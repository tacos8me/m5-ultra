"""Token-level response resume for the ds41 children (og worker and q3 fallback).

The ds41-og supervisor continues a response that failed mid-stream (box lost)
on another backend by re-sending the same request with the header
`x-ds41-resume: 1` and a body field {"ds41_resume": {"output": [token ids]}}:
the tokens the failed backend had already generated. This module admits such a
request with prompt + those tokens as its prompt, then replays the tokens
through the scheduler's normal output path (detokenizer, stop strings,
reasoning and tool parsers) ahead of the first new token, exactly as if they
had just been generated. The API layer therefore produces the whole response
again and the supervisor forwards only what the client has not received.

max_tokens counts the replayed tokens; usage.prompt_tokens includes them (the
supervisor subtracts them). A resumed request never stores to the prefix cache,
because its prompt key would repeat the replayed output. Requests without the
header are not touched.
"""
import contextvars
import copy
import json
import logging
import uuid

logger = logging.getLogger('ds41.resume')
HEADER = b'x-ds41-resume'
FIELD = 'ds41_resume'
_current = contextvars.ContextVar('ds41_resume', default=None)
PENDING = {}     # request_id -> tokens, between EngineCore.add_request and Scheduler.add_request
REPLAYING = set()
STATS = dict(resumed=0, replayed_tokens=0)


class ResumeMiddleware:
    """Pure ASGI: pops `ds41_resume` from marked request bodies into a context variable."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or not any(
                k == HEADER for k, _ in scope.get('headers', ())):
            return await self.app(scope, receive, send)
        parts, more = [], True
        while more:
            message = await receive()
            if message['type'] != 'http.request':
                return await self.app(scope, receive, send)
            parts.append(message.get('body', b''))
            more = message.get('more_body', False)
        body, tokens = b''.join(parts), None
        try:
            data = json.loads(body)
            resume = data.pop(FIELD, None)
            if isinstance(resume, dict) and isinstance(resume.get('output'), list):
                tokens = [int(t) for t in resume['output']]
                body = json.dumps(data).encode()
        except (ValueError, TypeError) as exc:
            logger.warning('ds41 resume body ignored: %r', exc)
        headers = [(k, v) for k, v in scope['headers'] if k != b'content-length']
        headers.append((b'content-length', str(len(body)).encode()))
        scope = dict(scope, headers=headers)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()
        token = _current.set(tokens) if tokens is not None else None
        try:
            await self.app(scope, replay, send)
        finally:
            if token is not None:
                _current.reset(token)


def install(app=None):
    from omlx.engine_core import EngineCore
    from omlx.scheduler import Scheduler
    if app is None:
        from omlx.server import app
    app.add_middleware(ResumeMiddleware)

    engine_add = EngineCore.add_request

    async def add_request(self, prompt, sampling_params=None, request_id=None, *args, **kwargs):
        tokens = _current.get()
        if tokens is not None:
            _current.set(None)  # one engine request per HTTP request
            request_id = request_id or str(uuid.uuid4())
            PENDING[request_id] = tokens
        return await engine_add(self, prompt, sampling_params, request_id, *args, **kwargs)

    scheduler_add = Scheduler.add_request

    def scheduler_add_request(self, request):
        tokens = PENDING.pop(request.request_id, None)
        if tokens:
            if request.prompt_token_ids is None:
                request.prompt_token_ids = (self.tokenizer.encode(request.prompt) if isinstance(request.prompt, str)
                                            else list(request.prompt))
            original = len(request.prompt_token_ids)
            request.prompt_token_ids = list(request.prompt_token_ids) + tokens
            request.num_prompt_tokens = len(request.prompt_token_ids)
            params = request.sampling_params
            params.max_tokens = max(1, int(params.max_tokens) - len(tokens))
            request.skip_cache_store = True
            request._ds41_replay = tokens
            REPLAYING.add(request.request_id)
            STATS['resumed'] += 1
            STATS['replayed_tokens'] += len(tokens)
            logger.warning('ds41 resume %s: prompt %d + %d replayed output tokens, max_tokens now %d',
                           request.request_id, original, len(tokens), params.max_tokens)
        return scheduler_add(self, request)

    process = Scheduler._process_batch_responses

    def process_batch_responses(self, responses):
        if REPLAYING:
            expanded = []
            for response in responses:
                request_id = self.uid_to_request_id.get(getattr(response, 'uid', None))
                request = self.running.get(request_id) if request_id is not None else None
                tokens = getattr(request, '_ds41_replay', None)
                if tokens:
                    request._ds41_replay = None
                    REPLAYING.discard(request_id)
                    for token in tokens:
                        fake = copy.copy(response)
                        fake.token, fake.finish_reason = token, None
                        for name in ('logprobs', 'error'):
                            if hasattr(fake, name):
                                setattr(fake, name, None)
                        expanded.append(fake)
                expanded.append(response)
            responses = expanded
        return process(self, responses)

    abort = Scheduler._do_abort_request

    def do_abort_request(self, request_id):
        REPLAYING.discard(request_id)
        PENDING.pop(request_id, None)
        return abort(self, request_id)

    EngineCore.add_request = add_request
    Scheduler.add_request = scheduler_add_request
    Scheduler._process_batch_responses = process_batch_responses
    Scheduler._do_abort_request = do_abort_request
    logger.info('ds41 resume hook installed')
