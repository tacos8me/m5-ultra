# SPDX-License-Identifier: MIT
"""Env-gated front-end timestamps for the ds41-og worker (DS41_FE_TRACE=1; off: no-ops).

One JSON line per inference request (logger ds41.fe, prefix "ds41-fe trace"), wall-clock
seconds (time.time(), comparable with the supervisor's and a local client's stamps):

  http_start     ASGI scope entered (headers read)        body_done      last body chunk received
  handler        API handler entered (JSON parsed)        count_start/end, preflight_start/end
  engine_start   VLM stream_chat/chat entered             pcm_start/end  template render + tokenize
  add_request    EngineCore.add_request (scheduler arrival follows)
  og.*           og admission (og_model): deferred, job_start, box_ack, box_end, job_end, prepare_start,
                 import_*, prepare_end, first_step
  first_output   first engine output with text           first_body     first response bytes
  first_content  first SSE event carrying generated text  http_end
plus encode_s (tokenizer time accumulated over the request) and the prompt token count.
"""
import contextvars
import json
import logging
import os
import threading
import time

ON = os.environ.get('DS41_FE_TRACE', '0') == '1'
logger = logging.getLogger('ds41.fe')
_current = contextvars.ContextVar('ds41_fe_trace', default=None)
_local = threading.local()
BY_ID = {}      # engine request id -> trace dict (og admission stamps land here)
_lock = threading.Lock()
PATHS = ('/v1/chat/completions', '/v1/completions', '/v1/messages', '/v1/responses')
CONTENT = (b'"content":"', b'"reasoning_content":"', b'"text":"', b'"delta":"', b'"text_delta"', b'"arguments":"')


def now():
    return time.time()


def current():
    return _current.get() if ON else None


def stamp(key, trace=None, when=None):
    trace = trace if trace is not None else current()
    if trace is not None and key not in trace:
        trace[key] = when if when is not None else now()


def add(key, value, trace=None):
    trace = trace if trace is not None else (current() or getattr(_local, 'trace', None))
    if trace is not None:
        trace[key] = trace.get(key, 0.0) + value


def for_request(request_id):
    """The trace of an engine request (og admission side), or None."""
    if not ON or request_id is None:
        return None
    with _lock:
        return BY_ID.get(request_id)


def og_stamp(request_id, key, when=None):
    trace = for_request(request_id)
    if trace is not None:
        stamp('og.' + key, trace, when)


def merge(trace, record):
    for key, value in record.items():
        trace[key] = trace.get(key, 0.0) + value if key == 'encode_s' else value


def _has_content(chunk):
    for marker in CONTENT:
        at = chunk.find(marker)
        while at >= 0:
            if chunk[at + len(marker):at + len(marker) + 1] not in (b'"', b''):
                return True
            at = chunk.find(marker, at + 1)
    return False


class TraceMiddleware:
    """Pure ASGI: stamps receive/send times of inference requests and logs one line at the end."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not ON or scope['type'] != 'http' or scope.get('path') not in PATHS:
            return await self.app(scope, receive, send)
        trace = dict(path=scope['path'], http_start=now())
        token = _current.set(trace)

        async def recv():
            message = await receive()
            if message.get('type') == 'http.request' and not message.get('more_body'):
                stamp('body_done', trace)
            return message

        async def snd(message):
            if message.get('type') == 'http.response.body':
                body = message.get('body') or b''
                if body:
                    stamp('first_body', trace)
                    if 'first_content' not in trace and _has_content(body):
                        stamp('first_content', trace)
            await send(message)
        try:
            await self.app(scope, recv, snd)
        finally:
            _current.reset(token)
            trace['http_end'] = now()
            rid = trace.get('request_id')
            if rid is not None:
                with _lock:
                    BY_ID.pop(rid, None)
                _dump_profile(rid)
            logger.info('ds41-fe trace %s', json.dumps(trace, sort_keys=True))


PROFILE = [os.environ.get('DS41_FE_PROFILE', '0') == '1']  # runtime switch (/og/fe 'profile')
PROFILE_MIN = int(os.environ.get('DS41_FE_PROFILE_MIN', '100000'))
PROFILES = {}  # request id -> cProfile.Profile over its scheduler work until the first response


class _profiled:
    """Profile this thread's work for the given requests (MLX executor thread: add + steps)."""

    def __init__(self, *rids):
        self.profiles = [PROFILES[r] for r in rids if r in PROFILES]

    def __enter__(self):
        for p in self.profiles[:1]:  # one profiler per thread may be active; charge the first
            p.enable()

    def __exit__(self, *exc):
        for p in self.profiles[:1]:
            p.disable()


def _dump_profile(rid):
    profile = PROFILES.pop(rid, None)
    if profile is None:
        return
    import io
    import pstats
    out = io.StringIO()
    pstats.Stats(profile, stream=out).sort_stats('cumulative').print_stats(30)
    logger.info('ds41-fe profile %s:\n%s', rid, out.getvalue())


def _timed(fn, start, end):
    def wrapper(*args, **kwargs):
        trace = current()
        stamp(start, trace)
        try:
            return fn(*args, **kwargs)
        finally:
            if trace is not None:
                trace[end] = now()
    wrapper.__wrapped__ = fn
    return wrapper


def install(app):
    """Wire the stamps into omlx (worker process only). No-op unless DS41_FE_TRACE=1."""
    if not ON:
        return
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.engine_core import EngineCore
    from omlx import server

    app.add_middleware(TraceMiddleware)
    VLMBatchedEngine.count_chat_tokens = _timed(VLMBatchedEngine.count_chat_tokens, 'count_start', 'count_end')

    preflight = VLMBatchedEngine.preflight_chat

    async def preflight_chat(self, *args, **kwargs):
        trace = current()
        stamp('preflight_start', trace)
        try:
            return await preflight(self, *args, **kwargs)
        finally:
            if trace is not None:
                trace['preflight_end'] = now()
    VLMBatchedEngine.preflight_chat = preflight_chat

    process_messages = VLMBatchedEngine._process_chat_messages
    handoff = {}

    def process_chat_messages(self, messages, tools, kwargs):
        # Runs on the MLX executor thread (no context): hand the stamps back by messages identity.
        record = dict(pcm_start=now())
        _local.trace = record
        try:
            return process_messages(self, messages, tools, kwargs)
        finally:
            _local.trace = None
            record['pcm_end'] = now()
            handoff[id(messages)] = record
    VLMBatchedEngine._process_chat_messages = process_chat_messages

    for name in ('stream_chat', 'chat'):
        original = getattr(VLMBatchedEngine, name)
        if name == 'stream_chat':
            async def wrapped(self, messages, *args, _original=original, **kwargs):
                trace = current()
                stamp('engine_start', trace)
                try:
                    async for output in _original(self, messages, *args, **kwargs):
                        if trace is not None:
                            merge(trace, handoff.pop(id(messages), {}))
                            if 'first_output' not in trace and (getattr(output, 'new_text', None)
                                                                or getattr(output, 'tool_calls', None)):
                                trace['first_output'] = now()
                        yield output
                finally:
                    handoff.pop(id(messages), None)
        else:
            async def wrapped(self, messages, *args, _original=original, **kwargs):
                trace = current()
                stamp('engine_start', trace)
                try:
                    return await _original(self, messages, *args, **kwargs)
                finally:
                    if trace is not None:
                        merge(trace, handoff.pop(id(messages), {}))
        setattr(VLMBatchedEngine, name, wrapped)

    engine_add = EngineCore.add_request

    async def add_request(self, prompt, sampling_params=None, request_id=None, *args, **kwargs):
        trace = current()
        if trace is not None:
            import uuid
            request_id = request_id or str(uuid.uuid4())
            trace['add_request'] = now()
            trace['request_id'] = request_id
            trace['prompt_tokens'] = len(prompt) if isinstance(prompt, (list, tuple)) else None
            with _lock:
                BY_ID[request_id] = trace
            if PROFILE[0] and (trace['prompt_tokens'] or 0) >= PROFILE_MIN:
                PROFILES[request_id] = __import__('cProfile').Profile()
        try:
            return await engine_add(self, prompt, sampling_params, request_id, *args, **kwargs)
        finally:
            if trace is not None:
                trace['add_request_end'] = now()
    EngineCore.add_request = add_request

    from omlx.scheduler import Scheduler
    sched_add = Scheduler.add_request

    def scheduler_add_request(self, request):
        rid = request.request_id
        og_stamp(rid, 'sched_add')
        try:
            with _profiled(rid):
                return sched_add(self, request)
        finally:
            og_stamp(rid, 'sched_add_end')
    Scheduler.add_request = scheduler_add_request

    process_responses = Scheduler._process_batch_responses

    def process_batch_responses(self, responses):
        if BY_ID:
            for response in responses:
                rid = self.uid_to_request_id.get(getattr(response, 'uid', None))
                trace = for_request(rid)
                if trace is not None and 'og.first_response' not in trace:
                    trace['og.first_response'] = now()
                    _dump_profile(rid)
        return process_responses(self, responses)
    Scheduler._process_batch_responses = process_batch_responses

    step = Scheduler.step

    def scheduler_step(self):
        if not PROFILES:
            return step(self)
        with _profiled(*list(PROFILES)):
            return step(self)
    Scheduler.step = scheduler_step

    # The API handlers call get_engine_for_model right after FastAPI parsed the JSON body.
    get_engine = server.get_engine_for_model

    async def get_engine_for_model(*args, **kwargs):
        stamp('handler')
        return await get_engine(*args, **kwargs)
    server.get_engine_for_model = get_engine_for_model
    logger.info('ds41-fe trace on')
