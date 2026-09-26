"""ds41-og: OpenAI-compatible supervisor for the original-weight split pipeline.

DS41_OG_Q3=off (default): og only. No q3 child is ever started. A box engine
restart is ridden out inside the og worker (bit-identical rebuild, up to
DS41_OG_RESTART_WAIT_S); if the box does not come back, a stream ends with a
clean retryable error (SSE error + [DONE]) and other requests get HTTP 503 with
Retry-After; og failing to start is a 503 too. DS41_OG_Q3=on restores the q3
fallback child described below (legacy).

CPU only (no MLX import, never under gpu-exec). It owns exactly one GPU child
at a time and proxies every request to it:

  og   original pipeline worker (box layers 0-19 over 10.10.10.1:10052, Mac
       layers 20-39 + head + DSpark), og_server.py under gpu-exec, ~150 GiB
  q3   the served q3 build (ds41-server settings, profile-c2) through ds41-q3,
       which only adds the inert resume hook (og_resume.py), ~224 GiB

The mode is chosen per request at admission: box trusted -> og, otherwise q3.
A switch waits until no upstream request is in flight, TERMs the child's own
PID (gpu-exec execs in place, so that PID owns the memory and the lock) and
waits for it to exit before the other child is started; the two never
co-reside. q3 -> og happens only after the box has answered for
DS41_OG_RECOVER_S (default 60 s) and the q3 child is idle.

Failure handling (chat and text completions):
  tier A  the og worker rebuilds a lost box session on the committed tokens
          and resends the step (bit-identical continuation, DS41_OG_RESUME_WAIT_S);
          the client only sees a pause (SSE keepalives continue).
  tier B  if the box stays down, the worker fails the request with a resume
          marker carrying the output tokens already generated. The supervisor
          keeps the client stream open (keepalive chunks), switches backend
          (q3, or og again once the box is trusted), re-sends the request with
          those tokens (og_resume.py replays them) and forwards only what the
          client has not received yet: same id, same text, no duplicate. At
          most DS41_OG_MAX_RESUMES times within DS41_OG_FAILOVER_S.
  else    a clean, retryable error: an SSE error event
          {"error": {"code": "backend_unavailable", ...}} + [DONE], or HTTP 503
          with Retry-After for non-streaming requests.
A child that dies mid-response (no marker) ends that response with the same
clean error. Other paths (/v1/messages, /v1/responses, ...) are proxied as is.

Images (chat image_url parts, /v1/messages image blocks, /v1/responses input_image):
DS41_OG_VISION=og (default) serves them on the og worker like text. The box runs
the vision tower (og_images.py); there is no q3 involvement: while the box is down
an image request gets a retryable 503, and image requests skip the q3 resume path
(an engine restart mid-stream is still ridden out inside the og worker, images
included). DS41_OG_VISION=reject answers HTTP 400 in the OpenAI error shape.
DS41_OG_VISION=q3 is the old interim (images on the q3 child); never the default.
"""
import argparse
import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HOME = Path.home()
HERE = Path(__file__).resolve().parent
LOG = logging.getLogger('ds41-og')
BOX = os.environ.get('DS41_OG_BOX', '10.10.10.1:10052')
OG_PORT = int(os.environ.get('DS41_OG_WORKER_PORT', '12147'))
Q3_PORT = int(os.environ.get('DS41_OG_Q3_PORT', '12148'))
RECOVER_S = float(os.environ.get('DS41_OG_RECOVER_S', '60'))
Q3 = os.environ.get('DS41_OG_Q3', 'off') == 'on'  # legacy q3 fallback child; off: og only, never q3
VISION = os.environ.get('DS41_OG_VISION', 'og')  # og: images on the og worker (box vision); reject: 400; q3: legacy
IMAGE_TYPES = {'image_url', 'input_image', 'image'}
# While the box host refuses the port (engine restarting, 2.5-3 min) og keeps serving: its worker
# waits for the engine (same limit, pipe_wire.RESTART_WAIT_S) instead of a double model swap.
RESTART_WAIT_S = float(os.environ.get('DS41_OG_RESTART_WAIT_S', '240'))
READY_S = float(os.environ.get('DS41_OG_READY_S', '900'))
FAILOVER_S = float(os.environ.get('DS41_OG_FAILOVER_S', '900'))
MAX_RESUMES = int(os.environ.get('DS41_OG_MAX_RESUMES', '2'))
KEEPALIVE_S = float(os.environ.get('DS41_OG_KEEPALIVE_S', '5'))
# Each child takes gpu.lock through gpu-exec. A test harness that already holds the lock for its
# whole lease sets DS41_OG_GPU_EXEC='' so no other job can take the GPU between two children.
GPU_EXEC = os.environ.get('DS41_OG_GPU_EXEC', str(HOME/'llm/bin/gpu-exec'))
PY = str(HOME/'llm/.venv-ds41-omlx-tiles/bin/python')
LOGS = Path(os.environ.get('DS41_OG_LOGS', str(HOME/'llm/ds41/og/logs')))
INFERENCE = ('/v1/chat/completions', '/v1/completions', '/v1/responses', '/v1/messages')
RESUMABLE = ('/v1/chat/completions', '/v1/completions')
DROP = {'host', 'content-length', 'transfer-encoding', 'connection', 'keep-alive', 'accept-encoding',
        'content-encoding'}
MARK = 'ds41-og-resume:'
DONE = b'data: [DONE]\n\n'


def box_state(timeout=0.5):
    """'up', 'refused' (host answers, engine not listening: restarting) or 'down' (host/link gone)."""
    host, port = BOX.rsplit(':', 1)
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return 'up'
    except ConnectionRefusedError:
        return 'refused'
    except OSError:
        return 'down'


def child_spec(mode):
    """(argv, env, port, model id) for one GPU child."""
    env = os.environ.copy()
    port, model_id = (OG_PORT, os.environ.get('DS41_OG_MODEL_ID', 'ds41-og')) if mode == 'og' else (Q3_PORT, 'ds41-q3g128')
    override = os.environ.get(f'DS41_OG_CHILD_{mode.upper()}')  # CPU tests: argv template with {port}
    if override:
        return [arg.format(port=port) for arg in shlex.split(override)], env, port, model_id
    if mode == 'og':
        env.update(MLX_ENABLE_TF32='0', OMLX_BONJOUR='0', OMLX_DISCOVERY='0', DS41_NATIVE_VERIFY='0',
                   DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1',
                   DS41_NATIVE_DECODE='1', DS41_MTP_COST_POLICY='1', DS41_COPY_DRAFT='1',
                   DS41_PREFIX_CACHE_GIB='0')
        env.setdefault('DS41_TREE', str(HERE.parent))
        argv = [PY, '-u', str(HERE/'og_server.py'), '--host', '127.0.0.1', '--port', str(OG_PORT)]
        return [GPU_EXEC] * bool(GPU_EXEC) + argv, env, OG_PORT, os.environ.get('DS41_OG_MODEL_ID', 'ds41-og')
    env.update(DS41_TREE=os.environ.get('DS41_OG_Q3_TREE', str(HOME/'src/wt/ds41-serve-c2')),
               DS41_PREFIX_CACHE_GIB='2', DS41_MTP_COST_POLICY='1', DS41_EXTRA_DRAFT='1')
    argv = [str(HERE/'ds41-q3'), '--host', '127.0.0.1', '--port', str(Q3_PORT),
            '--base-path', str(HOME/'llm/ds41/serve/profile-c2'), '--max-concurrent-requests', '2']
    return [GPU_EXEC] * bool(GPU_EXEC) + argv, env, Q3_PORT, 'ds41-q3g128'


def data_of(event):
    """'DONE', a JSON object, or None (comment / other) for one SSE event."""
    line = event.strip()
    if not line.startswith(b'data:'):
        return None
    payload = line[5:].strip()
    if payload == b'[DONE]':
        return 'DONE'
    with contextlib.suppress(ValueError):
        return json.loads(payload)
    return None


def marker_in(message):
    """Resume info embedded by the og worker (og_failover), or None."""
    if not isinstance(message, str):
        return None
    at = message.find(MARK)
    if at < 0:
        return None
    with contextlib.suppress(ValueError):
        info, _ = json.JSONDecoder().raw_decode(message, at + len(MARK))
        if isinstance(info, dict) and isinstance(info.get('output'), list):
            return info
    return None


MARKER_RE = re.compile(rb' ?\[ds41-og-resume:\{.*?\}\]')


def strip_marker(data):
    """Remove the worker's internal resume marker from bytes the client will see."""
    return MARKER_RE.sub(b'', data) if b'ds41-og-resume:' in data else data


def error_marker(obj):
    err = obj.get('error') if isinstance(obj, dict) else None
    if isinstance(err, dict):
        return marker_in(err.get('message'))
    return marker_in(err)


def fix_usage(usage, replayed):
    """Usage of a resumed attempt as the client's request saw it (the replayed tokens were output)."""
    usage = dict(usage)
    for key in ('prompt_tokens', 'input_tokens'):
        if isinstance(usage.get(key), int):
            usage[key] = max(0, usage[key] - replayed)
    usage['total_tokens'] = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
    return usage


def has_image(data):
    """Image content in a request: chat/Anthropic message content or Responses input (not tool schemas)."""
    def scan(value, depth=0):
        if depth > 8:
            return False
        if isinstance(value, dict):
            if value.get('type') in IMAGE_TYPES:
                return True
            return any(scan(v, depth + 1) for v in value.values() if isinstance(v, (dict, list)))
        if isinstance(value, list):
            return any(scan(v, depth + 1) for v in value if isinstance(v, (dict, list)))
        return False
    return any(scan(data.get(key)) for key in ('messages', 'input', 'system') if isinstance(data.get(key), list))


def image_error(reason):
    return JSONResponse(dict(error=dict(message=f'ds41-og: this request contains images, which {reason}; send it '
                                                'without images', type='invalid_request_error',
                                        param='messages', code='images_unavailable')), status_code=400)


def image_retry(reason):
    """Images need the box (it runs the vision tower); a clean, retryable 503 while it is away."""
    return JSONResponse(dict(error=dict(message=f'ds41-og: images need the RTX box, which {reason}; retry shortly',
                                        type='server_error', code='images_unavailable')),
                        status_code=503, headers={'Retry-After': '30'})


def stream_error(path, message, code):
    if path.endswith('/messages'):
        body = dict(type='error', error=dict(type='invalid_request_error', message=message))
        return b'event: error\ndata: ' + json.dumps(body).encode() + b'\n\n'
    body = dict(error=dict(message=message, type='invalid_request_error', code=code))
    return b'data: ' + json.dumps(body).encode() + b'\n\n' + DONE


def fix_keepalive(event, model):
    """The children's keepalive chunks say model "keepalive"; clients see the real model id instead."""
    if b'"model":"keepalive"' in event:
        return event.replace(b'"model":"keepalive"', b'"model":' + json.dumps(model or 'ds41-og').encode())
    return event


def stream_keepalive(path, model='ds41-og'):
    if path.endswith('/chat/completions'):
        return (b'data: {"id":"chatcmpl-keepalive","object":"chat.completion.chunk","created":0,"model":'
                + json.dumps(model).encode() +
                b',"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n')
    if path.endswith('/messages'):
        return b'event: ping\ndata: {"type":"ping"}\n\n'
    return b': keepalive\n\n'


def clean_message(reason):
    return (f'ds41-og: the backend became unavailable while generating ({reason}); the response is '
            'incomplete, retry the request')


def error_event(reason):
    body = dict(error=dict(message=clean_message(reason), type='server_error', code='backend_unavailable'))
    return b'data: ' + json.dumps(body).encode() + b'\n\n'


async def sse_events(upstream):
    buf = b''
    async for chunk in upstream.aiter_bytes():
        buf += chunk
        while (end := buf.find(b'\n\n')) >= 0:
            yield buf[:end + 2]
            buf = buf[end + 2:]
    if buf.strip():
        yield buf


class Splice:
    """What one client response has received, and how a resumed attempt continues it.

    The first attempt is relayed byte for byte. A resumed attempt re-emits the whole
    response (og_resume replays the earlier tokens); its text channels are skipped up
    to what was already sent, its id/model are rewritten to the first attempt's, and
    usage.prompt_tokens excludes the replayed tokens.
    """

    CHANNELS = ('content', 'reasoning_content', 'text')

    def __init__(self, chat):
        self.chat = chat
        self.id = self.model = None
        self.sent = dict.fromkeys(self.CHANNELS, '')
        self.tools = {}
        self.seen = None
        self.tools_seen = None
        self.replayed = 0
        self.divergent = 0
        self.resumes = 0

    def begin_resume(self, replayed):
        self.seen = dict.fromkeys(self.CHANNELS, 0)
        self.tools_seen = {}
        self.replayed = replayed
        self.resumes += 1

    def record(self, obj):
        self.id = self.id or obj.get('id')
        if obj.get('model') not in (None, 'keepalive'):
            self.model = self.model or obj['model']
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            for key in ('content', 'reasoning_content'):
                if delta.get(key):
                    self.sent[key] += delta[key]
            if choice.get('text'):
                self.sent['text'] += choice['text']
            for call in delta.get('tool_calls') or []:
                self._tool(call)

    def _tool(self, call):
        entry = self.tools.setdefault(call.get('index', 0), dict(args=''))
        for key in ('id', 'type'):
            if call.get(key):
                entry[key] = call[key]
        function = call.get('function') or {}
        if function.get('name'):
            entry['name'] = function['name']
        entry['args'] += function.get('arguments') or ''

    def _skip(self, channel, text):
        seen, sent = self.seen[channel], self.sent[channel]
        self.seen[channel] = seen + len(text)
        if seen + len(text) <= len(sent):
            self.divergent += sent[seen:seen + len(text)] != text
            return ''
        if seen < len(sent):
            self.divergent += sent[seen:] != text[:len(sent) - seen]
            text = text[len(sent) - seen:]
        self.sent[channel] += text
        return text

    def _skip_tool(self, call):
        index = call.get('index', 0)
        entry = self.tools.get(index)
        if entry is None:
            self._tool(call)
            return call
        seen = self.tools_seen.get(index, 0)
        function = dict(call.get('function') or {})
        args = function.get('arguments') or ''
        self.tools_seen[index] = seen + len(args)
        if seen + len(args) <= len(entry['args']):
            args = ''
        elif seen < len(entry['args']):
            args = args[len(entry['args']) - seen:]
        entry['args'] += args
        if not args:
            return None
        return dict(index=index, function=dict(arguments=args))

    def rewrite(self, obj):
        """A resumed attempt's chunk as the client should see it (None: nothing new)."""
        if 'error' in obj:
            return obj
        obj = dict(obj)
        if self.id:
            obj['id'] = self.id
        if self.model and obj.get('model') != 'keepalive':
            obj['model'] = self.model
        usage = obj.get('usage')
        if usage:
            obj['usage'] = usage = fix_usage(usage, self.replayed)
        choices = []
        for choice in obj.get('choices') or []:
            choice = dict(choice)
            if self.chat:
                delta = dict(choice.get('delta') or {})
                delta.pop('role', None)
                for key in ('content', 'reasoning_content'):
                    if delta.get(key):
                        delta[key] = self._skip(key, delta[key])
                    if key in delta and not delta[key]:
                        del delta[key]
                calls = [c for c in (self._skip_tool(c) for c in delta.pop('tool_calls', None) or []) if c]
                if calls:
                    delta['tool_calls'] = calls
                if not delta and choice.get('finish_reason') is None:
                    continue
                choice['delta'] = delta
            else:
                choice['text'] = self._skip('text', choice.get('text') or '')
                if not choice['text'] and choice.get('finish_reason') is None:
                    continue
            choices.append(choice)
        if not choices and not usage:
            return None
        obj['choices'] = choices
        return obj

    def keepalive(self):
        if self.chat:
            body = dict(id=self.id or 'chatcmpl-keepalive', object='chat.completion.chunk', created=0,
                        model=self.model or 'ds41-og', choices=[dict(index=0, delta=dict(role='assistant', content=''),
                                                         finish_reason=None)])
        else:
            body = dict(id=self.id or 'cmpl-keepalive', object='text_completion', created=0,
                        model=self.model or 'ds41-og',
                        choices=[dict(index=0, text='', logprobs=None, finish_reason=None)])
        return b'data: ' + json.dumps(body, separators=(',', ':')).encode() + b'\n\n'


class Supervisor:
    def __init__(self):
        self.mode = None
        self.ready = False
        self.proc = None
        self.port = None
        self.model_id = None
        self.inflight = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.switch = asyncio.Lock()
        self.box_since = None
        self.refused_since = None
        self.lost_at = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))
        self.vision_at = None
        self.vision_waiting = 0
        self.stats = dict(og=0, q3=0, failovers=0, switches=0, box_lost=0, resumed=0, resume_failed=0,
                          vision=0, vision_rejected=0,
                          broken=0, divergent=0)

    async def start(self, mode):
        argv, env, port, model_id = child_spec(mode)
        LOGS.mkdir(parents=True, exist_ok=True)
        log = (LOGS/f'{mode}-child.log').open('a')
        log.write(f'\n=== {time.strftime("%F %T")} start {mode}: {" ".join(argv)}\n'); log.flush()
        self.proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Detached reaper: if this supervisor dies without reaping (e.g. SIGKILL),
        # TERM the GPU child so it cannot hold gpu.lock and its memory orphaned.
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--reap', str(os.getpid()), str(self.proc.pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.mode, self.port, self.model_id = mode, port, model_id
        self.stats['switches'] += 1
        LOG.info('started %s child pid %d on %d', mode, self.proc.pid, port)
        deadline = time.monotonic() + READY_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'{mode} child exited {self.proc.returncode} before ready')
            with contextlib.suppress(httpx.HTTPError):
                r = await self.client.get(f'http://127.0.0.1:{port}/health', timeout=2)
                if r.status_code == 200:
                    LOG.info('%s child ready', mode)
                    self.ready = True
                    if mode == 'q3' and not os.environ.get('DS41_OG_CHILD_Q3'):
                        # External 245 GiB SIGKILL guard (the og worker starts its own); started after
                        # readiness so it binds to the final exec'd process.
                        subprocess.Popen(['/opt/homebrew/bin/python3', str(HERE/'watch.py'), str(self.proc.pid),
                                          str(LOGS/'q3-child.memory.json'), '0'], start_new_session=True,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
            await asyncio.sleep(1)
        raise TimeoutError(f'{mode} child not ready after {READY_S:.0f}s')

    async def stop(self):
        mode, proc, self.proc, self.mode, self.ready = self.mode, self.proc, None, None, False
        if proc is None or proc.poll() is not None:
            return
        LOG.info('stopping %s child pid %d', mode, proc.pid)
        proc.terminate()
        # omlx drains in-flight requests on TERM; bound it, then KILL the memory owner.
        for _ in range(60):
            if proc.poll() is not None:
                return
            await asyncio.sleep(.5)
        proc.kill()
        await asyncio.get_running_loop().run_in_executor(None, proc.wait)

    async def probe(self):
        state = await asyncio.get_running_loop().run_in_executor(None, box_state)
        now = time.monotonic()
        self.box_since = (self.box_since or now) if state == 'up' else None
        self.refused_since = (self.refused_since or now) if state == 'refused' else None
        return state

    def box_restarting(self):
        """The engine is restarting (port refused) for less than DS41_OG_RESTART_WAIT_S."""
        return self.refused_since is not None and time.monotonic() - self.refused_since < RESTART_WAIT_S

    def box_ok(self):
        """Box reachable now, and for DS41_OG_RECOVER_S if it was lost mid-response."""
        if self.box_since is None:
            return False
        if not Q3:
            return True  # no fallback to protect: use the box as soon as it answers
        if self.lost_at is not None and time.monotonic() - self.box_since >= RECOVER_S:
            self.lost_at = None  # trusted again
        return self.lost_at is None

    def box_lost(self):
        self.lost_at = time.monotonic()
        self.box_since = None
        self.stats['box_lost'] += 1

    async def watch_box(self):
        # CPU-only probe; the recovery clock runs even while no request arrives.
        while True:
            await self.probe()
            await asyncio.sleep(2)

    def want(self, vision=False):
        """The child the next request needs, or None when the current one can serve it now."""
        alive = self.proc is not None and self.proc.poll() is None
        if not Q3:
            return None if alive and self.mode == 'og' else 'og'
        if vision or self.vision_waiting or (self.vision_at is not None
                                             and time.monotonic() - self.vision_at < RECOVER_S):
            # Images (and text while an image request is active or recent) go to q3.
            return None if alive and self.mode == 'q3' else 'q3'
        ok = self.box_ok()
        if alive and self.mode == 'og' and (ok or (self.box_restarting() and self.lost_at is None)):
            return None
        if alive and self.mode == 'q3' and (not ok or self.inflight):
            return None
        return 'og' if ok else 'q3'

    async def ensure(self, vision=False):
        """Pick the backend for the next request; switch only when no upstream is in flight."""
        if vision:
            self.vision_at = time.monotonic()
            self.vision_waiting += 1
        try:
            await self.probe()
            if self.want(vision) is None:
                return
            async with self.switch:
                want = self.want(vision)
                if want is None:
                    return
                await self.idle.wait()
                want = self.want(vision) or self.mode
                if self.mode == want and self.proc and self.proc.poll() is None:
                    return
                if self.mode is not None and self.mode != want:
                    self.stats['failovers'] += want == 'q3' and not vision
                await self.stop()
                try:
                    await self.start(want)
                except (RuntimeError, TimeoutError, OSError) as exc:
                    if want != 'og' or not Q3:
                        raise
                    # e.g. the box went away again during og's warm-up: serve from q3 instead.
                    LOG.error('og child did not start (%s); falling back to q3', exc)
                    self.box_lost()
                    self.stats['failovers'] += 1
                    await self.stop()
                    await self.start('q3')
        finally:
            if vision:
                self.vision_waiting -= 1
                # Hold q3 for text only if it is actually serving; a q3 that failed must not keep og away.
                q3_up = self.mode == 'q3' and self.proc is not None and self.proc.poll() is None
                self.vision_at = time.monotonic() if q3_up else None

    async def open(self, method, path, query, headers, data=None, body=b'', resume=None, vision=False):
        """Send one request to the current child; returns (streaming response, mode)."""
        await self.ensure(vision)
        if vision and self.mode != 'q3':
            raise RuntimeError(f'images need the q3 child, current backend is {self.mode}')
        mode, port = self.mode, self.port
        headers = dict(headers)
        if isinstance(data, dict):
            data = dict(data)
            if 'model' in data:
                data['model'] = self.model_id
            if resume is not None:
                data['ds41_resume'] = dict(output=resume)
                headers['x-ds41-resume'] = '1'
            body = json.dumps(data).encode()
        url = f'http://127.0.0.1:{port}{path}' + (f'?{query}' if query else '')
        self.inflight += 1
        self.idle.clear()
        self.stats[mode] += 1
        try:
            upstream = await self.client.send(self.client.build_request(method, url, headers=headers, content=body),
                                              stream=True)
        except BaseException:
            self._done()
            raise
        upstream._ds41_vision = vision
        upstream._ds41_model = self.model_id
        return upstream, mode

    def _done(self):
        self.inflight -= 1
        if not self.inflight:
            self.idle.set()

    async def forward(self, request: Request, body: bytes):
        path, query = request.url.path, request.url.query
        headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
        data = None
        if body and request.headers.get('content-type', '').startswith('application/json'):
            with contextlib.suppress(ValueError):
                data = json.loads(body)
        if path in INFERENCE and isinstance(data, dict) and has_image(data):
            if VISION == 'reject' or (VISION == 'q3' and Q3):
                return await self.forward_images(request.method, path, query, headers, data)
            return await self.forward_images_og(request.method, path, query, headers, data, body)
        if not Q3 and path in INFERENCE:
            await self.probe()
            if not self.box_ok() and not self.box_restarting():
                return self.unavailable_response('the RTX box is not reachable')
        resumable = (path in RESUMABLE and isinstance(data, dict) and not data.get('logprobs')
                     and data.get('n') in (None, 1))
        if not resumable:
            try:
                upstream, mode = await self.open(request.method, path, query, headers, data, body)
            except (OSError, RuntimeError, httpx.HTTPError) as exc:
                LOG.exception('no backend for %s', path)
                return self.error_response(f'no backend: {type(exc).__name__}')
            return self.relay_raw(upstream, mode)
        if data.get('stream'):
            try:
                upstream, mode = await self.open(request.method, path, query, headers, data)
            except (OSError, RuntimeError, httpx.HTTPError) as exc:
                LOG.exception('no backend for a stream')
                return self.error_response(f'no backend: {type(exc).__name__}')
            if upstream.status_code != 200:
                return self.relay_raw(upstream, mode)
            out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP}
            out_headers['x-ds41-og-backend'] = mode
            return StreamingResponse(self.splice(upstream, request.method, path, query, headers, data),
                                     status_code=200, headers=out_headers)
        return await self.complete(request.method, path, query, headers, data)

    async def forward_images_og(self, method, path, query, headers, data, body):
        """Image request on the og worker (the box runs the vision tower); never on q3."""
        self.stats['vision'] += 1
        await self.probe()
        if not self.box_ok() and not self.box_restarting():
            self.stats['vision_rejected'] += 1
            return image_retry('is not reachable')
        try:
            upstream, mode = await self.open(method, path, query, headers, data, body)
        except (OSError, RuntimeError, httpx.HTTPError) as exc:
            self.stats['vision_rejected'] += 1
            LOG.exception('no og worker for an image request')
            return image_retry(f'could not be reached ({type(exc).__name__})')
        if mode != 'og':
            with contextlib.suppress(Exception):
                await upstream.aclose()
            self.stats['vision_rejected'] += 1
            return image_retry('went away')
        return self.relay_raw(upstream, mode)

    async def forward_images(self, method, path, query, headers, data):
        """Serve an image request on the q3 child (og is text only)."""
        self.stats['vision'] += 1
        if VISION != 'q3':
            self.stats['vision_rejected'] += 1
            return image_error('this server is configured not to serve')
        if data.get('stream') and self.want(vision=True) is not None:
            # A model swap first (~45 s): stream keepalives meanwhile so the client does not time out.
            out_headers = {'x-ds41-og-backend': 'q3', 'cache-control': 'no-cache'}
            return StreamingResponse(self.stream_after_switch(method, path, query, headers, data),
                                     media_type='text/event-stream', headers=out_headers)
        try:
            upstream, mode = await self.open(method, path, query, headers, data, vision=True)
        except (OSError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
            self.stats['vision_rejected'] += 1
            LOG.exception('no q3 child for an image request')
            return image_error(f'need the q3 fallback, which could not start ({type(exc).__name__})')
        return self.relay_raw(upstream, mode)

    async def stream_after_switch(self, method, path, query, headers, data):
        pending = asyncio.ensure_future(self.open(method, path, query, headers, data, vision=True))
        try:
            while not pending.done():
                await asyncio.wait({pending}, timeout=KEEPALIVE_S)
                if not pending.done():
                    yield stream_keepalive(path, self.model_id or 'ds41-og')
            try:
                upstream, _ = pending.result()
            except Exception as exc:  # noqa: BLE001 -- surfaced as a well-formed stream error
                self.stats['vision_rejected'] += 1
                LOG.exception('no q3 child for an image stream')
                yield stream_error(path, f'ds41-og: this request contains images, which need the q3 fallback; it '
                                         f'could not start ({type(exc).__name__}). Retry, or send it without images.',
                                   'images_unavailable')
                return
            pending = None
            try:
                if upstream.status_code != 200:
                    raw = await upstream.aread()
                    message = raw.decode(errors='replace')[:500]
                    with contextlib.suppress(ValueError, AttributeError, TypeError):
                        err = json.loads(raw).get('error')
                        message = err.get('message', message) if isinstance(err, dict) else str(err or message)
                    yield stream_error(path, message, f'upstream_{upstream.status_code}')
                    return
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await self.release(upstream)
        finally:
            if pending is not None and not pending.done():
                pending.add_done_callback(lambda task: self._drop_opened(task))

    def relay_raw(self, upstream, mode):
        out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP}
        out_headers['x-ds41-og-backend'] = mode
        sse = upstream.headers.get('content-type', '').startswith('text/event-stream')

        async def relay():
            try:
                if sse:  # event by event, so a resume marker (not resumable on this path) never leaks
                    async for event in sse_events(upstream):
                        yield fix_keepalive(strip_marker(event), getattr(upstream, '_ds41_model', None))
                else:
                    async for chunk in upstream.aiter_bytes():
                        yield strip_marker(chunk)
            finally:
                await self.release(upstream)
        return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers)

    async def release(self, upstream):
        """Count the upstream done, then close it (shielded: a client disconnect cancels the
        relay task, and the child only aborts its generation once this connection closes)."""
        self._done()
        if getattr(upstream, '_ds41_vision', False):
            self.vision_at = time.monotonic()  # q3 stays until images have been idle for RECOVER_S
        with anyio.CancelScope(shield=True):
            await upstream.aclose()

    async def splice(self, upstream, method, path, query, headers, data):
        splice = Splice(chat=path.endswith('/chat/completions'))
        started = time.monotonic()
        pending = None
        try:
            while True:
                info = broken = None
                try:
                    async for event in sse_events(upstream):
                        obj = data_of(event)
                        if isinstance(obj, dict) and 'error' in obj and (marker := error_marker(obj)) is not None:
                            info = marker
                            continue
                        if obj == 'DONE' and info is not None:
                            continue
                        if splice.seen is None or not isinstance(obj, dict):
                            if isinstance(obj, dict):
                                splice.record(obj)
                            yield fix_keepalive(event, getattr(upstream, '_ds41_model', None))
                            continue
                        obj = splice.rewrite(obj)
                        if obj is not None:
                            yield b'data: ' + json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode() + b'\n\n'
                except httpx.HTTPError as exc:
                    broken = exc
                finally:
                    await self.release(upstream)
                if info is None and broken is None:
                    return
                if info is None:
                    self.stats['broken'] += 1
                    LOG.error('upstream broke mid-response: %r', broken)
                    yield error_event(f'{type(broken).__name__}')
                    yield DONE
                    return
                elapsed = time.monotonic() - started
                if info.get('kind') != 'box' or not Q3 or splice.resumes >= MAX_RESUMES or elapsed > FAILOVER_S:
                    self.stats['resume_failed'] += 1
                    LOG.error('cannot resume (%s, %d resumes, %.0fs): %s', info.get('kind'), splice.resumes, elapsed,
                              info.get('reason'))
                    yield error_event(info.get('reason') or 'box unavailable')
                    yield DONE
                    return
                self.box_lost()
                output = [int(t) for t in info['output']]
                LOG.warning('box lost mid-response after %d output tokens (%s); resuming', len(output),
                            info.get('reason'))
                splice.begin_resume(len(output))
                pending = asyncio.ensure_future(self.open(method, path, query, headers, data, resume=output))
                while not pending.done():
                    await asyncio.wait({pending}, timeout=KEEPALIVE_S)
                    if not pending.done():
                        yield splice.keepalive()
                try:
                    upstream, mode = pending.result()
                except Exception as exc:  # noqa: BLE001 -- surfaced as the clean error
                    self.stats['resume_failed'] += 1
                    LOG.exception('resume could not start')
                    yield error_event(f'fallback did not start: {type(exc).__name__}')
                    yield DONE
                    return
                finally:
                    pending = None
                if upstream.status_code != 200:
                    try:
                        raw = (await upstream.aread())[:300]
                    finally:
                        await self.release(upstream)
                    self.stats['resume_failed'] += 1
                    yield error_event(f'fallback answered {upstream.status_code}: {raw!r}')
                    yield DONE
                    return
                self.stats['resumed'] += 1
                LOG.warning('resuming on %s after %.1fs (%d replayed tokens)', mode, time.monotonic() - started,
                            len(output))
        finally:
            self.stats['divergent'] += splice.divergent
            if pending is not None and not pending.done():
                # Client went away while a backend was starting; let the switch finish, drop the request.
                pending.add_done_callback(lambda task: self._drop_opened(task))

    def _drop_opened(self, task):
        if task.cancelled() or task.exception() is not None:
            return
        upstream, _ = task.result()
        asyncio.ensure_future(self.release(upstream))

    async def complete(self, method, path, query, headers, data):
        """Non-streaming chat/text completion with the same failover (nothing sent yet)."""
        started, resumes, resume = time.monotonic(), 0, None
        while True:
            try:
                upstream, mode = await self.open(method, path, query, headers, data, resume=resume)
            except (OSError, RuntimeError, httpx.HTTPError) as exc:
                LOG.exception('no backend for a completion')
                return self.error_response(f'no backend: {type(exc).__name__}')
            try:
                raw = await upstream.aread()
            except httpx.HTTPError as exc:
                self.stats['broken'] += 1
                return self.error_response(type(exc).__name__)
            finally:
                await self.release(upstream)
            info = None
            if upstream.status_code >= 500:
                with contextlib.suppress(ValueError):
                    info = error_marker(json.loads(raw))
            out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP}
            out_headers['x-ds41-og-backend'] = mode
            if info is None:
                if resume and upstream.status_code == 200:
                    with contextlib.suppress(ValueError, TypeError, KeyError):
                        body = json.loads(raw)
                        body['usage'] = fix_usage(body['usage'], len(resume))
                        raw = json.dumps(body).encode()
                return Response(raw, status_code=upstream.status_code, headers=out_headers)
            if info.get('kind') != 'box' or not Q3 or resumes >= MAX_RESUMES or time.monotonic() - started > FAILOVER_S:
                self.stats['resume_failed'] += 1
                return self.error_response(info.get('reason') or 'box unavailable')
            self.box_lost()
            resumes += 1
            resume = [int(t) for t in info['output']]
            self.stats['resumed'] += 1
            LOG.warning('box lost in a non-streaming request after %d output tokens; resuming', len(resume))

    @staticmethod
    def unavailable_response(reason):
        return JSONResponse(dict(error=dict(message=f'ds41-og: {reason}; retry shortly', type='server_error',
                                            code='backend_unavailable')),
                            status_code=503, headers={'Retry-After': '30'})

    @staticmethod
    def error_response(reason):
        return JSONResponse(dict(error=dict(message=clean_message(reason), type='server_error',
                                            code='backend_unavailable')),
                            status_code=503, headers={'Retry-After': '30'})


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def reap(supervisor, child):
    """Exit with the child; TERM (then KILL) it if the supervisor vanishes first."""
    while alive(child):
        if not alive(supervisor):
            os.kill(child, 15)
            for _ in range(240):
                if not alive(child):
                    return
                time.sleep(.5)
            os.kill(child, 9)
            return
        time.sleep(1)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--reap':
        return reap(int(sys.argv[2]), int(sys.argv[3]))
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s ds41-og %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    sup = Supervisor()
    app = FastAPI()

    @app.on_event('startup')
    async def boot():
        # Load in the background so a stop during loading is still handled.
        sup.booting = asyncio.create_task(sup.ensure())
        sup.watcher = asyncio.create_task(sup.watch_box())

    @app.on_event('shutdown')
    async def halt():
        await sup.stop()

    @app.get('/health')
    async def health():
        ok = sup.ready and sup.proc is not None and sup.proc.poll() is None
        # Without q3, a failed og start at boot (e.g. box down) must not leave llama-swap waiting 15 min
        # for health: report ready and answer requests with 503 (or start og once the box is back).
        degraded = not ok and not Q3 and getattr(sup, 'booting', None) is not None and sup.booting.done()
        return JSONResponse(dict(status='ok' if ok else 'degraded' if degraded else 'starting', backend=sup.mode,
                                 inflight=sup.inflight,
                                 pid=sup.proc.pid if sup.proc is not None else None,
                                 box_up=sup.box_since is not None, box_trusted=sup.box_ok(),
                                 box_restarting=sup.box_restarting(), q3_enabled=Q3, **sup.stats),
                            status_code=200 if ok or degraded else 503)

    @app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
    async def proxy(path: str, request: Request):
        body = await request.body()
        if sup.proc is None and request.url.path not in INFERENCE:
            return Response(status_code=503)
        return await sup.forward(request, body)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
    finally:
        # Reap the GPU child before exiting (llama-swap stops us with SIGTERM).
        proc = sup.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()


if __name__ == '__main__':
    main()
