"""CPU-only tests: ds41-og supervisor tier-B failover with fake children (fakechild.py).

Run: python og_serve/test_supervisor_failover.py   (no MLX, no GPU, no real box)
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
PY = sys.executable
BASE_PORT = int(os.environ.get('SUP_TEST_PORT', '12190'))  # uses 12190-12199
EXPECTED = ''.join(f'w{t} ' for t in range(40))
NOSTART = Path(tempfile.mkdtemp(prefix='sv-nostart-'))/'flag'
os.environ['FAKE_OG_NOSTART'] = str(NOSTART)
NOQ3 = NOSTART.parent/'noq3'
os.environ['FAKE_Q3_NOSTART'] = str(NOQ3)


class BoxPort:
    """A plain listener the supervisor's TCP probe sees as the box."""

    def __init__(self, port):
        self.port, self.srv = port, None
        self.up()

    def up(self):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', self.port))
        srv.listen(64)
        self.srv = srv

        def loop():
            while True:
                try:
                    conn, _ = srv.accept()
                    conn.close()
                except OSError:
                    return
        threading.Thread(target=loop, daemon=True).start()

    def down(self):
        try:
            self.srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.srv.close()


class Sup:
    def __init__(self, port, box_port, **env):
        self.port = port
        self.base = f'http://127.0.0.1:{port}'
        logs = Path(tempfile.mkdtemp(prefix='sv-sup-'))
        child = f'{PY} {HERE/"fakechild.py"} --port {{port}} --mode'
        e = dict(os.environ, DS41_OG_BOX=f'127.0.0.1:{box_port}', DS41_OG_WORKER_PORT=str(port + 1),
                 DS41_OG_Q3_PORT=str(port + 2), DS41_OG_CHILD_OG=child + ' og', DS41_OG_CHILD_Q3=child + ' q3',
                 DS41_OG_RECOVER_S='3', DS41_OG_KEEPALIVE_S='0.5', DS41_OG_LOGS=str(logs), DS41_OG_RESTART_WAIT_S='0')
        e.update(env)
        self.proc = subprocess.Popen([PY, str(HERE/'ds41_og.py'), '--port', str(port)], env=e,
                                     stdout=(logs/'sup.log').open('w'), stderr=subprocess.STDOUT)
        self.logs = logs
        deadline = time.time() + 30
        while time.time() < deadline:
            if (self.health() or {}).get('status') == 'ok':
                return
            time.sleep(0.3)
        raise RuntimeError('supervisor not ready: ' + (logs/'sup.log').read_text()[-2000:])

    def health(self):
        try:
            with urllib.request.urlopen(self.base + '/health', timeout=3) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            return json.load(e)
        except OSError:
            return None

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)


def post(base, path, body, timeout=60):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def stream(base, body, path='/v1/chat/completions', stop_after=None):
    body = dict(dict(model='ds41', stream=True, stream_options={'include_usage': True}), **body)
    out = dict(content='', reasoning='', text='', ids=set(), errors=[], finish=None, usage=None, keepalives=0,
               done=False, events=0, keepalive_model=0)
    with post(base, path, body) as r:
        out['backend'] = r.headers.get('x-ds41-og-backend')
        for line in r:
            line = line.strip()
            out['keepalive_model'] += b'"model":"keepalive"' in line or b'"model": "keepalive"' in line
            if not line.startswith(b'data: '):
                continue
            if line == b'data: [DONE]':
                out['done'] = True
                continue
            d = json.loads(line[6:])
            out['events'] += 1
            if 'error' in d:
                out['errors'].append(d['error'])
                continue
            if d.get('model') == 'keepalive' or (d.get('created') == 0 and [c.get('delta') for c in d.get('choices') or []]
                                                 == [{'role': 'assistant', 'content': ''}]):
                out['keepalives'] += 1
                continue
            out['ids'].add(d.get('id'))
            out['usage'] = d.get('usage') or out['usage']
            for c in d.get('choices') or []:
                delta = c.get('delta') or {}
                out['content'] += delta.get('content') or ''
                out['reasoning'] += delta.get('reasoning_content') or ''
                out['text'] += c.get('text') or ''
                out['finish'] = c.get('finish_reason') or out['finish']
            if stop_after is not None and out['events'] >= stop_after:
                return out
    return out


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f': {detail}'), flush=True)
    return bool(cond)


def main():
    box = BoxPort(BASE_PORT)
    sup = Sup(BASE_PORT + 1, BASE_PORT, DS41_OG_Q3='on')  # legacy q3 fallback child
    results = []
    try:
        r = stream(sup.base, {})
        results.append(check('plain stream on og', r['content'] == EXPECTED and r['backend'] == 'og' and r['done']
                             and not r['errors'], r))

        r = stream(sup.base, {'fake_fail_after': 12})
        h = sup.health()
        results.append(check('mid-stream box loss resumed on q3: full text once, no error',
                             r['content'] == EXPECTED and not r['errors'] and r['done'] and r['finish'] == 'length', r))
        results.append(check('one response id across the splice', len(r['ids']) == 1, r['ids']))
        results.append(check('usage excludes replayed tokens', r['usage'] and r['usage']['prompt_tokens'] == 17
                             and r['usage']['completion_tokens'] == 40, r['usage']))
        results.append(check('supervisor now on q3, 1 resume', h['backend'] == 'q3' and h['resumed'] == 1, h))

        time.sleep(4)
        r = stream(sup.base, {})
        results.append(check('back to og after DS41_OG_RECOVER_S at an idle boundary', r['backend'] == 'og'
                             and r['content'] == EXPECTED, r))

        r = stream(sup.base, {'fake_fail_after': 5, 'chat_template_kwargs': {'enable_thinking': True}})
        exp_r = ''.join(f'w{t} ' for t in range(10))
        results.append(check('loss inside reasoning: channels intact', r['reasoning'] == exp_r
                             and r['content'] == EXPECTED[len(exp_r):] and not r['errors'], r))

        time.sleep(4)
        with post(sup.base, '/v1/chat/completions', dict(model='ds41', fake_fail_after=9)) as resp:
            d = json.load(resp)
        results.append(check('non-streaming loss resumed', d['choices'][0]['message']['content'] == EXPECTED
                             and d['usage']['prompt_tokens'] == 17, d))

        time.sleep(4)
        r = stream(sup.base, {'prompt': 'x', 'fake_fail_after': 7}, path='/v1/completions')
        results.append(check('text completion resumed', r['text'] == EXPECTED and not r['errors'], r))

        time.sleep(4)
        with ThreadPoolExecutor(2) as pool:
            pair = list(pool.map(lambda n: stream(sup.base, {'fake_fail_after': n}), [8, 20]))
        results.append(check('c2: both streams resumed', all(p['content'] == EXPECTED and not p['errors'] for p in pair),
                             pair))

        time.sleep(4)
        r = stream(sup.base, {'fake_die_after': 6})
        results.append(check('child died mid-stream: clean error + [DONE]', r['done'] and r['errors']
                             and r['errors'][0].get('code') == 'backend_unavailable', r))
        r = stream(sup.base, {})
        results.append(check('next request after a child death works', r['content'] == EXPECTED and not r['errors'], r))

        r = stream(sup.base, {}, stop_after=5)
        time.sleep(1.5)
        h = sup.health()
        results.append(check('client disconnect releases the upstream', h['inflight'] == 0, h))

        box.down()
        r = stream(sup.base, {})
        results.append(check('box down at request start -> q3', r['backend'] == 'q3' and r['content'] == EXPECTED, r))
        box.up()

        NOSTART.write_text('1')
        time.sleep(4)
        r = stream(sup.base, {})
        h = sup.health()
        results.append(check('og fails to start on the way back -> served by q3', r['backend'] == 'q3'
                             and r['content'] == EXPECTED and not r['errors'] and h['backend'] == 'q3', (r, h)))
        NOSTART.unlink()
    finally:
        sup.stop()

    sup = Sup(BASE_PORT + 4, BASE_PORT, DS41_OG_RESTART_WAIT_S='4', DS41_OG_Q3='on')
    try:
        box.down()
        r = stream(sup.base, {})
        h = sup.health()
        results.append(check('box engine restarting (port refused): og keeps serving', r['backend'] == 'og'
                             and r['content'] == EXPECTED and h['box_restarting'], (r, h)))
        time.sleep(5)
        r = stream(sup.base, {})
        results.append(check('refused past DS41_OG_RESTART_WAIT_S -> q3', r['backend'] == 'q3' and r['content'] == EXPECTED, r))
        box.up()
    finally:
        sup.stop()

    image_msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'what is this?'},
                                               {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]
    # DS41_OG_VISION=og (default): images on the og worker (box vision), never on q3
    sup = Sup(BASE_PORT + 4, BASE_PORT, FAKE_OG_VISION='1', DS41_OG_RESTART_WAIT_S='1')
    try:
        r = stream(sup.base, {'messages': image_msgs})
        results.append(check('og vision: image stream served by og, no swap, no keepalives', r['backend'] == 'og'
                             and r['content'] == EXPECTED and not r['errors'] and r['done'] and not r['keepalives'], r))
        with post(sup.base, '/v1/chat/completions', dict(model='ds41', messages=image_msgs)) as resp:
            d = json.load(resp)
            results.append(check('og vision: non-streaming image on og', resp.headers.get('x-ds41-og-backend') == 'og'
                                 and d['choices'][0]['message']['content'] == EXPECTED, d))
        with post(sup.base, '/v1/messages', dict(model='ds41', max_tokens=8, messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAAA'}}]}])) as resp:
            d = json.load(resp)
            results.append(check('og vision: /v1/messages image block on og', resp.headers.get('x-ds41-og-backend') == 'og'
                                 and d['content'][0]['text'] == 'saw it', d))
        r = stream(sup.base, {})
        results.append(check('og vision: text right after an image stays on og', r['backend'] == 'og' and not r['errors'], r))
        box.down()
        time.sleep(3)
        try:
            post(sup.base, '/v1/chat/completions', dict(model='ds41', messages=image_msgs)).read()
            results.append(check('og vision: box down -> retryable 503, never q3', False, 'accepted'))
        except urllib.error.HTTPError as e:
            body = json.load(e)
            h = sup.health()
            results.append(check('og vision: box down -> retryable 503, never q3', e.code == 503
                                 and e.headers.get('Retry-After') and body['error']['code'] == 'images_unavailable'
                                 and 'q3' not in body['error']['message'] and h.get('mode') != 'q3', (body, h)))
        box.up()
    finally:
        sup.stop()

    # DS41_OG_VISION=q3: the old interim, kept only as an explicit, non-default setting
    sup = Sup(BASE_PORT + 4, BASE_PORT, FAKE_Q3_START_DELAY='2', DS41_OG_VISION='q3', DS41_OG_Q3='on')
    try:
        r = stream(sup.base, {})
        results.append(check('images: starts on og', r['backend'] == 'og' and r['content'] == EXPECTED, r))
        r = stream(sup.base, {'messages': image_msgs})
        results.append(check('images: stream switched og -> q3 with keepalives, no error', r['backend'] == 'q3'
                             and r['content'] == EXPECTED and not r['errors'] and r['done'] and r['keepalives'] >= 1, r))
        r = stream(sup.base, {})
        results.append(check('images: text right after stays on q3 (no swap back)', r['backend'] == 'q3' and not r['errors'], r))
        with post(sup.base, '/v1/chat/completions', dict(model='ds41', messages=image_msgs)) as resp:
            d = json.load(resp)
            results.append(check('images: non-streaming on q3', resp.headers.get('x-ds41-og-backend') == 'q3'
                                 and d['choices'][0]['message']['content'] == EXPECTED, d))
        with post(sup.base, '/v1/messages', dict(model='ds41', max_tokens=8, messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAAA'}}]}])) as resp:
            d = json.load(resp)
            results.append(check('images: /v1/messages image block on q3', resp.headers.get('x-ds41-og-backend') == 'q3'
                                 and d['content'][0]['text'] == 'saw it', d))
        time.sleep(4)
        r = stream(sup.base, {})
        results.append(check('images: back to og after RECOVER_S without images', r['backend'] == 'og' and not r['errors'], r))
        NOQ3.write_text('1')
        try:
            post(sup.base, '/v1/chat/completions', dict(model='ds41', messages=image_msgs)).read()
            results.append(check('images: q3 cannot start -> HTTP 400', False, 'accepted'))
        except urllib.error.HTTPError as e:
            body = json.load(e)
            results.append(check('images: q3 cannot start -> HTTP 400, OpenAI error shape', e.code == 400
                                 and body['error']['code'] == 'images_unavailable' and body['error']['message'], body))
        r = stream(sup.base, {'messages': image_msgs})
        results.append(check('images: q3 cannot start mid-stream -> well-formed SSE error + [DONE]', r['done']
                             and r['errors'] and r['errors'][0].get('code') == 'images_unavailable', r))
        NOQ3.unlink()
        r = stream(sup.base, {})
        results.append(check('images: text after a failed q3 start is served by og', r['backend'] == 'og'
                             and r['content'] == EXPECTED and not r['errors'], r))
    finally:
        sup.stop()

    sup = Sup(BASE_PORT + 4, BASE_PORT, DS41_OG_VISION='reject')
    try:
        try:
            post(sup.base, '/v1/chat/completions', dict(model='ds41', stream=True, messages=image_msgs)).read()
            results.append(check('images: DS41_OG_VISION=reject -> 400', False, 'accepted'))
        except urllib.error.HTTPError as e:
            body = json.load(e)
            results.append(check('images: DS41_OG_VISION=reject -> 400 JSON, no ds41-q3 suggestion', e.code == 400
                                 and body['error']['type'] == 'invalid_request_error' and 'q3' not in body['error']['message'], body))
    finally:
        sup.stop()

    # DS41_OG_Q3=off (default): og only, never a q3 child
    sup = Sup(BASE_PORT + 4, BASE_PORT, DS41_OG_RESTART_WAIT_S='1')
    try:
        def no_q3(h):
            return h and h.get('q3') == 0 and h.get('backend') in ('og', None) and not h.get('q3_enabled')
        r = stream(sup.base, {'fake_keepalive': 3})
        results.append(check('no-q3: stream on og; child keepalives carry the real model, never "keepalive"',
                             r['backend'] == 'og' and r['content'] == EXPECTED and r['keepalives'] == 3
                             and r['keepalive_model'] == 0, r))
        r = stream(sup.base, {'fake_fail_after': 12})
        h = sup.health()
        results.append(check('no-q3: box lost mid-stream -> clean retryable SSE error + [DONE], no q3',
                             r['done'] and r['errors'] and r['errors'][-1].get('code') == 'backend_unavailable'
                             and r['content'] == EXPECTED[:len('w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 ')]
                             and no_q3(h) and h['switches'] == 1, (r, h)))
        try:
            post(sup.base, '/v1/chat/completions', dict(model='ds41', fake_fail_after=9)).read()
            results.append(check('no-q3: non-streaming box loss -> 503', False, 'accepted'))
        except urllib.error.HTTPError as e:
            body = json.load(e)
            results.append(check('no-q3: non-streaming box loss -> 503 + Retry-After', e.code == 503
                                 and e.headers.get('Retry-After') and body['error']['code'] == 'backend_unavailable', body))
        r = stream(sup.base, {})
        results.append(check('no-q3: next request right after a loss is served by og (no trust delay)',
                             r['backend'] == 'og' and r['content'] == EXPECTED and not r['errors'], r))
        box.down()
        time.sleep(3)  # refused for longer than DS41_OG_RESTART_WAIT_S: the box is gone
        for streaming in (True, False):
            try:
                post(sup.base, '/v1/chat/completions', dict(model='ds41', stream=streaming,
                                                            messages=[{'role': 'user', 'content': 'hi'}])).read()
                results.append(check(f'no-q3: box down, new request (stream={streaming}) -> 503', False, 'accepted'))
            except urllib.error.HTTPError as e:
                body = json.load(e)
                results.append(check(f'no-q3: box down, new request (stream={streaming}) -> 503 + Retry-After, no q3',
                                     e.code == 503 and e.headers.get('Retry-After')
                                     and body['error']['code'] == 'backend_unavailable' and no_q3(sup.health()), body))
        box.up()
        time.sleep(2.5)
        r = stream(sup.base, {})
        results.append(check('no-q3: box back -> og at once', r['backend'] == 'og' and not r['errors'], r))
        r = stream(sup.base, {'fake_die_after': 6})
        NOSTART.write_text('1')
        try:
            post(sup.base, '/v1/chat/completions', dict(model='ds41', messages=[{'role': 'user', 'content': 'hi'}])).read()
            results.append(check('no-q3: og cannot start -> 503', False, 'accepted'))
        except urllib.error.HTTPError as e:
            h = sup.health()
            results.append(check('no-q3: og cannot start -> 503, never q3', e.code == 503
                                 and e.headers.get('Retry-After') and no_q3(h), (json.load(e), h)))
        NOSTART.unlink()
        r = stream(sup.base, {})
        results.append(check('no-q3: og starts again once it can', r['backend'] == 'og' and r['content'] == EXPECTED, r))
    finally:
        sup.stop()

    NOSTART.write_text('1')  # boot with og unable to start: health must not hang llama-swap
    proc = subprocess.Popen([PY, str(HERE/'ds41_og.py'), '--port', str(BASE_PORT + 4)], env=dict(
        os.environ, DS41_OG_BOX=f'127.0.0.1:{BASE_PORT}', DS41_OG_WORKER_PORT=str(BASE_PORT + 5),
        DS41_OG_CHILD_OG=f'{PY} {HERE/"fakechild.py"} --port {{port}} --mode og', DS41_OG_LOGS=tempfile.mkdtemp()),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base, h = f'http://127.0.0.1:{BASE_PORT + 4}', None
        for _ in range(60):
            try:
                with urllib.request.urlopen(base + '/health', timeout=3) as r:
                    h = (r.status, json.load(r))
                    break
            except urllib.error.HTTPError:
                time.sleep(0.5)
            except OSError:
                time.sleep(0.5)
        results.append(check('no-q3: og cannot start at boot -> health 200 "degraded" (llama-swap does not hang)',
                             h and h[0] == 200 and h[1]['status'] == 'degraded', h))
        try:
            post(base, '/v1/chat/completions', dict(model='ds41', messages=[{'role': 'user', 'content': 'hi'}])).read()
            results.append(check('no-q3: degraded -> request gets 503', False, 'accepted'))
        except urllib.error.HTTPError as e:
            results.append(check('no-q3: degraded -> request gets 503 + Retry-After', e.code == 503
                                 and e.headers.get('Retry-After')))
    finally:
        NOSTART.unlink()
        proc.terminate()
        proc.wait(timeout=60)

    sup = Sup(BASE_PORT + 4, BASE_PORT, DS41_OG_RESTART_WAIT_S='30')
    try:
        box.down()
        r = stream(sup.base, {})
        h = sup.health()
        results.append(check('no-q3: box engine restarting (refused, within the wait) -> og keeps serving',
                             r['backend'] == 'og' and r['content'] == EXPECTED and h['box_restarting'] and h['q3'] == 0, (r, h)))
        box.up()
    finally:
        sup.stop()

    sup = Sup(BASE_PORT + 7, BASE_PORT, DS41_OG_RECOVER_S='0', DS41_OG_MAX_RESUMES='2', DS41_OG_Q3='on')
    try:
        r = stream(sup.base, {'fake_fail_after': 4})
        h = sup.health()
        results.append(check('resumes exhausted -> clean retryable error', r['done'] and r['errors']
                             and r['errors'][-1].get('code') == 'backend_unavailable' and h['resume_failed'] == 1, (r, h)))
    finally:
        sup.stop()
    ok = all(results)
    print('ALL PASS' if ok else 'SOME FAILED', f'({sum(results)}/{len(results)})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
