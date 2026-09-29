"""CPU-only robustness tests for the ds41-og supervisor (fake children, no MLX/GPU/box).

Run: SUP_TEST_PORT=12600 python og_serve/test_robust.py   (uses SUP_TEST_PORT .. +9)
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('SUP_TEST_PORT', '12600')
from test_supervisor_failover import EXPECTED, BoxPort, Sup, check, stream  # noqa: E402

BASE = int(os.environ['SUP_TEST_PORT'])


def get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def wait_for(fn, limit):
    deadline = time.time() + limit
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.2)
    return None


def raw_post_then_hang_up(port, body, after_s):
    """A non-streaming client that gives up (closes its socket) after `after_s`."""
    data = json.dumps(body).encode()
    s = socket.create_connection(('127.0.0.1', port))
    s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
              + f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
    time.sleep(after_s)
    s.close()


class OldChild:
    """A previous supervisor's og child still draining on the fixed worker port: answers /health, then exits."""

    def __init__(self, port, life_s):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.srv = ThreadingHTTPServer(('127.0.0.1', port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        threading.Timer(life_s, self.stop).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def main():
    box = BoxPort(BASE)
    results = []

    # 1. A worker that dies is restarted in the background (no request needed); a request that arrives while it
    #    loads waits for it instead of getting a 503 "no backend".
    sup = Sup(BASE + 1, BASE, DS41_OG_RESTART_WAIT_S='1', FAKE_OG_START_DELAY='3', DS41_OG_WORKER_RESTART_S='5')
    try:
        first = sup.health()['pid']
        r = stream(sup.base, {'fake_die_after': 6})
        results.append(check('worker death mid-stream -> clean error', r['errors'] and r['done'], r))
        h = wait_for(lambda: (sup.health() or {}).get('worker_restarting') and sup.health(), 5)
        results.append(check('dead worker noticed and restarted without a request', h and h['worker_exits'] == 1
                             and h['worker_exit_code'] == 3, h))
        r = stream(sup.base, {})
        results.append(check('request during the restart waits for the new worker (no 503)',
                             r['content'] == EXPECTED and not r['errors'] and r['backend'] == 'og', r))
        h = sup.health()
        results.append(check('health: new worker alive, 1 restart', h['worker_alive'] and h['worker_restarts'] == 1
                             and h['pid'] != first and h['status'] == 'ok', h))
    finally:
        sup.stop()

    # 2. A non-streaming client that hangs up cancels its request upstream (the worker aborts it).
    sup = Sup(BASE + 3, BASE, DS41_OG_RESTART_WAIT_S='1')
    try:
        raw_post_then_hang_up(BASE + 3, dict(model='ds41', messages=[{'role': 'user', 'content': 'hi'}], fake_slow_s=8), 1.0)
        stats = wait_for(lambda: get(f'http://127.0.0.1:{BASE + 4}/fake/stats').get('cancelled') and
                         get(f'http://127.0.0.1:{BASE + 4}/fake/stats'), 5)
        h = sup.health()
        results.append(check('non-streaming client gone -> upstream cancelled within seconds', stats and h['inflight'] == 0
                             and h['client_gone'] == 1, (stats, h)))
        with urllib.request.urlopen(urllib.request.Request(
                sup.base + '/v1/chat/completions', json.dumps(dict(model='ds41', fake_slow_s=0.5)).encode(),
                {'Content-Type': 'application/json'}), timeout=30) as resp:
            d = json.load(resp)
        results.append(check('a patient non-streaming client still gets its answer',
                             d['choices'][0]['message']['content'] == EXPECTED, d))
    finally:
        sup.stop()

    # 3. llama-swap reload: the previous child still answers /health on the fixed port for a while. The new
    #    supervisor must not report ready off the old child's /health.
    old = OldChild(BASE + 7, 3.0)
    t0 = time.time()
    sup = Sup(BASE + 6, BASE, DS41_OG_RESTART_WAIT_S='1')
    try:
        ready_after = time.time() - t0
        r = stream(sup.base, {})
        results.append(check('reload: ready only after the old child left, served by the new child',
                             ready_after >= 2.5 and r['content'] == EXPECTED and not r['errors'], (ready_after, r)))
    finally:
        sup.stop()
        old.stop() if old.srv.socket.fileno() >= 0 else None
    box.down()
    ok = all(results)
    print('ALL PASS' if ok else 'SOME FAILED', f'({sum(results)}/{len(results)})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
