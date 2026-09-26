"""Supervisor failover test (one Mac lease, <=15 min).

A local TCP relay stands in for the box link so the test can take the box
away and bring it back: og request -> relay down -> q3 fallback request ->
relay up -> og again -> TERM the supervisor and check that its GPU child is
reaped. Stops the served llama-swap ds41 first (by its own PID).
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HOME = Path.home()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'benchmarks/og'))
from lease import stop_served  # noqa: E402

LOGS = Path(os.environ.get('OG_LOGS', str(HOME/'llm/ds41/og-mac')))
PORT, RELAY = 12149, 12150
BASE = f'http://127.0.0.1:{PORT}'
out = (LOGS/'og-s1.sup.jsonl').open('a')


def log(**record):
    record = dict(t=round(time.time(), 1), **record)
    out.write(json.dumps(record)+'\n'); out.flush(); print(json.dumps(record), flush=True)


class Relay(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.srv = socket.socket(); self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(('127.0.0.1', RELAY)); self.srv.listen(16); self.conns = []; self.alive = True

    def run(self):
        while self.alive:
            try:
                a, _ = self.srv.accept()
            except OSError:
                return
            try:
                b = socket.create_connection(('10.10.10.1', 10052), timeout=3)
                b.settimeout(None)
            except OSError:
                a.close()  # behave like a refusing box
                continue
            for s in (a, b):
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.conns += [a, b]
            for x, y in ((a, b), (b, a)):
                threading.Thread(target=self.pipe, args=(x, y), daemon=True).start()

    @staticmethod
    def pipe(x, y):
        try:
            while True:
                data = x.recv(1 << 20)
                if not data:
                    break
                y.sendall(data)
        except OSError:
            pass
        for s in (x, y):
            try:
                s.close()
            except OSError:
                pass

    def stop(self):
        self.alive = False
        self.srv.close()
        for s in self.conns:
            try:
                s.close()
            except OSError:
                pass


def chat(content, max_tokens=48):
    body = json.dumps(dict(model='ds41-og', messages=[{'role': 'user', 'content': content}], max_tokens=max_tokens,
                           temperature=0)).encode()
    t0 = time.time()
    req = urllib.request.Request(BASE+'/v1/chat/completions', body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
        return dict(backend=r.headers.get('x-ds41-og-backend'), text=d['choices'][0]['message']['content'][:80],
                    seconds=round(time.time()-t0, 1))


def health():
    try:
        with urllib.request.urlopen(BASE+'/health', timeout=3) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)
    except OSError:
        return None


def children(pid):
    return subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True).stdout.split()


def main():
    deadline = time.monotonic() + 840
    stop_served(lambda r: log(**r))
    relay = Relay(); relay.start()
    env = dict(os.environ, DS41_OG_BOX=f'127.0.0.1:{RELAY}', DS41_OG_RECOVER_S='15', DS41_OG_LEASE_S='840',
               DS41_OG_WORKER_PORT='12147', DS41_OG_Q3_PORT='12148', DS41_OG_LOGS=str(LOGS/'og-s1-children'))
    sup = subprocess.Popen([str(HERE/'ds41-og'), '--host', '127.0.0.1', '--port', str(PORT)], env=env,
                           stdout=(LOGS/'og-s1.supervisor.log').open('a'), stderr=subprocess.STDOUT)
    log(event='supervisor', pid=sup.pid)
    try:
        while (h := health()) is None or h.get('status') != 'ok':
            if sup.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'supervisor not ready: {h}')
            time.sleep(2)
        log(event='ready', health=h)
        log(event='request1', **chat('Say ready.'))
        relay.stop()
        log(event='relay_down')
        log(event='request2', **chat('Name three prime numbers.'))
        log(event='health', health=health())
        relay = Relay(); relay.start()
        log(event='relay_up')
        time.sleep(20)
        log(event='request3', **chat('Say ready again.'))
        log(event='health', health=health())
    finally:
        kids = children(sup.pid)
        sup.terminate()
        try:
            sup.wait(timeout=180)
        except subprocess.TimeoutExpired:
            sup.kill(); sup.wait()
        time.sleep(2)
        alive = [k for k in kids if subprocess.run(['ps', '-p', k], capture_output=True).returncode == 0]
        log(event='supervisor_stopped', rc=sup.returncode, children=kids, children_alive=alive,
            lock=subprocess.run(['lsof', '-t', str(HOME/'llm/locks/gpu.lock')], capture_output=True, text=True).stdout.split())
        relay.stop()


if __name__ == '__main__':
    main()
