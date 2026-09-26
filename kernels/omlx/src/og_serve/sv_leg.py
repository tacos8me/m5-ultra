"""One guarded Mac GPU leg for og-serve: supervisor (this tree) behind a controllable box relay.

Stops the served llama-swap ds41 by its own PID, starts a TCP relay 127.0.0.1:12164 ->
10.10.10.1:10052 that the test can cut, starts og_serve/ds41-og on :12161 (og worker :12162,
q3 child :12163, both through gpu-exec with their 245 GiB watchdogs), runs the scenarios, then
TERMs the supervisor, waits for its children to exit and checks gpu.lock is free.

Scenarios: parity, tierA, resume_og, isolation, tierB, back_to_og.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

HOME = Path.home()
HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE/'benchmarks/og'))
sys.path.insert(0, str(HERE))
from lease import stop_served  # noqa: E402
import parity  # noqa: E402

LOGS = Path(os.environ.get('OG_LOGS', str(HOME/'llm/ds41/og-serve')))
SUP, WORKER, Q3, RELAY = 12161, 12162, 12163, 12164
BOX = ('10.10.10.1', 10052)


class Relay:
    """127.0.0.1:RELAY -> box. up() / down() (refuse connections) / kill() (drop live ones) / drop_new."""

    def __init__(self):
        self.conns, self.srv, self.drop_new = [], None, False
        self.lock = threading.Lock()
        self.up()

    def up(self):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', RELAY))
        srv.listen(32)
        self.srv = srv
        threading.Thread(target=self._accept, args=(srv,), daemon=True).start()

    def _accept(self, srv):
        while True:
            try:
                a, _ = srv.accept()
            except OSError:
                return
            if self.drop_new:
                a.close()
                continue
            try:
                b = socket.create_connection(BOX, timeout=3)
                b.settimeout(None)
            except OSError:
                a.close()
                continue
            for s in (a, b):
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.conns += [a, b]
            for x, y in ((a, b), (b, a)):
                threading.Thread(target=self._pipe, args=(x, y), daemon=True).start()

    @staticmethod
    def _pipe(x, y):
        try:
            while data := x.recv(1 << 20):
                y.sendall(data)
        except OSError:
            pass
        for s in (x, y):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def kill(self):
        with self.lock:
            conns, self.conns = self.conns, []
        for s in conns:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def down(self):
        if self.srv is not None:
            try:
                self.srv.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.srv.close()
            self.srv = None
        self.kill()


def hold_gpu_lock(wait_s):
    """flock ~/llm/locks/gpu.lock for this process (as gpu-exec does); the children run without gpu-exec."""
    path = HOME/'llm/locks/gpu.lock'

    def timeout(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, timeout)
    deadline = time.monotonic() + wait_s
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        signal.setitimer(signal.ITIMER_REAL, max(1, deadline - time.monotonic()))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except TimeoutError:
            os.close(fd)
            raise SystemExit(f'gpu.lock busy for {wait_s:.0f}s; not starting')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if os.fstat(fd).st_ino == os.stat(path).st_ino:
            return fd
        os.close(fd)


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except ValueError:
            return {'status': e.code}
    except OSError:
        return None


def document(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
    src = TREE/'omlx/patches/deepseek_v41'
    corpus = '\n\n'.join(f'File: {f.name}\n{f.read_text()}' for f in sorted(src.glob('*.py')))
    for s in tok.all_special_tokens:
        corpus = corpus.replace(s, '[special token literal]')
    corpus = corpus.replace('｜', '|').replace('<think>', '[think tag]').replace('</think>', '[/think tag]')
    base = tok.encode(corpus, add_special_tokens=False)
    return tok, tok.decode((base * (n // len(base) + 2))[:n])


def stream_with_fault(base, model, messages, max_tokens, fault=None, at_chars=300):
    """Stream; once `at_chars` characters of content have arrived, run fault() once (in a thread)."""
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0, stream=True,
                stream_options={'include_usage': True})
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    out = dict(content='', errors=[], usage=None, done=False, chunks=0, keepalives=0, ids=set(), gaps=[], finish=None)
    t0 = last = time.perf_counter()
    fired = False
    with urllib.request.urlopen(req, timeout=1800) as r:
        out['backend'] = r.headers.get('x-ds41-og-backend')
        for raw in r:
            line = raw.strip()
            if not line.startswith(b'data: '):
                continue
            if line == b'data: [DONE]':
                out['done'] = True
                continue
            d = json.loads(line[6:])
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
                piece = (c.get('delta') or {}).get('content') or ''
                if piece:
                    now = time.perf_counter()
                    out['gaps'].append(now - last)
                    last = now
                    out['content'] += piece
                    out['chunks'] += 1
                out['finish'] = c.get('finish_reason') or out['finish']
            if fault is not None and not fired and len(out['content']) >= at_chars:
                fired = True
                out['fault_at_chars'] = len(out['content'])
                threading.Thread(target=fault, daemon=True).start()
    out['s'] = time.perf_counter() - t0
    out['fired'] = fired
    out['max_gap_s'] = max(out['gaps'] or [0])
    out['ids'] = sorted(i for i in out['ids'] if i)
    del out['gaps']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--scenarios', default='parity,tierA,resume_og,isolation')
    ap.add_argument('--parity-cases', default='')
    ap.add_argument('--lease-seconds', type=int, default=840)
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--direct-box', action='store_true', help='no relay: the worker talks to 10.10.10.1:10052')
    ap.add_argument('--soak-hours', type=float, default=0.33)
    ap.add_argument('--bench', default='--ttft 8192,131072,524288,1048576 --resume 131072:2 '
                                       '--c1 8192:3,131072:3,524288:2 --c2 8192:2,131072:1')
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    events = (LOGS/f'{args.label}.leg.jsonl').open('a')
    deadline = time.monotonic() + args.lease_seconds - 60

    def log(**record):
        record = dict(t=round(time.time(), 1), **record)
        events.write(json.dumps(record, default=str) + '\n'); events.flush()
        print(json.dumps(record, default=str)[:1500], flush=True)

    with socket.socket() as s:
        s.settimeout(3)
        if s.connect_ex(BOX):
            raise SystemExit('box 10052 down; not starting')
    waited = time.monotonic()
    stop_served(lambda r: log(**r))
    lock_fd = hold_gpu_lock(float(os.environ.get('SV_LOCK_WAIT', '1500')))
    log(event='lock_held', waited_s=round(time.monotonic() - waited, 1))
    deadline = time.monotonic() + args.lease_seconds - 60
    relay = Relay()
    box = f'{BOX[0]}:{BOX[1]}' if args.direct_box else f'127.0.0.1:{RELAY}'
    env = dict(os.environ, DS41_TREE=str(TREE), DS41_OG_BOX=box, DS41_OG_WORKER_PORT=str(WORKER),
               DS41_OG_Q3_PORT=str(Q3), DS41_OG_LOGS=str(LOGS/f'{args.label}-children'), DS41_OG_LEASE_S=str(args.lease_seconds),
               DS41_OG_GPU_EXEC='')
    if not args.direct_box:  # relay tests: shorter waits (relay down = port refused)
        env.update(DS41_OG_RESUME_WAIT_S='20', DS41_OG_RESTART_WAIT_S='20', DS41_OG_RECOVER_S='20')
    env.update(dict(x.split('=', 1) for x in args.env))
    sup = subprocess.Popen([str(HERE/'ds41-og'), '--host', '127.0.0.1', '--port', str(SUP)], env=env,
                           stdout=(LOGS/f'{args.label}.supervisor.log').open('a'), stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{SUP}'
    log(event='supervisor', pid=sup.pid)
    results = {}
    try:
        while (h := get(base + '/health')) is None or h.get('status') != 'ok':
            if sup.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'supervisor not ready: {h}')
            time.sleep(2)
        log(event='ready', health=h)
        scenarios = [s for s in args.scenarios.split(',') if s]
        tok = doc8k = None
        if any(s in scenarios for s in ('tierA', 'resume_og', 'tierB')):
            tok, doc8k = document(8000)
        messages = [{'role': 'user', 'content': (doc8k or '') + '\n\nExplain in detail what this code does, file by file.'}]

        if 'parity' in scenarios:
            cases = args.parity_cases or ','.join(parity.BASE)
            rc = subprocess.run([sys.executable, str(HERE/'parity.py'), '--label', args.label, '--cases', cases,
                                 '--target', f'og={base}/v1@ds41-og'], env=dict(env, OG_LOGS=str(LOGS)),
                                timeout=max(60, deadline - time.monotonic())).returncode
            results['parity'] = json.loads((LOGS/f'{args.label}.parity.json').read_text())['failures']
            log(event='parity', rc=rc, failures=results['parity'])

        if 'soak' in scenarios:
            rc = subprocess.run([sys.executable, str(HERE/'soak.py'), '--label', args.label, '--base', f'{base}/v1',
                                 '--model', 'ds41-og', '--hours', str(args.soak_hours), '--health', base,
                                 '--og-stats', f'http://127.0.0.1:{WORKER}/og/stats'],
                                env=dict(env, OG_LOGS=str(LOGS))).returncode
            results['soak'] = json.loads((LOGS/f'{args.label}.soak.json').read_text())
            log(event='soak', rc=rc, requests=results['soak']['requests'], errors=results['soak']['errors'],
                footprint=results['soak']['footprint'])

        if 'bench' in scenarios and time.monotonic() < deadline:
            rc = subprocess.run([sys.executable, str(HERE/'bench_api.py'), '--label', args.label, '--base', base,
                                 '--model', 'ds41-og', '--budget', str(max(60, deadline - time.monotonic() - 30)),
                                 *args.bench.split()], env=dict(env, OG_LOGS=str(LOGS))).returncode
            results['bench'] = json.loads((LOGS/f'{args.label}.bench.json').read_text())
            log(event='bench', rc=rc, summary=results['bench'])

        if 'tierA' in scenarios and time.monotonic() < deadline:
            ref = stream_with_fault(base, 'ds41-og', messages, 400)
            log(event='tierA_reference', chars=len(ref['content']), usage=ref['usage'], s=ref['s'])
            before = len((get(f'http://127.0.0.1:{WORKER}/og/stats') or {}).get('recoveries') or [])
            kill = stream_with_fault(base, 'ds41-og', messages, 400, fault=relay.kill, at_chars=400)
            recs = (get(f'http://127.0.0.1:{WORKER}/og/stats') or {}).get('recoveries') or []
            log(event='tierA_kill', fired=kill['fired'], identical=kill['content'] == ref['content'], errors=kill['errors'],
                max_gap_s=kill['max_gap_s'], usage=kill['usage'], recoveries=recs[before:])
            kill_ok = kill['fired'] and len(recs) > before and kill['content'] == ref['content'] and not kill['errors']

            def outage():
                relay.down()
                time.sleep(8)
                relay.up()
            down = stream_with_fault(base, 'ds41-og', messages, 400, fault=outage, at_chars=700)
            recs2 = (get(f'http://127.0.0.1:{WORKER}/og/stats') or {}).get('recoveries') or []
            log(event='tierA_outage8s', fired=down['fired'], identical=down['content'] == ref['content'], errors=down['errors'],
                max_gap_s=down['max_gap_s'], keepalives=down['keepalives'], usage=down['usage'], recoveries=recs2[len(recs):])
            results['tierA'] = dict(kill=bool(kill_ok),
                                    outage=bool(down['fired'] and len(recs2) > len(recs) and down['content'] == ref['content']
                                                and not down['errors'] and down['max_gap_s'] > 7))
            if not results['tierA']['kill'] or not results['tierA']['outage']:
                for name, r in (('ref', ref), ('kill', kill), ('outage', down)):
                    (LOGS/f'{args.label}.tierA.{name}.txt').write_text(r['content'])

        if 'resume_og' in scenarios and time.monotonic() < deadline:
            ref = stream_with_fault(f'http://127.0.0.1:{WORKER}', 'ds41-og', messages, 200)
            ids = tok.encode(ref['content'], add_special_tokens=False)[:80]
            prefix = tok.decode(ids)
            body = dict(model='ds41-og', messages=messages, max_tokens=200, temperature=0, stream=True,
                        stream_options={'include_usage': True}, ds41_resume=dict(output=ids))
            req = urllib.request.Request(f'http://127.0.0.1:{WORKER}/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json', 'x-ds41-resume': '1'})
            content, usage = '', None
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw in r:
                    line = raw.strip()
                    if line.startswith(b'data: {'):
                        d = json.loads(line[6:])
                        usage = d.get('usage') or usage
                        for c in d.get('choices') or []:
                            content += (c.get('delta') or {}).get('content') or ''
            ok = content.startswith(prefix) and usage and usage['completion_tokens'] <= 200
            results['resume_og'] = dict(replay_prefix_ok=content.startswith(prefix),
                                        same_as_reference=content == ref['content'])
            log(event='resume_og', **results['resume_og'], replayed=len(ids), usage=usage, ref_usage=ref['usage'],
                ok=bool(ok), stats=(get(f'http://127.0.0.1:{WORKER}/og/stats') or {}).get('resume'))
            if not content == ref['content']:
                (LOGS/f'{args.label}.resume_og.ref.txt').write_text(ref['content'])
                (LOGS/f'{args.label}.resume_og.resumed.txt').write_text(content)

        if 'isolation' in scenarios and time.monotonic() < deadline:
            holder = {}

            def long_stream():
                holder['a'] = stream_with_fault(f'http://127.0.0.1:{WORKER}', 'ds41-og',
                                                [{'role': 'user', 'content': 'Write a long, detailed story about a lighthouse.'}], 600)
            th = threading.Thread(target=long_stream)
            th.start()
            time.sleep(6)
            relay.drop_new = True
            t0 = time.perf_counter()
            b = stream_with_fault(f'http://127.0.0.1:{WORKER}', 'ds41-og',
                                  [{'role': 'user', 'content': 'Say hello.'}], 20)
            relay.drop_new = False
            th.join(timeout=300)
            a = holder.get('a', {})
            results['isolation'] = dict(b_failed_cleanly=bool(b['errors']) and b['done'],
                                        b_has_marker='ds41-og-resume' in json.dumps(b['errors']),
                                        a_unaffected=bool(a) and not a['errors'] and a.get('done'))
            log(event='isolation', **results['isolation'], b_s=round(time.perf_counter() - t0, 1),
                b_error=json.dumps(b['errors'])[:300], a_usage=a.get('usage'))

        if 'tierB' in scenarios and time.monotonic() < deadline:
            ref = stream_with_fault(base, 'ds41-og', messages, 500)
            peer = {}

            def second():
                peer['r'] = stream_with_fault(base, 'ds41-og', [{'role': 'user', 'content':
                                                                 'Write a long, detailed story about a lighthouse keeper.'}], 700)
            th = threading.Thread(target=second)
            th.start()
            time.sleep(1.5)
            r = stream_with_fault(base, 'ds41-og', messages, 500, fault=relay.down, at_chars=400)
            th.join(timeout=600)
            h = get(base + '/health')
            cut = r.get('fault_at_chars') or 0
            p = peer.get('r') or {}
            results['tierB'] = dict(fired=r['fired'], no_error=not r['errors'] and r['done'], one_id=len(r['ids']) == 1,
                                    prefix_matches_og=r['content'][:cut] == ref['content'][:cut],
                                    peer_no_error=bool(p) and not p['errors'] and p['done'],
                                    backend_after=h and h.get('backend'), resumed=h and h.get('resumed'),
                                    divergent=h and h.get('divergent'))
            log(event='tierB', **results['tierB'], chars=len(r['content']), max_gap_s=r['max_gap_s'],
                keepalives=r['keepalives'], usage=r['usage'], peer_usage=p.get('usage'), fault_at=cut, health=h)
            (LOGS/f'{args.label}.tierB.txt').write_text(r['content'])
            (LOGS/f'{args.label}.tierB.ref.txt').write_text(ref['content'])
            if 'back_to_og' in scenarios:
                relay.up()
                time.sleep(25)
                r2 = stream_with_fault(base, 'ds41-og', [{'role': 'user', 'content': 'Say ready.'}], 8)
                h = get(base + '/health')
                results['back_to_og'] = dict(backend=r2['backend'], ok=not r2['errors'] and r2['done'])
                log(event='back_to_og', **results['back_to_og'], s=round(r2['s'], 1), health=h)
    finally:
        kids = subprocess.run(['pgrep', '-P', str(sup.pid)], capture_output=True, text=True).stdout.split()
        sup.terminate()
        try:
            sup.wait(timeout=180)
        except subprocess.TimeoutExpired:
            sup.kill(); sup.wait()
        for _ in range(60):
            alive = [k for k in kids if subprocess.run(['ps', '-p', k], capture_output=True).returncode == 0]
            if not alive:
                break
            time.sleep(1)
        holders = subprocess.run(['lsof', '-t', str(HOME/'llm/locks/gpu.lock')], capture_output=True, text=True).stdout.split()
        relay.down()
        log(event='stopped', rc=sup.returncode, children=kids, children_alive=alive, lock_holders=holders,
            results=results)
        os.close(lock_fd)


if __name__ == '__main__':
    main()
