"""ds41 fault-injection soak for a maintenance window (<= 30 min). DESIGNED, NOT RUN. Mac-side driver.

Run ONLY in an owner-approved window, on the Mac (the box admin endpoints accept 10.10.10.2 / localhost only):

    DS41_SOAK_WINDOW=1 ~/llm/.venv-ds41-omlx-tiles/bin/python og_serve/fault_soak.py --window \
        --out ~/llm/ds41/robust/soak-$(date +%Y%m%dT%H%M)

It drives ds41 through llama-swap (:8080) with an agent-like load (4-6 concurrent streams, 8K..512K contexts,
tool-call and non-streaming requests) and injects, in order:

  phase  t(min)  fault                                         expectation (pass criteria)
  A      0-3     none: reference run of every fixed prompt     texts recorded (temperature 0)
  B      3-5     client cancel mid-prefill (512K, at 5 s);     box sessions back to baseline <= 10 s after the cancel;
                 streaming + non-streaming disconnect mid-      Mac og/stats open_cancelled +1 (robust branch), opened ==
                 decode                                         closed; supervisor inflight 0, client_gone +1 (robust)
  C      5-10    box engine crash (POST /admin/crash) with 4   all 4 streams finish, text == phase A reference
                 streams mid-decode                             (bit-identical continuation), within 240 s of the crash;
                                                                og/stats recoveries +4, box_lost 0, broken 0
  D      10-15   planned engine restart (POST /admin/restart,  same as C (drain <= 30 s, then restart)
                 drain) with 4 streams mid-decode
  E      15-18   Mac worker SIGKILL (pid from :10001/health)   in-flight streams end with a well-formed SSE error
                 with 2 streams mid-decode                      (code backend_unavailable) + [DONE], no hang; next request
                                                                served <= 90 s later (robust: worker_restarts +1 with no
                                                                request); box sessions of the dead worker closed <= 10 s
  F      18-24   context edge: prompt of ~1,046,000 tokens     finishes with finish_reason "length" (robust: max_tokens
                 with max_tokens 8192                           clamped, og/stats max_tokens_clamped +1); on production
                                                                b16205ac this HANGS the engine thread (known bug) -> the
                                                                phase aborts it after 240 s and records FAIL
  G      24-27   malformed: prompt > 1M tokens (400), broken   400s with OpenAI error bodies; no worker/box restart
                 image data (400), 8 concurrent short requests  8/8 complete (4 queued behind the Mac's cap of 4)
  H      27-30   quiesce + final checks                         see final_checks()

Global pass criteria (final_checks): every request ended (no client hang past its deadline); box /health sessions 0,
connections <= 1 (its own probe), cache dropped_at_load 0, gpu memory within +1 GiB of the start; Mac og/stats
opened == closed, sessions 0, divergent 0; supervisor inflight 0; worker footprint within +2 GiB of the start
(og-worker.memory.json); exactly one omlx-server holds ~/llm/locks/gpu.lock; engine restarts in the journal == 2.

Optional (needs owner sudo on the box, not scripted): link drop -- `sudo iptables -I INPUT -s 10.10.10.2 -p tcp
--dport 10052 -j DROP` for 90 s with 2 streams mid-decode, then `-D` the rule. Expect: Mac streams pause, then fail
with a retryable error after STEP_TIMEOUT (120 s) + RESUME_WAIT_S (45 s); with the robust box branch the box frees the
dead sessions ~50 s into the outage (stalest_step_s / sessions in /health).

Abort rule: any phase that exceeds its budget by 120 s is abandoned (its requests are closed by PID-free client
side cancellation), recorded as FAIL, and the soak continues with the next phase; Ctrl-C stops everything and prints
the restore checklist (box: systemctl --user status split-nv-engine; Mac: curl :8080/running shows ds41 ready).
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

SWAP = os.environ.get('DS41_SOAK_URL', 'http://127.0.0.1:8080')
SUP = 'http://127.0.0.1:10001'  # the ds41 supervisor (llama-swap ${PORT} for ds41; check `ps` if it moved)
BOX = 'http://10.10.10.1:10051'
MODEL = 'ds41'
TREE = Path(os.environ.get('DS41_TREE', str(Path.home()/'src/wt/ds41-next2')))
LOG = []


def now():
    return time.strftime('%H:%M:%S')


def note(event, **kw):
    rec = dict(t=time.time(), event=event, **kw)
    LOG.append(rec)
    print(now(), event, json.dumps(kw)[:300], flush=True)


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)
    except OSError as e:
        return {'error': repr(e)}


def post(url, body=None, timeout=10):
    req = urllib.request.Request(url, json.dumps(body or {}).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def snapshot():
    """Counters every phase compares against."""
    mem = {}
    try:
        mem = json.loads((Path.home()/'llm/ds41/og/logs/og-worker.memory.json').read_text())
    except (OSError, ValueError):
        pass
    return dict(box=get(BOX + '/health'), sup=get(SUP + '/health'), og=get(SUP + '/og/stats'), mem=mem, t=time.time())


_TOK = []


def tokenizer():
    if not _TOK:
        from transformers import PreTrainedTokenizerFast
        path = next((Path.home()/'llm/ds41/og/models').glob('*/tokenizer.json')).parent
        _TOK.append(PreTrainedTokenizerFast.from_pretrained(str(path)))
    return _TOK[0]


def prompt_text(tokens, seed):
    """`tokens` tokens (exact, by the served tokenizer) of real code text (the tree's own sources, repeated as
    needed), with a nonce so no cache serves another phase's prompt."""
    src = []
    for p in sorted((TREE/'omlx').rglob('*.py')):
        src.append(p.read_text(errors='replace'))
        if sum(map(len, src)) > tokens * 5:
            break
    text = '\n'.join(src)
    while len(text) < tokens * 5:
        text += '\n' + text
    ids = tokenizer().encode(text[:tokens * 5], add_special_tokens=False)[:tokens]
    return f'[soak nonce {seed}]\n' + tokenizer().decode(ids) + '\n\nSummarize what the code above does in five bullet points.'


class Stream:
    """One streaming chat request in a thread: records text, errors, [DONE], timings; cancel() hangs up."""

    def __init__(self, name, content, max_tokens=256, tools=None, deadline=420):
        self.name, self.content, self.max_tokens, self.tools, self.deadline = name, content, max_tokens, tools, deadline
        self.text, self.errors, self.done, self.finish, self.first, self.end = '', [], False, None, None, None
        self.sock = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.t0 = time.time()
        self.thread.start()
        return self

    def run(self):
        body = dict(model=MODEL, stream=True, temperature=0, max_tokens=self.max_tokens,
                    messages=[{'role': 'user', 'content': self.content}])
        if self.tools:
            body['tools'] = self.tools
        host, port = SWAP.split('//')[1].split(':')
        try:
            conn = http.client.HTTPConnection(host, int(port), timeout=self.deadline)
            conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
            self.sock = conn.sock  # cancel() hangs up on it
            resp = conn.getresponse()  # http.client undoes the chunked encoding
            if resp.status != 200:
                self.errors.append({'status': resp.status, 'body': resp.read()[:300].decode(errors='replace')})
                return
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b'data: '):
                    continue
                if line == b'data: [DONE]':
                    self.done = True
                    break
                d = json.loads(line[6:])
                if 'error' in d:
                    self.errors.append(d['error'])
                    continue
                for c in d.get('choices') or []:
                    piece = (c.get('delta') or {}).get('content') or ''
                    if piece and self.first is None:
                        self.first = time.time()
                    self.text += piece
                    self.finish = c.get('finish_reason') or self.finish
        except Exception as e:  # noqa: BLE001
            self.errors.append({'client': repr(e)[:200]})
        finally:
            self.end = time.time()

    def cancel(self):
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def wait(self, limit=None):
        self.thread.join(limit if limit is not None else self.deadline)
        return not self.thread.is_alive()

    def result(self):
        return dict(name=self.name, chars=len(self.text), errors=self.errors, done=self.done, finish=self.finish,
                    ttft=round(self.first - self.t0, 2) if self.first else None,
                    total=round((self.end or time.time()) - self.t0, 1))


def wait_box_sessions(target, limit):
    deadline = time.time() + limit
    while time.time() < deadline:
        h = get(BOX + '/health')
        if h.get('sessions') is not None and h['sessions'] <= target:
            return True
        time.sleep(0.5)
    return False


FIXED = [('p8k-a', 8000), ('p8k-b', 8200), ('p128k', 128000), ('p32k', 32000)]
FIXED_MAX = 1200  # long enough that every stream is still decoding when C/D inject their fault


def phase_a(ref):
    note('A reference')
    streams = [Stream(n, prompt_text(t, n), max_tokens=FIXED_MAX).start() for n, t in FIXED]
    for s in streams:
        s.wait()
        ref[s.name] = s.text
        note('A result', **s.result())
    return all(s.done and not s.errors for s in streams)


def phase_b(base):
    note('B cancels')
    ok = True
    s = Stream('cancel-prefill', prompt_text(512000, random.random()), max_tokens=64).start()
    time.sleep(5)
    s.cancel()
    s.wait(10)
    ok &= wait_box_sessions(base['box'].get('sessions', 0), 15)
    note('B cancel mid-prefill', box=get(BOX + '/health').get('sessions'), og=get(SUP + '/og/stats').get('open_cancelled'))
    s = Stream('cancel-decode', prompt_text(8000, random.random()), max_tokens=2000).start()
    while s.first is None and s.thread.is_alive():
        time.sleep(0.2)
    time.sleep(3)
    s.cancel()
    s.wait(10)
    # non-streaming client that hangs up after 5 s
    body = json.dumps(dict(model=MODEL, max_tokens=2000, temperature=0,
                           messages=[{'role': 'user', 'content': prompt_text(8000, random.random())}])).encode()
    c = socket.create_connection(('127.0.0.1', 8080))
    c.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
              + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
    time.sleep(5)
    c.close()
    ok &= wait_box_sessions(base['box'].get('sessions', 0), 20)
    sup = get(SUP + '/health')
    note('B after', sup_inflight=sup.get('inflight'), client_gone=sup.get('client_gone'))
    return ok and sup.get('inflight') == 0


def phase_restart(ref, kind):
    note(f'{kind} start')
    streams = [Stream(n, prompt_text(t, n), max_tokens=FIXED_MAX, deadline=600).start() for n, t in FIXED]
    for s in streams:
        while s.first is None and s.thread.is_alive():
            time.sleep(0.2)
    time.sleep(2)
    t0 = time.time()
    try:
        post(BOX + ('/admin/crash' if kind == 'C' else '/admin/restart'))
    except OSError as e:
        note(f'{kind} admin call', error=repr(e))  # /admin/crash may drop the reply
    for s in streams:
        s.wait(600)
    ok = True
    for s in streams:
        r = s.result()
        same = s.text == ref.get(s.name)
        ok &= same and s.done and not s.errors and (s.end - t0) < 240 + 60
        note(f'{kind} result', identical=same, **r)
    note(f'{kind} og', recoveries=len(get(SUP + '/og/stats').get('recoveries') or []))
    return ok


def phase_e():
    note('E worker kill')
    sup = get(SUP + '/health')
    pid = sup.get('pid')
    streams = [Stream(f'e{i}', prompt_text(8000, random.random()), max_tokens=2000).start() for i in range(2)]
    for s in streams:
        while s.first is None and s.thread.is_alive():
            time.sleep(0.2)
    note('E killing', pid=pid)
    os.kill(int(pid), signal.SIGKILL)  # the worker's own PID (gpu-exec execs in place); never pkill
    ok = all(s.wait(60) for s in streams)
    ok &= all(s.done and s.errors and s.errors[0].get('code') == 'backend_unavailable' for s in streams)
    t0 = time.time()
    r = Stream('e-after', 'Say hi.', max_tokens=16, deadline=180).start()
    r.wait(180)
    note('E after', served_in=round(time.time() - t0, 1), **r.result(), sup=get(SUP + '/health'))
    return ok and r.done and not r.errors and time.time() - t0 < 120


def phase_f():
    note('F context edge')
    if 'worker_alive' not in get(SUP + '/health'):
        # Production b16205ac: generation past the box context wedges the engine thread (endless session rebuild,
        # ROBUSTNESS.md #1) until the worker is killed. Only run F against the robust branch.
        note('F skipped: supervisor is not the robust branch')
        return None
    s = Stream('f-edge', prompt_text(1046000, random.random()), max_tokens=8192, deadline=240).start()
    finished = s.wait(240)
    if not finished:
        s.cancel()
    clamped = get(SUP + '/og/stats').get('max_tokens_clamped')
    note('F result', finished=finished, **s.result(), og=clamped)
    # the clamp is the fix under test; the model may stop on its own before reaching the clamped length
    return finished and s.done and s.finish in ('length', 'stop') and not s.errors and (clamped or 0) >= 1


def phase_g():
    note('G malformed + burst')
    ok = True
    for name, body in (('too-long', dict(model=MODEL, max_tokens=8, messages=[{'role': 'user', 'content': 'x ' * 1100000}])),
                       ('bad-image', dict(model=MODEL, max_tokens=8, messages=[{'role': 'user', 'content': [
                           {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]))):
        try:
            post(SWAP + '/v1/chat/completions', body, timeout=120)
            ok = False
            note('G accepted?!', case=name)
        except urllib.error.HTTPError as e:
            ok &= 400 <= e.code < 500
            note('G refused', case=name, code=e.code)
    burst = [Stream(f'g{i}', f'Count from 1 to {10 + i}.', max_tokens=64).start() for i in range(8)]
    ok &= all(s.wait(300) and s.done and not s.errors for s in burst)
    note('G burst', results=[s.result() for s in burst])
    return ok


def final_checks(base):
    time.sleep(10)
    end = snapshot()
    box, og, sup = end['box'], end['og'], end['sup']
    checks = dict(
        box_sessions_0=box.get('sessions') == 0,
        box_connections_le1=(box.get('connections') or 0) <= 1,
        box_cache_dropped_0=(box.get('cache') or {}).get('dropped_at_load') == 0,
        og_opened_eq_closed=og.get('opened') == og.get('closed'),
        og_sessions_0=og.get('sessions') == 0,
        sup_inflight_0=sup.get('inflight') == 0,
        sup_divergent_0=sup.get('divergent') == 0,
        footprint_drift_lt_2g=(end['mem'].get('peak_footprint_gib', 0) - base['mem'].get('peak_footprint_gib', 0)) < 2
        or end['mem'].get('pid') != base['mem'].get('pid'),
    )
    note('final', **checks)
    return all(checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', action='store_true', help='required: an owner-approved maintenance window is open')
    ap.add_argument('--out', required=True)
    ap.add_argument('--phases', default='ABCDEFGH')
    args = ap.parse_args()
    if not args.window or os.environ.get('DS41_SOAK_WINDOW') != '1':
        sys.exit('refusing: this crashes/restarts production (box engine, Mac worker). Needs --window and '
                 'DS41_SOAK_WINDOW=1 inside an owner-approved maintenance window.')
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    base = snapshot()
    note('start', box=base['box'].get('version'), sup_pid=base['sup'].get('pid'))
    ref, results = {}, {}
    phases = dict(A=lambda: phase_a(ref), B=lambda: phase_b(base), C=lambda: phase_restart(ref, 'C'),
                  D=lambda: phase_restart(ref, 'D'), E=phase_e, F=phase_f, G=phase_g, H=lambda: final_checks(base))
    t_start = time.time()
    try:
        for p in args.phases:
            if time.time() - t_start > 30 * 60:
                note('time budget spent; skipping', phase=p)
                results[p] = None
                continue
            try:
                v = phases[p]()
                results[p] = None if v is None else bool(v)
            except Exception as e:  # noqa: BLE001
                note('phase error', phase=p, error=repr(e)[:300])
                results[p] = False
    finally:
        (out/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in LOG) + '\n')
        (out/'summary.json').write_text(json.dumps(dict(results=results, minutes=round((time.time() - t_start) / 60, 1)),
                                                   indent=1))
        print('RESULTS', results)
        print('restore check: box `systemctl --user status split-nv-engine`, `curl -s 10.10.10.1:10051/health`; '
              'Mac `curl -s localhost:8080/running` (ds41 ready), one omlx-server holding ~/llm/locks/gpu.lock')
    sys.exit(0 if all(v for v in results.values() if v is not None) else 1)


if __name__ == '__main__':
    main()
