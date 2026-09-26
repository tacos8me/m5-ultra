"""One guarded Mac maintenance window for front-end A/B runs of this tree (<= 15 min).

Checks that no other window is open (PROGRESS.md), that ds41 is the only llama-swap model and is
idle, and that the box has no sessions; announces the window; unloads ds41 through llama-swap's
API and waits for the production PID and gpu.lock to go; starts this tree's supervisor on another
port (its og worker under gpu-exec, with the 245 GiB / DS41_OG_LEASE_S watchdog); runs fe_bench once
per phase, switching the front-end paths at runtime (trace mode /og/fe); then TERMs the supervisor
by PID, waits for the lock to be free, restores production with a small chat request through
llama-swap, checks the answer and closes the window.

  python og_serve/fe_window.py --label fe-w1 --phases 'A:encode_cache=0,kickoff=0,wake=0;B:' \
      --bench '--ttft 8192:3 --resume 8192:3,131072:3,524288:2 --identity'
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

HOME = Path.home()
HERE = Path(__file__).resolve().parent
PY = str(HOME/'llm/.venv-ds41-omlx-tiles/bin/python')
PROGRESS = HOME/'llm/ds41/PROGRESS.md'
LOCK = HOME/'llm/locks/gpu.lock'
SWAP = 'http://127.0.0.1:8080'
SUP_PORT, WORKER_PORT = 12648, 12647


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post(url, body, timeout=60):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def announce(text):
    with PROGRESS.open('a') as f:
        f.write(f'- {utc()} {text}\n')
    print('PROGRESS:', text, flush=True)


def window_open():
    last = None
    for line in PROGRESS.read_text().splitlines():
        m = re.match(r'- \d{4}-\d\d-\d\dT[\d:]+Z .*?\b(WINDOW|LEASE) (OPEN|CLOSED)\b', line)
        if m:
            last = (m.group(2), line)
    return last is not None and last[0] == 'OPEN', last and last[1]


def listeners(port):
    out = subprocess.run(['lsof', '-t', f'-iTCP:{port}', '-sTCP:LISTEN'], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def lock_holders():
    out = subprocess.run(['lsof', '-t', str(LOCK)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_gone(pids, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.5)
    return False


def og_home():
    """A private OG home: copies of the served profile settings, the same model, own logs."""
    home = HOME/'llm/ds41/fe/og-home'
    (home/'profile').mkdir(parents=True, exist_ok=True)
    (home/'models').mkdir(exist_ok=True)
    (home/'logs').mkdir(exist_ok=True)
    for name in ('settings.json', 'model_settings.json'):
        shutil.copy2(HOME/'llm/ds41/og/profile'/name, home/'profile'/name)
    link = home/'models/ds41-og'
    if not link.exists():
        link.symlink_to(os.path.realpath(HOME/'llm/ds41/og/models/ds41-og'))
    return home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--phases', required=True,
                    help="'NAME:key=v,key=v[|bench args];NAME2:...' (/og/fe switches, optional bench override)")
    ap.add_argument('--bench', default='--ttft 8192:3 --resume 8192:3,131072:3,524288:2 --identity')
    ap.add_argument('--env', action='append', default=[], help='KEY=VALUE for the supervisor/worker')
    ap.add_argument('--minutes', type=float, default=14)
    ap.add_argument('--dry-run', action='store_true', help='preflight checks only')
    ap.add_argument('--wait-idle', type=float, default=180, help='start only after ds41 was idle this long (s)')
    ap.add_argument('--max-wait', type=float, default=60, help='give up after this many minutes of waiting')
    args = ap.parse_args()
    t_start = time.monotonic()
    logs = HOME/'llm/ds41/fe'/args.label
    logs.mkdir(parents=True, exist_ok=True)

    served_log = HOME/'llm/ds41/og/logs/og-child.log'
    waited = quiet_since = time.monotonic()
    while args.wait_idle:
        # Idle = no admission logged and nothing in flight, continuously for --wait-idle seconds.
        inflight = 0
        try:
            for item in get(SWAP + '/running').get('running', []):
                inflight += get(f"http://127.0.0.1:{item['proxy'].rsplit(':', 1)[1]}/health").get('inflight') or 0
        except Exception:  # noqa: BLE001 -- loading or unreachable: not idle
            inflight = 1
        if inflight or (served_log.exists() and time.time() - served_log.stat().st_mtime < 5):
            quiet_since = time.monotonic()
        if time.monotonic() - quiet_since >= args.wait_idle and not window_open()[0]:
            break
        if time.monotonic() - waited > args.max_wait * 60:
            raise SystemExit(f'ds41 not idle for {args.wait_idle:.0f} s within {args.max_wait:.0f} min')
        time.sleep(5)
    t_start = time.monotonic()
    busy, line = window_open()
    if busy:
        raise SystemExit(f'another window is open: {line}')
    running = get(SWAP + '/running').get('running', [])
    if [r['model'] for r in running] not in ([], ['ds41']):
        raise SystemExit(f'unexpected llama-swap models: {[r["model"] for r in running]}')
    prod_pids = []
    if running:
        port = running[0]['proxy'].rsplit(':', 1)[1]
        health = get(f'http://127.0.0.1:{port}/health')
        if health.get('inflight'):
            raise SystemExit(f'ds41 busy: {health}')
        prod_pids = listeners(port) + ([health['pid']] if health.get('pid') else [])
    box = get('http://10.10.10.1:10051/health')
    if box.get('sessions'):
        raise SystemExit(f'box has sessions: {box.get("sessions")}')

    if args.dry_run:
        print('preflight ok: production pids', prod_pids, 'lock holders', lock_holders(), 'last window line:', line)
        return
    announce(f'MAC WINDOW OPEN (Opus ds41-fe): front-end A/B `{args.label}` of ~/src/wt/ds41-fe on :{SUP_PORT}; '
             f'ds41 unloaded via llama-swap, restored by a chat request at the end; <= {args.minutes:.0f} min.')
    sup = None
    summary = {}
    try:
        with urllib.request.urlopen(SWAP + '/unload', timeout=150) as r:
            r.read()
        if not wait_gone(prod_pids, 150):
            raise RuntimeError(f'production PIDs {prod_pids} still alive after unload')
        deadline = time.monotonic() + 60
        while lock_holders() and time.monotonic() < deadline:
            time.sleep(0.5)
        if lock_holders():
            raise RuntimeError(f'gpu.lock still held by {lock_holders()}')

        home = og_home()
        env = os.environ.copy()
        env.update(DS41_TREE=str(HERE.parent), DS41_OG_CACHE_GIB='16', DS41_OG_WORKER_PORT=str(WORKER_PORT),
                   DS41_OG_LOGS=str(logs), DS41_OG_HOME=str(home), DS41_OG_LEASE_S=str(int(args.minutes * 60)),
                   DS41_FE_TRACE='1', DS41_FE_DIGEST='1')
        env.update(dict(x.split('=', 1) for x in args.env))
        out = (logs/'supervisor.log').open('a')
        sup = subprocess.Popen([PY, '-u', str(HERE/'ds41_og.py'), '--host', '127.0.0.1', '--port', str(SUP_PORT)],
                               env=env, stdout=out, stderr=subprocess.STDOUT)
        print('supervisor pid', sup.pid, flush=True)
        deadline = time.monotonic() + 300
        while True:
            if sup.poll() is not None:
                raise RuntimeError(f'supervisor exited {sup.returncode}')
            try:
                if get(f'http://127.0.0.1:{SUP_PORT}/health', 2).get('status') == 'ok':
                    break
            except Exception:  # noqa: BLE001 -- still starting
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('supervisor not healthy')
            time.sleep(1)
        print('healthy after', round(time.monotonic() - t_start), 's', flush=True)
        for spec in [p for p in args.phases.split(';') if p]:
            spec, _, bench = spec.partition('|')  # optional per-phase bench arguments
            name, _, flags = spec.partition(':')
            switches = {k: (int(v) if k == 'replay_eval_every' else bool(int(v)))
                        for k, _, v in (f.partition('=') for f in flags.split(',') if f)}
            defaults = dict(encode_cache=True, kickoff=True, wake=True, replay_eval_every=1, profile=False)
            state = post(f'http://127.0.0.1:{WORKER_PORT}/og/fe', dict(defaults, **switches))
            left = args.minutes * 60 - (time.monotonic() - t_start) - 120
            if left < 60:
                print('skipping phase', name, '(window time)', flush=True)
                continue
            print('phase', name, state, flush=True)
            cmd = [PY, str(HERE/'fe_bench.py'), '--base', f'http://127.0.0.1:{SUP_PORT}', '--model', 'ds41-og',
                   '--label', f'{args.label}-{name}', '--budget', str(int(left)), *shlex.split(bench or args.bench)]
            subprocess.run(cmd, timeout=left + 60)
            try:
                summary[name] = json.loads((HOME/'llm/ds41/fe'/f'{args.label}-{name}.summary.json').read_text())
            except (OSError, ValueError) as exc:
                summary[name] = repr(exc)
        try:
            summary['og_stats'] = get(f'http://127.0.0.1:{WORKER_PORT}/og/stats')
        except Exception:  # noqa: BLE001
            pass
    finally:
        if sup is not None and sup.poll() is None:
            child = None
            try:
                child = get(f'http://127.0.0.1:{SUP_PORT}/health', 2).get('pid')
            except Exception:  # noqa: BLE001
                pass
            sup.terminate()
            try:
                sup.wait(150)
            except subprocess.TimeoutExpired:
                sup.kill()
                sup.wait()
            if child and not wait_gone([child], 60):
                os.kill(child, 15)
                wait_gone([child], 60)
        deadline = time.monotonic() + 60
        while lock_holders() and time.monotonic() < deadline:
            time.sleep(0.5)
        restored = False
        try:
            reply = post(SWAP + '/v1/chat/completions', dict(
                model='ds41', max_tokens=8, temperature=0,
                messages=[{'role': 'user', 'content': 'Reply with exactly the word: ready'}]), timeout=600)
            text = reply['choices'][0]['message']['content']
            restored = 'ready' in text.lower()
            print('restore reply:', repr(text), flush=True)
        except Exception as exc:  # noqa: BLE001
            print('restore failed:', repr(exc), flush=True)
        (logs/'summary.json').write_text(json.dumps(summary, indent=1))
        announce(f'MAC WINDOW CLOSED (Opus ds41-fe): `{args.label}` done in {(time.monotonic() - t_start) / 60:.1f} min; '
                 f'ds41 {"restored and answering" if restored else "RESTORE NOT VERIFIED - check"}.')


if __name__ == '__main__':
    main()
