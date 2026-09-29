"""ds41-ttft2 partial-load guard: start only when production ds41 is idle (supervisor inflight 0, llama-swap ds41
ready, no open window) and no other heavy process (ds41-ffn harness etc.) runs; SIGTERM the child on production
traffic, another heavy process appearing, RSS > 35 GB, or the deadline. usage: guard.py <cmd...>
env: GUARD_MAX_S (default 600), GUARD_RSS_GB (35)."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

def read(url):
    with urllib.request.urlopen(url, timeout=1) as r:
        return json.load(r)

def window_open():
    last = None
    for line in (Path.home()/'llm/ds41/PROGRESS.md').read_text().splitlines():
        if 'WINDOW OPEN' in line or 'WINDOW CLOSED' in line:
            last = 'OPEN' if 'WINDOW OPEN' in line else 'CLOSED'
    return last == 'OPEN'

def heavy(exclude, root=None):
    out = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss=,command='], text=True)
    rows = [line.split(None, 3) for line in out.splitlines()]
    mine = set()
    if root is not None:  # the child's whole process tree is ours
        mine.add(root)
        grew = True
        while grew:
            grew = False
            for pid, ppid, _, _ in rows:
                if int(ppid) in mine and int(pid) not in mine:
                    mine.add(int(pid)); grew = True
    found = []
    for pid, ppid, rss, cmd in rows:
        if int(pid) in exclude or int(pid) in mine or int(rss) < 2_000_000:
            continue
        found.append((int(pid), int(rss) // 1024, cmd[:100]))
    return found, mine

def state(exclude, root=None):
    try:
        h = read('http://127.0.0.1:10001/health')
        r = read('http://127.0.0.1:8080/running')
    except Exception as e:  # noqa: BLE001
        return False, 'health ' + repr(e)[:80], None
    ok = (h.get('status') == 'ok' and h.get('inflight') == 0
          and any(m['model'] == 'ds41' and m['state'] == 'ready' for m in r['running']))
    if not ok:
        return False, dict(inflight=h.get('inflight'), status=h.get('status')), h
    if window_open():
        return False, 'window open', h
    others, mine = heavy(exclude | {h.get('pid') or -1}, root)
    if others:
        return False, dict(other_heavy=others), h
    return True, mine, h

ok, why, h = state({os.getpid()})
if not ok:
    print(json.dumps(dict(guard='busy', why=why)), flush=True)
    sys.exit(75)
p = subprocess.Popen(sys.argv[1:])
print(json.dumps(dict(guard='start', pid=p.pid, og=h.get('og'))), flush=True)
deadline = time.monotonic() + float(os.environ.get('GUARD_MAX_S', '600'))
limit = float(os.environ.get('GUARD_RSS_GB', '35')) * 1e9
reason = None
try:
    while p.poll() is None:
        time.sleep(0.1)
        ok, why, _ = state({os.getpid()}, p.pid)
        if not ok:
            reason = why
            break
        mine = why
        if time.monotonic() > deadline:
            reason = 'deadline'
            break
        rss = subprocess.run(['ps', '-o', 'rss=', '-p', ','.join(str(x) for x in mine)], capture_output=True, text=True).stdout.split()
        total = sum(int(x) for x in rss) * 1024
        if total > limit:
            reason = dict(rss=total)
            break
except BaseException as e:  # noqa: BLE001
    reason = repr(e)
finally:
    tree = set(locals().get('mine') or ()) - {p.pid, os.getpid()}
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)  # the child's trap stops its own servers by PID
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    # Anything the child started that outlived it (a server it did not stop): TERM, then KILL, by PID.
    left = [pid for pid in tree if alive(pid)]
    for pid in left:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 45
    while left and time.monotonic() < deadline:
        time.sleep(0.5)
        left = [pid for pid in left if alive(pid)]
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if tree:
        print(json.dumps(dict(guard='tree', stopped=sorted(tree), killed=left)), flush=True)
    print(json.dumps(dict(guard='end', pid=p.pid, reason=reason, returncode=p.returncode)), flush=True)
sys.exit(75 if reason else p.returncode)
