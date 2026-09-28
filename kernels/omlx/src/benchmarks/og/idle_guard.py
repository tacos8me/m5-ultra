"""Run one bounded Mac partial-load experiment; terminate its PID on production traffic."""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def read(url):
    with urllib.request.urlopen(url, timeout=0.5) as r:
        return json.load(r)


def idle():
    h = read('http://127.0.0.1:10001/health')
    r = read('http://127.0.0.1:8080/running')
    return (h.get('status') == 'ok' and h.get('inflight') == 0
            and any(m['model'] == 'ds41' and m['state'] == 'ready' for m in r['running'])), h


ok, initial = idle()
if not ok:
    print('GUARD busy; skipped', flush=True)
    sys.exit(75)
p = subprocess.Popen(sys.argv[1:])
print(json.dumps(dict(guard='start', pid=p.pid, health=initial)), flush=True)
deadline = time.monotonic() + float(os.environ.get('DECODE_MAX_SECONDS', '90'))
reason = None
try:
    while p.poll() is None:
        time.sleep(.1)
        ok, h = idle()
        if not ok:
            reason = {'production_not_idle': h}
            break
        if time.monotonic() > deadline:
            reason = 'deadline'
            break
        rss = subprocess.check_output(['ps','-o','rss=','-p',str(p.pid)],text=True).strip()
        if rss and int(rss)*1024 > 35_000_000_000:
            reason = {'rss_limit_bytes':int(rss)*1024}
            break
except BaseException as e:
    reason = repr(e)
finally:
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    print(json.dumps(dict(guard='end', pid=p.pid, reason=reason, returncode=p.returncode)), flush=True)
sys.exit(75 if reason else p.returncode)
