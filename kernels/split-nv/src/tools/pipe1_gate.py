"""LEGACY (pre-s4): stops the engine and restores the phase-2 run_encoder.sh; do NOT use while the split-nv-engine
user unit runs (it would restart the engine underneath). Use tools/og_gate.py + step_client validate instead.

Bounded phase-3 gate; restore phase-2 including a real prefill on every exit.

Announce the window in PROGRESS.md before running. Default budgets are 330s
for the experiment and 230s for recovery; explicit budgets total at most 870s.
Always check health and a real prefill after restoring the original launcher.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path('/home/ian/split-nv')


def stop():
    subprocess.run(['docker', 'stop', '-t', '5', 'split-nv-encoder'],
                   timeout=30, check=False, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True)
        if 'split-nv-encoder' not in names.splitlines():
            return
        time.sleep(.5)
    raise RuntimeError('Container did not exit')


def health():
    with urllib.request.urlopen('http://127.0.0.1:10051/health', timeout=2) as r:
        return json.load(r).get('encoder') == 'up'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--selftest', default='')
    ap.add_argument('--graphs', default='2,3,4,5')
    ap.add_argument('--experiment-seconds', type=int, default=330)
    ap.add_argument('--restore-seconds', type=int, default=230)
    args = ap.parse_args()
    if min(args.experiment_seconds, args.restore_seconds) <= 0 or args.experiment_seconds + args.restore_seconds > 870:
        ap.error('Experiment + restore budgets must be positive and total <=870s')
    if shutil.disk_usage('/').free < 100 * 1024**3:
        raise RuntimeError('Root has less than 100 GiB free')
    for name in ('run_engine.sh', 'run_encoder.sh'):
        assert '--ulimit core=0' in (ROOT / name).read_text(), name
    started = time.monotonic()
    deadline = started + args.experiment_seconds
    logpath = ROOT / 'logs' / (args.label + '.log')
    child = None
    def run(cmd):
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=max(1, deadline-time.monotonic()))
    try:
        stop()
        env = dict(os.environ, SPLIT_NV_GRAPHS=args.graphs,
                   SPLIT_NV_SELFTEST=args.selftest, CUDA_LAUNCH_BLOCKING='0')
        with logpath.open('x') as log:
            child = subprocess.Popen([str(ROOT / 'run_engine.sh')], env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            if args.selftest:
                child.wait(timeout=max(1, deadline-time.monotonic()))
            else:
                while time.monotonic() < deadline:
                    if child.poll() is not None:
                        raise RuntimeError('Engine exited')
                    if 'step api on' in logpath.read_text():
                        break
                    if 'Traceback (most recent call last)' in logpath.read_text():
                        raise RuntimeError('Engine startup failed; see log')
                    time.sleep(2)
                else:
                    raise TimeoutError('Engine startup deadline')
                for n in (256, 8190, 8192):
                    run(['docker', 'exec', 'split-nv-encoder', 'python3',
                         str(ROOT/'tools/step_client.py'), 'validate', str(ROOT/'ref/ids-8192.json'), '--n', str(n)])
                run([str(Path.home()/'.venv/bin/python'), 'tools/step_client.py',
                     'latency', 'ref/ids-8192.json', '--reps', '16'])
                run(['bash', 'tools/mac_latency.sh', 'ids-8192.json', '16'])
        print(json.dumps({'experiment_s':time.monotonic()-started, 'returncode':child.returncode}), flush=True)
    finally:
        stop()
        if child is not None:
            child.wait(timeout=10)
        with (ROOT/'logs'/(args.label+'-restore.log')).open('x') as log:
            restore = subprocess.Popen([str(ROOT/'run_encoder.sh')], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        end = min(started+args.experiment_seconds+args.restore_seconds+30,
                  time.monotonic()+args.restore_seconds)
        while time.monotonic() < end:
            if restore.poll() is not None:
                raise RuntimeError('Phase-2 restore exited')
            try:
                if health():
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Phase-2 health recovery deadline')
        req = urllib.request.Request('http://127.0.0.1:10051/v1/prefill',
            json.dumps({'tokens':[1,2,3,4,5,6,7,8,9]}).encode(), {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=max(1, end-time.monotonic())) as r:
            size = len(r.read())
        print(json.dumps({'restored':True, 'prefill_bytes':size,
                          'total_s':time.monotonic()-started,
                          'root_free_GiB':shutil.disk_usage('/').free/1024**3}), flush=True)


if __name__ == '__main__':
    main()
