#!/bin/bash
# One box maintenance window: production engine -> nsys profiling variant (same image/args) -> traffic -> production.
set -u
P=/mnt/nvme-1/split-nv-ops/box-perf/prof
LOG=$P/window.log
PY=/home/ian/.venv/bin/python
R=/home/ian/split-nv/ref
ts() { date -u +%T; }
say() { echo "$(ts) $*" | tee -a $LOG; }
healthy() { curl -s -m 2 127.0.0.1:10050/health | grep -q '"ok": true'; }
restore() {
  say "RESTORE: stopping profiling container, starting production"
  docker stop -t 20 split-nv-encoder >/dev/null 2>&1
  for i in $(seq 30); do docker ps -a --format '{{.Names}}' | grep -qx split-nv-encoder || break; sleep 1; done
  docker rm -f split-nv-encoder >/dev/null 2>&1
  systemctl --user start split-nv-engine
  for i in $(seq 120); do healthy && break; sleep 3; done
  curl -s 127.0.0.1:10050/health | tee -a $LOG; echo | tee -a $LOG
  $PY $P/drive.py dec $R/ids-8192.json --n 8193 --reps 6 --widths 5 --tight 0 2>&1 | grep -v '^{' | tee -a $LOG
  say "production restored"
}
nsys_start() { docker exec split-nv-encoder nsys start --session=prof -o /traces/$1 --force-overwrite=true --gpu-metrics-devices=all --gpu-metrics-frequency=$2 >>$LOG 2>&1; }
nsys_stop() { docker exec split-nv-encoder nsys stop --session=prof 2>&1 | tail -2 | tee -a $LOG; }

say "window start"
systemctl --user stop split-nv-engine
for i in $(seq 60); do docker ps -a --format '{{.Names}}' | grep -qx split-nv-encoder || break; sleep 1; done
say "production stopped"
SPLIT_NV_DEV=0 SPLIT_NV_ROOT=/home/ian/split-nv-deploy nohup setsid $P/run_engine_prof.sh > $P/engine-prof.log 2>&1 &
ok=0
for i in $(seq 110); do
  sleep 3
  healthy && { ok=1; break; }
  grep -q Traceback $P/engine-prof.log && break
  docker ps --format '{{.Names}}' | grep -qx split-nv-encoder || { [ $i -gt 5 ] && break; }
done
if [ $ok != 1 ]; then say "profiling engine failed to start"; tail -30 $P/engine-prof.log >> $LOG; restore; exit 1; fi
say "profiling engine up"; curl -s 127.0.0.1:10050/health >> $LOG; echo >> $LOG

nsys_start boxA 20000; say "A: collection started"
$PY $P/drive.py dec $R/ids-8192.json --n 8193 --reps 30 --tight 30 --eager $P > $P/A-dec8k.json 2>&1; grep -v '^{' $P/A-dec8k.json | tee -a $LOG
$PY $P/drive.py open $R/ids-8192.json --n 8193 > $P/A-open8k.json 2>&1; tee -a $LOG < $P/A-open8k.json
$PY $P/drive.py dec $R/ids-131072.json --n 131073 --reps 30 --widths 1,2,5 --tight 30 > $P/A-dec128k.json 2>&1; grep -v '^{' $P/A-dec128k.json | tee -a $LOG
nsys_stop; say "A: report written"

nsys_start boxB 10000; say "B: collection started"
$PY $P/drive.py open $R/ids-524310.json --n 524289 > $P/B-open512k.json 2>&1; tee -a $LOG < $P/B-open512k.json
$PY $P/drive.py interf $R/ids-8192.json $R/ids-131072.json --n 131073 > $P/B-interf.json 2>&1; cut -c1-300 $P/B-interf.json | tee -a $LOG
nsys_stop; say "B: report written"

restore
say "window end"
