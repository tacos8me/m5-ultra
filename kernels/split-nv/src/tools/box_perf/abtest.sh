#!/bin/bash
# In-window A/B of the runtime box-perf switches against the live engine. usage: abtest.sh OUTDIR
set -u
O=${1:?outdir}; mkdir -p $O
P=/home/ian/.venv/bin/python; D=/mnt/nvme-1/split-nv-ops/box-perf/prof; R=/home/ian/split-nv/ref
F=/dev/shm/split-nv/box-perf-flags.json
cfg() { echo "$2" > $F; sleep 1.3; echo "== $1 $2" | tee -a $O/ab.txt; date -u +%T.%N >> $O/ab.txt
  $P $D/drive.py dec $R/ids-8192.json --n 8193 --reps 30 --widths 1,5 --tight 30 > $O/$1.json 2>&1; grep -v '^{' $O/$1.json | tee -a $O/ab.txt; }
cfg A '{"step_log": true}'
cfg B '{"step_log": true, "inline": false, "spin_s": 0, "r1_spin_s": 0, "early_d2h": false}'
cfg C '{"step_log": true, "r1_spin_s": 0}'
cfg D '{"step_log": true, "spin_s": 0}'
cfg E '{"step_log": true, "early_d2h": false}'
cfg F '{"step_log": true, "inline": false}'
cfg A2 '{"step_log": true}'
echo '{}' > $F
