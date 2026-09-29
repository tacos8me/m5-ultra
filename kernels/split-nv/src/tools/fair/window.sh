#!/bin/bash
# Fairness / memory maintenance window, one stage per engine configuration. The coordinator deploys first:
#   systemctl --user stop llama-swap                       # box llama-swap grabs GPUs otherwise
#   tools/s4_deploy.sh <fair commit> fair-<stage>           # from /home/ian/split-nv-deploy
#   drop-in ~/.config/systemd/user/split-nv-engine.service.d/fair.conf with the stage's Environment= line, then
#   systemctl --user daemon-reload && systemctl --user restart split-nv-engine (wait for /health ok)
# usage: tools/fair/window.sh <stage> <label>
#   gates  byte gates for ANY stage (run first after each restart): og-s4.4 references 301/8193/131073 plain+stream,
#          1M, then tools/topk_det_window.sh (og_moe_gates + repetitive overflow prompts + strict step==prefill)
#   mem    memory ladder 8K/128K/512K/1M with allocator stats (needs SPLIT_NV_MEMLOG=1 or any fair commit)
#   fair   decode-during-prefill and short-behind-long, byte-checked, with runtime A/B of the preemption knobs
# Exit code = failed byte gates. Logs: /mnt/nvme-1/split-nv-ops/fair/window-<label>/
set -u
cd "$(dirname "$0")/../.."
PY=/home/ian/.venv/bin/python
R=/home/ian/split-nv/ref
REF=/mnt/nvme-1/split-nv-ops/og-moe-deploy/box-perf-2/ref
OUT=/mnt/nvme-1/split-nv-ops/fair/window-${2:?label}
FLAGS=/dev/shm/split-nv/box-perf-flags.json
mkdir -p "$OUT"
fail=0
g() { local f="$OUT/$(echo "$*" | md5sum | cut -c1-8).txt"; echo "## $*"; "$@" > "$f" 2>&1; rc=$?
      grep -hE '"gate"|"pass"|max_abs|FAIL|Error|during_|open_s' "$f" | cut -c1-600 | tail -6
      [ $rc -eq 0 ] || { fail=$((fail+1)); echo "FAILED: $*"; }; }
H=$(curl -s 127.0.0.1:10051/health); echo "$H" | cut -c1-900
echo "$H" | grep -q '"numerics": "og-s4.4"' || { echo "FAILED: numerics"; exit 1; }
[ "$(echo "$H" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sessions"])')" = 0 ] || { echo "sessions open: not a quiet box"; exit 1; }
case "${1:?stage}" in
gates)
  for n in 8193 301 131073; do
    ids=$R/ids-8192.json; [ $n -gt 8192 ] && ids=$R/ids-131072.json
    g $PY tools/og_gate.py check --ids $ids --n $n --ref $REF
    g $PY tools/og_gate.py check --ids $ids --n $n --ref $REF --stream
  done
  g $PY tools/og_gate.py check --ids $R/ids-1048576.json --n 1040000 --ref /mnt/nvme-1/split-nv-ops/ref-1m
  g tools/topk_det_window.sh "fair-$2"
  ;;
mem)
  g $PY tools/fair/mem_window.py --label "$2"
  journalctl --user -u split-nv-engine --since "-20 min" --no-pager | grep -E "mem after prefill|allocation failed with OOM" | tail -30 | tee "$OUT/memlog.txt"
  ;;
fair)
  L=$R/ids-131072.json; S=$R/ids-131072.json  # the 8193 ref was recorded from ids-131072 (STALE-STEPS.md)
  ab() { echo "$1" > $FLAGS; sleep 1.5; echo "### flags $1"
         g $PY tools/fair/fair_gate.py decode --ids $L --n 131073 --ref $REF --out "$OUT/fair.jsonl"
         g $PY tools/fair/fair_gate.py decode --decoders 2 --ids $L --n 131073 --ref $REF --out "$OUT/fair.jsonl"
         g $PY tools/fair/fair_gate.py mixed --ids $L --n 131073 --short-ids $S --short-n 8193 --ref $REF --out "$OUT/fair.jsonl"; }
  ab '{"preempt": false, "bypass_tokens": 0}'               # production behaviour (2048-row pieces, FIFO)
  ab '{"preempt": false, "bypass_tokens": 0, "share_chunk": 1024}'
  ab '{"preempt": true, "preempt_lag": 2, "bypass_tokens": 16384}'
  ab '{"preempt": true, "preempt_lag": 1, "bypass_tokens": 16384}'
  ab '{"preempt": true, "preempt_lag": 0, "bypass_tokens": 16384}'
  ab '{"preempt": true, "preempt_lag": 2, "preempt_share": 0.25, "bypass_tokens": 16384}'
  # streamed + cache=1 variants under preemption (the paths production uses)
  echo '{"preempt": true, "preempt_lag": 2, "bypass_tokens": 16384}' > $FLAGS; sleep 1.5
  g $PY tools/fair/fair_gate.py decode --stream --ids $L --n 131073 --ref $REF --out "$OUT/fair.jsonl"
  g $PY tools/fair/fair_gate.py mixed --stream --ids $L --n 131073 --short-ids $S --short-n 8193 --ref $REF --out "$OUT/fair.jsonl"
  g $PY tools/fair/fair_gate.py decode --ids $R/ids-1048576.json --n 1040000 --ref /mnt/nvme-1/split-nv-ops/ref-1m --out "$OUT/fair.jsonl"
  rm -f $FLAGS
  ;;
esac
echo "FAILED GATES: $fail"
exit $fail
