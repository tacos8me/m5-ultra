#!/bin/bash
# topk-det maintenance-window gates. Run AFTER `tools/s4_deploy.sh <topk-det commit> <label>` (box llama-swap stopped,
# no user sessions). Exit code = failed gates. Logs: /mnt/nvme-1/split-nv-ops/topk/window-<label>/.
#   usage: tools/topk_det_window.sh <label>
set -u
cd "$(dirname "$0")/.."
PY=/home/ian/.venv/bin/python
R=/home/ian/split-nv/ref
REF=/mnt/nvme-1/split-nv-ops/og-moe-deploy/box-perf-2/ref
REP=/mnt/nvme-1/split-nv-ops/topk/ids   # repetitive prompts that overflow (built by tools/make_rep_ids.py)
OUT=/mnt/nvme-1/split-nv-ops/topk/window-${1:?label}
mkdir -p "$OUT"
fail=0
g() { echo "## $*"; "$@" > "$OUT/$(echo "$*" | md5sum | cut -c1-8).txt" 2>&1; rc=$?
      grep -hE '"gate"|"pass"|max_abs|FAIL|Error' "$OUT/$(echo "$*" | md5sum | cut -c1-8).txt" | cut -c1-300 | tail -8
      [ $rc -eq 0 ] || { fail=$((fail+1)); echo "FAILED: $*"; }; }

H=$(curl -s 127.0.0.1:10051/health); echo "$H" | cut -c1-600
echo "$H" | grep -q '"numerics": "og-s4.4"' || { echo "FAILED: numerics"; fail=$((fail+1)); }
echo "$H" | grep -q '"topk_det": "[0-9a-f]\{16\}"' || { echo "FAILED: topk_det not installed"; fail=$((fail+1)); }

# 1. og-s4.4 byte references (non-overflow rows must be unchanged)
for n in 301 8193 131073; do
  ids=$R/ids-8192.json; [ $n -gt 8192 ] && ids=$R/ids-131072.json
  g $PY tools/og_gate.py check --ids $ids --n $n --ref $REF
  g $PY tools/og_gate.py check --ids $ids --n $n --ref $REF --stream
done
g $PY tools/og_gate.py check --ids $R/ids-1048576.json --n 1040000 --ref /mnt/nvme-1/split-nv-ops/ref-1m

# 2. strict step == prefill, prefix invariance, resume == fresh, HTTP, image, new-ref determinism, latency
g tools/og_moe_gates.sh "topk-det-${1}"

# 3. repetitive prompts that overflow: determinism (two fresh runs), resume == fresh, step == prefill
for f in "$REP"/ids-*.json; do
  [ -e "$f" ] || continue
  n=$(python3 -c "import json,sys; t=json.load(open('$f')); t=t['tokens'] if isinstance(t,dict) else t; print(len(t)-600)")
  b=$(basename "$f" .json)
  g $PY tools/og_gate.py ref --ids "$f" --n "$n" --out "$OUT/ref-$b"
  g $PY tools/og_gate.py check --ids "$f" --n "$n" --ref "$OUT/ref-$b"
  g $PY tools/og_gate.py resume --nonce --ids "$f" --base $((n / 2)) --n "$n"
  $PY tools/step_client.py validate "$f" --n "$n" > "$OUT/validate-$b.txt" 2>&1 || { fail=$((fail+1)); echo "FAILED: validate $b"; }
  bad=$(grep -oE '"max_abs": [0-9.e+-]+' "$OUT/validate-$b.txt" | grep -vc '"max_abs": 0.0$')
  nz=$(grep -oE '"max_abs": [0-9.e+-]+' "$OUT/validate-$b.txt" | wc -l)
  echo "## strict validate $b n=$n: $nz max_abs values, $bad non-zero"
  [ "$bad" = 0 ] && [ "$nz" -gt 0 ] || { fail=$((fail+1)); echo "FAILED: strict validate $b"; }
done
echo "FAILED GATES: $fail"
exit $fail
