#!/bin/bash
# Post-deploy gates for og-moe (numerics og-s4.4; the old byte references are og-s4.3 by design).
# usage: tools/og_moe_gates.sh <label>   (box should be quiet). Exit code = failed gates.
cd /home/ian/split-nv-moe
PY=/home/ian/.venv/bin/python
R=/home/ian/split-nv/ref
S=$R/ids-8192.json; L=$R/ids-131072.json
OUT=/mnt/nvme-2/og-moe/deploy/${1:?label}
mkdir -p $OUT
fail=0
g() { echo "## $*"; "$@" 2>&1 | grep -E '"gate"|numerical_gate|rel_rms|resumed_tokens|warning|"ref"|FAIL|Error|context' | cut -c1-400; [ ${PIPESTATUS[0]} -eq 0 ] || { fail=$((fail+1)); echo "FAILED: $*"; }; }
curl -s 127.0.0.1:10051/health; echo
curl -s 127.0.0.1:10051/health | grep -q '"numerics": "og-s4.4"' || { echo "FAILED: numerics is not og-s4.4"; fail=$((fail+1)); }
# 1. step == prefill, strict (every rel_rms exactly 0), incl. rollback and the 8192 page boundary
for n in 256 8190 8192; do
  $PY tools/step_client.py validate $S --n $n > $OUT/validate-$n.txt 2>&1 || { fail=$((fail+1)); echo "FAILED: validate $n"; }
  bad=$(grep -oE '"max_abs": [0-9.e+-]+' $OUT/validate-$n.txt | grep -vc '"max_abs": 0.0$')
  nz=$(grep -oE '"max_abs": [0-9.e+-]+' $OUT/validate-$n.txt | wc -l)
  echo "## validate n=$n: $nz max_abs values, $bad non-zero"; [ "$bad" = 0 ] && [ "$nz" -gt 0 ] || { fail=$((fail+1)); echo "FAILED: strict validate $n"; }
done
# 2. prefix invariance incl. the 1-row chunk cases
g $PY tools/og_gate.py prefix --ids $S --n 8194 --base 8200
g $PY tools/og_gate.py prefix --ids $L --n 16386 --base 16443
# 3. resume == fresh (state bytes + 6 steps)
for pair in "20000 30001" "20001 30001" "8192 20001" "8191 16385" "8193 8250" "30000 30001" "40000 131073" "130000 131073"; do
  set -- $pair; g $PY tools/og_gate.py resume --nonce --ids $L --base $1 --n $2
done
g $PY tools/og_gate.py resume --nonce --ids $L --base 20000 --n 30001 --stream
g $PY tools/og_gate.py resume --nonce --ids $L --base 20000 --n 30001 --stream --lean
# 4. HTTP /v1/prefill == OPEN STAT; vision gates
g $PY tools/og_http_check.py $S 8193
g $PY tools/og_image_gate.py
# 5. new og-s4.4 byte reference + determinism against it (a second fresh run must be identical)
for n in 301 8193 131073; do
  ids=$S; [ $n -gt 8192 ] && ids=$L
  g $PY tools/og_gate.py ref --ids $ids --n $n --out $OUT/ref
  g $PY tools/og_gate.py check --ids $ids --n $n --ref $OUT/ref
  g $PY tools/og_gate.py check --ids $ids --n $n --ref $OUT/ref --stream
done
# 6. box step latency (8K and 128K)
$PY tools/step_client.py latency $S --reps 24 2>&1 | grep "box median" | tee $OUT/latency-8k.txt
$PY tools/step_client.py latency $L --reps 24 2>&1 | grep "box median" | tee $OUT/latency-128k.txt
echo "FAILED GATES: $fail"
exit $fail
