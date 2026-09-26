#!/bin/bash
# Post-restart gates for the s4 engine (live, bounded ~5 min). Exit code = number of failed gates.
cd /home/ian/split-nv
PY=/home/ian/.venv/bin/python
REF=/mnt/nvme-2/og-box-s4/ref-2256aec
fail=0
g() { echo "## $*"; "$@" || { fail=$((fail+1)); echo "FAILED: $*"; }; }
S=ref/ids-8192.json; L=ref/ids-131072.json
curl -s 127.0.0.1:10051/health; echo
# 1. unchanged numerics (non-stream + stream) vs the 2256aec engine, state bytes + 6 steps
for n in 301 8193; do g $PY tools/og_gate.py check --ids $S --n $n --ref $REF; g $PY tools/og_gate.py check --ids $S --n $n --ref $REF --stream; done
for n in 16385 20001 40000 131073; do g $PY tools/og_gate.py check --ids $L --n $n --ref $REF; g $PY tools/og_gate.py check --ids $L --n $n --ref $REF --stream; done
# 2. n=8194 has a 1-row final chunk in the old engine: it must now equal the prefix of a longer prompt
g $PY tools/og_gate.py prefix --ids $S --n 8194 --base 8200
g $PY tools/og_gate.py prefix --ids $L --n 16386 --base 16443
# 3. resume == fresh (state bytes + 6 steps), various resume points / suffixes
for pair in "20000 30001" "20001 30001" "8192 20001" "8191 16385" "8193 8250" "30000 30001" "40000 131073" "130000 131073"; do
  set -- $pair; g $PY tools/og_gate.py resume --ids $L --base $1 --n $2
done
g $PY tools/og_gate.py resume --ids $L --base 20000 --n 30001 --stream
g $PY tools/og_gate.py resume --ids $L --base 20000 --n 30001 --stream --lean
g $PY tools/og_http_check.py ref/ids-8192.json 8193
# 4. step numerics (pipe1 validate)
for n in 256 8190 8192; do g docker exec split-nv-encoder python3 /home/ian/split-nv/tools/step_client.py validate /home/ian/split-nv/ref/ids-8192.json --n $n; done
curl -s 127.0.0.1:10051/v1/cache; echo
echo "FAILED GATES: $fail"
exit $fail
