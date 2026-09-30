#!/bin/bash
# CPU-only ds41 test suite for one tree: the og_serve scripts, then pytest on the ds41 file set with MLX pinned to the
# CPU and Metal kernels blocked (ds41_cpu_plugin.py). No GPU, no model, no gpu.lock; safe while ds41 serves.
# Usage: og_serve/cpu_suite.sh [TREE] [LABEL]   -> logs and failed.txt in /tmp/ds41-cpu-tests/LABEL
set -uo pipefail
TREE=${1:-$(cd "$(dirname "$0")/.." && pwd)}; LABEL=${2:-$(date +%Y%m%dT%H%M%S)}
PY=${PY:-$HOME/llm/.venv-ds41-omlx-tiles/bin/python}
OUT=/tmp/ds41-cpu-tests/$LABEL; mkdir -p "$OUT"
for p in 12169 12190 12191 12192 12193 12194 12195 12196 12197 12198 12199 12600 12601 12602 12603 12604 12605 12606 12607 12608 12609 12630 12631 12632 12633 12634 12635; do
  lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $p busy; abort"; exit 2; }
done
cd "$TREE"
export MLX_ENABLE_TF32=0
for t in test_import_fallback.py test_wire_failover.py test_robust.py test_supervisor_failover.py test_proxy_allowlist.py test_stepd.py; do
  [[ -f og_serve/$t ]] || { echo "SKIP og_serve/$t (absent)"; continue; }
  s=$(date +%s)
  $PY og_serve/$t > "$OUT/og_$t.log" 2>&1; rc=$?
  echo "og_serve/$t rc=$rc $(( $(date +%s) - s ))s :: $(grep -E 'ALL PASS|SOME FAILED|passed|failed|OK|FAIL' "$OUT/og_$t.log" | tail -1)"
done
PLUGIN=$(mktemp -d); cp og_serve/ds41_cpu_plugin.py "$PLUGIN/"
FILES=$(ls tests/test_deepseek_v41*.py tests/test_deepseek_v4_dspark.py tests/test_dspark_thinking_budget.py tests/test_mlx_lm_mtp_patch.py 2>/dev/null | grep -v -E 'test_deepseek_v41_(activation|fast_rope|ffn_fuse|grouped_expert|ssd|woa_compact)\.py')
s=$(date +%s)
PYTHONPATH=$PLUGIN $PY -m pytest -p ds41_cpu_plugin -p no:cacheprovider -q -o addopts='-m "not slow and not integration"' -rfE $FILES > "$OUT/pytest.log" 2>&1
echo "pytest rc=$? $(( $(date +%s) - s ))s :: $(tail -1 "$OUT/pytest.log")"
rm -rf "$PLUGIN"
grep -E '^(FAILED|ERROR)' "$OUT/pytest.log" | sed 's/ - .*//' | sort > "$OUT/failed.txt"
echo "failed/error ids: $(wc -l < "$OUT/failed.txt") -> $OUT/failed.txt"
