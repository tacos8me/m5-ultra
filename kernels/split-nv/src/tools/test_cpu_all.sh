#!/bin/bash
# Every CPU test of this tree, no GPU (CUDA_VISIBLE_DEVICES emptied; containers run without --gpus, --network none,
# private IPC). Safe while production serves: it touches no /dev/shm/split-nv, unit, engine or GPU.
#   usage: tools/test_cpu_all.sh            (IMAGE=... to test another image; SKIP_IMAGE=1 host-only)
# Exit code = number of failed tests. The macpack round trips need a Mac state file: MACPACK_STATE=<file>.
set -u
cd "$(dirname "$0")/.."
WT=$(pwd)
PY=${PY:-/home/ian/.venv/bin/python}
IMAGE=${IMAGE:-sglang-dsv41-split:6152b54}
SG=${SGLANG_SRC:-/home/ian/split-nv/sglang/sglang}/srt/layers/attention/dsv4/dsv41_indexer_select.py
LOG=$(mktemp -d "${TMPDIR:-/tmp}/cpu-tests.XXXXXX")
export CUDA_VISIBLE_DEVICES=
fails=0
run() {  # name, command...
  local name=$1; shift
  local f="$LOG/${name//\//_}.log"
  if "$@" >"$f" 2>&1; then echo "PASS $name ($(tail -1 "$f" | cut -c1-100))"
  else echo "FAIL $name (log $f)"; tail -5 "$f" | sed 's/^/    /'; fails=$((fails + 1)); fi
}
for t in test_dspark_markov test_dspark_stepd test_dspark_stream test_w11_fairness test_front_robust test_fast_restart test_rank_failstop test_imagekeys \
         test_topk_det_cpu; do
  run "$t" "$PY" "tools/$t.py"
done
for t in test_gate test_preempt_proto test_stallwatch; do
  run "fair/$t" "$PY" "tools/fair/$t.py"
done
run fair/test_idx_lowmem "$PY" tools/fair/test_idx_lowmem.py "$SG"
run fair/test_idx_rowsplit "$PY" tools/fair/test_idx_rowsplit.py "$SG"
run test_cache_mirror bash tools/test_cache_mirror.sh
if [ -n "${MACPACK_STATE:-}" ]; then
  run test_macpack_roundtrip "$PY" tools/test_macpack_roundtrip.py "$MACPACK_STATE"
  run test_macpack_values "$PY" tools/test_macpack_values.py "$MACPACK_STATE"
fi
if [ "${SKIP_IMAGE:-0}" != 1 ]; then
  D=(docker run --rm --network none --ipc=private --ulimit core=0 --cpus 4 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1)
  # the fused Markov kernels under the image's Triton (the engine's), in the interpreter
  run image/test_dspark_markov "${D[@]}" -v "$WT":/work:ro -w /work --entrypoint python3 "$IMAGE" tools/test_dspark_markov.py
  run image/test_engram_keep "${D[@]}" -e PYTHONPATH=/home/ian/split-nv/hooks -v "$WT":/home/ian/split-nv:ro \
    --entrypoint python3 "$IMAGE" /home/ian/split-nv/tools/test_engram_keep.py
fi
echo "FAILED: $fails (logs in $LOG)"
exit $fails
