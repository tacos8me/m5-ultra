#!/bin/bash
# Run tools/test_topk_det_gpu.py on ONE GPU in a throwaway container of the production image (never the engine's).
# Guards: the engine must be idle (health ok, sessions 0, gpu_job null) and the GPU must have >= MIN_FREE_MB free.
# Default = small shapes (<= 64 MB per tensor, allocator capped at 1% of the GPU): safe next to production.
# FULL=1 TIMING=1 (production-size shapes + timings) only in a maintenance window with the engine stopped.
#   usage: tools/topk_det_gpu.sh [GPU=1] [FULL=1] [TIMING=1] [RUNS=10]
set -u
TREE=$(cd "$(dirname "$0")/.." && pwd)
GPU=${GPU:-1}
IMAGE=${SPLIT_NV_IMAGE:-sglang-dsv41-split:6152b54}
JIT=${JIT:-/mnt/nvme-1/split-nv-ops/topk/jit-cache}
MIN_FREE_MB=${MIN_FREE_MB:-3072}
H=$(curl -s -m 3 127.0.0.1:10051/health || true)
if [ -n "$H" ]; then
  echo "$H" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("sessions")==0 and d.get("gpu_job") is None else 1)' \
    || { echo "engine busy: $H" | cut -c1-300; exit 2; }
  [ "${FULL:-0}" = 1 ] && { echo "FULL=1 needs the engine stopped (maintenance window)"; exit 2; }
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
[ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU $GPU has only $FREE MiB free"; exit 3; }
echo "GPU $GPU free ${FREE} MiB; engine idle or down"
exec timeout 600 docker run --rm --name topk-det-test --gpus "\"device=$GPU\"" --entrypoint python3 \
  -v "$TREE":/w:ro -v "$JIT":/root/.cache/sglang -e PYTHONPATH=/w/hooks \
  -e FULL="${FULL:-0}" -e TIMING="${TIMING:-0}" -e RUNS="${RUNS:-10}" \
  "$IMAGE" /w/tools/test_topk_det_gpu.py
