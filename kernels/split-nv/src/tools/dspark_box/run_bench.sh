#!/bin/bash
# Kill gate (b): the TP2 CUDA-graph DSpark drafter microbench (tools/dspark_box/drafter_bench.py) in the production
# image on both GPUs. Box window only, with the engine STOPPED (the bench needs ~6 GB/GPU and clean timing):
#   systemctl --user stop llama-swap; systemctl --user stop split-nv-engine
#   SPLIT_NV_WINDOW=1 tools/dspark_box/run_bench.sh [drafter_bench.py args, e.g. --iters 300 --widths 4,5]
#   systemctl --user start split-nv-engine   (then the ballast gate, or the normal post-restart checks)
# Smoke first (~2-3 min of engine downtime): run_bench.sh --load-only  -> "LOAD-ONLY PASS|FAIL" (model + drafter load,
#   checkpoint/derived-parameter checks, og-moe layout on the real swizzled scales, og3 vs SGLang MoE, rope self-check,
#   one eager + one graph-captured drafter cycle, graph == eager); exit code 0 only on PASS.
# The load path itself is proven without a GPU by tools/dspark_box/load_dryrun.sh.
# Prebuild the og_moe3 kernel any time without a GPU (compiles only):  BUILD_ONLY=1 tools/dspark_box/run_bench.sh
# Private /dev/shm and no network (loopback only): nothing here can touch the engine's /dev/shm/split-nv or the Mac.
# A watcher kills the container if the engine or the box llama-swap becomes active while it runs.
set -u
cd "$(dirname "$0")/../.."
WT=$(pwd)
IMAGE=${IMAGE:-sglang-dsv41-split:6152b54}
M=/home/ian/models/DeepSeek-V4.1-Flash-original
OPS=${OPS:-/mnt/nvme-1/split-nv-ops/dspark-box}
NAME=dspark-bench-$$
mkdir -p "$OPS"/{view,out,shm,build/og_moe3,cache/fi,cache/sgl}
[ -f "$OPS/view2/config.json" ] || python3 tools/dspark_box/bench_view.py "$OPS/view2" || exit 1
COMMON=(--rm --name $NAME --ulimit core=0 --network none --ulimit memlock=-1 --shm-size 16g
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OG_CKPT=/ckpt -e VIEW=/dsb/view2 -e PYTHONPATH=/work/hooks
  -e OG_MOE3_BUILD=/dsb/build/og_moe3 -e TORCH_EXTENSIONS_DIR=/dsb/build/ext -e TRITON_CACHE_DIR=/dsb/build/triton
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e AUTOTUNE=${AUTOTUNE:-1} -e MEM_GB=${MEM_GB:-0}
  -v "$WT":/work:ro -v $M:/ckpt:ro -v $M:$M:ro -v /home/ian/split-nv/ref:/ref:ro -v "$OPS":/dsb
  -v "$OPS"/cache/fi:/root/.cache/flashinfer -v "$OPS"/cache/sgl:/root/.cache/sglang -w /work/tools/dspark_box)
if [ "${BUILD_ONLY:-0}" = 1 ]; then
  # nvcc only: no --gpus, low priority
  exec docker run "${COMMON[@]}" --cpus 4 "$IMAGE" nice -n 19 python3 drafter_bench.py --build-only
fi
[ "${SPLIT_NV_WINDOW:-0}" = 1 ] || { echo "refusing: set SPLIT_NV_WINDOW=1 inside an announced box window"; exit 2; }
if systemctl --user is-active --quiet split-nv-engine; then
  echo "refusing: split-nv-engine is active (systemctl --user stop split-nv-engine first)"; exit 2
fi
if systemctl --user is-active --quiet llama-swap; then
  echo "refusing: box llama-swap is active (systemctl --user stop llama-swap first)"; exit 2
fi
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
[ -z "$busy" ] || { echo "refusing: GPU processes present: $busy"; exit 2; }
# first run: seed the JIT caches from production's (copies; production's directories are never written)
[ -n "$(ls -A "$OPS"/cache/fi 2>/dev/null)" ] || cp -a /mnt/nvme-1/dsv41-fi-cache/. "$OPS"/cache/fi/ 2>/dev/null
[ -n "$(ls -A "$OPS"/cache/sgl 2>/dev/null)" ] || cp -a /mnt/nvme-1/dsv41-sglang-jit/. "$OPS"/cache/sgl/ 2>/dev/null
(
  sleep 5
  while docker ps --format '{{.Names}}' | grep -qx $NAME; do
    if systemctl --user is-active --quiet split-nv-engine || systemctl --user is-active --quiet llama-swap; then
      echo "[run_bench] engine or llama-swap started: killing $NAME" >&2
      docker kill $NAME >/dev/null 2>&1
      break
    fi
    sleep 2
  done
) &
WATCH=$!
cleanup() { docker kill $NAME >/dev/null 2>&1; kill $WATCH 2>/dev/null; }
trap 'cleanup; exit 143' TERM INT HUP
LOG="$OPS/out/bench-$(date -u +%Y%m%dT%H%M%S).log"
docker run "${COMMON[@]}" --gpus '"device=0,1"' "$IMAGE" python3 drafter_bench.py --out /dsb/out "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
kill $WATCH 2>/dev/null
echo "log: $LOG"
exit $rc
