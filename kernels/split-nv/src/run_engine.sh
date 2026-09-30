#!/bin/bash
# split-nv phase-3 engine: same weights/config as run_encoder.sh, driven by split_nv.engine (sessions + step API).
# Step API 0.0.0.0:10052 (STEP_API.md), HTTP 0.0.0.0:10051 + 127.0.0.1:10050 (POST /v1/prefill, GET /health).
# Runs in the foreground (docker run --rm): supervised by the user unit split-nv-engine.service (restart + boot).
set -u
NAME=${NAME:-split-nv-encoder}
MAX_TOTAL=${MAX_TOTAL:-3145728}
MAX_REQS=${MAX_REQS:-8}
MEMFRAC=${MEMFRAC:-0.94}
ENGRAM_PINNED=${ENGRAM_PINNED-copy}
# Everything comes from ROOT, mounted read-only at /home/ian/split-nv: in production the pinned deploy clone
# /home/ian/split-nv-deploy (its own sglang/ tree, encoder-model/ and ref/ copies; nothing from the dev tree).
ROOT=${SPLIT_NV_ROOT:-/home/ian/split-nv}
# encoder-model-vl: layers 0-20 + vision tower/aligner/bias_vl (tools/build_encoder_view.py --vision);
# encoder-model: the text-only view (rollback)
MODEL=${SPLIT_NV_MODEL:-encoder-model-vl}
VERSION=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo -dirty)
# Image: sglang-dsv41-split:<m5-ultra commit> is built from public parts only (m5-ultra kernels/split-nv/Dockerfile:
# lmsysorg/sglang:dev-dsv41 + SGLang and FlashInfer patches) and carries the tested SGLang tree itself; proven
# bit-identical to the old local image. local/sglang-dsv41:base (private base) still works: it mounts ROOT/sglang.
IMAGE=${SPLIT_NV_IMAGE:-sglang-dsv41-split:6152b54}
if [ "$IMAGE" = local/sglang-dsv41:base ]; then
  SGLANG_MOUNT=(-v "$ROOT"/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro)
  SGLANG_VERSION=$(git -C "$ROOT/sglang" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT/sglang" diff --quiet 2>/dev/null || echo -dirty)
else
  SGLANG_MOUNT=(); SGLANG_VERSION=$IMAGE
fi
mkdir -p /dev/shm/split-nv
# PyTorch caching-allocator options for memory windows only (e.g. garbage_collection_threshold:0.9 together with
# SPLIT_NV_MEM_FRACTION). Never expandable_segments: the step graphs' custom all-reduce needs IPC-able segments.
ALLOC_CONF=(); [ -n "${SPLIT_NV_ALLOC_CONF:-}" ] && ALLOC_CONF=(-e PYTORCH_CUDA_ALLOC_CONF="$SPLIT_NV_ALLOC_CONF")
# Bytecode cache outside the source trees (the image's SGLang tree ships no .pyc): each of the 3 processes (main +
# 2 ranks) otherwise recompiles every imported module, ~11 s -> ~5.6 s of imports per process, twice in series.
# Timestamp-validated, so a deploy checkout (new mtimes) recompiles what changed. SPLIT_NV_PYCACHE= (empty) = old mode.
PYCACHE=${SPLIT_NV_PYCACHE-/mnt/nvme-1/dsv41-pycache}
if [ -n "$PYCACHE" ]; then
  PYC=(-v "$PYCACHE":/root/.cache/pyc -e PYTHONPYCACHEPREFIX=/root/.cache/pyc)
else
  PYC=(-e PYTHONDONTWRITEBYTECODE=1)
fi
# OOM priority of every process in the container (docker sets it as root; no extra privilege). Each rank's RSS
# (~200 GB) counts the ~189 GiB of kept Engram SysV segments that its death does not free, so with adj 0 the kernel
# picks a rank (oom_score ~920) before any other process except traefik/coredns (adj 1000), frees a few GB and takes
# production down. -500 (oom_score ~590) puts them after every adj >= 0 process (user services ~800, containers
# >= ~667) but before dockerd (-500, small RSS; killing it would stop the engine anyway). 0 = kernel default.
OOM_ADJ=${SPLIT_NV_OOM_SCORE_ADJ:--500}
# Profiling windows only (default off): SPLIT_NV_NSYS=<host dir> starts the engine under `nsys launch` (session "prof",
# CUDA + NVTX + OS runtime) through tools/box_perf/prof_entry.py (NVTX command/layer ranges, no numerical change), with
# every flag of this script. Collect: docker exec split-nv-encoder nsys start --session=prof -o /traces/<name>
# --force-overwrite=true; ...; docker exec split-nv-encoder nsys stop --session=prof  -> <host dir>/<name>.nsys-rep
PYPATH=/home/ian/split-nv/hooks; ENTRY=(python3 -m split_nv.engine); NSYS=()
if [ -n "${SPLIT_NV_NSYS:-}" ]; then
  mkdir -p "$SPLIT_NV_NSYS"
  PYPATH=$PYPATH:/home/ian/split-nv/tools/box_perf
  NSYS=(--cap-add SYS_ADMIN -v "$SPLIT_NV_NSYS":/traces)
  ENTRY=(nsys launch --session-new=prof --trace=cuda,nvtx,osrt --cuda-graph-trace=node:nvtx-precapture python3 -m prof_entry)
fi
# DSpark drafter on the box (STEPD-SPEC.md; default off = today's engine, nothing below is passed): SPLIT_NV_DSPARK=1
# loads the 3 stages (~3.9 GiB/GPU) from the original checkpoint (mounted below) and advertises the capability.
# og_moe3 builds into /root/.cache/sglang/og_moe3 (persistent JIT dir); the fused Markov kernels' Triton cache too.
DSPARK_ENV=()
if [ "${SPLIT_NV_DSPARK:-0}" = 1 ]; then
  DSPARK_ENV=(-e SPLIT_NV_DSPARK=1 -e SPLIT_NV_DSPARK_HEAD="${SPLIT_NV_DSPARK_HEAD-fp8}"
    -e SPLIT_NV_DSPARK_MARKOV="${SPLIT_NV_DSPARK_MARKOV-fused}" -e SPLIT_NV_DSPARK_MOE="${SPLIT_NV_DSPARK_MOE-og3}"
    -e SPLIT_NV_DSPARK_SLOTS="${SPLIT_NV_DSPARK_SLOTS-8}" -e SPLIT_NV_DSPARK_WIDTHS="${SPLIT_NV_DSPARK_WIDTHS-4}"
    -e SPLIT_NV_DSPARK_FREE_BF16_HEAD="${SPLIT_NV_DSPARK_FREE_BF16_HEAD-1}"
    -e SPLIT_NV_DSPARK_CKPT=/home/ian/models/DeepSeek-V4.1-Flash-original -e TRITON_CACHE_DIR=/root/.cache/sglang/triton-dspark)
fi
# Warm the page cache with the checkpoint tensors the ranks will load while the container starts (NVMe idle then).
if [ "${SPLIT_NV_PREFETCH-1}" = 1 ] && [ -f "$ROOT/$MODEL/model.safetensors.index.json" ]; then
  python3 "$ROOT"/tools/prefetch_weights.py "$ROOT/$MODEL" --threads "${SPLIT_NV_PREFETCH_THREADS:-12}" &
fi
exec docker run --name "$NAME" --init --rm --ulimit core=0 --oom-score-adj "$OOM_ADJ" --gpus all --runtime nvidia --ipc=host --network host \
  --stop-timeout 60 --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
  -e SGLANG_SM120_FLASHMLA_BACKEND=flashinfer -e SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DSV41_ENGRAM_HOST_TABLE_DIR=/engram \
  -e SGLANG_DSV41_ENGRAM_PINNED="$ENGRAM_PINNED" -e SGLANG_DSV41_ENGRAM_PREWARM=1 \
  -e SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB="${SPLIT_NV_IDX_BUDGET_MB:-1024}" -e SGLANG_OPT_USE_TOPK_V2=1 \
  "${PYC[@]}" -e PYTHONPATH="$PYPATH" \
  -e SPLIT_NV_HOOKS=1 -e SPLIT_NV_CONFIG=/home/ian/split-nv/$MODEL/config.json \
  -e SPLIT_NV_DIR=/dev/shm/split-nv -e SPLIT_NV_MAX_TOKENS=1056768 -e SPLIT_NV_STEP_LOG="${SPLIT_NV_STEP_LOG:-}" \
  -e SPLIT_NV_VERSION="$VERSION" -e SPLIT_NV_SGLANG_VERSION="$SGLANG_VERSION" -e SPLIT_NV_CACHE_GB="${SPLIT_NV_CACHE_GB:-96}" -e SPLIT_NV_DRAIN_S="${SPLIT_NV_DRAIN_S:-30}" \
  -e SPLIT_NV_PUBLIC_HTTP="${SPLIT_NV_PUBLIC_HTTP-0.0.0.0:10051}" \
  -e SPLIT_NV_SELFTEST="${SPLIT_NV_SELFTEST:-}" -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" -e SPLIT_NV_GRAPHS="${SPLIT_NV_GRAPHS-2,3,4,5}" \
  -e SPLIT_NV_TRACE="${SPLIT_NV_TRACE:-}" -e SPLIT_NV_DEV="${SPLIT_NV_DEV:-}" -e SPLIT_NV_TRIM="${SPLIT_NV_TRIM-1}" -e SPLIT_NV_B12X="${SPLIT_NV_B12X-1}" \
  -e SPLIT_NV_OG_MOE="${SPLIT_NV_OG_MOE-1}" -e SPLIT_NV_SPIN_S="${SPLIT_NV_SPIN_S-0.2}" \
  -e SPLIT_NV_PF_OVERLAP="${SPLIT_NV_PF_OVERLAP-0}" -e SPLIT_NV_CE_AR="${SPLIT_NV_CE_AR-0}" -e SPLIT_NV_Q_NOCOPY="${SPLIT_NV_Q_NOCOPY-0}" \
  -e SPLIT_NV_TOPK_DET="${SPLIT_NV_TOPK_DET-1}" -e SPLIT_NV_TOPK_AUDIT="${SPLIT_NV_TOPK_AUDIT:-}" \
  -e SPLIT_NV_PREEMPT="${SPLIT_NV_PREEMPT-0}" -e SPLIT_NV_PREEMPT_LAG="${SPLIT_NV_PREEMPT_LAG-2}" -e SPLIT_NV_PREEMPT_SHARE="${SPLIT_NV_PREEMPT_SHARE-0.5}" \
  -e SPLIT_NV_BYPASS_TOKENS="${SPLIT_NV_BYPASS_TOKENS-0}" -e SPLIT_NV_BYPASS_SHARE="${SPLIT_NV_BYPASS_SHARE-0.5}" \
  -e SPLIT_NV_SHARE_CHUNK="${SPLIT_NV_SHARE_CHUNK-2048}" -e SPLIT_NV_TRIM_MIN_TOKENS="${SPLIT_NV_TRIM_MIN_TOKENS-0}" \
  -e SPLIT_NV_IDX_LOWMEM="${SPLIT_NV_IDX_LOWMEM-0}" -e SPLIT_NV_IDX_ROWSPLIT="${SPLIT_NV_IDX_ROWSPLIT-0}" -e SPLIT_NV_MEMLOG="${SPLIT_NV_MEMLOG:-}" -e SPLIT_NV_MEMHIST="${SPLIT_NV_MEMHIST:-}" \
  -e SPLIT_NV_IDX_FIXED_TILES="${SPLIT_NV_IDX_FIXED_TILES-0}" -e SPLIT_NV_IDX_TILE_MARGIN="${SPLIT_NV_IDX_TILE_MARGIN-0.125}" \
  -e SPLIT_NV_IDX_TILE_MIN_MB="${SPLIT_NV_IDX_TILE_MIN_MB-64}" "${DSPARK_ENV[@]}" \
  -e SPLIT_NV_STALLWATCH="${SPLIT_NV_STALLWATCH-0}" -e SPLIT_NV_GC_FREEZE="${SPLIT_NV_GC_FREEZE-0}" -e SPLIT_NV_PARK_LOG_MS="${SPLIT_NV_PARK_LOG_MS-150}" \
  -e SPLIT_NV_MEM_FRACTION="${SPLIT_NV_MEM_FRACTION:-}" "${ALLOC_CONF[@]}" \
  -e SPLIT_NV_ENGRAM_KEEP="${SPLIT_NV_ENGRAM_KEEP-0}" -e SPLIT_NV_ENGRAM_ASYNC_REGISTER="${SPLIT_NV_ENGRAM_ASYNC_REGISTER-1}" \
  -v "$ROOT":/home/ian/split-nv:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/home/ian/models/DeepSeek-V4.1-Flash-original:ro \
  "${SGLANG_MOUNT[@]}" "${NSYS[@]}" \
  -v /home/ian/models/dsv41-engram:/engram \
  -v /mnt/nvme-1/dsv41-fi-cache:/root/.cache/flashinfer -v /mnt/nvme-1/dsv41-sglang-jit:/root/.cache/sglang \
  "$IMAGE" "${ENTRY[@]}" \
  --model-path /home/ian/split-nv/$MODEL --trust-remote-code --served-model-name split-nv-encoder \
  --tp 2 --host 127.0.0.1 --port 10050 --mem-fraction-static "$MEMFRAC" \
  --context-length 1048576 --max-total-tokens "$MAX_TOTAL" --max-running-requests "$MAX_REQS" \
  --chunked-prefill-size 8192 --enable-deepseek-v4-fp4-indexer --fp8-gemm-backend flashinfer_cutlass \
  --disable-cuda-graph --disable-radix-cache "$@"
