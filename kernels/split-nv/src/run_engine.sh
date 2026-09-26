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
SGLANG_VERSION=$(git -C "$ROOT/sglang" rev-parse --short HEAD 2>/dev/null)$(git -C "$ROOT/sglang" diff --quiet 2>/dev/null || echo -dirty)
mkdir -p /dev/shm/split-nv
exec docker run --name "$NAME" --init --rm --ulimit core=0 --gpus all --runtime nvidia --ipc=host --network host \
  --stop-timeout 60 --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
  -e SGLANG_SM120_FLASHMLA_BACKEND=flashinfer -e SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DSV41_ENGRAM_HOST_TABLE_DIR=/engram \
  -e SGLANG_DSV41_ENGRAM_PINNED="$ENGRAM_PINNED" -e SGLANG_DSV41_ENGRAM_PREWARM=1 \
  -e SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB=256 -e SGLANG_OPT_USE_TOPK_V2=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/home/ian/split-nv/hooks \
  -e SPLIT_NV_HOOKS=1 -e SPLIT_NV_CONFIG=/home/ian/split-nv/$MODEL/config.json \
  -e SPLIT_NV_DIR=/dev/shm/split-nv -e SPLIT_NV_MAX_TOKENS=1056768 -e SPLIT_NV_STEP_LOG="${SPLIT_NV_STEP_LOG:-}" \
  -e SPLIT_NV_VERSION="$VERSION" -e SPLIT_NV_SGLANG_VERSION="$SGLANG_VERSION" -e SPLIT_NV_CACHE_GB="${SPLIT_NV_CACHE_GB:-64}" -e SPLIT_NV_DRAIN_S="${SPLIT_NV_DRAIN_S:-30}" \
  -e SPLIT_NV_PUBLIC_HTTP="${SPLIT_NV_PUBLIC_HTTP-0.0.0.0:10051}" \
  -e SPLIT_NV_SELFTEST="${SPLIT_NV_SELFTEST:-}" -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" -e SPLIT_NV_GRAPHS="${SPLIT_NV_GRAPHS-2,3,4,5}" \
  -e SPLIT_NV_TRACE="${SPLIT_NV_TRACE:-}" -e SPLIT_NV_DEV="${SPLIT_NV_DEV:-}" -e SPLIT_NV_TRIM="${SPLIT_NV_TRIM-1}" -e SPLIT_NV_B12X="${SPLIT_NV_B12X-1}" \
  -e SPLIT_NV_OG_MOE="${SPLIT_NV_OG_MOE-1}" \
  -v "$ROOT":/home/ian/split-nv:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/home/ian/models/DeepSeek-V4.1-Flash-original:ro \
  -v "$ROOT"/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /home/ian/models/dsv41-engram:/engram \
  -v /mnt/nvme-1/dsv41-fi-cache:/root/.cache/flashinfer -v /mnt/nvme-1/dsv41-sglang-jit:/root/.cache/sglang \
  local/sglang-dsv41:base python3 -m split_nv.engine \
  --model-path /home/ian/split-nv/$MODEL --trust-remote-code --served-model-name split-nv-encoder \
  --tp 2 --host 127.0.0.1 --port 10050 --mem-fraction-static "$MEMFRAC" \
  --context-length 1048576 --max-total-tokens "$MAX_TOTAL" --max-running-requests "$MAX_REQS" \
  --chunked-prefill-size 8192 --enable-deepseek-v4-fp4-indexer --fp8-gemm-backend flashinfer_cutlass \
  --disable-cuda-graph --disable-radix-cache "$@"
