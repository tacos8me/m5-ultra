#!/bin/bash
# split-nv encoder: DS-V4.1-Flash layers 0-20 (all 384 experts, FP4/FP8 originals) on both RTX PRO 6000,
# SGLang SM120 stack with the split_nv capture hooks. Internal SGLang port 10050 (loopback only);
# the public service is server.py on 10.10.10.1:10051.
#   ENGRAM_PINNED=copy  (default) pinned in-RAM copies of the host tables (needs the .bin+.complete files)
#   ENGRAM_PINNED=      disk-backed mmap; first boot in this mode builds sglang_engram_1.bin from shards 47/48
set -u
NAME=${NAME:-split-nv-encoder}
PORT=${PORT:-10050}
MAX_TOTAL=${MAX_TOTAL:-1056768}
MEMFRAC=${MEMFRAC:-0.94}
ENGRAM_PINNED=${ENGRAM_PINNED-copy}
mkdir -p /dev/shm/split-nv
exec docker run --name "$NAME" --init --rm --ulimit core=0 --gpus all --runtime nvidia --ipc=host --network host \
  --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
  -e SGLANG_SM120_FLASHMLA_BACKEND=flashinfer -e SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DSV41_ENGRAM_HOST_TABLE_DIR=/engram \
  -e SGLANG_DSV41_ENGRAM_PINNED="$ENGRAM_PINNED" -e SGLANG_DSV41_ENGRAM_PREWARM=1 \
  -e SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB=256 -e SGLANG_OPT_USE_TOPK_V2=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/home/ian/split-nv/hooks \
  -e SPLIT_NV_HOOKS=1 -e SPLIT_NV_CONFIG=/home/ian/split-nv/encoder-model/config.json \
  -e SPLIT_NV_DIR=/dev/shm/split-nv -e SPLIT_NV_MAX_TOKENS="$MAX_TOTAL" \
  -v /home/ian/split-nv:/home/ian/split-nv:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/home/ian/models/DeepSeek-V4.1-Flash-original:ro \
  -v /home/ian/split-nv/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /home/ian/models/dsv41-engram:/engram \
  -v /mnt/nvme-1/dsv41-fi-cache:/root/.cache/flashinfer -v /mnt/nvme-1/dsv41-sglang-jit:/root/.cache/sglang \
  local/sglang-dsv41:base python3 -m sglang.launch_server \
  --model-path /home/ian/split-nv/encoder-model --trust-remote-code --served-model-name split-nv-encoder \
  --tp 2 --host 127.0.0.1 --port "$PORT" --mem-fraction-static "$MEMFRAC" \
  --context-length 1048576 --max-total-tokens "$MAX_TOTAL" --max-running-requests 1 \
  --chunked-prefill-size 8192 --enable-deepseek-v4-fp4-indexer --fp8-gemm-backend flashinfer_cutlass \
  --disable-cuda-graph --disable-radix-cache "$@"
