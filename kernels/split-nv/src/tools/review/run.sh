#!/bin/bash
# Review microbench container on GPU ${GPU:-1}; only when the engine is idle. usage: run.sh <python args...>
/mnt/nvme-1/split-nv-ops/review/idle.sh || { echo "engine busy, not starting"; exit 3; }
exec docker run --rm --name og-review-$$ --gpus "\"device=${GPU:-1}\"" --ulimit core=0 --ipc=host --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OG_CKPT=/ckpt -e TORCH_EXTENSIONS_DIR=/review/build \
  -e OG_MOE_BUILD=/review/build/og -e TRITON_CACHE_DIR=/review/build/triton \
  -v /home/ian/split-nv-moe:/work -v /home/ian/split-nv/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/ckpt:ro -v /dev/shm/split-nv/og:/traces:ro \
  -v /mnt/nvme-1/split-nv-ops/review:/review -v /mnt/nvme-1/split-nv-ops/traces:/tr:ro \
  -w /work local/sglang-dsv41:base python3 "$@"
