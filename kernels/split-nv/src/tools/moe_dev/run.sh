#!/bin/bash
# Dev container on GPU ${GPU:-1}: small allocations only, core=0. usage: run.sh <python args...>
exec docker run --rm --name og-moe-dev-$$ --gpus "\"device=${GPU:-1}\"" --ulimit core=0 --ipc=host --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OG_MOE_PTW=${OG_MOE_PTW:-} -e OG_CKPT=/ckpt -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -v /home/ian/split-nv-moe:/work -v /home/ian/split-nv/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/ckpt:ro -v /dev/shm/split-nv/og:/traces:ro \
  -v /mnt/nvme-2/og-moe:/scratch -v /mnt/nvme-2/og-moe/cache/fi:/root/.cache/flashinfer \
  -v /mnt/nvme-2/og-moe/cache/sgl2:/root/.cache/sglang -v /mnt/nvme-2/og-moe/cache/torch_ext:/cache/torch_ext \
  -w /work local/sglang-dsv41:base python3 "$@"
