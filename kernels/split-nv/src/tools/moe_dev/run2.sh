#!/bin/bash
# Two-GPU dev container (TP2 tests): small allocations only, core=0.
exec docker run --rm --name og-moe-dev2-$$ --gpus '"device=0,1"' --ulimit core=0 --ipc=host --network host \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OG_CKPT=/ckpt -e NCCL_DEBUG=WARN -e LAYER=${LAYER:-3} -e NE=${NE:-32} \
  -v /home/ian/split-nv-moe:/work -v /home/ian/split-nv/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/home/ian/models/DeepSeek-V4.1-Flash-original:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/ckpt:ro -v /home/ian/split-nv/${MODEL:-encoder-model}:/model:ro -e VL=${VL:-0} -e TIME=${TIME:-0} -e PROF=${PROF:-0} \
  -v /dev/shm/split-nv/og:/traces:ro -v /mnt/nvme-2/og-moe:/scratch -v /mnt/nvme-2/og-moe/cache/fi:/root/.cache/flashinfer \
  -v /mnt/nvme-2/og-moe/cache/sgl2:/root/.cache/sglang -w /work local/sglang-dsv41:base python3 "$@"
