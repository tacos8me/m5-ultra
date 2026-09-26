#!/bin/bash
# usage: ncu.sh <kernel regex> <launch-skip> <ncu args...> -- python args
K=$1; S=$2; shift 2
exec docker run --rm --gpus '"device=1"' --cap-add SYS_ADMIN --ulimit core=0 --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e OG_CKPT=/ckpt -v /home/ian/split-nv-moe:/work -v /home/ian/split-nv/sglang/sglang:/sgl-workspace/sglang/python/sglang:ro \
  -v /home/ian/models/DeepSeek-V4.1-Flash-original:/ckpt:ro -v /dev/shm/split-nv/og:/traces:ro -v /mnt/nvme-2/og-moe:/scratch \
  -v /mnt/nvme-2/og-moe/cache/sgl2:/root/.cache/sglang -w /work local/sglang-dsv41:base \
  ncu --clock-control none -k regex:$K --launch-skip $S --launch-count 1 "$@"
