#!/bin/bash
# CPU-only dry run of the drafter microbench's weight load (tools/dspark_box/load_dryrun.py). No --gpus: the
# container cannot see a GPU, so it is safe any time (it reads the ~7.4 GiB mtp.* shards twice; ~8 GiB host RAM).
set -u
cd "$(dirname "$0")/../.."
WT=$(pwd)
IMAGE=${IMAGE:-sglang-dsv41-split:6152b54}
M=/home/ian/models/DeepSeek-V4.1-Flash-original
OPS=${OPS:-/mnt/nvme-1/split-nv-ops/dspark-box}
mkdir -p "$OPS"/{out,shm}
[ -f "$OPS/view2/config.json" ] || python3 tools/dspark_box/bench_view.py "$OPS/view2" || exit 1
exec docker run --rm --name dspark-dryrun-$$ --cpus ${CPUS:-6} --memory 40g --ulimit core=0 --network none \
  --shm-size 8g -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OG_CKPT=/ckpt -e VIEW=/dsb/view2 \
  -e PYTHONPATH=/work/hooks -e OG_MOE3_BUILD=/dsb/build/og_moe3 -e DRYRUN_OUT=/dsb/out/load_dryrun \
  -v "$WT":/work:ro -v $M:/ckpt:ro -v $M:$M:ro -v "$OPS":/dsb -w /work/tools/dspark_box \
  "$IMAGE" nice -n 19 python3 load_dryrun.py
