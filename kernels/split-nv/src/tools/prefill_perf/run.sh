#!/bin/bash
# Prefill-perf prototype container (production image sglang-dsv41-split:6152b54), only when the engine is idle.
# GPUS=1 (default) or GPUS=0,1 (TP2 tests). Every process must stay <= 5 GB per GPU (the harness caps torch).
# Private /dev/shm (no --ipc=host): nothing here can touch the production engine's /dev/shm/split-nv.
# A watcher kills this container (by its own name) as soon as the engine is no longer idle.
# usage: run.sh <python args...>
IDLE=/mnt/nvme-1/split-nv-ops/review/idle.sh
$IDLE || { echo "engine busy, not starting"; exit 3; }
G=${GPUS:-1}
P=/mnt/nvme-1/split-nv-ops/prefill
M=/home/ian/models/DeepSeek-V4.1-Flash-original
NAME=pf-perf-$$
(
  sleep 5
  while docker ps --format '{{.Names}}' | grep -qx $NAME; do
    if ! $IDLE; then
      echo "[run.sh] engine busy: killing $NAME" >&2
      docker kill $NAME >/dev/null 2>&1
      break
    fi
    # the 5 GB/GPU rule: any process between 5000 MiB and 80 GB (production holds ~88 GB) is ours
    big=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$2 > 5000 && $2 < 80000')
    if [ -n "$big" ]; then
      echo "[run.sh] over 5 GB per GPU ($big): killing $NAME" >&2
      docker kill $NAME >/dev/null 2>&1
      break
    fi
    sleep 2
  done
) &
WATCH=$!
# never leave the container behind (a `timeout` around this script only kills the docker CLI otherwise)
cleanup() { docker kill $NAME >/dev/null 2>&1; kill $WATCH 2>/dev/null; }
trap 'cleanup; exit 143' TERM INT HUP
docker run --rm --name $NAME --gpus "\"device=$G\"" --ulimit core=0 --network none \
  --ulimit memlock=-1 --shm-size 8g \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e OG_CKPT=/ckpt -e PYTHONPATH=/work/hooks \
  -e TORCH_EXTENSIONS_DIR=/pf/build/ext -e TRITON_CACHE_DIR=/pf/build/triton -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} ${EXTRA_ENV:-} \
  -v /home/ian/split-nv-moe:/work:ro -v $M:/ckpt:ro -v $M:$M:ro -v /home/ian/split-nv/ref:/ref:ro \
  -v $P:/pf -v $P/cache/fi:/root/.cache/flashinfer -v $P/cache/sgl:/root/.cache/sglang ${EXTRA_MOUNTS:-} \
  -w /work sglang-dsv41-split:6152b54 python3 "$@" &
DOCKER=$!
wait $DOCKER
rc=$?
kill $WATCH 2>/dev/null
exit $rc
