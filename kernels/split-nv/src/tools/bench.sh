#!/bin/bash
# Warm throughput sweep through the service; samples peak GPU memory during each run.
# usage: tools/bench.sh 8192 131072 524310 1048576
cd /home/ian/split-nv || exit 1
mkdir -p states logs
for n in "$@"; do
  ids=ref/ids-$n.json
  [ -f "$ids" ] || { echo "missing $ids"; continue; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > logs/gpumem-$n.log 2>/dev/null &
  smi=$!
  ~/.venv/bin/python tools/prefill_client.py "$ids" states/mine-$n.safetensors 2>&1 | tail -1
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  echo "peak_gpu_mem_mib n=$n: $(sort -n logs/gpumem-$n.log | tail -1) (per GPU max over both)"
done
