#!/bin/bash
# Swap the phase-1 SGLang server for the phase-3 engine: self-test first (exits), then the real engine, then health checks.
set -u
cd /home/ian/split-nv || exit 1
stop() { docker stop -t 10 split-nv-encoder >/dev/null 2>&1; for i in $(seq 1 40); do docker ps -a --format '{{.Names}}' | grep -q '^split-nv-encoder$' || break; sleep 1; done; }
stop; rm -f /dev/shm/split-nv/*.safetensors
echo "== self-test ($(date +%H:%M:%S))"
SPLIT_NV_SELFTEST=${SELFTEST:-aligned256,unaligned8,unaligned300,long9000} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} ./run_engine.sh > logs/engine-selftest.log 2>&1
grep -E "engine\] selftest|failed|Error|Traceback" logs/engine-selftest.log | cut -c1-220
if ! grep -q "selftest done" logs/engine-selftest.log; then echo "SELFTEST FAILED: restoring the phase-1 server"; stop; nohup ./run_encoder.sh > logs/encoder-restore.log 2>&1 & exit 1; fi
stop
echo "== engine boot ($(date +%H:%M:%S))"
nohup ./run_engine.sh > logs/engine.log 2>&1 &
until grep -qE "\[engine\] step api on|failed|Traceback" logs/engine.log 2>/dev/null; do docker ps --format '{{.Names}}' | grep -q split-nv-encoder || { echo ENGINE_GONE; exit 1; }; sleep 3; done
grep -E "engine\]" logs/engine.log | tail -5
curl -s localhost:10050/health; echo; curl -s localhost:10051/health; echo
