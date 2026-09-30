#!/bin/bash
# DSpark-on-box kill gates in ONE announced box window (~60-75 min, no deploy: production code and config unchanged).
# Before: the coordinator announces the window (Mac ds41 told the box is down) and gets the owner's go.
# usage: tools/dspark_box/window.sh <all|load|bench|ballast> <label>
#   load     gate (b) smoke, ~2-3 min of engine downtime: run_bench.sh --load-only (model + drafter load, post-load
#            checks, og-moe layout on the real swizzled scales, one eager + one graph-captured drafter cycle)
#   bench    gate (b): engine stopped, TP2 CUDA-graph drafter microbench (run_bench.sh). GO if job <= 1.8 ms at W=5.
#   ballast  gate (a): engine running, 3.9 GiB/GPU ballast vs control, fresh engine per arm (ballast_gate.py).
#            PASS if 1M prefill <= +1% and no OOM / crash; WARN if allocator retries grow.
#   all      bench first (a NO-GO there makes the ballast arms moot: stop and report), then ballast.
# Ends with the engine up (fresh restart), one og-s4.4 byte gate (8193), and the box llama-swap started again.
set -u
cd "$(dirname "$0")/../.."
STAGE=${1:?all|load|bench|ballast}; LABEL=${2:?label}
OPS=/mnt/nvme-1/split-nv-ops/dspark-box
OUT=$OPS/window-$LABEL
PY=/home/ian/.venv/bin/python
mkdir -p "$OUT"
export SPLIT_NV_WINDOW=1
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/window.log"; }
healthy() { curl -s -m 3 127.0.0.1:10051/health | grep -q '"ok": true'; }
wait_healthy() { for _ in $(seq 1 300); do healthy && return 0; sleep 3; done; return 1; }
quiet() { for _ in $(seq 1 120); do
            s=$(curl -s -m 3 127.0.0.1:10051/health | python3 -c 'import json,sys; print(json.load(sys.stdin)["sessions"])' 2>/dev/null)
            [ "$s" = 0 ] && return 0; sleep 5; done; return 1; }

log "window $LABEL stage $STAGE; engine $(curl -s -m 3 127.0.0.1:10051/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print(h["version"], h["numerics"])' 2>/dev/null)"
systemctl --user stop llama-swap && log "box llama-swap stopped"
# the 10-min prefix-cache mirror (rsync of /dev/shm to NVMe) would add I/O noise to the 1% prefill comparison
systemctl --user stop split-nv-cache-sync.timer && log "cache mirror timer paused"
# early exits leave the box llama-swap STOPPED on purpose (it grabs the GPUs if the engine is down); the timer resumes
trap 'systemctl --user start split-nv-cache-sync.timer; systemctl --user is-active --quiet llama-swap || echo "NOTE: box llama-swap is stopped; start it once the engine is healthy"' EXIT
rc=0
if [ "$STAGE" = all ] || [ "$STAGE" = load ] || [ "$STAGE" = bench ]; then
  quiet || { log "sessions still open after 10 min: not a quiet box, abort"; exit 2; }
  log "stopping split-nv-engine for the microbench"
  systemctl --user stop split-nv-engine
  for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
  if [ "$STAGE" != bench ]; then
    OPS=$OPS tools/dspark_box/run_bench.sh --load-only --out /dsb/out/window-$LABEL-load 2>&1 | tail -25 | tee -a "$OUT/window.log"
    lrc=${PIPESTATUS[0]}
    cp -p "$OPS/out/window-$LABEL-load"/*.json "$OUT/" 2>/dev/null
    log "load-only: rc=$lrc"
    if [ "$lrc" != 0 ] || [ "$STAGE" = load ]; then
      [ "$lrc" = 0 ] || { rc=1; log "load-only failed: skipping the bench and the ballast gate"; }
      STAGE=done
    fi
  fi
  if [ "$STAGE" = done ]; then
    log "starting split-nv-engine"
    systemctl --user start split-nv-engine
    wait_healthy || { log "ENGINE DID NOT COME UP"; exit 3; }
  fi
fi
if [ "$STAGE" = all ] || [ "$STAGE" = bench ]; then
  OPS=$OPS tools/dspark_box/run_bench.sh --iters 300 --widths 4,5 --out /dsb/out/window-$LABEL 2>&1 | tail -40 | tee -a "$OUT/window.log"
  cp -p "$OPS/out/window-$LABEL"/*.json "$OUT/" 2>/dev/null
  log "starting split-nv-engine"
  systemctl --user start split-nv-engine
  wait_healthy || { log "ENGINE DID NOT COME UP"; exit 3; }
  v=$(python3 -c 'import json; print(json.load(open("'"$OUT"'/verdict.json"))["verdict"])' 2>/dev/null || echo "NO RESULT")
  log "gate (b): $v"
  case "$v" in GO*) ;; *) rc=1; [ "$STAGE" = all ] && { log "gate (b) not GO: skipping the ballast gate"; STAGE=done; } ;; esac
fi
if [ "$STAGE" = all ] || [ "$STAGE" = ballast ]; then
  wait_healthy || { log "engine not healthy"; exit 3; }
  $PY tools/dspark_box/ballast_gate.py --gib 3.9 --pairs 2 --out "$OUT/ballast" 2>&1 | tee -a "$OUT/window.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
fi
wait_healthy || { log "ENGINE NOT HEALTHY AT THE END"; exit 3; }
R=/home/ian/split-nv/ref
REF=/mnt/nvme-1/split-nv-ops/og-moe-deploy/box-perf-2/ref
$PY tools/og_gate.py check --ids $R/ids-131072.json --n 8193 --ref $REF > "$OUT/og_gate-8193.txt" 2>&1 \
  && log "og_gate 8193: pass" || { log "og_gate 8193: FAIL (see $OUT/og_gate-8193.txt)"; rc=4; }
systemctl --user start llama-swap && log "box llama-swap started"
log "done rc=$rc; results in $OUT"
exit $rc
