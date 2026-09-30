#!/bin/bash
# Metal System Trace of the served og worker during a c4ab run at c2 or c4 (engine-loop-pipeline go/no-go).
# WINDOW ONLY (observation; sends c4ab traffic to :8080). Usage: benchmarks/og/c4ab_trace.sh LABEL C [SECONDS]
# Waits for the discarded warm-up group and measured group 0, then records SECONDS (default 4) inside group 1,
# i.e. in steady cN decode. Output in ~/llm/ds41/next4: LABEL.trace, LABEL.c4ab.log, og/stats before/after,
# and the xt_rounds.py split (GO if within_step_a_ms >= 1.5).
set -uo pipefail
LABEL=$1; C=$2; SECS=${3:-4}
TREE=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$HOME/llm/ds41/next4; mkdir -p "$OUT"
PY=$HOME/llm/.venv-ds41-omlx-tiles/bin/python
H=$(curl -s --max-time 3 localhost:10001/health)
PID=$(printf '%s' "$H" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["pid"] if d.get("status")=="ok" and d.get("inflight")==0 else "")')
[[ -n $PID ]] || { echo "og worker not ready/idle: $H"; exit 1; }
lsof -p "$PID" 2>/dev/null | grep -q "$TREE/omlx/custom_kernels" || { echo "worker $PID does not map $TREE kernels (omlx-server renames its command)"; exit 1; }
curl -s localhost:12147/og/stats > "$OUT/$LABEL.stats0.json"
"$PY" -u "$TREE/benchmarks/og/c4ab.py" run "$LABEL" --c "$C" --groups 4 --url http://127.0.0.1:8080 --out "$OUT" \
    > "$OUT/$LABEL.c4ab.log" 2>&1 &
BP=$!
for _ in $(seq 1 1500); do grep -q '"group": 0' "$OUT/$LABEL.c4ab.log" 2>/dev/null && break; sleep 0.2; done
sleep 1.0  # group 1 admission (box OPENs + Mac import) before steady decode
rm -rf "$OUT/$LABEL.trace"
xcrun xctrace record --template "Metal System Trace" --attach "$PID" --time-limit "${SECS}s" \
    --output "$OUT/$LABEL.trace" > "$OUT/$LABEL.xctrace.log" 2>&1
wait $BP
curl -s localhost:12147/og/stats > "$OUT/$LABEL.stats1.json"
tail -1 "$OUT/$LABEL.c4ab.log"
"$PY" "$TREE/benchmarks/og/xt_rounds.py" "$OUT/$LABEL.trace" --c "$C" --skip-ms 0 | tee "$OUT/$LABEL.rounds.txt"
