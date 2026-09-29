#!/bin/zsh
# ds41-ttft2 served partial-load leg: <tree>'s supervisor on :12698 with a host_server.py partial worker (~31 GB,
# no gpu.lock) as its og child, traced (DS41_FE_TRACE/DIGEST), then fe_bench and optional extra client scripts.
# Run it under ttft2_guard.py (production idle, nothing else heavy); the supervisor is stopped by PID on exit.
# usage: leg.sh <tree-name> <window> <phase> "<fe_bench args>" [env...]   LEG_EXTRA="<script args>" runs after fe_bench
unsetopt BG_NICE
tree=$1; window=$2; phase=$3; bench=$4; shift 4
PY=~/llm/.venv-ds41-omlx-tiles/bin/python
HERE=~/src/wt/ds41-ttft2
logs=~/llm/ds41/ttft2/fe/$window
mkdir -p $logs
cd ~/llm/ds41/ttft2
sup=
cleanup() {
  [[ -n $sup ]] && kill $sup 2>/dev/null
  for i in $(seq 1 60); do [[ -n $sup ]] && ps -p $sup >/dev/null || break; sleep 1; done
  w=$(lsof -t -iTCP:12697 -sTCP:LISTEN 2>/dev/null); [[ -n $w ]] && { echo "worker $w still listening: TERM"; kill $w; }
  echo "leg $window-$phase stopped"
}
trap cleanup EXIT INT TERM
start=$(cat $logs/og-child.log 2>/dev/null | wc -l)
env "$@" DS41_TREE=$HOME/src/wt/$tree DS41_FE_TRACE=1 DS41_FE_DIGEST=1 HS_DRAFTS=1 DS41_OG_GPU_EXEC= \
  DS41_OG_WORKER_PORT=12697 DS41_OG_LOGS=$logs DS41_OG_HOME=$HOME/llm/ds41/ttft2/og DS41_OG_CONCURRENCY=4 \
  DS41_OG_CHILD_OG="$PY -u $HERE/benchmarks/og/ttft2_host.py --host 127.0.0.1 --port {port}" \
  $PY -u $HOME/src/wt/$tree/og_serve/ds41_og.py --host 127.0.0.1 --port 12698 >> $logs/supervisor.log 2>&1 &
sup=$!
echo "$(date -u +%FT%TZ) leg $window-$phase tree=$tree env=$* sup=$sup"
for i in $(seq 1 150); do sleep 2; [[ $(curl -s -m 2 -o /dev/null -w '%{http_code}' localhost:12698/health) == 200 ]] && break; done
curl -s -m 2 localhost:12697/og/stats >/dev/null || { echo WORKER_NOT_READY; exit 1; }
if [[ -n $bench ]]; then
  FE_DOC_TREE=$HOME/src/wt/ds41-og $PY -u $HERE/og_serve/fe_bench.py --base http://127.0.0.1:12698 --model ds41-og \
    --label $window-$phase ${=bench} 2>&1 | tail -1
fi
if [[ -n $LEG_EXTRA ]]; then
  $PY -u ${=LEG_EXTRA} http://127.0.0.1:12698 $window-$phase 2>&1 | tail -20
fi
curl -s -m 3 localhost:12697/og/stats; echo
tail -n +$((start + 1)) $logs/og-child.log | grep '"event": "drafts"' > $logs/drafts-$phase.jsonl
