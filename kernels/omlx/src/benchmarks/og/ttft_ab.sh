#!/bin/zsh
# One partial-load TTFT leg (ds41-ttft): this tree's supervisor on :12698 with a host_server.py partial worker
# (~31 GB, no gpu.lock) as its og child, both traced (DS41_FE_TRACE/DIGEST), then fe_bench (TTFT/resume/identity,
# one box session at a time) and optionally batch_identity (groups <= 2 box sessions). Production idle only.
# usage: ttft_ab.sh <tree-name> <window> <phase> "<fe_bench args>" [env...]
#   TTFT_GROUPS="1,2": also run batch_identity (alone vs 2 concurrent) after fe_bench.
#   TTFT_GAP="131072 524288": also run ttft_gap.py (one-message doc + new question follow-ups) at those sizes.
# Output: ~/llm/ds41/fe/<window>/{supervisor,og-child}.log + ~/llm/ds41/fe/<window>-<phase>.jsonl (fe_report.py layout).
unsetopt BG_NICE
tree=$1; window=$2; phase=$3; bench=$4; shift 4
PY=~/llm/.venv-ds41-omlx-tiles/bin/python
HERE=~/src/wt/ds41-ttft
logs=~/llm/ds41/fe/$window
pidf=~/llm/ds41/ttft/sup.pid
mkdir -p $logs ~/llm/ds41/ttft
if [[ -f $pidf ]] && ps -p $(cat $pidf) >/dev/null; then echo "supervisor $(cat $pidf) still running"; exit 1; fi
python3 ~/src/wt/ds41-prof/benchmarks/og/prof_idle.py || { echo NOT_IDLE; exit 1; }
cd ~/llm/ds41/ttft
start=$(cat $logs/og-child.log 2>/dev/null | wc -l)
env "$@" DS41_TREE=$HOME/src/wt/$tree DS41_FE_TRACE=1 DS41_FE_DIGEST=1 HS_DRAFTS=1 DS41_OG_GPU_EXEC= \
  DS41_OG_WORKER_PORT=12697 DS41_OG_LOGS=$logs DS41_OG_HOME=$HOME/llm/ds41/ttft/og DS41_OG_CONCURRENCY=4 \
  DS41_OG_CHILD_OG="$PY -u $HERE/benchmarks/og/host_server.py --host 127.0.0.1 --port {port}" \
  nohup $PY -u $HOME/src/wt/$tree/og_serve/ds41_og.py --host 127.0.0.1 --port 12698 >> $logs/supervisor.log 2>&1 &
echo $! > $pidf
echo "$(date -u +%FT%TZ) leg $window-$phase tree=$tree env=$* sup=$(cat $pidf)"
for i in $(seq 1 150); do sleep 2; [[ $(curl -s -m 2 -o /dev/null -w '%{http_code}' localhost:12698/health) == 200 ]] && break; done
curl -s -m 2 localhost:12697/og/stats >/dev/null || { echo WORKER_NOT_READY; kill $(cat $pidf); exit 1; }
if [[ -n $bench ]]; then
  FE_DOC_TREE=$HOME/src/wt/ds41-og $PY -u $HERE/og_serve/fe_bench.py --base http://127.0.0.1:12698 --model ds41-og \
    --label $window-$phase ${=bench} 2>&1 | tail -1
fi
for n in ${=TTFT_GAP}; do  # bench-leg2-shaped follow-ups (doc + a new question in one message)
  FE_DOC_TREE=$HOME/src/wt/ds41-og $PY -u $HERE/benchmarks/og/ttft_gap.py http://127.0.0.1:12698 $window-$phase $n 2>&1 | grep rep
done
if [[ -n $TTFT_GROUPS ]]; then
  DS41_TREE=$HERE $PY -u $HERE/benchmarks/og/batch_identity.py http://127.0.0.1:12697 $window-$phase $TTFT_GROUPS ${TTFT_TOKENS:-128} 2>&1 | grep '"label"'
fi
curl -s -m 3 localhost:12697/og/stats; echo
tail -n +$((start + 1)) $logs/og-child.log | grep '"event": "drafts"' > $logs/drafts-$phase.jsonl
pid=$(cat $pidf); kill $pid
for i in $(seq 1 90); do ps -p $pid >/dev/null || break; sleep 1; done
ps -p $pid >/dev/null && echo "supervisor $pid still alive" || echo "supervisor $pid stopped"
lsof -t -iTCP:12697 -sTCP:LISTEN && echo "WORKER STILL LISTENING" || echo "worker stopped"
