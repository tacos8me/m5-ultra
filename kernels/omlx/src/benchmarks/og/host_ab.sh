#!/bin/zsh
# One served-flow A/B leg: (re)start benchmarks/og/host_server.py on ~/src/wt/<tree> at :12699, run host_cycle.py, print
# the summary and the median HS_TRACE per-step timeline. Needs ~/llm/ds41/host/og (scratch og home). Production idle only.
# usage: ab.sh <tree-name> <label> [extra env...]
unsetopt BG_NICE
tree=$1; label=$2; shift 2
pidf=~/llm/ds41/host/server.pid
if [[ -f $pidf ]] && ps -p $(cat $pidf) >/dev/null; then kill $(cat $pidf); while ps -p $(cat $pidf) >/dev/null; do sleep 1; done; fi
python3 ~/src/wt/ds41-prof/benchmarks/og/prof_idle.py >/dev/null || { echo NOT_IDLE; exit 1; }
cd ~/llm/ds41/host
env "$@" HS_TRACE=1 DS41_TREE=$HOME/src/wt/$tree DS41_OG_HOME=$HOME/llm/ds41/host/og nohup ~/llm/.venv-ds41-omlx-tiles/bin/python -u ~/src/wt/ds41-host/benchmarks/og/host_server.py --host 127.0.0.1 --port 12699 > server-$label.log 2>&1 &
echo $! > $pidf
for i in $(seq 1 120); do sleep 2; curl -s -m 2 localhost:12699/v1/models >/dev/null && break; done
DS41_TREE=$HOME/src/wt/ds41-host-base HC_REPS=${HC_REPS:-3} ~/llm/.venv-ds41-omlx-tiles/bin/python -u ~/src/wt/ds41-host/benchmarks/og/host_cycle.py http://127.0.0.1:12699 $label 2>&1 | grep summary
grep '"trace"' server-$label.log | tail -4 | python3 -c "
import sys, json, statistics
rows=[json.loads(l)['per_step_ms'] for l in sys.stdin]
rows=[r for r in rows if r.get('between_steps',0) < 0.5] or rows
print(json.dumps({k: round(statistics.median(r[k] for r in rows),3) for k in rows[0]}))"
