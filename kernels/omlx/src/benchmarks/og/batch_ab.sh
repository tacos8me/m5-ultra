#!/bin/zsh
# One served-flow leg: (re)start host_server.py (partial model) on a tree at :12699, run batch_cycle.py for each c.
# usage: batch_ab.sh <tree-name> <label> <cs e.g. "2 4"> [env...]   BC_IDENTITY=1: <cs> = identity group sizes "1,4,3,2"   (production idle only; <= 4 box sessions)
unsetopt BG_NICE
tree=$1; label=$2; cs=$3; shift 3
pidf=~/llm/ds41/batch/server.pid
if [[ -f $pidf ]] && ps -p $(cat $pidf) >/dev/null; then kill $(cat $pidf); while ps -p $(cat $pidf) >/dev/null; do sleep 1; done; fi
python3 ~/src/wt/ds41-prof/benchmarks/og/prof_idle.py >/dev/null || { echo NOT_IDLE; exit 1; }
mkdir -p ~/llm/ds41/batch/og && cd ~/llm/ds41/batch
env "$@" DS41_TREE=$HOME/src/wt/$tree DS41_OG_HOME=$HOME/llm/ds41/batch/og nohup ~/llm/.venv-ds41-omlx-tiles/bin/python -u ~/src/wt/ds41-draft/benchmarks/og/host_server.py --host 127.0.0.1 --port 12699 > server-$label.log 2>&1 &
echo $! > $pidf
for i in $(seq 1 120); do sleep 2; curl -s -m 2 localhost:12699/v1/models >/dev/null && break; done
if [[ -n $BC_IDENTITY ]]; then
  DS41_TREE=$HOME/src/wt/ds41-draft ~/llm/.venv-ds41-omlx-tiles/bin/python -u ~/src/wt/ds41-draft/benchmarks/og/batch_identity.py http://127.0.0.1:12699 $label $cs ${BC_TOKENS:-256} 2>&1 | grep '"label"' | tee -a identity.jsonl
  cs=""
fi
for c in ${=cs}; do
  DS41_TREE=$HOME/src/wt/ds41-draft ~/llm/.venv-ds41-omlx-tiles/bin/python -u ~/src/wt/ds41-draft/benchmarks/og/batch_cycle.py http://127.0.0.1:12699 $label $c ${BC_CTX:-8192} ${BC_REPS:-3} ${BC_TOKENS:-256} 2>&1 | grep '"label"' | tee -a runs.jsonl
done
kill $(cat $pidf)
