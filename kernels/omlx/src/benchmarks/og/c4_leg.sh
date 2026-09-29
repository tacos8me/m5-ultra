#!/bin/zsh
# One traced served-flow leg on the partial model: c4_host.py under idle_guard.py at :12699, then a client.
# usage: c4_leg.sh <label> <client cmd...>     env passes through (e.g. DS41_OG_FUSE_MIN=2 HS_FIXED_DEPTH=1)
# Production idle only; <= 2 box sessions (DS41_OG_CONCURRENCY=2); the guard TERMs the server on traffic.
unsetopt BG_NICE
label=$1; shift
T=$HOME/src/wt/ds41-c4; PY=$HOME/llm/.venv-ds41-omlx-tiles/bin/python; D=$HOME/llm/ds41/c4
python3 $HOME/src/wt/ds41-prof/benchmarks/og/prof_idle.py >/dev/null || { echo NOT_IDLE; exit 1; }
cd $D
DS41_OG_CONCURRENCY=${DS41_OG_CONCURRENCY:-2} DS41_TREE=$T DS41_OG_HOME=$D/og DECODE_MAX_SECONDS=${LEG_S:-420} \
  nohup $PY $T/benchmarks/og/idle_guard.py $PY -u $T/benchmarks/og/c4_host.py --host 127.0.0.1 --port 12699 \
  > $D/server-$label.log 2>&1 &
gpid=$!
for i in $(seq 1 240); do sleep 2; curl -sf -m 2 localhost:12699/health >/dev/null && break; ps -p $gpid >/dev/null || break; done
DS41_TREE=$T "$@" 2>&1 | grep -v Warning | tee -a $D/client-$label.log
spid=$(grep -m1 '"guard": "start"' $D/server-$label.log | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["pid"])')
[[ -n $spid ]] && ps -p $spid >/dev/null && kill $spid
wait $gpid
tail -2 $D/server-$label.log
