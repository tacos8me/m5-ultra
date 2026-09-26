#!/bin/bash
# One-time move of the engine to the user unit (s4). Waits until 10052 has been idle for 60 s.
set -u
idle=0
while [ $idle -lt 12 ]; do
  s=$(curl -s -m 3 127.0.0.1:10050/health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sessions", 1))' 2>/dev/null || echo 0)
  if [ "$s" = "0" ]; then idle=$((idle+1)); else idle=0; fi
  sleep 5
done
date -u +"%T idle 60 s; stopping old engine"
docker stop -t 10 split-nv-encoder >/dev/null 2>&1
kill -TERM 157505 2>/dev/null; sleep 2; ps -p 157505 >/dev/null && echo "server.py still alive" || echo "server.py stopped"
for i in $(seq 1 20); do docker ps -a --format '{{.Names}}' | grep -qx split-nv-encoder || break; sleep 1; done
mkdir -p ~/.config/systemd/user
cp /home/ian/split-nv/deploy/split-nv-engine.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now split-nv-engine
date -u +"%T unit started"
for i in $(seq 1 120); do
  journalctl --user -u split-nv-engine --since "-10min" --no-pager 2>/dev/null | grep -q "step api on" && break
  sleep 3
done
date -u +"%T step api"
journalctl --user -u split-nv-engine --since "-10min" --no-pager | grep -E "DSV4 pool sizes|SWA sizing|model ready|warm-up done|http on|step api on|Traceback|Error" | tail -12
