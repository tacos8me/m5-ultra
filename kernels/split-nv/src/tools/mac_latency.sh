#!/bin/bash
# Round-trip step latency measured from the Mac over the direct link. usage: tools/mac_latency.sh ids-8192.json [reps]
scp -q /home/ian/split-nv/tools/step_client.py m5:~/llm/ds41/split-nv/step_client.py
scp -q /home/ian/split-nv/ref/$1 m5:~/llm/ds41/split-nv/$1
ssh m5 "cd ~/llm/ds41/split-nv && ~/llm/.venv-omlx/bin/python step_client.py latency $1 --host 10.10.10.1 --port 10052 --reps ${2:-12} --state none"
