#!/bin/zsh
# run one ttft2 GPU job under the guard, waiting (up to WAIT_S, default 1200) for a free slot; retries once if pre-empted.
unsetopt BG_NICE
PY=~/llm/.venv-ds41-omlx-tiles/bin/python
H=~/src/wt/ds41-ttft2/benchmarks/og
cd ~/llm/ds41/ttft2
deadline=$(( $(date +%s) + ${WAIT_S:-1200} ))
tries=0
while true; do
  if [[ $1 == *.py ]]; then cmd=($PY -u "$@"); else cmd=("$@"); fi
  out=$($PY $H/ttft2_guard.py "${cmd[@]}" 2>&1); rc=$?
  if [[ $rc != 75 ]]; then print -r -- "$out"; exit $rc; fi
  (( tries % 20 == 1 )) && echo "$out" | grep -o "guard.*" | tail -1 >&2
  tries=$((tries+1))
  (( $(date +%s) > deadline )) && { echo GAVE_UP; exit 75; }
  sleep 20
done
