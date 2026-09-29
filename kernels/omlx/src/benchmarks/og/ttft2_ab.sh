#!/bin/zsh
# ds41-ttft2 replay A/B: same profiler, baseline tree (production ds41-probe, read-only) then this tree.
# usage: ab.sh <label> [env...]   (RP_CASES / RP_TESTS / RP_REPS from the environment)
unsetopt BG_NICE
label=$1; shift
PY=~/llm/.venv-ds41-omlx-tiles/bin/python
H=~/src/wt/ds41-ttft2/benchmarks/og
cd ~/llm/ds41/ttft2
: ${RP_CASES:=8217@8217,8290@8290,8600@8600,300@300,40@40,180@180,131162@8290,131500@8600,524400@8600}
export RP_CASES
for side in A B; do
  tree=$([[ $side == A ]] && echo ds41-probe || echo ds41-ttft2)
  rm -f ~/llm/ds41/ttft2/$label-$side.jsonl; env "$@" DS41_TREE=$HOME/src/wt/$tree RP_LABEL=$label-$side GUARD_MAX_S=400 $H/ttft2_run.sh $H/ttft2_replay_prof.py 2>&1 \
    | grep -v '^\[transformers\]' | grep -o '"kind": "time".*\|guard.*"reason.*\|Error.*'
  sleep 2
done
$PY $H/ttft2_cmp.py $label-A $label-B
