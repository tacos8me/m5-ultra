#!/bin/bash
# Post-deploy gates (box must be quiet). usage: tools/s4_after.sh <label>
LOG=/mnt/nvme-2/og-box-s4/${1:?label}-gates.log
S=/tmp/claude-1000/-home-ian-mac/3b1e9369-a329-4d79-992c-bf60818fcab6/scratchpad
cd /home/ian/split-nv
{
echo "== byte gates"; timeout 1200 tools/s4_gates.sh 2>&1 | grep -E '"pass": false|FAILED|numerical_gate|FAILED GATES'; 
echo "== resume gates"; bash $S/resume_gates.sh 2>&1 | grep -E "FAILED"
echo "== grid gate"; /home/ian/.venv/bin/python tools/og_grid_gate.py grid 2>&1 | grep -E '"gate"' | cut -c1-200
echo "== 512K x3 determinism"; /home/ian/.venv/bin/python $S/rep2.py 524288 21 3
echo "== og-cache fresh_pair"; (cd $S/ogc && timeout 300 /home/ian/.venv/bin/python benchmarks/og/fresh_pair.py 2>&1 | grep -c "differ \[\]")
} 2>&1 | tee $LOG
