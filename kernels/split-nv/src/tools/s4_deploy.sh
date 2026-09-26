#!/bin/bash
# Deploy a gated commit from the deploy worktree and run the gates. usage: tools/s4_deploy.sh <commit> <label>
set -u
C=${1:?commit}; LABEL=${2:?label}
LOG=/mnt/nvme-2/og-box-s4/$LABEL.log
git -C /home/ian/split-nv-deploy fetch -q /home/ian/split-nv '+refs/heads/*:refs/remotes/dev/*' && git -C /home/ian/split-nv-deploy checkout -q --detach "$C" || exit 1
cp /home/ian/split-nv-deploy/deploy/split-nv-engine.service ~/.config/systemd/user/split-nv-engine.service
systemctl --user daemon-reload
T=$(date -u +%H:%M:%S); echo "restart $T" | tee $LOG
systemctl --user restart split-nv-engine
until curl -s -m 2 127.0.0.1:10051/health | grep -q '"ok": true'; do
  journalctl --user -u split-nv-engine --since "$T" --no-pager | grep -q Traceback && { echo "START FAILED" | tee -a $LOG; exit 2; }
  sleep 3
done
curl -s 127.0.0.1:10051/health | tee -a $LOG; echo | tee -a $LOG
date -u +"%T up" | tee -a $LOG
