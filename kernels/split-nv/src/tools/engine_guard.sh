#!/bin/bash
# engine_guard.sh <cmd...>: prefix for box llama-swap model commands that use the GPUs.
# Refuses (exit 75) while the split-nv engine owns the GPUs or is (re)starting: during an engine restart
# (crash -> RestartSec 15 s -> ~2.5 min load) a box llama-swap request (e.g. the 15-minute glm-5.3-flash poller)
# otherwise loads a model into the gap and the engine OOMs at weight load in a loop (2026-09-28 14:45 and 17:03:
# 12 torch.OutOfMemoryError at load), which also defeats the Mac's 240 s ride-through of an engine restart.
# `systemctl --user stop split-nv-engine` (inactive) lets models start again. SPLIT_NV_GUARD=off disables it.
set -u
if [ "${SPLIT_NV_GUARD:-on}" != off ]; then
  state=$(systemctl --user is-active split-nv-engine.service 2>/dev/null)
  case "$state" in
    active|activating|reloading|deactivating)
      echo "engine_guard: split-nv-engine is $state (it owns the GPUs); refusing to start: $*" >&2
      exit 75 ;;
  esac
fi
exec "$@"
