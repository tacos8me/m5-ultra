"""Runtime switches for the box-perf step/prefill changes, read from SPLIT_NV_DIR/box-perf-flags.json (re-read at
most once a second; a missing or invalid file means the defaults). Lets one maintenance window A/B them without a
restart. Keys: inline, spin_s, r1_spin_s, early_d2h, step_log, engram_prefetch; read on rank 0 only and sent in
the command (collective decisions): pf_overlap, preempt, share_chunk, idx_rowsplit."""
import json
import os
import time

PATH = os.path.join(os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"), "box-perf-flags.json")
_state = {"t": -1e9, "v": {}}


def flag(name, default):
    now = time.monotonic()
    if now - _state["t"] > 1.0:
        _state["t"] = now
        try:
            with open(PATH) as f:
                v = json.load(f)
            _state["v"] = v if isinstance(v, dict) else {}
        except (OSError, ValueError):
            _state["v"] = {}
    return _state["v"].get(name, default)
