"""Hold several long sessions at once and step each (capacity check), e.g. 2 x 512K + 1M.

  og_capacity.py ref/ids-524310.json:524310 ref/ids-524310.json:524000 ref/ids-1048576.json:1048576
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from og_client import Session  # noqa: E402

sessions = []
out = []
try:
    for spec in sys.argv[1:]:
        path, n = spec.rsplit(":", 1)
        t = json.load(open(path))
        t = (t["tokens"] if isinstance(t, dict) else t)[:int(n)]
        s = Session()
        t0 = time.perf_counter()
        ack, _, _, info = s.open(t, state="none")
        sessions.append((s, t))
        out.append({"tokens": len(t), "open_s": round(time.perf_counter() - t0, 1)})
        print(json.dumps(out[-1]), flush=True)
    for s, t in sessions:
        b = len(t) - 1
        L = min(5, 1048576 - b)
        rtts = [s.step(b, [x % 100000 + 1000 for x in t[-L:]])[1]["rtt_s"] * 1e3 for _ in range(10)]
        print(json.dumps({"tokens": len(t), "step_L5_ms": sorted(rtts)[5]}), flush=True)
finally:
    for s, _ in sessions:
        s.close()
