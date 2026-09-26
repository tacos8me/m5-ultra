"""Decode stall: session A steps continuously (L=5, rollback each step) while session B prefills a long prompt.

  og_stall.py --ids-a ref/ids-8192.json --ids-b ref/ids-524310.json --nb 524310 [--hold]
Reports A's step round trips before / during / after B's OPEN (median, p99, max) and B's prefill time.
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from og_client import Session  # noqa: E402


def load(path):
    t = json.load(open(path))
    return t["tokens"] if isinstance(t, dict) else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-a", default="/home/ian/split-nv/ref/ids-8192.json")
    ap.add_argument("--na", type=int, default=8193)
    ap.add_argument("--ids-b", required=True)
    ap.add_argument("--nb", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pre-s", type=float, default=3.0)
    ap.add_argument("--post-s", type=float, default=3.0)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    ta, tb = load(args.ids_a)[:args.na], load(args.ids_b)[:args.nb]
    a = Session(args.host)
    a.open(ta, state="none")
    base = len(ta) - 1
    ids = [t % 100000 + 1000 for t in ta[-5:]]
    samples, stop = [], threading.Event()
    marks = {}

    def run_b():
        time.sleep(args.pre_s)
        b = Session(args.host)
        marks["b_start"] = time.perf_counter()
        opts = {"stream": 1} if args.stream else {}
        ack, _, _, info = b.open(tb, state="full" if args.stream else "none", **opts)
        marks["b_end"] = time.perf_counter()
        marks["b_info"] = {k: v for k, v in info.items() if k in ("open_s", "bytes", "first_part_s")}
        time.sleep(args.post_s)
        stop.set()
        b.close()

    th = threading.Thread(target=run_b)
    th.start()
    while not stop.is_set():
        t0 = time.perf_counter()
        _, t = a.step(base, ids)
        samples.append((t0, t["rtt_s"], t["box_s"]))
    th.join()
    a.close()
    s = np.array(samples)
    b0, b1 = marks["b_start"], marks["b_end"]
    out = {"b_tokens": len(tb), "b_open_s": round(b1 - b0, 2), "b_info": marks["b_info"]}
    for name, m in (("before", s[:, 0] < b0), ("during", (s[:, 0] >= b0) & (s[:, 0] < b1)), ("after", s[:, 0] >= b1)):
        r = s[m, 1] * 1e3
        if len(r):
            out[name] = {"steps": int(len(r)), "median_ms": round(float(np.median(r)), 1), "p99_ms": round(float(np.percentile(r, 99)), 1),
                         "max_ms": round(float(r.max()), 1), "tok_s_L5": round(5 * len(r) / max(1e-9, (s[m, 0][-1] - s[m, 0][0]) if len(r) > 1 else 1), 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
