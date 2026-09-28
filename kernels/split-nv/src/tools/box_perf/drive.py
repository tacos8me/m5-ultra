"""Traffic for box profiling / A-B timing against the live step API.

  drive.py dec IDS [--n N] [--reps 30] [--gap 0.025] [--eager DIR] [--widths 1,2,3,4,5] [--tight 30]
      OPEN(state none) on tokens[:N], then per width `reps` STEPs that advance like a verify loop
      (keep = base + a, a in 1..L), `gap` s between STPR and the next STEP; then `tight` back-to-back L=5 steps;
      with --eager, a few eager steps (DIR/eager toggle) for live per-module NVTX.
  drive.py open IDS --n N          one fresh prefill of tokens[:N] via OPEN (state none), then CLOS
  drive.py interf IDS8 IDSL --n N  a session stepping L=5 every 25 ms while another OPEN prefills tokens[:N]
"""
import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/ian/split-nv-moe/tools")
from step_client import Session  # noqa: E402

HOST, PORT = "127.0.0.1", 10052


def load(p):
    t = json.load(open(p))
    return t["tokens"] if isinstance(t, dict) else t


def stepper(s, tokens, L, reps, gap, rng, out, stop=None):
    base = s.length
    last_L = None
    for i in range(reps):
        if stop is not None and stop.is_set():
            break
        keep = base if last_L is None else base + rng.randint(1, last_L)
        ids = [tokens[(keep + j) % len(tokens)] % 100000 + 1000 for j in range(L)]
        r = s.step(keep, ids)
        out.append((L, r["box_s"] * 1e3, r["rtt_s"] * 1e3, time.time()))
        base, last_L = keep, L
        if gap:
            time.sleep(gap)
    return base


def summary(tag, rows):
    by = {}
    for L, b, r, _ in rows:
        by.setdefault(L, []).append((b, r))
    res = {}
    for L, v in sorted(by.items()):
        a = np.array(v[2:] if len(v) > 4 else v)
        res[L] = {"n": len(a), "box_med": round(float(np.median(a[:, 0])), 3), "rtt_med": round(float(np.median(a[:, 1])), 3),
                  "box_p10": round(float(np.percentile(a[:, 0], 10)), 3), "box_p90": round(float(np.percentile(a[:, 0], 90)), 3)}
        print(f"{tag} L={L} n={len(a)} box med {res[L]['box_med']:.3f} ms (p10 {res[L]['box_p10']:.3f} p90 {res[L]['box_p90']:.3f})"
              f" rtt med {res[L]['rtt_med']:.3f} ms", flush=True)
    return res


def dec(a):
    tokens = load(a.ids)
    n = a.n or len(tokens)
    rng = random.Random(0)
    t0 = time.time()
    s = Session(HOST, PORT, tokens[:n], state="none")
    print(f"open {n} tokens: {s.open_seconds:.3f} s", flush=True)
    s.length = n - 1
    rows, tight, eager = [], [], []
    try:
        for L in [int(x) for x in a.widths.split(",")]:
            s.length = stepper(s, tokens, L, a.reps, a.gap, rng, rows)
            # the next width starts from a keep inside the last step
            s.length += 1
        if a.tight:
            s.length = stepper(s, tokens, 5, a.tight, 0.0, rng, tight) + 1
        if a.eager:
            flag = os.path.join(a.eager, "eager")
            open(flag, "w").close()
            try:
                for L in (5, 2):
                    s.length = stepper(s, tokens, L, 4, a.gap, rng, eager) + 1
            finally:
                os.unlink(flag)
    finally:
        s.close()
    res = {"gap": summary(f"gap{int(a.gap * 1e3)}ms", rows)}
    if tight:
        res["tight"] = summary("tight", tight)
    if eager:
        res["eager"] = summary("eager", eager)
    print(json.dumps({"mode": "dec", "n": n, "open_s": s.open_seconds, "wall_s": time.time() - t0, "res": res,
                      "rows": rows, "tight_rows": tight}), flush=True)


def open_(a):
    tokens = load(a.ids)
    n = a.n or len(tokens)
    s = Session(HOST, PORT, tokens[:n], state="none")
    print(json.dumps({"mode": "open", "n": n, "open_s": round(s.open_seconds, 3)}), flush=True)
    s.close()


def interf(a):
    t8, tl = load(a.ids), load(a.ids2)
    s = Session(HOST, PORT, t8[:8193], state="none")
    s.length = 8192
    rows, stop = [], threading.Event()
    th = threading.Thread(target=lambda: stepper(s, t8, 5, 100000, 0.025, random.Random(1), rows, stop))
    th.start()
    time.sleep(0.5)
    t0 = time.time()
    s2 = Session(HOST, PORT, tl[:a.n], state="none")
    t_open = time.time() - t0
    time.sleep(0.3)
    stop.set()
    th.join()
    s2.close()
    s.close()
    during = [r for r in rows if t0 <= r[3] <= t0 + t_open]
    rtts = np.array([r[2] for r in during]) if during else np.zeros(1)
    print(json.dumps({"mode": "interf", "n": a.n, "open_s": round(t_open, 3), "steps_during": len(during),
                      "rtt_med": float(np.median(rtts)), "rtt_max": float(rtts.max()),
                      "rows": rows}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dec", "open", "interf"])
    ap.add_argument("ids")
    ap.add_argument("ids2", nargs="?")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--gap", type=float, default=0.025)
    ap.add_argument("--widths", default="1,2,3,4,5")
    ap.add_argument("--tight", type=int, default=30)
    ap.add_argument("--eager", default="")
    a = ap.parse_args()
    {"dec": dec, "open": open_, "interf": interf}[a.mode](a)
