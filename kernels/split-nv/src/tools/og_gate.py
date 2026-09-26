"""Byte-exact gates for box engine changes, against the live engine (no restart).

  og_gate.py ref   --ids ref/ids-8192.json --n 8193 --out DIR       record OPEN state + step payloads
  og_gate.py check --ids ref/ids-8192.json --n 8193 --ref DIR [--stream] [--cache]
      the same OPEN/steps on the live engine; every state tensor and every STPR payload must be byte-identical.
  og_gate.py resume --ids ... --n N --base P [--stream]
      OPEN(tokens[:P+1]) with cache, CLOS; then OPEN(tokens[:N]) with cache (resumes at P) vs a fresh OPEN(tokens[:N]):
      state tensors and step payloads byte-identical.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from og_client import Session, step_script  # noqa: E402


def load_ids(path):
    t = json.load(open(path))
    return t["tokens"] if isinstance(t, dict) else t


def run(args, tokens, n, **opts):
    s = Session(args.host, args.port)
    ack, tensors, manifest, info = s.open(tokens[:n], **opts)
    steps = []
    for keep, ids in step_script(tokens, n):
        payload, t = s.step(keep, ids)
        steps.append(payload)
    s.close()
    return ack, tensors, manifest, info, steps


def digest_state(tensors):
    return {k: [v[0], v[1], hashlib.sha256(v[2]).hexdigest()] for k, v in sorted(tensors.items())}


def compare(a_state, b_state, a_steps, b_steps, label):
    bad = sorted(k for k in set(a_state) | set(b_state) if a_state.get(k) != b_state.get(k))
    step_bad = [i for i, (x, y) in enumerate(zip(a_steps, b_steps)) if x != y] + (
        ["count"] if len(a_steps) != len(b_steps) else [])
    res = {"gate": label, "tensors": len(a_state), "tensor_mismatch": bad, "steps": len(a_steps), "step_mismatch": step_bad,
           "pass": not bad and not step_bad}
    print(json.dumps(res), flush=True)
    return res["pass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "check", "resume", "prefix"])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--ref")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--nonce", action="store_true", help="resume mode: random prefix tokens instead of clearing the cache")
    ap.add_argument("--http", default="http://127.0.0.1:10050")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10052)
    args = ap.parse_args()
    tokens = load_ids(args.ids)
    opts = {}
    if args.stream:
        opts["stream"] = 1
    if args.mode == "ref":
        ack, tensors, manifest, info, steps = run(args, tokens, args.n)
        os.makedirs(args.out, exist_ok=True)
        tag = f"n{args.n}"
        json.dump({"state": digest_state(tensors), "steps": [hashlib.sha256(p).hexdigest() for p in steps],
                   "info": info, "manifest_identity": manifest.get("identity")},
                  open(os.path.join(args.out, tag + ".json"), "w"))
        print(json.dumps({"ref": tag, "tensors": len(tensors), "bytes": info["bytes"], "open_s": round(info["open_s"], 3)}))
        return
    if args.mode == "check":
        if args.cache:
            opts["cache"] = 1
        ack, tensors, manifest, info, steps = run(args, tokens, args.n, **opts)
        ref = json.load(open(os.path.join(args.ref, f"n{args.n}.json")))
        ok = compare({k: v for k, v in ref["state"].items()}, {k: list(v) for k, v in digest_state(tensors).items()},
                     ref["steps"], [hashlib.sha256(p).hexdigest() for p in steps], f"check n={args.n} {opts}")
        print(json.dumps({"open_s": round(info["open_s"], 3), "first_part_s": info.get("first_part_s"), "ack": ack}))
        sys.exit(0 if ok else 1)
    if args.lean:
        opts["state"] = "lean"
    if args.mode == "prefix":
        # OPEN(tokens[:n]) vs OPEN(tokens[:base]) with base > n: every row of the shorter state equals the longer's
        _, a, _, _ = Session(args.host, args.port).open(tokens[:args.n], **opts)
        _, b, _, _ = Session(args.host, args.port).open(tokens[:args.base], **opts)
        n1, bad = args.n - 1, []
        for L, r in ((2, 2), (8, 2), (14, 2), (20, 1)):
            for slot, w in ((2, 288), (3, 68)):
                k = f"layer.{L}.slot.{slot}"
                if a[k][2] != b[k][2][:(n1 // r) * w]:
                    bad.append(k)
        R = a["tail.hidden"][1][1]
        m1 = args.base - 1
        Rb = b["tail.hidden"][1][1]
        off = (n1 - R) - (m1 - Rb)  # row of a's first tail row inside b's tail
        if off >= 0:
            row = 4 * 5120 * 2
            if a["tail.hidden"][2][max(0, -off) * row:] != b["tail.hidden"][2][off * row:(off + R) * row]:
                bad.append("tail.hidden")
        print(json.dumps({"gate": f"prefix n={args.n} within {args.base}", "mismatch": bad, "pass": not bad}))
        sys.exit(1 if bad else 0)
    # resume: warm the cache with a shorter prompt, then compare resumed vs fresh
    base = args.base
    if args.nonce:
        # fresh keys instead of clearing the live cache: 8 random tokens after BOS shift nothing else
        import numpy as np
        tokens = tokens[:1] + [int(x) for x in np.random.default_rng(int.from_bytes(os.urandom(4), "little")).integers(1000, 100000, 8)] + tokens[1:]
    else:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(args.http + "/v1/cache/clear", b"{}"), timeout=30).read()
    s = Session(args.host, args.port)
    s.open(tokens[:base + 1], state="none", cache=1)
    s.close()
    t0 = time.perf_counter()
    ack_r, tr, _, info_r, steps_r = run(args, tokens, args.n, cache=1, **opts)
    t_r = time.perf_counter() - t0
    ack_f, tf, _, info_f, steps_f = run(args, tokens, args.n, **opts)
    ok = compare({k: list(v) for k, v in digest_state(tf).items()}, {k: list(v) for k, v in digest_state(tr).items()},
                 [hashlib.sha256(p).hexdigest() for p in steps_f], [hashlib.sha256(p).hexdigest() for p in steps_r],
                 f"resume n={args.n} base={base} {opts}")
    print(json.dumps({"resumed_tokens": ack_r.get("resumed_tokens"), "resumed_open_s": round(info_r["open_s"], 3),
                      "fresh_open_s": round(info_f["open_s"], 3), "resumed_total_s": round(t_r, 3)}))
    if ack_r.get("resumed_tokens") != base:
        print(json.dumps({"warning": f"expected resume at {base}, got {ack_r.get('resumed_tokens')}"}))
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
