"""Prefix-cache gates beyond the prompt-end resume (live engine; byte-exact state + steps vs a fresh OPEN).

  og_grid_gate.py grid --warm 40001 --diverge 35000 --n 60001
      cache OPEN(tokens[:warm]); then prompt = tokens[:diverge] + tokens[70000:...] (a different continuation):
      it must resume from the 8K grid entry at floor8192(diverge) and equal a fresh OPEN of the same prompt.
  og_grid_gate.py warm --n 30001        clear, then a cache OPEN(tokens[:n]) (before an engine restart)
  og_grid_gate.py persist --n 30001 --m 31001
      after a restart: cache OPEN(tokens[:m]) resumes at n-1 and equals a fresh OPEN.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from og_client import Session, step_script  # noqa: E402

HTTP = "http://127.0.0.1:10050"


def run(prompt, **opts):
    s = Session()
    ack, tensors, _, info = s.open(prompt, **opts)
    steps = [hashlib.sha256(s.step(keep, ids)[0]).hexdigest() for keep, ids in step_script(prompt + FILLER, len(prompt))]
    s.close()
    return ack, {k: hashlib.sha256(v[2]).hexdigest() for k, v in tensors.items()}, steps, info


def compare(label, fresh, cached, expect_resume):
    bad = sorted(k for k in fresh[1] if fresh[1][k] != cached[1].get(k))
    sbad = [i for i, (a, b) in enumerate(zip(fresh[2], cached[2])) if a != b]
    got = cached[0].get("resumed_tokens")
    ok = not bad and not sbad and got == expect_resume
    print(json.dumps({"gate": label, "tensor_mismatch": bad, "step_mismatch": sbad, "resumed_tokens": got,
                      "expected_resume": expect_resume, "cached_open_s": round(cached[3]["open_s"], 3),
                      "fresh_open_s": round(fresh[3]["open_s"], 3), "pass": ok}))
    return ok


def clear():
    urllib.request.urlopen(urllib.request.Request(HTTP + "/v1/cache/clear", b"{}"), timeout=60).read()


def cache():
    return json.load(urllib.request.urlopen(HTTP + "/v1/cache", timeout=10))


FILLER = []


def main():
    global FILLER
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grid", "warm", "persist"])
    ap.add_argument("--ids", default="/home/ian/split-nv/ref/ids-131072.json")
    ap.add_argument("--warm", type=int, default=40001)
    ap.add_argument("--diverge", type=int, default=35000)
    ap.add_argument("--n", type=int, default=60001)
    ap.add_argument("--m", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    t = json.load(open(a.ids))
    t = t["tokens"] if isinstance(t, dict) else t
    FILLER = t[100000:100040]
    opts = {"stream": 1} if a.stream else {}
    if a.mode == "grid":
        clear()
        s = Session()
        s.open(t[:a.warm], state="none", cache=1)
        s.close()
        prompt = t[:a.diverge] + t[70000:70000 + (a.n - a.diverge)]
        cached = run(prompt, cache=1, **opts)
        fresh = run(prompt, **opts)
        ok = compare(f"grid warm={a.warm} diverge={a.diverge} n={a.n}", fresh, cached, a.diverge // 8192 * 8192)
        # the diverged prompt's own snapshot must serve its continuation, and a second divergence reuses shared blocks
        prompt2 = prompt + t[90000:91000]
        ok &= compare("continuation of the diverged prompt", run(prompt2, **opts), run(prompt2, cache=1, **opts), len(prompt) - 1)
        print(json.dumps({"cache": cache()}))
        sys.exit(0 if ok else 1)
    if a.mode == "warm":
        clear()
        s = Session()
        s.open(t[:a.n], state="none", cache=1)
        s.close()
        print(json.dumps({"warmed": a.n, "cache": cache()}))
        return
    prompt = t[:a.m]
    ok = compare(f"persist n={a.n} m={a.m}", run(prompt, **opts), run(prompt, cache=1, **opts), a.n - 1)
    print(json.dumps({"cache": cache()}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
