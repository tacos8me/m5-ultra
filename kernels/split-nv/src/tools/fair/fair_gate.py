"""Window gates + measurements for the fairness prototypes (SPLIT_NV_PREEMPT, SPLIT_NV_BYPASS_TOKENS), against the
live engine. Byte checks reuse og_gate's reference digests.

  fair_gate.py decode --ids ref/ids-131072.json --n 131073 --ref REF [--stream] [--cache] [--decoders 1] [--think 0.022]
      D decoder sessions (8K prompts) each STEP the same rows over and over (keep = prompt, so every STPR payload of a
      decoder must be byte-identical to its idle one) while the og_gate check of the long prompt runs: its state and step
      payloads must equal REF. Reports decode step latency / rate during the prefill and the prefill's open_s.
  fair_gate.py mixed --ids ref/ids-131072.json --n 131073 --short-ids ref/ids-8192.json --short-n 8193 --ref REF [--gap 1]
      the long og_gate check, and after `gap` seconds the short one: both byte-identical to REF; reports both open_s.
Exit code 0 = every byte check passed.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from og_client import Session  # noqa: E402
from og_gate import compare, digest_state, load_ids, run  # noqa: E402


def health():
    return json.load(urllib.request.urlopen("http://127.0.0.1:10051/health", timeout=5))


class Decoder(threading.Thread):
    def __init__(self, args, tokens, n, L, think, idle_steps=20):
        super().__init__(daemon=True)
        self.s = Session(args.host, args.port)
        self.s.open(tokens[:n], state="none")
        self.N = n - 1
        self.ids = tokens[self.N:self.N + L]
        self.think = think
        self.idle = [self.step() for _ in range(idle_steps)]
        self.ref = {d for d, _, _ in self.idle}
        self.samples = []
        self.stop = threading.Event()

    def step(self):
        t0 = time.perf_counter()
        payload, _ = self.s.step(self.N, self.ids)
        return hashlib.sha256(payload).hexdigest(), time.perf_counter() - t0, time.monotonic()

    def run(self):
        while not self.stop.is_set():
            self.samples.append(self.step())
            if self.think > 0:
                time.sleep(self.think)

    def close(self):
        self.stop.set()
        self.join()
        self.s.close()


def _decoder_main(host, port, tokens, n, L, think, conn, ready, stop):
    args = argparse.Namespace(host=host, port=port)
    d = Decoder(args, tokens, n, L, think)
    ready.set()
    d.stop = stop
    d.run()
    d.s.close()
    conn.send((d.idle, d.samples))
    conn.close()


class DecoderProc:
    """A Decoder in its own process: the gate's own work in this process (receiving and parsing a ~0.9 GB STAT blob at
    1M: bytes copies of up to 300 MB that hold the GIL for ~130 ms each) must not show up as decode step latency."""

    def __init__(self, args, tokens, n, L, think):
        ctx = multiprocessing.get_context("fork")
        self.ready, self.stop = ctx.Event(), ctx.Event()
        self.rx, tx = ctx.Pipe(duplex=False)
        self.p = ctx.Process(target=_decoder_main, args=(args.host, args.port, tokens[:n + L], n, L, think, tx,
                                                          self.ready, self.stop), daemon=True)
        self.p.start()
        if not self.ready.wait(120):
            raise RuntimeError("decoder process did not open its session")
        self.idle = self.samples = None

    def start(self):
        pass  # already stepping

    def close(self):
        self.stop.set()
        self.idle, self.samples = self.rx.recv()
        self.p.join(30)
        self.ref = {dg for dg, _, _ in self.idle}


def tail(samples, t0, t1):
    """Step latencies (ms) completed while the long prompt was open, with the gaps over 100/200 ms and the five
    longest (latency ms, seconds after the open started, seconds before it returned)."""
    during = [(lat, t) for _, lat, t in samples if t0 + 0.3 < t < t1 - 0.3]
    top = sorted(during, reverse=True)[:5]
    return {"during_over_100ms": sum(lat > 0.1 for lat, _ in during), "during_over_200ms": sum(lat > 0.2 for lat, _ in during),
            "top": [[round(lat * 1e3, 1), round(t - t0, 2), round(t1 - t, 2)] for lat, t in top]}


def check(args, ids_path, n, label, out):
    tokens = load_ids(ids_path)
    opts = {}
    if args.stream:
        opts["stream"] = 1
    if args.cache:
        opts["cache"] = 1
    t0 = time.monotonic()
    ack, tensors, manifest, info, steps = run(args, tokens, n, **opts)
    t1 = time.monotonic()
    ref = json.load(open(os.path.join(args.ref, f"n{n}.json")))
    ok = compare(ref["state"], {k: list(v) for k, v in digest_state(tensors).items()}, ref["steps"],
                 [hashlib.sha256(p).hexdigest() for p in steps], f"{label} n={n} {opts}")
    out[label] = {"pass": ok, "open_s": round(info["open_s"], 3), "t0": t0, "t1": t1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["decode", "mixed"])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--short-ids")
    ap.add_argument("--short-n", type=int, default=8193)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--dec-ids", default="/home/ian/split-nv/ref/ids-8192.json")
    ap.add_argument("--dec-n", type=int, default=8193)
    ap.add_argument("--decoders", type=int, default=1)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--think", type=float, default=0.022, help="Mac time between a reply and the next STEP (c1 ~22 ms)")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10052)
    ap.add_argument("--out")
    ap.add_argument("--inproc", action="store_true", help="decoders as threads of this process (the old measurement)")
    ap.add_argument("--samples", help="append every decoder sample (t - open start, latency, before open end) here")
    args = ap.parse_args()
    h0 = health()
    res = {"mode": args.mode, "n": args.n, "decoders_inproc": args.inproc, "fair_before": h0.get("fair")}
    ok = True
    if args.mode == "decode":
        dec_tokens = load_ids(args.dec_ids)
        make = Decoder if args.inproc else DecoderProc
        decs = [make(args, dec_tokens, args.dec_n - 17 * i, args.L, args.think) for i in range(args.decoders)]
        for d in decs:
            d.start()
        time.sleep(1.0)
        out = {}
        check(args, args.ids, args.n, "long", out)
        time.sleep(0.5)
        for d in decs:
            d.close()
        t0, t1 = out["long"]["t0"], out["long"]["t1"]
        res["long"] = {k: v for k, v in out["long"].items() if k not in ("t0", "t1")}
        ok &= out["long"]["pass"]
        for i, d in enumerate(decs):
            during = [lat for _, lat, t in d.samples if t0 + 0.3 < t < t1 - 0.3]
            same = all(dg in d.ref for dg, _, _ in d.samples) and len(d.ref) == 1
            ok &= same
            res[f"decoder{i}"] = {
                "payload_identical": same, "idle_ms": round(statistics.median(x[1] for x in d.idle) * 1e3, 2),
                "during_n": len(during), "during_steps_per_s": round(len(during) / max(1e-9, t1 - t0 - 0.6), 2),
                "during_median_ms": round(statistics.median(during) * 1e3, 1) if during else None,
                "during_p90_ms": round(sorted(during)[int(0.9 * len(during))] * 1e3, 1) if during else None,
                "during_max_ms": round(max(during) * 1e3, 1) if during else None, **tail(d.samples, t0, t1)}
            if args.samples:
                with open(args.samples, "a") as f:
                    f.write(json.dumps({"n": args.n, "decoder": i, "inproc": args.inproc, "open_s": res["long"]["open_s"],
                                        "samples": [[round(t - t0, 4), round(lat * 1e3, 2)] for _, lat, t in d.samples]}) + "\n")
    else:
        out = {}
        th = threading.Thread(target=check, args=(args, args.ids, args.n, "long", out))
        th.start()
        time.sleep(args.gap)
        check(args, args.short_ids, args.short_n, "short", out)
        th.join()
        for k in ("long", "short"):
            ok &= out[k]["pass"]
            res[k] = {kk: v for kk, v in out[k].items() if kk not in ("t0", "t1")}
        res["short_finished_first"] = out["short"]["t1"] < out["long"]["t1"]
    res["fair_after"] = health().get("fair")
    res["pass"] = bool(ok)
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
