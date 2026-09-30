"""W11 fairness, measured per active decoder-second (window only; replaces the 2026-09-30 scratch w11.py).

Why a new script: the 09-30 W11 counted the decoders' SSE chunks over the whole TTFT window of the long prompt and
divided by the window length. Its decoders stop on their own after ~900-1500 tokens (finish=stop, even with
max_tokens 4096), and the long prompt arrived at a fixed 10 s. Box drafting makes the decoders ~30% faster, so they had
mostly finished before or early in the prefill: the box arm's "decode chunks/s during the prefill" fell (3.9 / 6.2 vs
9.9 / 15.3) while the rate of the decoders that were still running was the same or higher (8.6 / 9.9 vs 8.0 / 7.9
chunks per active decoder-second), and the long prompt's TTFT followed the decoder overlap, not the arm
(TTFT ~ 6.9 s + 0.095 s per decoder-second of overlap fits all four runs within 0.1 s). See DSPARK-FOLLOWUP-BOX.md.

This script:
- starts the long prompt LEAD_S after both decoders have streamed their first content (not at a fixed time), so the
  decoders are still running during the prefill;
- measures every decoder's own active interval and reports, over the long prompt's TTFT window [send, first content]:
  decoder-seconds of overlap, coverage (1.0 = both decoders ran through the whole window), content characters and SSE
  chunks per active decoder-second (T=0 outputs are identical in both arms, so characters are the same work);
- flips the box kill switch for the "kill" arm itself (merges {"dspark": false} into the live flags file, keeps every
  other key, restores it at exit), so arms interleave without a Mac reload: kill = the Mac drafts (today's schedule);
- `summary` keeps only full-coverage runs and compares the arms.

usage (box, inside an announced window, box engine with SPLIT_NV_DSPARK=1, Mac with DS41_OG_BOX_DRAFT=1):
  export SPLIT_NV_WINDOW=1; O=/mnt/nvme-1/split-nv-ops/dspark-followup/w11.jsonl
  for i in 1 2 3; do python3 tools/dspark_box/w11_fairness.py run --arm box --out $O
                     python3 tools/dspark_box/w11_fairness.py run --arm kill --out $O; done
  python3 tools/dspark_box/w11_fairness.py summary --out $O
Pass: >= 3 full-coverage runs per arm; box decode chars per active decoder-second >= 0.95 x kill; box long TTFT
<= 1.05 x kill (medians).
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = os.path.join(os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"), "box-perf-flags.json")
SRC = os.path.join(HERE, "..", "..", "hooks", "split_nv", "engine.py")


def read_flags():
    try:
        with open(FLAGS) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return None


def write_flags(d):
    tmp = FLAGS + ".w11"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, FLAGS)
    time.sleep(1.2)  # the engine re-reads the flags file at most once a second


def restore_flags(orig):
    if orig is None:
        try:
            os.unlink(FLAGS)
        except FileNotFoundError:
            pass
    else:
        write_flags(orig)


def prompt(src, n_chars, nonce, long_answer):
    ask = ("Explain the following code in exhaustive detail, function by function and line by line where it matters. "
           "Aim for at least 3000 words; do not summarize or stop early.\n" if long_answer else
           "Summarize this code and list its main functions:\n")
    return f"[{nonce}] {ask}" + (src * (n_chars // max(1, len(src)) + 2))[:n_chars]


def stream(url, model, text, max_tokens, out, key, first_evt=None):
    body = json.dumps({"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": text}]}).encode()
    rec = {"t0": time.time(), "chunks": [], "error": None}
    out[key] = rec
    try:
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                try:
                    d = json.loads(line[6:])
                except ValueError:
                    continue
                ch = (d.get("choices") or [{}])[0]
                delta = ch.get("delta", {})
                txt = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if txt:
                    rec["chunks"].append((time.time(), len(txt)))
                    if first_evt is not None and len(rec["chunks"]) == 1:
                        first_evt.set()
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:300]
    rec["t_end"] = time.time()


def window_stats(dec, w0, w1):
    """Per decoder: overlap of its active interval with [w0, w1], chunks and characters inside the window."""
    res = []
    for rec in dec:
        ts = rec["chunks"]
        if not ts:
            res.append({"overlap_s": 0.0, "chunks": 0, "chars": 0, "finish": rec.get("finish")})
            continue
        a, b = ts[0][0], ts[-1][0]
        ov = max(0.0, min(b, w1) - max(a, w0))
        inside = [(t, n) for t, n in ts if w0 <= t <= w1]
        res.append({"overlap_s": round(ov, 3), "chunks": len(inside), "chars": sum(n for _, n in inside),
                    "active": [round(a - w0, 3), round(b - w0, 3)], "finish": rec.get("finish")})
    return res


def pre_rate(dec, t_from, t_to):
    n = sum(1 for rec in dec for t, _ in rec["chunks"] if t_from <= t <= t_to)
    c = sum(k for rec in dec for t, k in rec["chunks"] if t_from <= t <= t_to)
    return round(n / (2 * (t_to - t_from)), 2), round(c / (2 * (t_to - t_from)), 1)


def run(a):
    if os.environ.get("SPLIT_NV_WINDOW") != "1":
        sys.exit("refusing: set SPLIT_NV_WINDOW=1 inside an announced box window")
    src = open(a.src).read()
    orig = read_flags()
    try:
        if a.arm == "kill":
            write_flags(dict(orig or {}, dspark=False))  # every other live flag kept
        elif orig is not None and orig.get("dspark") is False:
            sys.exit("refusing: the live flags file already has dspark=false (restore it first)")
        out, nonce = {}, random.randint(0, 10 ** 9)
        firsts = [threading.Event(), threading.Event()]
        ts = [threading.Thread(target=stream, args=(a.url, a.model, prompt(src, a.decoder_chars, f"{nonce}-d{i}", True),
                                                    a.max_tokens, out, f"d{i}", firsts[i])) for i in range(2)]
        for t in ts:
            t.start()
        for e in firsts:
            if not e.wait(120):
                sys.exit("a decoder produced no content in 120 s")
        time.sleep(a.lead_s)
        tl = threading.Thread(target=stream, args=(a.url, a.model, prompt(src, a.long_chars, f"{nonce}-L", False), 16,
                                                   out, "L"))
        tl.start()
        tl.join()
        for t in ts:
            t.join()
    finally:
        if a.arm == "kill":
            restore_flags(orig)
    L = out["L"]
    if L["error"] or not L["chunks"]:
        sys.exit(f"long prompt failed: {L['error']}")
    w0, w1 = L["t0"], L["chunks"][0][0]
    dec = [out["d0"], out["d1"]]
    per = window_stats(dec, w0, w1)
    dsec = sum(p["overlap_s"] for p in per)
    rec = {"label": a.label or a.arm, "arm": a.arm, "time": round(w0, 1), "long_ttft_s": round(w1 - w0, 3),
           "decoder_seconds": round(dsec, 3), "coverage": round(dsec / (2 * (w1 - w0)), 3),
           "chars_per_decoder_s": round(sum(p["chars"] for p in per) / dsec, 1) if dsec else None,
           "chunks_per_decoder_s": round(sum(p["chunks"] for p in per) / dsec, 2) if dsec else None,
           "legacy_chunks_per_s": round(sum(p["chunks"] for p in per) / (w1 - w0), 1),
           "pre_chunks_chars_per_decoder_s": pre_rate(dec, w0 - a.lead_s, w0),
           "decoders": per, "errors": [r["error"] for r in dec if r["error"]]}
    print(json.dumps(rec))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def summary(a):
    rows = [json.loads(line) for line in open(a.out) if line.strip()]
    res = {}
    for arm in ("box", "kill"):
        rs = [r for r in rows if r["arm"] == arm]
        full = [r for r in rs if r["coverage"] >= a.min_coverage and not r["errors"]]
        med = (lambda k: round(statistics.median([r[k] for r in full]), 3) if full else None)  # noqa: E731
        res[arm] = {"runs": len(rs), "full_coverage": len(full), "long_ttft_s": med("long_ttft_s"),
                    "chars_per_decoder_s": med("chars_per_decoder_s"), "chunks_per_decoder_s": med("chunks_per_decoder_s"),
                    "decoder_seconds": med("decoder_seconds")}
    b, k = res["box"], res["kill"]
    ok = (b["full_coverage"] >= 3 and k["full_coverage"] >= 3 and b["chars_per_decoder_s"] and k["chars_per_decoder_s"]
          and b["chars_per_decoder_s"] >= 0.95 * k["chars_per_decoder_s"] and b["long_ttft_s"] <= 1.05 * k["long_ttft_s"])
    if b["chars_per_decoder_s"] and k["chars_per_decoder_s"]:
        res["decode_ratio_box_over_kill"] = round(b["chars_per_decoder_s"] / k["chars_per_decoder_s"], 3)
        res["ttft_ratio_box_over_kill"] = round(b["long_ttft_s"] / k["long_ttft_s"], 3)
    res["verdict"] = "PASS" if ok else "FAIL"
    print("W11 " + json.dumps(res))
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=("box", "kill"), required=True)
    r.add_argument("--label", default="")
    r.add_argument("--out", default="")
    r.add_argument("--url", default="http://192.168.1.203:8080/v1/chat/completions")
    r.add_argument("--model", default="ds41")
    r.add_argument("--src", default=SRC)
    r.add_argument("--lead-s", type=float, default=2.0, help="long prompt this long after both decoders' first content")
    r.add_argument("--decoder-chars", type=int, default=14000)  # ~3.8K tokens
    r.add_argument("--long-chars", type=int, default=450000)  # ~126K tokens
    r.add_argument("--max-tokens", type=int, default=4096)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    s.add_argument("--min-coverage", type=float, default=0.98)
    a = p.parse_args()
    run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
