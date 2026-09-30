"""CPU test of tools/dspark_box/w11_fairness.py against a fake OpenAI streaming server (no GPU, no network).

Checks: the long prompt starts only after both decoders streamed content; per-decoder overlap, coverage and rates are
right for decoders that run through the window and for one that stops early (the 09-30 artifact: the legacy
chunks-per-window-second metric drops while the per-decoder-second rate does not); the kill arm merges dspark=false
into the flags file with every other key kept and restores it; summary keeps only full-coverage runs.
usage: CUDA_VISIBLE_DEVICES= python tools/test_w11_fairness.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "dspark_box", "w11_fairness.py")
TMP = tempfile.mkdtemp(prefix="w11-test-")
FLAGS = os.path.join(TMP, "box-perf-flags.json")
RESULTS = []
SEEN = []  # (kind, flags file content at request time, time)


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {str(detail)[:400]}"), flush=True)


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stop_early = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][0]["content"]
        kind = "L" if body["max_tokens"] == 16 else "d"
        try:
            flags = json.load(open(FLAGS))
        except (OSError, ValueError):
            flags = None
        SEEN.append((kind, flags, time.time(), text[:40]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(s, finish=None):
            d = {"choices": [{"delta": {"content": s}, "finish_reason": finish}]}
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()

        if kind == "L":
            time.sleep(0.6)  # the prefill
            emit("x")
            emit("", "length")
        else:
            n = 25 if (Fake.stop_early and "-d1]" in text) else 120  # d1 stops after ~0.5 s when stop_early
            for i in range(n):
                time.sleep(0.02)
                emit("abcd")
            emit("", "stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def run_arm(arm, url, out, lead):
    env = dict(os.environ, SPLIT_NV_WINDOW="1", SPLIT_NV_DIR=TMP, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, SCRIPT, "run", "--arm", arm, "--url", url, "--out", out, "--lead-s", str(lead),
                        "--decoder-chars", "2000", "--long-chars", "5000"], env=env, capture_output=True, text=True,
                       timeout=60)
    if p.returncode:
        print(p.stdout, p.stderr)
    return p.returncode, json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    out = os.path.join(TMP, "w11.jsonl")
    with open(FLAGS, "w") as f:
        json.dump({"idx_rowsplit": False}, f)

    # refusal outside a window
    p = subprocess.run([sys.executable, SCRIPT, "run", "--arm", "box", "--url", url], capture_output=True, text=True,
                       env=dict(os.environ, SPLIT_NV_DIR=TMP), timeout=30)
    check("refuses without SPLIT_NV_WINDOW=1", p.returncode != 0 and "refusing" in (p.stderr + p.stdout))

    for arm in ("box", "kill", "box", "kill", "box", "kill"):
        SEEN.clear()
        code, rec = run_arm(arm, url, out, 0.3)
        d_first = min(t for k, _, t, _ in SEEN if k == "d")
        L_t = [t for k, _, t, _ in SEEN if k == "L"]
        check(f"{arm}: ran, long prompt after both decoders streamed", code == 0 and L_t and L_t[0] - d_first >= 0.3, SEEN)
        check(f"{arm}: both decoders cover the whole TTFT window", rec and rec["coverage"] >= 0.98
              and abs(rec["long_ttft_s"] - 0.6) < 0.25, rec)
        check(f"{arm}: rates per active decoder-second ~ the fake stream (50 chunks/s, 200 chars/s)",
              rec and 35 <= rec["chunks_per_decoder_s"] <= 55 and 140 <= rec["chars_per_decoder_s"] <= 220, rec)
        fl = [f for k, f, _, _ in SEEN if k == "L"][0]
        want = {"idx_rowsplit": False, "dspark": False} if arm == "kill" else {"idx_rowsplit": False}
        check(f"{arm}: flags during the long prompt {want}", fl == want, fl)
        check(f"{arm}: operator flags restored afterwards", json.load(open(FLAGS)) == {"idx_rowsplit": False})

    # the 09-30 artifact: one decoder stops before the prefill ends -> legacy metric drops, per-decoder-second does not
    Fake.stop_early = True
    SEEN.clear()
    code, rec = run_arm("box", url, out, 0.3)
    Fake.stop_early = False
    check("early stop: coverage < 1, excluded from the summary", code == 0 and rec["coverage"] < 0.9, rec)
    full = [json.loads(line) for line in open(out)][:6]
    leg_full = sum(r["legacy_chunks_per_s"] for r in full) / len(full)
    check("early stop: legacy chunks/s falls while chunks per active decoder-second holds",
          rec["legacy_chunks_per_s"] < 0.8 * leg_full and rec["chunks_per_decoder_s"] >= 35, (rec, leg_full))

    p = subprocess.run([sys.executable, SCRIPT, "summary", "--out", out], capture_output=True, text=True, timeout=30)
    s = json.loads(p.stdout.split("W11 ", 1)[1])
    check("summary: 3 full-coverage runs per arm, the early-stop run excluded, PASS on equal arms",
          p.returncode == 0 and s["box"]["runs"] == 4 and s["box"]["full_coverage"] == 3 and s["kill"]["full_coverage"] == 3
          and s["verdict"] == "PASS", s)
    srv.shutdown()
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED", f"({sum(RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
