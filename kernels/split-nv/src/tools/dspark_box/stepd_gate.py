"""Box-local STEPD byte gate (GPU window, engine running with SPLIT_NV_DSPARK=1; no Mac needed).

Two sessions of the same prompt on the live engine (127.0.0.1:10052):
- A is granted box drafting: RING of synthetic rows, then STEPD cycles;
- B is plain: the STEP of exactly A's (keep, ids) of every cycle.

For each cycle, A's STPD payload must equal B's STPR payload byte for byte (STEPD-SPEC.md s0: "a STEPD reply's
payload is byte-identical to the STPR payload of STEP(keep, ids)"). A and B then follow the same keep / ids.

Accept is synthetic, since there is no target on the box: each cycle accepts a pseudo-random a in [0, L-1], and the
argmax array is made consistent with it (the box only checks consistency).

The mix covers:
- mode 0 (box drafts), and mode 1 (explicit);
- the V2.1 alias (STEPD under tag STEP);
- the V2.2 plain STEP + taps;
- a mid-run RING re-prime;
- a kill-switch stretch (--kill-switch, writes /dev/shm/split-nv/box-perf-flags.json: window only; the fillers of
  V2.3 are gated like every other step).

It reports the drafted share, reasons, box_s / drafter_ms and the round trip.
--prefill N runs a concurrent N-token OPEN on a third session: STEPD parks under preemption and PF_ACTIVE is set.

usage (window only):
  SPLIT_NV_WINDOW=1 python3 tools/dspark_box/stepd_gate.py --ids /home/ian/split-nv/ref/ids-131072.json --n 131073 \
      --cycles 300 [--prefill 131073] [--kill-switch] [--out FILE.json]
Exit code 0 = PASS (no payload mismatch, no ERR, drafted cycles > 0, every STPD header consistent).
"""
import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import struct
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hooks"))
from split_nv import dspark_wire as W  # noqa: E402

FRAME = struct.Struct("<4sIQ")
STEP_HDR = struct.Struct("<IIH")
STPR_HDR = struct.Struct("<IIHIf")
COSTS = {"pipe": [[524288, [25.6, 27.8, 30.1, 32.8]], [131072, [24.9, 26.9, 29.0, 31.7]], [0, [24.1, 26.1, 28.2, 30.8]]],
         "fused": [[524288, [13.7, 15.8, 17.9, 19.9]], [131072, [13.4, 15.3, 17.2, 19.2]], [0, [13.1, 15.0, 16.9, 18.8]]]}
FLAGS = os.path.join(os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"), "box-perf-flags.json")


def recv_exact(s, n):
    buf = bytearray(n)
    v, got = memoryview(buf), 0
    while got < n:
        k = s.recv_into(v[got:], n - got)
        if not k:
            raise ConnectionError("closed")
        got += k
    return bytes(buf)


class Conn:
    def __init__(self, addr, tokens, dspark):
        self.s = socket.create_connection(addr, timeout=600)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        h = {"proto": 1, "identity": "stepd-gate", "prompt_tokens": len(tokens), "state": "none", "request_id": "gate"}
        if dspark:
            h["dspark"] = {"ver": 1, "costs": COSTS}
        self.send(b"OPEN", json.dumps(h).encode(), struct.pack(f"<{len(tokens)}I", *tokens))
        tag, hh, _ = self.recv()
        self.ack = json.loads(hh)
        if tag != b"ACK ":
            raise RuntimeError(f"OPEN refused: {self.ack}")
        self.sid = self.ack["session"]
        self.s.settimeout(120)

    def send(self, tag, h, p=b""):
        self.s.sendall(FRAME.pack(tag, len(h), len(p)) + h + p)

    def recv(self):
        tag, hl, pl = FRAME.unpack(recv_exact(self.s, FRAME.size))
        return tag, recv_exact(self.s, hl) if hl else b"", recv_exact(self.s, pl) if pl else b""

    def step(self, keep, ids, taps=b""):
        t0 = time.perf_counter()
        self.send(b"STEP", STEP_HDR.pack(self.sid, keep, len(ids)), struct.pack(f"<{len(ids)}I", *ids) + taps)
        tag, h, p = self.recv()
        if tag != b"STPR":
            raise RuntimeError(f"STEP keep={keep}: {tag} {h[:300]}")
        sid, after, L, nb, box_s = STPR_HDR.unpack(h)
        if (sid, after, L, nb, len(p)) != (self.sid, keep + len(ids), len(ids), len(ids) * W.ROW_BYTES, len(ids) * W.ROW_BYTES):
            raise RuntimeError(f"STPR geometry {(sid, after, L, nb, len(p))}")
        return p, box_s, time.perf_counter() - t0

    def stepd(self, keep, anchor, nver, a, mode, dmax, argmax=(), explicit=(), taps=b"", flags=0, tag=W.TAG_STEPD):
        t0 = time.perf_counter()
        hdr, pay = W.encode_stepd(self.sid, keep, anchor, nver, a, mode, dmax, flags, argmax, explicit, taps)
        self.send(tag, hdr, pay)
        rt, h, p = self.recv()
        if rt != W.TAG_STPD:
            raise RuntimeError(f"STEPD keep={keep}: {rt} {h[:300]}")
        r = W.decode_stpd_header(h)
        return r, p, time.perf_counter() - t0

    def ring(self, offset, taps_rows):
        hdr, pay = W.encode_ring(self.sid, offset, b"", taps_rows, 0, len(taps_rows) // W.TAP_ROW_BYTES)
        self.send(W.TAG_RING, hdr, pay)

    def close(self):
        try:
            self.send(b"CLOS", b"{}")
        except OSError:
            pass
        self.s.close()


class Taps:
    """Synthetic committed-row taps (BF16, ~N(0, 0.5)), deterministic per position."""

    def __init__(self, seed):
        self.seed = seed

    def rows(self, lo, hi):
        import numpy as np
        out = []
        for p in range(lo, hi):
            r = np.random.default_rng(self.seed * 1_000_003 + p).standard_normal(W.TAP_DIM).astype("float32") * 0.5
            out.append((r.view("uint32") >> 16).astype("<u2").tobytes())  # truncation to BF16 bits
        return b"".join(out)


def read_flags():
    try:
        with open(FLAGS) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return None


def set_flags(d):
    """Write the flags file as d (other keys the operator set are kept by the caller: see main)."""
    tmp = FLAGS + ".stepd-gate"
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
        set_flags(orig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=131073)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10052)
    ap.add_argument("--prefill", type=int, default=0, help="concurrent OPEN of this many tokens (third session)")
    ap.add_argument("--kill-switch", action="store_true", help="flip {'dspark': false} for a stretch (window only)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if os.environ.get("SPLIT_NV_WINDOW") != "1":
        sys.exit("refusing: set SPLIT_NV_WINDOW=1 inside an announced box window")
    ids = json.load(open(a.ids))
    ids = ids["tokens"] if isinstance(ids, dict) else ids
    tokens = [int(t) for t in ids[:a.n]]
    if len(tokens) < 1100:
        sys.exit("need a prompt of >= 1100 tokens (box drafting starts at 1024)")
    addr = (a.host, a.port)
    rng = random.Random(a.seed)
    taps = Taps(a.seed)
    t0 = time.perf_counter()
    A = Conn(addr, tokens, True)
    B = Conn(addr, tokens, False)
    print(f"opened A={A.sid} B={B.sid} ({len(tokens)} tokens) in {time.perf_counter() - t0:.1f}s; grant {A.ack.get('dspark')}",
          flush=True)
    if "dspark" not in A.ack:
        sys.exit(f"FAIL: no grant ({A.ack.get('dspark_off')})")
    pre = {}
    if a.prefill:
        def bg():
            t = time.perf_counter()
            c = Conn(addr, [int(x) for x in (ids * (1 + a.prefill // len(ids)))[:a.prefill]][::-1], False)
            pre["s"] = time.perf_counter() - t
            c.close()
        th = threading.Thread(target=bg, daemon=True)
    keep = len(tokens) - 1
    anchor = tokens[-1]
    A.ring(keep, taps.rows(keep - W.RING, keep))
    stats = {"cycles": 0, "mismatch": 0, "modes": {}, "reasons": {}, "drafted": 0, "fillers": 0, "pf_active": 0,
             "box_ms": [], "drafter_ms": [], "rtt_ms": [], "step_rtt_ms": [], "digests": []}
    r, p, rtt = A.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    ids_cur = [anchor] + (r["drafts"][:r["depth"]] if r["mode_used"] == W.MODE_BOX else [])
    pb, _, _ = B.step(keep, ids_cur)
    ok = p == pb
    stats["mismatch"] += not ok
    stats["drafted"] += bool(r["flags"] & W.R_DRAFTED)
    kill_lo, kill_hi = (a.cycles // 3, a.cycles // 3 + 10) if a.kill_switch else (-1, -1)
    orig_flags = read_flags()
    started = False
    for c in range(a.cycles):
        if a.prefill and c == 20 and not started:
            th.start()
            started = True
        if c == kill_lo:
            set_flags(dict(orig_flags or {}, dspark=False))  # every other live flag kept
        if c == kill_hi:
            restore_flags(orig_flags)
        L = len(ids_cur)
        acc = rng.randint(0, L - 1)
        new_anchor = rng.randrange(1000, 120000)
        if acc + 1 < L and new_anchor == ids_cur[acc + 1]:
            new_anchor += 1
        argmax = [ids_cur[i + 1] for i in range(acc)] + [new_anchor] + [rng.randrange(1000, 120000) for _ in range(L - acc - 1)]
        keep_new = keep + acc + 1
        tp = taps.rows(keep_new - acc - 1, keep_new)
        kind = rng.random()
        if c == a.cycles // 2:  # a RING re-prime mid-run (the ring content equals what the appends built)
            A.ring(keep, taps.rows(keep - W.RING, keep))
        if kind < 0.70 or kill_lo <= c < kill_hi:
            tag = b"STEP" if kind < 0.05 else W.TAG_STEPD  # V2.1 alias now and then
            r, p, rtt = A.stepd(keep_new, new_anchor, L, acc, W.MODE_BOX, 4, argmax, (), tp, 0, tag)
            ids_next = [new_anchor] + (r["drafts"][:r["depth"]] if r["mode_used"] == W.MODE_BOX else [])
        elif kind < 0.90:
            ex = [rng.randrange(1000, 120000) for _ in range(rng.randint(1, 4))]
            r, p, rtt = A.stepd(keep_new, new_anchor, L, acc, W.MODE_EXPLICIT, 0, argmax, ex, tp)
            ids_next = [new_anchor] + ex
        else:  # V2.2: a plain STEP with taps (no STPD header)
            ex = [rng.randrange(1000, 120000) for _ in range(rng.randint(1, 3))]
            ids_next = [new_anchor] + ex
            p, _, rtt = A.step(keep_new, ids_next, tp)
            r = None
        if r is not None:
            if (r["length_after"], r["L"], r["a_box"]) != (keep_new + len(ids_next), len(ids_next), acc):
                print(f"FAIL cycle {c}: STPD header {r}", flush=True)
                stats["mismatch"] += 1
            stats["modes"][r["mode_used"]] = stats["modes"].get(r["mode_used"], 0) + 1
            stats["reasons"][r["reason"]] = stats["reasons"].get(r["reason"], 0) + 1
            stats["drafted"] += bool(r["flags"] & W.R_DRAFTED)
            stats["fillers"] += r["mode_used"] == W.MODE_BOX and not r["flags"] & W.R_DRAFTED
            stats["pf_active"] += bool(r["flags"] & W.R_PF_ACTIVE)
            stats["box_ms"].append(r["box_s"] * 1e3)
            if r["flags"] & W.R_DRAFTED:
                stats["drafter_ms"].append(r["drafter_ms"])
        pb, _, srtt = B.step(keep_new, ids_next)
        stats["rtt_ms"].append(rtt * 1e3)
        stats["step_rtt_ms"].append(srtt * 1e3)
        if p != pb:
            stats["mismatch"] += 1
            print(f"FAIL cycle {c}: payload differs from STEP(keep={keep_new}, ids={ids_next})", flush=True)
        stats["digests"].append(hashlib.sha256(p).hexdigest()[:16])
        stats["cycles"] += 1
        keep, ids_cur = keep_new, ids_next
    if a.kill_switch:
        restore_flags(orig_flags)
    A.close()
    B.close()
    if started:
        th.join(timeout=600)
    med = (lambda xs: round(statistics.median(xs), 3) if xs else None)
    summary = {"cycles": stats["cycles"], "payload_mismatches": stats["mismatch"], "drafted": stats["drafted"],
               "fillers": stats["fillers"], "modes": stats["modes"], "reasons": stats["reasons"],
               "pf_active_replies": stats["pf_active"], "box_ms_p50": med(stats["box_ms"]),
               "drafter_ms_p50": med(stats["drafter_ms"]), "stepd_rtt_ms_p50": med(stats["rtt_ms"]),
               "step_rtt_ms_p50": med(stats["step_rtt_ms"]), "prefill_s": pre.get("s"), "n": len(tokens)}
    ok = stats["mismatch"] == 0 and stats["drafted"] > 0
    summary["verdict"] = "PASS" if ok else "FAIL"
    print("STEPD-GATE " + json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(summary, digests=stats["digests"]), f)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
