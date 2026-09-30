"""CPU test of DSpark-on-box (STEPD-SPEC.md v1) end to end over TCP, no GPU.

What is real:
- split_nv.front.Front, with the connection handler, OPEN grant and ACK, RING and STEPD handling, run_now/_execute
  and peer broadcast;
- split_nv.engine.Engine.cmd_stepd / cmd_ring / cmd_step, run on TWO engines. Rank 1 gets every command through
  Front.peers, as over the pipe.
- the codec, split_nv.dspark_wire.

What is fake:
- the GPU parts. FakeSteps has StepRunner's keep/rollback/commit semantics, and its payload rows are a deterministic
  function of (committed prefix, position, token). FakeDrafter keeps a 128-slot ring per session, decoded from the
  taps/keys bytes, and drafts with the toy LM from that ring plus the anchor, injecting some wrong drafts.
- the Mac, played by a toy greedy LM (next token = f(last 3 tokens)), which verifies every row.
Exactness: every scenario's generated tokens must equal plain greedy decoding of the toy LM, whatever the drafts,
modes, fallbacks, kill switch, ring gaps, re-primes or restarts.

Scenarios:
- codec round trips and validation;
- OPEN grant / refusals (not asked, not loaded, disabled, no slot, bad version or costs);
- box drafting at >= 1024 (drafted cycles, acceptance, both ranks identical, payload == the STEP function);
- a short prompt crossing 896 -> 1024 (mode 1 with taps, mode 0 refused below 1024);
- a ring wrapping past 128 positions;
- a plain STEP gap -> ring_not_ok -> RING -> drafting again;
- the kill switch mid-stream -> bonus only, DISABLED -> back on + RING;
- sha256-rejected RING;
- NO_TAPS;
- a crash-restart continuation (fresh box: nver>0 -> no_pending ERR; reopen + RING + kickoff nver=0);
- accept mismatch -> ERR + closed;
- pf_active;
- parkable STEPD commands.
usage: CUDA_VISIBLE_DEVICES= python tools/test_dspark_stepd.py
"""
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TMP = tempfile.mkdtemp(prefix="dspark-stepd-test-")
os.environ["SPLIT_NV_DIR"] = TMP  # box-perf-flags.json of this test only
os.environ["SPLIT_NV_SPIN_S"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))
import numpy as np  # noqa: E402

from split_nv import dspark_wire as W  # noqa: E402
from split_nv import engine as EM  # noqa: E402
from split_nv import front as F  # noqa: E402
from split_nv import perf_flags  # noqa: E402

V = 997  # toy vocab
FLAGS = os.path.join(TMP, "box-perf-flags.json")
COSTS = {"pipe": [[524288, [25.6, 27.8, 30.1, 32.8]], [131072, [24.9, 26.9, 29.0, 31.7]], [0, [24.1, 26.1, 28.2, 30.8]]],
         "fused": [[524288, [13.7, 15.8, 17.9, 19.9]], [131072, [13.4, 15.3, 17.2, 19.2]], [0, [13.1, 15.0, 16.9, 18.8]]]}
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {str(detail)[:400]}"), flush=True)
    return bool(cond)


def set_flags(d):
    with open(FLAGS, "w") as f:
        json.dump(d, f)
    perf_flags._state["t"] = -1e9  # re-read now (the module caches for a second)


def lm(prefix):
    """The toy target: greedy next token of a prefix."""
    a, b, c = (prefix[-3:] if len(prefix) >= 3 else [0] * (3 - len(prefix)) + list(prefix))
    return (a * 7 + b * 31 + c * 131 + 17) % V


def greedy(prompt, n):
    out, seq = [], list(prompt)
    for _ in range(n):
        t = lm(seq)
        out.append(t)
        seq.append(t)
    return out


# ------------------------------------------------------------------------------------------------ fake GPU parts
def row_bytes(prefix_digest, pos, tok):
    seed = int.from_bytes(hashlib.sha256(f"{prefix_digest}:{pos}:{tok}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed).bytes(W.ROW_BYTES)


class FakeSession(EM.Session):
    def __init__(self, sid, tokens):
        super().__init__(sid)
        self.hist = list(tokens)  # committed tokens (positions 0..length-1 hold KV)
        self.length = len(tokens)


class FakeSlot:
    def __init__(self, W_):
        self.W, self.graph = W_, (object() if W_ <= 5 else None)


class FakeSteps:
    """StepRunner's host semantics: pick, commit_pending, prepare, run (keep/rollback, pending)."""
    widths = (1, 2, 3, 4, 5, 6, 8)

    def __init__(self, engine):
        self.engine = engine
        self.prepared = 0

    def pick(self, L):
        L = max(2, L)
        for w in self.widths:
            if w >= L:
                return FakeSlot(w)
        raise ValueError(f"L={L}")

    def commit_pending(self, sess, keep):
        if sess.pending is None:
            if keep != sess.length:
                raise ValueError(f"cannot rewind committed prefix {sess.length} to {keep}")
            return
        base, ids = sess.pending
        if keep < base:
            raise ValueError(f"keep {keep} < previous step base {base}")
        sess.pending = None
        a = min(keep - base, len(ids))
        sess.hist = sess.hist[:base] + list(ids[:a])

    def prepare(self, sess, keep, upto):
        self.commit_pending(sess, keep)
        sess.length = keep
        sess.alloc_len = max(sess.alloc_len, upto)
        self.prepared += 1

    def run(self, sess, keep, ids, use_graph=True):
        self.pick(len(ids))
        self.commit_pending(sess, keep)
        sess.length = keep
        if len(sess.hist) != keep:
            raise RuntimeError(f"fake: committed {len(sess.hist)} != keep {keep}")
        dig = hashlib.sha256(np.asarray(sess.hist, dtype="<u4").tobytes()).hexdigest()[:16]
        payload = b"".join(row_bytes(dig, keep + i, t) for i, t in enumerate(ids))
        sess.length = keep + len(ids)
        sess.pending = (keep, list(ids))
        return {"payload": payload, "L": len(ids)}, 0.001


def expected_payload(hist, keep, ids):
    dig = hashlib.sha256(np.asarray(hist[:keep], dtype="<u4").tobytes()).hexdigest()[:16]
    return b"".join(row_bytes(dig, keep + i, t) for i, t in enumerate(ids))


def enc_row(tok, pos, nbytes):
    return struct.pack("<II", tok, pos) + bytes(nbytes - 8)


class FakeDrafter:
    """128-slot ring per session slot holding (position, token) decoded from the taps / key rows; drafts = the toy LM
    over the ring's last 3 positions + anchor, with deterministic wrong drafts; probabilities vary with keep."""

    def __init__(self, nslots=3, strict=True):
        self.nslots, self.strict = nslots, strict
        self.graphs = {"append": object(), 4: object()}
        self.ring = [dict() for _ in range(nslots)]  # slot -> {p % 128: (p, tok)}
        self.pending = None
        self.calls = {"drafts": 0, "appends": 0, "rings": 0}

    def _put(self, slot, pos, tok):
        self.ring[slot][pos % W.RING] = (pos, tok)

    def run_append(self, slot, base, n, taps):
        mv = memoryview(taps)
        for i in range(n):
            tok, pos = struct.unpack_from("<II", mv, i * W.TAP_ROW_BYTES)
            if not self.strict:  # opaque taps (the window gate's synthetic rows): any deterministic token
                tok, pos = tok % V, base + i
            if pos != base + i:
                raise RuntimeError(f"fake drafter: tap row {i} is position {pos}, expected {base + i}")
            self._put(slot, pos, tok)
        self.calls["appends"] += n

    def launch_draft(self, slot, keep, anchor, base, n, taps, W_):
        if n:
            self.run_append(slot, base, n, taps)
        ctx = []
        for p in range(keep - 3, keep):
            e = self.ring[slot].get(p % W.RING)
            ctx.append(e[1] if e and e[0] == p else 0)  # a stale / missing slot drafts from garbage (speed only)
        seq, toks = ctx + [anchor], []
        for i in range(W_):
            t = lm(seq)
            if (keep + i) % 5 == 0 and i >= 1:
                t = (t + 1) % V  # a wrong draft now and then
            toks.append(t)
            seq.append(t)
        q = 0.3 + 0.6 * ((keep * 7) % 10) / 10
        self.pending = (toks, [0.97, 0.9, q, 0.25][:W_])
        self.calls["drafts"] += 1

    def collect(self, W_):
        toks, probs = self.pending
        return toks[:W_], probs[:W_]

    def ring_set(self, slot, offset, keys, kr, taps, tr):
        self.ring[slot] = {}
        mv = memoryview(keys)
        for s in range(W.STAGES):
            for j in range(kr):
                tok, pos = struct.unpack_from("<II", mv, (s * kr + j) * W.KEY_ROW_BYTES)
                if pos != offset - kr - tr + j:
                    raise RuntimeError("fake drafter: key row position")
                if s == 0:
                    self._put(slot, pos, tok)
        if tr:
            self.run_append(slot, offset - tr, tr, taps)
        self.calls["rings"] += 1

    def summary(self):
        return dict(self.calls)


class FakeEngine:
    """Enough of split_nv.engine.Engine for the step API; STEPD / RING / STEP run the real Engine methods."""

    def __init__(self, rank, drafter=True, nslots=3, strict=True):
        self.tp_rank = rank
        self.sessions = {}
        self.vision = False
        self.use_graph = True
        self.steps = FakeSteps(self)
        self.dspark = FakeDrafter(nslots, strict) if drafter else None
        self.job_failures = 0
        self.last_ids = []

    def execute(self, cmd):
        kind = cmd[0]
        if kind == "open":  # test helper: the prefill of an OPEN (state none)
            self.sessions[cmd[1]] = FakeSession(cmd[1], cmd[2])
            return None
        if kind == "close":
            self.sessions.pop(cmd[1], None)
            return None
        if kind == "step":
            return EM.Engine.cmd_step(self, cmd[1], cmd[2], cmd[3])
        if kind == "stepd":
            r = EM.Engine.cmd_stepd(self, *cmd[1:])
            self.last_ids.append(r[2]["ids"])
            return r
        if kind == "ring":
            return EM.Engine.cmd_ring(self, *cmd[1:])
        if kind == "capacity":
            return {"rows": 8, "full": 1 << 30, "swa": 1 << 30, "sessions": len(self.sessions)}
        raise ValueError(kind)


class Peer:
    """Front.peers entry: rank 1 executes each command as it is sent (a pipe with a synchronous reader)."""

    def __init__(self, engine):
        self.engine = engine
        self.errors = []

    def send(self, cmd):
        try:
            self.engine.execute(cmd)
        except Exception as e:  # noqa: BLE001 -- rank 1 logs and counts, as rank_main does
            self.errors.append(repr(e))


def make_front(drafter=True, nslots=3, max_pos=1 << 20, strict=True):
    e0, e1 = FakeEngine(0, drafter, nslots, strict), FakeEngine(1, drafter, nslots, strict)
    f = F.Front.__new__(F.Front)
    f.engine = e0
    f.peers = [Peer(e1)]
    f.jobs = None
    f.job_seq = 0
    f.prefill_lock = threading.RLock()
    f.next_sid, f.sid_lock = 1, threading.Lock()
    f.identity = "t"
    f.draining = False
    f.current = None
    f.open_conns, f.conn_lock = 0, threading.Lock()
    f.max_pos = max_pos
    f.image_token_id = 1 << 30
    f.last_step = {}
    f.gpu_lock, f.step_cv, f.step_waiters = threading.Lock(), threading.Condition(), 0
    f.gate = F.PrefillGate(f.prefill_lock, 0)
    f.pf_need = {}
    f.preempt_on = False
    f.pq_lock, f.pq_open, f.pq = threading.Lock(), False, []
    f.pstats = F.new_pstats()
    f.min_resume = 1024
    f.dsp, f.dsp_off, f.dsp_lock = {}, {}, threading.Lock()
    f.dsp_free = list(range(nslots)) if drafter else []
    f.dsp_stats = {"granted": 0, "refused": {}, "cycles": {"box": 0, "explicit": 0, "bonus": 0, "filler": 0}, "fallbacks": {},
                   "rings": 0, "hard_errors": {}, "drafter_ms": []}
    f.spin_readable = lambda conn, sid: None

    def prefill(sid, tokens, use_cache, **kw):
        f.run_now(("open", sid, list(tokens[:-1])))
        return {}, {"format": "t", "timing": {"prefill_seconds": 0.0}}, {"prefill_s": 0.0, "resumed_tokens": 0}

    f.prefill = prefill
    f.submit = lambda cmd, priority=0: f.run_now(cmd).wait()
    f.submit_async = lambda cmd, priority=0: f.run_now(cmd)
    f.e1 = e1
    return f


def serve(f):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop():
        while True:
            try:
                conn, peer = srv.accept()
            except OSError:
                return
            threading.Thread(target=f.handle_conn, args=(conn, peer), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return srv


# ------------------------------------------------------------------------------------------------ the fake Mac
class BoxError(RuntimeError):
    def __init__(self, h):
        super().__init__(str(h))
        self.h = h


class Mac:
    """The Mac side of STEPD-SPEC.md for one request: toy LM verify, accept, tap ring, modes, re-prime, recovery."""

    def __init__(self, addr, prompt, dspark=True, costs=COSTS, ver=1):
        self.addr, self.prompt = addr, list(prompt)
        self.dspark_req = {"ver": ver, "costs": costs} if dspark else None
        self.sock = None
        self.hist = []  # committed tokens (box KV rows), = prompt[:-1] + ... as the box holds them
        self.out = []  # emitted tokens
        self.pending = None  # (keep, ids) in flight / last sent
        self.ack = None
        self.stats = {"modes": {0: 0, 1: 0, 2: 0}, "used": {0: 0, 1: 0, 2: 0}, "drafted": 0, "accepted": 0, "rings": 0,
                      "reasons": {}, "steps": 0}
        self.flags_seen = []

    # -- frames
    def _send(self, tag, header, payload=b""):
        h = header if isinstance(header, bytes) else json.dumps(header).encode()
        self.sock.sendall(F.FRAME.pack(tag, len(h), len(payload)) + h + payload)

    def _recv(self):
        head = F.recv_exact(self.sock, F.FRAME.size)
        if head is None:
            raise ConnectionError("closed")
        tag, hl, pl = F.FRAME.unpack(head)
        h = F.recv_exact(self.sock, hl) if hl else b""
        p = F.recv_exact(self.sock, pl) if pl else b""
        return tag, h, p

    def open(self, tokens=None):
        tokens = list(self.prompt if tokens is None else tokens)
        self.sock = socket.create_connection(self.addr)
        self.sock.settimeout(30)
        h = {"proto": 1, "identity": "t", "prompt_tokens": len(tokens), "state": "none", "request_id": "x"}
        if self.dspark_req is not None:
            h["dspark"] = self.dspark_req
        self._send(b"OPEN", h, struct.pack(f"<{len(tokens)}I", *tokens))
        tag, hh, _ = self._recv()
        self.ack = json.loads(hh)
        if tag != b"ACK ":
            raise BoxError(self.ack)
        self.sid = self.ack["session"]
        self.hist = tokens[:-1]
        self.pending = None
        return self.ack

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def granted(self):
        return "dspark" in (self.ack or {})

    # -- steps
    def step(self, keep, ids, taps=b""):
        self._send(b"STEP", F.STEP_HDR.pack(self.sid, keep, len(ids)), struct.pack(f"<{len(ids)}I", *ids) + taps)
        tag, h, p = self._recv()
        if tag != b"STPR":
            raise BoxError(json.loads(h))
        self.pending = (keep, list(ids))
        self.stats["steps"] += 1
        return p

    def stepd(self, keep, anchor, nver, a, mode, dmax, argmax=(), explicit=(), taps=b"", flags=0, tag=W.TAG_STEPD):
        hdr, pay = W.encode_stepd(self.sid, keep, anchor, nver, a, mode, dmax, flags, argmax, explicit, taps)
        self._send(tag, hdr, pay)
        tag, h, p = self._recv()
        if tag != W.TAG_STPD:
            raise BoxError(json.loads(h))
        r = W.decode_stpd_header(h)
        if r["session"] != self.sid or r["length_after"] != keep + r["L"] or len(p) != r["L"] * W.ROW_BYTES:
            raise AssertionError(f"STPD geometry {r}")
        if r["mode_used"] == W.MODE_BOX:
            ids = [anchor] + r["drafts"][:r["depth"]]
        elif r["mode_used"] == W.MODE_EXPLICIT:
            ids = [anchor] + list(explicit)
        else:
            ids = [anchor]
        if len(ids) != r["L"]:
            raise AssertionError(f"STPD L {r['L']} vs ids {ids}")
        self.stats["modes"][mode] += 1
        self.stats["used"][r["mode_used"]] += 1
        self.stats["drafted"] += r["mode_used"] == W.MODE_BOX
        self.stats["reasons"][r["reason"]] = self.stats["reasons"].get(r["reason"], 0) + 1
        self.flags_seen.append(r["flags"])
        self.pending = (keep, ids)
        self.stats["steps"] += 1
        self.last = (r, p)
        return r, p

    def ring(self, offset, k_rows, t_rows, bad_digest=False):
        """Re-prime from the Mac's own state: keys (positions [offset-k-t, offset-t)) then taps (the rest)."""
        n = k_rows + t_rows
        keys = b"".join(enc_row(self.hist[p], p, W.KEY_ROW_BYTES) for _ in range(W.STAGES)
                        for p in range(offset - n, offset - t_rows))
        taps = b"".join(enc_row(self.hist[p], p, W.TAP_ROW_BYTES) for p in range(offset - t_rows, offset))
        hdr, pay = W.encode_ring(self.sid, offset, keys, taps, k_rows, t_rows)
        if bad_digest:
            hdr = json.dumps(dict(json.loads(hdr), sha256="0" * 64)).encode()
        self._send(W.TAG_RING, hdr, pay)
        self.stats["rings"] += 1

    # -- verify (the toy target) and accept
    def verify(self):
        """The pending step's verify: argmax per row, accepted count, new committed history, emitted tokens."""
        keep, ids = self.pending
        base_hist = self.hist[:keep]
        argmax = [lm(base_hist + ids[:i + 1]) for i in range(len(ids))]
        a = 0
        while a < len(ids) - 1 and ids[a + 1] == argmax[a]:
            a += 1
        self.hist = base_hist + ids[:a + 1]
        emitted = ids[1:a + 1] + [argmax[a]]
        self.out += emitted
        self.stats["accepted"] += a
        return argmax, a, argmax[a]

    def taps_of(self, keep, a):
        return b"".join(enc_row(self.hist[p], p, W.TAP_ROW_BYTES) for p in range(keep - a - 1, keep))

    def start(self):
        """Today's first two cycles (plain STEP [last prompt token], then STEP [first])."""
        n1 = len(self.prompt) - 1
        self.step(n1, [self.prompt[-1]])
        _, a, first = self.verify()
        self.step(len(self.hist), [first])
        _, a, anchor = self.verify()
        return anchor

    def cycle(self, mode, dmax=4, explicit=None, flags=0, no_taps=False, tag=W.TAG_STEPD):
        """Verify the pending step, then send the next as STEPD (mode) with its taps; returns the STPD."""
        argmax, a, anchor = self.verify()
        keep = len(self.hist)
        nver = len(self.pending[1])
        taps = b"" if no_taps else self.taps_of(keep, a)
        if mode == W.MODE_EXPLICIT and explicit is None:
            explicit = greedy(self.hist + [anchor], 3)
        return self.stepd(keep, anchor, nver, a, mode, dmax, argmax, explicit or (), taps,
                          flags | (W.F_NO_TAPS if no_taps else 0), tag)

    def plain_cycle(self, with_taps):
        """Verify, then a plain STEP [anchor] + Mac drafts (V2.2: optionally with the committed rows' taps)."""
        _, a, anchor = self.verify()
        keep = len(self.hist)
        self.step(keep, [anchor] + greedy(self.hist + [anchor], 2), self.taps_of(keep, a) if with_taps else b"")


# ------------------------------------------------------------------------------------------------ scenarios
def codec_tests():
    hdr, pay = W.encode_stepd(7, 69999, 1234, 3, 1, 0, 4, 0, [500, 1234, 9], (), bytes(2 * W.TAP_ROW_BYTES))
    f = W.decode_stepd(hdr, pay)
    check("codec: STEPD round trip", (f["session"], f["keep"], f["anchor"], f["nver"], f["a"], f["mode"], f["dmax"],
                                      f["argmax"], f["ntaps"], len(f["taps"])) == (7, 69999, 1234, 3, 1, 0, 4, [500, 1234, 9], 2,
                                                                                 2 * W.TAP_ROW_BYTES))
    check("codec: STEPD header vector (SPEC s15)", hdr.hex() == "070000006f110100d2040000030100040000", hdr.hex())
    s = W.encode_stpd_header(7, 69999, 4, 0.00912, 1, 0, W.R_DRAFTED | W.R_RING_OK, 0, [11, 22, 33, 44],
                             [0.9, 0.8, 0.5, 0.25], 1.5)
    check("codec: STPD header vector (SPEC s15)", s.hex() == "07000000731101000400d08502000d6c153c01030005000400000000c03f0b00000016000000210000002c0000006666663fcdcc4c3f0000003f0000803e")
    r = W.decode_stpd_header(s)
    check("codec: STPD round trip", (r["length_after"], r["L"], r["payload_bytes"], r["a_box"], r["depth"], r["drafts"]) ==
          (70003, 4, 4 * W.ROW_BYTES, 1, 3, [11, 22, 33, 44]))
    rh, rp = W.encode_ring(7, 69999, bytes(3 * 2 * 1024), bytes(W.TAP_ROW_BYTES), 2, 1)
    rr = W.decode_ring(json.loads(rh), rp)
    check("codec: RING round trip + digest", rr["digest_ok"] is True and rr["keys_rows"] == 2 and rr["taps_rows"] == 1
          and json.loads(rh)["sha256"] == "1c0273095382988333e2f2b5ae487cea460737ed9be65cbad9c5de537f95bf75")
    bad = {
        "payload length": (W.STEPD_HDR.pack(1, 10, 5, 2, 0, 0, 4, 0, 0), b"\0" * 7),
        "mode 3": (W.STEPD_HDR.pack(1, 10, 5, 0, 0, 3, 4, 0, 0), b""),
        "dmax 5": (W.STEPD_HDR.pack(1, 10, 5, 0, 0, 0, 5, 0, 0), b""),
        "explicit in mode 0": (W.STEPD_HDR.pack(1, 10, 5, 0, 0, 0, 4, 0, 1), b"\0" * 4),
        "a >= nver": (W.STEPD_HDR.pack(1, 10, 5, 2, 2, 0, 4, 0, 0), b"\0" * (8 + 3 * W.TAP_ROW_BYTES)),
        "a with nver 0": (W.STEPD_HDR.pack(1, 10, 5, 0, 1, 0, 4, 0, 0), b""),
        "header size": (W.STEPD_HDR.pack(1, 10, 5, 0, 0, 0, 4, 0, 0) + b"\0", b""),
    }
    for name, (h, p) in bad.items():
        try:
            W.decode_stepd(h, p)
            check(f"codec: rejects {name}", False)
        except W.WireError as e:
            check(f"codec: rejects {name}", e.code == "bad_stepd", e.code)
    try:
        W.decode_stepd(W.STEPD_HDR.pack(1, 1000, 5, 0, 0, 1, 4, 0, 3), b"\0" * 12, max_pos=1002)
        check("codec: context_exceeded", False)
    except W.WireError as e:
        check("codec: context_exceeded", e.code == "context_exceeded")
    for name, h, p in (("ring rows 0", {"session": 1, "offset": 5, "keys_rows": 0, "taps_rows": 0}, b""),
                       ("ring rows 129", {"session": 1, "offset": 500, "keys_rows": 129, "taps_rows": 0}, bytes(3 * 129 * 1024)),
                       ("ring offset < rows", {"session": 1, "offset": 1, "keys_rows": 2, "taps_rows": 0}, bytes(6 * 1024)),
                       ("ring payload", {"session": 1, "offset": 9, "keys_rows": 2, "taps_rows": 0}, bytes(5))):
        try:
            W.decode_ring(h, p)
            check(f"codec: rejects {name}", False)
        except W.WireError as e:
            check(f"codec: rejects {name}", e.code == "bad_ring")
    ok = W.parse_costs(COSTS)
    check("costs: tiers sorted by floor descending", [t[0] for t in ok["pipe"]] == [524288, 131072, 0])
    for name, c in (("no floor 0", {"pipe": [[5, [1, 2, 3, 4]]], "fused": COSTS["fused"]}),
                    ("3 costs", {"pipe": [[0, [1, 2, 3]]], "fused": COSTS["fused"]}),
                    ("negative", {"pipe": [[0, [1, -2, 3, 4]]], "fused": COSTS["fused"]}),
                    ("missing fused", {"pipe": COSTS["pipe"]})):
        try:
            W.parse_costs(c)
            check(f"costs: rejects {name}", False)
        except ValueError:
            check(f"costs: rejects {name}", True)
    C = W.parse_costs(COSTS)
    probs = [0.9, 0.8, 0.5, 0.25]
    check("width: SPEC s15 vector", (W.box_depth(probs, C, 69999, False, 4), W.box_depth(probs, C, 69999, True, 4),
                                     W.box_depth(probs, C, 69999, False, 2)) == (3, 3, 2))
    check("accept: SPEC s15 vector", W.accept_count([42, 500, 777], [500, 1234, 9]) == 1)
    try:
        W.check_accept((69997, [42, 500, 777]), 69999, 9, 3, 1, [500, 1234, 9])
        check("accept: anchor != argmax[a] rejected", False)
    except W.WireError as e:
        check("accept: anchor != argmax[a] rejected", e.code == "accept_mismatch")


def prompt_of(n, seed=3):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, V, n)]


def run_box_drafting(addr, f, n_prompt=1500, cycles=120):
    mac = Mac(addr, prompt_of(n_prompt))
    ack = mac.open()
    check("grant: ACK carries dspark capability", ack.get("dspark") == W.capability(), ack)
    anchor = mac.start()
    keep = len(mac.hist)
    mac.ring(keep, min(W.RING, keep) - 40, 40)  # keys for the older part, taps for the newest 40
    r, p = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    check("kickoff nver=0 after RING drafts on the box", r["mode_used"] == W.MODE_BOX and r["flags"] & W.R_RING_OK
          and r["a_box"] == W.A_BOX_NONE and r["ndraft"] == 4, r)
    check("STPD payload == the STEP function of (keep, ids)", p == expected_payload(mac.hist, keep, mac.pending[1]))
    for _ in range(cycles):
        r, p = mac.cycle(W.MODE_BOX)
        if not p == expected_payload(mac.hist, len(mac.hist), mac.pending[1]):
            check("payload per cycle", False, r)
            break
    mac.verify()
    ref = greedy(mac.prompt, len(mac.out))
    check("box drafting: output == greedy reference", mac.out == ref, (mac.out[:10], ref[:10]))
    check("box drafting: every cycle drafted on the box with the ring current",
          mac.stats["used"][0] == cycles + 1 and all(fl & W.R_RING_OK for fl in mac.flags_seen), mac.stats)
    check("box drafting: acceptance > 1 draft/cycle on average", mac.stats["accepted"] > cycles, mac.stats)
    check("ring wrapped past 128 positions and stayed consistent", len(mac.hist) - keep > W.RING and mac.stats["accepted"] > cycles)
    e0, e1 = f.engine, f.e1
    check("rank 1 ran the same steps with the same drafts and widths", e0.last_ids == e1.last_ids and not f.peers[0].errors,
          f.peers[0].errors[:2])
    s0, s1 = e0.sessions[mac.sid], e1.sessions[mac.sid]
    check("rank 0 / rank 1 session state identical", (s0.hist, s0.length, s0.pending) == (s1.hist, s1.length, s1.pending))
    check("drafted cycles prepared the step during the drafter graph", e0.steps.prepared == mac.stats["used"][0])
    return mac


def run_crossing(addr):
    mac = Mac(addr, prompt_of(880, seed=5))
    mac.open()
    anchor = mac.start()
    # below 896: plain STEP with Mac (explicit) drafts, today's path
    mac.step(len(mac.hist), [anchor] + greedy(mac.hist + [anchor], 3))
    while len(mac.hist) < 896:
        _, a, anchor = mac.verify()
        mac.step(len(mac.hist), [anchor] + greedy(mac.hist + [anchor], 2))
    _, a, anchor = mac.verify()
    keep = len(mac.hist)
    mac.ring(keep, 0, W.RING)
    r, _ = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    check("mode 0 below 1024 -> bonus only, short_context", r["mode_used"] == W.MODE_BONUS
          and r["reason"] == W.REASON["short_context"] and r["flags"] & W.R_RING_OK, r)
    n_mode1 = 0
    while len(mac.hist) < 1024:
        r, _ = mac.cycle(W.MODE_EXPLICIT)
        n_mode1 += 1
        if r["mode_used"] != W.MODE_EXPLICIT or not r["flags"] & W.R_RING_OK:
            check("mode 1 keeps the ring current below 1024", False, r)
            break
    r, _ = mac.cycle(W.MODE_BOX)
    check(f"crossing 1024: mode 1 x{n_mode1} with taps, then box drafting", r["mode_used"] == W.MODE_BOX, r)
    for _ in range(20):
        mac.cycle(W.MODE_BOX)
    mac.verify()
    check("crossing: output == greedy reference", mac.out == greedy(mac.prompt, len(mac.out)))
    return mac


def run_gap_and_switch(addr, f):
    mac = Mac(addr, prompt_of(1300, seed=7))
    mac.open()
    anchor = mac.start()
    keep = len(mac.hist)
    mac.ring(keep, W.RING, 0)
    mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    for _ in range(5):
        mac.cycle(W.MODE_BOX)
    # a plain STEP in between: the committed rows never reach the box ring
    _, a, anchor = mac.verify()
    mac.step(len(mac.hist), [anchor] + greedy(mac.hist + [anchor], 2))
    r, _ = mac.cycle(W.MODE_BOX)
    check("gap (plain STEP, no taps) -> ring_not_ok, filler [anchor, anchor] (V2.3)", r["mode_used"] == W.MODE_BOX and r["L"] == 2 and not r["flags"] & W.R_DRAFTED
          and r["drafts"][0] == mac.pending[1][0] == mac.pending[1][1] and r["reason"] == W.REASON["ring_not_ok"] and not r["flags"] & W.R_RING_OK, r)
    r, _ = mac.cycle(W.MODE_EXPLICIT)
    check("mode 1 while the ring is not ok: explicit, reason 2 as information", r["mode_used"] == W.MODE_EXPLICIT
          and r["reason"] == W.REASON["ring_not_ok"], r)
    _, a, anchor = mac.verify()
    keep = len(mac.hist)
    mac.ring(keep, 30, 98)
    r, _ = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    check("RING re-prime -> drafting again", r["mode_used"] == W.MODE_BOX and r["reason"] == 0, r)
    for _ in range(3):
        mac.cycle(W.MODE_BOX)
    # kill switch mid-stream
    set_flags({"dspark": False})
    r, _ = mac.cycle(W.MODE_BOX)
    check("kill switch: filler, DISABLED, reason disabled", r["mode_used"] == W.MODE_BOX and r["L"] == 2 and not r["flags"] & W.R_DRAFTED
          and r["drafts"][0] == mac.pending[1][0] == mac.pending[1][1] and r["flags"] & W.R_DISABLED and r["reason"] == W.REASON["disabled"], r)
    r, _ = mac.cycle(W.MODE_EXPLICIT)
    check("kill switch: mode 1 still served, taps ignored", r["mode_used"] == W.MODE_EXPLICIT and r["flags"] & W.R_DISABLED
          and not r["flags"] & W.R_RING_OK, r)
    m2 = Mac(mac.addr, prompt_of(1100, seed=8))
    ack = m2.open()
    check("kill switch: new OPEN gets dspark_off disabled", ack.get("dspark_off") == "disabled" and "dspark" not in ack, ack)
    m2.close()
    set_flags({})
    r, _ = mac.cycle(W.MODE_BOX)
    check("switch back on: ring still not ok until a RING (filler)", r["mode_used"] == W.MODE_BOX and r["L"] == 2 and not r["flags"] & W.R_DRAFTED
          and r["drafts"][0] == mac.pending[1][0] == mac.pending[1][1] and r["reason"] == W.REASON["ring_not_ok"] and not r["flags"] & W.R_DISABLED, r)
    _, a, anchor = mac.verify()
    keep = len(mac.hist)
    mac.ring(keep, 0, W.RING, bad_digest=True)
    r, _ = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    check("RING with a bad sha256 -> ring_rejected, filler", r["mode_used"] == W.MODE_BOX and r["L"] == 2 and not r["flags"] & W.R_DRAFTED
          and r["drafts"][0] == mac.pending[1][0] == mac.pending[1][1] and r["reason"] == W.REASON["ring_rejected"], r)
    _, a, anchor = mac.verify()
    keep = len(mac.hist)
    mac.ring(keep, 0, W.RING)
    r, _ = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    check("good RING after a rejected one -> drafting", r["mode_used"] == W.MODE_BOX, r)
    r, _ = mac.cycle(W.MODE_BOX, no_taps=True)
    check("NO_TAPS -> ring gap -> filler", r["mode_used"] == W.MODE_BOX and r["L"] == 2 and not r["flags"] & W.R_DRAFTED
          and r["drafts"][0] == mac.pending[1][0] == mac.pending[1][1] and r["reason"] == W.REASON["ring_not_ok"], r)
    r, _ = mac.cycle(W.MODE_BONUS)
    check("mode 2 bonus only while the ring is not ok", r["mode_used"] == W.MODE_BONUS and r["L"] == 1, r)
    # pf_active: another session's prefill admitted and unfinished
    f.pf_need[10 ** 6] = 5000
    r, _ = mac.cycle(W.MODE_EXPLICIT)
    check("pf_active while another prefill is admitted", bool(r["flags"] & W.R_PF_ACTIVE), r)
    del f.pf_need[10 ** 6]
    r, _ = mac.cycle(W.MODE_EXPLICIT)
    check("pf_active clear afterwards", not r["flags"] & W.R_PF_ACTIVE, r)
    # dmax = 0 in mode 0
    _, a, anchor = mac.verify()
    keep = len(mac.hist)
    mac.ring(keep, 0, W.RING)
    r, _ = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 0)
    check("mode 0 with dmax 0 -> bonus only, dmax_zero", r["mode_used"] == W.MODE_BONUS and r["reason"] == W.REASON["dmax_zero"], r)
    r, _ = mac.cycle(W.MODE_BOX, dmax=1)
    check("dmax 1 caps the box width", r["mode_used"] == W.MODE_BOX and r["depth"] == 1, r)
    # V2.1: a STEPD sent under the tag b"STEP" (18-byte header)
    r, _ = mac.cycle(W.MODE_BOX, tag=b"STEP")
    check("V2.1: STEPD under tag STEP (18-byte header) served as STEPD", r["mode_used"] == W.MODE_BOX
          and r["flags"] & W.R_DRAFTED, r)
    # V2.2: plain STEPs carrying the committed rows' taps keep the ring current (no RING needed afterwards)
    ring_before = mac.stats["rings"]
    for _ in range(3):
        mac.plain_cycle(with_taps=True)
    r, _ = mac.cycle(W.MODE_BOX)
    check("V2.2: plain STEP + taps x3 -> ring current, box drafts without a RING", r["mode_used"] == W.MODE_BOX
          and r["flags"] & W.R_DRAFTED and r["flags"] & W.R_RING_OK and mac.stats["rings"] == ring_before, r)
    mac.plain_cycle(with_taps=False)
    r, _ = mac.cycle(W.MODE_BOX)
    check("V2.2: plain STEP without taps still leaves a gap (filler, ring_not_ok)", r["mode_used"] == W.MODE_BOX
          and not r["flags"] & W.R_DRAFTED and r["reason"] == W.REASON["ring_not_ok"], r)
    mac.plain_cycle(with_taps=True)
    r, _ = mac.cycle(W.MODE_BOX)
    check("V2.2: taps after a gap are ignored until a RING", not r["flags"] & W.R_RING_OK
          and r["reason"] == W.REASON["ring_not_ok"], r)
    mac.verify()
    check("gap/switch/reprime: output == greedy reference", mac.out == greedy(mac.prompt, len(mac.out)))
    return mac


def run_restart(addr_old, f_old):
    """Box crash mid-stream: the Mac rebuilds on a fresh box and continues with identical output."""
    mac = Mac(addr_old, prompt_of(1400, seed=9))
    mac.open()
    anchor = mac.start()
    keep = len(mac.hist)
    mac.ring(keep, W.RING, 0)
    mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    for _ in range(10):
        mac.cycle(W.MODE_BOX)
    # the box dies with a STEPD in flight: its reply never comes; the Mac verified the previous step already
    argmax, a, anchor = mac.verify()
    keep, nver = len(mac.hist), len(mac.pending[1])
    taps = mac.taps_of(keep, a)
    mac.close()
    f2 = make_front()
    srv2 = serve(f2)
    mac.addr = srv2.getsockname()
    ack = mac.open(mac.hist + [anchor])  # pipe_wire.reopen: history[:keep] + [next token]
    check("reopen: fresh box grants again", "dspark" in ack and len(f2.engine.sessions[mac.sid].hist) == keep, ack)
    try:
        mac.stepd(keep, anchor, nver, a, W.MODE_BOX, 4, argmax, (), taps)
        check("fresh session: STEPD nver>0 -> hard ERR no_pending", False)
    except BoxError as e:
        check("fresh session: STEPD nver>0 -> hard ERR no_pending", e.h.get("code") == "no_pending" and e.h.get("retry") is False, e.h)
    mac.close()
    ack = mac.open(mac.hist + [anchor])
    mac.ring(keep, 20, W.RING - 20)  # the Mac's ring: prime keys still in the window + host tap rows
    r, _ = mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    check("restart: RING + kickoff nver=0 at the committed length drafts on the box", r["mode_used"] == W.MODE_BOX, r)
    for _ in range(15):
        mac.cycle(W.MODE_BOX)
    mac.verify()
    check("restart: continuation output == greedy reference", mac.out == greedy(mac.prompt, len(mac.out)))
    srv2.close()


def run_errors(addr, f):
    mac = Mac(addr, prompt_of(1200, seed=11))
    mac.open()
    anchor = mac.start()
    keep = len(mac.hist)
    mac.ring(keep, W.RING, 0)
    mac.stepd(keep, anchor, 0, 0, W.MODE_BOX, 4)
    argmax, a, anchor = mac.verify()
    keep, nver = len(mac.hist), len(mac.pending[1])
    wrong = (a + 1) % nver if nver > 1 else 0
    try:
        mac.stepd(keep - a + wrong, anchor, nver, wrong, W.MODE_BOX, 4, argmax, (), bytes((wrong + 1) * W.TAP_ROW_BYTES))
        check("accept mismatch -> ERR", False)
    except BoxError as e:
        check("accept mismatch -> hard ERR accept_mismatch", e.h.get("code") == "accept_mismatch", e.h)
    time.sleep(0.1)
    check("accept mismatch closed the session and freed its ring slot", mac.sid not in f.engine.sessions
          and mac.sid not in f.dsp)
    mac.close()
    # not asked / refusals
    m = Mac(addr, prompt_of(1100), dspark=False)
    ack = m.open()
    check("OPEN without dspark: ACK has neither key", "dspark" not in ack and "dspark_off" not in ack, ack)
    m.step(len(m.hist), [m.prompt[-1]])
    _, a0, anchor0 = m.verify()
    try:
        m.step(len(m.hist), [anchor0], m.taps_of(len(m.hist), a0))
        check("V2.2: STEP + taps without the grant -> ERR (bad STEP, as today)", False)
    except BoxError as e:
        check("V2.2: STEP + taps without the grant -> ERR (bad STEP, as today)", "bad STEP" in str(e.h.get("error")), e.h)
    m.close()
    m = Mac(addr, prompt_of(1100), dspark=False)
    m.open()
    try:
        m.ring(len(m.hist), 0, 1)
        m.stepd(len(m.hist), 5, 0, 0, W.MODE_BONUS, 0)
        check("RING/STEPD without the grant -> ERR", False)
    except BoxError as e:
        check("RING/STEPD without the grant -> ERR dspark_not_granted", e.h.get("code") == "dspark_not_granted", e.h)
    m.close()
    for name, kw, want in (("version", {"ver": 2}, "version"), ("bad costs", {"costs": {"pipe": []}}, "bad_request")):
        m = Mac(addr, prompt_of(1100), **kw)
        ack = m.open()
        check(f"OPEN with {name} -> dspark_off {want}", ack.get("dspark_off") == want, ack)
        m.close()
    # session id mismatch in a STEPD
    m = Mac(addr, prompt_of(1100))
    m.open()
    hdr, pay = W.encode_stepd(m.sid + 1000, 10, 5, 0, 0, W.MODE_BONUS, 0)
    m._send(W.TAG_STEPD, hdr, pay)
    tag, h, _ = m._recv()
    check("STEPD for another session -> ERR bad_session", tag == b"ERR " and json.loads(h).get("code") == "bad_session", h)
    m.close()


def run_slots_and_not_loaded():
    f = make_front(nslots=1)
    srv = serve(f)
    a = Mac(srv.getsockname(), prompt_of(1100))
    b = Mac(srv.getsockname(), prompt_of(1100))
    ack_a, ack_b = a.open(), b.open()
    check("one slot: first OPEN granted, second dspark_off no_slot", "dspark" in ack_a and ack_b.get("dspark_off") == "no_slot",
          (ack_a, ack_b))
    a.close()
    time.sleep(0.1)
    c = Mac(srv.getsockname(), prompt_of(1100))
    check("slot freed at close is granted again", "dspark" in c.open())
    b.close()
    c.close()
    srv.close()
    f = make_front(drafter=False)
    srv = serve(f)
    m = Mac(srv.getsockname(), prompt_of(1100))
    check("engine without the drafter -> dspark_off not_loaded", m.open().get("dspark_off") == "not_loaded")
    m.close()
    srv.close()


def run_parkable():
    f = make_front()
    D = f.engine.dspark
    f.engine.use_graph = True
    cmd = lambda draft, n_app, dmax, explicit, W_=4: ("stepd", 1, 0, 2000, 5, 1999, n_app, b"", draft, W_, dmax, None, explicit, None)  # noqa: E731
    check("parkable: drafted STEPD with graphs", f._parkable(cmd(True, 1, 4, [])))
    check("parkable: append-only STEPD with the append graph", f._parkable(cmd(False, 2, 0, [1, 2])))
    check("not parkable: explicit width 8 (eager step)", not f._parkable(cmd(False, 0, 0, [1] * 7)))
    D.graphs.pop(4)
    check("not parkable: drafter width without a graph", not f._parkable(cmd(True, 1, 4, [])))
    f.engine.dspark = None
    check("not parkable: drafter gone", not f._parkable(cmd(False, 0, 0, [1])))


def run_gate_client():
    """tools/dspark_box/stepd_gate.py (the window's box-local byte gate) against the fake box: PASS expected."""
    import importlib.util
    f = make_front(nslots=4, strict=False)
    srv = serve(f)
    ids_file = os.path.join(TMP, "gate-ids.json")
    with open(ids_file, "w") as fh:
        json.dump(prompt_of(1500, seed=21), fh)
    spec = importlib.util.spec_from_file_location("stepd_gate", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                            "dspark_box", "stepd_gate.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    set_flags({"idx_rowsplit": False})  # an operator flag the gate must keep
    argv, os.environ["SPLIT_NV_WINDOW"] = sys.argv, "1"
    sys.argv = ["stepd_gate.py", "--ids", ids_file, "--n", "1500", "--cycles", "120", "--port", str(srv.getsockname()[1]),
                "--kill-switch", "--prefill", "3000", "--out", os.path.join(TMP, "gate.json")]
    try:
        G.main()
        code = 0
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = argv
        del os.environ["SPLIT_NV_WINDOW"]
    out = json.load(open(os.path.join(TMP, "gate.json")))
    check("stepd_gate client vs the fake box: PASS", code == 0 and out["payload_mismatches"] == 0 and out["drafted"] > 0, out)
    check("stepd_gate exercised fillers (kill switch) and every mode", out["fillers"] > 0 and len(out["modes"]) >= 2, out)
    check("stepd_gate restored the operator's flags", json.load(open(FLAGS)) == {"idx_rowsplit": False})
    set_flags({})
    srv.close()


def main():
    set_flags({})
    codec_tests()
    f = make_front()
    srv = serve(f)
    addr = srv.getsockname()
    run_box_drafting(addr, f)
    run_crossing(addr)
    run_gap_and_switch(addr, f)
    run_errors(addr, f)
    run_restart(addr, f)
    run_slots_and_not_loaded()
    run_parkable()
    run_gate_client()
    st = f.dspark_summary() if hasattr(f, "dspark_summary") else {}
    check("health summary counts cycles and fallbacks", st.get("cycles", {}).get("box", 0) > 100 and st.get("fallbacks"), st)
    srv.close()
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED", f"({sum(RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
