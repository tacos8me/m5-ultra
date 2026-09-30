"""Rank-0 front end of the split-nv engine: sockets, prefill orchestration, prefix cache, health, drain.

Step API (TCP, STEP_API.md), default 0.0.0.0:10052:
  OPEN {..., "stream": 1}  -> ACK first, then the ds41-encoder-state-v1 tensors as split-wire TENS parts while the
                              prefill runs (layer 2/8/14/20 rows per 8K chunk), PROG per chunk, then MANI + END.
                              Without "stream" the reply is ACK after the prefill + one STAT frame (unchanged).
  OPEN {..., "cache": 1}   -> resume from the longest snapshot whose tokens are a prefix of prompt[:-1], prefill the
                              rest, and save a snapshot at the end of this prompt. Output bytes are those of a
                              fresh prefill. ACK carries "resumed_tokens".
HTTP (default 0.0.0.0:10051, also 127.0.0.1:10050): POST /v1/prefill (phase-2 server.py compatible, optional
"cache": true), GET /health, GET /v1/info, GET /v1/cache.
"""

import contextlib
import json
import os
import queue
import signal
import socket
import struct
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from split_nv import dspark_wire as WIRE
from split_nv import idx_rowsplit, preempt, stallwatch, topk_det
from split_nv.fair import PrefillGate
from split_nv.imagekeys import parse_images, prefix_digest, prompt_keys
from split_nv.perf_flags import flag
from split_nv.prefix_cache import PrefixIndex, _atomic_save, _load, budget_bytes
from split_nv.state_pack import (FORMAT, ENCODER_LAYERS, NUMERICS, SOURCE, RowChunks, TokenMap, assemble_parts, dtype_name,
                                 row_names, serialize_parts, tensor_bytes)

FRAME = struct.Struct("<4sIQ")
STEP_HDR = struct.Struct("<IIH")
STPR_HDR = struct.Struct("<IIHIf")
CHUNK = 8192
GRID = 8192
MAX_STEP_ROWS = 8
STEP_ROW_BYTES = 4 * 5120 * 2 + 4 * 4 + 288 + 68
PREFILL_TOK_S = 17000.0
SPIN_S = float(os.environ.get("SPLIT_NV_SPIN_S", "0.2"))
ADMIN_PEERS = set(os.environ.get("SPLIT_NV_ADMIN_PEERS", "127.0.0.1,10.10.10.2").split(","))
# Frame limits: a JSON header is a few KB; an OPEN payload is 4 B/token (<= 1M tokens) plus image patches.
MAX_HEADER = 4 << 20
MAX_PAYLOAD = int(os.environ.get("SPLIT_NV_MAX_PAYLOAD_MB", "1024")) << 20
# TCP keepalive on step connections: a Mac that vanished without a FIN (link down, power loss, kernel panic) left its
# sessions -- and their KV, up to a 1M context each -- allocated forever (the connection thread blocks in recv with no
# timeout). Probes are answered by the peer's kernel, so a live but idle client is never cut. 0 = off.
KEEPALIVE_S = int(os.environ.get("SPLIT_NV_KEEPALIVE_S", "30"))
# TCP_QUICKACK before every read of a large request frame (STEPD with taps, RING, STEP + taps, OPEN): without it the
# kernel delays the ACKs of a ~100 KB upload until the application reads, and the Mac's send stalls in ~200/400 us
# bursts: 1.05-1.5 ms per 92-154 KB of STEPD taps vs 0.08-0.23 ms with it, 8.9 vs 3.4 ms for a 3.9 MB RING
# (Mac netbench, DSPARK-FOLLOWUP-MAC.md s0). Linux clears the mode again, so it is re-armed before each recv. Frames
# up to QUICKACK_MIN payload bytes (a plain STEP: 4-32 B) are read exactly as before. Transport only: no byte changes.
# Env SPLIT_NV_QUICKACK=0 or the live perf flag {"quickack": false} turns it off.
TCP_QUICKACK = getattr(socket, "TCP_QUICKACK", None)
QUICKACK = os.environ.get("SPLIT_NV_QUICKACK", "1") == "1" and TCP_QUICKACK is not None
QUICKACK_MIN = int(os.environ.get("SPLIT_NV_QUICKACK_MIN", "4096"))


def idx_lowmem_summary():
    from split_nv import idx_lowmem
    return idx_lowmem.summary()


def keepalive(conn):
    if KEEPALIVE_S <= 0:
        return
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, KEEPALIVE_S)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4)
    except (OSError, AttributeError):
        pass


def gpu_memory():
    """Rank-0 CUDA allocator state (MiB): the caching allocator keeps its high-water mark reserved, so `reserved`
    near the card's capacity is expected; `free` is what is left for a larger prefill than any seen so far."""
    try:
        free, total = torch.cuda.mem_get_info()
        return {"allocated": torch.cuda.memory_allocated() >> 20, "reserved": torch.cuda.memory_reserved() >> 20,
                "peak": torch.cuda.max_memory_allocated() >> 20, "free": free >> 20, "total": total >> 20}
    except Exception:  # noqa: BLE001
        return None

log_lock = threading.Lock()


def log(*a):
    with log_lock:
        print("[engine]", *a, flush=True)


def quickack(plen):
    """Whether a request frame with plen payload bytes is read with TCP_QUICKACK (see QUICKACK)."""
    return QUICKACK and plen > QUICKACK_MIN and bool(flag("quickack", True))


def recv_exact(conn, n, quickack=False):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        if quickack:
            try:
                conn.setsockopt(socket.IPPROTO_TCP, TCP_QUICKACK, 1)  # not sticky on Linux: before every read
            except OSError:
                pass  # a dead socket: the read below reports it
        k = conn.recv_into(view[got:], min(n - got, 4 << 20))
        if k == 0:
            return None
        got += k
    return bytes(buf)


def send_frame(conn, tag, header, payload=b""):
    h = json.dumps(header, separators=(",", ":")).encode()
    conn.sendall(FRAME.pack(tag, len(h), len(payload)) + h)
    if len(payload):
        conn.sendall(payload)


def plan_chunks(P, n1, chunk=CHUNK):
    """Prefill chunks for rows [P, n1): the fresh 8K grid (first chunk [P, next multiple of chunk)), with no 1-row
    chunk: a 1-row extend takes M=1 GEMV paths that round differently from every M>=2 chunk and from STEP."""
    b = [P] + list(range((P // chunk + 1) * chunk, n1, chunk)) + [n1]
    ch = [[b[i], b[i + 1]] for i in range(len(b) - 1) if b[i + 1] > b[i]]
    while len(ch) > 1:
        i = next((i for i, (a, e) in enumerate(ch) if e - a == 1), None)
        if i is None:
            break
        j = i + 1 if i + 1 < len(ch) else i - 1
        lo, hi = min(ch[i][0], ch[j][0]), max(ch[i][1], ch[j][1])
        if hi - lo <= chunk:
            ch[min(i, j)] = [lo, hi]
            del ch[max(i, j)]
        elif j > i:
            ch[i][1] += 1
            ch[j][0] += 1
        else:
            ch[j][1] -= 1
            ch[i][0] -= 1
    return [(a, e) for a, e in ch]


def split_piece(a, e, size):
    """Split [a, e) at multiples of `size`, with no 1-row piece (merged into its neighbour)."""
    b = [a] + list(range((a // size + 1) * size, e, size)) + [e]
    ps = [[b[i], b[i + 1]] for i in range(len(b) - 1)]
    if len(ps) > 1 and ps[0][1] - ps[0][0] == 1:
        ps[1][0] = ps[0][0]
        del ps[0]
    if len(ps) > 1 and ps[-1][1] - ps[-1][0] == 1:
        ps[-2][1] = ps[-1][1]
        del ps[-1]
    return [(x, y) for x, y in ps]


class Rows:
    """Mac-packed source rows (layers 2/8/14/20) of one prefill in position order, kept as chunks."""

    def __init__(self):
        self.chunks = {L: [] for L in SOURCE}  # L -> [(first row, ckv, idxk)]
        self.n = {L: 0 for L in SOURCE}

    def __len__(self):
        return sum(self.n.values())

    def add(self, part):
        for L, (ck, ik) in part.items():
            if int(ck.shape[0]):
                self.chunks[L].append((self.n[L], ck, ik))
                self.n[L] += int(ck.shape[0])

    def parts(self):
        """Chunk-wise {L: (ckv, idxk)} dicts in position order (for streaming a resumed prefix)."""
        k = max((len(v) for v in self.chunks.values()), default=0)
        for i in range(k):
            yield {L: (v[i][1], v[i][2]) for L, v in self.chunks.items() if i < len(v)}

    def by_layer(self):
        return {L: ([c[1] for c in v], [c[2] for c in v]) for L, v in self.chunks.items()}

    def slice(self, L, r0, r1):
        ck, ik = [], []
        for first, c, i in self.chunks[L]:
            a, b = max(r0, first), min(r1, first + int(c.shape[0]))
            if a < b:
                ck.append(c[a - first:b - first])
                ik.append(i[a - first:b - first])
        cat = (lambda xs, w: torch.cat(xs) if xs else torch.zeros((0, w), dtype=torch.uint8))
        return cat(ck, 288).contiguous(), cat(ik, 68).contiguous()

    def save(self, path, a, b):
        """Rows of positions [a, b) (row ranges [a // ratio, b // ratio) per layer) to a safetensors file."""
        t = {}
        for L, r in SOURCE.items():
            t[f"ckv.{L}"], t[f"idxk.{L}"] = self.slice(L, a // r, b // r)
        _atomic_save(t, path)

    def load(self, path):
        t = _load(path)
        self.add({L: (t[f"ckv.{L}"], t[f"idxk.{L}"]) for L in SOURCE})


def tree_head(root):
    """Commit the mounted code tree points at now (a detached clone's HEAD); '' if unknown.
    `version` is the commit the engine started from; they differ only if someone moved the tree without a restart."""
    try:
        head = open(os.path.join(root, ".git", "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = os.path.join(root, ".git", ref)
            if os.path.exists(path):
                return open(path).read().strip()[:7]
            for line in open(os.path.join(root, ".git", "packed-refs")):
                if line.strip().endswith(ref):
                    return line.split()[0][:7]
            return ""
        return head[:7]
    except OSError:
        return ""


def hard_exit(code):
    """os._exit(code) from rank 0, after asking engine.main (SIGUSR2) to SIGKILL the other ranks right now, so their
    teardown (GPU context, ~190 GiB of pinned Engram mappings each) runs alongside this one instead of after it."""
    if os.environ.get("SPLIT_NV_FAST_EXIT", "1") == "1":
        try:
            os.kill(os.getppid(), signal.SIGUSR2)
        except OSError:
            pass
    os._exit(code)


def fatal_cuda_error(e):
    text = str(e)
    return any(m in text for m in ("device-side assert", "illegal memory access", "CUDA error: an illegal",
                                   "unspecified launch failure", "cudaErrorAssert", "CUBLAS_STATUS_EXECUTION_FAILED"))


class Busy(RuntimeError):
    """Retryable refusal (capacity, draining)."""


class DsSession:
    """Rank 0's state of a session granted box drafting (STEPD-SPEC.md): its ring slot and cost tables, a mirror of
    the engine's pending step and length (the accept check runs here, before any GPU work), and the ring bookkeeping
    every STEPD decision is made from. The decisions travel in the job command, so every rank executes the same."""
    __slots__ = ("slot", "costs", "length", "pending", "ring_ok", "ring_offset", "ring_reason")

    def __init__(self, slot, costs, length):
        self.slot, self.costs, self.length = slot, costs, length
        self.pending = None  # (base, ids) of the last STEP/STEPD
        self.ring_ok, self.ring_offset = False, 0
        self.ring_reason = WIRE.REASON["ring_not_ok"]  # why ring_ok is 0: ring_not_ok or ring_rejected


def new_pstats():
    """Preemption counters (/health fair.preempt). park_*: how long parked STEPs waited for a point or the chunk end;
    share_denied: points that had parked STEPs but no decode share left; timing: rank 0's per-chunk maxima (engine);
    by_kind: per job kind (step, stepd (Mac drafts / append only), stepd_draft (box drafts), ring) the jobs run at a
    point (inline) or after their chunk, and their summed / longest park, so a window can compare STEPD with STEP."""
    return {"chunks": 0, "inline_steps": 0, "after_chunk_steps": 0, "share_denied": 0, "park_max_ms": 0.0,
            "park_over_100": 0, "park_over_200": 0, "by_kind": {}}


def job_kind(cmd):
    if cmd[0] == "stepd":
        return "stepd_draft" if cmd[8] else "stepd"
    return cmd[0]


class Job:
    __slots__ = ("cmd", "done", "result", "error", "t_get", "t_bcast", "t_exec", "t_park")

    def __init__(self, cmd):
        self.cmd = cmd
        self.done = threading.Event()
        self.result = self.error = None
        self.t_get = self.t_bcast = self.t_exec = self.t_park = None

    def wait(self):
        self.done.wait()
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.result


class Streamer:
    """Sends a state as split-wire TENS parts; row tensors grow chunk by chunk."""

    def __init__(self, conn, n1, lean=False, delta_from=0):
        self.conn = conn
        self.rows = row_names(n1, lean, delta_from)
        self.sent = {name: 0 for name in self.rows}  # rows sent, counted from the tensor's first row
        self.pos = {name: 0 for name in self.rows}  # absolute rows seen
        self.bytes = 0

    def part(self, name, dtype, shape, offset, data):
        data = memoryview(data).cast("B")
        send_frame(self.conn, b"TENS", {"name": name, "dtype": dtype, "shape": list(shape), "offset": offset, "nbytes": len(data)},
                   data)
        self.bytes += len(data)

    def rows_chunk(self, rows):
        for L, (ckv, idxk) in sorted(rows.items()):
            for slot, t in ((2, ckv), (3, idxk)):
                name = f"layer.{L}.slot.{slot}"
                if name not in self.rows:
                    continue
                _, _, _, first, total, width = self.rows[name]
                n = int(t.shape[0])
                skip = max(0, first - self.pos[name])
                self.pos[name] += n
                if n <= skip:
                    continue
                self.part(name, "U8", (1, total, width), self.sent[name] * width, t[skip:].contiguous().numpy())
                self.sent[name] += n - skip

    def finish(self, arrays, manifest):
        for name, t in arrays.items():
            if isinstance(t, RowChunks):
                if self.sent[name] != t.rows:
                    raise RuntimeError(f"stream: {name} sent {self.sent[name]} of {t.rows} rows")
                continue
            data = tensor_bytes(t)
            self.part(name, dtype_name(t), t.shape, 0, data)
        send_frame(self.conn, b"MANI", manifest)
        send_frame(self.conn, b"END ", {"tensors": len(arrays), "bytes": manifest["bytes"],
                                        "prefill_s": manifest["timing"]["prefill_seconds"]})


class Front:
    def __init__(self, engine, args, cache_loader=None):
        self.engine = engine
        self.args = args
        self.jobs = queue.PriorityQueue()
        self.job_seq = 0
        # Re-entrant: a cache=1 OPEN holds it through its reply and snapshot, so an OPEN right behind it can resume.
        self.prefill_lock = threading.RLock()
        self.next_sid = 1
        self.sid_lock = threading.Lock()
        self.identity = os.environ.get("SPLIT_NV_IDENTITY", "split-nv:sglang-757e8f35+hooks:fp8-original:enc0-20:consistent-v1")
        text = json.loads(open(os.environ["SPLIT_NV_CONFIG"]).read())["text_config"]
        self.token_map = TokenMap(os.path.dirname(os.environ["SPLIT_NV_CONFIG"]), text["engram_compressed_vocab_size"])
        if cache_loader is not None:  # built during the model load (prefix_cache.IndexLoader)
            self.cache, waited = cache_loader.get()
            log(f"prefix cache: {self.cache.summary()} (index built in {cache_loader.seconds:.1f}s during the model "
                f"load, waited {waited:.1f}s)")
        else:
            self.cache = PrefixIndex(NUMERICS, budget_bytes())
            log(f"prefix cache: {self.cache.summary()}")
        self.min_resume = int(os.environ.get("SPLIT_NV_CACHE_MIN_RESUME", "1024"))
        self.draining = False
        self.drain_s = float(os.environ.get("SPLIT_NV_DRAIN_S", "30"))
        self.job_timeout = float(os.environ.get("SPLIT_NV_JOB_TIMEOUT_S", "300"))
        self.current = None  # (cmd kind, start time) of the running GPU job
        self.open_conns = 0
        self.conn_lock = threading.Lock()
        self.step_port = int(os.environ.get("SPLIT_NV_STEP_PORT", "10052"))
        self.share_chunk = int(os.environ.get("SPLIT_NV_SHARE_CHUNK", "2048"))
        self.max_pos = int(engine.mr.model_config.context_len)
        self.image_token_id = engine.image_token_id
        self.last_step = {}  # sid -> monotonic time of its last STEP
        self.t_start = time.time()
        self.version = os.environ.get("SPLIT_NV_VERSION", "")
        self.peers = []
        self.gpu_lock = threading.Lock()  # one GPU job at a time: gpu_loop (queued jobs) or run_now (STEPs)
        self.step_cv = threading.Condition()
        self.step_waiters = 0
        # Fairness prototypes (default off): prefill admission gate with a short-prompt bypass (split_nv.fair), STEPs
        # preempting preemptible prefill chunks between layers (split_nv.preempt), cache trim after long prefills.
        self.gate = PrefillGate(self.prefill_lock, int(os.environ.get("SPLIT_NV_BYPASS_TOKENS", "0") or 0),
                                float(os.environ.get("SPLIT_NV_BYPASS_SHARE", "0.5")))
        self.pf_need = {}  # sid -> KV tokens an unfinished prefill may still allocate (admission of interleaved prefills)
        self.preempt_on = preempt.enabled()
        self.pq_lock = threading.Lock()
        self.pq_open = False  # a preemptible chunk is running: STEPs park here and run between its layers
        self.pq = []
        self.pstats = new_pstats()
        self.rank_counts = None  # [[snapshot write errors, failed commands] per rank] as of the last barrier
        self.rank_counts_t = None
        self.trim_min = int(os.environ.get("SPLIT_NV_TRIM_MIN_TOKENS", "0") or 0)
        engine.preempt_take = self._take_parked
        # DSpark on the box (SPLIT_NV_DSPARK=1): granted sessions, ACK refusals, free ring slots, counters
        self.dsp = {}
        self.dsp_off = {}
        self.dsp_lock = threading.Lock()
        D = getattr(engine, "dspark", None)
        self.dsp_free = list(range(D.nslots)) if D is not None else []
        self.dsp_stats = {"granted": 0, "refused": {}, "cycles": {"box": 0, "explicit": 0, "bonus": 0, "filler": 0}, "fallbacks": {},
                          "rings": 0, "hard_errors": {}, "drafter_ms": []}
        c128 = getattr(engine.steps.be, "online_c128_mtp", None)
        if self.preempt_on and c128 is not None and c128.enabled():
            # its per-forward state is mutated in place by a verify step's metadata init: not restorable by preempt
            log("preempt: online c128 MTP is enabled; preemption stays off")
            self.preempt_on = False

    # ---- GPU job queue ---------------------------------------------------------------------------------------
    def submit_async(self, cmd, priority=0):
        """priority 0 (steps, control) is served before 1 (prefill chunks, cache copies)."""
        job = Job(cmd)
        with self.sid_lock:
            self.job_seq += 1
            seq = self.job_seq
        self.jobs.put((priority, seq, job))
        return job

    def submit(self, cmd, priority=0):
        return self.submit_async(cmd, priority).wait()

    def gpu_loop(self):
        while True:
            _, _, job = self.jobs.get()
            with self.step_cv:
                # a STEP running inline (run_now) goes before any queued job
                while self.step_waiters:
                    self.step_cv.wait()
                self.gpu_lock.acquire()
            try:
                park = job.cmd[0] == "prefill_chunk" and len(job.cmd) > 5 and bool(job.cmd[5])
                if park:
                    with self.pq_lock:
                        self.pq_open = True
                    self.pstats["chunks"] += 1
                try:
                    self._execute(job)
                finally:
                    if park:
                        with self.pq_lock:
                            self.pq_open = False
                            left, self.pq = self.pq, []
                        # STEPs parked too late for a point of this chunk run now, before anything else
                        self.pstats["after_chunk_steps"] += len(left)
                        self._park_waits(left, "after chunk", where_key="after_chunk")
                        for j in left:
                            self._execute(j)
            finally:
                self.gpu_lock.release()

    def _take_parked(self, win):
        """Rank 0's preemption source (split_nv.preempt): the STEPs parked since the last point, if this chunk's
        decode share allows."""
        with self.pq_lock:
            if not self.pq:
                return []
            if not win.share_ok(float(flag("preempt_share", preempt.SHARE))):
                self.pstats["share_denied"] = self.pstats.get("share_denied", 0) + 1
                return []
            jobs, self.pq = self.pq, []
        self.pstats["inline_steps"] += len(jobs)
        self._park_waits(jobs, f"point {win.k - 1}", win)
        return jobs

    def _park_waits(self, jobs, where, win=None, where_key="inline"):
        """Waits of parked STEPs (park -> released to a point or to the chunk's end): max, >100/>200 ms counts, and a
        log line for each wait >= SPLIT_NV_PARK_LOG_MS (150) naming where it was released and the chunk's timings."""
        now = time.perf_counter()
        st = self.pstats
        by = st.setdefault("by_kind", {})
        for j in jobs:
            if j.t_park is None:
                continue
            ms = (now - j.t_park) * 1e3
            k = by.setdefault(job_kind(j.cmd), {"inline": 0, "after_chunk": 0, "park_ms_sum": 0.0, "park_ms_max": 0.0})
            k[where_key] += 1
            k["park_ms_sum"] = round(k["park_ms_sum"] + ms, 1)
            k["park_ms_max"] = max(k["park_ms_max"], round(ms, 1))
            st["park_max_ms"] = max(st.get("park_max_ms", 0.0), round(ms, 1))
            st["park_over_100"] = st.get("park_over_100", 0) + (ms > 100)
            st["park_over_200"] = st.get("park_over_200", 0) + (ms > 200)
            if ms >= float(os.environ.get("SPLIT_NV_PARK_LOG_MS", "150")):
                w = win or getattr(self.engine, "last_window", None)
                extra = "" if w is None else (
                    f", chunk head {1e3 * (w.head_s or 0):.0f} ms, max point gap {1e3 * w.max_gap_s:.0f} ms, max lag wait "
                    f"{1e3 * w.max_lag_s:.0f} ms, share used {w.step_s:.2f}s of {time.perf_counter() - w.t0:.2f}s")
                log(f"preempt: STEP parked {ms:.0f} ms, released {where}{extra}")

    def _parkable(self, cmd):
        if cmd[0] == "ring":
            # ("ring", sid, slot, offset, keys, kr, taps, tr): index writes into the drafter's ring table plus
            # append-graph replays -- no step forward. Parked like a graph STEP, so a RING sent during another
            # session's long prefill (a request crossing 896 or admitted by the bypass, a re-prime after the kill
            # switch or a ring fault, a crash recovery) runs at the next point instead of waiting for the whole chunk
            # (~0.4 s for an 8K chunk) and then holding up the next one. Drafts only: the ring's bytes are the same
            # either way.
            D = self.engine.dspark
            return D is not None and (not cmd[7] or "append" in D.graphs)
        if cmd[0] == "stepd":
            # ("stepd", sid, slot, keep, anchor, app_base, n_app, taps, draft, W, dmax, costs, explicit, expect)
            steps, D = self.engine.steps, self.engine.dspark
            try:
                slot = steps.pick(1 + max(cmd[10], len(cmd[12])))
            except ValueError:
                return False
            if not (steps.engine.use_graph and slot.graph is not None) or D is None:
                return False
            return (cmd[9] in D.graphs) if cmd[8] else (not cmd[6] or "append" in D.graphs)
        if cmd[0] != "step":
            return False
        steps = self.engine.steps
        try:
            slot = steps.pick(len(cmd[3]))
        except ValueError:
            return False
        return steps.engine.use_graph and slot.graph is not None  # graph steps only (no eager forward inside a chunk)

    def run_now(self, cmd):
        """Execute a STEP (or STEPD / RING) on the calling (connection) thread: no hand-off to gpu_loop and back, each
        of which costs a thread wake-up on a cold core. Waits only for the GPU job already running; ahead of queued
        jobs. During a preemptible prefill chunk a parkable job runs at the chunk's next point instead."""
        job = Job(cmd)
        if self.pq_open and self._parkable(cmd):
            with self.pq_lock:
                parked = self.pq_open
                if parked:
                    job.t_park = time.perf_counter()
                    self.pq.append(job)
            if parked:
                job.done.wait()
                return job
        with self.step_cv:
            self.step_waiters += 1
        self.gpu_lock.acquire()
        try:
            with self.step_cv:
                self.step_waiters -= 1
                self.step_cv.notify_all()
            self._execute(job)
        finally:
            self.gpu_lock.release()
        return job

    def _execute(self, job):
        try:
            job.t_get = time.perf_counter()
            self.current = (job.cmd[0], time.monotonic())
            for conn in self.peers:
                conn.send(job.cmd)
            job.t_bcast = time.perf_counter()
            job.result = self.engine.execute(job.cmd)
            job.t_exec = time.perf_counter()
        except Exception as e:  # noqa: BLE001
            job.error = f"{e}\n{traceback.format_exc()}"
            self.engine.job_failures = getattr(self.engine, "job_failures", 0) + 1
            log("GPU job failed:", job.error)
            if fatal_cuda_error(e):
                # A sticky CUDA error (device-side assert, illegal access) poisons the context: every later job
                # would fail while /health still said ok. Exit so the unit restarts the engine.
                log("FATAL CUDA error: exiting for a restart")
                hard_exit(4)
        finally:
            self.current = None
            job.done.set()

    def spin_readable(self, conn, sid):
        """While a single session is decoding (its last STEP less than SPIN_S ago), busy-wait for its next frame
        instead of sleeping in recv: the core stays in C0 at full clock, so the next STEP's host work (~0.5 ms hot)
        does not run ~3x slower after the ~25 ms idle of a c1 cycle. Other sessions (c2+) block as before."""
        t_last = self.last_step.get(sid)
        spin_s = float(flag("spin_s", SPIN_S))
        if t_last is None or spin_s <= 0 or len(self.engine.sessions) != 1:
            return
        deadline = t_last + spin_s
        sessions = self.engine.sessions
        while time.monotonic() < deadline and len(sessions) == 1:
            try:
                conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                return  # data or EOF: recv_exact takes it from here
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                return

    def watchdog(self):
        """A GPU job that never returns means a wedged rank or collective: exit so the supervisor restarts us."""
        while True:
            time.sleep(5)
            cur = self.current
            if cur is not None and time.monotonic() - cur[1] > self.job_timeout:
                log(f"WATCHDOG: GPU job {cur[0]} running {time.monotonic() - cur[1]:.0f}s > {self.job_timeout:.0f}s; exiting")
                hard_exit(3)

    def contended(self, sid):
        """Another session stepped within the last second: it is decoding and waits on every prefill chunk."""
        now = time.monotonic()
        return any(s != sid and now - t < 1.0 for s, t in list(self.last_step.items()))

    def new_sid(self):
        with self.sid_lock:
            sid = self.next_sid
            self.next_sid += 1
            return sid

    # ---- prefill (+ prefix cache) ----------------------------------------------------------------------------
    def prefill(self, sid, tokens, *, use_cache, on_ready=None, on_rows=None, lean=False, delta_from=0, images=(),
                keys=None):
        """Prefill tokens[:-1] into a new session `sid`. Returns (arrays, manifest, info).

        images: imagekeys.parse_images() output; their span rows replace the embeddings at their positions, and the
        cache keys depend on their content (a different image never resumes a cached state).

        on_ready(resumed_tokens) runs once the start point is known; on_rows(rows) for every batch of source rows
        in position order (the resumed prefix first). Holds the prefill lock (the capture state is global).
        With use_cache: resume from the longest cached prefix, save every 8K grid block this prefill completes
        (plus a grid entry) and leave info["blocks"] for the prompt-end snapshot (save_snapshot).
        """
        ids = list(tokens[:-1])
        n1 = len(ids)
        if keys is None:
            keys = prompt_keys(ids, [(st, ln, dg) for st, ln, _, _, dg, _ in images])
        t0 = time.perf_counter()
        info = {"resumed_tokens": 0, "new_blocks": [], "grid_entries": []}
        rows = Rows()
        blocks = []
        entry = None
        with self.gate.enter(None if use_cache else n1):
            if use_cache:
                entry = self.cache.lookup(keys, min_len=min(self.min_resume, n1))
            self.admit(n1, sid)
            self.pf_need[sid] = n1
            if entry is not None:
                try:
                    _, info["restore_s"] = self.submit(("restore", sid, entry.key, entry.blocks, entry.P), priority=1)
                    info["resumed_tokens"] = entry.P
                    blocks = list(entry.blocks)
                    for bid in blocks:
                        rows.load(self.cache.p(f"rows-{bid}"))
                    rows.load(self.cache.p(f"rows-tail-{entry.key}"))
                except Exception as e:  # noqa: BLE001
                    log(f"session {sid}: restore of {entry.key} failed, prefilling from scratch: {e}")
                    self.submit(("close", sid))
                    entry, blocks, rows = None, [], Rows()
                    info["resumed_tokens"] = 0
            if entry is None:
                self.submit(("prefill_begin", sid))
            P = info["resumed_tokens"]
            todo = [(st, vh, vw, data) for st, ln, vh, vw, _, data in images if st + ln > P]
            if todo:
                info["vision_s"] = self.submit(("vision", sid, todo), priority=1)
                info["vision_images"] = len(todo)
            if on_ready:
                on_ready(P)
            plan = plan_chunks(P, n1)
            grids = self._grid_plan(plan, keys, len(blocks)) if use_cache else [None] * len(plan)
            timing = []
            job = None
            pending, next_k = [], [0]
            preemptible = False

            def submit_next():
                # Plan chunks are split into pieces when another session is decoding (re-checked per chunk): a
                # STEP waits for at most one piece. Grid work rides on the last piece of its chunk.
                nonlocal preemptible
                if not pending and next_k[0] < len(plan):
                    k = next_k[0]
                    next_k[0] += 1
                    contended = self.contended(sid)
                    # SPLIT_NV_PREEMPT: whole (overlapped) chunks whose layers STEPs can preempt; otherwise pieces of
                    # share_chunk rows (runtime perf flag share_chunk, 0 = never split)
                    preemptible = contended and self.preempt_on and bool(flag("preempt", True))
                    size = int(flag("share_chunk", self.share_chunk))
                    ps = split_piece(*plan[k], size) if contended and size > 0 and not preemptible else [plan[k]]
                    pending.extend((a, e, k, i == len(ps) - 1) for i, (a, e) in enumerate(ps))
                if not pending:
                    return None
                a, e, k, last = pending.pop(0)
                grid = grids[k][1] if grids[k] and last else None
                from split_nv.engine import pf_split
                cmd = ("prefill_chunk", sid, ids[a:e], grid, pf_split(e - a))
                rowsplit = idx_rowsplit.decide()  # read here, on rank 0 only: every rank gets the same decision
                if preemptible or rowsplit:
                    cmd += (preemptible, rowsplit)
                return self.submit_async(cmd, priority=1), a, e, k, last

            try:
                cur = submit_next()
                job = cur[0] if cur else None
                if on_rows and len(rows):  # the cached prefix's rows go out while the first new chunk computes
                    for part in rows.parts():
                        on_rows(part)
                while cur is not None:
                    job, a, e, k, last = cur
                    part, dt = job.wait()
                    self.pf_need[sid] = n1 - e
                    if not pending:  # between plan chunks, nothing of ours queued: short prompts may go first
                        info["suspended_s"] = info.get("suspended_s", 0.0) + self.gate.prefill_yield()
                    cur = submit_next()
                    job = cur[0] if cur else None
                    timing.append((e - a, dt))
                    rows.add(part)
                    if on_rows:
                        on_rows(part)
                    if last and grids[k] is not None:
                        blocks = self._grid_done(grids[k], keys, e, blocks, rows, info)
                final = self.submit(("prefill_end", sid, use_cache), priority=1)
                if self.trim_min and n1 - P >= self.trim_min and not self.gate.fifo and not self.gate.suspended:
                    # return the long prefill's cached activation segments to the device (asynchronous: the reply
                    # does not wait; the next STEP waits for it like for any GPU job)
                    self.submit_async(("trim",), priority=1)
            except Exception:
                if job is not None:
                    try:
                        job.wait()
                    except Exception:  # noqa: BLE001
                        pass
                self.cache.forget_blocks(info["new_blocks"])
                for gkey, *_ in info["grid_entries"]:
                    for f in self.cache.entry_files(gkey):
                        os.path.exists(f) and os.unlink(f)
                raise
            finally:
                self.pf_need.pop(sid, None)
        info["prefill_s"] = time.perf_counter() - t0
        arrays, manifest = assemble_parts(tokens, rows.by_layer(), final, self.identity, self.token_map,
                                          {"encoder_request_seconds": info["prefill_s"], "chunks": timing,
                                           "resumed_tokens": P}, lean=lean, delta_from=delta_from)
        if images:
            manifest["images"] = [{"start": st, "length": ln, "grid": [vh, vw], "sha256": dg} for st, ln, vh, vw, dg, _ in images]
        info["rows"], info["blocks"], info["keys"] = rows, blocks, keys
        return arrays, manifest, info

    def _grid_plan(self, plan, ids, nblocks):
        """Per chunk: (actions, command) where actions are ("block", bid, g) for each 8K grid block [g - 8192, g) the
        chunk completes (or ("adopt", entry blocks) when an entry already covers g: entries cannot vanish while this
        prefill holds the lock), and command = (blocks to write [(bid, g)], grid entry key or None) for the engine."""
        out = []
        n1 = len(ids)
        for _, e in plan:
            acts, write = [], []
            for g in range((nblocks + 1) * GRID, e + 1, GRID):
                cov = self.cache.find(g, ids[:g])
                if cov is not None:
                    acts.append(("adopt", list(cov.blocks)))
                else:
                    bid = self.cache.new_id("b")
                    acts.append(("block", bid, g))
                    write.append((bid, g))
                nblocks = g // GRID
            key = None
            if e % GRID == 0 and e < n1 and not self.cache.covered(e, ids[:e], capture=False):
                key = self.cache.new_id("g")
            out.append((acts, (write, key)) if acts or key else None)
        return out

    def _grid_done(self, grid, ids, e, blocks, rows, info):
        acts, (_, key) = grid
        for act in acts:
            if act[0] == "adopt":
                blocks = act[1]
                continue
            _, bid, g = act
            if len(blocks) != g // GRID - 1:
                raise RuntimeError(f"grid block order: {len(blocks)} blocks before {g}")
            rows.save(self.cache.p(f"rows-{bid}"), g - GRID, g)
            blocks = blocks + [bid]
            info["new_blocks"].append(bid)
        if key is not None:
            rows.save(self.cache.p(f"rows-tail-{key}"), e, e)
            info["grid_entries"].append((key, e, list(blocks)))
        return blocks

    def barrier(self, priority=1):
        """Engine barrier: every rank has executed every earlier command (its snapshot files are written). Keeps the
        ranks' [write errors, failed commands] for /health and logs new ones of the other ranks."""
        counts = self.submit(("barrier",), priority=priority)
        if counts:
            prev = self.rank_counts or [[0, 0]] * len(counts)
            for r in range(1, len(counts)):
                if counts[r] != prev[r]:
                    log(f"rank {r}: {counts[r][0] - prev[r][0]} new snapshot write error(s), {counts[r][1] - prev[r][1]} "
                        f"new failed command(s) since the last barrier (totals {counts[r]}; details in its log lines)")
            self.rank_counts, self.rank_counts_t = counts, time.monotonic()
        return counts

    def _complete(self, key, blocks, capture):
        """Both ranks' files of entry `key` (and of its unregistered blocks) are whole; else forget it (counted)."""
        bad = self.cache.incomplete(key, blocks, capture)
        if not bad:
            return True
        self.cache.stats["skipped_incomplete"] = self.cache.stats.get("skipped_incomplete", 0) + 1
        log(f"prefix cache: entry {key} not registered, {len(bad)} file(s) missing or incomplete: {bad[:3]}")
        for f in self.cache.entry_files(key):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(f)
        return False

    def save_snapshot(self, sid, ids, info):
        """After a cache=1 prefill (prefill lock held): snapshot the prompt end unless that exact prefix is cached,
        then register it and the prefill's grid entries. Entries become visible only after a barrier: both ranks'
        files are then complete, so an OPEN right behind this one can resume from them. An entry whose files are not
        all there (a failed write on either rank) is skipped and counted, never raised: resumes stay exact."""
        P = len(ids)
        blocks, new_blocks = info["blocks"], list(info["new_blocks"])
        final = len(blocks) == P // GRID and not self.cache.covered(P, ids, capture=True)
        key, dt = None, 0.0
        if final:
            key = self.cache.new_id("e")
            dt = self.submit(("snapshot", sid, key), priority=1)
            info["rows"].save(self.cache.p(f"rows-tail-{key}"), P // GRID * GRID, P)
        self.barrier()
        for gkey, g, gblocks in info["grid_entries"]:
            if self._complete(gkey, gblocks, capture=False):
                self.cache.add(gkey, ids[:g], gblocks, capture=False)
        if final and not self._complete(key, blocks, capture=True):
            final = False
        if final:
            self.cache.add(key, ids, blocks, capture=True)
        self.cache.forget_blocks(new_blocks)  # only those no registered entry references
        if self.cache.total() > self.cache.budget and not self.gate.suspended:
            # after the barrier: rank 1 no longer reads any file. Not while a prefill is suspended for this one (its
            # grid plan may adopt blocks of existing entries): that prefill evicts at its own snapshot.
            self.cache.evict()
        if not final:
            return None
        return {"key": key, "bytes": self.cache.entries[key].bytes if key in self.cache.entries else 0, "snapshot_s": dt}

    def admit(self, n1, sid=None):
        """Refuse (retryable) a prefill that cannot fit, before anything is allocated: an allocation failure half
        way through a prefill is the one path that can leave the pools inconsistent. Unfinished prefills (suspended
        for a short prompt) keep their remaining tokens reserved."""
        cap = self.submit(("capacity",))
        page = 256
        busy = cap["sessions"]
        need_full = (n1 // page + 2) * page + busy * 4 * page + sum(v for k, v in list(self.pf_need.items()) if k != sid)
        need_swa = min(n1, CHUNK) + 3 * page + busy * 2 * page
        if cap["rows"] < 1 or cap["full"] < need_full or cap["swa"] < need_swa:
            raise Busy(f"busy: KV capacity (free rows {cap['rows']}, full {cap['full']} < {need_full} or swa {cap['swa']} < {need_swa})")

    def cache_clear(self):
        with self.gate.enter(None):
            self.barrier()
            return {"dropped": self.cache.clear()}

    # ---- health ------------------------------------------------------------------------------------------------
    def rank_errors(self):
        """Per rank: snapshot write errors and failed commands since start (rank 0 live, the others as of the last
        barrier, which every cache=1 prefill ends with), plus rank 0's last write error."""
        store = getattr(self.engine, "store", None)
        live = [getattr(store, "n_errors", 0), getattr(self.engine, "job_failures", 0)]
        peers = (getattr(self, "rank_counts", None) or [live])[1:]
        t = getattr(self, "rank_counts_t", None)
        out = {"write_errors": [live[0]] + [c[0] for c in peers], "failed_commands": [live[1]] + [c[1] for c in peers],
               "peers_as_of_s": round(time.monotonic() - t, 1) if t else None}
        if getattr(store, "errors", None):
            out["last_write_error"] = store.errors[-1][:300]
        return out

    def health(self):
        cur = self.current
        busy = (time.monotonic() - cur[1]) if cur else 0.0
        step_api = False
        try:
            with socket.create_connection(("127.0.0.1", self.step_port), timeout=1):
                step_api = True
        except OSError:
            pass
        wedged = busy > 60
        ok = step_api and not wedged and not self.draining
        state = "draining" if self.draining else ("wedged" if wedged else "up")
        tree = tree_head("/home/ian/split-nv")
        now = time.monotonic()
        steps = list(self.last_step.values())
        return ok, {"ok": ok, "encoder": state, "step_api": "up" if step_api else "down", "token_map": True,
                    "sessions": len(self.engine.sessions), "connections": self.open_conns,
                    # seconds since the least recently stepped session's last STEP: a large value with sessions open
                    # is a client that holds a session without using it (leak), not load
                    "stalest_step_s": round(now - min(steps), 1) if steps else None,
                    "gpu_mem_mib": gpu_memory(),
                    "gpu_job": cur[0] if cur else None, "gpu_job_s": round(busy, 2), "queued_jobs": self.jobs.qsize(),
                    "cache": self.cache.summary(), "rank_errors": self.rank_errors(), "uptime_s": round(time.time() - self.t_start),
                    "version": self.version, "sglang": os.environ.get("SPLIT_NV_SGLANG_VERSION", ""), "numerics": NUMERICS,
                    "tree_head": tree, "restart_pending": bool(tree) and not self.version.startswith(tree[:len(self.version.split("-")[0])]),
                    "dev_hook": os.environ.get("SPLIT_NV_DEV") == "1", "vision": self.engine.vision,
                    "prefill_perf": {k: os.environ.get(f"SPLIT_NV_{k.upper()}", "0") == "1"
                                     for k in ("pf_overlap", "ce_ar", "q_nocopy")},
                    "topk_det": topk_det.digest() if topk_det.installed() else False,
                    **({"idx_rowsplit": dict(idx_rowsplit.summary(), flag=bool(flag("idx_rowsplit", True)))}
                       if idx_rowsplit.enabled() else {}),
                    **({"stallwatch": stallwatch.summary()} if stallwatch.enabled() else {}),
                    **({"idx_fixed_tiles": dict(idx_lowmem_summary(), flag=bool(flag("idx_fixed_tiles", True)))}
                       if os.environ.get("SPLIT_NV_IDX_FIXED_TILES") == "1" and os.environ.get("SPLIT_NV_IDX_LOWMEM") == "1" else {}),
                    **({"dspark": self.dspark_summary()} if os.environ.get("SPLIT_NV_DSPARK") == "1" else {}),
                    **({"fair": {"gate": self.gate.summary(), "preempt": dict(self.pstats, **getattr(self.engine, "preempt_timing", {})) if self.preempt_on else False,
                                 "trim_min_tokens": self.trim_min}}
                       if self.preempt_on or self.gate.bypass_tokens or self.trim_min else {})}

    # ---- HTTP: /v1/prefill (phase-2 compatible), /health, /v1/info, /v1/cache ------------------------------------
    def serve_http(self, host, port):
        front = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code, obj):
                self._send(code, json.dumps(obj).encode())

            def do_GET(self):
                if self.path.startswith("/debug/mem"):
                    if self.client_address[0] not in ADMIN_PEERS:
                        return self._json(403, {"error": "forbidden"})
                    try:
                        return self._json(200, front.memstats(reset="reset=1" in self.path, snapshot="snapshot=1" in self.path,
                                                              segments="segments=0" not in self.path))
                    except Exception as e:  # noqa: BLE001
                        return self._json(500, {"error": str(e)[:500]})
                if self.path.startswith("/health"):
                    ok, body = front.health()
                    return self._json(200 if ok else 503, body)
                if self.path.startswith("/v1/info"):
                    return self._json(200, {"format": FORMAT, "identity": front.identity, "encoder_layers": ENCODER_LAYERS,
                                            "source_layers": SOURCE, "max_tokens": 1048576, "chunk": CHUNK,
                                            "boundary": "state after prompt tokens[:-1] through layers 0-20",
                                            "step_api": f"tcp :{front.step_port} (STEP_API.md)"})
                if self.path.startswith("/v1/cache"):
                    return self._json(200, front.cache.summary())
                return self._json(404, {"error": "not found"})

            def do_POST(self):
                if self.path.startswith("/admin/"):
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    if self.client_address[0] not in ADMIN_PEERS:
                        return self._json(403, {"error": "forbidden"})
                    if self.path.startswith("/admin/restart"):
                        log(f"admin restart from {self.client_address[0]}")
                        self._json(200, {"ok": True, "action": "drain, exit, restart by the unit"})
                        front.drain_and_exit()
                        return None
                    if self.path.startswith("/admin/crash"):
                        log(f"admin crash from {self.client_address[0]}")
                        self._json(200, {"ok": True, "action": "exit(1) now"})
                        hard_exit(1)
                    return self._json(404, {"error": "not found"})
                if self.path.startswith("/v1/cache/clear"):
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    return self._json(200, front.cache_clear())
                if not self.path.startswith("/v1/prefill"):
                    return self._json(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length", "0"))
                try:
                    body = json.loads(self.rfile.read(n))
                    tokens = [int(t) for t in body["tokens"]]
                    if len(tokens) < 2:
                        raise ValueError("tokens: list of >= 2 ints")
                except Exception as e:  # noqa: BLE001
                    return self._json(400, {"error": str(e)})
                if front.draining:
                    return self._json(503, {"error": "draining", "retry": True})
                identity = body.get("identity") or front.identity
                sid = front.new_sid()
                t0 = time.perf_counter()
                try:
                    with (front.gate.enter(None) if body.get("cache") else contextlib.nullcontext()):
                        arrays, manifest, info = front.prefill(sid, tokens, use_cache=bool(body.get("cache")))
                        manifest["identity"] = identity
                        blob = serialize_parts(arrays, manifest)
                        if body.get("cache"):
                            front.save_snapshot(sid, info["keys"], info)
                except Exception as e:  # noqa: BLE001
                    log(f"http prefill failed: {e}")
                    return self._json(503 if isinstance(e, Busy) else 500, {"error": str(e).splitlines()[0][:500],
                                                                             "retry": isinstance(e, Busy)})
                finally:
                    try:
                        front.submit(("close", sid))
                    except Exception:  # noqa: BLE001
                        pass
                manifest["timing"]["total_seconds"] = time.perf_counter() - t0
                meta = json.dumps({k: v for k, v in manifest.items() if k != "layers"})
                log(f"http prefill {len(tokens)} tokens: {info['prefill_s']:.2f}s resumed {info['resumed_tokens']}, "
                    f"state {len(blob) / 1e6:.1f} MB")
                return self._send(200, blob, "application/octet-stream", {"X-Split-NV-Meta": meta})

        srv = ThreadingHTTPServer((host, port), Handler)
        srv.daemon_threads = True
        log(f"http on {host}:{port}")
        srv.serve_forever()

    def serve_dev(self, port):
        front = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                code = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
                try:
                    # Dev code may prefill through the global capture state: never between another prompt's chunks.
                    with front.gate.enter(None):
                        body = json.dumps({"result": front.submit(("exec", code))}, default=repr)
                    status = 200
                except Exception as e:  # noqa: BLE001
                    body, status = json.dumps({"error": str(e)}), 500
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        log(f"dev exec on 127.0.0.1:{port}")
        srv.serve_forever()

    def memstats(self, reset=False, snapshot=False, segments=True):
        """Allocator statistics of every rank (engine command "memstats"; read-only unless reset: peak counters)."""
        d = os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(d, f"memstats-{stamp}")
        self.submit(("memstats", path, reset, os.path.join(d, f"memsnap-{stamp}") if snapshot else None, segments))
        self.barrier(priority=0)
        out = {}
        for r in range(1 + len(self.peers)):
            f = f"{path}.rank{r}.json"
            with open(f) as fh:
                out[f"rank{r}"] = json.load(fh)
            os.unlink(f)
        return out

    # ---- drain -------------------------------------------------------------------------------------------------
    def drain_and_exit(self, *_):
        if self.draining:
            return
        self.draining = True
        log(f"draining: refusing new OPENs, waiting up to {self.drain_s:.0f}s for {self.open_conns} connection(s)")

        def run():
            deadline = time.monotonic() + self.drain_s
            while self.open_conns and time.monotonic() < deadline:
                time.sleep(0.2)
            log(f"drain done ({self.open_conns} connection(s) left); exiting")
            hard_exit(0)

        threading.Thread(target=run, daemon=True).start()

    # ---- step API (TCP) -----------------------------------------------------------------------------------------
    def serve_tcp(self, host, port):
        self.step_port = port
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(16)
        log(f"step api on {host}:{port}")
        while True:
            conn, peer = srv.accept()
            threading.Thread(target=self.handle_conn, args=(conn, peer), daemon=True).start()

    def handle_open(self, conn, peer, h, tokens, images=()):
        if self.draining:
            send_frame(conn, b"ERR ", {"error": "draining", "retry": True})
            return None
        state = h.get("state", "full")
        if state not in ("full", "lean", "none"):
            raise ValueError(f"bad state {state!r}")
        stream = h.get("stream") in (1, True) and state != "none"
        use_cache = h.get("cache") in (1, True)
        want_state = state != "none"
        lean = state == "lean"
        if images and not self.engine.vision:
            raise ValueError("this engine has no vision tower")
        delta_from = int(h.get("delta_from") or 0)
        if delta_from:
            if not 0 < delta_from <= len(tokens) - 1:
                raise ValueError(f"bad delta_from {delta_from}")
            keys = prompt_keys(tokens[:delta_from], [(st, ln, dg) for st, ln, _, _, dg, _ in images])
            if h.get("prefix_sha256") != prefix_digest(tokens, keys, delta_from, bool(images)):
                raise ValueError("delta_from: prefix_sha256 does not match the prefix (tokens, or keys with images)")
        sid = self.new_sid()
        self._dspark_grant(sid, h, len(tokens) - 1)
        try:
            return self._handle_open(conn, peer, h, tokens, images, sid, state, stream, use_cache, want_state, lean,
                                     delta_from)
        except BaseException:
            self._dspark_release(sid)
            raise

    def _handle_open(self, conn, peer, h, tokens, images, sid, state, stream, use_cache, want_state, lean, delta_from):
        t0 = time.perf_counter()
        streamer = Streamer(conn, len(tokens) - 1, lean, delta_from) if stream else None

        def on_ready(P):
            if stream:
                send_frame(conn, b"ACK ", {"ok": True, "session": sid, "stream": 1, "resumed_tokens": P, "numerics": NUMERICS,
                                           "est_s": round((len(tokens) - 1 - P) / PREFILL_TOK_S, 2), **self._ack_extra(sid)})

        def on_rows(rows):
            if stream:
                streamer.rows_chunk(rows)
                send_frame(conn, b"PROG", {"tokens_done": streamer.sent[f"layer.{ENCODER_LAYERS - 1}.slot.2"]})

        keys, est = None, len(tokens) - 1
        if use_cache and self.gate.bypass_tokens and est > self.gate.bypass_tokens:
            # a long prompt that resumes from the cache may still be short work: estimate from a read-only peek
            keys = prompt_keys(tokens[:-1], [(st, ln, dg) for st, ln, _, _, dg, _ in images])
            est -= self.cache.peek(keys, min_len=min(self.min_resume, est))
        with (self.gate.enter(est) if use_cache else contextlib.nullcontext()):
            return self._open(conn, peer, h, tokens, sid, t0, stream, streamer, use_cache, want_state, lean, delta_from,
                              on_ready, on_rows, images, keys)

    def _open(self, conn, peer, h, tokens, sid, t0, stream, streamer, use_cache, want_state, lean, delta_from, on_ready, on_rows,
              images, keys=None):
        try:
            arrays, manifest, info = self.prefill(sid, tokens, use_cache=use_cache, on_ready=on_ready, on_rows=on_rows,
                                                  lean=lean, delta_from=delta_from, images=images, keys=keys)
        except Exception:
            try:
                self.submit(("close", sid))
            except Exception:  # noqa: BLE001
                pass
            raise
        manifest["identity"] = h.get("identity") or self.identity
        dt = info["prefill_s"]
        if stream:
            streamer.finish(arrays, manifest)
        else:
            send_frame(conn, b"ACK ", {"ok": True, "session": sid, "prefill_s": dt, "resumed_tokens": info["resumed_tokens"],
                                       "numerics": NUMERICS, **self._ack_extra(sid)})
            if want_state:
                blob = serialize_parts(arrays, manifest)
                send_frame(conn, b"STAT", {"format": manifest["format"], "prompt_tokens": len(tokens), "bytes": len(blob),
                                           "prefill_s": manifest["timing"]["prefill_seconds"]}, blob)
        t_sent = time.perf_counter() - t0
        snap = self.save_snapshot(sid, info["keys"], info) if use_cache else None
        log(f"session {sid} from {peer[0]}: {len(tokens)} tokens"
            + (f", {len(images)} image(s) (ViT {info.get('vision_images', 0)} in {info.get('vision_s', 0):.2f}s)" if images else "")
            + f", resumed {info['resumed_tokens']}, prefill {dt:.2f}s, "
            f"open total {t_sent:.2f}s{' stream' if stream else ''}"
            + (f", snapshot {snap['bytes'] / 1e6:.0f} MB in {snap['snapshot_s']:.2f}s" if snap else ""))
        return sid

    def handle_conn(self, conn, peer):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        keepalive(conn)
        sid = None
        with self.conn_lock:
            self.open_conns += 1
        try:
            while True:
                if sid is not None:
                    self.spin_readable(conn, sid)
                head = recv_exact(conn, FRAME.size)
                if head is None:
                    break
                tag, hlen, plen = FRAME.unpack(head)
                if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
                    send_frame(conn, b"ERR ", {"error": f"frame too large (header {hlen}, payload {plen})", "retry": False})
                    break
                qa = quickack(plen)
                header = recv_exact(conn, hlen, qa) if hlen else b""
                payload = recv_exact(conn, plen, qa) if plen else b""
                if tag == b"OPEN":
                    h = json.loads(header)
                    n = int(h.get("prompt_tokens") or 0) if h.get("images") else plen // 4
                    if sid is not None or n < 2 or 4 * n > plen or (not h.get("images") and 4 * n != plen):
                        send_frame(conn, b"ERR ", {"error": "bad OPEN"})
                        break
                    arr = np.frombuffer(payload, dtype="<u4", count=n)
                    tokens = arr.tolist()
                    if h.get("prompt_tokens") != len(tokens):
                        send_frame(conn, b"ERR ", {"error": "bad OPEN"})
                        break
                    try:
                        images = parse_images(h, tokens, memoryview(payload)[4 * n:], self.image_token_id) \
                            if h.get("images") else ()
                        if not images and int(np.count_nonzero(arr == self.image_token_id)):
                            raise ValueError("image_token_id in a prompt without images")
                    except ValueError as e:
                        send_frame(conn, b"ERR ", {"error": f"bad OPEN images: {e}", "retry": False})
                        break
                    if len(tokens) > self.max_pos:
                        send_frame(conn, b"ERR ", {"error": f"prompt of {len(tokens)} tokens exceeds the context length {self.max_pos}", "retry": False})
                        break
                    sid = self.handle_open(conn, peer, h, tokens, images)
                    if sid is None:
                        break
                elif tag == b"STEP":
                    if len(header) == WIRE.STEPD_HDR.size and sid in self.dsp:
                        self.stepd(conn, sid, header, payload)  # SPEC V2.1: a STEPD sent under the tag b"STEP"
                        continue
                    s, keep, L = STEP_HDR.unpack(header)
                    if (s == sid and sid in self.dsp and 1 <= L <= MAX_STEP_ROWS and plen > 4 * L
                            and (plen - 4 * L) % WIRE.TAP_ROW_BYTES == 0):
                        # SPEC V2.2: a granted session's plain STEP carrying the committed rows' taps
                        self.step_taps(conn, sid, keep, list(struct.unpack_from("<%dI" % L, payload)),
                                       memoryview(payload)[4 * L:])
                        continue
                    if s != sid or L < 1 or L > MAX_STEP_ROWS or plen != 4 * L:
                        send_frame(conn, b"ERR ", {"error": f"bad STEP session={s} L={L}"})
                        break
                    ids = list(struct.unpack("<%dI" % L, payload))
                    if keep + L > self.max_pos:
                        send_frame(conn, b"ERR ", {"error": f"context length {self.max_pos} exceeded at {keep + L}", "retry": False})
                        break
                    t0 = time.perf_counter()
                    self.last_step[sid] = time.monotonic()
                    cmd = ("step", sid, keep, ids)
                    job = self.run_now(cmd) if flag("inline", True) else self.submit_async(cmd)
                    step, box_s = job.wait()
                    t_get, t_bcast, t_exec = job.t_get, job.t_bcast, job.t_exec
                    out = step["payload"]
                    if step["L"] != L or len(out) != L * STEP_ROW_BYTES:
                        send_frame(conn, b"ERR ", {"error": f"step payload {len(out)} bytes for L={L}"})
                        break
                    conn.sendall(FRAME.pack(b"STPR", STPR_HDR.size, len(out)) + STPR_HDR.pack(sid, keep + L, L, len(out), box_s) + out)
                    ds = self.dsp.get(sid)
                    if ds is not None:  # a granted session's plain STEP: the next STEPD checks accept against it
                        ds.pending, ds.length = (keep, ids), keep + L
                    if os.environ.get("SPLIT_NV_STEP_LOG") or flag("step_log", False):
                        tm = step.get("t")
                        ph = (" " + " ".join(f"{k}={(b - a) * 1e3:.3f}" for k, a, b in zip(tm[0::2], tm[1::2], tm[3::2])
                                             if isinstance(k, str))) if tm else ""
                        log(f"session {sid} step keep={keep} L={L} box={box_s * 1e3:.2f}ms total={(time.perf_counter() - t0) * 1e3:.2f}ms "
                            f"queue={(t_get - t0) * 1e3:.2f} bcast={(t_bcast - t_get) * 1e3:.2f} "
                            f"exec={(t_exec - t_bcast) * 1e3:.2f} reply={(time.perf_counter() - t_exec) * 1e3:.2f}{ph}")
                elif tag == WIRE.TAG_STEPD:
                    self.stepd(conn, sid, header, payload)
                elif tag == WIRE.TAG_RING:
                    self.ring(sid, header, payload)
                elif tag == b"CLOS":
                    break
                else:
                    send_frame(conn, b"ERR ", {"error": f"unknown tag {tag!r}"})
                    break
        except WIRE.WireError as e:
            log(f"connection {peer} session {sid}: {e}")
            st = self.dsp_stats["hard_errors"]
            st[e.code] = st.get(e.code, 0) + 1
            try:
                send_frame(conn, b"ERR ", {"error": str(e)[:500], "retry": False, "code": e.code})
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            log(f"connection {peer}: {e}")
            try:
                send_frame(conn, b"ERR ", {"error": str(e).splitlines()[0][:500], "retry": isinstance(e, Busy)})
            except Exception:  # noqa: BLE001
                pass
        finally:
            if sid is not None:
                self.last_step.pop(sid, None)
                self._dspark_release(sid)
                try:
                    self.submit(("close", sid))
                except Exception as e:  # noqa: BLE001
                    log(f"close {sid}: {e}")
            conn.close()
            with self.conn_lock:
                self.open_conns -= 1

    # ---- DSpark on the box: STEPD / RING (STEPD-SPEC.md v1) -----------------------------------------------------------
    def _dspark_grant(self, sid, h, length):
        """OPEN: grant box drafting (ring slot + cost tables) or record why not, for the ACK."""
        costs, why = WIRE.parse_open(h)
        if costs is None and why is None:
            return
        if why is None:
            if self.engine.dspark is None:
                why = "not_loaded"
            elif not flag("dspark", True):
                why = "disabled"
        with self.dsp_lock:
            if why is None:
                if self.dsp_free:
                    self.dsp[sid] = DsSession(self.dsp_free.pop(0), costs, length)
                    self.dsp_stats["granted"] += 1
                    return
                why = "no_slot"
            self.dsp_off[sid] = why
            self.dsp_stats["refused"][why] = self.dsp_stats["refused"].get(why, 0) + 1

    def _ack_extra(self, sid):
        if sid in self.dsp:
            return {"dspark": WIRE.capability()}
        why = self.dsp_off.pop(sid, None)
        return {"dspark_off": why} if why else {}

    def _dspark_release(self, sid):
        with self.dsp_lock:
            self.dsp_off.pop(sid, None)
            ds = self.dsp.pop(sid, None)
            if ds is not None:
                self.dsp_free.append(ds.slot)
                self.dsp_free.sort()

    def _dspark_enabled(self):
        return self.engine.dspark is not None and bool(flag("dspark", True))

    def pf_active(self, sid):
        """STPD PF_ACTIVE: another session's prefill is admitted and unfinished (running, queued or suspended). Runtime
        flag dspark_pf_share > 0 (default 0: always set then) clears it while a running preemptible chunk's decode share
        is below dspark_pf_share x preempt_share -- off by default: at c2 that rule oscillates (box drafting pushes the
        decode share over the threshold, Mac drafting pulls it back under; SPEC V2.4)."""
        if not (any(k != sid for k in list(self.pf_need)) or self.gate.fifo):
            return False
        thr = float(flag("dspark_pf_share", 0.0))
        w = preempt._win
        if thr <= 0 or w is None:
            return True
        return w.step_s >= thr * float(flag("preempt_share", preempt.SHARE)) * (time.perf_counter() - w.t0)

    def ring(self, sid, header, payload):
        """RING: prime / re-prime this session's ring. No reply; the result shows in the next STPD (RING_OK, reason)."""
        try:
            h = json.loads(header)
        except ValueError:
            raise WIRE.WireError("bad_ring", "RING header is not JSON")
        r = WIRE.decode_ring(h if isinstance(h, dict) else {}, payload, self.max_pos)
        if sid is None or r["session"] != sid:
            raise WIRE.WireError("bad_session", f"RING session {r['session']} on connection session {sid}")
        ds = self.dsp.get(sid)
        if ds is None:
            raise WIRE.WireError("dspark_not_granted", "RING on a session without the dspark grant")
        self.dsp_stats["rings"] += 1
        ds.ring_ok = False
        if not self._dspark_enabled():
            ds.ring_reason = WIRE.REASON["ring_not_ok"]
            return
        if r["digest_ok"] is False:
            ds.ring_reason = WIRE.REASON["ring_rejected"]
            log(f"session {sid}: RING sha256 mismatch (offset {r['offset']}): ring rejected")
            return
        cmd = ("ring", sid, ds.slot, r["offset"], bytes(r["keys"]), r["keys_rows"], bytes(r["taps"]), r["taps_rows"])
        try:
            self.run_now(cmd).wait()
        except RuntimeError as e:
            ds.ring_reason = WIRE.REASON["ring_rejected"]
            log(f"session {sid}: RING job failed, ring rejected: {str(e).splitlines()[0][:300]}")
            return
        ds.ring_ok, ds.ring_offset, ds.ring_reason = True, r["offset"], 0

    def stepd(self, conn, sid, header, payload):
        """STEPD: accept check -> [append committed taps] -> [draft] -> step -> STPD (one GPU job on every rank)."""
        t0 = time.perf_counter()
        f = WIRE.decode_stepd(header, payload, self.max_pos)
        if sid is None or f["session"] != sid:
            raise WIRE.WireError("bad_session", f"STEPD session {f['session']} on connection session {sid}")
        ds = self.dsp.get(sid)
        if ds is None:
            raise WIRE.WireError("dspark_not_granted", "STEPD on a session without the dspark grant")
        keep, anchor, nver, a, mode, flags = f["keep"], f["anchor"], f["nver"], f["a"], f["mode"], f["flags"]
        expect = None
        if nver:
            WIRE.check_accept(ds.pending, keep, anchor, nver, a, f["argmax"])
            expect = (ds.pending[0], nver)
        else:
            base = ds.pending[0] if ds.pending is not None else ds.length
            if not base <= keep <= ds.length:
                raise WIRE.WireError("bad_keep", f"kickoff keep {keep} outside [{base}, {ds.length}]")
        enabled = self._dspark_enabled()
        n_app = f["ntaps"]
        app_base = keep - n_app
        append = False
        if not enabled:
            if ds.ring_ok:
                ds.ring_ok, ds.ring_reason = False, WIRE.REASON["ring_not_ok"]
        elif nver and not n_app:  # NO_TAPS: the committed rows never reach the ring
            if ds.ring_ok:
                ds.ring_ok, ds.ring_reason = False, WIRE.REASON["ring_not_ok"]
        elif n_app:
            if ds.ring_ok and ds.ring_offset == app_base:
                append = True
            elif ds.ring_ok:  # a gap (a plain STEP in between, or a RING at another offset)
                ds.ring_ok, ds.ring_reason = False, WIRE.REASON["ring_not_ok"]
        ring_cur = ds.ring_ok and (append or ds.ring_offset == keep)
        dmax = max(0, min(f["dmax"], self.max_pos - keep - 1))
        if not enabled:
            reason = WIRE.REASON["disabled"]
        elif not ring_cur:
            reason = ds.ring_reason or WIRE.REASON["ring_not_ok"]
        elif mode == WIRE.MODE_BOX and keep < WIRE.MIN_CTX:
            reason = WIRE.REASON["short_context"]
        elif mode == WIRE.MODE_BOX and dmax < 1:
            reason = WIRE.REASON["dmax_zero"]
        else:
            reason = 0
        draft = mode == WIRE.MODE_BOX and reason == 0
        # SPEC V2.3: a mode 0 fallback at >= MIN_CTX answers [anchor, anchor] (a filler draft; any draft is exact), so the
        # Mac verify stays on its L >= 2 path; short context, dmax 0 and the context end stay bonus only
        filler = mode == WIRE.MODE_BOX and not draft and reason in (1, 2, 5) and keep >= WIRE.MIN_CTX and dmax >= 1
        costs = WIRE.tier(ds.costs["fused" if flags & WIRE.F_FUSED else "pipe"], keep) if draft else None
        explicit = f["explicit"] if mode == WIRE.MODE_EXPLICIT else ([anchor] if filler else [])
        cmd = ("stepd", sid, ds.slot, keep, anchor, app_base, n_app if append else 0, bytes(f["taps"]) if append else b"",
               draft, WIRE.WIDTH, dmax if draft else 0, costs, explicit, expect)
        self.last_step[sid] = time.monotonic()
        job = self.run_now(cmd) if flag("inline", True) else self.submit_async(cmd)
        step, box_s, info = job.wait()
        ids = info["ids"]
        L = len(ids)
        out = step["payload"]
        if step["L"] != L or len(out) != L * STEP_ROW_BYTES:
            raise RuntimeError(f"stepd payload {len(out)} bytes for L={L}")
        ds.pending, ds.length = (keep, ids), keep + L
        if append:
            ds.ring_offset = keep
        drafted = info["drafted"]
        rflags = ((WIRE.R_DRAFTED if drafted else 0) | (WIRE.R_PF_ACTIVE if self.pf_active(sid) else 0)
                  | (WIRE.R_RING_OK if ring_cur else 0) | (0 if enabled else WIRE.R_DISABLED))
        if drafted or filler:
            mode_used = WIRE.MODE_BOX
        elif mode == WIRE.MODE_EXPLICIT:
            mode_used = WIRE.MODE_EXPLICIT
        else:
            mode_used = WIRE.MODE_BONUS
        if drafted:
            drafts, maxprob = info["toks"], info["probs"]
        elif filler:
            drafts, maxprob = [anchor] * WIRE.WIDTH, [0.0] * WIRE.WIDTH
        else:
            drafts, maxprob = (), ()
        hdr = WIRE.encode_stpd_header(sid, keep, L, box_s, a if nver else WIRE.A_BOX_NONE, mode_used, rflags, reason,
                                      drafts=drafts, maxprob=maxprob, drafter_ms=info["drafter_ms"])
        conn.sendall(FRAME.pack(WIRE.TAG_STPD, len(hdr), len(out)) + hdr + out)
        st = self.dsp_stats
        kind = "filler" if filler else ("box", "explicit", "bonus")[mode_used]
        st["cycles"][kind] = st["cycles"].get(kind, 0) + 1
        if mode == WIRE.MODE_BOX and not drafted:
            name = WIRE.REASONS.get(reason, str(reason))
            st["fallbacks"][name] = st["fallbacks"].get(name, 0) + 1
        if drafted:
            st["drafter_ms"].append(round(info["drafter_ms"], 3))
            del st["drafter_ms"][:-512]
            if info.get("split_ms"):
                sp = st.setdefault("split_ms", [])
                sp.append(info["split_ms"])
                del sp[:-512]
        if os.environ.get("SPLIT_NV_STEP_LOG") or flag("step_log", False):
            sp = info.get("split_ms")
            log(f"session {sid} stepd keep={keep} nver={nver} a={a} mode={mode}->{mode_used} L={L} reason={reason} "
                f"app={n_app if append else 0} box={box_s * 1e3:.2f}ms drafter={info['drafter_ms']:.2f}ms "
                + (f"(launch {sp[0]:.3f} prepare {sp[1]:.3f} wait {sp[2]:.3f}) " if sp else "")
                + f"total={(time.perf_counter() - t0) * 1e3:.2f}ms")

    def step_taps(self, conn, sid, keep, ids, taps):
        """SPEC V2.2: STEP(keep, ids) of a granted session with the committed rows' taps [keep - n, keep) after the ids:
        exactly the plain STEP (STPR reply, same rows) plus the ring append a STEPD would do, so a Mac-side fallback to
        plain STEP leaves no ring gap. As a STEP: no accept check; keep follows the STEP rules."""
        ds = self.dsp[sid]
        n = len(taps) // WIRE.TAP_ROW_BYTES
        L = len(ids)
        base = ds.pending[0] if ds.pending is not None else ds.length
        if not base <= keep <= ds.length or n > keep:
            raise WIRE.WireError("bad_keep", f"STEP+taps keep {keep} outside [{base}, {ds.length}] or {n} taps rows")
        if keep + L > self.max_pos:
            raise WIRE.WireError("context_exceeded", f"context length {self.max_pos} exceeded at {keep + L}")
        append = self._dspark_enabled() and ds.ring_ok and ds.ring_offset == keep - n
        if not append and ds.ring_ok:
            ds.ring_ok, ds.ring_reason = False, WIRE.REASON["ring_not_ok"]
        cmd = ("stepd", sid, ds.slot, keep, ids[0], keep - n, n if append else 0, bytes(taps) if append else b"", False,
               WIRE.WIDTH, 0, None, ids[1:], None)
        self.last_step[sid] = time.monotonic()
        job = self.run_now(cmd) if flag("inline", True) else self.submit_async(cmd)
        step, box_s, info = job.wait()
        out = step["payload"]
        if step["L"] != L or len(out) != L * STEP_ROW_BYTES:
            raise RuntimeError(f"step payload {len(out)} bytes for L={L}")
        conn.sendall(FRAME.pack(b"STPR", STPR_HDR.size, len(out)) + STPR_HDR.pack(sid, keep + L, L, len(out), box_s) + out)
        ds.pending, ds.length = (keep, ids), keep + L
        if append:
            ds.ring_offset = keep
        self.dsp_stats["step_taps"] = self.dsp_stats.get("step_taps", 0) + 1

    def dspark_summary(self):
        D = self.engine.dspark
        ms = sorted(self.dsp_stats["drafter_ms"])
        pct = (lambda q: ms[min(len(ms) - 1, int(q * len(ms)))] if ms else None)
        sp = self.dsp_stats.get("split_ms") or []
        med = (lambda i: sorted(x[i] for x in sp)[len(sp) // 2] if sp else None)  # noqa: E731
        return {"loaded": D is not None, "flag": bool(flag("dspark", True)), "sessions": len(self.dsp),
                "free_slots": len(self.dsp_free), **({"drafter": D.summary(), "load": getattr(D, "load_info", None)} if D else {}),
                **{k: v for k, v in self.dsp_stats.items() if k not in ("drafter_ms", "split_ms")},
                "drafter_ms_p50": pct(0.5), "drafter_ms_p90": pct(0.9),
                # the drafter job's host timeline (engine.cmd_stepd): launch, step bookkeeping, wait for the draft
                "launch_ms_p50": med(0), "prepare_ms_p50": med(1), "wait_ms_p50": med(2)}


def install_signals(front):
    signal.signal(signal.SIGTERM, front.drain_and_exit)
    signal.signal(signal.SIGUSR1, front.drain_and_exit)
