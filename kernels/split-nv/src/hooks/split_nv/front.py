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

from split_nv.imagekeys import parse_images, prefix_digest, prompt_keys
from split_nv.prefix_cache import PrefixIndex, _atomic_save, _load
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
ADMIN_PEERS = set(os.environ.get("SPLIT_NV_ADMIN_PEERS", "127.0.0.1,10.10.10.2").split(","))

log_lock = threading.Lock()


def log(*a):
    with log_lock:
        print("[engine]", *a, flush=True)


def recv_exact(conn, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
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


def fatal_cuda_error(e):
    text = str(e)
    return any(m in text for m in ("device-side assert", "illegal memory access", "CUDA error: an illegal",
                                   "unspecified launch failure", "cudaErrorAssert", "CUBLAS_STATUS_EXECUTION_FAILED"))


class Busy(RuntimeError):
    """Retryable refusal (capacity, draining)."""


class Job:
    __slots__ = ("cmd", "done", "result", "error", "t_get", "t_bcast", "t_exec")

    def __init__(self, cmd):
        self.cmd = cmd
        self.done = threading.Event()
        self.result = self.error = None
        self.t_get = self.t_bcast = self.t_exec = None

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
    def __init__(self, engine, args):
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
        gib = os.environ.get("SPLIT_NV_CACHE_GIB") or os.environ.get("SPLIT_NV_CACHE_GB") or "64"
        self.cache = PrefixIndex(NUMERICS, int(float(gib) * (1 << 30)))
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
                log("GPU job failed:", job.error)
                if fatal_cuda_error(e):
                    # A sticky CUDA error (device-side assert, illegal access) poisons the context: every later job
                    # would fail while /health still said ok. Exit so the unit restarts the engine.
                    log("FATAL CUDA error: exiting for a restart")
                    os._exit(4)
            finally:
                self.current = None
                job.done.set()

    def watchdog(self):
        """A GPU job that never returns means a wedged rank or collective: exit so the supervisor restarts us."""
        while True:
            time.sleep(5)
            cur = self.current
            if cur is not None and time.monotonic() - cur[1] > self.job_timeout:
                log(f"WATCHDOG: GPU job {cur[0]} running {time.monotonic() - cur[1]:.0f}s > {self.job_timeout:.0f}s; exiting")
                os._exit(3)

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
    def prefill(self, sid, tokens, *, use_cache, on_ready=None, on_rows=None, lean=False, delta_from=0, images=()):
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
        keys = prompt_keys(ids, [(st, ln, dg) for st, ln, _, _, dg, _ in images])
        t0 = time.perf_counter()
        info = {"resumed_tokens": 0, "new_blocks": [], "grid_entries": []}
        rows = Rows()
        blocks = []
        entry = None
        with self.prefill_lock:
            if use_cache:
                entry = self.cache.lookup(keys, min_len=min(self.min_resume, n1))
            self.admit(n1)
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

            def submit_next():
                # Plan chunks are split into pieces when another session is decoding (re-checked per chunk): a
                # STEP waits for at most one piece. Grid work rides on the last piece of its chunk.
                if not pending and next_k[0] < len(plan):
                    k = next_k[0]
                    next_k[0] += 1
                    ps = split_piece(*plan[k], self.share_chunk) if self.contended(sid) else [plan[k]]
                    pending.extend((a, e, k, i == len(ps) - 1) for i, (a, e) in enumerate(ps))
                if not pending:
                    return None
                a, e, k, last = pending.pop(0)
                grid = grids[k][1] if grids[k] and last else None
                return self.submit_async(("prefill_chunk", sid, ids[a:e], grid), priority=1), a, e, k, last

            try:
                cur = submit_next()
                job = cur[0] if cur else None
                if on_rows and len(rows):  # the cached prefix's rows go out while the first new chunk computes
                    for part in rows.parts():
                        on_rows(part)
                while cur is not None:
                    job, a, e, k, last = cur
                    part, dt = job.wait()
                    cur = submit_next()
                    job = cur[0] if cur else None
                    timing.append((e - a, dt))
                    rows.add(part)
                    if on_rows:
                        on_rows(part)
                    if last and grids[k] is not None:
                        blocks = self._grid_done(grids[k], keys, e, blocks, rows, info)
                final = self.submit(("prefill_end", sid, use_cache), priority=1)
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

    def save_snapshot(self, sid, ids, info):
        """After a cache=1 prefill (prefill lock held): snapshot the prompt end unless that exact prefix is cached,
        then register it and the prefill's grid entries. Entries become visible only after a barrier: both ranks'
        files are then complete, so an OPEN right behind this one can resume from them."""
        P = len(ids)
        blocks, new_blocks = info["blocks"], list(info["new_blocks"])
        final = len(blocks) == P // GRID and not self.cache.covered(P, ids, capture=True)
        key, dt = None, 0.0
        if final:
            key = self.cache.new_id("e")
            dt = self.submit(("snapshot", sid, key), priority=1)
            info["rows"].save(self.cache.p(f"rows-tail-{key}"), P // GRID * GRID, P)
        self.submit(("barrier",), priority=1)
        for gkey, g, gblocks in info["grid_entries"]:
            self.cache.add(gkey, ids[:g], gblocks, capture=False)
        if final:
            self.cache.add(key, ids, blocks, capture=True)
        self.cache.forget_blocks(new_blocks)  # only those no registered entry references
        if self.cache.total() > self.cache.budget:
            self.cache.evict()  # after the barrier: rank 1 no longer reads any file
        if not final:
            return None
        return {"key": key, "bytes": self.cache.entries[key].bytes if key in self.cache.entries else 0, "snapshot_s": dt}

    def admit(self, n1):
        """Refuse (retryable) a prefill that cannot fit, before anything is allocated: an allocation failure half
        way through a prefill is the one path that can leave the pools inconsistent."""
        cap = self.submit(("capacity",))
        page = 256
        busy = cap["sessions"]
        need_full = (n1 // page + 2) * page + busy * 4 * page
        need_swa = min(n1, CHUNK) + 3 * page + busy * 2 * page
        if cap["rows"] < 1 or cap["full"] < need_full or cap["swa"] < need_swa:
            raise Busy(f"busy: KV capacity (free rows {cap['rows']}, full {cap['full']} < {need_full} or swa {cap['swa']} < {need_swa})")

    def cache_clear(self):
        with self.prefill_lock:
            self.submit(("barrier",), priority=1)
            return {"dropped": self.cache.clear()}

    # ---- health ------------------------------------------------------------------------------------------------
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
        return ok, {"ok": ok, "encoder": state, "step_api": "up" if step_api else "down", "token_map": True,
                    "sessions": len(self.engine.sessions), "connections": self.open_conns,
                    "gpu_job": cur[0] if cur else None, "gpu_job_s": round(busy, 2), "queued_jobs": self.jobs.qsize(),
                    "cache": self.cache.summary(), "uptime_s": round(time.time() - self.t_start),
                    "version": self.version, "sglang": os.environ.get("SPLIT_NV_SGLANG_VERSION", ""), "numerics": NUMERICS,
                    "tree_head": tree, "restart_pending": bool(tree) and not self.version.startswith(tree[:len(self.version.split("-")[0])]),
                    "dev_hook": os.environ.get("SPLIT_NV_DEV") == "1", "vision": self.engine.vision}

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
                        os._exit(1)
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
                    with (front.prefill_lock if body.get("cache") else contextlib.nullcontext()):
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
                    with front.prefill_lock:
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
            os._exit(0)

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
        t0 = time.perf_counter()
        streamer = Streamer(conn, len(tokens) - 1, lean, delta_from) if stream else None

        def on_ready(P):
            if stream:
                send_frame(conn, b"ACK ", {"ok": True, "session": sid, "stream": 1, "resumed_tokens": P, "numerics": NUMERICS,
                                           "est_s": round((len(tokens) - 1 - P) / PREFILL_TOK_S, 2)})

        def on_rows(rows):
            if stream:
                streamer.rows_chunk(rows)
                send_frame(conn, b"PROG", {"tokens_done": streamer.sent["layer.20.slot.2"]})

        with (self.prefill_lock if use_cache else contextlib.nullcontext()):
            return self._open(conn, peer, h, tokens, sid, t0, stream, streamer, use_cache, want_state, lean, delta_from,
                              on_ready, on_rows, images)

    def _open(self, conn, peer, h, tokens, sid, t0, stream, streamer, use_cache, want_state, lean, delta_from, on_ready, on_rows,
              images):
        try:
            arrays, manifest, info = self.prefill(sid, tokens, use_cache=use_cache, on_ready=on_ready, on_rows=on_rows,
                                                  lean=lean, delta_from=delta_from, images=images)
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
                                       "numerics": NUMERICS})
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
        sid = None
        with self.conn_lock:
            self.open_conns += 1
        try:
            while True:
                head = recv_exact(conn, FRAME.size)
                if head is None:
                    break
                tag, hlen, plen = FRAME.unpack(head)
                header = recv_exact(conn, hlen) if hlen else b""
                payload = recv_exact(conn, plen) if plen else b""
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
                    s, keep, L = STEP_HDR.unpack(header)
                    if s != sid or L < 1 or L > MAX_STEP_ROWS or plen != 4 * L:
                        send_frame(conn, b"ERR ", {"error": f"bad STEP session={s} L={L}"})
                        break
                    ids = list(struct.unpack("<%dI" % L, payload))
                    if keep + L > self.max_pos:
                        send_frame(conn, b"ERR ", {"error": f"context length {self.max_pos} exceeded at {keep + L}", "retry": False})
                        break
                    t0 = time.perf_counter()
                    self.last_step[sid] = time.monotonic()
                    job = self.submit_async(("step", sid, keep, ids))
                    step, box_s = job.wait()
                    t_get, t_bcast, t_exec = job.t_get, job.t_bcast, job.t_exec
                    out = step["payload"]
                    if step["L"] != L or len(out) != L * STEP_ROW_BYTES:
                        send_frame(conn, b"ERR ", {"error": f"step payload {len(out)} bytes for L={L}"})
                        break
                    conn.sendall(FRAME.pack(b"STPR", STPR_HDR.size, len(out)) + STPR_HDR.pack(sid, keep + L, L, len(out), box_s) + out)
                    if os.environ.get("SPLIT_NV_STEP_LOG"):
                        log(f"session {sid} step keep={keep} L={L} box={box_s * 1e3:.2f}ms total={(time.perf_counter() - t0) * 1e3:.2f}ms "
                            f"queue={(t_get - t0) * 1e3:.2f} bcast={(t_bcast - t_get) * 1e3:.2f} "
                            f"exec={(t_exec - t_bcast) * 1e3:.2f} reply={(time.perf_counter() - t_exec) * 1e3:.2f}")
                elif tag == b"CLOS":
                    break
                else:
                    send_frame(conn, b"ERR ", {"error": f"unknown tag {tag!r}"})
                    break
        except Exception as e:  # noqa: BLE001
            log(f"connection {peer}: {e}")
            try:
                send_frame(conn, b"ERR ", {"error": str(e).splitlines()[0][:500], "retry": isinstance(e, Busy)})
            except Exception:  # noqa: BLE001
                pass
        finally:
            if sid is not None:
                self.last_step.pop(sid, None)
                try:
                    self.submit(("close", sid))
                except Exception as e:  # noqa: BLE001
                    log(f"close {sid}: {e}")
            conn.close()
            with self.conn_lock:
                self.open_conns -= 1


def install_signals(front):
    signal.signal(signal.SIGTERM, front.drain_and_exit)
    signal.signal(signal.SIGUSR1, front.drain_and_exit)
