"""Box-side prefix cache: snapshots of a session's layers 0-20 state, kept as files in host shared memory
(/dev/shm/split-nv/cache/<numerics>/) so they survive an engine or container restart (not a host reboot).

Layout (one set per numerics version; `.r0`/`.r1` = TP rank):
  blk-<bid>.r<rank>    native pages of one 8K grid block [8192 b, 8192 (b+1)): compressed KV (ratio 2 layers 2/8/14,
                       ratio 1 layer 20) + packed index-K pages. Written once, shared by every entry that covers it.
  ent-<key>.r<rank>    per snapshot point P: the pages of [floor8192(P), P), the SWA pages holding the window
                       [P-128, P), the ratio-2 pending-pair rings, the Engram history row, and (rank 0, prompt-end
                       snapshots only) the capture state (Mac-packed window ring, hidden tail ring, ratio-2 tails).
  rows-<bid>, rows-tail-<key>   the Mac-packed layer 2/8/14/20 rows of the same position ranges (rank 0 writes).
  tok-<key>.npy, idx-<key>.json the entry's tokens and index record (written last: an entry exists once its json does).
Page-aligned allocation keeps pos % page == slot % page for the full and the SWA allocator, so pages copy 1:1.

A restore rebuilds the request's allocation through the ordinary chunked-extend path without a forward (same KV
rows, SWA window and eviction floor as a prefill of the P tokens), then writes the saved pages into the new slots.
"""

import glob
import hashlib
import json
import os
import queue
import threading
import time

import numpy as np
import torch

WINDOW = 128
GRID = 8192
GATHER_PAGES = 512


def cache_root(numerics):
    return os.path.join(os.environ.get("SPLIT_NV_CACHE_DIR", "/dev/shm/split-nv/cache"), numerics)


def _atomic_save(tensors, path, metadata=None):
    from safetensors.torch import save_file

    tmp = f"{path}.tmp{os.getpid()}"
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata=metadata)
    os.replace(tmp, path)


def _load(path):
    from safetensors.torch import load_file

    return load_file(path)


# --------------------------------------------------------------------------------------------- engine side (each rank)
class RankStore:
    """Per-rank snapshot files; writes go through one background thread so the GPU loop only pays the D2H copy."""

    def __init__(self, engine, numerics):
        self.engine = engine
        self.rank = engine.tp_rank
        self.root = cache_root(numerics)
        os.makedirs(self.root, exist_ok=True)
        self.q = queue.Queue()
        self.errors = []
        threading.Thread(target=self._writer, daemon=True).start()

    def path(self, name):
        return os.path.join(self.root, f"{name}.r{self.rank}")

    def _writer(self):
        while True:
            item = self.q.get()
            try:
                _atomic_save(*item)
            except Exception as e:  # noqa: BLE001
                self.errors.append(repr(e))
            finally:
                self.q.task_done()

    def put(self, name, tensors, metadata=None):
        self.q.put((tensors, self.path(name), metadata))

    def flush(self):
        self.q.join()

    # ---- pools ------------------------------------------------------------------------------------------------
    def pools(self):
        p = self.engine.mr.token_to_kv_pool
        kv = [buf for r in sorted(p.kv_pools) for buf in p.kv_pools[r].kv_buffer]
        index = []
        for r in sorted(p.index_pools):
            pool = p.index_pools[r]
            per_full = (p.page_size // r) // pool.page_size
            index += [(buf, per_full) for buf in pool.contiguous_page_row_buffers()]
        rings = [c for c in p.compress_state_pools if c is not None]
        return p, kv, index, rings

    def page_ids(self, sess, a, b):
        """Full-pool page ids holding positions [a, b) (a page aligned)."""
        mr, ps = self.engine.mr, self.engine.page_size
        row = sess.req.kv.req_pool_idx
        locs = mr.req_to_token_pool.req_to_token[row, a:b:ps].to(torch.int64)
        return torch.div(locs, ps, rounding_mode="floor")

    def swa_page_ids(self, sess, P):
        mr, ps = self.engine.mr, self.engine.page_size
        row = sess.req.kv.req_pool_idx
        lo = max(0, P - WINDOW)
        pos = torch.clamp(torch.arange(lo // ps, (P - 1) // ps + 1, dtype=torch.int64, device=mr.device) * ps, min=lo)
        swa = mr.token_to_kv_pool_allocator.full_to_swa_index_mapping[mr.req_to_token_pool.req_to_token[row, pos].to(torch.int64)]
        if bool((swa <= 0).any()):
            raise RuntimeError(f"snapshot: SWA window of length {P} is not resident")
        return torch.div(swa, ps, rounding_mode="floor")

    @staticmethod
    def _gather(buf, idx):
        out = torch.empty((idx.numel(), *buf.shape[1:]), dtype=buf.dtype)
        for a in range(0, idx.numel(), GATHER_PAGES):
            part = idx[a:a + GATHER_PAGES]
            out[a:a + part.numel()].copy_(buf.index_select(0, part))
        return out

    @staticmethod
    def _scatter(buf, idx, host):
        for a in range(0, idx.numel(), GATHER_PAGES):
            part = idx[a:a + GATHER_PAGES]
            buf.index_copy_(0, part, host[a:a + part.numel()].to(buf.device))

    def _pages(self, pages):
        _, kv, index, _ = self.pools()
        out = {f"kv{i}": self._gather(buf, pages) for i, buf in enumerate(kv)}
        for i, (buf, per) in enumerate(index):
            ip = (pages[:, None] * per + torch.arange(per, device=pages.device)).reshape(-1)
            out[f"ix{i}"] = self._gather(buf, ip)
        return out

    def _put_pages(self, pages, t):
        _, kv, index, _ = self.pools()
        for i, buf in enumerate(kv):
            self._scatter(buf, pages, t[f"kv{i}"])
        for i, (buf, per) in enumerate(index):
            ip = (pages[:, None] * per + torch.arange(per, device=pages.device)).reshape(-1)
            self._scatter(buf, ip, t[f"ix{i}"])

    # ---- commands ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def write_block(self, sess, bid, end):
        """Grid block [end - GRID, end) of `sess` (its pages are final once the prefill passed `end`)."""
        t = self._pages(self.page_ids(sess, end - GRID, end))
        torch.cuda.synchronize()
        self.put(f"blk-{bid}", t)

    @torch.no_grad()
    def write_entry(self, sess, key, capture=None):
        """Snapshot point P = sess.length: tail pages [floor8192(P), P), window, rings, history (+ capture)."""
        if sess.pending is not None:
            raise RuntimeError("snapshot: session has unverified step rows")
        P = sess.length
        g = P // GRID * GRID
        t = {}
        if P > g:
            t.update(self._pages(self.page_ids(sess, g, P)))
        p, _, _, rings = self.pools()
        swa = self.swa_page_ids(sess, P)
        for i, buf in enumerate(p.swa_kv_pool.kv_buffer):
            t[f"swa{i}"] = self._gather(buf, swa)
        row = sess.req.kv.req_pool_idx
        for i, c in enumerate(rings):
            t[f"ring{i}"] = c.kv_score_buffer.kv_score[row * c.ring_size:(row + 1) * c.ring_size].cpu()
        hasher = self.engine.mr.model.model.engram_hasher
        if hasher is not None:
            t["hist"] = hasher.history[row].cpu()
        meta = {"P": str(P)}
        if capture is not None:
            meta["capture"] = json.dumps({"ntok": capture["ntok"], "rows": {str(k): v for k, v in capture["rows"].items()},
                                          "swa": sorted(capture["swa"]), "tail": sorted(capture["tail"]),
                                          "h_ring": capture["h_ring"] is not None})
            for L, (b, pos) in capture["swa"].items():
                t[f"cap.swa.{L}"], t[f"cap.swapos.{L}"] = b, pos
            for L, v in capture["tail"].items():
                for j, x in enumerate(v):
                    t[f"cap.tail.{L}.{j}"] = x
            if capture["h_ring"] is not None:
                for j, x in enumerate(capture["h_ring"]):
                    t[f"cap.h.{j}"] = x
        torch.cuda.synchronize()
        self.put(f"ent-{key}", t, meta)

    @torch.no_grad()
    def restore(self, sess, key, blocks, P, chunk):
        from safetensors import safe_open

        from split_nv.imagekeys import key_tokens

        self.flush()
        tokens = key_tokens(np.load(os.path.join(self.root, f"tok-{key}.npy")), self.engine.image_token_id)
        if len(tokens) != P:
            raise RuntimeError(f"restore: entry {key} has {len(tokens)} tokens, expected {P}")
        ids = tokens.tolist()
        for a in range(0, P, chunk):
            self.engine._extend(sess, ids[a:a + chunk], forward=False)
        for b, bid in enumerate(blocks):
            self._put_pages(self.page_ids(sess, b * GRID, (b + 1) * GRID), _load(self.path(f"blk-{bid}")))
        path = self.path(f"ent-{key}")
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        t = _load(path)
        g = P // GRID * GRID
        if len(blocks) != g // GRID or int(meta.get("P", -1)) != P:
            raise RuntimeError(f"restore: entry {key} P={meta.get('P')} blocks={len(blocks)} does not match P={P}")
        if P > g:
            self._put_pages(self.page_ids(sess, g, P), t)
        p, _, _, rings = self.pools()
        swa = self.swa_page_ids(sess, P)
        for i, buf in enumerate(p.swa_kv_pool.kv_buffer):
            self._scatter(buf, swa, t[f"swa{i}"])
        row = sess.req.kv.req_pool_idx
        mr = self.engine.mr
        for i, c in enumerate(rings):
            c.kv_score_buffer.kv_score[row * c.ring_size:(row + 1) * c.ring_size].copy_(t[f"ring{i}"].to(mr.device))
        hasher = mr.model.model.engram_hasher
        if hasher is not None:
            hasher.history[row].copy_(t["hist"].to(mr.device))
        cap = self.engine.cap
        if cap.enabled:
            cap.reset()
            cap.ntok = P
            cap.tokens = [torch.as_tensor(tokens, dtype=torch.int64)]
            if "capture" in meta:
                c = json.loads(meta["capture"])
                cap.rows = {int(k): v for k, v in c["rows"].items()}
                cap.swa = {int(L): (t[f"cap.swa.{L}"].to(mr.device), t[f"cap.swapos.{L}"].to(mr.device)) for L in c["swa"]}
                cap.tail = {int(L): tuple(t[f"cap.tail.{L}.{j}"].to(mr.device) for j in range(3)) for L in c["tail"]}
                cap.h_ring = tuple(t[f"cap.h.{j}"].to(mr.device) for j in range(3)) if c["h_ring"] else None
            else:  # grid point: the prefill resuming here recomputes window, tails and hidden tail (>= 256 new rows)
                cap.rows = {L: P // r for L, r in ((2, 2), (8, 2), (14, 2), (20, 1))}
        torch.cuda.synchronize()
        sess.length = P
        sess.alloc_len = sess.req.kv.kv_allocated_len
        sess.pending = None


# --------------------------------------------------------------------------------------------- rank 0 index
class Entry:
    __slots__ = ("key", "P", "tokens", "blocks", "capture", "bytes", "last", "hits")


class PrefixIndex:
    """Prefix lookup over snapshot entries (rank 0 front threads), on uint64 keys (imagekeys.prompt_keys: token ids,
    or image-content keys inside image spans). Entries are LRU-evicted as a whole; a block file is deleted when no
    entry references it. The index is rebuilt from idx-*.json at start (uint32 token files of text-only entries
    written before keys existed load as text keys)."""

    def __init__(self, numerics, budget_bytes, ranks=2):
        self.root = cache_root(numerics)
        os.makedirs(self.root, exist_ok=True)
        self.budget = budget_bytes
        self.ranks = ranks
        self.entries = {}
        self.block_refs = {}
        self.block_bytes = {}
        self.lock = threading.Lock()
        self.stats = dict(lookups=0, hits=0, resumed_tokens=0, saved=0, evicted=0, loaded=0, dropped_at_load=0)
        self.seq = int(time.time() * 1000)
        self._load()

    def p(self, name):
        return os.path.join(self.root, name)

    def entry_files(self, key):
        return [self.p(f"idx-{key}.json"), self.p(f"tok-{key}.npy"), self.p(f"rows-tail-{key}")] + \
            [self.p(f"ent-{key}.r{r}") for r in range(self.ranks)]

    def block_files(self, bid):
        return [self.p(f"blk-{bid}.r{r}") for r in range(self.ranks)] + [self.p(f"rows-{bid}")]

    def _load(self):
        for path in glob.glob(self.p("idx-*.json")):
            key = os.path.basename(path)[4:-5]
            try:
                rec = json.load(open(path))
                e = Entry()
                e.key, e.P, e.blocks, e.capture = rec["key"], rec["P"], rec["blocks"], rec["capture"]
                e.tokens = np.load(self.p(f"tok-{key}.npy")).astype(np.uint64)
                missing = [f for f in self.entry_files(key) if not os.path.exists(f)]
                missing += [f for b in e.blocks for f in self.block_files(b) if not os.path.exists(f)]
                if missing or len(e.tokens) != e.P:
                    raise ValueError(f"incomplete: {missing[:3]}")
                e.bytes = sum(os.path.getsize(f) for f in self.entry_files(key))
                e.last, e.hits = os.path.getmtime(path), 0
                self.entries[key] = e
                for b in e.blocks:
                    self.block_refs[b] = self.block_refs.get(b, 0) + 1
                self.stats["loaded"] += 1
            except Exception:  # noqa: BLE001
                self.stats["dropped_at_load"] += 1
                for f in self.entry_files(key):
                    _unlink(f)
        for b in self.block_refs:
            self.block_bytes[b] = sum(os.path.getsize(f) for f in self.block_files(b))
        for f in glob.glob(self.p("blk-*")) + glob.glob(self.p("rows-b*")):
            bid = os.path.basename(f).split("-", 1)[1].split(".")[0]
            if bid not in self.block_refs:
                _unlink(f)
        for f in glob.glob(self.p("*.tmp*")):
            _unlink(f)

    def new_id(self, prefix):
        with self.lock:
            self.seq += 1
            return f"{prefix}{self.seq}"

    def lookup(self, ids, min_len=1):
        """Longest cached exact prefix of keys `ids` with P >= min_len leaving 0 or >= 2 tokens to prefill (a 1-row
        chunk rounds differently); a grid entry (no capture state) needs >= 256 new tokens."""
        arr = np.asarray(ids, dtype=np.uint64)
        n = len(ids)
        with self.lock:
            self.stats["lookups"] += 1
            cands = sorted((e for e in self.entries.values() if min_len <= e.P <= n and n - e.P != 1
                            and (e.capture or n - e.P >= 256)), key=lambda e: -e.P)
            for e in cands:
                if np.array_equal(e.tokens, arr[:e.P]):
                    e.last = time.time()
                    e.hits += 1
                    self.stats["hits"] += 1
                    self.stats["resumed_tokens"] += e.P
                    return e
        return None

    def find(self, P, ids_prefix):
        arr = np.asarray(ids_prefix, dtype=np.uint64)
        with self.lock:
            return next((e for e in self.entries.values() if e.P == P and np.array_equal(e.tokens, arr)), None)

    def covered(self, P, ids_prefix, capture):
        """An entry at exactly this prefix exists (with capture state, if asked)."""
        arr = np.asarray(ids_prefix, dtype=np.uint64)
        with self.lock:
            return any(e.P == P and (e.capture or not capture) and np.array_equal(e.tokens, arr) for e in self.entries.values())

    def add(self, key, ids, blocks, capture):
        """Register an entry whose files (and its blocks' files) are written."""
        e = Entry()
        e.key, e.P, e.blocks, e.capture = key, len(ids), list(blocks), capture
        e.tokens = np.asarray(ids, dtype=np.uint64)
        np.save(self.p(f"tok-{key}.npy"), e.tokens)
        tmp = self.p(f"idx-{key}.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"key": key, "P": e.P, "blocks": e.blocks, "capture": capture,
                       "digest": hashlib.sha256(e.tokens.tobytes()).hexdigest()}, f)
        os.replace(tmp, self.p(f"idx-{key}.json"))
        e.bytes = sum(os.path.getsize(f) for f in self.entry_files(key) if os.path.exists(f))
        e.last, e.hits = time.time(), 0
        with self.lock:
            for b in e.blocks:
                if b not in self.block_bytes:
                    self.block_bytes[b] = sum(os.path.getsize(f) for f in self.block_files(b) if os.path.exists(f))
            for b in e.blocks:
                self.block_refs[b] = self.block_refs.get(b, 0) + 1
            self.entries[key] = e
            self.stats["saved"] += 1

    def total(self):
        return sum(e.bytes for e in self.entries.values()) + sum(self.block_bytes.get(b, 0) for b in self.block_refs)

    def evict(self, reserve=0):
        """Drop LRU entries (and blocks nobody references) until the total fits the budget with `reserve` to spare."""
        dead = []
        with self.lock:
            while self.entries and self.total() + reserve > self.budget:
                e = min(self.entries.values(), key=lambda e: e.last)
                del self.entries[e.key]
                self.stats["evicted"] += 1
                dead += self.entry_files(e.key)
                for b in e.blocks:
                    self.block_refs[b] -= 1
                    if not self.block_refs[b]:
                        del self.block_refs[b]
                        self.block_bytes.pop(b, None)
                        dead += self.block_files(b)
        for f in dead:
            _unlink(f)

    def forget_blocks(self, bids):
        """Blocks written for entries that were never registered (e.g. the prefill failed)."""
        with self.lock:
            bids = [b for b in bids if b not in self.block_refs]
        for b in bids:
            for f in self.block_files(b):
                _unlink(f)

    def clear(self):
        with self.lock:
            n = len(self.entries)
            self.entries.clear()
            self.block_refs.clear()
            self.block_bytes.clear()
        for f in glob.glob(self.p("*")):
            _unlink(f)
        return n

    def summary(self):
        with self.lock:
            return dict(self.stats, entries=len(self.entries), blocks=len(self.block_refs), bytes=self.total(),
                        budget=self.budget, root=self.root)


def _unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
