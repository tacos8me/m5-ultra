"""Lower transient memory in the prefill indexer's row-tiled top-k (SPLIT_NV_IDX_LOWMEM=1, default off).

sglang's dense_indexer_topk (dsv41_indexer_select) scores one row tile of fp32 logits at a time, sized to
SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB (1024 in production). Two things keep more alive than the budget suggests:
  * `logits = score_rows(rows)` evaluates the new tile while the name still holds the previous one (and `scores`, a
    view of it), so two tiles coexist: 2 GiB at depth (>= 256K tokens for a 4096-row half, >= 128K for 8192 rows);
  * a candidate-consuming layer builds a bool [rows, lc] position mask (repeat_interleave of the kept blocks), its
    negation and the interleave temporary: three more quarter-tiles (~0.75 GiB at budget 1024).
This version drops the previous tile before scoring the next and masks the dropped blocks in place through a
[rows, blocks, block_size] view of the scores. Every kernel that computes a value is the same call on the same inputs;
masked_fill_ writes -inf to exactly the positions the expanded mask selected, so logits, selections and published
block ids are bitwise those of the original (tools/fair/test_idx_lowmem.py).

Fixed-size tiles (SPLIT_NV_IDX_FIXED_TILES=1, default off; needs SPLIT_NV_IDX_LOWMEM=1; runtime flag idx_fixed_tiles,
read per call on each rank -- memory placement only, no collective depends on it). DeepGEMM allocates each logits tile
itself, slightly above the budget and growing with the width (1073741824, 1075838976, 1077936128, ... in consecutive
OOM-retries): every chunk asked for a block a little larger than any cached one, so the caching allocator mapped a new
~1 GiB segment per chunk and kept the old ones, 8.3-8.4 GiB over the allocation peak at 1M (FAIRNESS-MEMORY s3). Here
the tiles of a call whose tile is >= SPLIT_NV_IDX_TILE_MIN_MB (64) are allocated in a private CUDA MemPool
(torch.cuda.use_mem_pool: only this thread's score_rows allocation is routed there), which holds one block reserved at
budget x (1 + SPLIT_NV_IDX_TILE_MARGIN) (0.125) before the first tile of every call: each tile then reuses that one
segment, and no other allocation can split it between chunks. A tile above the cap (never seen at margin 0.125) raises
the cap for later calls and is counted (over_cap). Only addresses change: every kernel, input and output is the same
(tools/fair/test_idx_lowmem.py runs the fixed-tile path through a recording pool and compares bitwise).
"""
import contextlib
import os

import torch

FIXED = os.environ.get("SPLIT_NV_IDX_FIXED_TILES", "0") == "1"
MARGIN = float(os.environ.get("SPLIT_NV_IDX_TILE_MARGIN", "0.125"))
MIN_BYTES = int(float(os.environ.get("SPLIT_NV_IDX_TILE_MIN_MB", "64")) * (1 << 20))
ROUND = 2 << 20  # the caching allocator's large-block rounding


def _round(n):
    return -(-int(n) // ROUND) * ROUND


class TilePool:
    """Private CUDA memory pool for the logits tiles (one per device), reserved at the cap before every call."""

    def __init__(self, margin=MARGIN, min_bytes=MIN_BYTES):
        self.margin, self.min_bytes = margin, min_bytes
        self.pools = {}
        self.cap = 0
        self.stats = {"calls": 0, "tiles": 0, "reserved_bytes": 0, "max_tile_bytes": 0, "over_cap": 0}

    def active(self, tile_bytes):
        if tile_bytes < self.min_bytes:
            return False
        from split_nv.perf_flags import flag
        return bool(flag("idx_fixed_tiles", True))

    def _pool(self, device):
        key = torch.device(device).index
        p = self.pools.get(key)
        if p is None:
            p = self.pools[key] = torch.cuda.MemPool()
        return p

    @contextlib.contextmanager
    def scope(self, device):
        with torch.cuda.use_mem_pool(self._pool(device), device):
            yield

    def reserve(self, device, budget_bytes):
        """Before a call's first tile: make sure the pool holds a free block of the cap (no-op once it does)."""
        self.cap = max(self.cap, _round(budget_bytes * (1 + self.margin)))
        with self.scope(device):
            r = torch.empty(self.cap, dtype=torch.uint8, device=device)
            del r
        self.stats["calls"] += 1
        self.stats["reserved_bytes"] = self.cap

    def note(self, logits):
        nb = logits.untyped_storage().nbytes()
        st = self.stats
        st["tiles"] += 1
        st["max_tile_bytes"] = max(st["max_tile_bytes"], nb)
        if nb > self.cap:
            st["over_cap"] += 1
            self.cap = _round(nb * (1 + self.margin))


POOL = TilePool() if FIXED else None


def summary():
    return {"fixed_tiles": FIXED, **(dict(POOL.stats, cap=POOL.cap) if POOL is not None else {})}


def mask_blocks_(scores, block_ids, block_size):
    """In place: scores.masked_fill_(~candidate_block_mask(block_ids, block_size, width), -inf)."""
    width = scores.shape[-1]
    num_blocks = -(-width // block_size)
    keep = torch.zeros(block_ids.shape[0], num_blocks + 1, dtype=torch.bool, device=block_ids.device)
    keep.scatter_(-1, block_ids.masked_fill(block_ids < 0, num_blocks).long(), True)
    drop = ~keep[:, :num_blocks]
    full = width - width % block_size
    nf = full // block_size
    if nf:
        scores[:, :full].unflatten(-1, (nf, block_size)).masked_fill_(drop[:, :nf, None], -torch.inf)
    if full < width:
        scores[:, full:].masked_fill_(drop[:, nf:nf + 1], -torch.inf)
    return scores


def make(S, pool=None):
    """The low-memory dense_indexer_topk over module S (sglang's dsv41_indexer_select). pool: a TilePool (fixed-size
    tiles) or None; tests pass a recording stand-in."""

    def dense_indexer_topk(*, score_rows, topk_rows, num_tokens, width, compress_lens, ks, q_lens_cpu, lc_per_req, topk,
                           budget_bytes, publish_blocks=None, consume_blocks=None):
        assert publish_blocks is None or consume_blocks is None
        device = compress_lens.device
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        spans = S.request_spans(q_lens_cpu)
        published = None
        columns = None
        if publish_blocks is not None:
            published = [[] for _ in spans]
            columns = torch.arange(width, device=device)
        tiles = S.row_tiles(num_tokens, width, S.FP32_BYTES, budget_bytes)
        fixed = pool is not None and pool.active((tiles[0].stop - tiles[0].start) * width * S.FP32_BYTES)
        if fixed:
            pool.reserve(device, budget_bytes)
        for rows in tiles:
            if fixed:
                with pool.scope(device):  # the tile only: nothing that outlives it is allocated in the pool
                    logits = score_rows(rows)
                pool.note(logits)
            else:
                logits = score_rows(rows)
            for b, (r0, r1) in enumerate(spans):
                lo, hi = max(r0, rows.start), min(r1, rows.stop)
                lc = lc_per_req[b]
                if lo >= hi or lc == 0:
                    continue
                scores = logits[lo - rows.start : hi - rows.start, :lc]
                if consume_blocks is not None:
                    ids, block_size = consume_blocks
                    mask_blocks_(scores, ids[b][lo - r0 : hi - r0], block_size)
                elif publish_blocks is not None:
                    topk_blocks, block_size = publish_blocks
                    lens = compress_lens[lo:hi, None]
                    scores.masked_fill_(columns[None, :lc] >= lens, -torch.inf)
                    published[b].append(S.select_candidate_block_ids(scores, lens, topk_blocks, block_size))
                scores = None
            out = selected[rows]
            topk_rows(logits, rows, out)
            if consume_blocks is not None:
                out.copy_(S.mask_topk_scores(logits, out, ks[rows]))
            logits = out = None  # one tile alive at a time: freed (stream-ordered) before the next tile is scored
        if published is None:
            return selected, None
        empty = torch.zeros(0, 0, dtype=torch.int32, device=device)
        return selected, [torch.cat(parts) if len(parts) > 1 else (parts[0] if parts else empty) for parts in published]

    dense_indexer_topk.lowmem = True
    return dense_indexer_topk


def install():
    from sglang.srt.layers.attention import deepseek_v4_backend as B
    from sglang.srt.layers.attention.dsv4 import dsv41_indexer_select as S

    if getattr(B.dense_indexer_topk, "lowmem", False):
        return
    if getattr(B.dense_indexer_topk, "rowsplit_inner", None) is not None:
        # replacing the wrapper would silently drop the row split: install order is lowmem, then rowsplit
        raise RuntimeError("idx_lowmem.install() after idx_rowsplit.install(); install idx_lowmem first")
    B.dense_indexer_topk = make(S, POOL)
    if POOL is not None:
        print(f"[split-nv] idx-lowmem fixed-size tiles: private pool, cap = budget x {1 + POOL.margin:g}, "
              f"tiles >= {POOL.min_bytes >> 20} MiB (live flag idx_fixed_tiles)", flush=True)
