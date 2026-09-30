"""CPU test: split_nv.idx_lowmem.dense_indexer_topk == sglang's dense_indexer_topk, bitwise (selections, published
block ids, and the logits every top-k call saw), and it keeps one logits tile alive instead of two.
Fixed-size tiles (SPLIT_NV_IDX_FIXED_TILES): the same comparison through a recording pool, which also checks that the
pool is reserved (at >= every tile's bytes) before a call's first tile, that only score_rows runs inside the pool scope
(top-k, masks, selections and published ids never land in the pool), and TilePool's cap / over-cap arithmetic.
usage: CUDA_VISIBLE_DEVICES= python tools/fair/test_idx_lowmem.py [path to dsv41_indexer_select.py]"""
import contextlib
import gc
import importlib.util
import os
import sys
import tempfile
import threading
import weakref

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hooks"))
os.environ["SPLIT_NV_DIR"] = tempfile.mkdtemp(prefix="idx-lowmem-test-")  # never the live box-perf-flags.json
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "sglang", "sglang", "srt", "layers", "attention", "dsv4",
                                                         "dsv41_indexer_select.py")
spec = importlib.util.spec_from_file_location("dsv41_indexer_select", SRC)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from split_nv import idx_lowmem  # noqa: E402

NEW = idx_lowmem.make(S)


class RecordingPool:
    """CPU stand-in for idx_lowmem.TilePool: records reservations, and which callbacks ran inside the tile scope."""

    def __init__(self):
        self.local = threading.local()
        self.cap = 0
        self.tiles = 0
        self.violations = []

    def active(self, tile_bytes):
        return True

    def inside(self):
        return getattr(self.local, "inside", False)

    @contextlib.contextmanager
    def scope(self, device):
        self.local.inside = True
        try:
            yield
        finally:
            self.local.inside = False

    def reserve(self, device, budget_bytes):
        self.cap = max(self.cap, budget_bytes)
        self.local.reserved = True

    def note(self, logits):
        nb = logits.untyped_storage().nbytes()
        if not getattr(self.local, "reserved", False):
            self.violations.append("tile before any reservation")
        if nb > self.cap:
            self.violations.append(f"tile {nb} B > reserved cap {self.cap} B")
        self.tiles += 1


POOL = RecordingPool()
FIXED = idx_lowmem.make(S, POOL)


def case(seed, q_lens, lcs, width, topk, budget_rows, mode, block_size=64, topk_blocks=4, ties=False):
    g = torch.Generator().manual_seed(seed)
    n = sum(q_lens)
    full = torch.randn(n, width, generator=g)
    if ties:
        full = (full * 4).round() / 4
    full[torch.rand(n, width, generator=g) < 0.02] = -torch.inf
    # per-row reachable positions (causal inside each request)
    comp = []
    for b, q in enumerate(q_lens):
        lc = lcs[b]
        comp += [max(1, min(lc, lc - q + 1 + i)) if lc else 0 for i in range(q)]
    compress_lens = torch.tensor(comp, dtype=torch.int32)
    ks = torch.tensor(sum(([sum(lcs[:b])] * q for b, q in enumerate(q_lens)), []), dtype=torch.int32)
    consume = None
    if mode == "consume":
        consume = ([torch.randint(-1, max(1, -(-lc // block_size)), (q, topk_blocks), generator=g, dtype=torch.int32)
                    for q, lc in zip(q_lens, lcs)], block_size)
    publish = (topk_blocks, block_size) if mode == "publish" else None

    def run(fn, pool=None):
        seen, alive, live_max = [], [], [0]

        def score_rows(rows):
            live_max[0] = max(live_max[0], sum(1 for r in alive if r() is not None) + 1)
            if pool is not None and not pool.inside():
                pool.violations.append("score_rows outside the tile pool")
            t = full[rows].clone()
            alive.append(weakref.ref(t))
            return t

        def topk_rows(logits, rows, out):
            if pool is not None and pool.inside():
                pool.violations.append("topk_rows inside the tile pool")
            seen.append(logits.clone())
            lg = logits.clone()
            cl = compress_lens[rows].long()
            lg[torch.arange(lg.shape[1])[None, :] >= cl[:, None]] = -torch.inf
            v, i = lg.topk(min(topk, lg.shape[1]), dim=-1)
            i = torch.where(v > -torch.inf, i + ks[rows].long()[:, None], torch.full_like(i, -1))
            out.fill_(-1)
            out[:, :i.shape[1]] = i.to(out.dtype)

        sel, pub = fn(score_rows=score_rows, topk_rows=topk_rows, num_tokens=n, width=width, compress_lens=compress_lens,
                      ks=ks, q_lens_cpu=q_lens, lc_per_req=lcs, topk=topk, budget_bytes=budget_rows * width * 4,
                      publish_blocks=publish, consume_blocks=consume)
        gc.collect()
        return sel, pub, seen, live_max[0]

    a = run(S.dense_indexer_topk)
    b = run(NEW)
    c = run(FIXED, POOL)
    eq = True
    for o in (b, c):
        eq = eq and torch.equal(a[0], o[0]) and len(a[2]) == len(o[2]) and all(
            torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a[2], o[2]))
        if a[1] is not None:
            eq = eq and len(a[1]) == len(o[1]) and all(torch.equal(x, y) for x, y in zip(a[1], o[1]))
    return ((eq and c[3] == 1) if len(a[2]) > 1 else eq), a[3], b[3], len(a[2])


cases = 0
fails = 0
peaks = set()
for seed in range(40):
    for mode in ("plain", "consume", "publish"):
        for q_lens, lcs, width in (([37], [300], 300), ([64, 33], [257, 190], 260), ([128], [1000], 1000),
                                   ([5, 7, 0, 11], [70, 0, 33, 129], 132)):
            for budget_rows in (1, 3, 16, 1000):
                for bs in (64, 16, 7):
                    eq, pa, pb, tiles = case(seed, q_lens, lcs, width, 32, budget_rows, mode, block_size=bs,
                                             ties=seed % 2 == 0)
                    cases += 1
                    fails += not eq
                    if tiles > 1:
                        peaks.add((pa, pb))
print(f"{cases} cases, {fails} mismatches; live logits tiles at a score call (orig, lowmem): {sorted(peaks)}")
print(f"fixed tiles: {POOL.tiles} tiles through the recording pool, violations {POOL.violations[:3]}")
assert fails == 0 and all(pb == 1 for _, pb in peaks)
assert POOL.tiles > 0 and not POOL.violations

# TilePool arithmetic (no CUDA needed): cap = budget x (1 + margin) rounded to 2 MiB; an over-cap tile raises the cap
def tile_of(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8, device="meta")  # storage size only, no memory


tp = idx_lowmem.TilePool(margin=0.125, min_bytes=64 << 20)
tp.cap = max(tp.cap, idx_lowmem._round((1 << 30) * 1.125))
assert tp.cap == 1152 << 20, tp.cap
tp.note(tile_of(1080033280))  # the largest tile request seen in the 1M OOM-retries
assert tp.stats["over_cap"] == 0 and tp.stats["max_tile_bytes"] == 1080033280
tp.note(tile_of(1300 << 20))
assert tp.stats["over_cap"] == 1 and tp.cap == idx_lowmem._round((1300 << 20) * 1.125), tp.cap
assert not tp.active(32 << 20) and tp.active(512 << 20)
print("PASS")
