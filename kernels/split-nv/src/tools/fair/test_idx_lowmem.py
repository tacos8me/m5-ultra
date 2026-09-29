"""CPU test: split_nv.idx_lowmem.dense_indexer_topk == sglang's dense_indexer_topk, bitwise (selections, published
block ids, and the logits every top-k call saw), and it keeps one logits tile alive instead of two.
usage: CUDA_VISIBLE_DEVICES= python tools/fair/test_idx_lowmem.py [path to dsv41_indexer_select.py]"""
import gc
import importlib.util
import os
import sys
import weakref

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hooks"))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "sglang", "sglang", "srt", "layers", "attention", "dsv4",
                                                         "dsv41_indexer_select.py")
spec = importlib.util.spec_from_file_location("dsv41_indexer_select", SRC)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from split_nv import idx_lowmem  # noqa: E402

NEW = idx_lowmem.make(S)


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

    def run(fn):
        seen, alive, live_max = [], [], [0]

        def score_rows(rows):
            live_max[0] = max(live_max[0], sum(1 for r in alive if r() is not None) + 1)
            t = full[rows].clone()
            alive.append(weakref.ref(t))
            return t

        def topk_rows(logits, rows, out):
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
    eq = torch.equal(a[0], b[0]) and len(a[2]) == len(b[2]) and all(
        torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a[2], b[2]))
    if a[1] is not None:
        eq = eq and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
    return eq, a[3], b[3], len(a[2])


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
assert fails == 0 and all(pb == 1 for _, pb in peaks)
print("PASS")
