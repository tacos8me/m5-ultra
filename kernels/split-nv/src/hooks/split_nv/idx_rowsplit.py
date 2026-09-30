"""Split the replicated prefill indexer's rows across the two TP ranks (SPLIT_NV_IDX_ROWSPLIT=1, default off).

DeepseekV41Indexer is replicated (n_local_heads = n_heads; wq_b and weights_proj are ReplicatedLinear), so on layers
2/8/14 both ranks score every row of a prefill chunk against every compressed position and run the same top-k: at
1M tokens each rank spends ~10.5 s on work the other rank also does. The fp4 MQA logits kernel has no cross-row
reduction and the deterministic top-k (topk_v2_det) runs one CTA per row, so a row's logits and selections do not
depend on which other rows share the call (the 256 -> 1024 MB tiling change, the pf_overlap halves and the 2048-row
contended pieces all left the 1M reference byte-identical). Here each rank scores and reduces its contiguous half of
the rows through the installed dense_indexer_topk, and one all_gather_into_tensor of the int32 [rows, topk]
selections gives both ranks the full result; the backend then builds page_indices / raw_indices from it on both ranks
exactly as before. Queries and head weights are still computed for all rows (width-independent, small).

Collective safety -- the all-gather must be issued on both ranks or on neither, so every input to that decision is
identical on both ranks:
  * the per-chunk switch travels in the prefill_chunk command: rank 0 reads SPLIT_NV_IDX_ROWSPLIT and the box-perf
    flag `idx_rowsplit` (default on) once per chunk (decide), every rank enters chunk(on) from the command;
  * the TP size, the call's row count (a tensor shape) and its width (from seq_lens_cpu), and publish/consume (fixed
    per layer: layers 2/8/14 on the box do neither; layer 20 runs with run_indexer=False).
The gather goes to the TP torch.distributed NCCL group, the group and stream pf_overlap's asynchronous all-reduces use,
so it queues behind the other half's in-flight all-reduce in the same order on both ranks. It is synchronous: the
compute stream waits for it before anything after it runs. It is NOT routed through the pf_overlap baton (no hand-over
while it is in flight, so no new interleave point inside attention) and is never in flight at a preempt point (those
sit between layers; preempt.fence only knows the all-reduces). Steps are TARGET_VERIFY and use the decode indexer, and
cmd_prefill (warm-up) never enters chunk(True), so neither can reach the split.

Order with SPLIT_NV_IDX_LOWMEM: this wraps whatever B.dense_indexer_topk is when install() runs, so idx_lowmem must be
installed first (engine.Engine.__init__ does): rowsplit(lowmem(rows)), each rank running the low-memory tiling on its
own half at the same tile shape. idx_lowmem.install() refuses to run over this wrapper (it would replace it).

Bitwise: tools/fair/test_idx_rowsplit.py (both halves through sglang's and idx_lowmem's function, gathered == the full
call, including tie-heavy rows, multi-request chunks and uneven halves).
"""
import contextlib
import os

import torch

ALIGN = 4  # the fp4 MQA logits kernel scores rows in q-blocks of 4
MIN_WIDTH = int(os.environ.get("SPLIT_NV_IDX_ROWSPLIT_MIN_WIDTH", "32768"))  # compressed columns (ratio 2: 64K tokens)
MIN_ROWS = int(os.environ.get("SPLIT_NV_IDX_ROWSPLIT_MIN_ROWS", "1024"))  # <= 752 rows is one MQA wave anyway

_on = False  # the running prefill chunk's switch, set on every rank from its command (chunk)
stats = {"split_calls": 0, "split_rows": 0}


def enabled():
    return os.environ.get("SPLIT_NV_IDX_ROWSPLIT", "0") == "1"


def decide():
    """Rank 0, once per prefill_chunk command: the row split for that chunk (sent to every rank in the command)."""
    if not enabled():
        return False
    from split_nv.perf_flags import flag

    return bool(flag("idx_rowsplit", True))


@contextlib.contextmanager
def chunk(on):
    """Every rank, around one prefill chunk's forward (both pf_overlap halves): the command's decision."""
    global _on
    _on = bool(on)
    try:
        yield
    finally:
        _on = False


def split_point(n):
    """Rows [0, h) go to rank 0 and [h, n) to rank 1: h = n // 2 rounded down to a multiple of ALIGN."""
    return (n // 2) // ALIGN * ALIGN


def sub_lens(q_lens_cpu, lo, hi):
    """Per-request row counts of rows [lo, hi) when requests own consecutive rows, q_lens_cpu[b] each."""
    out, start = [], 0
    for n in q_lens_cpu:
        out.append(max(0, min(start + n, hi) - max(start, lo)))
        start += n
    return out


def rows_topk(inner, kw, lo, hi):
    """inner's selections for rows [lo, hi) of the plain call `kw` (score_rows / topk_rows keep chunk row numbers)."""
    score_rows, topk_rows = kw["score_rows"], kw["topk_rows"]

    def shift(rows):
        return slice(rows.start + lo, rows.stop + lo)

    sel, _ = inner(score_rows=lambda rows: score_rows(shift(rows)),
                   topk_rows=lambda logits, rows, out: topk_rows(logits, shift(rows), out),
                   num_tokens=hi - lo, width=kw["width"], compress_lens=kw["compress_lens"][lo:hi], ks=kw["ks"][lo:hi],
                   q_lens_cpu=sub_lens(kw["q_lens_cpu"], lo, hi), lc_per_req=kw["lc_per_req"], topk=kw["topk"],
                   budget_bytes=kw["budget_bytes"])
    return sel


def gather_rows(sel, h, n, all_gather):
    """[n, topk] from rank 0's rows [0, h) and rank 1's [h, n): each rank sends m = n - h rows (rank 0 pads with -1)."""
    m = n - h
    if sel.shape[0] < m:
        pad = torch.full((m, sel.shape[1]), -1, dtype=sel.dtype, device=sel.device)
        pad[:sel.shape[0]] = sel
        sel = pad
    out = torch.empty((2 * m, sel.shape[1]), dtype=sel.dtype, device=sel.device)
    all_gather(out, sel.contiguous())
    return out if h == m else torch.cat((out[:h], out[m:]))


def _tp():
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


def _nccl_all_gather(group):
    def all_gather(out, inp):
        torch.distributed.all_gather_into_tensor(out, inp, group=group.device_group)
    return all_gather


def make(inner, tp=_tp, all_gather=None):
    """dense_indexer_topk that splits the plain calls of a split chunk across the two TP ranks, around `inner`.
    tp() -> the TP GroupCoordinator (world_size, rank_in_group, device_group); all_gather(out, inp) for tests."""

    def dense_indexer_topk(**kw):
        n = kw["num_tokens"]
        if not (_on and kw["width"] > MIN_WIDTH and n >= max(MIN_ROWS, 2 * ALIGN)
                and kw.get("publish_blocks") is None and kw.get("consume_blocks") is None):
            return inner(**kw)
        group = tp()
        if group.world_size != 2:
            return inner(**kw)
        h = split_point(n)
        lo, hi = (0, h) if group.rank_in_group == 0 else (h, n)
        sel = rows_topk(inner, kw, lo, hi)
        stats["split_calls"] += 1
        stats["split_rows"] += hi - lo
        return gather_rows(sel, h, n, all_gather or _nccl_all_gather(group)), None

    dense_indexer_topk.rowsplit_inner = inner
    return dense_indexer_topk


def wrap(B):
    """Wrap module B's dense_indexer_topk (idempotent)."""
    if getattr(B.dense_indexer_topk, "rowsplit_inner", None) is None:
        B.dense_indexer_topk = make(B.dense_indexer_topk)
    return B.dense_indexer_topk


def install():
    """After idx_lowmem.install() (if SPLIT_NV_IDX_LOWMEM=1): wraps the function the backend calls now."""
    from sglang.srt.layers.attention import deepseek_v4_backend as B

    fn = wrap(B)
    inner = fn.rowsplit_inner
    print(f"[split-nv] idx-rowsplit installed around {'idx_lowmem' if getattr(inner, 'lowmem', False) else 'sglang'}"
          f" dense_indexer_topk (width > {MIN_WIDTH}, rows >= {MIN_ROWS}; live flag idx_rowsplit)", flush=True)


def summary():
    return {"enabled": enabled(), **stats}
