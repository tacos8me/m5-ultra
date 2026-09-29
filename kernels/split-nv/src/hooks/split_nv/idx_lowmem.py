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
"""
import torch


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


def make(S):
    """The low-memory dense_indexer_topk over module S (sglang's dsv41_indexer_select)."""

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
        for rows in S.row_tiles(num_tokens, width, S.FP32_BYTES, budget_bytes):
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
    B.dense_indexer_topk = make(S)
