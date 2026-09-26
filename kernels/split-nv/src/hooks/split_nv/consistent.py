"""Phase-3 arithmetic shared by prefill and speculative steps.

The original checkpoint is unchanged. Low precision attention and MoE amplify
even tiny changes in projection reductions, so batch size must not choose a
different reduction or a different ordering of sparse keys.
"""
import torch


def install():
    from sglang.srt.layers.attention import deepseek_v4_backend as B
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse as S
    from sglang.srt.models.deepseek_v2 import MoEGate
    from sglang.kernels.ops.attention import flash_mla_sm120 as A
    from split_nv.fixed_linear import linear

    if getattr(B, '_pipe1_consistent', False):
        return
    B._pipe1_consistent = True

    original_topk = B.topk_transform_paged_v2

    def sorted_topk(scores, lens, pages, out, page_size, metadata):
        # Prefill sorts logical positions. Atomic decode top-k output does not;
        # its order can change even when replaying an identical CUDA graph.
        raw = torch.empty_like(out)
        original_topk(scores, lens, None, raw, page_size, metadata)
        sentinel = torch.iinfo(raw.dtype).max
        raw = raw.masked_fill(raw < 0, sentinel).sort(-1).values
        valid = raw != sentinel
        safe = raw.masked_fill(~valid, 0).long()
        slots = safe if pages is None else pages.gather(1, safe // page_size) * page_size + safe % page_size
        out.copy_(torch.where(valid, slots, -1))

    B.topk_transform_paged_v2 = sorted_topk
    MoEGate.forward = lambda self, x, *a, **kw: linear(x, self.weight)
    S.linear_bf16_fp32 = linear
    original_project = S.DeepseekV41Compressor.project

    def project(self, x):
        if self.compress_ratio == 1:
            return linear(x, self.wkv.weight).to(x.dtype), None
        return original_project(self, x)

    S.DeepseekV41Compressor.project = project
    S.DeepseekV41Indexer.head_weights_raw = lambda self, x: linear(x, self.weights_proj.weight).to(x.dtype)
    S.DeepseekV41Indexer.forward_wk = lambda self, x: linear(x, self.wk.weight).to(x.dtype)

    original_attention = A.flash_mla_with_kvcache_sm120

    def attention(*args, **kwargs):
        # FlashInfer switches at 64 rows from split-K decode to prefill; those
        # kernels round differently. Dummy zero-length rows force the same
        # prefill kernel without adding keys or changing a real query's mask.
        n = kwargs['q'].shape[0]
        if n <= 64:
            for name in ('q', 'indices', 'topk_length', 'extra_indices_in_kvcache', 'extra_topk_length'):
                x = kwargs.get(name)
                if x is not None:
                    value = -1 if 'indices' in name else 0
                    kwargs[name] = torch.cat((x, x.new_full((65 - n, *x.shape[1:]), value)))
        out, aux = original_attention(*args, **kwargs)
        return out[:n], aux

    A.flash_mla_with_kvcache_sm120 = attention

    def rms_norm(self, x):
        # The dsv41 RMSNorm (compressor norm, indexer k_norm) used a Triton kernel for <= 64 rows and this torch
        # reduction above: they round differently in ~0.1% of rows, so a pooled row could differ between a 41-row
        # and a 78-row chunk, or between STEP and prefill. The torch path is row-invariant at every M (and in
        # graphs), and it is what every prefill chunk > 64 rows already used, so exported states are unchanged.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)

    S.RMSNorm.forward = rms_norm
