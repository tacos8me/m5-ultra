"""DS-V4.1 sparse MLA with the official arithmetic (inference/kernel.py sparse_attn) on SGLang's FlashMLA caches.

One program per (query token, 16-head group): the window and compressed index lists are concatenated
(window first, in metadata order) and consumed in blocks of 64 keys, as the official kernel does.
BF16 Q, FP32 scores, online softmax in FP32, BF16 probabilities for the PV product, FP32 accumulation,
attention sink added to the denominator at the end. Each query row is computed independently of the
batch, so a speculative step and a full prefill produce identical bytes for the same row.

Cache rows are read exactly: FP8 nope x UE8M0 64-tile scale and BF16 RoPE tail, dequantized to BF16
(exact) before the tensor-core products.
"""
import torch
import triton
import triton.language as tl

NOPE, ROPE, STRIDE = 448, 64, 576


@triton.jit
def _load_kv(cache_u8, cache_f8, cache_bf, idx, valid, page_size, page_bytes, offs_d, offs_r):
    safe = tl.where(valid, idx, 0)
    page = (safe // page_size).to(tl.int64)
    off = (safe % page_size).to(tl.int64)
    base = page * page_bytes + off * 576
    dmask = valid[:, None] & (offs_d[None, :] < 448)
    nope = tl.load(cache_f8 + base[:, None] + offs_d[None, :], mask=dmask, other=0.0).to(tl.float32)
    sbase = page * page_bytes + page_size * 576 + off * 8
    sc = tl.load(cache_u8 + sbase[:, None] + (offs_d // 64)[None, :], mask=dmask, other=127)
    nope = nope * tl.math.exp2(sc.to(tl.float32) - 127.0)
    rope = tl.load(cache_bf + (base + 448)[:, None] // 2 + offs_r[None, :], mask=valid[:, None], other=0.0)
    return nope.to(tl.bfloat16), rope


@triton.jit
def _og_sparse_mla(
    Q, O, SINK,
    SWA_U8, SWA_F8, SWA_BF, SWA_IDX, SWA_LEN,
    EXT_U8, EXT_F8, EXT_BF, EXT_IDX, EXT_LEN,
    scale, swa_ps, swa_pb, ext_ps, ext_pb,
    s_qt, s_qh, s_ot, s_oh, s_si, s_ei,
    NW: tl.constexpr, NE: tl.constexpr, HB: tl.constexpr, BN: tl.constexpr,
):
    t = tl.program_id(0)
    h0 = tl.program_id(1) * HB
    offs_h = h0 + tl.arange(0, HB)
    offs_d = tl.arange(0, 512)
    offs_r = tl.arange(0, 64)
    offs_n = tl.arange(0, BN)
    qp = Q + t.to(tl.int64) * s_qt + offs_h[:, None] * s_qh
    q_nope = tl.load(qp + offs_d[None, :], mask=(offs_d[None, :] < 448), other=0.0)
    q_rope = tl.load(qp + 448 + offs_r[None, :])
    m = tl.full([HB], -1e30, tl.float32)
    l = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, 512], tl.float32)
    acc_r = tl.zeros([HB, 64], tl.float32)
    swa_len = tl.load(SWA_LEN + t)
    for b in range(0, NW, BN):
        j = b + offs_n
        idx = tl.load(SWA_IDX + t.to(tl.int64) * s_si + j)
        valid = (idx >= 0) & (j < swa_len)
        kv_n, kv_r = _load_kv(SWA_U8, SWA_F8, SWA_BF, idx, valid, swa_ps, swa_pb, offs_d, offs_r)
        s = tl.dot(q_nope, tl.trans(kv_n)) + tl.dot(q_rope, tl.trans(kv_r))
        s = tl.where(valid[None, :], s * scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(pb, kv_n)
        acc_r = acc_r * alpha[:, None] + tl.dot(pb, kv_r)
        m = m_new
    if NE > 0:
        ext_len = tl.load(EXT_LEN + t)
        for b in range(0, NE, BN):
            j = b + offs_n
            idx = tl.load(EXT_IDX + t.to(tl.int64) * s_ei + j)
            valid = (idx >= 0) & (j < ext_len)
            kv_n, kv_r = _load_kv(EXT_U8, EXT_F8, EXT_BF, idx, valid, ext_ps, ext_pb, offs_d, offs_r)
            s = tl.dot(q_nope, tl.trans(kv_n)) + tl.dot(q_rope, tl.trans(kv_r))
            s = tl.where(valid[None, :], s * scale, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, 1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new[:, None])
            l = l * alpha + tl.sum(p, 1)
            pb = p.to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(pb, kv_n)
            acc_r = acc_r * alpha[:, None] + tl.dot(pb, kv_r)
            m = m_new
    l = l + tl.exp(tl.load(SINK + offs_h) - m)
    op = O + t.to(tl.int64) * s_ot + offs_h[:, None] * s_oh
    tl.store(op + offs_d[None, :], (acc / l[:, None]).to(tl.bfloat16), mask=(offs_d[None, :] < 448))
    tl.store(op + 448 + offs_r[None, :], (acc_r / l[:, None]).to(tl.bfloat16))


def _views(cache):
    pages = cache.shape[0]
    flat = cache.as_strided((pages * cache.stride(0),), (1,)).view(torch.uint8)
    return flat, flat.view(torch.float8_e4m3fn), flat.view(torch.bfloat16), cache.shape[1], cache.stride(0)


def _canonical(idx, length):
    """Keep the metadata order (logical order in prefill and in steps); entries >= length are skipped."""
    T, N = idx.shape
    if length is None:
        length = torch.full((T,), N, dtype=torch.int32, device=idx.device)
    return idx.to(torch.int32).contiguous(), length.to(torch.int32).contiguous()


def sparse_mla(q, k_cache, indices, topk_length, attn_sink, softmax_scale,
               extra_k_cache=None, extra_indices=None, extra_topk_length=None):
    """q [T, 1, H, 512] bf16 -> [T, 1, H, 512] bf16. Same arguments as flash_mla_with_kvcache_sm120."""
    q3 = q.reshape(q.shape[0], q.shape[-2], q.shape[-1])
    T, H, D = q3.shape
    assert D == 512 and H % 16 == 0
    widx, wlen = _canonical(indices.reshape(T, -1), topk_length)
    NW = widx.shape[1]
    if extra_k_cache is not None and extra_indices is not None:
        eidx, elen = _canonical(extra_indices.reshape(T, -1), extra_topk_length)
        ext = _views(extra_k_cache)
    else:
        eidx, elen, ext = widx, wlen, _views(k_cache)
        eidx = eidx[:, :0]
    NE = eidx.shape[1]
    assert NW % 64 == 0 and NE % 64 == 0
    swa = _views(k_cache)
    out = torch.empty((T, H, D), dtype=torch.bfloat16, device=q.device)
    sink = attn_sink.float().contiguous()
    if T:
        _og_sparse_mla[(T, H // 16)](
            q3, out, sink,
            swa[0], swa[1], swa[2], widx, wlen,
            ext[0], ext[1], ext[2], eidx if NE else widx, elen,
            float(softmax_scale), swa[3], swa[4], ext[3], ext[4],
            q3.stride(0), q3.stride(1), out.stride(0), out.stride(1), widx.stride(0), (eidx if NE else widx).stride(0),
            NW=NW, NE=NE, HB=16, BN=64, num_warps=4, num_stages=2)
    return out.unsqueeze(1)


def install():
    from sglang.kernels.ops.attention import flash_mla_sm120 as A

    def attention(**kw):
        return sparse_mla(kw["q"], kw["k_cache"], kw["indices"], kw.get("topk_length"), kw["attn_sink"],
                          kw.get("softmax_scale") or kw["q"].shape[-1] ** -0.5,
                          kw.get("extra_k_cache"), kw.get("extra_indices_in_kvcache"),
                          kw.get("extra_topk_length")), None

    A.flash_mla_with_kvcache_sm120 = attention
