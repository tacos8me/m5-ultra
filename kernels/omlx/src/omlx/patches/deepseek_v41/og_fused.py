# SPDX-License-Identifier: MIT
"""One Mac pass over layers 20-39 + head for several og requests' verify rows.

Two (or more) requests' boundaries run as one row block: the dense weights
(MXFP8 attention/shared-expert projections, the BF16 wo_a and the BF16 head) are
read once for all rows, and the routed experts run as one MXFP4 pair launch.
Attention, the indexer, the router and every cache stay request-local.

Each request's rows get exactly its own forward_boundary arithmetic: the
projection kernels used here (fast_qmv rows, grouped_gemv, head rows, the MoE
pair kernels, mHC and combine) compute every row independently with the same
lanes, order and reductions for any row count, so a row's result does not
depend on how many rows share the launch. Requests of one row keep their own
path (MLX selects different M=1 kernels), see ``eligible``.
"""

import os

import mlx.core as mx

from . import attn_fusions, decode_fusions, fast_qmv, head, hc_fuse, moe_decode
from .activation import quantize_swiglu_activation
from .quantization import QuantizedProjection, quantize_activation

MAX_ROWS = 16
ENABLED = os.environ.get("DS41_OG_FUSE", "1") == "1"


def _rows(p, x):
    """fast_qmv's rows kernel for any M (per-row arithmetic independent of M)."""
    k, n, m = x.shape[-1], p.weight.shape[0], x.shape[1]
    r = fast_qmv.ROWS_PER_LANE
    return fast_qmv._kernel()(
        inputs=[p.weight, p.scales, x],
        template=[("T", x.dtype), ("K", k), ("N", n), ("M", m), ("R", r)],
        grid=(32, (n + 4 * r - 1) // (4 * r) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(1, m, n)],
        output_dtypes=[x.dtype],
    )[0]


def _mxfp8(p):
    return isinstance(p, QuantizedProjection) and p.quantize_input and fast_qmv._static_ok(p)[1]


def unsupported(lm):
    """Why this model cannot take the fused path (None = it can): the kernels it replicates must be on."""
    from . import language
    flags = dict(DS41_HC_FUSE=hc_fuse.DS41_HC_FUSE, DS41_DECODE_KERNELS_V2=decode_fusions.DS41_DECODE_KERNELS_V2,
                 DS41_MHC=language.DS41_MHC, DS41_ATTN_FUSE=attn_fusions.ENABLED,
                 DS41_FAST_ROPE=language.DS41_FAST_ROPE, DS41_MOE_FUSED=moe_decode.ENABLED,
                 DS41_FAST_QMV=fast_qmv.ENABLED, DS41_HEAD_ROWS=head.DS41_HEAD_ROWS)
    off = [k for k, v in flags.items() if not v]
    if off:
        return "off: " + ",".join(off)
    w = getattr(lm.head, "weight", None)
    if w is None or hasattr(lm.head, "bits") or w.dtype != mx.bfloat16:
        return "head is not BF16"
    for i in range(20, 40):
        layer = lm.layers[i]
        a, f = layer.attn, layer.ffn
        if not all(_mxfp8(p) for p in (a.wq_a, a.wkv, a.wq_b, a.wo_b, f.shared_experts.w1,
                                       f.shared_experts.w3, f.shared_experts.w2)):
            return f"layer {i}: projections are not MXFP8"
        if a.wo_a.weight.dtype != mx.bfloat16 or not f.experts.quantizes_input or "engram" in layer:
            return f"layer {i}: unexpected wo_a/experts/engram"
    return None


def _static_ok(lm):
    cached = lm.__dict__.get("_og_fused_ok")
    if cached is None:
        reason = unsupported(lm)
        if reason is not None:
            import logging
            logging.getLogger(__name__).warning("og_fused disabled: %s", reason)
        cached = lm.__dict__["_og_fused_ok"] = reason is None
    return cached


def eligible(lm, lengths):
    from .language import verify_tile
    return (
        ENABLED
        and len(lengths) > 1
        and all(2 <= n <= 5 for n in lengths)
        and sum(lengths) <= MAX_ROWS
        and not verify_tile()
        and mx.default_device() == mx.gpu
        and _static_ok(lm)
    )


def _attention(attn, x, bounds, caches, shareds, starts, prebuilt):
    c = attn._config
    xq = quantize_activation(x)
    query, kv_input = _rows(attn.wq_a, xq), _rows(attn.wkv, xq)
    qrs, qr8s = [], []
    for b, e in bounds:
        qr, qr8 = attn_fusions.rms_quant(query[:, b:e], attn.q_norm.weight, attn.q_norm.eps, True)
        qrs.append(qr)
        qr8s.append(qr8)
    q_input = _rows(attn.wq_b, mx.concatenate(qr8s, 1))
    heads = []
    for s, (b, e) in enumerate(bounds):
        # The indexer's wq_b reuses this FP8 round trip, as in Attention.__call__.
        shareds[s]["qr8"] = (qrs[s], qr8s[s])
        heads.append(attn(x[:, b:e], caches[s], shareds[s], starts[s],
                          projections=(qrs[s], kv_input[:, b:e], q_input[:, b:e]),
                          prebuilt_end=prebuilt[s], return_heads=True))
    out = mx.concatenate(heads, 1)
    grouped = out.reshape(1, out.shape[1], c.o_groups, -1)
    weight = attn.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
    projected = decode_fusions.grouped_gemv(grouped, weight).flatten(-2)
    return _rows(attn.wo_b, quantize_activation(projected))


def _moe(moe, x, bounds):
    routes = [moe.gate(x[:, b:e], None) for b, e in bounds]
    idx = mx.concatenate([r[0] for r in routes], 1)
    weights = mx.concatenate([r[1] for r in routes], 1)
    xq = quantize_activation(x)
    routed = moe.experts(xq[..., None, None, :], idx, weights, input_quantized=True,
                         max_grouped_tokens=MAX_ROWS).squeeze(-2)
    sh = moe.shared_experts
    y = quantize_swiglu_activation(_rows(sh.w1, xq), _rows(sh.w3, xq), None, x.dtype, sh._limit)
    shared = _rows(sh.w2, y)
    return moe_decode._combine_kernel()(
        inputs=[routed, shared],
        template=[("T", shared.dtype), ("D", shared.shape[-1]), ("TOPK", routed.shape[-2])],
        grid=(shared.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[shared.dtype],
    )[0]


def _block(layer, h, pre, bounds, caches, shareds, starts, prebuilt, first):
    """hc_fuse.block_forward over all requests' rows (mHC kernels are per row)."""
    c = layer._config
    mix_a, x = hc_fuse.project_pre_norm(h, pre, layer.hc_attn_fn, layer.attn_norm.weight,
                                        c.norm_eps, layer.attn_norm.eps)
    a = _attention(layer.attn, x, bounds, caches, shareds, starts, prebuilt)
    if first and hc_fuse.EARLY_SUBMIT:
        mx.async_eval(a)
    h, ap = hc_fuse.post_mix(a, h, mix_a, layer.hc_attn_scale, layer.hc_attn_base, c.hc_eps,
                             c.hc_sinkhorn_iters)
    mix_f, x = hc_fuse.project_pre_norm(h, ap, layer.hc_ffn_fn, layer.ffn_norm.weight,
                                        c.norm_eps, layer.ffn_norm.eps)
    f = _moe(layer.ffn, x, bounds)
    return hc_fuse.post_mix(f, h, mix_f, layer.hc_ffn_scale, layer.hc_ffn_base, c.hc_eps,
                            c.hc_sinkhorn_iters)


def _head_rows(weight, x):
    """head.project_logits' BF16 rows kernel for any row count (per-row arithmetic independent of M)."""
    rows, width = weight.shape
    lanes = 16
    return head._rows_kernel()(
        inputs=[x.astype(mx.float32), weight],
        template=[("N", rows), ("K", width), ("KL", lanes), ("M", x.shape[1])],
        grid=(((rows * lanes + 63) // 64) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.float32],
    )[0]


def _logits(lm, h, pre, bounds):
    from .language import hc_pre
    x = mx.concatenate([lm.norm(hc_pre(h[:, b:e], pre[:, b:e])) for b, e in bounds], 1)
    out = _head_rows(lm.head.weight, x)
    return [out[:, b:e] for b, e in bounds]


def forward(lm, items, capture=True):
    """items: dicts with h, pre, kv, index, cache, start, verify_states (forward_boundary's
    arguments, verify=True). Returns [(logits, hidden)] in item order; each request's
    caches, verify stash and outputs are those of its own forward_boundary call."""
    from .encoder_replay import empty_slot

    c = lm._config
    opened = [lm._open_boundary(it["h"], it["pre"], it["cache"], it["start"], it.get("kv"),
                                it.get("index"), True, it.get("verify_states")) for it in items]
    caches = [it["cache"] for it in items]
    starts = [it["start"] for it in items]
    lengths = [o[0] for o in opened]
    bounds, begin = [], 0
    for n in lengths:
        bounds.append((begin, begin + n))
        begin += n
    prebuilt = [s + n if it.get("kv") is not None else None for it, s, n in zip(items, starts, lengths)]
    h = mx.concatenate([it["h"] for it in items], 1)
    pre = mx.concatenate([it["pre"] for it in items], 1)
    shareds = [{} for _ in items]
    captured = [{} for _ in items]
    for i in range(20, 40):
        for cache, (_, _, states) in zip(caches, opened):
            cache[i]._mtp_verify_state = states[i]
        if capture and i in c.dspark_target_layer_ids:
            for s, (b, e) in enumerate(bounds):
                captured[s][i] = mx.mean(h[:, b:e], axis=2)
        h, pre = _block(lm.layers[i], h, pre, bounds, [cache[i] for cache in caches], shareds,
                        starts, [p if i == 20 else None for p in prebuilt], i == 20)
        for cache, start, n in zip(caches, starts, lengths):
            cache[i][0] = mx.array([start + n], mx.int32)
            for slot in range(1, 7):
                if cache[i][slot] is None:
                    cache[i][slot] = empty_slot(c, slot, h.dtype)
            del cache[i]._mtp_verify_state
        if lm._async_every and ((i - 19) % lm._async_every == 0 or (i - 19) in lm._async_front):
            mx.async_eval(h, pre)
    for cache, start, n in zip(caches, starts, lengths):
        for i in range(20):
            cache[i][0] = mx.array([start + n], mx.int32)
    logits = _logits(lm, h, pre, bounds)
    out = []
    for s, (cache, start, n, (_, snapshots, states)) in enumerate(zip(caches, starts, lengths, opened)):
        hidden = (mx.concatenate([captured[s][i] for i in c.dspark_target_layer_ids], -1)
                  if capture else None)
        cache[0]._pipe1_verify = (start, n, snapshots, states)
        out.append((logits[s], hidden))
    return out
