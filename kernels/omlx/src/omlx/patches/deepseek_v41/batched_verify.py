# SPDX-License-Identifier: MIT
"""Ragged DSpark verification with independent arithmetic and cache commits.

Each request keeps its verification length, hidden capture and rollback state.
Routed experts share one call; MXFP8 projections share weights with the same
per-token reductions as singleton verification. Attention and router keep
request-local shapes.
"""

from contextlib import nullcontext

import mlx.core as mx

from .quantization import QuantizedProjection, quantize_activation
from .routed_batch import eligible
from .ragged_projection import project
from .activation import quantize_swiglu_activation


def routed_forward(moe, values):
    if not all(eligible(moe, x) for x in values):
        return [moe(x, None) for x in values]
    lengths = [x.shape[1] for x in values]
    routes = [moe.gate(x, None) for x in values]
    ids = mx.concatenate([r[0] for r in routes], axis=1)
    weights = mx.concatenate([r[1] for r in routes], axis=1)
    rq, sq = moe.experts.quantizes_input, moe.shared_experts.quantizes_input
    quantized = [quantize_activation(x) for x in values] if rq or sq else values
    shared_inputs = quantized if sq else values
    expert = moe.shared_experts
    if isinstance(expert.w2, QuantizedProjection) and expert.w2.quantize_input:
        gate, up = (
            project(expert.w1, shared_inputs, quantized=sq),
            project(expert.w3, shared_inputs, quantized=sq),
        )
        middle = [
            quantize_swiglu_activation(g, u, None, x.dtype, expert._limit)
            for g, u, x in zip(gate, up, values)
        ]
        shared = project(expert.w2, middle, quantized=True)
    else:
        shared = [expert(x, input_quantized=sq) for x in shared_inputs]
    joined = mx.concatenate(quantized if rq else values, axis=1)
    routed = moe.experts(
        joined[..., None, None, :],
        ids,
        weights,
        input_quantized=rq,
        max_grouped_tokens=32,
    ).squeeze(-2)
    results, begin = [], 0
    for x, sh, length in zip(values, shared, lengths):
        part = routed[:, begin : begin + length]
        results.append(
            (part.astype(mx.float32).sum(-2) + sh.astype(mx.float32)).astype(x.dtype)
        )
        begin += length
    return results


def forward(model, inputs, caches):
    from .language import (
        hc_mixes,
        hc_pre,
        hc_pre_norm,
        hc_post,
        project_logits,
        DS41_DECODE_ASYNC,
    )

    if not inputs or len(inputs) != len(caches):
        raise ValueError("DSpark verify needs one cache per request")
    c = model._config
    lengths = [ids.shape[1] for ids in inputs]
    if (
        any(ids.shape[0] != 1 or not 1 <= n <= 8 for ids, n in zip(inputs, lengths))
        or sum(lengths) > 32
    ):
        raise ValueError("DSpark ragged verify supports at most 32 unpadded token rows")
    starts = [cache[0].size() for cache in caches]
    snapshots = [
        [(list(item.cache), item.left_padding, item.lengths) for item in cache]
        for cache in caches
    ]
    rows = [
        [item.extract(0, offset=start) for item in cache]
        for cache, start in zip(caches, starts)
    ]
    states = [[{} for _ in cache] for cache in caches]
    for cache, start, state in zip(rows, starts, states):
        if any(item.size() != start for item in cache):
            raise ValueError("DSpark cache offsets diverged across layers")
        for item, undo in zip(cache, state):
            item._mtp_verify_state = undo
    hash_history = [
        (None, None) if model._hasher is None else model._hasher(ids, cache[0][6], None)
        for ids, cache in zip(inputs, rows)
    ]
    h = [mx.repeat(model.embed(ids)[..., None, :], c.hc_mult, -2) for ids in inputs]
    pre = [
        mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), x.shape[:-1])
        for x in h
    ]
    shared = [{} for _ in inputs]
    captured = [{} for _ in inputs]
    prefetch = getattr(model, "_engram_prefetch", None)
    for i, layer in enumerate(model.layers):
        values, mixes = [], []
        ax, amixes = [], []
        for row in range(len(inputs)):
            if "engram" in layer:
                ix = list(c.engram_layer_ids).index(i)
                with prefetch.forward() if prefetch is not None else nullcontext():
                    h[row] = layer.engram(h[row], hash_history[row][0][:, :, ix], None)
            if i in c.dspark_target_layer_ids:
                captured[row][i] = mx.mean(h[row], axis=2)
            ap, ao, ac = hc_mixes(
                h[row], layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, c
            )
            x = hc_pre_norm(
                h[row], pre[row], layer.attn_norm.weight, layer.attn_norm.eps
            )
            ax.append(x)
            amixes.append((ap, ao, ac))
        if all(
            isinstance(p, QuantizedProjection) and p.quantize_input
            for p in (layer.attn.wq_a, layer.attn.wkv)
        ):
            quantized = [quantize_activation(x) for x in ax]
            queries = project(layer.attn.wq_a, quantized, quantized=True)
            kv_inputs = project(layer.attn.wkv, quantized, quantized=True)
        else:
            pairs = [layer.attn._input_projections(x) for x in ax]
            queries, kv_inputs = [p[0] for p in pairs], [p[1] for p in pairs]
        qr = [layer.attn.q_norm(q) for q in queries]
        q_inputs = project(layer.attn.wq_b, qr)
        projected = [
            layer.attn(
                x,
                rows[row][i],
                shared[row],
                starts[row],
                projections=(qr[row], kv_inputs[row], q_inputs[row]),
                return_projected=True,
            )
            for row, x in enumerate(ax)
        ]
        attention = project(layer.attn.wo_b, projected)
        for row, (attn, (ap, ao, ac)) in enumerate(zip(attention, amixes)):
            h[row] = hc_post(attn, h[row], ao, ac)
            fp, fo, fc = hc_mixes(
                h[row], layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, c
            )
            values.append(
                hc_pre_norm(h[row], ap, layer.ffn_norm.weight, layer.ffn_norm.eps)
            )
            mixes.append((fp, fo, fc))
        outputs = routed_forward(layer.ffn, values)
        for row, (ffn, (fp, fo, fc)) in enumerate(zip(outputs, mixes)):
            h[row], pre[row] = hc_post(ffn, h[row], fo, fc), fp
            rows[row][i][0] = mx.array([starts[row] + lengths[row]], mx.int32)
            if i == 0 and hash_history[row][1] is not None:
                rows[row][i][6] = mx.array(hash_history[row][1], mx.int64)
        if DS41_DECODE_ASYNC and (i + 1) % DS41_DECODE_ASYNC == 0:
            mx.async_eval(h, pre)
    results = []
    for row, (ids, cache) in enumerate(zip(inputs, caches)):
        logits = project_logits(model.norm(hc_pre(h[row], pre[row])), model.head)
        hidden = mx.concatenate(
            [captured[row][i] for i in c.dspark_target_layer_ids], axis=-1
        )
        for item, updated in zip(cache, rows[row]):
            item.cache = updated.cache
            item.advance(lengths[row])
        cache[0]._mtp_draft_stash = (ids, snapshots[row], starts[row], states[row])
        results.append((logits, hidden, None))
    return results
