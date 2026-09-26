# SPDX-License-Identifier: MIT
"""Layer-major plain decode with request-local attention and compressor state.

Only unpadded one-token calls use this path. Prefill and DSpark capture retain
LanguageModel's reference row loop. Router, shared-expert projections, and
attention keep their one-row arithmetic. Routed experts, mHC, and the output
head share existing operators. No kernel or cache format changes.
"""

import os
from contextlib import nullcontext
import mlx.core as mx
import numpy as np
from .cache import DeepseekV41Cache
from . import routed_batch


def forward(model, input_ids, cache):
    from .language import hc_mixes, hc_pre, hc_pre_norm, hc_post, project_logits

    c = model._config
    batch = input_ids.shape[0]
    if not getattr(model, "_batch_decode_logged", False):
        object.__setattr__(model, "_batch_decode_logged", True)
        import logging

        logging.getLogger(__name__).info(
            "DeepSeek V4.1 layer-major plain decode active"
        )
    # Materialize tiny offset vectors once, not one GPU wait per row/layer.
    offsets = np.asarray(mx.stack([item.offset for item in cache]))
    if np.any(offsets != offsets[0]):
        raise ValueError("DeepSeek V4.1 cache offsets diverged across layers")
    starts = [int(x) for x in offsets[0]]
    rows = [
        [item.extract(row, offset=starts[row]) for row in range(batch)]
        for item in cache
    ]
    shared = [{} for _ in range(batch)]
    hashes, histories = [], []
    for row in range(batch):
        hs, hist = (
            (None, None)
            if model._hasher is None
            else model._hasher(input_ids[row : row + 1], rows[0][row][6], None)
        )
        hashes.append(hs)
        histories.append(hist)
    h = mx.repeat(model.embed(input_ids)[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
    prefetch = getattr(model, "_engram_prefetch", None)
    for i, layer in enumerate(model.layers):
        if "engram" in layer:
            ix = list(c.engram_layer_ids).index(i)
            outputs = []
            for row in range(batch):
                # The SSD prefetcher's context and pending rows belong to one
                # request. Keep the existing per-row contract unchanged.
                with prefetch.forward() if prefetch is not None else nullcontext():
                    outputs.append(
                        layer.engram(h[row : row + 1], hashes[row][:, :, ix], None)
                    )
            h = mx.concatenate(outputs, 0)
        ap, ao, ac = hc_mixes(
            h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, c
        )
        x = hc_pre_norm(h, pre, layer.attn_norm.weight, layer.attn_norm.eps)
        attn = mx.concatenate(
            [
                layer.attn(x[row : row + 1], rows[i][row], shared[row], starts[row])
                for row in range(batch)
            ],
            0,
        )
        h = hc_post(attn, h, ao, ac)
        fp, fo, fc = hc_mixes(
            h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, c
        )
        x = hc_pre_norm(h, ap, layer.ffn_norm.weight, layer.ffn_norm.eps)
        # Router and shared projections batch through scalar-order GEMV
        # kernels; routed affine q3 experts share their existing gather call.
        # Attention and cache state remain request-local.
        if os.environ.get("DS41_BATCH_ROUTED", "1") == "1" and routed_batch.eligible(
            layer.ffn, x
        ):
            ffn = routed_batch.forward(layer.ffn, x)
        else:
            ffn = mx.concatenate(
                [layer.ffn(x[row : row + 1], None) for row in range(batch)], 0
            )
        h, pre = hc_post(ffn, h, fo, fc), fp
        for row in range(batch):
            rc = rows[i][row]
            rc[0] = mx.array([starts[row] + 1], mx.int32)
            if i == 0 and histories[row] is not None:
                rc[6] = mx.array(histories[row], mx.int64)
        cache[i].adopt(DeepseekV41Cache.merge(rows[i]))
        cache[i].advance(1)
    final = model.norm(hc_pre(h, pre))
    return project_logits(final, model.head)
