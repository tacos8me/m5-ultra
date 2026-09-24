# SPDX-License-Identifier: MIT
"""One routed-expert call across requests, with row-local projection math."""

import os
import mlx.core as mx
from . import scalar_batch
from .quantization import QuantizedProjection, quantize_activation


def eligible(moe, x):
    return x.dtype == mx.bfloat16 and all(
        isinstance(p, QuantizedProjection)
        and p.mode == "affine"
        and p.bits == 3
        and p.group_size == 128
        and p.quantize_input
        for p in (moe.experts.w1, moe.experts.w3, moe.experts.w2)
    )


def forward(moe, x):
    use_m1 = os.environ.get("DS41_BATCH_M1", "1") == "1"
    selected = scalar_batch.router(moe.gate, x) if use_m1 else None
    ids, weights = selected if selected is not None else (
        mx.concatenate(parts, 0)
        for parts in zip(*[moe.gate(x[r : r + 1], None) for r in range(x.shape[0])])
    )
    rq, sq = moe.experts.quantizes_input, moe.shared_experts.quantizes_input
    quantized = quantize_activation(x) if rq or sq else x
    routed = moe.experts(
        (quantized if rq else x)[..., None, None, :], ids, weights, input_quantized=rq
    ).squeeze(-2)
    shared_input = quantized if sq else x
    shared = (
        scalar_batch.shared_expert(moe.shared_experts, shared_input) if use_m1 else None
    )
    if shared is None:
        shared = mx.concatenate(
            [
                moe.shared_experts(shared_input[r : r + 1], input_quantized=sq)
                for r in range(x.shape[0])
            ],
            0,
        )
    return (routed.astype(mx.float32).sum(-2) + shared.astype(mx.float32)).astype(
        x.dtype
    )
