# SPDX-License-Identifier: MIT
"""Batch independent rows through the same reductions as scalar decode."""
from functools import cache

import mlx.core as mx

from . import decode_fusions
from .activation import quantize_swiglu_activation
from .quantization import QuantizedProjection


def supported_rows(x):
    return (
        x.ndim == 3
        and 2 <= x.shape[0] <= 8
        and x.shape[1] == 1
        and x.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
    )


@cache
def _row_indices(rows):
    return mx.arange(rows, dtype=mx.uint32), mx.zeros((rows,), mx.uint32)


def router(gate, x):
    from .language import DS41_MHC

    c = gate._config
    if not (
        supported_rows(x)
        and DS41_MHC
        and c.n_routed_experts == 384
        and c.n_activated_experts == 6
        and c.norm_topk_prob
        and c.score_func not in ("sigmoid", "softmax")
    ):
        return None
    # Ordinary batched matmul collapses B into M and selects GEMM/split-K.
    # Gather-MM keeps M=1 and the scalar GEMV's lanes and reduction order.
    lhs, rhs = _row_indices(x.shape[0])
    raw = mx.gather_mm(
        x.astype(mx.float32),
        gate.weight.astype(mx.float32).T[None],
        lhs_indices=lhs,
        rhs_indices=rhs,
    )
    if c.gate_temp != 1:
        raw = raw / c.gate_temp
    return decode_fusions.router(raw, gate.bias, c.route_scale)


def shared_expert(expert, x):
    if not supported_rows(x) or not all(
        isinstance(p, QuantizedProjection)
        and p.mode == "mxfp8"
        and p.bits == 8
        and p.group_size == 32
        and p.quantize_input
        and p.get("biases") is None
        for p in (expert.w1, expert.w3, expert.w2)
    ):
        return None

    def project(p, value):
        # The singleton weight batch prevents MLX from flattening input rows
        # into M. This chooses fp_qmv_fast batch_1, with scalar M=1 arithmetic.
        return mx.quantized_matmul(
            value,
            p.weight[None],
            p.scales[None],
            group_size=p.group_size,
            bits=p.bits,
            mode=p.mode,
        )

    gate, up = project(expert.w1, x), project(expert.w3, x)
    hidden = quantize_swiglu_activation(gate, up, None, x.dtype, expert._limit)
    return project(expert.w2, hidden)
