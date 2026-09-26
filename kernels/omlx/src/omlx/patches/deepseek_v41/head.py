# SPDX-License-Identifier: MIT
"""Vocabulary projection with BF16 storage and FP32 accumulation/output."""

import os
from functools import cache

import mlx.core as mx

# One pass over the BF16 head for all 2-5 query rows (bitwise the per-row
# kernel below: same lanes, k order, dot/accumulate and shuffle reduction).
DS41_HEAD_ROWS = os.environ.get("DS41_HEAD_ROWS", "1") == "1"

_SOURCE = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    const uint query = threadgroup_position_in_grid.y;
    float sum = 0;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1],
                                w[offset + 2], w[offset + 3]);
        const float4 b = float4(x[query * K + k], x[query * K + k + 1], x[query * K + k + 2], x[query * K + k + 3]);
        sum += dot(a, b);
    }
    for (uint offset = KL / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }
    if (lane == 0) y[query * N + row] = sum;
"""

# Each 16-lane group reads its weight row once and keeps M running sums; the
# arithmetic per (query, row) is exactly the per-row kernel's.
_ROWS_SOURCE = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    float sum[M];
    for (int q = 0; q < M; q++) sum[q] = 0;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(*((const device vec<bfloat16_t, 4>*)(w + offset)));
        for (int q = 0; q < M; q++) {
            const float4 b = *((const device float4*)(x + q * K + k));
            sum[q] += dot(a, b);
        }
    }
    for (int q = 0; q < M; q++) {
        float s = sum[q];
        for (uint offset = KL / 2; offset > 0; offset /= 2) {
            s += simd_shuffle_down(s, offset);
        }
        if (lane == 0) y[q * N + row] = s;
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="v41_bf16_head_fp32_output",
        input_names=["x", "w"],
        output_names=["y"],
        source=_SOURCE,
    )


@cache
def _rows_kernel():
    return mx.fast.metal_kernel(
        name="v41_bf16_head_rows_fp32_output",
        input_names=["x", "w"],
        output_names=["y"],
        source=_ROWS_SOURCE,
    )


def project_logits(x, weight):
    """Keep the full logits contract for prefill and multi-token verification."""
    if hasattr(weight, "weight"):
        if hasattr(weight, "bits"):
            if weight.quantize_input:
                raise ValueError("Vocabulary head must not quantize activations")
            from .language import row_tiles, verify_tile

            tile = verify_tile()
            rows = (
                row_tiles(x.shape[1], max(tile, 11))
                if tile and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] > 5
                else [(0, x.shape[-2])]
            )
            # The affine QMV keeps one reduction order up to 11 rows.
            parts = [
                mx.quantized_matmul(
                    (x[:, b:e] if len(rows) > 1 else x).astype(mx.float32),
                    weight.weight,
                    weight.scales.astype(mx.float32),
                    weight.biases.astype(mx.float32),
                    transpose=True,
                    group_size=weight.group_size,
                    bits=weight.bits,
                    mode=weight.mode,
                )
                for b, e in rows
            ]
            return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        weight = weight.weight
    rows, width = weight.shape
    if (
        weight.dtype != mx.bfloat16
        or x.shape[-1] != width
        or not 1 <= x.size // width <= 5
        or width % 4
        or width < 256
        or rows < 4096
        or mx.default_device() == mx.cpu
    ):
        return x.astype(mx.float32) @ weight.astype(mx.float32).T
    lanes = 16
    queries = x.size // width
    if DS41_HEAD_ROWS and queries > 1:
        return _rows_kernel()(
            inputs=[x.astype(mx.float32), weight],
            template=[("N", rows), ("K", width), ("KL", lanes), ("M", queries)],
            grid=(((rows * lanes + 63) // 64) * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(*x.shape[:-1], rows)],
            output_dtypes=[mx.float32],
        )[0]
    return _kernel()(
        inputs=[x.astype(mx.float32), weight],
        template=[("N", rows), ("K", width), ("KL", lanes)],
        grid=(((rows * lanes + 63) // 64) * 64, queries, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.float32],
    )[0]
