# SPDX-License-Identifier: MIT
"""wo_a grouped GEMV + the FP8 round trip of its output in one launch (bitwise).

Attention ends with wo_a (a grouped GEMV: decode_fusions.grouped_gemv, or
woa_compact's byte-coded replica at singleton verify widths 2-5), then
wo_b(projected.flatten(-2)), whose first step is quantization's FP8 round trip
over 32-value groups. A group is 32 consecutive wo_a output rows of one
o-group, so a threadgroup that computes 32 consecutive rows can finish them:
the same per-row kernels (loads, dot products, unroll-8 blocks, shuffle-down
reduction, rounding to T) run with 32 rows per threadgroup instead of 4 or 8,
the rows go through threadgroup memory, and one simdgroup per input row runs
activation's round trip. wo_b then projects the rounded rows directly.
Opt-in (DS41_ATTN_OUT_FUSE=1); bitwise either way.
"""
import os
from functools import cache

import mlx.core as mx

from . import decode_fusions, ffn_fuse, woa_compact
from .quantization import QuantizedProjection

# Off by default: measured neutral to slightly slower (32-row threadgroups cost what the saved
# launch gains; singleton widths 2-5 +0.00..0.15 ms, drafts +0.04 ms, 5+5 pairs -0.08 ms).
ENABLED = os.environ.get("DS41_ATTN_OUT_FUSE", "0") == "1"

_EPILOGUE = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid < M) {
        const float val = float(tg_y[simd_gid * 32 + simd_lid]);
        y[(size_t(simd_gid) * G + g) * N + row_base + simd_lid] = T(ds41_fp8_round(val));
    }
"""

# decode_fusions._grouped_gemv (R = 2 rows per simdgroup), 16 simdgroups = 32 rows per threadgroup.
_BF16 = r"""
    threadgroup T tg_y[M * 32];
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int R = 2;
    constexpr int unroll = 8;
    constexpr int n_v4 = K / 4;
    constexpr int n_main = n_v4 - n_v4 % (32 * unroll);
    const int g = tid.z;
    const int row_base = tid.y * 32;
    const int row0 = row_base + simd_gid * R;
    const device vec<T, 4>* w4[R];
    for (int r = 0; r < R; r++) {
        const int row = row0 + r;
        w4[r] = (const device vec<T, 4>*)(weight + (size_t(g) * N + row) * K);
    }
    const device vec<T, 4>* x4[M];
    for (int v = 0; v < M; v++) x4[v] = (const device vec<T, 4>*)(x + (size_t(v) * G + g) * K);
    float result[R][M];
    for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] = 0;
    for (int base = 0; base < n_main; base += 32 * unroll) {
        float acc[R][M];
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) acc[r][v] = 0;
        for (int i = 0; i < unroll; i++) {
            const int idx = base + i * 32 + simd_lid;
            float4 xq[M];
            for (int v = 0; v < M; v++) xq[v] = float4(x4[v][idx]);
            for (int r = 0; r < R; r++) {
                const float4 wf = float4(w4[r][idx]);
                for (int v = 0; v < M; v++) acc[r][v] += dot(wf, xq[v]);
            }
        }
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] += acc[r][v];
    }
    for (int idx = n_main + simd_lid; idx < n_v4; idx += 32) {
        for (int r = 0; r < R; r++) {
            const float4 wf = float4(w4[r][idx]);
            for (int v = 0; v < M; v++) result[r][v] += dot(wf, float4(x4[v][idx]));
        }
    }
    for (int r = 0; r < R; r++)
        for (int v = 0; v < M; v++)
            for (ushort off = 16; off >= 1; off >>= 1)
                result[r][v] += simd_shuffle_down(result[r][v], off);
    if (simd_lid == 0)
        for (int r = 0; r < R; r++)
            for (int v = 0; v < M; v++)
                tg_y[v * 32 + simd_gid * R + r] = static_cast<T>(result[r][v]);
""" + _EPILOGUE

# woa_compact._gemv (one row per simdgroup), 32 simdgroups = 32 rows per threadgroup.
_COMPACT = r"""
    threadgroup T tg_y[M * 32];
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int unroll = 8;
    constexpr int n_v4 = K / 4;
    constexpr int n_main = n_v4 - n_v4 % (32 * unroll);
    const int g = tid.z;
    const int row_base = tid.y * 32;
    const int row = row_base + simd_gid;
    const size_t offset = (size_t(g) * N + row) * K;
    const device uchar4* c4 = (const device uchar4*)(codes + offset);
    const device T* w = weight + offset;
    const device vec<T, 4>* x4[M];
    for (int v = 0; v < M; v++) x4[v] = (const device vec<T, 4>*)(x + (size_t(v) * G + g) * K);
    #define DS41_WOA_DECODE(wf, idx) { \
        const uchar4 c = c4[idx]; \
        for (int j = 0; j < 4; j++) \
            wf[j] = as_type<float>((((uint(c[j]) & 128) << 8) | (((uint(c[j]) & 127) + BASE * 8) << 4)) << 16); \
        if (any((c & uchar4(120)) == uchar4(120))) \
            for (int j = 0; j < 4; j++) \
                if ((c[j] & 120) == 120) wf[j] = float(w[idx * 4 + j]); \
    }
    float result[M];
    for (int v = 0; v < M; v++) result[v] = 0;
    for (int base = 0; base < n_main; base += 32 * unroll) {
        float acc[M];
        for (int v = 0; v < M; v++) acc[v] = 0;
        for (int i = 0; i < unroll; i++) {
            const int idx = base + i * 32 + simd_lid;
            float4 xq[M];
            for (int v = 0; v < M; v++) xq[v] = float4(x4[v][idx]);
            float4 wf;
            DS41_WOA_DECODE(wf, idx);
            for (int v = 0; v < M; v++) acc[v] += dot(wf, xq[v]);
        }
        for (int v = 0; v < M; v++) result[v] += acc[v];
    }
    for (int idx = n_main + simd_lid; idx < n_v4; idx += 32) {
        float4 wf;
        DS41_WOA_DECODE(wf, idx);
        for (int v = 0; v < M; v++) result[v] += dot(wf, float4(x4[v][idx]));
    }
    for (int v = 0; v < M; v++)
        for (ushort off = 16; off >= 1; off >>= 1)
            result[v] += simd_shuffle_down(result[v], off);
    if (simd_lid == 0)
        for (int v = 0; v < M; v++)
            tg_y[v * 32 + simd_gid] = static_cast<T>(result[v]);
""" + _EPILOGUE


@cache
def _kernel(compact):
    return mx.fast.metal_kernel(
        name="ds41_woa_compact_gemv_q" if compact else "ds41_grouped_gemv_q",
        input_names=["x", "weight", "codes"] if compact else ["x", "weight"], output_names=["y"],
        header=ffn_fuse._ROUND, source=_COMPACT if compact else _BF16)


def supported(wo_b, grouped, weight, max_rows=5):
    """Where the caller runs decode_fusions/woa_compact grouped_gemv (2-5 rows; up to max_rows for
    og_fused / batched drafts; one row stays on MLX's einsum), whole 32-row groups, and a wo_b
    that FP8-rounds its input."""
    return (
        ENABLED
        and decode_fusions.DS41_DECODE_KERNELS_V2
        and isinstance(wo_b, QuantizedProjection) and wo_b.quantize_input
        and grouped.ndim == 4 and grouped.shape[0] == 1
        and 2 <= grouped.shape[1] <= min(max_rows, 16)
        and grouped.dtype in (mx.bfloat16, mx.float16) and weight.dtype == grouped.dtype
        and weight.ndim == 3 and weight.shape[0] == grouped.shape[2] and weight.shape[2] == grouped.shape[3]
        and grouped.shape[3] % 4 == 0 and weight.shape[1] % 32 == 0
        and mx.default_device() == mx.gpu
    )


def grouped_gemv_q(linear, grouped, weight, compact=True):
    """quantize_activation(grouped_gemv(grouped, weight)) in one launch, [1, rows, G, N].

    compact: use linear's woa_compact byte codes when it has them (singleton widths 2-5
    and single-stream drafts, where woa_compact.grouped_gemv would)."""
    length, groups, k = grouped.shape[1:]
    n = weight.shape[1]
    codes = woa_compact.codes_of(linear) if compact and woa_compact.ENABLED else None
    if codes is not None:
        return _kernel(True)(
            inputs=[grouped, codes.weight, codes.codes],
            template=[("T", grouped.dtype), ("K", k), ("N", n), ("M", length), ("G", groups), ("BASE", codes.base)],
            grid=(32, n, groups), threadgroup=(32, 32, 1),
            output_shapes=[(1, length, groups, n)], output_dtypes=[grouped.dtype])[0]
    return _kernel(False)(
        inputs=[grouped, weight],
        template=[("T", grouped.dtype), ("K", k), ("N", n), ("M", length), ("G", groups)],
        grid=(32, n // 2, groups), threadgroup=(32, 16, 1),
        output_shapes=[(1, length, groups, n)], output_dtypes=[grouped.dtype])[0]
