# SPDX-License-Identifier: MIT
"""Bitwise replicas of MLX's short-M MXFP8 GEMV kernels with better layouts.

fp_qmv_wide (group 32, k_lanes 16, one x tile of 4-5 vectors): every output
keeps the same 16 K lanes, per-lane group order, expressions and
shuffle-down reduction, but each lane serves two consecutive rows, so the
activation chunks it loads per quantization group feed both rows.

fp_qmv_fast (M == 1): each lane keeps its 8-value slice of every 256-value
block and the same qdot/accumulation order, but a simdgroup serves one row
instead of four, so thin projections run four times as many simdgroups.
"""

import os
from functools import cache

import mlx.core as mx

ENABLED = os.environ.get("DS41_FAST_QMV", "1") == "1"
ENABLED_M1 = True
# Measured in the full model: faster for M=4-5, slower for M=2 (fewer threadgroups).
MIN_ROWS = 4

_HEADER = r"""
inline float ds41_fp8_e4m3(uchar bits) {
    ushort v = bits & 127;
    ushort sign_bit = ((ushort)((bits >> 7) & 1)) << 15;
    ushort u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    half scaled = converted * 256.0;
    return static_cast<float>(scaled);
}
inline float ds41_e8m0(uchar bits) {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
}
// MLX's qdot for 8-bit values, kept as a separate function like the original.
inline float ds41_qdot8(const device uint8_t* w, const thread float* x_thread, float scale) {
    float accum = 0;
    for (int i = 0; i < 8; i++) {
        accum += x_thread[i] * ds41_fp8_e4m3(w[i]);
    }
    return scale * accum;
}
"""

_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int G = K / 32;
    const short k_lane = simd_lid % 16;
    const short half_id = simd_lid / 16;
    const int row0 = tid.y * (4 * R) + simd_gid * (2 * R) + half_id * R;
    const device uint8_t* wrow[R];
    const device uint8_t* srow[R];
    for (int r = 0; r < R; r++) {
        const int row = min(row0 + r, N - 1);
        wrow[r] = (const device uint8_t*)w + size_t(row) * K;
        srow[r] = scales + size_t(row) * G;
    }
    const device T* xv[M];
    for (int v = 0; v < M; v++) xv[v] = x + v * K;
    float result[R][M];
    for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] = 0;
    for (int g = k_lane; g < G; g += 16) {
        const int k0 = g * 32;
        float s[R];
        for (int r = 0; r < R; r++) s[r] = ds41_e8m0(srow[r][g]);
        float acc[R][M];
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) acc[r][v] = 0;
        for (int j = 0; j < 8; j++) {
            float4 xq[M];
            for (int v = 0; v < M; v++) xq[v] = float4(((const device vec<T, 4>*)(xv[v] + k0))[j]);
            for (int r = 0; r < R; r++) {
                const device uint8_t* wg = wrow[r] + k0 + 4 * j;
                const float4 w4 = float4(ds41_fp8_e4m3(wg[0]), ds41_fp8_e4m3(wg[1]),
                                         ds41_fp8_e4m3(wg[2]), ds41_fp8_e4m3(wg[3]));
                for (int v = 0; v < M; v++) acc[r][v] += dot(w4, xq[v]);
            }
        }
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] += s[r] * acc[r][v];
    }
    for (int r = 0; r < R; r++) {
        for (int v = 0; v < M; v++) {
            result[r][v] += simd_shuffle_down(result[r][v], 8);
            result[r][v] += simd_shuffle_down(result[r][v], 4);
            result[r][v] += simd_shuffle_down(result[r][v], 2);
            result[r][v] += simd_shuffle_down(result[r][v], 1);
        }
    }
    if (k_lane == 0)
        for (int r = 0; r < R; r++)
            if (row0 + r < N)
                for (int v = 0; v < M; v++) y[v * N + row0 + r] = static_cast<T>(result[r][v]);
"""

ROWS_PER_LANE = 2

# fp_qmv_fast (M == 1) replica: each lane keeps its 8-value slice of every
# 256-value block and the same qdot/accumulation order, but a simdgroup
# serves one row instead of four, so thin projections run four times as
# many simdgroups.
_FAST_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int G = K / 32;
    const int out_row = tid.y * 2 + simd_gid;
    const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * K + simd_lid * 8;
    const device uint8_t* sl = scales + size_t(out_row) * G + simd_lid / 4;
    const device T* xp = x + simd_lid * 8;
    float result = 0;
    for (int k = 0; k < K; k += 256) {
        float x_thread[8];
        for (int i = 0; i < 8; i++) x_thread[i] = xp[i];
        const float s = ds41_e8m0(sl[0]);
        result += ds41_qdot8(ws, x_thread, s);
        ws += 256;
        sl += 8;
        xp += 256;
    }
    result = simd_sum(result);
    if (simd_lid == 0) y[out_row] = static_cast<T>(result);
"""


@cache
def _fast_kernel():
    return mx.fast.metal_kernel(
        name="ds41_mxfp8_qmv_one_row",
        input_names=["w", "scales", "x"],
        output_names=["y"],
        source=_FAST_SOURCE,
        header=_HEADER,
    )


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="ds41_mxfp8_qmv_rows",
        input_names=["w", "scales", "x"],
        output_names=["y"],
        source=_SOURCE,
        header=_HEADER,
    )


def _static_ok(projection):
    cached = projection.__dict__.get("_ds41_fast_qmv")
    if cached is None or cached[0] is not projection.weight:
        w = projection.weight
        k = w.shape[-1] * 4 if w.ndim == 2 else 0
        ok = (
            projection.mode == "mxfp8"
            and projection.bits == 8
            and projection.group_size == 32
            and projection.get("biases") is None
            and w.ndim == 2
            and w.dtype == mx.uint32
            and k % 32 == 0
            and k not in (64, 128)
            and projection.scales.shape == (w.shape[0], k // 32)
        )
        cached = projection.__dict__["_ds41_fast_qmv"] = (w, ok, w.shape[0] % 8 == 0 and k % 256 == 0)
    return cached


def supported(projection, x):
    if not ENABLED or x.dtype not in (mx.bfloat16, mx.float16) or mx.default_device() != mx.gpu:
        return False
    k = x.shape[-1]
    m = x.size // k
    _, ok, fast_ok = _static_ok(projection)
    if not ok or projection.weight.shape[1] * 4 != k:
        return False
    if m == 1:
        return ENABLED_M1 and fast_ok
    return MIN_ROWS <= m <= 5


def mxfp8_qmv(x, weight, scales):
    """Bitwise mx.quantized_matmul(x, weight, scales, mode='mxfp8') for M <= 5."""
    k = x.shape[-1]
    n = weight.shape[0]
    m = x.size // k
    if m == 1:
        return _fast_kernel()(
            inputs=[weight, scales, x],
            template=[("T", x.dtype), ("K", k)],
            grid=(32, n, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(*x.shape[:-1], n)],
            output_dtypes=[x.dtype],
        )[0]
    rows = 4 * ROWS_PER_LANE
    return _kernel()(
        inputs=[weight, scales, x],
        template=[("T", x.dtype), ("K", k), ("N", n), ("M", m), ("R", ROWS_PER_LANE)],
        grid=(32, (n + rows - 1) // rows * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]
