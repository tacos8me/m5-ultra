# SPDX-License-Identifier: MIT
"""Attention sublayer input for short decode/verify rows (bitwise the DS41_MHC path).

hc_fuse.project_pre_norm, then Attention._input_projections' FP8 round trip of
the normalized rows and its two MXFP8 projections (wq_a, wkv) ran as four
launches. Here it is two:

* ``project_pre_norm_q``: hc_fuse.project_pre_norm whose pre-norm also writes
  the FP8 round trip (quantization.quantize_activation), the norm's serial
  per-element tail spread over J threadgroups per row (each recomputes the
  row's norm with the same 256-thread reduction).
* ``input_projections``: wq_a and wkv in one launch, every output row with the
  arithmetic of the kernel the unfused path picks for its row count
  (fast_qmv's one-row kernel for one row, its rows kernel otherwise, which is
  bitwise MLX's M = 2 quantized_matmul).

The normalized rows (still needed by the compressor, the indexer and the
fallbacks) are unchanged. DS41_ATTN_IN_FUSE=0 restores the unfused path.
"""
import os
from functools import cache

import mlx.core as mx

from . import fast_qmv, ffn_fuse, hc_fuse
from .quantization import QuantizedProjection

ENABLED = os.environ.get("DS41_ATTN_IN_FUSE", "1") == "1"

_PRE_NORM_Y_Q = ffn_fuse._PRE_NORM_Q.replace(
    "inline void ds41_pre_norm_row_q(", "inline void ds41_pre_norm_row_yq(").replace(
    "device float* xf, device T* xq)", "device T* y, device T* yq)").replace(
    "        xf[row * D + d] = v;\n        xq[row * D + d] = T(ds41_fp8_round(v));\n",
    "        y[row * D + d] = T(v);\n        yq[row * D + d] = T(ds41_fp8_round(v));\n")
assert "device T* y, device T* yq)" in _PRE_NORM_Y_Q and "yq[row * D + d]" in _PRE_NORM_Y_Q

_PROJECT_PRE_NORM_Y_Q = hc_fuse._PROJECT_PRE_NORM.replace("if (o == 24) {", "if (o >= 24) {").replace(
    "ds41_pre_norm_row<T, D>(x, pre, weight, eps[1], row, t, lane, sg, sums, y);",
    "ds41_pre_norm_row_yq<T, D, J>(o - 24, x, pre, weight, eps[1], row, t, lane, sg, sums, y, yq);")
assert "ds41_pre_norm_row_yq<T, D, J>" in _PROJECT_PRE_NORM_Y_Q

# Two MXFP8 projections of the same rows. Threadgroups below TG1 are the first
# projection's, the rest the second's; per output row the arithmetic of
# fast_qmv._FAST_SOURCE (M == 1, two rows per threadgroup) or fast_qmv._SOURCE
# (R = 1, four rows per threadgroup).
_DUAL = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const bool second = threadgroup_position_in_grid.y >= TG1;
    const uint ty = second ? threadgroup_position_in_grid.y - TG1 : threadgroup_position_in_grid.y;
    const device uint8_t* w = (const device uint8_t*)(second ? w2 : w1);
    const device uint8_t* scales = second ? s2 : s1;
    device T* y = second ? y2 : y1;
    const int N = second ? N2 : N1;
    constexpr int G = K / 32;
    if (M == 1) {
        const int out_row = ty * 2 + simd_gid;
        const device uint8_t* ws = w + size_t(out_row) * K + simd_lid * 8;
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
    } else {
        constexpr int R = 1;
        const short k_lane = simd_lid % 16;
        const short half_id = simd_lid / 16;
        const int row0 = ty * (4 * R) + simd_gid * (2 * R) + half_id * R;
        const device uint8_t* wrow[R];
        const device uint8_t* srow[R];
        for (int r = 0; r < R; r++) {
            const int row = min(row0 + r, N - 1);
            wrow[r] = w + size_t(row) * K;
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
    }
"""


@cache
def _pre_norm_kernel():
    return mx.fast.metal_kernel(
        name="ds41_attn_project_pre_norm_q", input_names=["x", "pre", "fn", "weight", "eps"],
        output_names=["mix", "y", "yq"], header=hc_fuse._HEADER + ffn_fuse._ROUND + _PRE_NORM_Y_Q,
        source=_PROJECT_PRE_NORM_Y_Q)


@cache
def _dual_kernel():
    return mx.fast.metal_kernel(
        name="ds41_attn_dual_mxfp8", input_names=["x", "w1", "s1", "w2", "s2"],
        output_names=["y1", "y2"], header=fast_qmv._HEADER, source=_DUAL)


def project_pre_norm_q(h, pre, fn, weight, proj_eps, norm_eps):
    """(hc_project mix, pre-norm rows, their FP8 round trip) in one launch."""
    rows, d = h.size // (4 * h.shape[-1]), h.shape[-1]
    j = ffn_fuse.NORM_SPLIT_1 if rows == 1 else ffn_fuse.NORM_SPLIT
    return _pre_norm_kernel()(
        inputs=[h, pre, fn, weight, hc_fuse._consts(float(proj_eps), float(norm_eps))],
        template=[("T", h.dtype), ("D", d), ("DH", 4 * d), ("J", j)],
        grid=(rows * 256, 24 + j, 1), threadgroup=(256, 1, 1),
        output_shapes=[(*h.shape[:-2], 24), (*h.shape[:-2], d), (*h.shape[:-2], d)],
        output_dtypes=[mx.float32, h.dtype, h.dtype])


def _mxfp8(p):
    return isinstance(p, QuantizedProjection) and p.quantize_input and fast_qmv._static_ok(p)[1]


def eligible(attn, h, max_rows=5):
    """Rows the fused input serves: DS41_MHC short rows with MXFP8 wq_a/wkv (the caller
    checked hc_fuse.eligible). One row needs fast_qmv's one-row layout (N % 8, K % 256).
    Singleton rows stop at 5 (MLX's quantized_matmul serves 6-8); og_fused passes 16
    (its projections always use the rows kernel)."""
    rows = h.shape[1]
    if not (ENABLED and fast_qmv.ENABLED and h.shape[-1] % 256 == 0 and 1 <= rows <= max_rows):
        return False
    ok = attn.__dict__.get("_ds41_attn_in")
    if ok is None or ok[0] is not attn.wq_a.weight:
        good = (_mxfp8(attn.wq_a) and _mxfp8(attn.wkv)
                and attn.wq_a.weight.shape[1] == attn.wkv.weight.shape[1]
                and all(fast_qmv._static_ok(p)[2] for p in (attn.wq_a, attn.wkv)))
        ok = attn.__dict__["_ds41_attn_in"] = (attn.wq_a.weight, good)
    return ok[1] and (rows > 1 or fast_qmv.ENABLED_M1)


def input_projections(xq, p1, p2):
    """(p1.project_quantized(xq), p2.project_quantized(xq)) in one launch; xq: [1, rows, K]."""
    k = xq.shape[-1]
    m = xq.size // k
    n1, n2 = p1.weight.shape[0], p2.weight.shape[0]
    per = 2 if m == 1 else 4
    tg1 = (n1 + per - 1) // per
    tgs = tg1 + (n2 + per - 1) // per
    return _dual_kernel()(
        inputs=[xq, p1.weight, p1.scales, p2.weight, p2.scales],
        template=[("T", xq.dtype), ("K", k), ("M", m), ("N1", n1), ("N2", n2), ("TG1", tg1)],
        grid=(32, tgs * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(*xq.shape[:-1], n1), (*xq.shape[:-1], n2)],
        output_dtypes=[xq.dtype, xq.dtype])


# wq_b and the query's partial RoPE (language.rope_range -> fast_rope.partial_rope) in one
# launch. A RoPE pair is two adjacent wq_b output rows (2p, 2p + 1 of a head), which the
# projection kernels keep in one simdgroup (rows kernel: its two half-simdgroups) or one
# threadgroup (one-row kernel: its two simdgroups); each row is rounded to T first, then
# rotated with fast_rope's expressions and tables.
_WQB_ROPE = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int G = K / 32;
    constexpr uint KEEP = (HD - RD) / 2;
    threadgroup float pair_tg[2];
    float mine[M];
    int row;
    bool odd;
    if (M == 1) {
        row = threadgroup_position_in_grid.y * 2 + simd_gid;
        odd = simd_gid == 1;
        const device uint8_t* ws = (const device uint8_t*)w + size_t(row) * K + simd_lid * 8;
        const device uint8_t* sl = scales + size_t(row) * G + simd_lid / 4;
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
        mine[0] = simd_sum(result);
    } else {
        const short k_lane = simd_lid % 16;
        const short half_id = simd_lid / 16;
        row = threadgroup_position_in_grid.y * 4 + simd_gid * 2 + half_id;
        odd = half_id == 1;
        const device uint8_t* wrow = (const device uint8_t*)w + size_t(row) * K;
        const device uint8_t* srow = scales + size_t(row) * G;
        const device T* xv[M];
        for (int v = 0; v < M; v++) xv[v] = x + v * K;
        float result[M];
        for (int v = 0; v < M; v++) result[v] = 0;
        for (int g = k_lane; g < G; g += 16) {
            const int k0 = g * 32;
            const float s = ds41_e8m0(srow[g]);
            float acc[M];
            for (int v = 0; v < M; v++) acc[v] = 0;
            for (int j = 0; j < 8; j++) {
                float4 xq[M];
                for (int v = 0; v < M; v++) xq[v] = float4(((const device vec<T, 4>*)(xv[v] + k0))[j]);
                const device uint8_t* wg = wrow + k0 + 4 * j;
                const float4 w4 = float4(ds41_fp8_e4m3(wg[0]), ds41_fp8_e4m3(wg[1]),
                                         ds41_fp8_e4m3(wg[2]), ds41_fp8_e4m3(wg[3]));
                for (int v = 0; v < M; v++) acc[v] += dot(w4, xq[v]);
            }
            for (int v = 0; v < M; v++) result[v] += s * acc[v];
        }
        for (int v = 0; v < M; v++) {
            result[v] += simd_shuffle_down(result[v], 8);
            result[v] += simd_shuffle_down(result[v], 4);
            result[v] += simd_shuffle_down(result[v], 2);
            result[v] += simd_shuffle_down(result[v], 1);
            mine[v] = result[v];
        }
    }
    const uint p = (uint(row) % HD) / 2;
    for (int v = 0; v < M; v++) {
        float other;
        if (M == 1) {
            if (simd_lid == 0) pair_tg[simd_gid] = mine[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            other = pair_tg[1 - simd_gid];
        } else {
            other = simd_shuffle_xor(mine[v], 16);
        }
        if (simd_lid % (M == 1 ? 32 : 16) != 0) continue;
        const T mine_t = static_cast<T>(mine[v]);
        if (p < KEEP) {
            y[size_t(v) * N + row] = mine_t;
            continue;
        }
        const T other_t = static_cast<T>(other);
        const uint j = v * (RD / 2) + (p - KEEP);
        float c = cos_t[j];
        float s = sin_t[j];
        float a = static_cast<float>(odd ? other_t : mine_t);
        float b2 = static_cast<float>(odd ? mine_t : other_t);
        float a_c = a * c;
        float b_s = b2 * s;
        float a_s = a * s;
        float b_c = b2 * c;
        y[size_t(v) * N + row] = odd ? static_cast<T>(a_s + b_c) : static_cast<T>(a_c - b_s);
    }
"""


@cache
def _wqb_rope_kernel():
    return mx.fast.metal_kernel(
        name="ds41_attn_wqb_rope", input_names=["x", "w", "scales", "cos_t", "sin_t"],
        output_names=["y"], header=fast_qmv._HEADER, source=_WQB_ROPE)


def wqb_rope_supported(wq_b, qr8, heads, head_dim, rope_dim, max_rows=5):
    """Singleton rows 1-5 whose wq_b the unfused path runs with fast_qmv's kernels (or MLX's
    M = 2 quantized_matmul, bitwise the rows kernel), and fast_rope's partial RoPE."""
    from .language import DS41_FAST_ROPE
    rows = qr8.size // qr8.shape[-1]
    return (
        ENABLED and DS41_FAST_ROPE and fast_qmv.ENABLED and 1 <= rows <= max_rows
        and (rows > 1 or fast_qmv.ENABLED_M1)
        and _mxfp8(wq_b) and fast_qmv._static_ok(wq_b)[2]
        and wq_b.weight.shape[0] == heads * head_dim and head_dim % 2 == 0
        and 0 < rope_dim <= head_dim and rope_dim % 2 == 0
        and qr8.dtype in (mx.bfloat16, mx.float16)
    )


def wqb_rope(qr8, wq_b, cos, sin, rope_dim, heads, head_dim):
    """rope_range(wq_b.project_quantized(qr8).reshape(1, L, heads, head_dim), ...) with the
    tables rope_tables returns; [1, L, heads, head_dim]."""
    k = qr8.shape[-1]
    m = qr8.size // k
    n = wq_b.weight.shape[0]
    per = 2 if m == 1 else 4
    return _wqb_rope_kernel()(
        inputs=[qr8, wq_b.weight, wq_b.scales, cos, sin],
        template=[("T", qr8.dtype), ("K", k), ("M", m), ("N", n), ("HD", head_dim), ("RD", rope_dim)],
        grid=(32, n // per * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(1, m, heads, head_dim)], output_dtypes=[qr8.dtype])[0]
