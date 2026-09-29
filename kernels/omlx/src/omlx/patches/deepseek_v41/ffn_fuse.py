# SPDX-License-Identifier: MIT
"""Fused FFN sublayer for short decode/verify rows (bitwise the DS41_MHC path).

A decode Block's FFN sublayer (hc_fuse.block_forward's second half) ran about
fifteen dependent launches per layer: the hc projection + pre-norm, an FP32
cast of the normalized rows, MLX's router GEMM (gemv, or split-K GEMM + accum),
the top-6 router, the FP8 activation round trip, the routed gate/up pair
kernel, its SwiGLU/FP8 kernel, the routed down kernel, the shared expert's gate
and up projections, a ``mx.ones`` weight placeholder, its SwiGLU/FP8 kernel,
its down projection, the routed + shared combine and the hc post/mix. Here it
is six:

* ``pre_norm_q``: hc_fuse.project_pre_norm whose pre-norm also writes the FP32
  rows the router multiplies and their FP8 round trip; the norm's serial
  per-element tail is spread over J threadgroups per row.
* ``shared_router``: the shared expert's gate/up rows + SwiGLU/FP8, and in the
  same launch bitwise replicas of MLX's router GEMM reading the BF16 gate
  weight (independent work: the latency-bound router hides under the
  bandwidth-bound shared expert).
* decode_fusions.router (top-6, unchanged).
* ``routed_up``: each pair's gate/up rows, each threadgroup finishing its
  32-column group with the weighted SwiGLU/FP8 round trip.
* ``down``: the routed pairs' MXFP4 down rows and the shared MXFP8 down rows.
* ``post_combine``: hc_fuse.post_mix computing moe_decode.combine in place.

Every output keeps its original per-element arithmetic: the projection rows
use the same lane -> K mapping, decode, per-group scale, accumulation order
and reductions as moe_decode (routed) and fast_qmv (shared: the one-row kernel
for one row, the rows kernel otherwise, which MLX's M = 2 quantized_matmul
matches); the SwiGLU/FP8 round trips keep activation's expressions with a
simdgroup per 32-value scale group; the router keeps MLX's gemv lane order or
split-K partitions and MMA chains; intermediates are rounded to the same dtypes
at the same points. Only the grouping of rows into threadgroups and launches
changes. DS41_FFN_FUSE=0 restores the unfused path.
"""
import os
from functools import cache

import mlx.core as mx

from . import decode_fusions, fast_qmv, hc_fuse, moe_decode
from .quantization import QuantizedProjection

ENABLED = os.environ.get("DS41_FFN_FUSE", "1") == "1"
MAX_ROWS = 16
# Threadgroups per row for the pre-norm/FP8 part of pre_norm_q (divisors of D/256 = 20).
NORM_SPLIT_1 = int(os.environ.get("DS41_FFN_NORM_SPLIT_1", "10"))
NORM_SPLIT = int(os.environ.get("DS41_FFN_NORM_SPLIT", "5"))

# FP8 E4M3 round trip of one 32-value group held by the 32 lanes of a simdgroup
# (activation._ROUND, verbatim).
_ROUND = r"""
inline float ds41_fp8_round(float v) {
    const float amax = max(simd_max(abs(v)), 0x1.cp-118f);
    const int scale_exponent = max(int(ceil(log2(amax / 448.0f))), -126);
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    const float scaled = clamp(v / scale, -448.0f, 448.0f);
    const float a = abs(scaled);
    const int step_exponent = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);
    const float step = as_type<float>(uint(step_exponent + 127) << 23);
    const float q = sign(scaled) * min(rint(a / step) * step, 448.0f);
    return q * scale;
}

// activation._TAIL_SOURCE before the round trip: SwiGLU of one gate/up pair,
// rounded to T.
template <typename T, bool WEIGHTED, typename PL>
inline float ds41_swiglu(float g, float u, PL limit, float weight) {
    if (limit[0] != 0.0f) {
        g = min(g, limit[0]);
        u = clamp(u, -limit[0], limit[0]);
    }
    const float neg_sigmoid = 1.0f / (1.0f + exp(abs(g)));
    const float sigmoid = g < 0 ? neg_sigmoid : 1.0f - neg_sigmoid;
    float value = (g * sigmoid) * u;
    if (WEIGHTED) value *= weight;
    return float(T(value));
}
"""

_PRE_NORM_Q = r"""
// hc_fuse.ds41_pre_norm_row, also writing the FP32 copy of each output and its
// FP8 round trip (quantization.quantize_activation). D % 256 == 0: every lane is
// live and a simdgroup covers one aligned 32-value group. Each of J threadgroups
// per row computes the whole row's norm (the same 256-thread reduction) and
// writes the slice js of its outputs: the serial per-element tail (FP8 round
// trips) is spread over J threadgroups instead of one.
template <typename T, uint D, uint J, typename PX, typename PP, typename PW>
inline void ds41_pre_norm_row_q(uint js, PX x, PP pre, PW weight, float eps,
                                uint row, uint tid, uint lane, uint simd, threadgroup float* sums,
                                device float* xf, device T* xq) {
    #pragma clang fp contract(off)
    constexpr uint S = (D + 255) / 256;
    float values[S];
    float total = 0.0f;
    for (uint t = 0; t < S; ++t) {
        const uint d = tid + 256 * t;
        float value = 0.0f;
        if (d < D) {
            for (uint i = 0; i < 4; ++i)
                value = float(x[(row * 4 + i) * D + d]) * pre[row * 4 + i] + value;
            value = float(T(value));
        }
        values[t] = value;
        total = total + value * value;
    }
    total = simd_sum(total);
    if (lane == 0) sums[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float sum = lane < 8 ? sums[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) sums[0] = rsqrt(sum / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = js * S / J; t < (js + 1) * S / J; ++t) {
        const uint d = tid + 256 * t;
        const float v = float(T((values[t] * sums[0]) * float(weight[d])));
        xf[row * D + d] = v;
        xq[row * D + d] = T(ds41_fp8_round(v));
    }
}
"""

_PROJECT_PRE_NORM_Q = hc_fuse._PROJECT_PRE_NORM.replace("if (o == 24) {", "if (o >= 24) {").replace(
    "ds41_pre_norm_row<T, D>(x, pre, weight, eps[1], row, t, lane, sg, sums, y);",
    "ds41_pre_norm_row_q<T, D, J>(o - 24, x, pre, weight, eps[1], row, t, lane, sg, sums, xf, xq);",
)
assert "ds41_pre_norm_row_q<T, D, J>" in _PROJECT_PRE_NORM_Q and "o >= 24" in _PROJECT_PRE_NORM_Q

# Routed MXFP4 gate/up rows (moe_decode._GATE_UP per simdgroup) of one pair and
# one 32-column group, then SwiGLU + FP8. 16 simdgroups: 0-7 gate, 8-15 up.
_ROUTED_UP = r"""
    const uint p = threadgroup_position_in_grid.x;
    const uint grp = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const bool second = sg >= 8;
    const int base_row = grp * 32 + (sg % 8) * 4;
    threadgroup T tg_gate[32], tg_up[32];
    threadgroup T* out = second ? tg_up : tg_gate;
    {
        // moe_decode._GATE_UP
        const uint e = ids[p];
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int in_vec_size_w = K / 2;
        constexpr int in_vec_size_g = K / 32;
        const device uint8_t* ws = (const device uint8_t*)(second ? w3 : w1)
            + size_t(e) * N * in_vec_size_w + base_row * in_vec_size_w + simd_lid * 8;
        const device uint8_t* sl = (second ? s3 : s1)
            + size_t(e) * N * in_vec_size_g + base_row * in_vec_size_g + simd_lid / 2;
        const device T* xs = x + size_t(p / TOPK) * K + simd_lid * values_per_thread;
        float result[4] = {0, 0, 0, 0};
        for (int k = 0; k < K; k += block_size) {
            T x_thread[values_per_thread];
            for (int i = 0; i < values_per_thread; i++) x_thread[i] = xs[k + i];
            for (int row = 0; row < 4; row++) {
                float wv[values_per_thread];
                ds41_decode4<values_per_thread>(ws + row * in_vec_size_w, wv);
                const float s = ds41_e8m0(sl[row * in_vec_size_g]);
                result[row] += ds41_qdot4_pre<values_per_thread>(x_thread, wv, s);
            }
            ws += block_size / 2;
            sl += block_size / 32;
        }
        for (int row = 0; row < 4; row++) {
            const float v = simd_sum(result[row]);
            if (simd_lid == 0) out[(sg % 8) * 4 + row] = static_cast<T>(v);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float v = ds41_swiglu<T, true>(float(tg_gate[simd_lid]), float(tg_up[simd_lid]), limit, weights[p]);
        yr[size_t(p) * N + grp * 32 + simd_lid] = T(ds41_fp8_round(v));
    }
"""

# Shared MXFP8 gate/up rows (fast_qmv one-row kernel for one row, rows kernel
# otherwise; per output row) of one 32-column group for every row, then SwiGLU + FP8.
# The router threadgroups come first (_ROUTER, substituted below).
_SHARED_UP = r"""
    const uint tgx = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    threadgroup T tg_gate[M * 32], tg_up[M * 32];
    threadgroup float partial[M == 1 ? 1 : 32 * M * 8];
ROUTER
    const uint grp = tgx - RTG;
    const bool second = sg >= 8;
    const int base_row = grp * 32 + (sg % 8) * 4;
    threadgroup T* out = second ? tg_up : tg_gate;
    if (M == 1) {
        // fast_qmv._FAST_SOURCE, four rows per simdgroup
        constexpr int G = K / 32;
        const device uint8_t* wbase = (const device uint8_t*)(second ? sw3 : sw1);
        const device uint8_t* sbase = second ? ss3 : ss1;
        float result[4] = {0, 0, 0, 0};
        for (int k = 0; k < K; k += 256) {
            float x_thread[8];
            for (int i = 0; i < 8; i++) x_thread[i] = x[k + simd_lid * 8 + i];
            for (int row = 0; row < 4; row++) {
                const int out_row = base_row + row;
                const float s = ds41_e8m0(sbase[size_t(out_row) * G + simd_lid / 4 + k / 32]);
                result[row] += ds41_qdot8(wbase + size_t(out_row) * K + simd_lid * 8 + k, x_thread, s);
            }
        }
        for (int row = 0; row < 4; row++) {
            const float v = simd_sum(result[row]);
            if (simd_lid == 0) out[(sg % 8) * 4 + row] = static_cast<T>(v);
        }
    } else {
        // fast_qmv._SOURCE with R = 2
        constexpr int R = 2;
        constexpr int G = K / 32;
        const short k_lane = simd_lid % 16;
        const short half_id = simd_lid / 16;
        const int row0 = base_row + half_id * R;
        const device uint8_t* w = (const device uint8_t*)(second ? sw3 : sw1);
        const device uint8_t* scales = second ? ss3 : ss1;
        const device uint8_t* wrow[R];
        const device uint8_t* srow[R];
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
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
                float4 xq4[M];
                for (int v = 0; v < M; v++) xq4[v] = float4(((const device vec<T, 4>*)(xv[v] + k0))[j]);
                for (int r = 0; r < R; r++) {
                    const device uint8_t* wg = wrow[r] + k0 + 4 * j;
                    const float4 w4 = float4(ds41_fp8_e4m3(wg[0]), ds41_fp8_e4m3(wg[1]),
                                             ds41_fp8_e4m3(wg[2]), ds41_fp8_e4m3(wg[3]));
                    for (int v = 0; v < M; v++) acc[r][v] += dot(w4, xq4[v]);
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
                for (int v = 0; v < M; v++)
                    out[v * 32 + (sg % 8) * 4 + half_id * R + r] = static_cast<T>(result[r][v]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < M) {
        const float v = ds41_swiglu<T, false>(float(tg_gate[sg * 32 + simd_lid]), float(tg_up[sg * 32 + simd_lid]), limit, 1.0f);
        ys[size_t(sg) * N + grp * 32 + simd_lid] = T(ds41_fp8_round(v));
    }
"""

# Router logits = MLX's x.astype(f32) @ weight.astype(f32).T for the checkpoint
# shape (384 x 5120), reading the BF16 weight (the same FP32 values), in the
# shared gate/up launch (independent work: the latency-bound router runs next to
# the bandwidth-bound shared expert). Threadgroups below RTG are the router's.
# M == 1: MLX gemv (bm4 bn1 sm1 sn32 tm4 tn4): lane l accumulates columns
# 128 i + 4 l + [0, 4) in order, then the shuffle-down tree; one row per simdgroup.
# M >= 2: MLX steel_gemm_splitk (bm16 bn32 bk16 wm2 wn2; 32 partitions of K / 32
# for every M <= 32) + steel_gemm_splitk_accum: per 8-column tile, each
# partition's chain of 8x8x8 simdgroup MMAs from zero on the same A (unused
# fragment rows zero) and B values, then the partitions summed in order from
# zero. Every output element depends only on its own row and column, so the
# rows of several requests (og_fused) share fragments.
_ROUTER = r"""
    if (tgx < RTG) {
        if (M == 1) {
            const uint row = tgx * 16 + sg;
            const device T* wr = gw + size_t(row) * K + 4 * simd_lid;
            const device float* xr = xf + 4 * simd_lid;
            float r = 0.0f;
            constexpr int ITERS = K / 128;
            constexpr int U = 8;
            for (int i0 = 0; i0 < ITERS; i0 += U) {
                float wv[U][4], xv[U][4];
                for (int u = 0; u < U; ++u) {
                    for (int t = 0; t < 4; ++t) {
                        wv[u][t] = float(wr[128 * (i0 + u) + t]);
                        xv[u][t] = xr[128 * (i0 + u) + t];
                    }
                }
                for (int u = 0; u < U; ++u)
                    for (int t = 0; t < 4; ++t)
                        r += wv[u][t] * xv[u][t];
            }
            for (ushort sn = 16; sn >= 1; sn >>= 1) r += simd_shuffle_down(r, sn);
            if (simd_lid == 0) raw[row] = r;
        } else {
            const uint n0 = tgx * 8;
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            constexpr int PART = K / 32;
            constexpr int F = (M + 7) / 8;
            for (int pp = 0; pp < 2; ++pp) {
                const int p = sg + 16 * pp;
                const device T* w0 = gw + size_t(n0 + fn) * K + p * PART + fm;
                const device T* w1 = w0 + K;
                for (int f = 0; f < F; ++f) {
                    const int m = f * 8 + fm;
                    const device float* xa = xf + size_t(m) * K + p * PART + fn;
                    simdgroup_float8x8 C = simdgroup_float8x8(0);
                    for (int c = 0; c < PART / 8; ++c) {
                        simdgroup_float8x8 A, B;
                        A.thread_elements()[0] = m < M ? xa[8 * c] : 0.0f;
                        A.thread_elements()[1] = m < M ? xa[8 * c + 1] : 0.0f;
                        B.thread_elements()[0] = float(w0[8 * c]);
                        B.thread_elements()[1] = float(w1[8 * c]);
                        simdgroup_multiply_accumulate(C, A, B, C);
                    }
                    if (m < M) {
                        partial[(p * M + m) * 8 + fn] = C.thread_elements()[0];
                        partial[(p * M + m) * 8 + fn + 1] = C.thread_elements()[1];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint t = sg * 32 + simd_lid;
            if (t < M * 8) {
                const uint m = t / 8, j = t % 8;
                float acc = 0;
                for (int q = 0; q < 32; q++) acc += partial[(q * M + m) * 8 + j];
                raw[m * NR + n0 + j] = acc;
            }
        }
        return;
    }
"""

# Routed MXFP4 down (moe_decode._DOWN) and shared MXFP8 down (fast_qmv rows /
# one-row kernels): x = PAIRS routed slots, then SLOTS shared slots per 8 rows.
_DOWN = r"""
    const uint p = threadgroup_position_in_grid.x;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    if (p < PAIRS) {
        const uint e = ids[p];
        constexpr int values_per_thread = 8;
        constexpr int block_size = values_per_thread * 32;
        constexpr int in_vec_size_w = K / 2;
        constexpr int in_vec_size_g = K / 32;
        const int out_row = threadgroup_position_in_grid.y * 8 + simd_gid * 4;
        const device uint8_t* ws = (const device uint8_t*)w2
            + size_t(e) * N * in_vec_size_w + out_row * in_vec_size_w + simd_lid * 4;
        const device uint8_t* sl = s2 + size_t(e) * N * in_vec_size_g + out_row * in_vec_size_g + simd_lid / 4;
        const device T* xs = yr + size_t(p) * K + simd_lid * values_per_thread;
        float result[4] = {0, 0, 0, 0};
        int k = 0;
        for (; k < K - block_size; k += block_size) {
            T x_thread[values_per_thread];
            for (int i = 0; i < values_per_thread; i++) x_thread[i] = xs[k + i];
            for (int row = 0; row < 4; row++) {
                float wv[values_per_thread];
                ds41_decode4<values_per_thread>(ws + row * in_vec_size_w, wv);
                const float s = ds41_e8m0(sl[row * in_vec_size_g]);
                result[row] += ds41_qdot4_pre<values_per_thread>(x_thread, wv, s);
            }
            ws += block_size / 2;
            sl += block_size / 32;
        }
        {
            T x_thread[values_per_thread];
            for (int i = 0; i < values_per_thread; i++) x_thread[i] = xs[k + i];
            for (int row = 0; row < 4; row++) {
                float wv[values_per_thread];
                ds41_decode4<values_per_thread>(ws + row * in_vec_size_w, wv);
                const float s = ds41_e8m0(sl[row * in_vec_size_g]);
                result[row] += ds41_qdot4_pre<values_per_thread>(x_thread, wv, s);
            }
        }
        for (int row = 0; row < 4; row++) {
            const float v = simd_sum(result[row]);
            if (simd_lid == 0) down_r[size_t(p) * N + out_row + row] = static_cast<T>(v);
        }
    } else if (M == 1) {
        // fast_qmv._FAST_SOURCE: 4 slots x 2 rows per 8-row block
        constexpr int G = K / 32;
        const int out_row = (threadgroup_position_in_grid.y * 4 + (p - PAIRS)) * 2 + simd_gid;
        const device uint8_t* ws = (const device uint8_t*)sw2 + size_t(out_row) * K + simd_lid * 8;
        const device uint8_t* sl = ss2 + size_t(out_row) * G + simd_lid / 4;
        const device T* xp = ys + simd_lid * 8;
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
        if (simd_lid == 0) down_s[out_row] = static_cast<T>(result);
    } else {
        // fast_qmv._SOURCE with R = 1: 2 slots x 4 rows per 8-row block
        constexpr int R = 1;
        constexpr int G = K / 32;
        const short k_lane = simd_lid % 16;
        const short half_id = simd_lid / 16;
        const int row0 = (threadgroup_position_in_grid.y * 2 + (p - PAIRS)) * (4 * R) + simd_gid * (2 * R) + half_id * R;
        const device uint8_t* wrow[R];
        const device uint8_t* srow[R];
        for (int r = 0; r < R; r++) {
            const int row = min(row0 + r, N - 1);
            wrow[r] = (const device uint8_t*)sw2 + size_t(row) * K;
            srow[r] = ss2 + size_t(row) * G;
        }
        const device T* xv[M];
        for (int v = 0; v < M; v++) xv[v] = ys + v * K;
        float result[R][M];
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] = 0;
        for (int g = k_lane; g < G; g += 16) {
            const int k0 = g * 32;
            float s[R];
            for (int r = 0; r < R; r++) s[r] = ds41_e8m0(srow[r][g]);
            float acc[R][M];
            for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) acc[r][v] = 0;
            for (int j = 0; j < 8; j++) {
                float4 xq4[M];
                for (int v = 0; v < M; v++) xq4[v] = float4(((const device vec<T, 4>*)(xv[v] + k0))[j]);
                for (int r = 0; r < R; r++) {
                    const device uint8_t* wg = wrow[r] + k0 + 4 * j;
                    const float4 w4 = float4(ds41_fp8_e4m3(wg[0]), ds41_fp8_e4m3(wg[1]),
                                             ds41_fp8_e4m3(wg[2]), ds41_fp8_e4m3(wg[3]));
                    for (int v = 0; v < M; v++) acc[r][v] += dot(w4, xq4[v]);
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
                    for (int v = 0; v < M; v++) down_s[v * N + row0 + r] = static_cast<T>(result[r][v]);
    }
"""

# hc_fuse._POST_MIX with x = moe_decode.combine(routed, shared) computed in place.
_POST_COMBINE = r"""
    const uint z = thread_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = (z - thread_position_in_threadgroup.x) / D;
    threadgroup float tpre[4], tpost[4], tcomb[16];
    if (sg == 0)
        ds41_mix_row<ITERS>(mix, scale, base, eps[0], row, lane, tpre, tpost, tcomb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = z % D;
    float acc = 0.0f;
    for (int j = 0; j < TOPK; j++) acc += float(routed[(row * TOPK + j) * D + d]);
    const T combined = static_cast<T>(acc + float(shared[z]));
    ds41_post_elem_value<T, D>(float(combined), residual, tpost, tcomb, z, y);
    if (d < 4) pre_out[row * 4 + d] = tpre[d];
"""

_POST_VALUE = r"""
// hc_fuse.ds41_post_elem with x[z] already loaded.
template <typename T, uint D, typename PR>
inline void ds41_post_elem_value(float value, PR residual, threadgroup const float* post,
                                 threadgroup const float* comb, uint z, device T* y) {
    #pragma clang fp contract(off)
    const uint row = z / D, d = z % D;
    float values[4];
    for (uint i = 0; i < 4; ++i)
        values[i] = float(residual[(row * 4 + i) * D + d]);
    for (uint j = 0; j < 4; ++j) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i)
            sum = fma(comb[i * 4 + j], values[i], sum);
        y[(row * 4 + j) * D + d] = T(post[j] * value + sum);
    }
}
"""


@cache
def _pre_norm_q_kernel():
    return mx.fast.metal_kernel(
        name="ds41_ffn_project_pre_norm_q", input_names=["x", "pre", "fn", "weight", "eps"],
        output_names=["mix", "xf", "xq"], header=hc_fuse._HEADER + _ROUND + _PRE_NORM_Q,
        source=_PROJECT_PRE_NORM_Q)


@cache
def _routed_up_kernel():
    return mx.fast.metal_kernel(
        name="ds41_ffn_routed_up", input_names=["x", "w1", "s1", "w3", "s3", "ids", "weights", "limit"],
        output_names=["yr"], header=moe_decode._HEADER + _ROUND, source=_ROUTED_UP)


@cache
def _shared_router_kernel():
    return mx.fast.metal_kernel(
        name="ds41_ffn_shared_router", input_names=["x", "sw1", "ss1", "sw3", "ss3", "limit", "gw", "xf"],
        output_names=["ys", "raw"], header=_qmv8_header(full=True) + _ROUND,
        source=_SHARED_UP.replace("ROUTER\n", _ROUTER))


@cache
def _down_kernel():
    return mx.fast.metal_kernel(
        name="ds41_ffn_down", input_names=["yr", "w2", "s2", "ids", "ys", "sw2", "ss2"],
        output_names=["down_r", "down_s"], header=moe_decode._HEADER + _qmv8_header(), source=_DOWN)


@cache
def _post_combine_kernel():
    return mx.fast.metal_kernel(
        name="ds41_ffn_post_combine",
        input_names=["routed", "shared", "residual", "mix", "scale", "base", "eps"],
        output_names=["y", "pre_out"], header=hc_fuse._HEADER + _POST_VALUE, source=_POST_COMBINE)


def _qmv8_header(full=False):
    # fast_qmv's FP8/E8M0 helpers; its ds41_e8m0 duplicates moe_decode's (same body).
    if full:
        return fast_qmv._HEADER
    return fast_qmv._HEADER.replace(
        "inline float ds41_e8m0(uchar bits) {\n"
        "    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));\n"
        "    return as_type<float>(out);\n"
        "}\n", "")


@cache
def _limit(value):
    out = mx.array([float(value or 0)], mx.float32)
    mx.eval(out)
    return out


def _mxfp8(p):
    return isinstance(p, QuantizedProjection) and p.quantize_input and fast_qmv._static_ok(p)[1]


def _static_ok(moe):
    """Checkpoint shapes/formats the kernels hard-code (cached per MoE module)."""
    cached = moe.__dict__.get("_ds41_ffn_fuse")
    if cached is not None and cached[0] is moe.experts.w1.weight:
        return cached[1]
    ex, sh, gate = moe.experts, moe.shared_experts, moe.gate
    c = gate._config
    ok = False
    try:
        w1, w3, w2 = ex.w1, ex.w3, ex.w2
        routed = all(isinstance(p, QuantizedProjection) and p.mode == "mxfp4" and p.group_size == 32
                     and p.bits == 4 and p.quantize_input and p.get("biases") is None and p.weight.ndim == 3
                     for p in (w1, w3, w2))
        n, k = w1.weight.shape[1], w1.weight.shape[2] * 8
        ok = (routed and w3.weight.shape == w1.weight.shape and k % 512 == 0 and n % 32 == 0
              and w2.weight.shape[2] * 8 == n and w2.weight.shape[1] % 8 == 0
              and all(_mxfp8(p) for p in (sh.w1, sh.w3, sh.w2))
              and sh.w1.weight.shape == (n, k // 4) and sh.w3.weight.shape == (n, k // 4)
              and sh.w2.weight.shape == (k, n // 4) and n % 256 == 0 and k % 256 == 0
              and ex._limit == sh._limit
              and c.n_routed_experts == 384 and c.n_activated_experts == 6 and c.norm_topk_prob
              and c.score_func not in ("sigmoid", "softmax") and c.gate_temp == 1
              # The router replicas reproduce MLX's kernel choice for this GEMM shape only.
              and gate.weight.dtype == mx.bfloat16 and gate.weight.shape == (384, 5120) and k == 5120)
    except Exception:  # noqa: BLE001 -- any unexpected layout keeps the unfused path
        ok = False
    moe.__dict__["_ds41_ffn_fuse"] = (moe.experts.w1.weight, ok)
    return ok


def eligible(block, h, image_mask, max_rows=5):
    """hc_fuse.block_forward's FFN on the DS41_MHC decode path (the caller checked
    hc_fuse.eligible). Singleton rows stop at 5: MLX's quantized_matmul takes the
    shared expert at 6-8 rows (fast_qmv.MIN_ROWS..5 otherwise); og_fused passes
    max_rows=MAX_ROWS (its shared expert always uses the rows kernel)."""
    from .language import DS41_MHC
    return (
        ENABLED
        and DS41_MHC
        and moe_decode.ENABLED
        and fast_qmv.ENABLED
        and fast_qmv.ENABLED_M1
        and image_mask is None
        and 1 <= h.shape[1] <= max_rows
        and _static_ok(block.ffn)
    )


def pre_norm_q(h, pre, fn, weight, proj_eps, norm_eps):
    """(hc_project mix, FP32 pre-norm rows, their FP8 round trip) in one launch."""
    rows, d = h.size // (4 * h.shape[-1]), h.shape[-1]
    j = NORM_SPLIT_1 if rows == 1 else NORM_SPLIT
    return _pre_norm_q_kernel()(
        inputs=[h, pre, fn, weight, hc_fuse._consts(float(proj_eps), float(norm_eps))],
        template=[("T", h.dtype), ("D", d), ("DH", 4 * d), ("J", j)],
        grid=(rows * 256, 24 + j, 1), threadgroup=(256, 1, 1),
        output_shapes=[(*h.shape[:-2], 24), (*h.shape[:-2], d), (*h.shape[:-2], d)],
        output_dtypes=[mx.float32, mx.float32, h.dtype])


def shared_router(moe, xq, xf):
    """(shared expert gate/up + SwiGLU/FP8 [rows, N], router logits [rows, 384]) in one launch.

    xq: [rows, K] FP8 round trip; xf: [rows, K] FP32 (rows <= 16)."""
    sh, gate = moe.shared_experts, moe.gate
    rows, k = xq.shape
    n = sh.w1.weight.shape[0]
    nr = gate.weight.shape[0]
    rtg = nr // 16 if rows == 1 else nr // 8
    return _shared_router_kernel()(
        inputs=[xq, sh.w1.weight, sh.w1.scales, sh.w3.weight, sh.w3.scales, _limit(sh._limit), gate.weight, xf],
        template=[("T", xq.dtype), ("K", k), ("N", n), ("M", rows), ("NR", nr), ("RTG", rtg)],
        grid=(32 * (rtg + n // 32), 16, 1), threadgroup=(32, 16, 1),
        output_shapes=[(rows, n), (rows, nr)], output_dtypes=[xq.dtype, mx.float32])


def routed_up(moe, xq, ids, weights):
    """Routed gate/up + weighted SwiGLU/FP8: [pairs, N]."""
    ex = moe.experts
    rows, k = xq.size // xq.shape[-1], xq.shape[-1]
    n = ex.w1.weight.shape[1]
    pairs = ids.size
    return _routed_up_kernel()(
        inputs=[xq, ex.w1.weight, ex.w1.scales, ex.w3.weight, ex.w3.scales, ids, weights, _limit(ex._limit)],
        template=[("T", xq.dtype), ("K", k), ("N", n), ("TOPK", pairs // rows)],
        grid=(32 * pairs, 16 * (n // 32), 1), threadgroup=(32, 16, 1),
        output_shapes=[(pairs, n)], output_dtypes=[xq.dtype])[0]


def down(moe, yr, ys, ids):
    """Routed + shared down projections: ([pairs, D], [rows, D])."""
    ex, sh = moe.experts, moe.shared_experts
    pairs, rows, n = yr.shape[0], ys.shape[0], yr.shape[-1]
    k = ex.w2.weight.shape[1]
    slots = 4 if rows == 1 else 2
    return _down_kernel()(
        inputs=[yr, ex.w2.weight, ex.w2.scales, ids, ys, sh.w2.weight, sh.w2.scales],
        template=[("T", yr.dtype), ("K", n), ("N", k), ("PAIRS", pairs), ("M", rows)],
        grid=(32 * (pairs + slots), (k // 8) * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(pairs, k), (rows, k)], output_dtypes=[yr.dtype, yr.dtype])


def post_combine(routed, shared, residual, mix, scale, base, hc_eps, iters, topk):
    """hc_fuse.post_mix(moe_decode.combine(routed, shared), residual, mix, ...)."""
    return _post_combine_kernel()(
        inputs=[routed, shared, residual, mix, scale, base, hc_fuse._consts(float(hc_eps))],
        template=[("T", residual.dtype), ("D", residual.shape[-1]), ("ITERS", max(1, iters)), ("TOPK", topk)],
        grid=(shared.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[residual.shape, residual.shape[:-1]],
        output_dtypes=[residual.dtype, mx.float32])


def ffn_forward(block, h, pre, bounds=None):
    """The FFN half of hc_fuse.block_forward (h, pre after the attention post_mix).

    bounds: og_fused row ranges, one per request (each request's router rows are
    computed with its own MLX kernel's arithmetic; rows are independent)."""
    c = block._config
    moe = block.ffn
    gate = moe.gate
    mix, xf, xq = pre_norm_q(h, pre, block.hc_ffn_fn, block.ffn_norm.weight, c.norm_eps, block.ffn_norm.eps)
    rows = xq.size // xq.shape[-1]
    if bounds is not None and rows == 1:
        raise ValueError("og_fused rows are never single")
    ys, raw = shared_router(moe, xq.reshape(rows, -1), xf.reshape(rows, -1))
    ids, weights = decode_fusions.router(raw.reshape(*xf.shape[:-1], -1), gate.bias, c.route_scale)
    flat = ids.reshape(-1)
    yr = routed_up(moe, xq.reshape(rows, -1), flat, weights.reshape(-1))
    routed, shared = down(moe, yr, ys, flat)
    return post_combine(routed, shared, h, mix, block.hc_ffn_scale, block.hc_ffn_base, c.hc_eps,
                        c.hc_sinkhorn_iters, ids.shape[-1])
