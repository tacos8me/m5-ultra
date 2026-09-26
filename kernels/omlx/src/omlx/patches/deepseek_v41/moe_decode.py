# SPDX-License-Identifier: MIT
"""MXFP4 routed experts for short decode/verify blocks (1-5 rows).

Replaces MLX's three gather_qmv launches (gate, up, down) with two: gate and
up share one launch, and every kernel dispatches the pair index fastest, so the
pairs of an expert chosen by several verify rows read the same weight rows at
about the same time and the duplicates come from cache. Each output keeps
MLX's per-pair arithmetic: the lane -> K mapping, qdot expression, E8M0/E2M1
decode, per-group scale and simd_sum of fp_qmv_fast (gate/up, K % 512 == 0)
and fp_qmv (down, K = 2304), so results are bitwise those of
mx.gather_qmm(..., mode="mxfp4"). (An expert-major variant that kept all pairs
of an expert in one threadgroup was slower: register pressure.)
"""

import os
from functools import cache

import mlx.core as mx

ENABLED = os.environ.get("DS41_MOE_FUSED", "1") == "1"

_HEADER = r"""
inline float ds41_fp4(uint8_t bits) {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return static_cast<float>(bits & 8 ? -converted : converted);
}
inline float ds41_e8m0(uint8_t bits) {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
}
// qdot with the weights already decoded: identical products and sums.
template <int V, typename T>
inline float ds41_qdot4_pre(const thread T* x, const thread float* w, float scale) {
    float accum = 0;
    for (int i = 0; i < (V / 4); i++) {
        accum +=
            (float(x[4 * i]) * w[4 * i] +
             float(x[4 * i + 1]) * w[4 * i + 1] +
             float(x[4 * i + 2]) * w[4 * i + 2] +
             float(x[4 * i + 3]) * w[4 * i + 3]);
    }
    return scale * accum;
}
template <int V>
inline void ds41_decode4(const device uint8_t* w, thread float* out) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (V / 4); i++) {
        const uint16_t b = ws[i];
        out[4 * i] = ds41_fp4(b);
        out[4 * i + 1] = ds41_fp4(b >> 4);
        out[4 * i + 2] = ds41_fp4(b >> 8);
        out[4 * i + 3] = ds41_fp4(b >> 12);
    }
}
"""

_PAIR = r"""
    const uint p = threadgroup_position_in_grid.x;
    const uint e = ids[p];
"""

# fp_qmv_fast_impl, 4-bit, group 32: 16 values/thread, 512-value blocks,
# 2 simdgroups x 4 rows. grid.y blocks cover gate rows then up rows.
_GATE_UP = _PAIR + r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int in_vec_size_w = K / 2;
    constexpr int in_vec_size_g = K / 32;
    const uint blocks = N / 8;
    const bool second = threadgroup_position_in_grid.y >= blocks;
    const int out_row = (threadgroup_position_in_grid.y % blocks) * 8 + simd_gid * 4;
    const device uint8_t* ws = (const device uint8_t*)(second ? w3 : w1)
        + size_t(e) * N * in_vec_size_w + out_row * in_vec_size_w + simd_lid * 8;
    const device uint8_t* sl = (second ? s3 : s1)
        + size_t(e) * N * in_vec_size_g + out_row * in_vec_size_g + simd_lid / 2;
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
    device T* out = second ? up : gate;
    for (int row = 0; row < 4; row++) {
        const float v = simd_sum(result[row]);
        if (simd_lid == 0) out[size_t(p) * N + out_row + row] = static_cast<T>(v);
    }
"""

# fp_qmv_impl, 4-bit, group 32, K = 2304: 8 values/thread, 256-value blocks
# (eight in the loop, the ninth as MLX's tail), 2 simdgroups x 4 rows.
_DOWN = _PAIR + r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int values_per_thread = 8;
    constexpr int block_size = values_per_thread * 32;
    constexpr int in_vec_size_w = K / 2;
    constexpr int in_vec_size_g = K / 32;
    const int out_row = threadgroup_position_in_grid.y * 8 + simd_gid * 4;
    const device uint8_t* ws = (const device uint8_t*)w2
        + size_t(e) * N * in_vec_size_w + out_row * in_vec_size_w + simd_lid * 4;
    const device uint8_t* sl = s2 + size_t(e) * N * in_vec_size_g + out_row * in_vec_size_g + simd_lid / 4;
    const device T* xs = x + size_t(p) * K + simd_lid * values_per_thread;
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
        if (simd_lid == 0) y[size_t(p) * N + out_row + row] = static_cast<T>(v);
    }
"""


@cache
def _gate_up_kernel():
    return mx.fast.metal_kernel(
        name="ds41_mxfp4_moe_gate_up",
        input_names=["x", "w1", "s1", "w3", "s3", "ids"],
        output_names=["gate", "up"],
        source=_GATE_UP,
        header=_HEADER,
    )


@cache
def _down_kernel():
    return mx.fast.metal_kernel(
        name="ds41_mxfp4_moe_down",
        input_names=["x", "w2", "s2", "ids"],
        output_names=["y"],
        source=_DOWN,
        header=_HEADER,
    )


def supported(expert, x, indices):
    """Short blocks of the original checkpoint's MXFP4 experts (gate/up K % 512, down K 2304)."""
    if not ENABLED or x.dtype != mx.bfloat16 or mx.default_device() != mx.gpu:
        return False
    w1, w3, w2 = expert.w1, expert.w3, expert.w2
    if not all(getattr(p, "mode", None) == "mxfp4" and p.group_size == 32 and p.bits == 4
               and p.get("biases") is None and p.weight.ndim == 3 for p in (w1, w3, w2)):
        return False
    n, k = w1.weight.shape[1], w1.weight.shape[2] * 8
    rows = indices.size // indices.shape[-1]
    return (w3.weight.shape == w1.weight.shape and k % 512 == 0 and n % 8 == 0
            and w2.weight.shape[1] % 8 == 0 and w2.weight.shape[2] * 8 == n
            and n % 256 == 0 and (n // 256) >= 2 and 1 <= rows <= 5)


def gate_up(xq, w1, w3, ids, topk):
    """xq: [rows, K] quantized activations; ids: [rows * topk] uint32 expert ids."""
    n, k = w1.weight.shape[1], w1.weight.shape[2] * 8
    pairs = ids.size
    gate, up = _gate_up_kernel()(
        inputs=[xq, w1.weight, w1.scales, w3.weight, w3.scales, ids],
        template=[("T", xq.dtype), ("K", k), ("N", n), ("TOPK", topk)],
        grid=(32 * pairs, (n // 8) * 2 * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, 1, n), (pairs, 1, n)],
        output_dtypes=[xq.dtype, xq.dtype],
    )
    return gate, up


def down(y, w2, ids):
    """y: [pairs, 1, K] activations; returns [pairs, 1, N]."""
    n, k = w2.weight.shape[1], w2.weight.shape[2] * 8
    pairs = ids.size
    return _down_kernel()(
        inputs=[y, w2.weight, w2.scales, ids],
        template=[("T", y.dtype), ("K", k), ("N", n)],
        grid=(32 * pairs, (n // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, 1, n)],
        output_dtypes=[y.dtype],
    )[0]


# (routed.astype(f32).sum(-2) + shared.astype(f32)).astype(bf16) in one pass:
# MLX's reduction over the expert axis adds the rows in order from 0, which
# this kernel repeats (checked bitwise against MLX at FP32 and BF16).
_COMBINE = r"""
    const uint i = thread_position_in_grid.x;
    const uint r = i / D, d = i % D;
    float acc = 0.0f;
    for (int j = 0; j < TOPK; j++) acc += float(routed[(r * TOPK + j) * D + d]);
    y[i] = static_cast<T>(acc + float(shared[i]));
"""


@cache
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="ds41_moe_combine",
        input_names=["routed", "shared"],
        output_names=["y"],
        source=_COMBINE,
    )


def combine(routed, shared):
    """routed: [..., topk, D] BF16 expert outputs; shared: [..., D] BF16."""
    if (not ENABLED or routed.dtype != mx.bfloat16 or shared.dtype != mx.bfloat16
            or routed.shape[:-2] != shared.shape[:-1] or routed.shape[-1] != shared.shape[-1]
            or shared.size // shared.shape[-1] > 8 or mx.default_device() != mx.gpu):
        # Validated against MLX's reduction for decode/verify rows only.
        return None
    return _combine_kernel()(
        inputs=[routed, shared],
        template=[("T", shared.dtype), ("D", shared.shape[-1]), ("TOPK", routed.shape[-2])],
        grid=(shared.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[shared.dtype],
    )[0]
