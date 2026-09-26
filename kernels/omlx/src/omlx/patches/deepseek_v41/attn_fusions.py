# SPDX-License-Identifier: MIT
"""Fused decode kernels for the attention sublayer (1..8 rows), bitwise equal to the unfused ops.

Every kernel replaces a chain of short dependent dispatches (about 4 us each on
M5 Ultra) with one, and keeps each output element's arithmetic:

* rms_quant: RMSNorm (MLX row_reduce_looped Sum order over f*f, then the
  compiled mean/eps/rsqrt/weight chain) and, optionally, the FP8 activation
  round trip of its bf16 output (activation._ROUND) for the MXFP8 projections.
* kv_rows: kv_norm -> partial RoPE (fast_rope) -> pack_fp8 (decode_fusions),
  written after the retained window rows (the concatenate).
* the attention merge applies the inverse output RoPE (kernels.py).
"""

import os
from functools import cache

import mlx.core as mx

from .decode_fusions import _MINIMUM_LITERAL

ENABLED = os.environ.get("DS41_ATTN_FUSE", "1") == "1"
# Decode index top-k (layer-20 keys and blocks, candidate layers) by select_rows.
DECODE_SELECT = os.environ.get("DS41_INDEX_SELECT", "1") == "1"
MAX_ROWS = 8

_HEADER = f"""
#define MINIMUM {_MINIMUM_LITERAL}
""" + r"""
inline float ds41_maximum(float x, float y) { if (metal::isnan(x)) return x; return x > y ? x : y; }
inline float ds41_minimum(float x, float y) { if (metal::isnan(x)) return x; return x < y ? x : y; }
inline float ds41_pow2(float e) { return as_type<float>(uint32_t(e + 127.0f) << 23); }
inline uint8_t ds41_to_fp8(float f) {
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(f);
    uint32_t sign = f_bits & 0x80000000;
    uint8_t bits;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        bits = 0x7E;
    } else if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
    }
    bits |= static_cast<uint8_t>(sign >> 24);
    return bits;
}
// activation._ROUND: FP8 E4M3 round trip with a power-of-two group scale.
inline float ds41_fp8_round(float v, float amax_in) {
    const float amax = max(amax_in, 0x1.cp-118f);
    const int scale_exponent = max(int(ceil(log2(amax / 448.0f))), -126);
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    const float scaled = clamp(v / scale, -448.0f, 448.0f);
    const float a = abs(scaled);
    const int step_exponent = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);
    const float step = as_type<float>(uint(step_exponent + 127) << 23);
    const float q = sign(scaled) * min(rint(a / step) * step, 448.0f);
    return q * scale;
}
// MLX row_reduce_looped Sum of f*f over one contiguous row (REDUCE_N_READS 4):
// thread lid adds its 4-value runs of each TG*4 block, then simd_sum per
// simdgroup and simd_sum of the simdgroup totals in simdgroup 0.
template <typename T, int D, int TG>
inline float ds41_row_sumsq(const device T* row, uint lid, threadgroup float* shared) {
    constexpr int BLOCKS = D / (TG * 4), EXTRA = D - BLOCKS * TG * 4;
    float total = 0.0f;
    const device T* p = row + lid * 4;
    for (int b = 0; b < BLOCKS; ++b) {
        for (int i = 0; i < 4; ++i) {
            const float f = float(p[i]);
            const float sq = ds41_square(f);
            total = sq + total;
        }
        p += TG * 4;
    }
    const int index = int(lid) * 4;
    if (index + 4 <= EXTRA) {
        for (int i = 0; i < 4; ++i) { const float f = float(p[i]); const float sq = ds41_square(f); total = sq + total; }
    } else {
        for (int i = 0; index + i < EXTRA; ++i) { const float f = float(p[i]); const float sq = ds41_square(f); total = sq + total; }
    }
    float result = 0.0f + total;
    result = simd_sum(result);
    if (TG > 32) {
        const uint lane = lid % 32, sg = lid / 32;
        if (lane == 0) shared[sg] = result;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const float value = lid < uint(TG / 32) ? shared[lid] : 0.0f;
            result = simd_sum(value);
            if (lid == 0) shared[32] = result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        result = shared[32];
    }
    return result;
}
"""

# f*f is a separate MLX Multiply kernel: keep the product rounded before the add.
_SQUARE = r"""
inline float ds41_square(float f) {
#pragma clang fp contract(off)
    return f * f;
}
"""


def _tg(d):
    """MLX threadgroup_size_from_row_size."""
    if d <= 512:
        return 32
    if d <= 1024:
        return 128
    return min(1024, ((d + 3) // 4 + 31) // 32 * 32)


@cache
def _rms_quant_kernel():
    return mx.fast.metal_kernel(
        name="ds41_attn_rms_quant",
        input_names=["x", "weight", "consts"],
        output_names=["y", "yq"],
        header=_SQUARE + _HEADER,
        source=r"""
        const uint lid = thread_index_in_threadgroup;
        const uint row = threadgroup_position_in_grid.y;
        threadgroup float shared[33];
        const device T* xr = x + size_t(row) * D;
        const float total = ds41_row_sumsq<T, D, TG>(xr, lid, shared);
        const float r = metal::precise::rsqrt(total * consts[0] + consts[1]);
        // Each thread owns 4 consecutive values; 8 threads cover one FP8 group of 32.
        for (int base = int(lid) * 4; base < D; base += TG * 4) {
            float out[4];
            float amax = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const float f = float(xr[base + i]);
                const float n = f * r;
                out[i] = float(T(n * float(weight[base + i])));
                y[size_t(row) * D + base + i] = T(out[i]);
                amax = max(amax, abs(out[i]));
            }
            if (QUANT) {
                amax = max(amax, simd_shuffle_xor(amax, 1));
                amax = max(amax, simd_shuffle_xor(amax, 2));
                amax = max(amax, simd_shuffle_xor(amax, 4));
                for (int i = 0; i < 4; ++i)
                    yq[size_t(row) * D + base + i] = T(ds41_fp8_round(out[i], amax));
            }
        }
        """,
    )


def rms_supported(x, weight):
    d = x.shape[-1]
    tg = _tg(d)
    return (
        ENABLED
        and x.dtype == mx.bfloat16
        and x.ndim == 3
        and x.shape[0] == 1
        and 1 <= x.shape[1] <= MAX_ROWS
        and d % 32 == 0
        and d % (tg * 4) == 0
        and weight.shape == (d,)
        and mx.default_device() == mx.gpu
    )


def rms_quant(x, weight, eps, quant):
    """(norm(x, weight, eps), quantize_activation(that) if quant) in one dispatch."""
    d, rows = x.shape[-1], x.shape[1]
    tg = _tg(d)
    y, yq = _rms_quant_kernel()(
        inputs=[x, weight, mx.array([1.0 / d, eps], mx.float32)],
        template=[("T", x.dtype), ("D", d), ("TG", tg), ("QUANT", bool(quant))],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape, x.shape if quant else (1,)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return y, (yq if quant else None)


_KV_SOURCE = r"""
        // D threads per row: rows [0, OLD) copy the retained window, the rest
        // normalize, rotate and pack one new row each (thread d owns value d).
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.y;
        constexpr int W = D + D / 32;
        const int OLD = meta[0];
        if (int(row) < OLD) {
            for (int i = tid; i < W; i += D) kv[size_t(row) * W + i] = old[size_t(row) * W + i];
            return;
        }
        const uint l = row - OLD;
        threadgroup float shared[33];
        const device T* xr = x + size_t(l) * D;
        if (tid < 32) {
            // The MLX reduction runs in one simdgroup for D <= 512.
            const float total = ds41_row_sumsq<T, D, 32>(xr, tid, shared);
            if (tid == 0) shared[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float r = metal::precise::rsqrt(shared[0] * consts[0] + consts[1]);
        const uint d = tid;
        const float n = float(xr[d]) * r;
        const float normed = float(T(n * float(weight[d])));
        const float other = simd_shuffle_xor(normed, ushort(1));
        constexpr uint KEEP = (D - R) / 2;
        const uint p = d / 2;
        float rotated = normed;
        if (p >= KEEP) {
            const uint j = l * (R / 2) + (p - KEEP);
            float c = cos_t[j];
            float s = sin_t[j];
            float a = (d & 1) ? other : normed;
            float b2 = (d & 1) ? normed : other;
            float a_c = a * c;
            float b_s = b2 * s;
            float a_s = a * s;
            float b_c = b2 * c;
            rotated = (d & 1) ? float(T(a_s + b_c)) : float(T(a_c - b_s));
        }
        // decode_fusions.pack_fp8: one simdgroup per 32-value scale group.
        const float f = rotated;
        const float amax = ds41_maximum(simd_max(metal::abs(f)), static_cast<float>(MINIMUM));
        const float exponent = ds41_maximum(metal::ceil(metal::precise::log2(amax / 448.0f)), -126.0f);
        const float scaled = ds41_minimum(ds41_maximum(f / ds41_pow2(exponent), -448.0f), 448.0f);
        const float a = ds41_minimum(metal::abs(scaled), 448.0f);
        const float step_exponent = metal::floor(metal::precise::log2(ds41_maximum(a, 0x1p-9f)));
        const float step = ds41_pow2(ds41_maximum(step_exponent - 3.0f, -9.0f));
        const float sgn = float(int(scaled > 0.0f) - int(scaled < 0.0f));
        const float q = sgn * ds41_minimum(metal::rint(a / step) * step, 448.0f);
        const size_t out_row = size_t(row) * W;
        kv[out_row + d] = ds41_to_fp8(q);
        if (lane == 0) kv[out_row + D + d / 32] = uint8_t(exponent + 127.0f);
"""


@cache
def _kv_rows_kernel(source=None):
    return mx.fast.metal_kernel(
        name="ds41_attn_kv_rows" + ("" if source is None else str(abs(hash(source)))),
        input_names=["x", "weight", "consts", "cos_t", "sin_t", "old", "meta"],
        output_names=["kv"],
        header=_SQUARE + _HEADER,
        source=_KV_SOURCE if source is None else source,
    )


def kv_supported(kv_input, weight, old_len):
    d = kv_input.shape[-1]
    return (
        ENABLED
        and kv_input.dtype == mx.bfloat16
        and kv_input.ndim == 3
        and kv_input.shape[0] == 1
        and 1 <= kv_input.shape[1] <= MAX_ROWS
        and d == 512
        and weight.shape == (d,)
        and mx.default_device() == mx.gpu
    )


def kv_rows(kv_input, weight, eps, cos, sin, rope_dim, old, old_len):
    """concatenate([old[:, :old_len], pack_fp8(rope(norm(kv_input)))], 1) in one dispatch."""
    d, rows = kv_input.shape[-1], kv_input.shape[1]
    width = d + d // 32
    return _kv_rows_kernel()(
        inputs=[
            kv_input,
            weight,
            mx.array([1.0 / d, eps], mx.float32),
            cos,
            sin,
            old if old_len else mx.zeros((16,), mx.uint8),
            mx.array([old_len, 0], mx.int32),
        ],
        template=[("T", kv_input.dtype), ("D", d), ("R", rope_dim)],
        grid=(d, old_len + rows, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(1, old_len + rows, width)],
        output_dtypes=[mx.uint8],
    )[0]


_FP4_MINIMUM_LITERAL = "%.7g" % (6.0 * 2.0**-126)


@cache
def _index_q_kernel():
    # rope_range(q, ...) (fast_rope) then quantization._quantize_activation(bits=4)
    # (the compiled MLX graph: NaN-propagating maximum/minimum, precise log2,
    # 7-digit minimum literal, (x>0)-(x<0) sign, ascending E2M1 thresholds).
    return mx.fast.metal_kernel(
        name="ds41_attn_index_q",
        input_names=["q", "cos_t", "sin_t"],
        output_names=["y"],
        header=f"#define FP4_MINIMUM {_FP4_MINIMUM_LITERAL}\n" + _SQUARE + _HEADER,
        source=r"""
        const uint gid = thread_position_in_grid.x;
        const uint d = gid % D;
        const uint l = gid / (D * H);
        float v = float(q[gid]);
        if (d >= D - R) {
            const float other = simd_shuffle_xor(v, ushort(1));
            const uint j = l * (R / 2) + (d / 2 - (D - R) / 2);
            float c = cos_t[j];
            float s = sin_t[j];
            float a = (d & 1) ? other : v;
            float b2 = (d & 1) ? v : other;
            float a_c = a * c;
            float b_s = b2 * s;
            float a_s = a * s;
            float b_c = b2 * c;
            v = (d & 1) ? float(T(a_s + b_c)) : float(T(a_c - b_s));
        }
        const float amax = ds41_maximum(simd_max(metal::abs(v)), static_cast<float>(FP4_MINIMUM));
        const float exponent = ds41_maximum(metal::ceil(metal::precise::log2(amax / 6.0f)), -126.0f);
        const float scale = as_type<float>(uint32_t(exponent + 127.0f) << 23);
        const float scaled = ds41_minimum(ds41_maximum(v / scale, -6.0f), 6.0f);
        const float a = metal::abs(scaled);
        float level = 0.0f;
        level = a > 0.25f ? 0.5f : level;
        level = a >= 0.75f ? 1.0f : level;
        level = a > 1.25f ? 1.5f : level;
        level = a >= 1.75f ? 2.0f : level;
        level = a > 2.5f ? 3.0f : level;
        level = a >= 3.5f ? 4.0f : level;
        level = a > 5.0f ? 6.0f : level;
        const float sgn = float(int(scaled > 0.0f) - int(scaled < 0.0f));
        const float quantized = sgn * level;
        y[gid] = T(quantized * scale);
        """,
    )


def index_q_supported(q, rope_dim):
    return (
        ENABLED
        and q.dtype == mx.bfloat16
        and q.ndim == 4
        and q.shape[0] == 1
        and 1 <= q.shape[1] <= MAX_ROWS
        and q.shape[-1] % 32 == 0
        and rope_dim % 64 == 0
        and 0 < rope_dim <= q.shape[-1]
        and mx.default_device() == mx.gpu
    )


def index_q(q, cos, sin, rope_dim):
    """quantize_activation(rope(q), bits=4) for the indexer queries in one dispatch."""
    _, rows, heads, d = q.shape
    return _index_q_kernel()(
        inputs=[q, cos, sin],
        template=[("T", q.dtype), ("D", d), ("H", heads), ("R", rope_dim)],
        grid=(q.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


# Exact top-k of one row per threadgroup (W <= 1024 * E values): radix select
# of the k-th largest (score desc, position asc) key, then an ordered
# compaction. Returns what the decode index paths return after their top-k:
# the selected ids in ascending order with -1 (first) for -inf selections.
# Valid ids ascend with position (key positions, block ids or the sorted
# candidate lists), so position order is id order.
_SELECT = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const int SW = meta[0], W = meta[1], K = meta[2], offset = meta[3], start = meta[4];
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[32];
    threadgroup uint state[2];
    uint keys[E];
    bool finite[E];
    for (int e = 0; e < E; ++e) {
        const int pos = int(tid) * E + e;
        // Keys are built from bits (fast math may fold infinity compares):
        // key(-inf) = 0x007FFFFF, key(+inf) = 0xFF800000, -0 and +0 share one key.
        uint u = 0xFF800000u;
        bool forced = false;
        if (pos < W) {
            if (BLOCK > 1) {
                bool any = false;
                float v = 0.0f;
                for (int b = 0; b < BLOCK; ++b) {
                    const int src = pos * BLOCK + b;
                    if (src < SW) {
                        const float s = scores[size_t(row) * SW + src];
                        v = any ? max(v, s) : s;
                        any = true;
                    }
                }
                u = any ? as_type<uint>(v) : 0xFF800000u;
                const int length = (start + int(row) + 1) / RATIO;
                forced = FORCE && length > 0 && pos + offset == (length - 1) / BLOCK;
            } else {
                u = as_type<uint>(scores[size_t(row) * SW + pos]);
            }
        }
        if ((u & 0x7FFFFFFFu) == 0u) u = 0u;
        keys[e] = forced ? 0xFF800000u : ((u & 0x80000000u) ? ~u : (u | 0x80000000u));
        finite[e] = pos < W && (forced || u != 0xFF800000u);
    }
    uint prefix = 0, mask = 0, remaining = uint(K);
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += TG) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = 0; e < E; ++e) {
            const int pos = int(tid) * E + e;
            if (pos < W && (keys[e] & mask) == prefix)
                atomic_fetch_add_explicit(&hist[(keys[e] >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            // Lane l sums bins 255-8l .. 248-8l (top down).
            uint counts[8], total = 0;
            for (int i = 0; i < 8; ++i) {
                counts[i] = atomic_load_explicit(&hist[255 - 8 * lane - i], memory_order_relaxed);
                total += counts[i];
            }
            const uint before = simd_prefix_exclusive_sum(total);
            if (before < remaining && remaining <= before + total) {
                uint acc = before;
                for (int i = 0; i < 8; ++i) {
                    if (remaining <= acc + counts[i]) {
                        state[0] = prefix | ((255u - 8u * lane - uint(i)) << shift);
                        state[1] = remaining - acc;
                        break;
                    }
                    acc += counts[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = state[0];
        remaining = state[1];
        mask |= 255u << shift;
    }
    // prefix = the k-th key; take every larger key and the first `remaining` equal ones.
    uint eq = 0;
    for (int e = 0; e < E; ++e) eq += (int(tid) * E + e < W && keys[e] == prefix) ? 1u : 0u;
    uint eq_before = simd_prefix_exclusive_sum(eq);
    if (lane == 31) scan[sg] = eq_before + eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint v = lane < TG / 32 ? scan[lane] : 0u;
        scan[lane] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    eq_before += scan[sg];
    bool take[E];
    uint fin = 0;
    for (int e = 0; e < E; ++e) {
        const int pos = int(tid) * E + e;
        bool t = false;
        if (pos < W) {
            if (keys[e] > prefix) t = true;
            else if (keys[e] == prefix) { t = eq_before < remaining; ++eq_before; }
        }
        take[e] = t;
        fin += (t && finite[e]) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint fin_before = simd_prefix_exclusive_sum(fin);
    if (lane == 31) scan[sg] = fin_before + fin;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint v = lane < TG / 32 ? scan[lane] : 0u;
        const uint excl = simd_prefix_exclusive_sum(v);
        scan[lane] = excl;
        if (lane == TG / 32 - 1) state[0] = excl + v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fin_before += scan[sg];
    const uint negatives = uint(K) - state[0];
    for (uint i = tid; i < negatives; i += TG) out[size_t(row) * K + i] = -1;
    for (int e = 0; e < E; ++e) {
        if (take[e] && finite[e]) {
            const int pos = int(tid) * E + e;
            out[size_t(row) * K + negatives + fin_before] = EXPLICIT ? ids[size_t(row) * W + pos] : offset + pos;
            ++fin_before;
        }
    }
"""


@cache
def _select_kernel():
    return mx.fast.metal_kernel(
        name="ds41_attn_select_rows",
        input_names=["scores", "ids", "meta"],
        output_names=["out"],
        source=_SELECT,
    )


SELECT_MAX = 32768


def select_supported(width, count):
    return DECODE_SELECT and 0 < count <= width <= SELECT_MAX and mx.default_device() == mx.gpu


def select_rows(scores, count, *, ids=None, offset=0, block_size=1, force_latest=False, start=0, ratio=1):
    """sort(where(top_value > -inf, top_id, -1)) of the count best (value desc, position asc).

    Values are scores, or with block_size > 1 the block maxima (the latest
    causal block forced to +inf), as kernels._tile_topk ranks them.
    """
    rows, score_width = scores.shape[1], scores.shape[-1]
    width = (score_width + block_size - 1) // block_size
    per = 1
    while per * 1024 < width:
        per *= 2
    tg = min(1024, max(32, (width + per - 1) // per + 31) // 32 * 32)
    return _select_kernel()(
        inputs=[
            scores,
            ids if ids is not None else mx.zeros((1,), mx.int32),
            mx.array([score_width, width, count, offset, start], mx.int32),
        ],
        template=[("E", per), ("TG", tg), ("BLOCK", block_size), ("FORCE", bool(force_latest)),
                  ("RATIO", ratio), ("EXPLICIT", ids is not None)],
        grid=(tg, rows, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(1, rows, count)],
        output_dtypes=[mx.int32],
    )[0]
