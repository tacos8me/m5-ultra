# SPDX-License-Identifier: Apache-2.0
"""Packed CSA2 Metal kernels, following local DSV4 WSDPA and DSA scan patterns.

BF16 attention uses 64-key online maxima and BF16 PV probabilities.
Other floating-point inputs retain the split-K FP32 probability path.
Index scoring fuses the head reduction and optionally visits explicit candidates.
The matrix fragment layout follows MLX-derived Bonsai kernels (Apple Inc.). No checkpoint or
compiled extension is required; compilation/dispatch errors propagate.
"""

from functools import cache
import os

DS41_SPARSE = os.environ.get("DS41_SPARSE", "0") == "1"

DS41_INDEX_NAX = os.environ.get("DS41_INDEX_NAX", "0") == "1"
DS41_PREFILL_INDEX = int(os.environ.get("DS41_PREFILL_INDEX", "2"))
DS41_DECODE_SINGLE_TILE = os.environ.get("DS41_DECODE_SINGLE_TILE", "1") == "1"
DS41_INDEX_ROWS = os.environ.get("DS41_INDEX_ROWS", "1") == "1"
DS41_DECODE_RADIX = os.environ.get("DS41_DECODE_RADIX", "1") == "1"
DS41_PREFILL_MID_INDEX = int(os.environ.get("DS41_PREFILL_MID_INDEX", "1"))

import mlx.core as mx

from . import decode_fusions, decode_topk
from .packed_attention import rounded_packed_attention

_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
inline float v41_fp8(uchar code) {
    const uint a = code & 127, exponent = a >> 3, mantissa = a & 7;
    const float value = exponent == 0 ? ldexp(float(mantissa), -9)
        : ldexp(1.0f + float(mantissa) * 0.125f, int(exponent) - 7);
    return code & 128 ? -value : value;
}
// Bitwise-equal v41_fp8 without ldexp: e4m3 values are built from their bits.
inline float v41_fp8_bits(uchar code) {
    const uint a = code & 127, exponent = a >> 3, mantissa = a & 7;
    const float value = exponent == 0 ? float(mantissa) * 0x1p-9f
        : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
    return code & 128 ? -value : value;
}
constant float v41_fp4_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
inline float v41_fp4_lut(uchar code) {
    return (code & 8 ? -1.0f : 1.0f) * v41_fp4_table[code & 7];
}
inline float v41_fp4(uchar code) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (code & 8 ? -1.0f : 1.0f) * table[code & 7];
}
"""

_ATTENTION = r"""
    const uint lane = thread_index_in_threadgroup % 32;
    const uint head = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const uint split = threadgroup_position_in_grid.z;
    const int H = meta[0], W = meta[1], C = meta[2];
    const int NW = meta[3], NC = meta[4], NS = meta[5];
    if (head >= H) return;
    // Contiguous lanes stay inside one quantization group at these widths.
    constexpr bool REUSE_SCALE = D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
    float query_values[D / 32], acc[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        query_values[v] = float(q[(query * H + head) * D + (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32)]);
        acc[v] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    const int end = min(int((split + 1) * CHUNK), W + C);
    for (int j = split * CHUNK; j < end; ++j) {
        const bool compressed = j >= W;
        const int row = compressed ? ci[query * C + j - W] : wi[query * W + j];
        if (row < 0 || row >= (compressed ? NC : NW)) continue;
        const size_t base = size_t(row) * (compressed
            ? (D / 2 + D / 16) : (D + D / 32));
        const uint first = lane * (D / 32);
        const float pooled_scale = REUSE_SCALE && compressed
            ? v41_fp8(pooled[base + D / 2 + first / 16]) : 0.0f;
        const int window_exp = REUSE_SCALE && !compressed
            ? int(window[base + D + first / 32]) - 127 : 0;
        float values[D / 32], dot = 0.0f;
        for (int v = 0; v < D / 32; ++v) {
            const uint d = (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32);
            float value;
            if (compressed) {
                const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;
                value = v41_fp4(code) * (REUSE_SCALE ? pooled_scale : v41_fp8(pooled[base + D / 2 + d / 16]));
            } else {
                value = ldexp(v41_fp8(window[base + d]), REUSE_SCALE ? window_exp : int(window[base + D + d / 32]) - 127);
            }
            values[v] = value;
            dot += query_values[v] * value;
        }
        const float score = simd_sum(dot) * scalep[0];
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int v = 0; v < D / 32; ++v)
            acc[v] = acc[v] * correction + probability * values[v];
        maximum = next_maximum;
    }
    const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);
    for (int v = 0; v < D / 32; ++v) partial[base + (REUSE_SCALE ? lane * (D / 32) + v : lane + v * 32)] = acc[v];
    if (lane == 0) {
        partial[base + D] = maximum;
        partial[base + D + 1] = denominator;
    }
"""

# Bitwise-equal short-query attention for D=512: decodes each key with one
# vector load, branch-free per-value e4m3/e2m1 bit decoding and the packed
# layout's per-lane scale hoisted out of the value loop. Per-key arithmetic
# and its order are unchanged from _ATTENTION.
_ATTENTION_FASTDEC = r"""
    const uint lane = thread_index_in_threadgroup % 32;
    const uint head = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const uint split = threadgroup_position_in_grid.z;
    const int H = meta[0], W = meta[1], C = meta[2];
    const int NW = meta[3], NC = meta[4], NS = meta[5];
    if (head >= uint(H)) return;
    static_assert(D == 512, "fast decode attention layout");
    float query_values[D / 32], acc[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        query_values[v] = float(q[(query * H + head) * D + lane * (D / 32) + v]);
        acc[v] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    const uint first = lane * (D / 32);
    const int end = min(int((split + 1) * CHUNK), W + C);
    for (int j = split * CHUNK; j < end; ++j) {
        const bool compressed = j >= W;
        const int row = compressed ? ci[query * C + j - W] : wi[query * W + j];
        if (row < 0 || row >= (compressed ? NC : NW)) continue;
        float values[D / 32], dot = 0.0f;
        if (compressed) {
            const size_t base = size_t(row) * (D / 2 + D / 16);
            const uint2 packed = *(const device uint2*)(pooled + base + first / 2);
            const float pooled_scale = v41_fp8_bits(pooled[base + D / 2 + first / 16]);
            for (int v = 0; v < D / 32; ++v) {
                const uint byte = ((v < 8 ? packed.x : packed.y) >> (8 * ((v / 2) % 4))) & 255u;
                const uchar code = uchar((byte >> ((v % 2) * 4)) & 15u);
                const float value = v41_fp4_lut(code) * pooled_scale;
                values[v] = value;
                dot += query_values[v] * value;
            }
        } else {
            const size_t base = size_t(row) * (D + D / 32);
            const uint4 packed = *(const device uint4*)(window + base + first);
            const int window_exp = int(window[base + D + first / 32]) - 127;
            if (window_exp >= -117 && window_exp <= 118) {
                // x * 2^e equals ldexp(x, e) while every e4m3 product stays normal.
                const float window_scale = as_type<float>(uint(window_exp + 127) << 23);
                for (int v = 0; v < D / 32; ++v) {
                    const uchar code = uchar((packed[v / 4] >> (8 * (v % 4))) & 255u);
                    const float value = v41_fp8_bits(code) * window_scale;
                    values[v] = value;
                    dot += query_values[v] * value;
                }
            } else {
                for (int v = 0; v < D / 32; ++v) {
                    const uchar code = uchar((packed[v / 4] >> (8 * (v % 4))) & 255u);
                    const float value = ldexp(v41_fp8_bits(code), window_exp);
                    values[v] = value;
                    dot += query_values[v] * value;
                }
            }
        }
        const float score = simd_sum(dot) * scalep[0];
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int v = 0; v < D / 32; ++v)
            acc[v] = acc[v] * correction + probability * values[v];
        maximum = next_maximum;
    }
    const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);
    for (int v = 0; v < D / 32; ++v) partial[base + first + v] = acc[v];
    if (lane == 0) {
        partial[base + D] = maximum;
        partial[base + D + 1] = denominator;
    }
"""

_ATTENTION_MMA = r"""
    const uint tid = thread_index_in_threadgroup, lane = tid % 32, sg = tid / 32;
    const uint query = threadgroup_position_in_grid.y, split = threadgroup_position_in_grid.z;
    const uint first_head = threadgroup_position_in_grid.x * 8;
    const int H = meta[0], W = meta[1], C = meta[2], NW = meta[3], NC = meta[4], NS = meta[5];
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[8 * D], dots[4 * 64], probability[64];
    threadgroup float maxima[8], denominator[8], correction[8];
    threadgroup bool valid[8];
    simdgroup_matrix<float, 8, 8> output[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        output[v].thread_elements()[0] = 0.0f;
        output[v].thread_elements()[1] = 0.0f;
    }
    if (tid < 8) { maxima[tid] = -INFINITY; denominator[tid] = 0.0f; }
    const int end = min(int((split + 1) * CHUNK), W + C);
    for (int begin = split * CHUNK; begin < end; begin += 8) {
        for (uint i = tid; i < 8 * D; i += 128) {
            const uint r = i / D, d = i % D, j = begin + r;
            const bool compressed = j >= W;
            int row = -1;
            if (j < end) row = compressed ? ci[query * C + j - W] : wi[query * W + j];
            const bool ok = row >= 0 && row < (compressed ? NC : NW);
            float value = 0.0f;
            if (ok) {
                if (compressed) {
                    const size_t base = size_t(row) * (D / 2 + D / 16);
                    const uchar code = (pooled[base + d / 2] >> ((d % 2) * 4)) & 15;
                    value = v41_fp4(code) * v41_fp8(pooled[base + D / 2 + d / 16]);
                } else {
                    const size_t base = size_t(row) * (D + D / 32);
                    value = ldexp(v41_fp8(window[base + d]), int(window[base + D + d / 32]) - 127);
                }
            }
            tile[i] = value;
            if (d == 0) valid[r] = ok;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Four simdgroups split the QK dot product's channel dimension.
        simdgroup_matrix<float, 8, 8> score, a, b;
        score.thread_elements()[0] = 0.0f;
        score.thread_elements()[1] = 0.0f;
        for (int k = sg * (D / 4); k < (sg + 1) * (D / 4); k += 8) {
            const uint head = first_head + fm;
            a.thread_elements()[0] = head < H ? float(q[(query * H + head) * D + k + fn]) : 0.0f;
            a.thread_elements()[1] = head < H ? float(q[(query * H + head) * D + k + fn + 1]) : 0.0f;
            b.thread_elements()[0] = tile[fn * D + k + fm];
            b.thread_elements()[1] = tile[(fn + 1) * D + k + fm];
            simdgroup_multiply_accumulate(score, a, b, score);
        }
        dots[sg * 64 + fm * 8 + fn] = score.thread_elements()[0];
        dots[sg * 64 + fm * 8 + fn + 1] = score.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float values[8], next_max = maxima[tid];
            for (uint r = 0; r < 8; ++r) {
                float sum = dots[tid * 8 + r] + dots[64 + tid * 8 + r]
                    + dots[128 + tid * 8 + r] + dots[192 + tid * 8 + r];
                values[r] = valid[r] ? sum * scalep[0] : -INFINITY;
                next_max = max(next_max, values[r]);
            }
            correction[tid] = maxima[tid] == -INFINITY ? 0.0f : exp(maxima[tid] - next_max);
            float total = denominator[tid] * correction[tid];
            for (uint r = 0; r < 8; ++r) {
                const float prob = valid[r] ? exp(values[r] - next_max) : 0.0f;
                probability[tid * 8 + r] = prob;
                total += prob;
            }
            maxima[tid] = next_max;
            denominator[tid] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // PV reuses the decoded tile. Float matrices retain FP32 probabilities.
        a.thread_elements()[0] = probability[fm * 8 + fn];
        a.thread_elements()[1] = probability[fm * 8 + fn + 1];
        for (int v = 0; v < D / 32; ++v) {
            const uint d = sg * (D / 4) + v * 8 + fn;
            b.thread_elements()[0] = tile[fm * D + d];
            b.thread_elements()[1] = tile[fm * D + d + 1];
            output[v].thread_elements()[0] *= correction[fm];
            output[v].thread_elements()[1] *= correction[fm];
            simdgroup_multiply_accumulate(output[v], a, b, output[v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint head = first_head + fm;
    if (head < H) {
        const size_t base = ((size_t(query) * H + head) * NS + split) * (D + 2);
        for (int v = 0; v < D / 32; ++v) {
            const uint d = sg * (D / 4) + v * 8 + fn;
            partial[base + d] = output[v].thread_elements()[0];
            partial[base + d + 1] = output[v].thread_elements()[1];
        }
    }
    if (tid < 8 && first_head + tid < H) {
        const size_t base = ((size_t(query) * H + first_head + tid) * NS + split) * (D + 2);
        partial[base + D] = maxima[tid];
        partial[base + D + 1] = denominator[tid];
    }
"""

_INDEX = r"""
    const uint lane = thread_index_in_threadgroup % 32;
    const uint position = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;
    if (position >= M) return;
    const int row = CANDIDATES ? candidates[query * M + position] : int(position);
    if (row < 0 || row >= N || row + meta[5] >= (start + int(query) + 1) / ratio) {
        if (lane == 0) scores[query * M + position] = -INFINITY;
        return;
    }
    const size_t base = size_t(row) * (D / 2 + D / 32);
    float key[D / 32];
    for (int v = 0; v < D / 32; ++v) {
        const uint d = lane + v * 32;
        const uchar code = (keys[base + d / 2] >> ((d % 2) * 4)) & 15;
        key[v] = ldexp(v41_fp4(code), int(keys[base + D / 2 + d / 32]) - 127);
    }
    float result = 0.0f;
    for (int h = 0; h < H; ++h) {
        float dot = 0.0f;
        for (int v = 0; v < D / 32; ++v)
            dot += float(q[(query * H + h) * D + lane + v * 32]) * key[v];
        result += max(simd_sum(dot), 0.0f) * weights[query * H + h];
    }
    if (lane == 0) scores[query * M + position] = result;
"""


_INDEX_MMA = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint lane = tid % 32, sg = tid / 32;
    const uint query = threadgroup_position_in_grid.y;
    constexpr uint COLS = HEAD_SPLIT ? 16 : 32;
    const uint key_group = HEAD_SPLIT ? sg % 2 : sg;
    const uint block = threadgroup_position_in_grid.x * COLS;
    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[D * COLS];
    for (uint i = tid; i < D * COLS; i += 128) {
        const uint d = i / COLS, col = i % COLS, pos = block + col;
        const int row = pos < M ? (CANDIDATES ? candidates[query * M + pos] : int(pos)) : -1;
        float value = 0.0f;
        if (row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio) {
            const size_t base = size_t(row) * (D / 2 + D / 32);
            const uchar code = (keys[base + d / 2] >> ((d % 2) * 4)) & 15;
            value = ldexp(v41_fp4(code), int(keys[base + D / 2 + d / 32]) - 127);
        }
        tile[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total0 = 0.0f, total1 = 0.0f;
    for (int h = HEAD_SPLIT ? int(sg / 2) * 8 : 0; h < H; h += HEAD_SPLIT ? 16 : 8) {
        simdgroup_matrix<float, 8, 8> acc, a, b;
        acc.thread_elements()[0] = 0.0f;
        acc.thread_elements()[1] = 0.0f;
        #pragma clang loop unroll(full)
        for (int d = 0; d < D; d += 8) {
            a.thread_elements()[0] = h + fm < H ? float(q[(query * H + h + fm) * D + d + fn]) : 0.0f;
            a.thread_elements()[1] = h + fm < H ? float(q[(query * H + h + fm) * D + d + fn + 1]) : 0.0f;
            b.thread_elements()[0] = tile[(d + fm) * COLS + key_group * 8 + fn];
            b.thread_elements()[1] = tile[(d + fm) * COLS + key_group * 8 + fn + 1];
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        const float weight = h + fm < H ? weights[query * H + h + fm] : 0.0f;
        float v0 = max(acc.thread_elements()[0], 0.0f) * weight;
        float v1 = max(acc.thread_elements()[1], 0.0f) * weight;
        // Lane bits 1, 2 and 4 span the eight head rows of each column.
        for (uint i = 0; i < 3; ++i) {
            const uint mask = i == 0 ? 2 : (i == 1 ? 4 : 16);
            v0 += simd_shuffle_xor(v0, mask);
            v1 += simd_shuffle_xor(v1, mask);
        }
        total0 += v0;
        total1 += v1;
    }
    threadgroup float head_totals[32];
    if (HEAD_SPLIT) {
        if (fm == 0) {
            head_totals[sg * 8 + fn] = total0;
            head_totals[sg * 8 + fn + 1] = total1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < COLS && block + tid < M) {
            const uint pos = block + tid;
            const int row = CANDIDATES ? candidates[query * M + pos] : int(pos);
            const bool valid = row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio;
            scores[query * M + pos] = valid ? head_totals[tid] + head_totals[tid + COLS] : -INFINITY;
        }
    } else if (fm == 0) {
        for (uint j = 0; j < 2; ++j) {
            const uint pos = block + sg * 8 + fn + j;
            if (pos >= M) continue;
            const int row = CANDIDATES ? candidates[query * M + pos] : int(pos);
            const bool valid = row >= 0 && row < N && row + meta[5] < (start + int(query) + 1) / ratio;
            scores[query * M + pos] = valid ? (j == 0 ? total0 : total1) : -INFINITY;
        }
    }
"""

# Bitwise-equal _INDEX_MMA (HEAD_SPLIT, no candidates) for short query blocks:
# one threadgroup scores 32 keys for every query row, so each key tile is read
# and decoded once instead of once per row, and each simdgroup keeps its eight
# keys' B fragments in registers. Per (query, head, key) the MMA sequence over
# D, the ReLU/weight, the three-step butterfly over each 8-head block and the
# block order ((0 + B0) + B16) + ((0 + B8) + B24) are those of _INDEX_MMA.
# Keys decode as fp4 * 2^(e-127); a multiply equals ldexp while 2^(e-127) and
# every product stay normal (3 <= e <= 254), and ldexp covers the rest.
_INDEX_MMA_ROWS = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint lane = tid % 32, sg = tid / 32;
    const uint block = threadgroup_position_in_grid.x * 32;
    const int H = HEADS, N = meta[1], M = meta[2], start = meta[3], ratio = RATIO;
    const int L = meta[6];
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float tile[D * 32];
    for (uint i = tid; i < 32 * (D / 8); i += 128) {
        const uint col = i / (D / 8), word = i % (D / 8);
        const int row = int(block + col);
        if (row < N) {
            const size_t base = size_t(row) * (D / 2 + D / 32);
            const uint bits = *((const device uint*)(keys + base) + word);
            const int e = int(keys[base + D / 2 + word / 4]);
            const bool fast = e >= 3 && e <= 254;
            const float scale = as_type<float>(uint(e) << 23);
            for (uint k = 0; k < 8; ++k) {
                const float x = v41_fp4((bits >> (4 * k)) & 15);
                tile[(word * 8 + k) * 32 + col] = fast ? x * scale : ldexp(x, e - 127);
            }
        } else {
            for (uint k = 0; k < 8; ++k) tile[(word * 8 + k) * 32 + col] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint col = sg * 8 + fn;
    float2 key[D / 8];
    for (uint s = 0; s < D / 8; ++s)
        key[s] = float2(tile[(s * 8 + fm) * 32 + col], tile[(s * 8 + fm) * 32 + col + 1]);
    for (int query = 0; query < L; ++query) {
        float2 first = 0.0f, second = 0.0f;
        for (int hb = 0; hb < 4; ++hb) {
            const int h = hb == 0 ? 0 : (hb == 1 ? 16 : (hb == 2 ? 8 : 24));
            simdgroup_matrix<float, 8, 8> acc, a, b;
            acc.thread_elements()[0] = 0.0f;
            acc.thread_elements()[1] = 0.0f;
            auto qrow = q + (size_t(query) * H + h + fm) * D + fn;
            #pragma clang loop unroll(full)
            for (uint s = 0; s < D / 8; ++s) {
                a.thread_elements()[0] = float(qrow[s * 8]);
                a.thread_elements()[1] = float(qrow[s * 8 + 1]);
                b.thread_elements()[0] = key[s].x;
                b.thread_elements()[1] = key[s].y;
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            const float weight = weights[query * H + h + fm];
            float v0 = max(acc.thread_elements()[0], 0.0f) * weight;
            float v1 = max(acc.thread_elements()[1], 0.0f) * weight;
            for (uint i = 0; i < 3; ++i) {
                const uint mask = i == 0 ? 2 : (i == 1 ? 4 : 16);
                v0 += simd_shuffle_xor(v0, mask);
                v1 += simd_shuffle_xor(v1, mask);
            }
            if (hb < 2) { first.x += v0; first.y += v1; }
            else { second.x += v0; second.y += v1; }
        }
        if (fm == 0) {
            const int limit = (start + query + 1) / ratio;
            for (uint j = 0; j < 2; ++j) {
                const uint pos = block + col + j;
                if (pos >= uint(M)) continue;
                const bool valid = int(pos) < N && int(pos) + meta[5] < limit;
                scores[size_t(query) * M + pos] = valid ? first[j] + second[j] : -INFINITY;
            }
        }
    }
"""

_MERGE = r"""
    const uint lane = thread_index_in_threadgroup % 32;
    const uint head = threadgroup_position_in_grid.x * 4
        + thread_index_in_threadgroup / 32;
    const uint query = threadgroup_position_in_grid.y;
    const int H = meta[0], NS = meta[1];
    if (head >= H) return;
    const size_t base = (size_t(query) * H + head) * NS * (D + 2);
    float maximum = float(sink[head]);
    for (int s = 0; s < NS; ++s)
        maximum = max(maximum, partial[base + s * (D + 2) + D]);
    float denominator = exp(float(sink[head]) - maximum);
    float acc[D / 32];
    for (int v = 0; v < D / 32; ++v) acc[v] = 0.0f;
    for (int s = 0; s < NS; ++s) {
        const size_t pos = base + s * (D + 2);
        const float correction = exp(partial[pos + D] - maximum);
        denominator += partial[pos + D + 1] * correction;
        for (int v = 0; v < D / 32; ++v)
            acc[v] += partial[pos + lane + v * 32] * correction;
    }
    for (int v = 0; v < D / 32; ++v)
        out[(query * H + head) * D + lane + v * 32] = T(acc[v] / denominator);
"""


@cache
def _kernel(kind):
    if kind in ("attention", "attention_mma", "attention_fastdec"):
        inputs = ["q", "window", "pooled", "wi", "ci", "meta", "scalep"]
        outputs, source = (
            ["partial"],
            {
                "attention": _ATTENTION,
                "attention_mma": _ATTENTION_MMA,
                "attention_fastdec": _ATTENTION_FASTDEC,
            }[kind],
        )
    elif kind == "merge":
        inputs = ["partial", "sink", "meta"]
        outputs, source = ["out"], _MERGE
    elif kind == "index_mma_rows":
        inputs = ["q", "keys", "weights", "meta"]
        outputs, source = ["scores"], _INDEX_MMA_ROWS
    else:
        inputs = ["q", "keys", "weights", "candidates", "meta"]
        outputs, source = ["scores"], _INDEX_MMA if kind == "index_mma" else _INDEX
    return mx.fast.metal_kernel(
        name="deepseek_v41_packed_" + kind,
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_HEADER,
    )


def packed_sparse_attention(q, window, pooled, wi, ci, sink, scale):
    """Read packed rows directly; scratch scales with split count, not selected D."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] % 32:
        raise ValueError("Prepare one request with 32-aligned attention heads")
    _, length, heads, dim = q.shape
    if not length or not heads or not dim or sink.shape != (heads,):
        raise ValueError("Invalid query or attention sink shape")
    if any(x.ndim != 3 or x.shape[:2] != (1, length) for x in (wi, ci)):
        raise ValueError("Sparse indices must match the query sequence")
    if any(x.ndim != 3 or x.shape[0] != 1 for x in (window, pooled)):
        raise ValueError("Prepare one request's KV rows")
    if window.dtype != mx.uint8 or pooled.dtype != mx.uint8:
        raise ValueError("Expected packed uint8 KV")
    if window.shape[-1] != dim + dim // 32 or pooled.shape[-1] != dim // 2 + dim // 16:
        raise ValueError("Invalid packed KV row width")
    if q.dtype == mx.bfloat16 and not (DS41_SPARSE and length <= 8 and dim == 512):
        return rounded_packed_attention(q, window, pooled, wi, ci, sink, scale)

    # Short verification blocks share the decode reduction and split geometry.
    # Each query owns its causal indices while all rows share one dispatch.
    chunk = 32 if length <= 8 else 128
    splits = max(1, (wi.shape[-1] + ci.shape[-1] + chunk - 1) // chunk)
    mma = length > 8 and dim <= 512
    head_group = 8 if mma else 4
    kind = "attention_mma" if mma else "attention"
    # The fast decoder reads packed rows as vectors, so placeholders must stay
    # device buffers (MLX passes inputs under 8 elements in constant memory).
    placeholder = 1
    if kind == "attention" and decode_fusions.DS41_DECODE_KERNELS_V2 and dim == 512:
        kind, placeholder = "attention_fastdec", 16
    partial = _kernel(kind)(
        inputs=[
            q,
            window if window.size else mx.zeros((placeholder,), mx.uint8),
            pooled if pooled.size else mx.zeros((placeholder,), mx.uint8),
            wi.astype(mx.int32) if wi.size else mx.zeros((1,), mx.int32),
            ci.astype(mx.int32) if ci.size else mx.zeros((1,), mx.int32),
            mx.array(
                [
                    heads,
                    wi.shape[-1],
                    ci.shape[-1],
                    window.shape[1],
                    pooled.shape[1],
                    splits,
                ],
                mx.int32,
            ),
            mx.array([scale]),
        ],
        template=[("D", dim), ("CHUNK", chunk)],
        grid=((heads + head_group - 1) // head_group * 128, length, splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, heads, splits, dim + 2)],
        output_dtypes=[mx.float32],
    )[0]
    return _kernel("merge")(
        inputs=[partial, sink, mx.array([heads, splits], mx.int32)],
        template=[("D", dim), ("T", q.dtype)],
        grid=((heads + 3) // 4 * 128, length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def packed_index_scores(
    q, keys, weights, start, ratio, candidates=None, *, key_start=0
):
    """Fuse dot/ReLU/weighted-head reduction without [query, head, context] scratch."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] % 32 or keys.dtype != mx.uint8:
        raise ValueError("Expected one request and packed FP4 index keys")
    _, length, heads, dim = q.shape
    if not length or not heads or not dim or ratio <= 0 or start < 0 or key_start < 0:
        raise ValueError("Invalid index query geometry or cache position")
    if keys.ndim != 3 or keys.shape[0] != 1 or weights.shape != (1, length, heads):
        raise ValueError("Index keys or weights do not match the query")
    if candidates is not None and (
        candidates.ndim != 3 or candidates.shape[:2] != (1, length)
    ):
        raise ValueError("Candidate rows must match the query sequence")
    width = keys.shape[1] if candidates is None else candidates.shape[-1]
    if keys.shape[-1] != dim // 2 + dim // 32:
        raise ValueError("Invalid packed index row width")
    if not width:
        return mx.zeros((1, length, 0), mx.float32)
    if DS41_INDEX_NAX and length == 1 and heads == 32 and dim == 128 and width >= 8192:
        from .index_nax import packed_scores

        return packed_scores(
            q, keys, weights, start, ratio, candidates, key_start=key_start
        )
    if (
        DS41_INDEX_ROWS
        and candidates is None
        # CED midpoint batches use the same per-row reductions as verify.
        # Keep their key tile in registers across the bounded 128-query tail.
        and 2 <= length <= (128 if DS41_PREFILL_MID_INDEX else 8)
        and heads == 32
        and dim == 128
        and q.dtype == mx.bfloat16
    ):
        return _kernel("index_mma_rows")(
            inputs=[
                q,
                keys,
                weights.astype(mx.float32),
                mx.array(
                    [heads, keys.shape[1], width, start, ratio, key_start, length],
                    mx.int32,
                ),
            ],
            template=[("D", dim), ("HEADS", heads), ("RATIO", ratio)],
            grid=((width + 31) // 32 * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, length, width)],
            output_dtypes=[mx.float32],
        )[0]
    mma = dim <= 128 and heads >= 8
    head_split = mma and heads >= 32
    cols = 16 if head_split else 32
    return _kernel("index_mma" if mma else "index")(
        inputs=[
            q,
            keys if keys.size else mx.zeros((1,), mx.uint8),
            weights.astype(mx.float32),
            (
                mx.zeros((1,), mx.int32)
                if candidates is None
                else candidates.astype(mx.int32)
            ),
            mx.array([heads, keys.shape[1], width, start, ratio, key_start], mx.int32),
        ],
        template=[
            ("D", dim),
            ("CANDIDATES", candidates is not None),
            ("HEAD_SPLIT", head_split),
            ("HEADS", heads),
            ("RATIO", ratio),
        ],
        grid=(
            ((width + cols - 1) // cols if mma else (width + 3) // 4) * 128,
            length,
            1,
        ),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, width)],
        output_dtypes=[mx.float32],
    )[0]


_TOPK = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint query = threadgroup_position_in_grid.y;
    const uint tile = threadgroup_position_in_grid.x;
    const int width = meta[0], base = meta[1];
    threadgroup float exchange_values[TILE];
    threadgroup int exchange_ids[TILE];
    float values[TILE / THREADS], next_values[TILE / THREADS];
    int indices[TILE / THREADS], next_ids[TILE / THREADS];
    for (uint j = 0; j < TILE / THREADS; ++j) {
        const uint i = tid + j * THREADS, pos = tile * TILE + i;
        float value = -INFINITY;
        if (pos < width) {
            for (uint k = 0; k < BLOCK; ++k) {
                const uint src = pos * BLOCK + k;
                if (src < uint(meta[2])) value = max(value, scores[query * meta[2] + src]);
            }
            const int length = (meta[3] + int(query) + 1) / RATIO;
            if (FORCE && length > 0 && int(pos) + base == (length - 1) / BLOCK)
                value = INFINITY;
        }
        values[j] = value;
        indices[j] = pos < width ? (EXPLICIT ? ids[query * width + pos] : base + int(pos)) : INT_MAX;
    }
    // Only cross-SIMD exchanges need threadgroup storage and barriers.
    for (uint size = 2; size <= TILE; size <<= 1) {
        for (uint stride = size >> 1; stride; stride >>= 1) {
            const bool cross_simd = stride >= 32 && stride < THREADS;
            if (cross_simd) {
                for (uint j = 0; j < TILE / THREADS; ++j) {
                    exchange_values[tid + j * THREADS] = values[j];
                    exchange_ids[tid + j * THREADS] = indices[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint j = 0; j < TILE / THREADS; ++j) {
                const uint i = tid + j * THREADS;
                float other;
                int other_id;
                if (stride < 32) {
                    other = simd_shuffle_xor(values[j], stride);
                    other_id = simd_shuffle_xor(indices[j], stride);
                } else if (cross_simd) {
                    other = exchange_values[i ^ stride];
                    other_id = exchange_ids[i ^ stride];
                } else {
                    other = values[j ^ (stride / THREADS)];
                    other_id = indices[j ^ (stride / THREADS)];
                }
                const bool better = other > values[j] || (other == values[j] && other_id < indices[j]);
                const bool worse = other < values[j] || (other == values[j] && other_id > indices[j]);
                const bool descending = ((i & size) == 0) == ((i & stride) == 0);
                const bool take = descending ? better : worse;
                next_values[j] = take ? other : values[j];
                next_ids[j] = take ? other_id : indices[j];
            }
            if (cross_simd) threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < TILE / THREADS; ++j) {
                values[j] = next_values[j]; indices[j] = next_ids[j];
            }
        }
    }
    for (uint j = 0; j < TILE / THREADS; ++j) {
        const uint i = tid + j * THREADS;
        if (i < K) {
            const size_t out = (size_t(query) * NT + tile) * K + i;
            selected_scores[out] = values[j];
            selected_ids[out] = indices[j];
        }
    }

"""


@cache
def _topk_kernel(runtime_k=False):
    return mx.fast.metal_kernel(
        name="deepseek_v41_tile_topk" + ("_rt" if runtime_k else ""),
        input_names=["scores", "ids", "meta"],
        output_names=["selected_scores", "selected_ids"],
        source=_TOPK_RT if runtime_k else _TOPK,
        header=_HEADER,
    )


_TOPK_MERGE = r"""
    const uint pos = thread_position_in_grid.x;
    const uint query = threadgroup_position_in_grid.y;
    const uint group = threadgroup_position_in_grid.z;
    const uint na = meta[0], nb = meta[1], runs = meta[2];
    const uint groups = GROUPED ? (runs + FANIN - 1) / FANIN : 1;
    if (GROUPED) {
        const uint source = pos / K, own = pos % K;
        const uint first = group * FANIN, n = min(uint(FANIN), runs - first);
        if (source >= n) return;
        const size_t base = (size_t(query) * runs + first) * K;
        const float value = a_scores[base + source * K + own];
        const int id = a_ids[base + source * K + own];
        uint rank = own;
        for (uint run = 0; run < n; ++run) {
            if (run == source) continue;
            uint lo = 0, hi = K - rank;
            while (lo < hi) {
                const uint mid = (lo + hi) / 2;
                const float other = a_scores[base + run * K + mid];
                const int other_id = a_ids[base + run * K + mid];
                const bool before = other > value || (other == value &&
                    (other_id < id || (run < source && other_id == id)));
                if (before) lo = mid + 1; else hi = mid;
            }
            rank += lo;
            // Once K entries precede this item, later runs cannot rescue it.
            if (rank >= K) return;
        }
        const size_t out = (size_t(query) * groups + group) * K + rank;
        selected_scores[out] = value;
        selected_ids[out] = id;
        return;
    }
    const uint left_n = na, right_n = nb;
    if (pos >= left_n + right_n) return;
    const size_t abase = size_t(query) * na, bbase = size_t(query) * nb;
    const bool right = pos >= left_n;
    const uint own = right ? pos - left_n : pos;
    const float value = right ? b_scores[bbase + own] : a_scores[abase + own];
    const int id = right ? b_ids[bbase + own] : a_ids[abase + own];
    if (own >= K) return;
    uint lo = 0, hi = min(right ? left_n : right_n, uint(K) - own);
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        const float other = right ? a_scores[abase + mid] : b_scores[bbase + mid];
        const int other_id = right ? a_ids[abase + mid] : b_ids[bbase + mid];
        // Identical entries in the left run precede those in the right run.
        const bool before = other > value || (other == value &&
            (other_id < id || (right && other_id == id)));
        if (before) lo = mid + 1; else hi = mid;
    }
    const uint rank = own + lo;
    if (rank < K) {
        const size_t out = (size_t(query) * groups + group) * K + rank;
        selected_scores[out] = value;
        selected_ids[out] = id;
    }
"""

_TOPK_RT = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint query = threadgroup_position_in_grid.y;
    const uint tile = threadgroup_position_in_grid.x;
    const int width = meta[0], base = meta[1];
    // Selection width and tile count arrive at run time so decode does not
    // compile a new pipeline whenever the causal key count grows.
    const uint K = uint(meta[5]), NT = uint(meta[6]);
    threadgroup float exchange_values[TILE];
    threadgroup int exchange_ids[TILE];
    float values[TILE / THREADS], next_values[TILE / THREADS];
    int indices[TILE / THREADS], next_ids[TILE / THREADS];
    for (uint j = 0; j < TILE / THREADS; ++j) {
        const uint i = tid + j * THREADS, pos = tile * TILE + i;
        float value = -INFINITY;
        if (pos < width) {
            for (uint k = 0; k < BLOCK; ++k) {
                const uint src = pos * BLOCK + k;
                if (src < uint(meta[2])) value = max(value, scores[query * meta[2] + src]);
            }
            const int length = (meta[3] + int(query) + 1) / RATIO;
            if (FORCE && length > 0 && int(pos) + base == (length - 1) / BLOCK)
                value = INFINITY;
        }
        values[j] = value;
        indices[j] = pos < width ? (EXPLICIT ? ids[query * width + pos] : base + int(pos)) : INT_MAX;
    }
    // Only cross-SIMD exchanges need threadgroup storage and barriers.
    for (uint size = 2; size <= TILE; size <<= 1) {
        for (uint stride = size >> 1; stride; stride >>= 1) {
            const bool cross_simd = stride >= 32 && stride < THREADS;
            if (cross_simd) {
                for (uint j = 0; j < TILE / THREADS; ++j) {
                    exchange_values[tid + j * THREADS] = values[j];
                    exchange_ids[tid + j * THREADS] = indices[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint j = 0; j < TILE / THREADS; ++j) {
                const uint i = tid + j * THREADS;
                float other;
                int other_id;
                if (stride < 32) {
                    other = simd_shuffle_xor(values[j], stride);
                    other_id = simd_shuffle_xor(indices[j], stride);
                } else if (cross_simd) {
                    other = exchange_values[i ^ stride];
                    other_id = exchange_ids[i ^ stride];
                } else {
                    other = values[j ^ (stride / THREADS)];
                    other_id = indices[j ^ (stride / THREADS)];
                }
                const bool better = other > values[j] || (other == values[j] && other_id < indices[j]);
                const bool worse = other < values[j] || (other == values[j] && other_id > indices[j]);
                const bool descending = ((i & size) == 0) == ((i & stride) == 0);
                const bool take = descending ? better : worse;
                next_values[j] = take ? other : values[j];
                next_ids[j] = take ? other_id : indices[j];
            }
            if (cross_simd) threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < TILE / THREADS; ++j) {
                values[j] = next_values[j]; indices[j] = next_ids[j];
            }
        }
    }
    for (uint j = 0; j < TILE / THREADS; ++j) {
        const uint i = tid + j * THREADS;
        if (i < K) {
            const size_t out = (size_t(query) * NT + tile) * K + i;
            selected_scores[out] = values[j];
            selected_ids[out] = indices[j];
        }
    }

"""

_TOPK_MERGE_RT = r"""
    const uint pos = thread_position_in_grid.x;
    const uint query = threadgroup_position_in_grid.y;
    const uint group = threadgroup_position_in_grid.z;
    const uint na = meta[0], nb = meta[1], runs = meta[2];
    const uint K = uint(meta[3]);
    const uint groups = GROUPED ? (runs + FANIN - 1) / FANIN : 1;
    if (GROUPED) {
        const uint source = pos / K, own = pos % K;
        const uint first = group * FANIN, n = min(uint(FANIN), runs - first);
        if (source >= n) return;
        const size_t base = (size_t(query) * runs + first) * K;
        const float value = a_scores[base + source * K + own];
        const int id = a_ids[base + source * K + own];
        uint rank = own;
        for (uint run = 0; run < n; ++run) {
            if (run == source) continue;
            uint lo = 0, hi = K - rank;
            while (lo < hi) {
                const uint mid = (lo + hi) / 2;
                const float other = a_scores[base + run * K + mid];
                const int other_id = a_ids[base + run * K + mid];
                const bool before = other > value || (other == value &&
                    (other_id < id || (run < source && other_id == id)));
                if (before) lo = mid + 1; else hi = mid;
            }
            rank += lo;
            // Once K entries precede this item, later runs cannot rescue it.
            if (rank >= K) return;
        }
        const size_t out = (size_t(query) * groups + group) * K + rank;
        selected_scores[out] = value;
        selected_ids[out] = id;
        return;
    }
    const uint left_n = na, right_n = nb;
    if (pos >= left_n + right_n) return;
    const size_t abase = size_t(query) * na, bbase = size_t(query) * nb;
    const bool right = pos >= left_n;
    const uint own = right ? pos - left_n : pos;
    const float value = right ? b_scores[bbase + own] : a_scores[abase + own];
    const int id = right ? b_ids[bbase + own] : a_ids[abase + own];
    if (own >= K) return;
    uint lo = 0, hi = min(right ? left_n : right_n, uint(K) - own);
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        const float other = right ? a_scores[abase + mid] : b_scores[bbase + mid];
        const int other_id = right ? a_ids[abase + mid] : b_ids[bbase + mid];
        // Identical entries in the left run precede those in the right run.
        const bool before = other > value || (other == value &&
            (other_id < id || (right && other_id == id)));
        if (before) lo = mid + 1; else hi = mid;
    }
    const uint rank = own + lo;
    if (rank < K) {
        const size_t out = (size_t(query) * groups + group) * K + rank;
        selected_scores[out] = value;
        selected_ids[out] = id;
    }
"""


@cache
def _topk_merge_kernel(runtime_k=False):
    return mx.fast.metal_kernel(
        name="deepseek_v41_sorted_topk_merge" + ("_rt" if runtime_k else ""),
        input_names=["a_scores", "a_ids", "b_scores", "b_ids", "meta"],
        output_names=["selected_scores", "selected_ids"],
        source=_TOPK_MERGE_RT if runtime_k else _TOPK_MERGE,
        header=_HEADER,
    )


def _merge_topk(a, b, count, *, runs=0):
    """Merge sorted winners by parallel rank search, without concatenation."""
    if not count:
        return a
    na, nb = a[0].shape[-1], b[0].shape[-1]
    if count > 2048:
        return _tile_topk(
            mx.concatenate([a[0], b[0]], -1),
            count,
            ids=mx.concatenate([a[1], b[1]], -1),
        )
    fanin = 4 if count <= 512 else 2
    groups = (runs + fanin - 1) // fanin if runs else 1
    width = fanin * count if runs else na + nb
    return tuple(
        _topk_merge_kernel(runtime_k=True)(
            inputs=[*a, *b, mx.array([na, nb, runs, count], mx.int32)],
            template=[("GROUPED", bool(runs)), ("FANIN", fanin)],
            grid=((width + 255) // 256 * 256, a[0].shape[1], groups),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, a[0].shape[1], groups * count)] * 2,
            output_dtypes=[mx.float32, mx.int32],
        )
    )


def _tile_topk(
    scores,
    count,
    *,
    offset=0,
    ids=None,
    block_size=1,
    force_latest=False,
    start=0,
    ratio=1,
):
    """Sort each initial tile once, then merge sorted winners by rank."""
    score_width = scores.shape[-1]
    width = (score_width + block_size - 1) // block_size
    count = min(count, width)
    if count <= 0:
        return scores[..., :0], mx.zeros((*scores.shape[:-1], 0), mx.int32)
    if count > 2048:
        if block_size != 1 or force_latest:
            padded = mx.pad(
                scores,
                [(0, 0), (0, 0), (0, -score_width % block_size)],
                constant_values=-float("inf"),
            )
            scores = padded.reshape(1, scores.shape[1], -1, block_size).max(-1)
            lengths = (mx.arange(start + 1, start + scores.shape[1] + 1) // ratio)[
                None, :, None
            ]
            if force_latest:
                scores = mx.where(
                    (lengths > 0)
                    & (mx.arange(width) + offset == (lengths - 1) // block_size),
                    float("inf"),
                    scores,
                )
        if ids is None:
            ids = mx.broadcast_to(
                mx.arange(width, dtype=mx.int32) + offset, scores.shape
            )
        order = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
        return mx.take_along_axis(scores, order, -1), mx.take_along_axis(ids, order, -1)
    tile_limit = 4096 if scores.shape[1] == 1 else max(1024, 2 * count)
    tile = max(32, 1 << (min(width, tile_limit) - 1).bit_length())
    tiles = (width + tile - 1) // tile
    threads = min(256, tile)
    # Prompt tails and causal key counts change on real traffic. Only the
    # bounded sorting geometry is specialized; K and NT are runtime metadata.
    winners = tuple(
        _topk_kernel(runtime_k=True)(
            inputs=[
                scores,
                ids if ids is not None else mx.zeros((1,), mx.int32),
                mx.array([width, offset, score_width, start, ratio, count, tiles], mx.int32),
            ],
            template=[
                ("TILE", tile),
                ("EXPLICIT", ids is not None),
                ("BLOCK", block_size),
                ("FORCE", force_latest),
                ("THREADS", threads),
                ("RATIO", ratio),
            ],
            grid=(tiles * threads, scores.shape[1], 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(1, scores.shape[1], tiles * count)] * 2,
            output_dtypes=[mx.float32, mx.int32],
        )
    )
    while tiles > 1:
        winners = _merge_topk(winners, winners, count, runs=tiles)
        fanin = 4 if count <= 512 else 2
        tiles = (tiles + fanin - 1) // fanin
    return winners


def packed_index_topk(
    q,
    keys,
    weights,
    start,
    ratio,
    count,
    *,
    block_count=0,
    block_size=0,
    chunk_size=None,
    query_chunk_size=None,
):
    """Stream exact source top-k and optional block maxima with bounded scratch.

    All causally visible source keys are scanned. A single bounded tile remains
    lazy so decode can submit it with the rest of the layer. Larger scans use
    GPU batches that overlap CPU construction of the next batch.
    The prior batch is drained before
    submitting another, and the final batch is drained before returning.
    Returns chronological IDs, with -1 padding.
    """
    if (
        DS41_PREFILL_MID_INDEX
        and block_count
        and query_chunk_size is None
        and chunk_size is None
        and 8 < q.shape[1] <= 128
    ):
        # CED midpoint prefill: the 128 tail queries against every key. Score
        # all of them per batch over 256K keys (128 MiB of scores) instead of
        # 16 queries over 64K keys, so a 1M-token chunk drains 2 GPU batches
        # instead of 64. The tile top-k and rank merges are batch-invariant.
        query_chunk_size = q.shape[1]
        chunk_size = 262144
    if query_chunk_size is None:
        # Fill the score budget with queries when the key prefix is short.
        # A fixed 16-query tile creates many unnecessary submission barriers.
        query_chunk_size = min(512, max(16, 1048576 // max(1, keys.shape[1])))
    if count < 0 or block_count < 0 or query_chunk_size <= 0:
        raise ValueError("Invalid index selection budget")
    if chunk_size is None:
        chunk_size = min(
            524288 if q.shape[1] == 1 else 262144,
            1048576 // max(1, min(q.shape[1], query_chunk_size)),
        )
        if DS41_DECODE_SINGLE_TILE and q.shape[1] <= 8:
            # Decode and DSpark verification: one lazy tile over every key
            # (scores stay <= 8 x keys FP32). Chunked scans evaluated each
            # batch mid-forward, idling the GPU while Python rebuilt the graph.
            # Tile boundaries stay multiples of 64 and the rank order is
            # total, so the selection is unchanged.
            chunk_size = max(chunk_size, (keys.shape[1] + 63) // 64 * 64)
    if chunk_size <= 0:
        raise ValueError("Invalid index selection budget")
    if block_count and block_size <= 0:
        raise ValueError("Candidate block size must be positive")
    # Validate even an empty source through the scoring entry point.
    packed_index_scores(q, keys[:, :0], weights, start, ratio)
    width = keys.shape[1]
    count = min(count, width)
    if block_count:
        block_count = min(block_count, (width + block_size - 1) // block_size)
        chunk_size = max(block_size, chunk_size // block_size * block_size)
    if (DS41_PREFILL_INDEX and q.shape[0] == 1 and q.shape[1] > 8 and q.shape[2:] == (32, 128)
            and q.dtype == mx.bfloat16 and keys.shape[1] >= count and count > 0 and not block_count):
        from .prefill_index import select
        result = select(q, keys, weights, start, ratio, count, score_fn=packed_index_scores)
        return result, mx.zeros((*result.shape[:2], 0), mx.int32)
    results, block_results = [], []
    batch_chunks = 1 if q.shape[1] == 1 else 2
    pending = ()
    single_tile = (
        q.shape[1] <= query_chunk_size
        and min(width, (start + q.shape[1]) // ratio) <= chunk_size
    )
    for begin in range(0, q.shape[1], query_chunk_size):
        end = min(begin + query_chunk_size, q.shape[1])
        running = (
            mx.full((1, end - begin, count), -float("inf")),
            mx.full((1, end - begin, count), 2147483647, mx.int32),
        )
        blocks = (
            mx.full((1, end - begin, block_count), -float("inf")),
            mx.full((1, end - begin, block_count), 2147483647, mx.int32),
        )
        visible = min(width, (start + end) // ratio)
        selected = None
        for first in range(0, visible, chunk_size):
            stop = min(first + chunk_size, visible)
            scores = packed_index_scores(
                q[:, begin:end],
                keys[:, first:stop],
                weights[:, begin:end],
                start + begin,
                ratio,
                key_start=first,
            )
            if (
                DS41_DECODE_RADIX
                and single_tile
                and q.shape[1] <= 8
                and scores.shape[-1] >= max(count, 32768)
            ):
                # Decode/verify rows: exact multi-threadgroup radix selection
                # of the same ids (score desc, id asc) as the tile sort.
                selected = decode_topk.selection(scores, count)
            else:
                values, ids = _tile_topk(scores, count, offset=first)
                running = (
                    (values, ids)
                    if first == 0 and values.shape[-1] == count
                    else _merge_topk(running, (values, ids), count)
                )
            if block_count:
                values, ids = _tile_topk(
                    scores,
                    block_count,
                    offset=first // block_size,
                    block_size=block_size,
                    force_latest=True,
                    start=start + begin,
                    ratio=ratio,
                )
                blocks = (
                    (values, ids)
                    if first == 0 and values.shape[-1] == block_count
                    else _merge_topk(blocks, (values, ids), block_count)
                )
            if not single_tile and (
                stop == visible or (first // chunk_size + 1) % batch_chunks == 0
            ):
                # Prepare this bounded graph while the previous batch runs;
                # drain it before submission to keep only one GPU batch active.
                if pending:
                    mx.eval(*pending)
                pending = (*running, *blocks)
                mx.async_eval(*pending)
        results.append(
            selected
            if selected is not None
            else mx.sort(mx.where(running[0] > -float("inf"), running[1], -1), axis=-1)
        )
        block_results.append(
            mx.sort(mx.where(blocks[0] > -float("inf"), blocks[1], -1), axis=-1)
        )
    if pending:
        mx.eval(*pending)
    return mx.concatenate(results, 1), mx.concatenate(block_results, 1)
