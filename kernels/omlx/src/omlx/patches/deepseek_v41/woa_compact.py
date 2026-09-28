# SPDX-License-Identifier: MIT
"""Lossless byte storage for the BF16 wo_a at verify widths 2-5 (DS41_WOA_COMPACT=0 = off).

The pipe1 checkpoint holds wo_a as original FP8 E4M3 values times their block scales,
expanded to BF16, so the low four mantissa bits are clear. One byte keeps the sign, the
three live mantissa bits and a 4-bit code for the 15 consecutive exponents BASE..BASE+14
holding the most values. Code 15 (other exponents, set low bits) escapes to the BF16
weight, which stays resident for the 1-row, fused and prefill paths anyway.

The kernel is decode_fusions' grouped GEMV with one output row per simdgroup: the same
per-row loads, dot products, unroll-8 blocks and shuffle-down reduction, so outputs are
bitwise identical to decode_fusions.grouped_gemv and mx.einsum("bsgd,grd->bsgr").
"""
import json
import os
import time
from functools import cache

import mlx.core as mx
import numpy as np

from . import decode_fusions

ENABLED = os.environ.get("DS41_WOA_COMPACT", "1") == "1"
# A weight escaping more often than this keeps the BF16 kernel.
MAX_ESCAPE_RATE = 0.01
_ATTR = "_ds41_woa_codes"


class Codes:
    """Byte codes of one wo_a weight (a plain object: nn.Module keeps it out of parameters())."""

    __slots__ = ("weight", "codes", "base", "escape_rate")

    def __init__(self, weight, codes, base, escape_rate):
        self.weight, self.codes, self.base, self.escape_rate = weight, codes, base, escape_rate


@cache
def _histogram():
    # Bin e < 256: values with exponent e and the low four mantissa bits clear; bin 256: the rest.
    return mx.fast.metal_kernel(
        name="ds41_woa_exponent_histogram", input_names=["bits"], output_names=["hist"],
        atomic_outputs=True,
        source=r"""
        threadgroup atomic_uint local[257];
        const uint t = thread_position_in_threadgroup.x;
        for (uint i = t; i < 257; i += 256) atomic_store_explicit(&local[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = thread_position_in_grid.x; i < SIZE; i += THREADS) {
            const uint b = bits[i];
            atomic_fetch_add_explicit(&local[(b & 15u) ? 256u : (b >> 7) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < 257; i += 256) {
            const uint v = atomic_load_explicit(&local[i], memory_order_relaxed);
            if (v) atomic_fetch_add_explicit(&hist[i], v, memory_order_relaxed);
        }
        """)


def encode(weight):
    """Codes for a 2-D BF16 weight, or None when too many values escape."""
    bits = weight.view(mx.uint16).reshape(-1)
    threads = 256 * 256
    hist = _histogram()(
        inputs=[bits], template=[("SIZE", bits.size), ("THREADS", threads)],
        grid=(threads, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(257,)], output_dtypes=[mx.uint32], init_value=0)[0]
    hist = np.array(hist).astype(np.int64)
    # Exponent codes stay within the normal range 1..254.
    window = np.convolve(hist[:256], np.ones(15, np.int64), mode="valid")
    base = 1 + int(np.argmax(window[1:241]))
    escape_rate = 1 - int(window[base]) / bits.size
    if escape_rate > MAX_ESCAPE_RATE:
        return None
    exponent = (bits >> 7) & 255
    code = exponent - base  # uint16: exponents below BASE wrap to large codes
    code = mx.where(((bits & 15) == 0) & (code < 15), code, 15)
    codes = ((code << 3) | ((bits >> 8) & 128) | ((bits >> 4) & 7)).astype(mx.uint8)
    codes = codes.reshape(weight.shape)
    mx.eval(codes)
    return Codes(weight, codes, base, escape_rate)


def _attentions(model):
    for i, layer in enumerate(getattr(model, "layers", [])):
        if hasattr(layer, "attn"):
            yield f"layers.{i}", layer.attn
    for i, stage in enumerate(getattr(model, "mtp", None) or []):
        yield f"mtp.{i}", stage.attn


def install(model):
    """Encode every BF16 wo_a of model.layers and model.mtp (aliased modules once)."""
    if not ENABLED or mx.default_device() != mx.gpu:
        return None
    begin, done, report = time.perf_counter(), {}, {}
    for name, attn in _attentions(model):
        linear = attn.wo_a
        weight = getattr(linear, "weight", None)
        if weight is None or weight.dtype != mx.bfloat16 or weight.ndim != 2 or weight.shape[1] % 4:
            report[name] = None
            continue
        if id(linear) not in done:
            done[id(linear)] = encode(weight)
            if done[id(linear)] is not None:
                linear.__dict__[_ATTR] = done[id(linear)]
        codes = done[id(linear)]
        report[name] = None if codes is None else round(codes.escape_rate, 7)
    unique = [c for c in done.values() if c is not None]
    summary = dict(event="woa_compact", encoded=len(unique), skipped=sum(v is None for v in report.values()),
                   gib=sum(c.codes.nbytes for c in unique) / 2**30,
                   max_escape_rate=max((c.escape_rate for c in unique), default=None),
                   seconds=round(time.perf_counter() - begin, 3))
    print(json.dumps(summary), flush=True)
    return dict(summary, layers=report)


def codes_of(linear):
    codes = linear.__dict__.get(_ATTR)
    return codes if codes is not None and codes.weight is linear.weight else None


@cache
def _gemv():
    return mx.fast.metal_kernel(
        name="ds41_woa_compact_gemv", input_names=["x", "weight", "codes"], output_names=["y"],
        source=r"""
        const uint3 tid = threadgroup_position_in_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        constexpr int unroll = 8;
        constexpr int n_v4 = K / 4;
        constexpr int n_main = n_v4 - n_v4 % (32 * unroll);
        const int g = tid.z;
        const int row = tid.y * 4 + simd_gid;
        if (row >= N) return;
        const size_t offset = (size_t(g) * N + row) * K;
        const device uchar4* c4 = (const device uchar4*)(codes + offset);
        const device T* w = weight + offset;
        const device vec<T, 4>* x4[M];
        for (int v = 0; v < M; v++) x4[v] = (const device vec<T, 4>*)(x + (size_t(v) * G + g) * K);
        // The BF16 bits, rebuilt: sign, exponent BASE + code, three mantissa bits, four zero bits.
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
                y[(size_t(v) * G + g) * N + row] = static_cast<T>(result[v]);
        """)


def grouped_gemv(linear, grouped, weight):
    """decode_fusions.grouped_gemv(grouped, weight), from linear's byte codes when it has them.

    Call only where decode_fusions.grouped_gemv_supported(grouped, weight) holds."""
    codes = codes_of(linear) if ENABLED else None
    if codes is None:
        return decode_fusions.grouped_gemv(grouped, weight)
    length, groups, k = grouped.shape[1:]
    n = weight.shape[1]
    return _gemv()(
        inputs=[grouped, codes.weight, codes.codes],
        template=[("T", grouped.dtype), ("K", k), ("N", n), ("M", length), ("G", groups), ("BASE", codes.base)],
        grid=(32, (n + 3) // 4 * 4, groups), threadgroup=(32, 4, 1),
        output_shapes=[(1, length, groups, n)], output_dtypes=[grouped.dtype])[0]
