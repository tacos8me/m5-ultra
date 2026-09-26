# SPDX-License-Identifier: MIT
"""Exact decode/verify top-k over long score rows with a multi-threadgroup radix select.

Decode scores at most 8 rows, so the per-row tile sort plus rank merges ran on a few
threadgroups per row. Here every threadgroup histograms a chunk of every row (top 12
bits of the order-preserving key), one pass per row picks the threshold bin, a second
chunked pass writes every id above it and gathers the bin's members, and one
threadgroup per row sorts that bin by (key desc, id asc). The selected set is the
K largest scores with ties (including -0 == +0) broken by ascending id, which is
exactly what the tile sort selects. A threshold bin larger than CAP falls back to
an exact single-threadgroup 8-bit radix over the row.
"""
import mlx.core as mx
from functools import cache as _cache

_HDR = r"""
#include <metal_stdlib>
using namespace metal;
inline uint score_key(float x) { uint u = as_type<uint>(x == 0.0f ? 0.0f : x); return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u); }
"""

# Pass 1: per-chunk 4096-bin histogram of the top 12 key bits, added into a per-row global histogram.
_HIST = r"""
    const uint tid = thread_index_in_threadgroup, chunk = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
    const uint n = meta[0], per = meta[1];
    threadgroup atomic_uint h[4096];
    for (uint b = tid; b < 4096; b += 256) atomic_store_explicit(h + b, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint begin = chunk * per, end = min(n, begin + per);
    const device float* s = scores + size_t(row) * n;
    for (uint i = begin + tid; i < end; i += 256)
        atomic_fetch_add_explicit(h + (score_key(s[i]) >> 20), 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint* g = (device atomic_uint*)hist + row * 4096;
    for (uint b = tid; b < 4096; b += 256) {
        const uint c = atomic_load_explicit(h + b, memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(g + b, c, memory_order_relaxed);
    }
"""

# Pass 2 (one threadgroup per row): bins above the threshold bin hold `above` < K ids; `need` come from it.
_THRESH = r"""
    const uint tid = thread_index_in_threadgroup, row = threadgroup_position_in_grid.y;
    const uint K = meta[2];
    threadgroup uint part[256];
    uint local = 0;
    for (uint j = 0; j < 16; ++j) local += hist[row * 4096 + 4095 - (tid * 16 + j)];
    part[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0, t = 0;
        for (; t < 255; ++t) { if (acc + part[t] >= K) break; acc += part[t]; }
        for (uint j = 0; j < 16; ++j) {
            const uint bin = 4095 - (t * 16 + j), c = hist[row * 4096 + bin];
            if (acc + c >= K || j == 15) { plan[row * 3] = bin; plan[row * 3 + 1] = acc; plan[row * 3 + 2] = K - acc; break; }
            acc += c;
        }
    }
"""

# Pass 3: write every id above the threshold bin; gather the threshold bin's members as (key, ~id).
_COLLECT = r"""
    const uint tid = thread_index_in_threadgroup, chunk = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
    const uint n = meta[0], per = meta[1], K = meta[2], CAP = meta[3];
    const uint bin = plan[row * 3];
    const uint begin = chunk * per, end = min(n, begin + per);
    const device float* s = scores + size_t(row) * n;
    device atomic_uint* c = (device atomic_uint*)counts + row * 2;
    for (uint i = begin + tid; i < end; i += 256) {
        const uint key = score_key(s[i]), hb = key >> 20;
        if (hb > bin) ids[size_t(row) * K + atomic_fetch_add_explicit(c, 1, memory_order_relaxed)] = int(i);
        else if (hb == bin) {
            const uint k = atomic_fetch_add_explicit(c + 1, 1, memory_order_relaxed);
            if (k < CAP) cand[size_t(row) * CAP + k] = (ulong(key) << 32) | ulong(0xFFFFFFFFu - i);
        }
    }
"""

# Pass 4 (one threadgroup per row): copy the ids above the threshold, sort the threshold bin by
# (key desc, id asc) and take `need`. A bin above CAP falls back to an exact single-threadgroup
# 8-bit radix over the row (the prefill compact kernel's fallback).
_FINISH = r"""
    const uint tid = thread_index_in_threadgroup, row = threadgroup_position_in_grid.y;
    const uint n = meta[0], K = meta[2], CAP = meta[3];
    const uint above = plan[row * 3 + 1], need = plan[row * 3 + 2];
    const uint count = counts[row * 2 + 1];
    device int* out = ids + size_t(row) * K;
    const device float* s = scores + size_t(row) * n;
    threadgroup ulong buf[2048];
    if (count <= CAP) {
        for (uint r = tid; r < above; r += 256) out[r] = above_ids[size_t(row) * K + r];
        uint P = 1; while (P < count) P <<= 1;
        for (uint i = tid; i < P; i += 256) buf[i] = i < count ? cand[size_t(row) * CAP + i] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 2; k <= P; k <<= 1) for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < P; i += 256) {
                const uint l = i ^ j;
                if (l > i) {
                    const ulong a = buf[i], b = buf[l];
                    const bool desc = (i & k) == 0;
                    if (desc ? (a < b) : (a > b)) { buf[i] = b; buf[l] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint r = tid; r < need; r += 256) out[above + r] = int(0xFFFFFFFFu - uint(buf[r] & 0xFFFFFFFFul));
        return;
    }
    threadgroup uint prefix, remaining, greater[256], equal[256];
    threadgroup atomic_uint* h8 = (threadgroup atomic_uint*)buf;
    if (tid == 0) { prefix = 0; remaining = K; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int shift = 24; shift >= 0; shift -= 8) {
        atomic_store_explicit(h8 + tid, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n; i += 256) {
            const uint code = score_key(s[i]);
            if (shift == 24 || (code >> (shift + 8)) == prefix)
                atomic_fetch_add_explicit(h8 + ((code >> shift) & 255u), 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) for (int b = 255; b >= 0; --b) {
            const uint c = atomic_load_explicit(h8 + b, memory_order_relaxed);
            if (remaining > c) remaining -= c; else { prefix = (prefix << 8) | uint(b); break; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint first = (n * tid) / 256, last = (n * (tid + 1)) / 256;
    uint ng = 0, ne = 0;
    for (uint i = first; i < last; ++i) { const uint code = score_key(s[i]); ng += code > prefix; ne += code == prefix; }
    greater[tid] = ng; equal[tid] = ne;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint g = 0, e = 0;
        for (uint t = 0; t < 256; ++t) { const uint a = greater[t], b = equal[t]; greater[t] = g; equal[t] = e; g += a; e += b; }
        remaining = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g = greater[tid], e = equal[tid];
    for (uint i = first; i < last; ++i) {
        const uint code = score_key(s[i]);
        if (code > prefix) out[g++] = int(i);
        else if (code == prefix) { if (remaining + e < K) out[remaining + e] = int(i); ++e; }
    }
"""

_SPEC = {
    'hist': (_HIST, ['scores', 'meta'], ['hist']),
    'thresh': (_THRESH, ['hist', 'meta'], ['plan']),
    'collect': (_COLLECT, ['scores', 'plan', 'meta'], ['ids', 'counts', 'cand']),
    'finish': (_FINISH, ['scores', 'plan', 'counts', 'cand', 'above_ids', 'meta'], ['ids']),
}


@_cache
def _k(name):
    src, ins, outs = _SPEC[name]
    return mx.fast.metal_kernel(name='ds41_decode_radix_' + name, input_names=ins, output_names=outs,
                                source=src, header=_HDR)


def radix_ids(scores, k, cap=2048, per=8192):
    """Unordered ids of the top-k per row (the set _tile_topk selects)."""
    _, L, n = scores.shape
    chunks = (n + per - 1) // per
    meta = mx.array([n, per, k, cap], mx.uint32)
    hist = _k('hist')(inputs=[scores, meta], grid=(256 * chunks, L, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(L * 4096,)], output_dtypes=[mx.uint32], init_value=0)[0]
    plan = _k('thresh')(inputs=[hist, meta], grid=(256, L, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(L * 3,)], output_dtypes=[mx.uint32])[0]
    above, counts, cand = _k('collect')(inputs=[scores, plan, meta], grid=(256 * chunks, L, 1),
                                        threadgroup=(256, 1, 1),
                                        output_shapes=[(L * k,), (L * 2,), (L * cap,)],
                                        output_dtypes=[mx.int32, mx.uint32, mx.uint64], init_value=0)
    return _k('finish')(inputs=[scores, plan, counts, cand, above, meta], grid=(256, L, 1),
                        threadgroup=(256, 1, 1), output_shapes=[(1, L, k)], output_dtypes=[mx.int32])[0]


def selection(scores, k):
    """The decode path's final ids: ascending, -1 where the selected score is -inf."""
    k = min(k, scores.shape[-1])
    ids = radix_ids(scores, k)
    vals = mx.take_along_axis(scores, ids, -1)
    return mx.sort(mx.where(vals > -float('inf'), ids, -1), axis=-1)
