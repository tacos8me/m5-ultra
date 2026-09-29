# SPDX-License-Identifier: MIT
"""Prototype: MXFP4 sorted experts, small blocks (<= S rows) by a scalar chain kernel, the rest by steel v0.
Each output = sequential fp32 chain over k of float(x_bf16) * float(w_bf16) from +0, rounded to bf16 (the
steel BlockMMA arithmetic, see moe_rows)."""
from functools import cache
import mlx.core as mx

_HEADER = r"""
inline float ds41s_fp4(uint b) {
    half v = as_type<half>(ushort((b & 7) << 9));
    v *= 16384.0;
    return static_cast<float>(b & 8 ? -v : v);
}
inline float ds41s_scale(uint8_t b) {
    uint16_t o = (b == 0 ? 0x40 : (uint16_t(b) << 7));
    return float(as_type<bfloat16_t>(o));
}
"""

# split: meta [B, 3] (bm plan over sorted routes), count -> meta_small/count_small (blocks of experts with
# <= S rows in total, i.e. one block with rows <= S) and meta_big/count_big (every other block).
_SPLIT = r"""
    uint b = thread_position_in_grid.x;
    if (b >= uint(MAXB)) return;
    int n = count[0];
    if (int(b) >= n) return;
    int rs = meta[b * 3], e = meta[b * 3 + 1], r = meta[b * 3 + 2];
    bool alone = (b == 0 || meta[(b - 1) * 3 + 1] != e) && (int(b) + 1 >= n || meta[(b + 1) * 3 + 1] != e);
    if (alone && r <= S) {
        uint k = atomic_fetch_add_explicit((device atomic_uint*)&cs[0], 1u, memory_order_relaxed);
        ms[k * 3] = rs; ms[k * 3 + 1] = e; ms[k * 3 + 2] = r;
    } else {
        uint k = atomic_fetch_add_explicit((device atomic_uint*)&cb[0], 1u, memory_order_relaxed);
        mb[k * 3] = rs; mb[k * 3 + 1] = e; mb[k * 3 + 2] = r;
    }
"""

# one thread = one output column of one small block (rows <= S); chain over k in order.
_SMALL = r"""
    const uint col = thread_position_in_grid.x;
    const int block = int(thread_position_in_grid.y);
    if (block >= int(count[0]) || col >= uint(NOUT)) return;
    const int row_start = meta[block * 3], expert = meta[block * 3 + 1], rows = meta[block * 3 + 2];
    const bool second = PAIR && col >= uint(N);
    const int n = second ? int(col) - N : int(col);
    const device uint4* wr = (const device uint4*)((const device uint8_t*)(second ? w1 : w0) + (size_t(expert) * N + n) * (K / 2));
    const device uint8_t* sr = (second ? s1 : s0) + (size_t(expert) * N + n) * (K / 32);
    const device bfloat16_t* x0 = x + size_t(row_start) * K;
    float acc[S];
    for (int r = 0; r < S; r++) acc[r] = 0.0f;
    for (int g = 0; g < K / 32; g++) {
        const float s = ds41s_scale(sr[g]);
        const uint4 q = wr[g];
        const uint words[4] = {q.x, q.y, q.z, q.w};
        for (int wi = 0; wi < 4; wi++) {
            const uint word = words[wi];
            const int k0 = 32 * g + 8 * wi;
            for (int j = 0; j < 8; j++) {
                const float w = float(static_cast<bfloat16_t>(s * ds41s_fp4(word >> (4 * j))));
                for (int r = 0; r < S; r++) {
                    if (r < rows) acc[r] = fma(float(x0[size_t(r) * K + k0 + j]), w, acc[r]);
                }
            }
        }
    }
    for (int r = 0; r < S; r++) {
        if (r < rows) y[size_t(row_start + r) * NOUT + col] = static_cast<bfloat16_t>(acc[r]);
    }
"""


@cache
def _split_kernel():
    return mx.fast.metal_kernel(name="ds41_moe_small_split", input_names=["meta", "count"],
                                output_names=["ms", "cs", "mb", "cb"], source=_SPLIT, atomic_outputs=False)


@cache
def _small_kernel(pair):
    names = ["x", "w0", "s0", "w1", "s1", "meta", "count"] if pair else ["x", "w0", "s0", "meta", "count"]
    src = _SMALL if pair else _SMALL.replace("(second ? w1 : w0)", "w0").replace("(second ? s1 : s0)", "s0")
    return mx.fast.metal_kernel(name="ds41_moe_small_" + ("pair" if pair else "one"), input_names=names,
                                output_names=["y"], source=src, header=_HEADER)


def split(meta, count, small_rows):
    maxb = meta.shape[0]
    ms, cs, mb, cb = _split_kernel()(inputs=[meta, count], template=[("S", small_rows), ("MAXB", maxb)],
                                     grid=(max(maxb, 1), 1, 1), threadgroup=(min(1024, max(maxb, 32)), 1, 1),
                                     output_shapes=[(maxb, 3), (1,), (maxb, 3), (1,)],
                                     output_dtypes=[mx.int32, mx.uint32, mx.int32, mx.uint32], init_value=0)
    return ms, cs, mb, cb


def small(x, weights, meta, count, rows, *, tn=64, out=None):
    pair = len(weights) == 2
    w0, s0 = weights[0]
    n, k = w0.shape[1], w0.shape[2] * 8
    nout = n * len(weights)
    inputs = [x, w0, s0] + ([weights[1][0], weights[1][1]] if pair else []) + [meta, count]
    return _small_kernel(pair)(inputs=inputs, template=[("K", k), ("N", n), ("NOUT", nout), ("PAIR", int(pair)), ("S", rows)],
                               grid=(-(-nout // tn) * tn, meta.shape[0], 1), threadgroup=(tn, 1, 1),
                               output_shapes=[(x.shape[0], 1, nout)], output_dtypes=[mx.bfloat16], init_value=0)[0]


_MASK = r"""
    uint b = thread_position_in_grid.x;
    if (int(b) >= int(count[0])) return;
    int rs = meta[b * 3], r = meta[b * 3 + 2];
    for (int i = 0; i < r; i++) mask[rs + i] = 1;
"""


@cache
def _mask_kernel():
    return mx.fast.metal_kernel(name="ds41_moe_small_mask", input_names=["meta", "count"], output_names=["mask"], source=_MASK)


def mask(meta, count, routes):
    return _mask_kernel()(inputs=[meta, count], grid=(max(meta.shape[0], 1), 1, 1), threadgroup=(64, 1, 1),
                          output_shapes=[(routes,)], output_dtypes=[mx.uint8], init_value=0)[0]
