# SPDX-License-Identifier: MIT
"""Prototype: MXFP4 sorted-block experts with simdgroup MMA fed straight from device memory (no weight staging)."""
from functools import cache
import mlx.core as mx

_HEADER = r"""
inline float ds41m_fp4(uint8_t bits) {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return static_cast<float>(bits & 8 ? -converted : converted);
}
inline float ds41m_scale(uint8_t b) {
    uint16_t o = (b == 0 ? 0x40 : (uint16_t(b) << 7));
    return float(as_type<bfloat16_t>(o));
}
inline float ds41m_w(uint8_t nib, float s) {
    return float(static_cast<bfloat16_t>(s * ds41m_fp4(nib)));
}
"""

# TG = NSG simdgroups; simdgroup g owns NT tiles of 8 output features; the block's (<= 8) token rows are
# the MMA's column dimension: C[n, t] += W[n, k..k+8) . x[t, k..k+8) in k order (the steel kernels' chain).
_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int block = int(threadgroup_position_in_grid.y);
    if (block >= count[0]) return;
    const int row_start = meta[block * 3 + 0];
    const int expert = meta[block * 3 + 1];
    const int rows = meta[block * 3 + 2];
    if (rows <= 0) return;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    constexpr int KB = K / 2;
    constexpr int KG = K / 32;
    const int col0 = int(threadgroup_position_in_grid.x) * (NSG * NT * 8) + int(sg) * (NT * 8);
    threadgroup bfloat16_t xs[8 * KC];
    simdgroup_matrix<float, 8, 8> acc[NT];
    for (int j = 0; j < NT; j++) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const device uint8_t* wrow[NT];
    const device uint8_t* srow[NT];
    bool tile_ok[NT];
    for (int j = 0; j < NT; j++) {
        const int col = col0 + j * 8 + fm;
        tile_ok[j] = col0 + j * 8 < NOUT;
        const bool second = PAIR && col >= N;
        const int n = tile_ok[j] ? (second ? col - N : col) : 0;
        wrow[j] = (const device uint8_t*)(second ? w1 : w0) + (size_t(expert) * N + n) * KB;
        srow[j] = (second ? s1 : s0) + (size_t(expert) * N + n) * KG;
    }
    for (int k0 = 0; k0 < K; k0 += KC) {
        for (int i = int(tid); i < 8 * KC; i += NSG * 32) {
            const int t = i / KC, kk = i % KC;
            xs[i] = t < rows ? x[size_t(row_start + t) * K + k0 + kk] : bfloat16_t(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < KC; g += 32) {
            float sc[NT];
            for (int j = 0; j < NT; j++) sc[j] = ds41m_scale(srow[j][(k0 + g) >> 5]);
            for (int s = 0; s < 32; s += 8) {
                const int kk = g + s;
                simdgroup_matrix<float, 8, 8> b;
                b.thread_elements()[0] = float(xs[fn * KC + kk + fm]);
                b.thread_elements()[1] = float(xs[(fn + 1) * KC + kk + fm]);
                for (int j = 0; j < NT; j++) {
                    const uint8_t byte = wrow[j][(k0 + kk + fn) >> 1];
                    simdgroup_matrix<float, 8, 8> a;
                    a.thread_elements()[0] = ds41m_w(byte & 15, sc[j]);
                    a.thread_elements()[1] = ds41m_w(byte >> 4, sc[j]);
                    simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = 0; j < NT; j++) {
        if (!tile_ok[j]) continue;
        const int col = col0 + j * 8 + fm;
        if (fn < rows) y[size_t(row_start + fn) * NOUT + col] = static_cast<bfloat16_t>(acc[j].thread_elements()[0]);
        if (fn + 1 < rows) y[size_t(row_start + fn + 1) * NOUT + col] = static_cast<bfloat16_t>(acc[j].thread_elements()[1]);
    }
"""


@cache
def _kernel(pair):
    names = ["x", "w0", "s0", "w1", "s1", "meta", "count"] if pair else ["x", "w0", "s0", "meta", "count"]
    src = _SRC if pair else _SRC.replace("(second ? w1 : w0)", "w0").replace("(second ? s1 : s0)", "s0")
    return mx.fast.metal_kernel(name="ds41_mxfp4_mma_" + ("pair" if pair else "one"), input_names=names,
                                output_names=["y"], source=src, header=_HEADER)


def run(x, weights, meta, count, *, nsg=4, nt=4, kc=256):
    pair = len(weights) == 2
    w0, s0 = weights[0]
    n, k = w0.shape[1], w0.shape[2] * 8
    nout = n * len(weights)
    per_tg = nsg * nt * 8
    inputs = [x, w0, s0] + ([weights[1][0], weights[1][1]] if pair else []) + [meta, count]
    return _kernel(pair)(
        inputs=inputs,
        template=[("K", k), ("N", n), ("NOUT", nout), ("PAIR", int(pair)), ("NSG", nsg), ("NT", nt), ("KC", kc)],
        grid=(-(-nout // per_tg) * nsg * 32, meta.shape[0], 1),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(x.shape[0], 1, nout)],
        output_dtypes=[mx.bfloat16],
        init_value=0,
    )[0]
