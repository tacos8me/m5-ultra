# SPDX-License-Identifier: MIT
"""MXFP4 routed experts over sorted block lists (replay-sized blocks), bitwise the block kernels.

deepseek_mxfp4_gather_qmm_{pair_concat_,}blocks (steel BlockMMA, fp32 simdgroup MMA)
computes every output element as one sequential fp32 chain over k = 0..K-1 of
float(x_bf16) * float(w_bf16) added to an accumulator that starts at +0, then
rounds to bf16 (products are exact: bf16 x bf16 fits fp32). Checked on the M5
Ultra against six other summation orders on adversarial inputs: only the
sequential chain matches, bit for bit, and it matches everywhere. So any
parallelization that keeps each element's chain in k order reproduces those
kernels exactly. These kernels give one thread one output column and all rows
of a block (<= R rows of one expert) as R accumulators: no padded 8x8 tiles and
no per-BK threadgroup barriers for the weights, which the steel kernels pay for
blocks of a few rows (the replay's 128-row segments: ~5 rows per expert).
"""
import os
from functools import cache

import mlx.core as mx

ENABLED = os.environ.get("DS41_MOE_ROWS", "1") == "1"

_HEADER = r"""
constant float DS41R_LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float ds41r_scale(uint8_t b) {
    // fp8_e8m0 -> bfloat16 -> float, as dequantize_scale<bfloat16_t, 32> + float() in the block loader.
    uint16_t o = (b == 0 ? 0x40 : (uint16_t(b) << 7));
    return float(as_type<bfloat16_t>(o));
}
"""

# One threadgroup = TN output columns of one block (row_start, expert, rows <= R).
# x rows of the block are staged in threadgroup memory KC values at a time.
_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const int block = int(threadgroup_position_in_grid.y);
    if (block >= count[0]) return;
    const int row_start = meta[block * 3 + 0];
    const int expert = meta[block * 3 + 1];
    const int rows = meta[block * 3 + 2];
    if (rows <= 0) return;
    const int col = int(threadgroup_position_in_grid.x) * TN + int(tid);
    const bool valid = col < NOUT;
    const bool second = PAIR && col >= N;
    const int n = second ? col - N : col;
    constexpr int KW = K / 8;
    constexpr int KG = K / 32;
    const device uint32_t* wr = (second ? w1 : w0) + (size_t(expert) * N + (valid ? n : 0)) * KW;
    const device uint8_t* sr = (second ? s1 : s0) + (size_t(expert) * N + (valid ? n : 0)) * KG;
    threadgroup float xs[R * KC];
    float acc[R];
    for (int r = 0; r < R; r++) acc[r] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += KC) {
        for (int i = int(tid); i < R * KC; i += TN) {
            const int r = i / KC, kk = i % KC;
            xs[i] = r < rows ? float(x[size_t(row_start + r) * K + k0 + kk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            for (int g = 0; g < KC / 32; g++) {
                const float s = ds41r_scale(sr[(k0 >> 5) + g]);
                const device uint4* wp = (const device uint4*)(wr + ((k0 + 32 * g) >> 3));
                const uint4 q = *wp;
                const uint words[4] = {q.x, q.y, q.z, q.w};
                for (int wi = 0; wi < 4; wi++) {
                    const uint word = words[wi];
                    for (int j = 0; j < 8; j++) {
                        const float wv = float(static_cast<bfloat16_t>(s * DS41R_LUT[(word >> (4 * j)) & 15]));
                        const int kk = 32 * g + 8 * wi + j;
                        for (int r = 0; r < R; r++) {
                            if (r < rows) acc[r] = fma(xs[r * KC + kk], wv, acc[r]);
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        for (int r = 0; r < R; r++) {
            if (r < rows) y[size_t(row_start + r) * NOUT + col] = static_cast<bfloat16_t>(acc[r]);
        }
    }
"""


@cache
def _kernel(pair):
    names = ["x", "w0", "s0", "w1", "s1", "meta", "count"] if pair else ["x", "w0", "s0", "meta", "count"]
    src = _SRC if pair else _SRC.replace("(second ? w1 : w0)", "w0").replace("(second ? s1 : s0)", "s0")
    return mx.fast.metal_kernel(name="ds41_mxfp4_rows_" + ("pair" if pair else "one"), input_names=names,
                                output_names=["y"], source=src, header=_HEADER)


def run(x, weights, meta, count, *, rows_per_block, tn=128, kc=128):
    """x: [M, 1, K] bf16 sorted rows; weights: [(w, s)] (1 or 2 projections, same N); -> [M, 1, len(weights)*N]."""
    pair = len(weights) == 2
    w0, s0 = weights[0]
    n, k = w0.shape[1], w0.shape[2] * 8
    nout = n * len(weights)
    inputs = [x, w0, s0] + ([weights[1][0], weights[1][1]] if pair else []) + [meta, count]
    return _kernel(pair)(
        inputs=inputs,
        template=[("K", k), ("N", n), ("NOUT", nout), ("PAIR", int(pair)), ("R", rows_per_block),
                  ("TN", tn), ("KC", kc)],
        grid=(-(-nout // tn) * tn, meta.shape[0], 1),
        threadgroup=(tn, 1, 1),
        output_shapes=[(x.shape[0], 1, nout)],
        output_dtypes=[mx.bfloat16],
        init_value=0,
    )[0]
