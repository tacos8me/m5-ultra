"""Router GEMM replicas: bitwise vs MLX (xf @ w.astype(f32).T) and timing. Gate weights only (~12 MB)."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('MLX_ENABLE_TF32', '0')
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path.home()/'src/wt/ds41-ffn'))
from omlx.patches.deepseek_v41.loading import _load_shard

MODEL = Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx'

GEMV = r"""
    // MLX gemv (bm4 bn1 sm1 sn32 tm4 tn4) per output row: lane l owns columns 128 i + 4 l + [0, 4).
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * SGS + simdgroup_index_in_threadgroup;
    const device T* wr = w + size_t(row) * K + 4 * lane;
    const device float* xr = x + 4 * lane;
    float r = 0.0f;
    constexpr int ITERS = K / 128;
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
                ACC
    }
    for (ushort sn = 16; sn >= 1; sn >>= 1) r += simd_shuffle_down(r, sn);
    if (lane == 0) out[row] = r;
"""

SPLITK = r"""
    // MLX steel_gemm_splitk (bm16 bn32 bk16, 32 partitions of K/32) + splitk_accum, per 8-column tile:
    // simdgroup p = partition p's 8x8 MMA chain from zero; then the partitions are summed in order.
    const uint lane = thread_index_in_simdgroup;
    const uint p = simdgroup_index_in_threadgroup;
    const uint n0 = threadgroup_position_in_grid.x * 8;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    constexpr int PART = K / 32;
    threadgroup float partial[32 * M * 8];
    simdgroup_float8x8 C = simdgroup_float8x8(0);
    const device T* w0 = w + size_t(n0 + fn) * K + p * PART + fm;
    const device T* w1 = w0 + K;
    const device float* xa = x + size_t(fm) * K + p * PART + fn;
    for (int c = 0; c < PART / 8; ++c) {
        simdgroup_float8x8 A, B;
        A.thread_elements()[0] = fm < M ? xa[8 * c] : 0.0f;
        A.thread_elements()[1] = fm < M ? xa[8 * c + 1] : 0.0f;
        B.thread_elements()[0] = float(w0[8 * c]);
        B.thread_elements()[1] = float(w1[8 * c]);
        simdgroup_multiply_accumulate(C, A, B, C);
    }
    if (fm < M) {
        partial[(p * M + fm) * 8 + fn] = C.thread_elements()[0];
        partial[(p * M + fm) * 8 + fn + 1] = C.thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = p * 32 + lane;
    if (t < M * 8) {
        const uint m = t / 8, j = t % 8;
        float acc = 0;
        for (int q = 0; q < 32; q++) acc += partial[(q * M + m) * 8 + j];
        out[m * N + n0 + j] = acc;
    }
"""


def kernel(name, src):
    return mx.fast.metal_kernel(name=name, input_names=['w', 'x'], output_names=['out'], source=src)


_k = {}


def gemv(w, x, variant, sgs=4, u=8):
    acc = {'fma': 'r = fma(wv[u][t], xv[u][t], r);',
           'nofma': '{\n#pragma clang fp contract(off)\n r = r + wv[u][t] * xv[u][t]; }',
           'plain': 'r += wv[u][t] * xv[u][t];'}[variant]
    key = ('gemv', variant)
    if key not in _k:
        _k[key] = kernel(f'router_gemv_{variant}', GEMV.replace('ACC', acc))
    n, k = w.shape
    return _k[key](inputs=[w, x], template=[('T', w.dtype), ('K', k), ('SGS', sgs), ('U', u)],
                   grid=(32 * n // sgs, sgs, 1), threadgroup=(32, sgs, 1),
                   output_shapes=[(n,)], output_dtypes=[mx.float32])[0]


def splitk(w, x):
    if 'splitk' not in _k:
        _k['splitk'] = kernel('router_splitk', SPLITK)
    n, k = w.shape
    m = x.shape[0]
    return _k['splitk'](inputs=[w, x], template=[('T', w.dtype), ('K', k), ('N', n), ('M', m)],
                        grid=(32 * n // 8, 32, 1), threadgroup=(32, 32, 1),
                        output_shapes=[(m, n)], output_dtypes=[mx.float32])[0]


def timeit(fn, reps=30):
    for _ in range(3):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def main():
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    ws = []
    for i in range(20, 40):
        key = f'language_model.layers.{i}.ffn.gate.weight'
        ws.append(_load_shard(MODEL/wm[key])[key])
    mx.eval(ws)
    w32 = [w.astype(mx.float32) for w in ws]
    mx.eval(w32)
    snap = mx.load(str(Path.home()/'llm/ds41/og-speed/snap-8k/step.safetensors'))
    h = snap['h'][0].astype(mx.float32)  # (5, 4, 5120)
    real = [h[:, j] * mx.rsqrt((h[:, j] ** 2).mean(-1, keepdims=True) + 1e-20) for j in range(4)]
    mx.random.seed(3)
    xs = [r.astype(mx.bfloat16).astype(mx.float32) for r in real]
    xs += [(mx.random.normal((8, 5120)) * s).astype(mx.bfloat16).astype(mx.float32) for s in (0.05, 1.0, 20.0)]
    xs = [x if x.shape[0] >= 8 else mx.concatenate([x, x[:8 - x.shape[0]] * 0.5], 0) for x in xs]
    mx.eval(xs)
    res = {}
    for variant in ('fma', 'nofma', 'plain'):
        bad = n = 0
        for w, w3 in zip(ws, w32):
            for x in xs:
                for r in range(8):
                    a = x[r:r + 1] @ w3.T
                    b = gemv(w, x[r], variant)
                    mx.eval(a, b)
                    n += 1
                    bad += not bool(mx.array_equal(a[0], b).item())
        res[f'gemv_{variant}'] = f'{n - bad}/{n}'
    for m in range(2, 9):
        bad = n = 0
        for w, w3 in zip(ws, w32):
            for x in xs:
                a = x[:m] @ w3.T
                b = splitk(w, x[:m])
                mx.eval(a, b)
                n += 1
                bad += not bool(mx.array_equal(a, b).item())
        res[f'splitk_m{m}'] = f'{n - bad}/{n}'
    print(json.dumps(res), flush=True)
    # timing: 20 dependent-free calls rotating the 20 layers, per call
    x1 = xs[0][:1]
    for m in (1, 2, 5):
        x = xs[0][:m]
        ref = lambda: [x @ w3.T for w3 in w32]
        if m == 1:
            new = lambda: [gemv(w, x[0], 'fma') for w in ws]
        else:
            new = lambda: [splitk(w, x) for w in ws]
        print(json.dumps(dict(rows=m, ref_ms_per20=round(timeit(ref), 3), new_ms_per20=round(timeit(new), 3))), flush=True)
    # serialized chains (each call depends on the previous): latency per call
    for m in (1, 5):
        x = xs[0][:m]

        def chain(new):
            def run():
                y = x
                for w, w3 in zip(ws, w32):
                    o = (gemv(w, y[0], 'fma')[None] if m == 1 else splitk(w, y)) if new else y @ w3.T
                    y = y + o[:, :1] * 0
                return y
            return run
        print(json.dumps(dict(chain_rows=m, ref_ms_per20=round(timeit(chain(False)), 3),
                              new_ms_per20=round(timeit(chain(True)), 3))), flush=True)


main()
