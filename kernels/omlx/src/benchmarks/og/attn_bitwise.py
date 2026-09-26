"""Bitwise + speed gate for the ds41 decode attention/projection kernels (synthetic data, ~1 GB, seconds).

Compares each new path against the previous one on the same inputs: every output bit must match.
"""
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(os.environ.get('DS41_TREE', str(Path.home()/'src/wt/ds41-attn')))))
os.environ.setdefault('DS41_SPARSE', '1')
os.environ.setdefault('MLX_ENABLE_TF32', '0')

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.set_memory_limit(8 << 30)
from omlx.patches.deepseek_v41 import kernels  # noqa: E402
from omlx.patches.deepseek_v41.quantization import pack_activation  # noqa: E402

FAIL = []


def same(name, a, b):
    a, b = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a), np.array(b.view(mx.uint16) if b.dtype == mx.bfloat16 else b)
    bad = int((a != b).sum())
    if bad:
        FAIL.append(name)
    print(f'{name}: {"OK" if not bad else f"MISMATCH {bad}/{a.size}"}', flush=True)


_SPIN = mx.fast.metal_kernel(
    name='attn_bitwise_spin', input_names=['n'], output_names=['y'],
    source="""
    float acc = 0.0f;
    for (int i = 0; i < n[0]; ++i) acc = metal::fma(acc, 0.999f, 1.0f);
    y[0] = acc;
    """)


def _spin():
    return _SPIN(inputs=[mx.array([2000000], mx.int32)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                 output_shapes=[(1,)], output_dtypes=[mx.float32])[0]


def bench(step, x, calls=20, reps=20):
    """GPU us per call of `calls` dependent step(x) calls queued behind a spin kernel (spin time subtracted)."""
    def run(n):
        ts = []
        for r in range(reps + 3):
            g = _spin()
            y = x + (g * 0).astype(x.dtype)
            for _ in range(n):
                y = step(y)
            t = time.perf_counter()
            mx.eval(y)
            ts.append(time.perf_counter() - t)
        return sorted(ts[3:])[reps // 2]
    return (run(calls) - run(0)) / calls * 1e6


def attention_case(length, n_pooled, window_rows, seed, empty_ci=False, invalid_frac=0.1, start=None):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    q = (mx.random.normal((1, length, 64, 512)) * 2).astype(mx.bfloat16)
    window = pack_activation((mx.random.normal((1, window_rows, 512)) * 3).astype(mx.bfloat16))
    pooled = pack_activation((mx.random.normal((1, max(n_pooled, 1), 512)) * 3).astype(mx.bfloat16), bits=4,
                             group_size=16, e4m3_scale=True)[:, :n_pooled]
    w = min(128, window_rows)
    wi = np.stack([np.where(rng.random(w) < invalid_frac, -1, rng.integers(0, window_rows, w)) for _ in range(length)])
    if empty_ci:
        ci = np.zeros((length, 0), np.int32)
    else:
        c = min(512, n_pooled)
        ci = np.stack([np.sort(np.where(rng.random(c) < invalid_frac, -1, rng.choice(n_pooled, c, replace=False)))
                       for _ in range(length)])
    sink = mx.array(rng.normal(size=64).astype(np.float32))
    return q, window, pooled, mx.array(wi[None].astype(np.int32)), mx.array(ci[None].astype(np.int32)), sink


def test_attention():
    cases = [(L, n, 128 + L, s) for L in (1, 2, 3, 4, 5, 8) for n, s in ((8192, 1), (600, 2))]
    cases += [(1, 40, 40, 3), (3, 7, 12, 4), (5, 0, 130, 5), (2, 513, 128, 6), (4, 131072, 132, 7)]
    for L, n, wr, seed in cases:
        args = attention_case(L, n, wr, seed, empty_ci=(n == 0))
        kernels.DS41_ATTN_SHARED = False
        kernels.DS41_MERGE_WIDE = False
        ref = kernels.packed_sparse_attention(*args[:5], args[5], 512 ** -0.5)
        kernels.DS41_ATTN_SHARED = True
        kernels.DS41_MERGE_WIDE = True
        new = kernels.packed_sparse_attention(*args[:5], args[5], 512 ** -0.5)
        same(f'attn L={L} pooled={n} window={wr}', ref, new)
    for L in (1, 5):
        args = attention_case(L, 8192, 128 + L, 11)
        out = {}
        for flag in (False, True):
            kernels.DS41_ATTN_SHARED = flag
            kernels.DS41_MERGE_WIDE = flag
            out[flag] = bench(lambda x: kernels.packed_sparse_attention(x, *args[1:5], args[5], 512 ** -0.5), args[0])
        print(f'attn L={L} us: old {out[False]:.1f} shared {out[True]:.1f}', flush=True)




def test_speed():
    from omlx.patches.deepseek_v41 import attn_fusions, language, decode_fusions
    from omlx.patches.deepseek_v41.quantization import quantize_activation
    c = _config()
    for L in (1, 5):
        x = _act((1, L, 1280), 1)
        w = mx.ones((1280,), mx.bfloat16)
        old = bench(lambda y: quantize_activation(language.norm(y, w, 1e-20)), x)
        new = bench(lambda y: attn_fusions.rms_quant(y, w, 1e-20, True)[1], x)
        print(f'L={L} q_norm+fp8 us: old {old:.1f} fused {new:.1f}', flush=True)
        x = _act((1, L, 512), 2)
        w5 = mx.ones((512,), mx.bfloat16)
        win = pack_activation(_act((1, 128, 512), 3))
        tabs = language.rope_tables(8192, L, c, True)

        def old_kv(y):
            new_rows = decode_fusions.pack_fp8(language.rope_range(language.norm(y, w5, 1e-20), 8192, L, c, True))
            kv = mx.concatenate([win, new_rows], 1)
            return y + kv[:, -L:, :1].astype(y.dtype) * 0

        def new_kv(y):
            kv = attn_fusions.kv_rows(y, w5, 1e-20, *tabs, win, 128)
            return y + kv[:, -L:, :1].astype(y.dtype) * 0
        print(f'L={L} kv path us (+2 glue ops): old {bench(old_kv, x):.1f} fused {bench(new_kv, x):.1f}', flush=True)
        print(f'L={L} glue ops alone us: {bench(lambda y: y + y[:, :, :1] * 0, x):.1f}', flush=True)


def test_split():
    """Time the split scan and the merge separately (diagnostic)."""
    for L in (1, 5):
        q, window, pooled, wi, ci, sink = attention_case(L, 8192, 128 + L, 11)
        splits = (wi.shape[-1] + ci.shape[-1] + 31) // 32
        meta = mx.array([64, wi.shape[-1], ci.shape[-1], window.shape[1], pooled.shape[1], splits], mx.int32)
        for kind, hg in (('attention_fastdec', 4), ('attention_shared', 8)):
            th = hg * 32 if kind == 'attention_shared' else 128

            def scan(x, kind=kind, hg=hg, th=th):
                return kernels._kernel(kind)(
                    inputs=[x, window, pooled, wi, ci, meta, mx.array([512 ** -0.5])],
                    template=[('D', 512), ('CHUNK', 32)] + ([('HG', hg)] if kind == 'attention_shared' else []),
                    grid=(64 // hg * th, L, splits), threadgroup=(th, 1, 1),
                    output_shapes=[(1, L, 64, splits, 514)], output_dtypes=[mx.float32])[0]
            print(f'L={L} {kind} scan us {bench(lambda x: scan(x)[:, :, :, 0, :512].astype(mx.bfloat16), q):.1f}', flush=True)
        part = scan(q)
        mx.eval(part)

        def merge(p):
            out = kernels._kernel('merge')(inputs=[p, sink, mx.array([64, splits], mx.int32)],
                                           template=[('D', 512), ('T', mx.bfloat16)], grid=(16 * 128, L, 1),
                                           threadgroup=(128, 1, 1), output_shapes=[q.shape], output_dtypes=[mx.bfloat16])[0]
            return p + out[:, :, :, None, :1].astype(mx.float32) * 0
        print(f'L={L} merge(+dep op) us {bench(merge, part):.1f}', flush=True)
        print(f'L={L} dep op only us {bench(lambda p: p + p[:, :, :, :1, :1] * 0, part):.1f}', flush=True)


def _config():
    import json
    from omlx.patches.deepseek_v41.config import ModelConfig
    raw = json.loads((Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx/config.json').read_text())
    return ModelConfig.from_dict(raw)


def _act(shape, seed, scale=1.0):
    mx.random.seed(seed)
    x = mx.random.normal(shape) * scale
    # heavy tails and exact zeros, like real activations
    x = mx.where(mx.random.uniform(shape=shape) < 0.01, x * 40, x)
    x = mx.where(mx.random.uniform(shape=shape) < 0.005, 0.0, x)
    return x.astype(mx.bfloat16)


def test_rms():
    from omlx.patches.deepseek_v41 import attn_fusions, language
    from omlx.patches.deepseek_v41.quantization import quantize_activation
    for d in (1280, 512):
        for L in (1, 2, 3, 4, 5, 8):
            for seed in range(3):
                x = _act((1, L, d), seed + 10 * L, scale=(0.02, 1.0, 30.0)[seed])
                mx.random.seed(99 + seed)
                w = (mx.random.uniform(0.2, 2.0, (d,))).astype(mx.bfloat16)
                ref = language.norm(x, w, 1e-20)
                refq = quantize_activation(ref)
                y, yq = attn_fusions.rms_quant(x, w, 1e-20, True)
                same(f'rms d={d} L={L} s={seed}', ref, y)
                same(f'rms+fp8 d={d} L={L} s={seed}', refq, yq)


def test_kv():
    from omlx.patches.deepseek_v41 import attn_fusions, language, decode_fusions
    c = _config()
    for L in (1, 2, 3, 4, 5, 8):
        for start, old_len in ((8192, 128), (40, 40), (5, 5), (0, 0), (131072 + 7, 128)):
            x = _act((1, L, 512), L + start, scale=(3.0 if L % 2 else 0.05))
            mx.random.seed(7)
            w = mx.random.uniform(0.2, 2.0, (512,)).astype(mx.bfloat16)
            old = pack_activation(_act((1, 128 + 3, 512), 3))[:, 3:]
            rotated = language.rope_range(language.norm(x, w, 1e-20), start, L, c, True)
            new = decode_fusions.pack_fp8(rotated)
            ref = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
            got = attn_fusions.kv_rows(x, w, 1e-20, *language.rope_tables(start, L, c, True), old, old_len)
            same(f'kv L={L} start={start} old={old_len}', ref, got)


def test_merge_rope():
    from omlx.patches.deepseek_v41 import language
    c = _config()
    for L in (1, 2, 3, 4, 5, 8):
        for start in (8192, 3, 1000003):
            args = attention_case(L, 8192, 128 + L, start % 97 + L)
            kernels.DS41_MERGE_WIDE = False
            ref = language.rope_range(kernels.packed_sparse_attention(*args[:5], args[5], 512 ** -0.5), start, L, c, True,
                                      inverse=True)
            kernels.DS41_MERGE_WIDE = True
            got = kernels.packed_sparse_attention(*args[:5], args[5], 512 ** -0.5,
                                                  rope_tables=language.rope_tables(start, L, c, True, inverse=True))
            same(f'attn+rope L={L} start={start}', ref, got)



def test_index_q():
    from omlx.patches.deepseek_v41 import attn_fusions, language
    from omlx.patches.deepseek_v41.quantization import quantize_activation
    c = _config()
    for L in (1, 2, 3, 4, 5, 8):
        for start, scale in ((8192, 1.0), (17, 0.001), (131072 + 5, 50.0)):
            q = _act((1, L, 32, 128), L + start, scale=scale)
            ref = quantize_activation(language.rope_range(q, start, L, c, True), bits=4)
            got = attn_fusions.index_q(q, *language.rope_tables(start, L, c, True))
            same(f'index_q L={L} start={start}', ref, got)


def test_cand_topk():
    from omlx.patches.deepseek_v41 import kernels as K
    rng = np.random.default_rng(0)
    for L in (1, 2, 3, 5, 8):
        for n_blocks, invalid, zero_frac in ((1024, 0, 0.3), (2048, 100, 0.5), (2048, 0, 0.0), (70, 3, 0.9)):
            blocks = np.sort(rng.choice(20000, n_blocks, replace=False))
            blocks[:invalid] = -1
            blocks = np.sort(blocks)
            blocks = mx.array(np.broadcast_to(blocks, (1, L, n_blocks)).astype(np.int32))
            cand = mx.where(blocks[..., None] >= 0, blocks[..., None] * 8 + mx.arange(8), -1).reshape(1, L, -1)
            sc = rng.standard_normal((1, L, n_blocks * 8)).astype(np.float32)
            sc[rng.random(sc.shape) < zero_frac] = 0.0
            sc = np.round(sc, 1)  # many exact ties
            sc[np.broadcast_to(np.array(cand) < 0, sc.shape)] = -np.inf
            sc[:, :, -37:] = -np.inf
            scores = mx.array(sc)
            count = min(512, scores.shape[-1])
            order = mx.argsort(-scores, axis=-1)[..., :count].astype(mx.int32)
            valid = mx.take_along_axis(scores, order, axis=-1) > -float('inf')
            ref = mx.sort(mx.where(valid, mx.take_along_axis(cand, order, -1), -1), axis=-1)
            values, ids = K._tile_topk(scores, count, ids=cand.astype(mx.int32))
            got = mx.sort(mx.where(values > -float('inf'), ids, -1), axis=-1)
            same(f'cand_topk L={L} blocks={n_blocks} invalid={invalid} zeros={zero_frac}', ref, got)


def test_select():
    from omlx.patches.deepseek_v41 import attn_fusions
    from omlx.patches.deepseek_v41 import kernels as K
    rng = np.random.default_rng(1)

    def scores_for(L, W, zero_frac, ninf_tail, neg_zero=True):
        sc = np.round(rng.standard_normal((1, L, W)).astype(np.float32), 2)
        sc[rng.random(sc.shape) < zero_frac] = 0.0
        if neg_zero:
            sc[rng.random(sc.shape) < 0.05] = -0.0
        if ninf_tail:
            sc[:, :, -ninf_tail:] = -np.inf
        return mx.array(sc)
    # candidate lists
    for L in (1, 2, 5, 8):
        for n_blocks, invalid, zf in ((1024, 0, 0.3), (2048, 100, 0.6), (2048, 0, 0.0), (70, 3, 0.9), (40, 0, 0.0)):
            blocks = np.sort(rng.choice(40000, n_blocks, replace=False)).astype(np.int32)
            blocks[:invalid] = -1
            blocks = mx.array(np.sort(blocks)[None, None].repeat(L, 1))
            cand = mx.where(blocks[..., None] >= 0, blocks[..., None] * 8 + mx.arange(8), -1).reshape(1, L, -1)
            sc = scores_for(L, n_blocks * 8, zf, 29)
            sc = mx.where(cand < 0, -float('inf'), sc)
            count = min(512, sc.shape[-1])
            order = mx.argsort(-sc, axis=-1)[..., :count].astype(mx.int32)
            valid = mx.take_along_axis(sc, order, axis=-1) > -float('inf')
            ref = mx.sort(mx.where(valid, mx.take_along_axis(cand, order, -1), -1), axis=-1)
            got = attn_fusions.select_rows(sc, count, ids=cand.astype(mx.int32))
            same(f'select cand L={L} blocks={n_blocks} invalid={invalid}', ref, got)
    # key positions and block maxima (layer-20 decode top-k)
    for L in (1, 3, 5, 8):
        for W, zf in ((8197, 0.2), (1000, 0.0), (32768, 0.5), (513, 0.9), (20000, 0.0)):
            start = W - L
            sc = scores_for(L, W, zf, 0)
            # causal: row r sees (start + r + 1) keys
            vis = np.arange(W)[None, None, :] >= (start + np.arange(L) + 1)[None, :, None]
            sc = mx.where(mx.array(vis), -float('inf'), sc)
            count = min(512, W)
            v, i = K._tile_topk(sc, count)
            ref = mx.sort(mx.where(v > -float('inf'), i, -1), axis=-1)
            same(f'select keys L={L} W={W}', ref, attn_fusions.select_rows(sc, count))
            nb = (W + 7) // 8
            bc = min(2048, nb)
            v, i = K._tile_topk(sc, bc, block_size=8, force_latest=True, start=start, ratio=1)
            ref = mx.sort(mx.where(v > -float('inf'), i, -1), axis=-1)
            got = attn_fusions.select_rows(sc, bc, block_size=8, force_latest=True, start=start, ratio=1)
            same(f'select blocks L={L} W={W}', ref, got)


if __name__ == '__main__':
    which = sys.argv[1:] or ['attention', 'merge_rope', 'rms', 'kv', 'index_q', 'cand_topk', 'select']
    for name in which:
        globals()['test_' + name]()
    print('FAIL' if FAIL else 'ALL BITWISE OK', FAIL)
    sys.exit(1 if FAIL else 0)
