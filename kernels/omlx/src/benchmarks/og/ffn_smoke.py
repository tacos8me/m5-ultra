"""Synthetic ffn_fuse kernel check (8 fake experts, real dims, <1 GB): up/down vs the unfused kernels."""
import os
import sys
import types
from pathlib import Path

ROOT = Path(os.environ.get('DS41_TREE', str(Path.home()/'src/wt/ds41-ffn')))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MLX_ENABLE_TF32', '0')
os.environ.setdefault('DS41_MHC', '1')
import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41 import ffn_fuse, fast_qmv, hc_fuse, moe_decode
from omlx.patches.deepseek_v41.activation import quantize_swiglu_activation
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, quantize_activation

rng = np.random.default_rng(0)
E, D, I = 8, 5120, 2304


def fp4(shape):
    return mx.array(rng.integers(0, 2**32, size=shape, dtype=np.uint32))


def e8m0(shape):
    return mx.array(rng.integers(118, 126, size=shape, dtype=np.uint8))


def fp8(shape):
    b = rng.integers(0, 256, size=shape[:-1] + (shape[-1] * 4,), dtype=np.uint8)
    b = np.where((b & 0x7f) >= 0x78, b & 0xb7, b).astype(np.uint8)  # no NaN codes, moderate range
    return mx.array(b.view(np.uint32))


def proj(w, s, mode, bits):
    return QuantizedProjection(w, s, bits, mode, group_size=32, quantize_input=True)


ex = types.SimpleNamespace(
    w1=proj(fp4((E, I, D // 8)), e8m0((E, I, D // 32)), 'mxfp4', 4),
    w3=proj(fp4((E, I, D // 8)), e8m0((E, I, D // 32)), 'mxfp4', 4),
    w2=proj(fp4((E, D, I // 8)), e8m0((E, D, I // 32)), 'mxfp4', 4), _limit=10.0)
sh = types.SimpleNamespace(
    w1=proj(fp8((I, D // 4)), e8m0((I, D // 32)), 'mxfp8', 8),
    w3=proj(fp8((I, D // 4)), e8m0((I, D // 32)), 'mxfp8', 8),
    w2=proj(fp8((D, I // 4)), e8m0((D, I // 32)), 'mxfp8', 8), _limit=10.0)
gate_ns = types.SimpleNamespace(weight=(mx.random.normal((384, D)) * 0.02).astype(mx.bfloat16))
moe = types.SimpleNamespace(experts=ex, shared_experts=sh, gate=gate_ns)
mx.eval(gate_ns.weight)
mx.eval([p.weight for p in (ex.w1, ex.w2, ex.w3, sh.w1, sh.w2, sh.w3)])


def ref_shared(xq):
    def p(w, x):
        m = x.shape[0]
        if m == 1:
            return fast_qmv.mxfp8_qmv(x[None], w.weight, w.scales)[0]
        if m == 2:
            return mx.quantized_matmul(x, w.weight, w.scales, None, group_size=32, bits=8, mode='mxfp8')
        return fast_qmv.mxfp8_qmv(x[None], w.weight, w.scales)[0] if m <= 5 else og_rows(w, x)
    y = quantize_swiglu_activation(p(sh.w1, xq), p(sh.w3, xq), None, xq.dtype, sh._limit)
    return p(sh.w2, y)


def og_rows(w, x):
    from omlx.patches.deepseek_v41 import og_fused
    return og_fused._rows(w, x[None])[0]


bad = 0
for rows in (1, 2, 3, 4, 5, 8, 10, 16):
    for scale in (1.0, 30.0, 0.01):
        x = (mx.random.normal((rows, D)) * scale).astype(mx.bfloat16)
        xq = quantize_activation(x)
        ids = mx.array(rng.integers(0, E, size=(rows * 6,), dtype=np.uint32))
        w = mx.array(rng.random((rows * 6,), dtype=np.float32))
        gate, up = moe_decode.gate_up(xq.reshape(-1, 1, D), ex.w1, ex.w3, ids, 6)
        y = quantize_swiglu_activation(gate, up, w, xq.dtype, ex._limit)
        r_ref = moe_decode.down(y, ex.w2, ids).reshape(rows * 6, D)
        s_ref = ref_shared(xq)
        xf = x.astype(mx.float32)
        ys, raw = ffn_fuse.shared_router(moe, xq, xf)
        yr = ffn_fuse.routed_up(moe, xq, ids, w)
        r_new, s_new = ffn_fuse.down(moe, yr, ys, ids)
        raw_ref = xf @ gate_ns.weight.astype(mx.float32).T if rows <= 8 else mx.concatenate(
            [xf[:5] @ gate_ns.weight.astype(mx.float32).T, xf[5:] @ gate_ns.weight.astype(mx.float32).T], 0)
        mx.eval(r_ref, s_ref, r_new, s_new, raw, raw_ref)
        ok_g = bool(mx.array_equal(raw, raw_ref).item())
        bad += not ok_g
        ok_r = bool(mx.array_equal(r_ref, r_new).item())
        ok_s = bool(mx.array_equal(s_ref, s_new).item())
        bad += not (ok_r and ok_s)
        print(dict(rows=rows, scale=scale, routed=ok_r, shared=ok_s, router=ok_g,
                   dr=float(mx.abs(r_ref.astype(mx.float32) - r_new.astype(mx.float32)).max().item()),
                   ds=float(mx.abs(s_ref.astype(mx.float32) - s_new.astype(mx.float32)).max().item())), flush=True)

# post_combine vs combine + post_mix, pre_norm_q vs project_pre_norm + quantize + astype
for rows in (1, 3, 5, 10):
    h = (mx.random.normal((1, rows, 4, D)) * 0.5).astype(mx.bfloat16)
    pre = mx.random.uniform(shape=(1, rows, 4)).astype(mx.float32)
    fn = (mx.random.normal((24, 4 * D)) * 0.01).astype(mx.float32)
    nw = (mx.random.normal((D,)) * 0.1 + 1).astype(mx.bfloat16)
    mix_a, x = hc_fuse.project_pre_norm(h, pre, fn, nw, 1e-6, 1e-20)
    mix_b, xf, xq = ffn_fuse.pre_norm_q(h, pre, fn, nw, 1e-6, 1e-20)
    ok1 = [bool(mx.array_equal(a, b).item()) for a, b in
           ((mix_a, mix_b), (x.astype(mx.float32), xf), (quantize_activation(x), xq))]
    routed = mx.random.normal((rows * 6, D)).astype(mx.bfloat16)
    shared = mx.random.normal((rows, D)).astype(mx.bfloat16)
    scale = mx.array([0.5, 0.7, 0.9], mx.float32)
    base = (mx.random.normal((24,)) * 0.1).astype(mx.float32)
    comb = moe_decode.combine(routed.reshape(1, rows, 6, D), shared.reshape(1, rows, D))
    ya, pa = hc_fuse.post_mix(comb, h, mix_a, scale, base, 1e-6, 20)
    yb, pb = ffn_fuse.post_combine(routed, shared, h, mix_a, scale, base, 1e-6, 20, 6)
    ok2 = [bool(mx.array_equal(ya, yb).item()), bool(mx.array_equal(pa, pb).item())]
    bad += not (all(ok1) and all(ok2))
    print(dict(rows=rows, pre_norm_q=ok1, post_combine=ok2), flush=True)
print('SMOKE', 'PASS' if bad == 0 else f'FAIL {bad}')
