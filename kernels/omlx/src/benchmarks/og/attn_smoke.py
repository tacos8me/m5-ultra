"""Synthetic attn_in check: pre-norm+FP8 and the dual wq_a/wkv launch vs the unfused kernels."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MLX_ENABLE_TF32', '0'); os.environ.setdefault('DS41_MHC', '1')
import mlx.core as mx
import numpy as np
from omlx.patches.deepseek_v41 import attn_in, fast_qmv, hc_fuse, og_fused
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, quantize_activation
rng = np.random.default_rng(1)
D = 5120
def fp8(n, k):
    b = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    b = np.where((b & 0x7f) >= 0x78, b & 0xb7, b).astype(np.uint8)
    return mx.array(b.view(np.uint32))
def e8(n, g):
    return mx.array(rng.integers(118, 126, size=(n, g), dtype=np.uint8))
wq_a = QuantizedProjection(fp8(1280, D), e8(1280, D // 32), 8, 'mxfp8')
wkv = QuantizedProjection(fp8(512, D), e8(512, D // 32), 8, 'mxfp8')
mx.eval(wq_a.weight, wkv.weight)
bad = 0
for m in (1, 2, 3, 4, 5, 6, 8, 10, 16):
    for scale in (1.0, 0.02, 40.0):
        x = (mx.random.normal((1, m, D)) * scale).astype(mx.bfloat16)
        xq = quantize_activation(x)
        if m <= 5:
            ref = (wq_a.project_quantized(xq), wkv.project_quantized(xq))
        else:
            ref = (og_fused._rows(wq_a, xq), og_fused._rows(wkv, xq))
        if m == 2:
            alt = (og_fused._rows(wq_a, xq), og_fused._rows(wkv, xq))
        new = attn_in.input_projections(xq, wq_a, wkv)
        mx.eval(ref, new)
        ok = [bool(mx.array_equal(a, b).item()) for a, b in zip(ref, new)]
        if m == 2:
            mx.eval(alt)
            ok += [bool(mx.array_equal(a, b).item()) for a, b in zip(ref, alt)]
        bad += not all(ok)
        print(dict(m=m, scale=scale, ok=ok), flush=True)
for rows in (1, 2, 5, 10):
    h = (mx.random.normal((1, rows, 4, D)) * 0.7).astype(mx.bfloat16)
    pre = mx.random.uniform(shape=(1, rows, 4)).astype(mx.float32)
    fn = (mx.random.normal((24, 4 * D)) * 0.01).astype(mx.float32)
    nw = (mx.random.normal((D,)) * 0.1 + 1).astype(mx.bfloat16)
    mix_a, x = hc_fuse.project_pre_norm(h, pre, fn, nw, 1e-6, 1e-20)
    mix_b, y, yq = attn_in.project_pre_norm_q(h, pre, fn, nw, 1e-6, 1e-20)
    ok = [bool(mx.array_equal(a, b).item()) for a, b in ((mix_a, mix_b), (x, y), (quantize_activation(x), yq))]
    bad += not all(ok)
    print(dict(rows=rows, pre_norm_q=ok), flush=True)
print('SMOKE', 'PASS' if bad == 0 else f'FAIL {bad}')
