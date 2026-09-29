"""Synthetic attn_out check: wo_a grouped GEMV + FP8 round trip vs grouped_gemv then quantize_activation."""
import os, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MLX_ENABLE_TF32', '0')
import mlx.core as mx
import numpy as np
from omlx.patches.deepseek_v41 import attn_out, decode_fusions, woa_compact
from omlx.patches.deepseek_v41.quantization import quantize_activation
rng = np.random.default_rng(3)
G, N, K = 8, 1024, 4096
bits = rng.normal(0, 0.02, size=(G * N, K)).astype(np.float32).view(np.uint32) >> 16
bits = (bits & 0xFFF0).astype(np.uint16)          # FP8-like: low 4 mantissa bits clear
bits[rng.integers(0, G * N, 50), rng.integers(0, K, 50)] = 0x3C81  # a few escapes
w2 = mx.array(bits).view(mx.bfloat16).reshape(G * N, K)
codes = woa_compact.encode(w2)
lin = types.SimpleNamespace(weight=w2)
lin.__dict__[woa_compact._ATTR] = codes
w3 = w2.reshape(G, N, K)
bad = 0
for m in (2, 3, 4, 5, 8, 10, 16):
    for scale in (1.0, 64.0, 1 / 64):
        x = (mx.random.normal((1, m, G, K)) * scale).astype(mx.bfloat16)
        ref_b = quantize_activation(decode_fusions.grouped_gemv(x, w3).flatten(-2))
        new_b = attn_out.grouped_gemv_q(lin, x, w3, compact=False).flatten(-2)
        oks = [bool(mx.array_equal(ref_b, new_b).item())]
        if m <= 5:
            ref_c = quantize_activation(woa_compact.grouped_gemv(lin, x, w3).flatten(-2))
            new_c = attn_out.grouped_gemv_q(lin, x, w3).flatten(-2)
            oks.append(bool(mx.array_equal(ref_c, new_c).item()))
            oks.append(bool(mx.array_equal(ref_c, ref_b).item()))
        bad += not all(oks)
        print(dict(m=m, scale=scale, ok=oks), flush=True)
print('SMOKE', 'PASS' if bad == 0 else f'FAIL {bad}')
