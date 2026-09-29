"""Synthetic check: attn_in.wqb_rope vs wq_b.project_quantized + rope_range (singleton rows 1-5)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MLX_ENABLE_TF32', '0'); os.environ.setdefault('DS41_MHC', '1'); os.environ.setdefault('DS41_FAST_ROPE', '1')
import mlx.core as mx
import numpy as np
from omlx.patches.deepseek_v41 import attn_in, language
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, quantize_activation
rng = np.random.default_rng(5)
import json
raw = json.load(open(Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx/config.json'))
c = ModelConfig.from_dict(raw)
H, HD, K = c.n_heads, c.head_dim, c.q_lora_rank
b = rng.integers(0, 256, size=(H * HD, K), dtype=np.uint8)
b = np.where((b & 0x7f) >= 0x78, b & 0xb7, b).astype(np.uint8)
wq_b = QuantizedProjection(mx.array(b.view(np.uint32)), mx.array(rng.integers(118, 126, size=(H * HD, K // 32), dtype=np.uint8)), 8, 'mxfp8')
mx.eval(wq_b.weight)
bad = 0
print('fast_rope', language.DS41_FAST_ROPE)
for ratio in (0, 2):
    for start in (100, 8191, 524287):
        for m in (1, 2, 3, 4, 5):
            qr8 = quantize_activation((mx.random.normal((1, m, K)) * 3).astype(mx.bfloat16))
            d = language.rope_params(c, bool(ratio))[0]
            assert attn_in.wqb_rope_supported(wq_b, qr8, H, HD, d)
            ref = language.rope_range(wq_b.project_quantized(qr8).reshape(1, m, H, HD), start, m, c, bool(ratio))
            new = attn_in.wqb_rope(qr8, wq_b, *language.rope_tables(start, m, c, bool(ratio)), H, HD)
            mx.eval(ref, new)
            ok = bool(mx.array_equal(ref, new).item())
            bad += not ok
            if not ok:
                print(dict(ratio=ratio, start=start, m=m, ok=ok, diff=float(mx.abs(ref.astype(mx.float32) - new.astype(mx.float32)).max().item())), flush=True)
print('SMOKE', 'PASS' if bad == 0 else f'FAIL {bad}')
