"""Compare FlashInfer mm_mxfp8 backends (cutlass vs b12x): bitwise equality per row and small-M speed."""
import time
import torch
from flashinfer import mm_mxfp8, mxfp8_quantize

torch.manual_seed(0)
dev = 'cuda'
for K, N in ((5120, 1792), (1280, 16384), (4096, 5120), (1152, 5120), (5120, 2304)):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True, alignment=32)
    res = {}
    for M in (5, 65, 8192):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        outs = {}
        for be in ('cutlass', 'b12x'):
            try:
                o = mm_mxfp8(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16, backend=be)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    mm_mxfp8(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16, backend=be)
                torch.cuda.synchronize()
                outs[be] = (o, (time.perf_counter() - t0) / 20 * 1e6)
            except Exception as e:  # noqa: BLE001
                outs[be] = (None, repr(e)[:120])
        c, b = outs['cutlass'], outs['b12x']
        eq = None if c[0] is None or b[0] is None else torch.equal(c[0], b[0])
        # batch invariance of b12x: first 5 rows computed alone vs inside the batch
        inv = None
        if b[0] is not None and M > 5:
            o5 = mm_mxfp8(xq[:5].contiguous(), wq.t(), mxfp8_quantize(x[:5], is_sf_swizzled_layout=True, alignment=32)[1], ws,
                          out_dtype=torch.bfloat16, backend='b12x')
            inv = torch.equal(o5, b[0][:5])
        print(f'K={K} N={N} M={M}: cutlass {c[1] if isinstance(c[1], str) else round(c[1], 1)} us | b12x '
              f'{b[1] if isinstance(b[1], str) else round(b[1], 1)} us | equal {eq} | b12x rows invariant {inv}')
