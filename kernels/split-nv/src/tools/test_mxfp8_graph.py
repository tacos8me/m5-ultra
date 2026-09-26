"""GPU time (CUDA graph replay) of FlashInfer mm_mxfp8 cutlass vs b12x at step sizes."""
import torch
from flashinfer import mm_mxfp8, mxfp8_quantize

torch.manual_seed(0)
dev = 'cuda'
for K, N in ((5120, 1792), (1280, 16384), (4096, 5120), (1152, 5120), (5120, 2304), (1280, 4096)):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True, alignment=32)
    line = f'K={K:5d} N={N:5d}:'
    for M in (2, 5, 8, 65):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        for be in ('cutlass', 'b12x'):
            mm_mxfp8(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16, backend=be)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    mm_mxfp8(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16, backend=be)
            g.replay(); torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                g.replay()
            e.record(); torch.cuda.synchronize()
            line += f' M{M} {be[:4]} {s.elapsed_time(e) / 100 * 1e3:5.1f}'
    print(line)
