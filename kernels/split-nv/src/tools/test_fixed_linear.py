"""fixed_linear variants: bitwise equality with the shipped config and graph-replay time at step sizes."""
import torch
import triton
import sys
sys.path.insert(0, '/home/ian/split-nv/hooks')
from split_nv.fixed_linear import _linear

torch.manual_seed(0)


def run(x, w, BM, BN, BK, warps, stages):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _linear[(triton.cdiv(m, BM), triton.cdiv(n, BN))](x, w, out, m, n, k, BM=BM, BN=BN, BK=BK,
                                                     num_warps=warps, num_stages=stages)
    return out


cfgs = [(16, 32, 64, 4, 2), (16, 16, 64, 4, 2), (16, 16, 64, 2, 2), (16, 16, 64, 1, 3), (16, 32, 64, 2, 3), (32, 32, 64, 4, 2)]
for n, k in ((384, 5120), (512, 5120), (1024, 5120), (32, 5120), (128, 512)):
    w = (torch.randn(n, k, device='cuda') * 0.02).to(torch.bfloat16)
    line = f'N={n:5d} K={k}:'
    for m in (5, 65, 8192):
        x = torch.randn(m, k, device='cuda').to(torch.bfloat16)
        ref = run(x, w, *cfgs[0])
        for c in cfgs:
            o = run(x, w, *c)
            eq = torch.equal(o, ref)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    run(x, w, *c)
            g.replay(); torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); [g.replay() for _ in range(5)]; e.record(); torch.cuda.synchronize()
            if True:
                line += f' | M{m} {c[:2]}w{c[3]}s{c[4]} {"=" if eq else "DIFF"} {s.elapsed_time(e) / 50 * 1e3:.1f}us'
    print(line)
