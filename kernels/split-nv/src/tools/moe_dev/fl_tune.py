"""fixed_linear small-M configs: bitwise equality with the shipped (16,16,64,w1,s3) and cold-weight time."""
import sys
import torch
import triton
sys.path.insert(0, '/work/hooks')
from split_nv.fixed_linear import _linear

torch.manual_seed(0)


def run(x, w, BM, BN, BK, warps, stages, out=None):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device) if out is None else out
    _linear[(triton.cdiv(m, BM), triton.cdiv(n, BN))](x, w, out, m, n, k, BM=BM, BN=BN, BK=BK, num_warps=warps,
                                                     num_stages=stages)
    return out


base = (16, 16, 64, 1, 3)
cfgs = [base, (16, 16, 128, 1, 4), (16, 16, 128, 1, 6), (16, 16, 256, 1, 4), (16, 16, 256, 2, 4), (16, 16, 128, 2, 6),
        (16, 16, 512, 4, 3), (16, 16, 256, 4, 4), (16, 16, 128, 4, 8)]
for n, k in ((384, 5120), (1024, 5120), (32, 5120)):
    ws = [(torch.randn(n, k, device='cuda') * 0.02).to(torch.bfloat16) for _ in range(max(2, int(160e6 / (n * k * 2))))]
    for m in (2, 5, 8, 64):
        x = torch.randn(m, k, device='cuda').to(torch.bfloat16)
        ref = run(x, ws[0], *base)
        line = f'N={n} K={k} M={m}:'
        for c in cfgs:
            try:
                o = run(x, ws[0], *c)
            except Exception as e:  # noqa: BLE001
                line += f' | {c} ERR'
                continue
            eq = torch.equal(o, ref)
            out = torch.empty((m, n), dtype=torch.float32, device='cuda')
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for w in ws:
                    run(x, w, *c, out=out)
            g.replay(); torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); [g.replay() for _ in range(3)]; e.record(); torch.cuda.synchronize()
            line += f' | {c[2]}w{c[3]}s{c[4]} {"=" if eq else "DIFF"} {s.elapsed_time(e) / (3 * len(ws)) * 1e3:.1f}us'
        print(line, flush=True)
