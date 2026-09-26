import sys, torch
sys.path.insert(0, '/work/tools/moe_dev'); sys.path.insert(0, '/work/hooks')
import test_og as T
from split_nv import og_moe
torch.manual_seed(0)
ne = 192
experts = torch.randperm(384)[:ne].tolist()
wt, _, lw = T.load_layer(3, experts, 0)
x8, _ = T.trace_rows(3)
from torch.profiler import profile, ProfilerActivity
for M, nd in ((1, 6), (5, 19), (8, 28)):
    sets = []
    for r in range(24):
        pool = torch.randperm(ne)[:nd]
        sets.append(torch.stack([pool[[(m * 6 + j) % nd for j in range(6)]] for m in range(M)]).to(torch.int32).cuda())
    x = x8[:M].contiguous(); wv = torch.rand(M, 6, device='cuda')
    for s in sets: og_moe.moe(x, s, wv, lw)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for s in sets: og_moe.moe(x, s, wv, lw)
        torch.cuda.synchronize()
    for e in prof.key_averages():
        if 'og::' in e.key:
            print(f'M={M} {e.key[:40]:40s} avg {e.device_time / e.count if hasattr(e, "device_time") else e.cuda_time:.1f} us', flush=True)
