import sys, torch
sys.path.insert(0, '/work/tools/moe_dev'); sys.path.insert(0, '/work/hooks')
import test_og as T
from split_nv import og_moe
torch.manual_seed(0)
ne = int(sys.argv[1]) if len(sys.argv) > 1 else 192
experts = torch.randperm(384)[:ne].tolist()
wt, _, lw = T.load_layer(3, experts, 0)
for M in (1024, 4096):
    x = (torch.randn(M, T.H, device='cuda') * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(ne, device='cuda')[:6] for _ in range(M)]).to(torch.int32)
    wv = torch.rand(M, 6, device='cuda')
    og_moe.moe(x, ids, wv, lw); torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); og_moe.moe(x, ids, wv, lw); e1.record(); e1.synchronize()
        best = min(best, e0.elapsed_time(e1))
    print(f'og-moe prefill M={M} ({ne} experts): {best:.3f} ms/layer', flush=True)
