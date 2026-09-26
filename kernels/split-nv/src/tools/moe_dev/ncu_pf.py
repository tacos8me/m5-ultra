import sys, torch
sys.path.insert(0, '/work/tools/moe_dev'); sys.path.insert(0, '/work/hooks')
import test_og as T
from split_nv import og_moe
torch.manual_seed(0)
ne = 64; M = 1536
experts = torch.randperm(384)[:ne].tolist()
wt, _, lw = T.load_layer(3, experts, 0)
x = (torch.randn(M, T.H, device='cuda') * 0.5).to(torch.bfloat16)
ids = torch.stack([torch.randperm(ne, device='cuda')[:6] for _ in range(M)]).to(torch.int32)
wv = torch.rand(M, 6, device='cuda')
og_moe.moe(x, ids, wv, lw); torch.cuda.synchronize()
og_moe.moe(x, ids, wv, lw); torch.cuda.synchronize()
