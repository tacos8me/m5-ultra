"""New small-M fixed_linear config (BK=256, 1 warp, 4 stages) vs the shipped (BK=64, 1 warp, 3 stages): bitwise."""
import sys
import torch
sys.path.insert(0, '/work/tools/moe_dev')
from fl_tune import run
torch.manual_seed(1)
bad = n = 0
for N, K in ((384, 5120), (32, 5120), (128, 512), (512, 5120), (1024, 5120), (64, 576), (96, 1000), (32, 64), (16, 4096)):
    w = (torch.randn(N, K, device='cuda') * 0.05).to(torch.bfloat16)
    for M in range(1, 65):
        for scale in (1.0, 30.0):
            x = (torch.randn(M, K, device='cuda') * scale).to(torch.bfloat16)
            a = run(x, w, 16, 16, 64, 1, 3)
            b = run(x, w, 16, 16, 256, 1, 4)
            n += 1
            if not torch.equal(a, b):
                bad += 1
                print('DIFF', N, K, M, scale, (a - b).abs().max().item())
print({'cases': n, 'mismatches': bad})
