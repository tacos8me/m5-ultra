"""Runtime-M fixed_linear == the previous constexpr-M kernel, bitwise, over the model's shapes and many M."""
import sys
import torch
import triton
import triton.language as tl
sys.path.insert(0, '/home/ian/split-nv/hooks')
from split_nv.fixed_linear import linear


@triton.jit
def _old(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr = 16, BN: tl.constexpr = 32, BK: tl.constexpr = 64):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + k
        a = tl.load(X + m[:, None] * K + ks[None, :], (m[:, None] < M) & (ks[None, :] < K), 0)
        b = tl.load(W + n[None, :] * K + ks[:, None], (n[None, :] < N) & (ks[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def old(x, w):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    BM, BN, warps, stages = (16, 16, 1, 3) if m <= 64 else (32, 32, 4, 2)
    _old[(triton.cdiv(m, BM), triton.cdiv(n, BN))](x, w, out, m, n, k, BM=BM, BN=BN, num_warps=warps, num_stages=stages)
    return out


torch.manual_seed(0)
bad = 0
for n, k in ((384, 5120), (512, 5120), (1024, 5120), (32, 5120), (128, 512), (64, 1024)):
    w = (torch.randn(n, k, device='cuda') * 0.02).to(torch.bfloat16)
    for m in (1, 2, 3, 5, 8, 16, 17, 56, 64, 65, 300, 1809, 3000, 8191, 8192):
        x = torch.randn(m, k, device='cuda').to(torch.bfloat16)
        a, b = old(x, w), linear(x, w)
        if not torch.equal(a, b):
            bad += 1
            print('DIFF', n, k, m, (a - b).abs().max().item())
print({'shapes': 6 * 15, 'different': bad})
sys.exit(1 if bad else 0)
