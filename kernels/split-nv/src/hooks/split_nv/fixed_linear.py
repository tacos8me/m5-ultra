"""BF16 projection matmul with an FP32 accumulation order independent of batch size."""
import torch
import triton
import triton.language as tl


# M is a runtime argument: a constexpr M compiled one kernel per distinct row count (~0.5 s JIT on the first
# prefill chunk of every new length). Masks only; each element's BK=64 MMA chain over K is unchanged.
@triton.jit(do_not_specialize=["M"])
def _linear(X, W, Y, M, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr = 16, BN: tl.constexpr = 32, BK: tl.constexpr = 64):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + k
        a = tl.load(X + m[:, None] * K + ks[None, :],
                    (m[:, None] < M) & (ks[None, :] < K), 0)
        b = tl.load(W + n[None, :] * K + ks[:, None],
                    (n[None, :] < N) & (ks[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def linear(x, weight):
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    m,k = x.shape
    n = weight.shape[0]
    out = torch.empty((m,n), dtype=torch.float32, device=x.device)
    # Tile shape, BK, warp count and stages only change how the work is spread and pipelined; each
    # element keeps the same ascending chain of k16 MMAs over K, so every config below is bitwise
    # equal (tools/test_fixed_linear.py, tools/moe_dev/fl_bitwise.py: 1152 cases). Small M is
    # latency-bound on the K chain: BK=256 x 4 stages cuts a K=5120 call from ~19 to ~7 us.
    BM, BN, BK, warps, stages = (16, 16, 256, 1, 4) if m <= 64 else (32, 32, 64, 4, 2)
    _linear[(triton.cdiv(m,BM),triton.cdiv(n,BN))](x,weight,out,m,n,k,BM=BM,BN=BN,BK=BK,
                                                num_warps=warps,num_stages=stages)
    return out
