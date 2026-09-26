"""dsv41 RMSNorm: the Triton per-row kernel is batch-invariant; the torch path (used for >64 rows) is a different
reduction. Measures how often they differ and checks row invariance of the Triton kernel across M."""
import torch
from sglang.kernels.ops.attention.dsv4.rmsnorm_fp32 import rmsnorm_fp32

torch.manual_seed(0)
for D in (128, 512):
    w = (1 + 0.1 * torch.randn(D, device="cuda")).float()
    x = torch.randn(8192, D, device="cuda").to(torch.bfloat16) * 3
    ref = rmsnorm_fp32(x, w, 1e-6)
    inv = all(torch.equal(rmsnorm_fp32(x[:m].contiguous(), w, 1e-6), ref[:m]) for m in (1, 2, 5, 41, 64, 65, 78, 300, 4096))
    xf = x.float()
    torch_path = (w * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6))).to(torch.bfloat16)
    rows_differ = int((torch_path != ref).any(-1).sum())
    tp_inv = all(torch.equal((w * (xf[:m] * torch.rsqrt(xf[:m].square().mean(-1, keepdim=True) + 1e-6))).to(torch.bfloat16), torch_path[:m])
                 for m in (65, 78, 300, 4096))
    print({"D": D, "triton_row_invariant": inv, "torch_vs_triton_rows_differ_of_8192": rows_differ, "torch_row_invariant_65..4096": tp_inv})
