"""With bias_vl loaded, DS-V4.1 MoE layers route through vision_topk (moe_fused_gate with bias_alt + input_ids,
renormalize_epsilon 1e-20) instead of the plain topk (moe_fused_gate, epsilon 0). For text tokens both must give
bitwise identical weights and ids (router logits at model scale, 384 experts, top-6, sqrtsoftplus, scale 1.5)."""
import torch
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

torch.manual_seed(0)
IMG = 129264
bad = 0
for M in (1, 2, 5, 8, 64, 300, 8192):
    for scale in (0.5, 2.0, 8.0):
        logits = (torch.randn(M, 384, device="cuda") * scale).to(torch.bfloat16)
        bias = (torch.randn(384, device="cuda") * 0.1).to(torch.bfloat16)
        bias_vl = (torch.randn(384, device="cuda") * 0.1).to(torch.bfloat16)
        ids = torch.randint(0, 129000, (M,), device="cuda")
        for apply_out in (False, True):
            common = dict(topk=6, scoring_func="sqrtsoftplus", renormalize=True, routed_scaling_factor=1.5,
                          apply_routed_scaling_factor_on_output=apply_out)
            w0, i0 = moe_fused_gate(logits, bias, **common)
            w1, i1 = moe_fused_gate(logits, bias, bias_alt=bias_vl, input_ids=ids, bias_alt_token_id=IMG,
                                    renormalize_epsilon=1e-20, **common)
            same = torch.equal(w0.float(), w1.float()) and torch.equal(i0.int(), i1.int())
            bad += not same
            if not same:
                print("DIFF", M, scale, apply_out, (w0.float() - w1.float()).abs().max().item(), (i0 != i1).sum().item())
        # image rows really use bias_vl (sanity): all-image ids route like the plain kernel with bias_vl
        wv, iv = moe_fused_gate(logits, bias_vl, topk=6, scoring_func="sqrtsoftplus", renormalize=True, routed_scaling_factor=1.5)
        wi, ii = moe_fused_gate(logits, bias, bias_alt=bias_vl, input_ids=torch.full((M,), IMG, device="cuda"),
                                bias_alt_token_id=IMG, renormalize_epsilon=1e-20, topk=6, scoring_func="sqrtsoftplus",
                                renormalize=True, routed_scaling_factor=1.5)
        bad += not torch.equal(iv.int(), ii.int())
print({"cases": 7 * 3, "mismatches": bad})
raise SystemExit(bad)
