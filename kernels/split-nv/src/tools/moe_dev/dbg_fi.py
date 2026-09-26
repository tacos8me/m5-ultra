import sys, torch, traceback
sys.path.insert(0, '/work/tools/moe_dev')
import weights as W
from flashinfer import mxfp8_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.autotuner import autotune
dev='cuda'; NE=16
wt = W.load_experts(3, list(range(NE)), rank=0)
gs = torch.ones(NE, dtype=torch.float32, device=dev)
limit = torch.full((NE,), 10.0, dtype=torch.float32, device=dev)
qs = [wt['w13_sf'].view(torch.int32), gs, wt['w2_sf'].view(torch.int32), gs]
M=int(sys.argv[1]); fuse = sys.argv[2]=='1'
x = torch.randn(M, W.H, device=dev, dtype=torch.bfloat16)
tid = torch.stack([torch.randperm(NE)[:6] for _ in range(M)]).to(torch.int32).to(dev)
tw = torch.rand(M, 6, device=dev)
out = torch.empty(M, W.H, device=dev, dtype=torch.bfloat16)
def fi(step):
    xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    print('q ok', step, flush=True)
    cutlass_fused_moe(input=xq, token_selected_experts=tid, token_final_scales=tw,
                      fc1_expert_weights=wt['w13'].view(torch.int64), fc2_expert_weights=wt['w2'].view(torch.int64),
                      output_dtype=torch.bfloat16, quant_scales=qs, input_sf=xsf, swiglu_limit=limit,
                      tp_size=2, tp_rank=0, use_mxfp8_act_scaling=True, tune_max_num_tokens=8,
                      output=out, use_fused_finalize=fuse)
with autotune(True):
    fi('tune')
fi('eager'); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        fi("cap"); fi("cap2")
    g.replay(); torch.cuda.synchronize(); print('graph OK')
except Exception as e:
    traceback.print_exc()
