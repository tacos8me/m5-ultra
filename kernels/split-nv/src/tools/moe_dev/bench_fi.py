"""Baseline: FlashInfer CUTLASS W4A8 MXFP4 MoE (the live path) on real layer weights, cold experts; plus routing stats."""
import sys
import time

import torch

sys.path.insert(0, '/work/tools/moe_dev')
import weights as W  # noqa: E402

torch.manual_seed(0)
dev = 'cuda'
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 160

# --- routing stats from the 8K-fixture traces (cuda side, rows 8205..8212) ---
tr = torch.load('/traces/e0-base.pt', map_location='cpu', weights_only=True)
distinct = {L: [] for L in range(1, 9)}
for layer in range(20):
    x = tr[f'{layer}.post_attention_layernorm'][1]
    idx, _ = W.route(x, layer)
    for L in range(1, 9):
        for s in range(0, 8 - L + 1):
            distinct[L].append(len(set(idx[s:s + L].flatten().tolist())))
dmean = {L: sum(v) / len(v) for L, v in distinct.items()}
print('distinct experts per layer (layers 0-19, 8K fixture rows), L=1..8:',
      ' '.join(f'{L}:{dmean[L]:.1f}' for L in range(1, 9)), flush=True)

# --- swizzle check vs flashinfer ---
from flashinfer import block_scale_interleave, mxfp8_quantize  # noqa: E402
from flashinfer.fused_moe import cutlass_fused_moe  # noqa: E402
from flashinfer.autotuner import autotune  # noqa: E402

t = torch.randint(0, 255, (3, 256, 36), dtype=torch.uint8)
ref = block_scale_interleave(t.to(dev)).reshape(t.shape).cpu()
assert torch.equal(ref, W.swizzle_sf(t)), 'swizzle mismatch'
print('swizzle == flashinfer block_scale_interleave', flush=True)

t0 = time.time()
ids = torch.randperm(384)[:NE].tolist()
wt = W.load_experts(LAYER, ids, rank=0)
print(f'loaded {NE} experts of layer {LAYER} in {time.time() - t0:.1f}s, '
      f'{sum(v.numel() for k, v in wt.items() if not k.endswith("lin")) / 1e9:.2f} GB', flush=True)
per_expert = (wt['w13'][0].numel() + wt['w2'][0].numel() + wt['w13_sf'][0].numel() + wt['w2_sf'][0].numel())
print(f'bytes per expert per GPU: {per_expert / 1e6:.2f} MB', flush=True)
gs = torch.ones(NE, dtype=torch.float32, device=dev)
limit = torch.full((NE,), 10.0, dtype=torch.float32, device=dev)
qs = [wt['w13_sf'].view(torch.int32), gs, wt['w2_sf'].view(torch.int32), gs]


def fi(x, tid, tw, out):
    xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    cutlass_fused_moe(input=xq, token_selected_experts=tid, token_final_scales=tw,
                      fc1_expert_weights=wt['w13'].view(torch.int64), fc2_expert_weights=wt['w2'].view(torch.int64),
                      output_dtype=torch.bfloat16, quant_scales=qs, input_sf=xsf, swiglu_limit=limit,
                      tp_size=2, tp_rank=0, use_mxfp8_act_scaling=True, tune_max_num_tokens=max(8, 1 << (x.shape[0] - 1).bit_length()),
                      output=out, use_fused_finalize=False)
    return out


def routes(M, n_distinct, reps):
    """reps routing sets with exactly n_distinct distinct experts (>= 6) each; consecutive sets disjoint-ish (cold)."""
    nd = max(n_distinct, 6)
    out = []
    for r in range(reps):
        pool = torch.randperm(NE)[:nd]
        tid = torch.stack([pool[[(m * 6 + j) % nd for j in range(6)]] for m in range(M)]).to(torch.int32)
        out.append(tid.to(dev))
    return out


def timeit(fn, M, sets, iters=3):
    x = torch.randn(M, W.H, device=dev, dtype=torch.bfloat16)
    tw = torch.rand(M, 6, device=dev) * 0.5
    out = torch.empty(M, W.H, device=dev, dtype=torch.bfloat16)
    with autotune(True):
        fn(x, sets[0], tw, out)
    torch.cuda.synchronize()
    for tid in sets:
        fn(x, tid, tw, out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for i, tid in enumerate(sets):
                fn(x, tid, tw, out)
        g.replay()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001  FlashInfer autotuner can break capture: time eagerly (queue many calls)
        torch.cuda.synchronize()

        class G:
            def replay(self):
                for tid in sets:
                    fn(x, tid, tw, out)
        g = G()
        print('  (eager timing)', flush=True)
    best = 1e9
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / len(sets))
    return best


for M in ([] if len(sys.argv) > 3 else (1, 2, 3, 4, 5, 8)):
    nd = round(dmean[M])
    sets = routes(M, nd, 24)
    real_nd = sum(len(set(t.flatten().tolist())) for t in sets) / len(sets)
    us = timeit(fi, M, sets)
    floor = real_nd * per_expert / 1.6e12 * 1e6
    print(f'FI M={M}: distinct {real_nd:.1f}  {us:7.1f} us/layer  (bytes floor @1.6TB/s {floor:5.1f} us, '
          f'{real_nd * per_expert / us / 1e6:.2f} TB/s eff)', flush=True)
for M in ([int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else (2048, 8192)):
    tid = torch.stack([torch.randperm(NE)[:6] for _ in range(M)]).to(torch.int32).to(dev)
    us = timeit(fi, M, [tid], iters=3)
    print(f'FI M={M}: {us:8.1f} us/layer', flush=True)
