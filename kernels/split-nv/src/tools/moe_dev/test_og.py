"""og-moe checks on real layer weights (standalone, one GPU, small allocations).

  test_og.py bitwise LAYER        decode(M) == prefill rows, step rows == rows inside a big prefill, repeatability
  test_og.py emul LAYER           routed part == float64 block-promotion emulation (shared weights zeroed)
  test_og.py fidelity L0,L1,...   og-moe (rank0 + rank1, bf16 all-reduce) vs the official inference/model.py arithmetic
                                  on the traced 8K-fixture FFN inputs; the traced CUDA engine output as the baseline
  test_og.py bench LAYER          decode us/layer for M=1..8 (cold experts) and prefill ms/layer
"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/work/tools/moe_dev')
sys.path.insert(0, '/work/hooks')
import weights as W  # noqa: E402
from split_nv import og_moe  # noqa: E402

dev = 'cuda'
H, I = W.H, W.I
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device=dev)


def load_layer(layer, experts, rank):
    wt = W.load_experts(layer, experts, rank)
    sh = W.shared_host(layer, rank)
    # engine layout (MergedColumnParallelLinear): gate (w1) rows first, then up (w3)
    s13 = torch.cat([sh['w1'], sh['w3']]).to(dev)
    s13_sf = torch.cat([sh['s1'], sh['s3']]).contiguous().to(dev)
    lw = og_moe.LayerWeights(wt['w13'], wt['w13_sf'], wt['w2'], wt['w2_sf'], s13, s13_sf, sh['w2'].to(dev),
                             sh['s2'].to(dev), s_up0=I, s_gate0=0)
    return wt, (s13, s13_sf, sh['w2'].to(dev), sh['s2'].to(dev)), lw


def trace_rows(layer):
    tr = torch.load('/traces/e0-base.pt', map_location='cpu', weights_only=True)
    return tr[f'{layer}.post_attention_layernorm'][1].to(dev), tr[f'{layer}.mlp'][1].to(dev)


def compact_routes(ids, experts):
    remap = {e: j for j, e in enumerate(experts)}
    return torch.tensor([[remap[int(e)] for e in row] for row in ids.tolist()], dtype=torch.int32, device=dev)


# ------------------------------------------------------------------------------------------ bitwise
def bitwise(layer):
    torch.manual_seed(0)
    x8, _ = trace_rows(layer)
    ids, w = W.route(x8, layer)
    used = sorted(set(ids.flatten().tolist()))
    extra = [e for e in torch.randperm(384).tolist() if e not in used][:64 - len(used)]
    experts = used + extra
    _, _, lw = load_layer(layer, experts, 0)
    cid = compact_routes(ids, experts)
    ok = True
    for M in range(1, 9):
        x, i_, w_ = x8[:M].contiguous(), cid[:M].contiguous(), w[:M].float().contiguous()
        a = og_moe.decode(x, i_, w_, lw)
        b = og_moe.prefill(x, i_, w_, lw)
        c = og_moe.decode(x, i_, w_, lw)
        eq, rep = torch.equal(a, b), torch.equal(a, c)
        ok &= eq and rep
        print(f'M={M}: decode == prefill {eq}; decode repeat {rep}; |out| {a.float().norm():.1f}', flush=True)
    # rows inside a large prefill (random other rows and routes), across tile boundaries
    for big in (9, 64, 129, 300, 2048, 8192):
        xb = (torch.randn(big, H, device=dev) * x8.float().std()).to(torch.bfloat16)
        ib = torch.stack([torch.randperm(len(experts), device=dev)[:6] for _ in range(big)]).to(torch.int32)
        wb = torch.rand(big, 6, device=dev)
        spots = [0] + ([137] if 142 <= big - 5 else []) + ([big - 5] if big >= 10 else [])
        for at in spots:
            xb[at:at + 5], ib[at:at + 5], wb[at:at + 5] = x8[:5], cid[:5], w[:5].float()
        full = og_moe.prefill(xb.contiguous(), ib.contiguous(), wb.contiguous(), lw)
        step = og_moe.decode(x8[:5].contiguous(), cid[:5].contiguous(), w[:5].float().contiguous(), lw)
        for at in spots:
            eq = torch.equal(full[at:at + 5], step)
            ok &= eq
            print(f'prefill M={big}: rows {at}..{at + 4} == decode(5) {eq}', flush=True)
    # padded decode: valid = 1 of W = 2 rows -> row 0 equals decode(1)
    v = torch.ones(1, dtype=torch.int32, device=dev)
    p2 = og_moe.decode(x8[:2].contiguous(), cid[:2].contiguous(), w[:2].float().contiguous(), lw, valid=v)
    one = og_moe.decode(x8[:1].contiguous(), cid[:1].contiguous(), w[:1].float().contiguous(), lw)
    eq = torch.equal(p2[:1], one) and bool((p2[1] == 0).all())
    ok &= eq
    print(f'padded W=2 valid=1: row 0 == decode(1), row 1 zero: {eq}')
    print('BITWISE', 'PASS' if ok else 'FAIL')


# ------------------------------------------------------------------------------------------ emulation
def act_quant(x):
    """official act_quant on bf16/fp32 rows -> (codes as float64 unscaled e4m3 values, exponent int)."""
    x = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    r = amax * torch.tensor(1 / 448, dtype=torch.float32)
    bits = r.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    q = (x * torch.exp2(-e.float())).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float64)
    return q, e.squeeze(-1)


def e8m0(sf):
    return sf.long() - 127


def unswz(sf, n, kb):
    """swizzled [n, kb] (one expert) -> linear"""
    x = sf.reshape(n // 128, kb // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)  # [nB, q, n&31, kbB, kb&3]
    return x.reshape(n, kb)


def promo_gemm(q, e, wv, we):
    """q [M, KB, 32] f64, e [M, KB], wv [N, KB, 32] f64, we [N, KB] -> fp32 [M, N]: groups of 4 k-blocks summed
    exactly (the tensor-core chain is ~exact), rounded to fp32, promoted in ascending order."""
    blk = torch.einsum('mkb,nkb->mnk', q, wv)  # exact in f64
    sc = torch.ldexp(torch.ones_like(blk), (e[:, None, :] + we[None, :, :]).to(torch.int32))
    d = (blk * sc).reshape(*blk.shape[:2], -1, 4).sum(-1).to(torch.float32)
    acc = torch.zeros(d.shape[:2], dtype=torch.float32, device=dev)
    for k in range(d.shape[2]):
        acc = acc + d[:, :, k]
    return acc


def emul(layer):
    torch.manual_seed(0)
    x8, _ = trace_rows(layer)
    ids, w = W.route(x8, layer)
    experts = sorted(set(ids.flatten().tolist()))
    wt, (s13, s13_sf, s2, s2_sf), _ = load_layer(layer, experts, 0)
    lw0 = og_moe.LayerWeights(wt['w13'], wt['w13_sf'], wt['w2'], wt['w2_sf'], torch.zeros_like(s13), s13_sf,
                              torch.zeros_like(s2), s2_sf, s_up0=I, s_gate0=0)
    cid = compact_routes(ids, experts)
    M = 5
    x = x8[:M].contiguous()
    out = og_moe.decode(x, cid[:M].contiguous(), w[:M].float().contiguous(), lw0)
    q, e = act_quant(x)
    D = torch.zeros(M, 7, H, dtype=torch.float32, device=dev)
    for j, ex in enumerate(experts):
        w13 = wt['w13'][j]
        codes = torch.stack((w13 & 15, w13 >> 4), -1).reshape(2 * I, 160, 32).long()
        wv = FP4[codes]
        we = e8m0(unswz(wt['w13_sf'][j], 2 * I, 160))
        acc = promo_gemm(q, e, wv, we)  # [M, 2I]
        U = acc[:, :I].to(torch.bfloat16).float().clamp(-10, 10)
        G = acc[:, I:].to(torch.bfloat16).float().clamp(max=10)
        hh = (G / (1 + torch.exp(-G))) * U
        for t in range(M):
            hit = (cid[t] == j).nonzero()
            if len(hit) == 0:
                continue
            slot = int(hit[0])
            hb = (w[t, slot].float() * hh[t]).to(torch.bfloat16)
            hq_, he = act_quant(hb[None])
            w2 = wt['w2'][j]
            c2 = torch.stack((w2 & 15, w2 >> 4), -1).reshape(H, 36, 32).long()
            d = promo_gemm(hq_, he, FP4[c2], e8m0(unswz(wt['w2_sf'][j], H, 36)))[0]
            pos = int((cid[t] < j).sum())
            D[t, pos] = d.to(torch.bfloat16).float()
    ref = torch.zeros(M, H, dtype=torch.float32, device=dev)
    for p in range(7):
        ref = ref + D[:, p]
    ref = ref.to(torch.bfloat16)
    eq = (ref == out).float().mean().item()
    rel = ((ref.float() - out.float()).norm() / ref.float().norm()).item()
    print(f'EMUL layer {layer}: exact-match fraction {eq:.6f}, relRMS {rel:.2e}  (chained MMA groups are ~exact, so rare bf16 ulp flips are expected)')


# ------------------------------------------------------------------------------------------ fidelity
def official(x, layer, ids, w, cache):
    """inference/model.py MoE arithmetic in float64 matmuls (og_ref.py rules): full expert (both TP halves)."""
    def deq_fp4(wu8, s):
        codes = torch.stack((wu8 & 15, wu8 >> 4), -1).flatten(-2).long()
        return FP4[codes].float() * torch.ldexp(torch.ones(1, device=dev), (s.long() - 127).int()).repeat_interleave(32, 1)

    def deq_fp8(wu8, s):
        v = wu8.view(torch.float8_e4m3fn).float()
        sc = torch.ldexp(torch.ones(1, device=dev), (s.long() - 127).int())
        return v * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)

    def aq(v):
        q, e = act_quant(v)
        return (q * torch.exp2(e.double())[..., None]).reshape(v.shape)

    def expert(xin, w1, w3, w2, wt=None):
        xq = aq(xin)
        g = (xq @ w1.double().T).to(torch.bfloat16).float()
        u = (xq @ w3.double().T).to(torch.bfloat16).float()
        u, g = u.clamp(-10, 10), g.clamp(max=10)
        h = (g / (1 + torch.exp(-g))) * u
        if wt is not None:
            h = wt * h
        return (aq(h.to(torch.bfloat16)) @ w2.double().T).to(torch.bfloat16)

    y = torch.zeros(x.shape, dtype=torch.float32, device=dev)
    for e in sorted(set(ids.flatten().tolist())):
        p = f'layers.{layer}.ffn.experts.{e}.'
        w1, w3, w2 = [deq_fp4(W.raw(p + n + '.weight').to(dev), W.raw(p + n + '.scale').to(dev)) for n in ('w1', 'w3', 'w2')]
        rows, slots = (ids == e).nonzero(as_tuple=True)
        y[rows] += expert(x[rows], w1, w3, w2, w[rows, slots, None].float()).float()
    p = f'layers.{layer}.ffn.shared_experts.'
    sw = [deq_fp8(W.raw(p + n + '.weight').to(dev), W.raw(p + n + '.scale').to(dev)) for n in ('w1', 'w3', 'w2')]
    y += expert(x, *sw).float()
    return y.to(torch.bfloat16)


def relrms(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def fidelity(layers):
    rows = []
    for layer in layers:
        x8, cuda_out = trace_rows(layer)
        ids, w = W.route(x8, layer)
        experts = sorted(set(ids.flatten().tolist()))
        cid = compact_routes(ids, experts)
        parts = []
        for rank in (0, 1):
            _, _, lw = load_layer(layer, experts, rank)
            parts.append(og_moe.moe(x8.contiguous(), cid.contiguous(), w.float().contiguous(), lw))
            del lw
            torch.cuda.empty_cache()
        ours = (parts[0].float() + parts[1].float()).to(torch.bfloat16)  # bf16 all-reduce of two ranks
        ref = official(x8, layer, ids, w, {})
        r_ours, r_cuda = relrms(ours, ref), relrms(cuda_out, ref)
        rows.append((layer, r_ours, r_cuda))
        print(f'FIDELITY layer {layer:2d}: og-moe vs official {r_ours:.5f} | live engine (FlashInfer) trace vs official {r_cuda:.5f}',
              flush=True)
    import statistics
    print(f'FIDELITY mean: og-moe {statistics.mean(r[1] for r in rows):.5f}, engine {statistics.mean(r[2] for r in rows):.5f}; '
          f'max og-moe {max(r[1] for r in rows):.5f}')


# ------------------------------------------------------------------------------------------ bench
def bench(layer, ne=192):
    torch.manual_seed(0)
    experts = torch.randperm(384)[:ne].tolist()
    wt, _, lw = load_layer(layer, experts, 0)
    per = (wt['w13'][0].numel() + wt['w2'][0].numel() + wt['w13_sf'][0].numel() + wt['w2_sf'][0].numel())
    shared_b = 3 * I * H + (2 * I // 32) * 160 + 160 * 36
    x8, _ = trace_rows(layer)
    dmean = {1: 6.0, 2: 9.7, 3: 12.9, 4: 16.1, 5: 19.3, 6: 22.0, 7: 24.7, 8: 27.7}
    for M in range(1, 9):
        nd = max(6, round(dmean[M]))
        sets = []
        for r in range(24):
            pool = torch.randperm(ne)[:nd]
            sets.append(torch.stack([pool[[(m * 6 + j) % nd for j in range(6)]] for m in range(M)]).to(torch.int32).to(dev))
        x = x8[:M].contiguous()
        wv = torch.rand(M, 6, device=dev)
        for s in sets:
            og_moe.moe(x, s, wv, lw)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for s in sets:
                og_moe.moe(x, s, wv, lw)
        g.replay()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(5):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1) * 1e3 / len(sets))
        byts = nd * per + shared_b
        print(f'og-moe decode M={M}: distinct {nd} + shared: {best:6.1f} us/layer; bytes {byts / 1e6:.0f} MB -> '
              f'{byts / best / 1e6:.2f} TB/s ({byts / 1.745e6 / best * 100:.0f}% of 1.745)', flush=True)
    for M in (1024, 4096):
        x = (torch.randn(M, H, device=dev) * x8.float().std()).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(ne, device=dev)[:6] for _ in range(M)]).to(torch.int32)
        wv = torch.rand(M, 6, device=dev)
        og_moe.moe(x, ids, wv, lw)
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(3):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); og_moe.moe(x, ids, wv, lw); e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1))
        print(f'og-moe prefill M={M} ({ne} experts): {best:.3f} ms/layer', flush=True)


if __name__ == '__main__':
    t0 = time.time()
    og_moe.ext()
    print(f'built in {time.time() - t0:.0f}s', flush=True)
    mode = sys.argv[1]
    if mode == 'bitwise':
        bitwise(int(sys.argv[2]))
    elif mode == 'emul':
        emul(int(sys.argv[2]))
    elif mode == 'fidelity':
        fidelity([int(v) for v in sys.argv[2].split(',')])
    elif mode == 'bench':
        bench(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 192)
