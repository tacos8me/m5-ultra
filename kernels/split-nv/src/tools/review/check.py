"""Reachability checks for the og_moe review (small GPU footprint).

1. #3: can the production router (sglang Triton moe_fused_gate, sqrtsoftplus, ungrouped, 384 experts, top-6; also the
   vision bias_alt path and num_token_non_padded padding) emit a duplicate expert id in a live row? Adversarial inputs.
2. #4: decode with a masked middle row / valid < M: are output rows >= m written (zeros) even when the allocator hands
   back a NaN-filled block?
"""
import sys

import torch

sys.path.insert(0, '/work/tools/moe_dev')
sys.path.insert(0, '/work/tools/review')
import variants as V  # noqa: E402
import weights as W  # noqa: E402

dev = 'cuda'
torch.manual_seed(0)


def router_check():
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
    _, gb = W.gate(3)
    bias = gb.to(torch.bfloat16).to(dev)
    bias_vl = W.raw('layers.3.ffn.gate.bias_vl').to(dev)
    N, M = 384, 4096
    cases = {
        'randn': torch.randn(M, N, device=dev),
        'randn x100': torch.randn(M, N, device=dev) * 100,
        'all zero (full ties)': torch.zeros(M, N, device=dev),
        'constant 5': torch.full((M, N), 5.0, device=dev),
        'quantized ties': (torch.randn(M, N, device=dev) * 2).round(),
        'some NaN': torch.where(torch.rand(M, N, device=dev) < 0.02, float('nan'), torch.randn(M, N, device=dev)),
        'all NaN': torch.full((M, N), float('nan'), device=dev),
        '+inf spots': torch.where(torch.rand(M, N, device=dev) < 0.01, float('inf'), torch.randn(M, N, device=dev)),
        '-inf spots': torch.where(torch.rand(M, N, device=dev) < 0.5, float('-inf'), torch.randn(M, N, device=dev)),
        'all -inf': torch.full((M, N), float('-inf'), device=dev),
        'huge 1e30': torch.randn(M, N, device=dev) * 1e30,
    }
    ok = True
    ids_img = torch.randint(0, 130000, (M,), device=dev, dtype=torch.int64)
    ids_img[::3] = 129264  # image_token_id
    pad = torch.tensor([M - 100], dtype=torch.int32, device=dev)
    for name, s in cases.items():
        for variant, kw in (('text', {}),
                            ('vision+pad', dict(bias_alt=bias_vl, input_ids=ids_img, bias_alt_token_id=129264,
                                                num_token_non_padded=pad, renormalize_epsilon=1e-20))):
            w, ids = moe_fused_gate(s.contiguous(), bias, topk=6, scoring_func='sqrtsoftplus', renormalize=True,
                                    routed_scaling_factor=1.5, **kw)
            live = ids[:M - 100] if variant != 'text' else ids
            srt = live.sort(-1).values
            dup = int((srt[:, 1:] == srt[:, :-1]).any(-1).sum())
            oob = int(((live < 0) | (live >= N)).any(-1).sum())
            padrows = ids[M - 100:] if variant != 'text' else None
            padinfo = '' if padrows is None else f', pad rows all -1: {bool((padrows == -1).all())}'
            ok &= dup == 0 and oob == 0
            print(f'  router {name:22s} {variant:10s}: rows with duplicate ids {dup}, out-of-range {oob}{padinfo}',
                  flush=True)
    print('ROUTER', 'no duplicates' if ok else 'DUPLICATES FOUND', flush=True)


def mask_check():
    experts = list(range(16))
    wt = W.load_experts(3, experts, 0)
    sh = W.shared_host(3, 0)
    s13 = torch.cat([sh['w1'], sh['w3']]).to(dev)
    s13_sf = torch.cat([sh['s1'], sh['s3']]).contiguous().to(dev)
    args = (wt['w13'], wt['w13_sf'], wt['w2'], wt['w2_sf'], s13, s13_sf, sh['w2'].to(dev), sh['s2'].to(dev), W.I, 0)
    m = V.build('base')
    ws = torch.empty(m.decode_workspace_bytes() + 256, dtype=torch.uint8, device=dev)
    M = 5
    x = (torch.randn(M, W.H, device=dev) * 0.35).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(16, device=dev)[:6] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, 6, device=dev)
    big = torch.tensor([1 << 30], dtype=torch.int32, device=dev)
    ref = m.decode(x, ids, w, big, ws, *args, 188)
    ok = True
    for tag, ids2, valid, m_exp in (('row 2 masked (middle)', ids.clone(), big, 2), ('valid=3', ids, torch.tensor([3], dtype=torch.int32, device=dev), 3),
                                    ('all rows masked', torch.full_like(ids, -1), big, 0)):
        if tag.startswith('row 2'):
            ids2[2] = -1
        for _ in range(3):  # poison the allocator block the output will reuse
            p = torch.full((M, W.H), float('nan'), dtype=torch.bfloat16, device=dev)
            del p
        out = m.decode(x, ids2, w, valid, ws, *args, 188)
        torch.cuda.synchronize()
        zeros = bool((out[m_exp:] == 0).all())
        same = torch.equal(out[:m_exp], ref[:m_exp])
        nan = bool(out.isnan().any())
        ok &= zeros and same and not nan
        print(f'  decode {tag:22s}: rows < {m_exp} == unmasked ref {same}; rows >= {m_exp} all zero {zeros}; any NaN {nan}',
              flush=True)
    print('MASK', 'rows >= m are written (zero)' if ok else 'FAIL', flush=True)


if __name__ == '__main__':
    router_check()
    mask_check()
    print(f'max reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB')
