"""Standalone check of split_nv.og_attn against a float64 emulation of the official sparse_attn (GPU)."""
import sys
import torch
sys.path.insert(0, '/home/ian/split-nv/hooks')
from split_nv.og_attn import sparse_mla

torch.manual_seed(0)
dev = 'cuda'


def make_cache(pages, ps):
    pb = ps * 584 + 64  # padded page stride
    raw = torch.zeros(pages, pb, dtype=torch.uint8, device=dev)
    vals = raw[:, :ps * 576].view(pages, ps, 576)
    nope = (torch.randn(pages, ps, 448, device=dev) * 60).clamp(-448, 448).to(torch.float8_e4m3fn)
    vals[:, :, :448] = nope.view(torch.uint8)
    rope = torch.randn(pages, ps, 64, device=dev).to(torch.bfloat16)
    vals[:, :, 448:] = rope.view(torch.uint8).view(pages, ps, 128)
    sc = torch.randint(118, 124, (pages, ps, 7), device=dev, dtype=torch.uint8)
    raw[:, ps * 576:ps * 584].view(pages, ps, 8)[:, :, :7] = sc
    kv = torch.cat((nope.float() * torch.exp2(sc.float() - 127).repeat_interleave(64, -1), rope.float()), -1)
    view = raw.as_strided((pages, ps, 1, 584), (pb, 584, 584, 1))
    return view, kv.view(pages * ps, 512).to(torch.bfloat16)


def ref(q, kvw, kve, sink, scale):
    out = []
    for t in range(q.shape[0]):
        keys = torch.cat([k for k in (kvw[t], kve[t]) if k is not None]).double()
        s = (q[t].double() @ keys.T).float() * scale
        m = s.amax(-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(s - m)
        den = p.sum(-1, keepdim=True) + torch.exp(sink[:, None] - m)
        out.append(((p.to(torch.bfloat16).double() @ keys) / den.double()).to(torch.bfloat16))
    return torch.stack(out)


for T, H, ps in ((5, 64, 128), (37, 32, 256), (200, 32, 64)):
    swa, swa_kv = make_cache(6, ps)
    ext, ext_kv = make_cache(12, ps)
    nw, ne = 128, 512
    widx = torch.randint(0, 6 * ps, (T, nw), device=dev, dtype=torch.int32)
    widx[:, :7] = -1
    wlen = torch.randint(100, nw + 1, (T,), device=dev, dtype=torch.int32)
    eidx = torch.randint(0, 12 * ps, (T, ne), device=dev, dtype=torch.int32)
    elen = torch.randint(0, ne + 1, (T,), device=dev, dtype=torch.int32)
    eidx[:, 400:] = -1
    q = (torch.randn(T, 1, H, 512, device=dev) * 0.05).to(torch.bfloat16)
    sink = torch.randn(H, device=dev)
    scale = 512 ** -0.5
    out = sparse_mla(q, swa, widx.unsqueeze(1), wlen, sink, scale, ext, eidx.unsqueeze(1), elen)[:, 0]
    kw = [swa_kv[widx[t][(widx[t] >= 0) & (torch.arange(nw, device=dev) < wlen[t])].long()] for t in range(T)]
    ke = [ext_kv[eidx[t][(eidx[t] >= 0) & (torch.arange(ne, device=dev) < elen[t])].long()] for t in range(T)]
    r = ref(q[:, 0], kw, ke, sink, scale)
    rel = ((out.double() - r.double()).norm() / r.double().norm()).item()
    print(f'T={T} H={H} ps={ps}: relRMS vs float64 official emulation {rel:.2e}, max|d| {(out.float() - r.float()).abs().max().item():.3e}')
    out2 = sparse_mla(q[:3], swa, widx[:3].unsqueeze(1), wlen[:3], sink, scale, ext, eidx[:3].unsqueeze(1), elen[:3])[:, 0]
    print('  batch-invariant (first 3 rows alone == in batch):', torch.equal(out2, out[:3]))
    no_ext = sparse_mla(q, swa, widx.unsqueeze(1), wlen, sink, scale)[:, 0]
    r2 = ref(q[:, 0], kw, [None] * T, sink, scale)
    print(f'  window only: relRMS {((no_ext.double() - r2.double()).norm() / r2.double().norm()).item():.2e}')
