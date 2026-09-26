import os, sys, torch
os.environ['OG_MOE_TRACE'] = '1'
sys.path.insert(0, '/work/tools/moe_dev'); sys.path.insert(0, '/work/hooks')
import test_og as T
from split_nv import og_moe
torch.manual_seed(0)
ne = 192
experts = torch.randperm(384)[:ne].tolist()
wt, _, lw = T.load_layer(3, experts, 0)
x8, _ = T.trace_rows(3)
grid = og_moe._grid(torch.device('cuda'))
nw = grid * 4
buf = torch.zeros(nw * 130, dtype=torch.int64, device='cuda')
og_moe.ext().set_trace(buf)
for M, nd in ((1, 6), (5, 19), (8, 28)):
    pool = torch.randperm(ne)[:nd]
    ids = torch.stack([pool[[(m * 6 + j) % nd for j in range(6)]] for m in range(M)]).to(torch.int32).cuda()
    x = x8[:M].contiguous(); wv = torch.rand(M, 6, device='cuda')
    # warm with other experts, then trace a cold call
    for r in range(4):
        p2 = torch.randperm(ne)[:nd]
        og_moe.moe(x, torch.stack([p2[[(m * 6 + j) % nd for j in range(6)]] for m in range(M)]).to(torch.int32).cuda(), wv, lw)
    buf.zero_(); torch.cuda.synchronize()
    og_moe.moe(x, ids, wv, lw); torch.cuda.synchronize()
    b = buf.view(nw, 130).cpu()
    tk0 = b[:, 0]; tw = b[:, 1]
    base = tk0.min().item()
    n_gu = (1 + nd) * 144
    gu, dn = [], []
    for w in range(nw):
        for i in range(64):
            v = b[w, 2 + 2 * i].item()
            if v == 0 and b[w, 3 + 2 * i].item() == 0: break
            item = v >> 40; t0 = (v & ((1 << 40) - 1)) + tk0[w].item() - base; t1 = b[w, 3 + 2 * i].item() + tk0[w].item() - base
            (gu if item < n_gu else dn).append((item, t0, t1))
    gu_t = torch.tensor([(a, s, e) for a, s, e in gu], dtype=torch.float64); dn_t = torch.tensor([(a, s, e) for a, s, e in dn], dtype=torch.float64)
    end = max(gu_t[:, 2].max(), dn_t[:, 2].max()).item()
    print(f'M={M} nd={nd}: warps start spread {(tk0.max()-base)/1e3:.1f} us, pdl_wait done {(tw.min()-base)/1e3:.1f}..{(tw.max()-base)/1e3:.1f} us, '
          f'kernel span {end/1e3:.1f} us')
    sh = gu_t[gu_t[:, 0] < 144]; rt = gu_t[gu_t[:, 0] >= 144]
    print(f'   gate/up: shared items {len(sh)} dur med {(sh[:,2]-sh[:,1]).median()/1e3:.1f} us, last end {sh[:,2].max()/1e3:.1f}; '
          f'routed {len(rt)} dur med {(rt[:,2]-rt[:,1]).median()/1e3:.1f} us, first start {rt[:,1].min()/1e3:.1f}, last end {rt[:,2].max()/1e3:.1f}')
    shd = dn_t[dn_t[:, 0] >= n_gu + (nd) * 320]; rtd = dn_t[dn_t[:, 0] < n_gu + nd * 320]
    print(f'   down: routed {len(rtd)} dur med {(rtd[:,2]-rtd[:,1]).median()/1e3:.1f} us, first start {rtd[:,1].min()/1e3:.1f}, last end {rtd[:,2].max()/1e3:.1f}; '
          f'shared {len(shd)} first start {shd[:,1].min()/1e3:.1f} last end {shd[:,2].max()/1e3:.1f}')
    # active warps over time
    for tt in range(0, int(end) + 5000, 10000):
        a_gu = ((gu_t[:, 1] <= tt) & (gu_t[:, 2] > tt)).sum().item(); a_dn = ((dn_t[:, 1] <= tt) & (dn_t[:, 2] > tt)).sum().item()
        print(f'     t={tt/1e3:6.1f} us: active gate/up {a_gu:4d}, down {a_dn:4d}')
