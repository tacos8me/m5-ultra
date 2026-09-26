"""Layer-k MoE attribution: official emulation from each side's own FFN input (CPU)."""
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0, '/home/ian/split-nv/tools')
import og_ref as R

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cuda_path = sys.argv[2] if len(sys.argv) > 2 else '/dev/shm/split-nv/og/e0-base.pt'
c = {k: v[1] for k, v in torch.load(cuda_path, map_location='cpu', weights_only=True).items()}
mac = {k: v[0] for k, v in load_file('/mnt/nvme-2/og-box/og-trace-8k.trace.safetensors').items()}
L = f'layers.{layer}.ffn.'


def m(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return round(((a - b).norm() / b.norm()).item(), 5)


def expert(x, pre, w=None, fp4=False):
    wt = R.fp4_weight if fp4 else R.fp8_weight
    xq = R.act_quant(x).double()
    g = (xq @ wt(pre + 'w1').double().T).to(torch.bfloat16).float()
    u = (xq @ wt(pre + 'w3').double().T).to(torch.bfloat16).float()
    u = u.clamp(-10, 10)
    g = g.clamp(max=10)
    y = F.silu(g) * u
    if w is not None:
        y = w * y
    return (R.act_quant(y.to(torch.bfloat16)).double() @ wt(pre + 'w2').double().T).to(torch.bfloat16)


def moe(x, weight_first=True):
    gw = R.W(L + 'gate.weight').float()
    logits = x.float() @ gw.T
    scores = F.softplus(logits).sqrt()
    idx = (scores + R.W(L + 'gate.bias').float()).topk(6, -1)[1]
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * 1.5
    y = torch.zeros(x.shape, dtype=torch.float32)
    for r in range(x.shape[0]):
        for j in range(6):
            e = int(idx[r, j])
            if weight_first:
                y[r] += expert(x[r:r + 1], f'{L}experts.{e}.', wts[r, j], fp4=True)[0].float()
            else:
                y[r] += wts[r, j] * expert(x[r:r + 1], f'{L}experts.{e}.', None, fp4=True)[0].float()
    shared = expert(x, L + 'shared_experts.').float()
    return (y + shared).to(torch.bfloat16), idx, logits


xc, xm = c[f'{layer}.post_attention_layernorm'], mac[f'{layer}.post_attention_layernorm']
oc, idc, lc = moe(xc)
om, idm, lm = moe(xm)
oc2, _, _ = moe(xc, weight_first=False)
print(f'L{layer} ffn-in cuda vs mac', m(xc, xm))
print('route idx equal cuda-ref/mac-ref', (idc.sort(-1)[0] == idm.sort(-1)[0]).all(-1).tolist(),
      'mac-trace vs mac-ref', (mac[f'{layer}.mlp.route_idx'].sort(-1)[0] == idm.sort(-1)[0]).all(-1).tolist())
print('logits ref vs cuda trace', m(lc, c[f'{layer}.mlp.gate']), 'ref vs mac trace', m(lm, mac[f'{layer}.mlp.gate']))
print('MoE: official(cuda in) vs cuda', m(oc, c[f'{layer}.mlp']), '| official(mac in) vs mac', m(om, mac[f'{layer}.mlp']))
print('MoE: weight-after-w2 emulation(cuda in) vs cuda', m(oc2, c[f'{layer}.mlp']), '| vs official', m(oc2, oc))
sh_m = expert(xm, L + 'shared_experts.')
print('shared: official(mac in) vs mac shared trace', m(sh_m, mac[f'{layer}.mlp.shared_experts']))
for r in range(8):
    print(r, 'cuda', m(oc[r], c[f'{layer}.mlp'][r]), 'mac', m(om[r], mac[f'{layer}.mlp'][r]), 'routes', idm[r].tolist(), idc[r].tolist())
