"""Layer-0 attention attribution: official emulation vs CUDA vs Mac traces (CPU)."""
import json
import sys
import torch
from safetensors.torch import load_file
sys.path.insert(0, '/home/ian/split-nv/tools')
import og_ref as R

cuda_path = sys.argv[1] if len(sys.argv) > 1 else '/dev/shm/split-nv/og/e0-base.pt'
c = {k: v[1] for k, v in torch.load(cuda_path, map_location='cpu', weights_only=True).items()}
mac = {k: v[0] for k, v in load_file('/mnt/nvme-2/og-box/og-trace-8k.trace.safetensors').items()}
ids = torch.tensor(json.load(open('/home/ian/split-nv/ref/ids-8192.json'))[:8213])
P0, P1 = 8077, 8213
pos = torch.arange(P0, P1)
rows = slice(8205 - P0, None)


def m(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return round(((a - b).norm() / b.norm()).item(), 5)


emb = R.W('embed.weight')[ids[P0:P1]]
x = R.rms(emb, R.W('layers.0.attn_norm.weight'))
print('x vs cuda/mac input_layernorm', m(x[rows], c['0.input_layernorm']), m(x[rows], mac['0.input_layernorm']))
L = 'layers.0.attn.'
ql = R.linear(x, L + 'wq_a')
kv0 = R.linear(x, L + 'wkv')
print('wq_a ref vs cuda/mac', m(ql[rows], c['0.self_attn.wqkv_a'][:, :1280]), m(ql[rows], mac['0.attn.wq_a']))
print('wkv  ref vs cuda/mac', m(kv0[rows], c['0.self_attn.wqkv_a'][:, 1280:]), m(kv0[rows], mac['0.attn.wkv']))
qr = R.rms(ql, R.W(L + 'q_norm.weight'))
q = R.linear(qr, L + 'wq_b').view(-1, 64, 512)
print('wq_b ref vs cuda (heads 0-31, pre-rope)', m(q[rows, :32].flatten(1), c['0.self_attn.wq_b']))
q = R.rope(q, pos, False)
kv = R.rope(R.rms(kv0, R.W(L + 'kv_norm.weight')), pos, False)
sink = R.W(L + 'attn_sink').float()
scale = 512 ** -0.5


def sglang_kv(kv):
    """SGLang SWA cache: nope 448 as FP8 with pow2 scale per 64-tile, rope 64 kept BF16."""
    nope = R.act_quant(kv[:, :448], g=64, floor=1e-8)
    return torch.cat((nope, kv[:, 448:].float()), 1).to(torch.bfloat16)


def attend(qv, kvv, pdtype=None):
    out = []
    for r in range(8205 - P0, P1 - P0):
        win = kvv[r - 127:r + 1]
        out.append(R.sparse_attn(qv[r], win, sink, scale))
    return torch.stack(out)


kv_off = R.act_quant(kv).to(torch.bfloat16)
core = attend(q, kv_off)
print('core official vs cuda/mac', m(core[:, :32], c['0.attn.core']), m(core[:, :32], mac['0.attn.core'][:, :32]))
core_sg = attend(q, sglang_kv(kv))
print('core sglang-kv vs cuda/mac', m(core_sg[:, :32], c['0.attn.core']), m(core_sg[:, :32], mac['0.attn.core'][:, :32]))
core_bf = attend(q, kv)
print('core unquantized-kv vs cuda/mac', m(core_bf[:, :32], c['0.attn.core']), m(core_bf[:, :32], mac['0.attn.core'][:, :32]))
qf8 = R.act_quant(q.flatten(1), g=64).view_as(q).to(torch.bfloat16)
core_q8 = attend(qf8, sglang_kv(kv))
print('core sglang-kv + fp8 q(g64) vs cuda/mac', m(core_q8[:, :32], c['0.attn.core']), m(core_q8[:, :32], mac['0.attn.core'][:, :32]))
print('cuda vs mac core', m(c['0.attn.core'], mac['0.attn.core'][:, :32]))
o = R.rope(core, pos[rows], False, inverse=True)
wo_a = R.fp8_weight(L + 'wo_a').to(torch.bfloat16).view(8, 1024, 4096)
oa = torch.einsum('sgd,grd->sgr', o.view(-1, 8, 4096).double(), wo_a.double()).to(torch.bfloat16)
out = R.linear(oa.flatten(1), L + 'wo_b')
print('self_attn official vs cuda/mac', m(out, c['0.self_attn']), m(out, mac['0.self_attn']))
