"""Layer-local attention fidelity at compressed layers, official emulation from CUDA's own inputs (CPU).

usage: og_l2.py RUN [layers]   RUN = /dev/shm/split-nv/og/RUN.pt traced with rows=136 (+ core/idx hooks)
Compressed KV comes from the matching raw state (RUN.raw) if present, else from e0-base.raw.
"""
import sys
import torch
from safetensors.torch import load_file
sys.path.insert(0, '/home/ian/split-nv/tools')
sys.path.insert(0, '/home/ian/split-nv/hooks')
import og_ref as R
from split_nv.macpack import unpack_activation

run = sys.argv[1]
layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '0,2,3,8,14').split(',')]
D = '/dev/shm/split-nv/og'
tr = torch.load(f'{D}/{run}.pt', map_location='cpu', weights_only=True)
pos_all = tr['0'][0]
c = {k: v[1] for k, v in tr.items()}
import os
raw = load_file(f"{D}/{os.environ.get('RAW', 'e0-base')}.raw.safetensors")
mac = {k: v[0] for k, v in load_file(f'{D}/og-trace-8k.trace.safetensors').items()}
T = len(pos_all)
Q0 = T - 8          # first of the 8 query rows compared (8205..8212)
SRC = {i: max(s for s in (2, 8, 14, 20) if s <= i) for i in range(2, 21)}


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return round(((a - b).norm() / b.norm()).item(), 5)


def sglang_store(kv, fp4=False):
    """FlashMLA cache: nope 448 re-quantized FP8 per 64-tile (pow2), RoPE 64 BF16."""
    nope = R.act_quant(kv[..., :448], g=64, floor=1e-8)
    return torch.cat((nope, kv[..., 448:].float()), -1).to(torch.bfloat16)


for L in layers:
    p = f'layers.{L}.attn.'
    ratio = R.CFG['compress_ratios'][L]
    q = c[f'{L}.self_attn.wq_b'].view(T, 32, 512)
    q = R.rope(q, pos_all, bool(ratio))
    kv0 = c[f'{L}.self_attn.wqkv_a'][:, 1280:]
    kv = R.rope(R.rms(kv0, R.W(p + 'kv_norm.weight')), pos_all, bool(ratio))
    kv_off, kv_sg = R.act_quant(kv).to(torch.bfloat16), sglang_store(kv)
    sink = R.W(p + 'attn_sink').float()[:32]
    if ratio:
        s = SRC[L]
        ck = unpack_activation(raw[f'ckv.{s}'], 4, 16, True).to(torch.bfloat16)
        ck_sg = sglang_store(ck)
        sel = c[f'{s}.attn.idx'] if f'{s}.attn.idx' in c else None
    outs = {'official': [], 'sglang-store': []}
    for r in range(Q0, T):
        win = slice(r - 127, r + 1)
        for name, kvw, ckk in (('official', kv_off, ck if ratio else None), ('sglang-store', kv_sg, ck_sg if ratio else None)):
            keys = kvw[win]
            if ratio:
                j = sel[r]
                keys = torch.cat((keys, ckk[j[j >= 0].long()]))
            outs[name].append(R.sparse_attn(q[r], keys, sink, 512 ** -0.5))
    o = {k: torch.stack(v) for k, v in outs.items()}
    cc, mc = c[f'{L}.attn.core'][Q0:], mac[f'{L}.attn.core'][:, :32]
    print(f'L{L} r{ratio}: core cuda vs official {rel(cc, o["official"])} | cuda vs sglang-store emu {rel(cc, o["sglang-store"])}'
          f' | sglang-store emu vs official {rel(o["sglang-store"], o["official"])} | (cuda vs mac {rel(cc, mc)})')
    if ratio and L == SRC[L] and f'{L}.attn.idx' in mac:
        a, b = c[f'{L}.attn.idx'][Q0:], mac[f'{L}.attn.idx']
        ov = [len(set(a[i][a[i] >= 0].tolist()) & set(b[i][b[i] >= 0].tolist())) / max(1, int((b[i] >= 0).sum())) for i in range(8)]
        print(f'   idx overlap cuda vs mac per row: {[round(x, 3) for x in ov]}')
