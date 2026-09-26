"""Compare two runs (CUDA trace .pt + raw state, or Mac trace/state) layer by layer.

usage: og_diff.py A B   where each is a run name under /dev/shm/split-nv/og (CUDA: NAME.pt + NAME.raw.safetensors)
or 'mac' (og-trace-8k.trace.safetensors + og-trace-8k.safetensors).
Prints block-input h relRMS for the 8 trace rows, SWA relRMS per layer (last 128 tokens), and the
layer-20 / source-layer compressed KV + index K relRMS over all rows, plus h19/L20-KV cosines.
"""
import sys
import torch
from safetensors.torch import load_file
sys.path.insert(0, '/home/ian/split-nv/hooks')
from split_nv.macpack import unpack_activation

D = '/dev/shm/split-nv/og'


def load(name):
    if name.startswith('mac'):
        stem = 'og-trace-8k' + name[3:]
        tr = {k: v[0] for k, v in load_file(f'{D}/{stem}.trace.safetensors').items()}
        st = load_file(f'{D}/{stem}.safetensors')
        swa = lambda i: st[f'layer.{i}.slot.1'][0]
        ckv = lambda i, s: st[f'layer.{i}.slot.{s}'][0]
        tail = st['tail.hidden'][0]
    else:
        tr = {k: v[1] for k, v in torch.load(f'{D}/{name}.pt', map_location='cpu', weights_only=True).items()}
        st = load_file(f'{D}/{name}.raw.safetensors')
        swa = lambda i: st[f'swa.{i}']
        ckv = lambda i, s: st[('ckv' if s == 2 else 'idxk') + f'.{i}']
        tail = st['tail_hidden']
    return tr, swa, ckv, tail


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a - b).norm().item() / b.norm().item()


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main(a, b):
    ta, sa, ca, ha = load(a)
    tb, sb, cb, hb = load(b)
    print(f'{a} vs {b}')
    rows = []
    for i in range(21):
        u = lambda f: unpack_activation(f, 8, 32, False).float()
        s = f'L{i:2d} h_in {rel(ta[str(i)], tb[str(i)]):.4f} swa {rel(u(sa(i)), u(sb(i))):.4f}'
        if i in (2, 8, 14, 20):
            k1 = lambda c: unpack_activation(c(i, 2), 4, 16, True).float()
            k2 = lambda c: unpack_activation(c(i, 3), 4, 32, False).float()
            s += f' ckv {rel(k1(ca), k1(cb)):.4f} idxk {rel(k2(ca), k2(cb)):.4f}'
        rows.append(s)
    print('\n'.join(rows))
    k1 = lambda c: unpack_activation(c(20, 2), 4, 16, True).float()
    print(f'GATE h19(tail 256) cos {cos(ha, hb):.5f} rel {rel(ha, hb):.4f} | L20 KV (all rows) cos {cos(k1(ca), k1(cb)):.5f} rel {rel(k1(ca), k1(cb)):.4f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
