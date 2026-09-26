"""Per-layer CUDA-trace vs Mac-trace comparison table (CPU)."""
import argparse
import torch
from safetensors.torch import load_file


def m(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            ((a - b).norm() / b.norm().clamp_min(1e-30)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', default='/mnt/nvme-2/pipe1-cuda-trace-8213.pt')
    ap.add_argument('--mac', default='/mnt/nvme-2/og-box/og-trace-8k.trace.safetensors')
    ap.add_argument('--layers', default='0-20')
    ap.add_argument('--raw', help='CUDA raw state (split-nv-raw-v1) to compare SWA/ckv/idxk rows')
    ap.add_argument('--mac-state', default='/mnt/nvme-2/og-box/og-trace-8k.safetensors')
    args = ap.parse_args()
    c = {k: v[1] for k, v in torch.load(args.cuda, map_location='cpu', weights_only=True).items()}
    mac = {k: v[0] for k, v in load_file(args.mac).items()}
    for i in range(21):
        if f'{i}.attn.core' in mac and f'{i}.attn.core' in c:
            mac[f'{i}.attn.core'] = mac[f'{i}.attn.core'][:, :32]
        if f'{i}.self_attn.wqkv_a' in c:
            c[f'{i}.attn.wq_a'] = c[f'{i}.self_attn.wqkv_a'][:, :1280]
            c[f'{i}.attn.wkv'] = c[f'{i}.self_attn.wqkv_a'][:, 1280:]
    cols = ['', 'attn.wq_a', 'attn.wkv', 'input_layernorm', 'attn.core', 'self_attn', 'post_attention_layernorm', 'mlp.gate',
            'mlp.shared_experts', 'mlp']
    lo, hi = map(int, args.layers.split('-'))
    print('relRMS per layer (cos of block input in last col)')
    print(f"{'L':>2} " + ' '.join(f'{(x or "h_in")[:12]:>12}' for x in cols) + f" {'cos(h_in)':>10}")
    for i in range(lo, hi + 1):
        row = []
        for col in cols:
            k = f'{i}.{col}' if col else str(i)
            if k in c and k in mac and c[k].numel() == mac[k].numel():
                row.append(f'{m(c[k], mac[k])[1]:12.5f}')
            else:
                row.append(f"{'-':>12}")
        print(f'{i:>2} ' + ' '.join(row) + f' {m(c[str(i)], mac[str(i)])[0]:10.6f}')
    if args.raw:
        import sys
        sys.path.insert(0, '/home/ian/split-nv/hooks')
        from split_nv.macpack import unpack_activation
        raw, st = load_file(args.raw), load_file(args.mac_state)
        print('state rows relRMS: swa (last 128 tokens) per layer; ckv/idxk for source layers')
        out = []
        for i in range(lo, hi + 1):
            a = unpack_activation(raw[f'swa.{i}'], 8, 32, False).float()
            b = unpack_activation(st[f'layer.{i}.slot.1'][0], 8, 32, False).float()
            s_ = f'L{i}: swa {m(a, b)[1]:.4f}'
            if f'ckv.{i}' in raw:
                for key, slot, fmt in (('ckv', 2, (4, 16, True)), ('idxk', 3, (4, 32, False))):
                    a = unpack_activation(raw[f'{key}.{i}'], *fmt).float()
                    b = unpack_activation(st[f'layer.{i}.slot.{slot}'][0], *fmt).float()
                    s_ += f' {key} {m(a, b)[1]:.4f}'
            out.append(s_)
        print('\n'.join(out))


if __name__ == '__main__':
    main()
