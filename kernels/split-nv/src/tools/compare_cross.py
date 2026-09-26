"""CPU-only original Mac/CUDA boundary and selected-layer trace comparison."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    assert a.shape == b.shape, (a.shape, b.shape)
    return dict(cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                rel_rms=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                max_abs=(a-b).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw')
    ap.add_argument('--native', default='/mnt/nvme-2/native-encoder-8k.safetensors')
    ap.add_argument('--cuda-trace', default='/dev/shm/split-nv/cuda-trace-8213.pt')
    ap.add_argument('--mac-trace')
    args = ap.parse_args()
    torch.set_num_threads(4)
    mac = load_file(args.native)
    c = torch.load(args.cuda_trace, map_location='cpu', weights_only=True)
    out = {'trace_h19': metrics(c['20'][1], mac['tail.hidden'][:, -len(c['20'][0]):]),
           'trace_positions': c['20'][0].tolist()}
    if args.raw:
        raw = load_file(args.raw)
        out['boundary'] = {k: metrics(raw[a], mac[b]) for k,a,b in
                          [('h19','tail_hidden','tail.hidden'), ('pre','tail_pre','tail.pre')]}
    if args.mac_trace:
        m = load_file(args.mac_trace)
        out['layers'] = {k: metrics(c[k][1], v) for k,v in m.items() if k in c}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
