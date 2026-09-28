"""og_moe prefill cost of one 8K chunk vs the same rows in 2 x 4K / 4 x 2K calls (real layer weights, one GPU, ~4.5 GB).

Rows are bitwise identical by construction (og-s4.4: every row's arithmetic is independent of M); checked here too.
usage: moe_split.py LAYER [reps]
"""
import statistics as st
import sys

import torch

sys.path.insert(0, '/work/tools/moe_dev')
sys.path.insert(0, '/work/tools/review')
sys.path.insert(0, '/work/hooks')
import weights as W  # noqa: E402
from split_nv.og_moe import LayerWeights, prefill  # noqa: E402

dev = 'cuda'
H, I = W.H, W.I


def main(layer, reps):
    torch.manual_seed(1)
    wt = W.load_experts(layer, list(range(384)), 0)
    sh = W.shared_host(layer, 0)
    s13 = torch.cat([sh['w1'], sh['w3']]).to(dev)
    s13_sf = torch.cat([sh['s1'], sh['s3']]).contiguous().to(dev)
    lw = LayerWeights(wt['w13'], wt['w13_sf'], wt['w2'], wt['w2_sf'], s13, s13_sf, sh['w2'].to(dev), sh['s2'].to(dev), I, 0)
    del wt
    M = 8192
    x = (torch.randn(M, H, device=dev) * 0.35).to(torch.bfloat16)
    parts = [W.route(x[a:a + 1024], layer) for a in range(0, M, 1024)]
    ids = torch.cat([p[0] for p in parts]).to(torch.int32).contiguous()
    w = torch.cat([p[1] for p in parts]).float().contiguous()
    ref = prefill(x, ids, w, lw)
    splits = {'1x8192': [8192], '2x4096': [4096, 4096], '4x2048': [2048] * 4, '8192=6144+2048': [6144, 2048]}
    for name, sizes in splits.items():
        outs, a = [], 0
        for n in sizes:
            outs.append(prefill(x[a:a + n].contiguous(), ids[a:a + n].contiguous(), w[a:a + n].contiguous(), lw))
            a += n
        print(f'{name}: bitwise == 1x8192: {torch.equal(torch.cat(outs), ref)}', flush=True)
    torch.cuda.synchronize()
    ts = {k: [] for k in splits}
    for r in range(reps):
        for name, sizes in splits.items():
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            a = 0
            for n in sizes:
                prefill(x[a:a + n], ids[a:a + n], w[a:a + n], lw)
                a += n
            e1.record()
            ts[name].append((e0, e1))
    torch.cuda.synchronize()
    base = st.median(e0.elapsed_time(e1) for e0, e1 in ts['1x8192'][2:])
    for name in splits:
        t = [e0.elapsed_time(e1) for e0, e1 in ts[name][2:]]
        m = st.median(t)
        print(f'layer {layer} {name:16s}: median {m:6.3f} ms  min {min(t):6.3f}  vs 1x8192 {m - base:+.3f} ms '
              f'(x20 layers {20 * (m - base):+.1f} ms/chunk)', flush=True)
    print(f'max reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 12)
