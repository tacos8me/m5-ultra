"""Task A: og_moe prefill variants (moe_variants.py) on real layer weights (all 384 experts, one GPU, ~4.6 GB):
bitwise vs base over several row counts and routings, and ms per call at 8192 rows (event timing, shuffled order).
usage: moe_fused.py LAYER [variants...]"""
import random
import statistics as st
import sys

import torch

sys.path.insert(0, "/work/tools/moe_dev")
sys.path.insert(0, "/work/tools/prefill_perf")
import moe_variants as V  # noqa: E402
import weights as W  # noqa: E402

dev = "cuda"
H, I = W.H, W.I


def main(layer, names):
    torch.manual_seed(1)
    random.seed(1)
    wt = W.load_experts(layer, list(range(384)), 0)
    sh = W.shared_host(layer, 0)
    args = (wt["w13"], wt["w13_sf"], wt["w2"], wt["w2_sf"], torch.cat([sh["w1"], sh["w3"]]).to(dev),
            torch.cat([sh["s1"], sh["s3"]]).contiguous().to(dev), sh["w2"].to(dev), sh["s2"].to(dev), I, 0)
    del wt
    mods = {n: V.build(n) for n in ["base"] + names}
    ok = True
    for M, kind in ((8192, "gate"), (4096, "gate"), (300, "gate"), (8192, "uniform"), (129, "gate")):
        x = (torch.randn(M, H, device=dev) * 0.35).to(torch.bfloat16)
        if kind == "gate":
            parts = [W.route(x[a:a + 1024], layer) for a in range(0, M, 1024)]
            ids = torch.cat([q[0] for q in parts]).to(torch.int32).contiguous()
            w = torch.cat([q[1] for q in parts]).float().contiguous()
        else:
            ids = torch.stack([torch.randperm(384, device=dev)[:6] for _ in range(M)]).to(torch.int32).contiguous()
            w = torch.rand(M, 6, device=dev).contiguous()
        if M == 129:
            ids[-3:] = -1  # masked rows (SGLang padding)
        ws = torch.empty(max(m.prefill_workspace_bytes(M) for m in mods.values()), dtype=torch.uint8, device=dev)
        ref = mods["base"].prefill(x, ids, w, ws, *args)
        for n in names:
            a = mods[n].prefill(x, ids, w, ws, *args)
            e = torch.equal(a, ref)
            del a
            b = mods[n].prefill(x, ids, w, ws, *args)  # workspace reuse (counters re-zeroed)
            e = e and torch.equal(b, ref)
            del b
            ok &= e
            print(f"M={M:5d} {kind:7s} {n}: bitwise == base {e}", flush=True)
        del ref, ws
    M = 8192
    x = (torch.randn(M, H, device=dev) * 0.35).to(torch.bfloat16)
    parts = [W.route(x[a:a + 1024], layer) for a in range(0, M, 1024)]
    ids = torch.cat([q[0] for q in parts]).to(torch.int32).contiguous()
    w = torch.cat([q[1] for q in parts]).float().contiguous()
    one = torch.empty(max(m.prefill_workspace_bytes(M) for m in mods.values()), dtype=torch.uint8, device=dev)
    ws = {n: one for n in mods}
    ts = {n: [] for n in mods}
    for r in range(24):
        order = list(mods)
        random.shuffle(order)
        for n in order:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            mods[n].prefill(x, ids, w, ws[n], *args)
            e1.record()
            if r >= 4:
                ts[n].append((e0, e1))
    torch.cuda.synchronize()
    b = st.median(a.elapsed_time(c) for a, c in ts["base"])
    for n in mods:
        t = [a.elapsed_time(c) for a, c in ts[n]]
        m = st.median(t)
        print(f"layer {layer} M=8192 {n:9s}: median {m:6.3f} ms  min {min(t):6.3f}  vs base {m - b:+.3f} ms/layer "
              f"({20 * (m - b):+.1f} ms per 8K chunk)", flush=True)
    print(f"check {'PASS' if ok else 'FAIL'}; max reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:] or ["lastw", "lastw_col"])
