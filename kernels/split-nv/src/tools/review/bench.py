"""A/B of og_moe.cu variants on real layer weights (one GPU, all 384 experts of one layer, ~4.5 GB).

  bench.py decode LAYER VARIANTS...   decode us/call (cold L2) at M = 1, 5, 8; bitwise vs base
  bench.py prefill LAYER VARIANTS...  prefill ms/call at M = 8192 (real gate routing on random rows); bitwise vs base
  bench.py profile LAYER              per-kernel prefill breakdown (base, M = 8192)
"""
import random
import statistics as st
import sys

import torch

sys.path.insert(0, '/work/tools/moe_dev')
sys.path.insert(0, '/work/tools/review')
import variants as V  # noqa: E402
import weights as W  # noqa: E402

dev = 'cuda'
H, I = W.H, W.I


def load_layer(layer):
    wt = W.load_experts(layer, list(range(384)), 0)
    sh = W.shared_host(layer, 0)
    s13 = torch.cat([sh['w1'], sh['w3']]).to(dev)
    s13_sf = torch.cat([sh['s1'], sh['s3']]).contiguous().to(dev)
    return (wt['w13'], wt['w13_sf'], wt['w2'], wt['w2_sf'], s13, s13_sf, sh['w2'].to(dev), sh['s2'].to(dev), I, 0)


def rows(n, layer, std):
    x = (torch.randn(n, H, device=dev) * std).to(torch.bfloat16)
    parts = [W.route(x[a:a + 1024], layer) for a in range(0, n, 1024)]
    ids = torch.cat([p[0] for p in parts])
    w = torch.cat([p[1] for p in parts])
    return x, ids.to(torch.int32).contiguous(), w.float().contiguous()


def decode(layer, names):
    torch.manual_seed(0)
    random.seed(0)
    args = load_layer(layer)
    mods = {n: V.build(n) for n in names}
    grids = {}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for n in names:
        grids[n] = [sms, 2 * sms] if n == 'dnst2' else [sms]
    ws = {n: torch.empty(m.decode_workspace_bytes() + 256, dtype=torch.uint8, device=dev) for n, m in mods.items()}
    valid = torch.full((1,), 1 << 30, dtype=torch.int32, device=dev)
    flush = torch.ones(64 << 20, dtype=torch.int32, device=dev)  # 256 MB > 128 MB L2
    keys = [(n, g) for n in names for g in grids[n]]
    for M in (1, 5, 8):
        R = 120
        x, ids, w = rows(R * M, layer, 0.35)
        nd = [len(set(ids[r * M:(r + 1) * M].flatten().tolist())) for r in range(R)]
        ev = {k: [] for k in keys}
        mism = {k: 0 for k in keys}
        for r in range(-3, R):
            rr = max(r, 0)
            xa, ia, wa = (t[rr * M:(rr + 1) * M].contiguous() for t in (x, ids, w))
            order = keys[:]
            random.shuffle(order)
            outs = {}
            for k in order:
                n, g = k
                flush.sum()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                outs[k] = mods[n].decode(xa, ia, wa, valid, ws[n], *args, g)
                e1.record()
                if r >= 0:
                    ev[k].append((e0, e1))
            base = outs.get(('base', sms))
            if base is not None and r >= 0:
                for k in keys:
                    if k[0] != 'nocomb' and not torch.equal(outs[k], base):
                        mism[k] += 1
        torch.cuda.synchronize()
        b = st.median(e0.elapsed_time(e1) for e0, e1 in ev[('base', sms)]) * 1e3
        print(f'decode M={M} (distinct experts med {st.median(nd)}), {R} cold calls/variant:', flush=True)
        for k in keys:
            t = [e0.elapsed_time(e1) * 1e3 for e0, e1 in ev[k]]
            med = st.median(t)
            print(f'  {k[0]:8s} grid {k[1]:3d}: median {med:7.2f} us  mean {st.mean(t):7.2f}  p10 {sorted(t)[R // 10]:7.2f}'
                  f'  d vs base {med - b:+6.2f} us ({(med - b) / b * 100:+.2f}%)  bitwise mismatches {mism[k]}', flush=True)


def prefill(layer, names, M=8192):
    torch.manual_seed(1)
    random.seed(1)
    args = load_layer(layer)
    mods = {}
    for n in names:
        try:
            mods[n] = V.build(n)
        except Exception as e:  # noqa: BLE001
            print(f'{n}: build failed: {e}', flush=True)
    for kind in ('gate', 'uniform'):
        if kind == 'gate':
            x, ids, w = rows(M, layer, 0.35)
        else:
            x = (torch.randn(M, H, device=dev) * 0.35).to(torch.bfloat16)
            ids = torch.stack([torch.randperm(384, device=dev)[:6] for _ in range(M)]).to(torch.int32).contiguous()
            w = torch.rand(M, 6, device=dev).contiguous()
        cnt = torch.bincount(ids.flatten().long(), minlength=384)
        n_routed = int(((cnt + 127) // 128).sum())
        max_routed = (M * 6 + 127) // 128 + 384
        print(f'prefill M={M} routes={kind}: experts hit {(cnt > 0).sum().item()}, max/expert {cnt.max().item()}, '
              f'routed tiles {n_routed} of grid.y {max_routed} ({max_routed - n_routed} empty rows of CTAs: '
              f'{(max_routed - n_routed) * 18} up + {(max_routed - n_routed) * 40} down CTAs)', flush=True)
        run = {}
        ws = torch.empty(mods['base'].prefill_workspace_bytes(M), dtype=torch.uint8, device=dev)  # same layout in all
        for n, m in list(mods.items()):
            if n == 'pfexact':
                m.set_grid_routed(n_routed)
            try:
                o = m.prefill(x, ids, w, ws, *args)
                torch.cuda.synchronize()
                run[n] = (m, ws, o if n == 'base' else None)
                del o
            except Exception as e:  # noqa: BLE001
                print(f'  {n}: launch failed: {str(e).splitlines()[0]}', flush=True)
                del mods[n]
        base = run['base'][2]
        ts = {n: [] for n in run}
        for r in range(12):
            order = list(run)
            random.shuffle(order)
            for n in order:
                m, ws, _ = run[n]
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                o = m.prefill(x, ids, w, ws, *args)
                e1.record()
                ts[n].append((e0, e1))
                if n != 'base' and r == 0:
                    torch.cuda.synchronize()
                    print(f'  {n}: bitwise == base {torch.equal(o, base)}', flush=True)
                del o
        torch.cuda.synchronize()
        b = st.median(e0.elapsed_time(e1) for e0, e1 in ts['base'][2:])
        for n in run:
            t = [e0.elapsed_time(e1) for e0, e1 in ts[n][2:]]
            med = st.median(t)
            print(f'  {n:8s}: median {med:7.3f} ms/layer  min {min(t):7.3f}  d vs base {med - b:+.3f} ms '
                  f'({(med - b) / b * 100:+.2f}%)', flush=True)
        if 'pfexact' in run:
            run['pfexact'][0].set_grid_routed(0)
        del run, ws, base, x, ids, w
        torch.cuda.empty_cache()


def profile(layer, M=8192):
    torch.manual_seed(1)
    args = load_layer(layer)
    m = V.build('base')
    x, ids, w = rows(M, layer, 0.35)
    ws = torch.empty(m.prefill_workspace_bytes(M), dtype=torch.uint8, device=dev)
    for Mw in (8192, 2048, 300, 9):
        print(f'prefill_workspace_bytes({Mw}) = {m.prefill_workspace_bytes(Mw) / 2**20:.1f} MiB', flush=True)
    m.prefill(x, ids, w, ws, *args)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile as prof
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            m.prefill(x, ids, w, ws, *args)
        torch.cuda.synchronize()
    print(p.key_averages().table(sort_by='cuda_time_total', row_limit=15, max_name_column_width=60))


def kern(layer, names, M=8192, reps=20):
    """Per-kernel device time (torch profiler) of prefill for each variant, same inputs."""
    torch.manual_seed(1)
    args = load_layer(layer)
    x, ids, w = rows(M, layer, 0.35)
    cnt = torch.bincount(ids.flatten().long(), minlength=384)
    n_routed = int(((cnt + 127) // 128).sum())
    mods = {n: V.build(n) for n in names}
    ws = torch.empty(mods['base'].prefill_workspace_bytes(M), dtype=torch.uint8, device=dev)
    for Mw in (8192, 2048, 300):
        print(f'prefill_workspace_bytes({Mw}) = {mods["base"].prefill_workspace_bytes(Mw) / 2**20:.1f} MiB', flush=True)
    if 'pfexact' in mods:
        mods['pfexact'].set_grid_routed(n_routed)
    from torch.profiler import ProfilerActivity, profile as prof
    res = {n: {} for n in names}
    for rnd in range(2):
        for n in (names if rnd == 0 else names[::-1]):
            m = mods[n]
            m.prefill(x, ids, w, ws, *args)
            torch.cuda.synchronize()
            with prof(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(reps):
                    m.prefill(x, ids, w, ws, *args)
                torch.cuda.synchronize()
            for e in p.key_averages():
                t = getattr(e, 'device_time_total', 0)
                if t and e.count:
                    a = res[n].setdefault(e.key, [0.0, 0])
                    a[0] += t
                    a[1] += e.count
    print(f'M={M}: routed tiles {n_routed} of grid.y {(M * 6 + 127) // 128 + 384}', flush=True)
    for n in names:
        tot = 0.0
        print(f'  {n}:')
        for k, (t, c) in sorted(res[n].items(), key=lambda kv: -kv[1][0]):
            if 'og::' not in k and 'Memset' not in k:
                continue
            per_call = t / (2 * reps)
            tot += per_call
            print(f'    {k[:70]:70s} {t / c:9.1f} us/launch  {per_call:9.1f} us/call', flush=True)
        print(f'    total og kernels {tot:9.1f} us/call', flush=True)


if __name__ == '__main__':
    mode, layer = sys.argv[1], int(sys.argv[2])
    if mode == 'decode':
        decode(layer, sys.argv[3:])
    elif mode == 'prefill':
        prefill(layer, sys.argv[3:])
    elif mode == 'kern':
        kern(layer, sys.argv[3:])
    else:
        profile(layer)
    print(f'max reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB', flush=True)
