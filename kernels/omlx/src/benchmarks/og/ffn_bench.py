"""FFN sublayer fusion (ffn_fuse): bitwise check + per-layer timing on 3 real layers' FFN.

Loads layers 20-22 ffn + ffn_norm + hc_ffn_* only (~22 GB, no gpu.lock). Run only through idle_guard.py
(production idle). Reference = hc_fuse.block_forward's FFN half (project_pre_norm -> MoE -> post_mix) for
singleton rows, og_fused._moe for fused row blocks. New = ffn_fuse.ffn_forward.

FB_TESTS=check,bench  FB_ROWS=1,2,3,4,5  FB_FUSED=2+2,3+3,5+5,2+5  FB_OUT=jsonl
  python benchmarks/og/idle_guard.py $PY -u benchmarks/og/ffn_bench.py
"""
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key, value in dict(MLX_ENABLE_TF32='0', DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1',
                       DS41_NATIVE_DECODE='1', DS41_INDEX_NAX='1', DS41_SPARSE='1', DS41_NATIVE_VERIFY='0').items():
    os.environ.setdefault(key, value)
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.set_memory_limit(34_000_000_000)
mx.set_cache_limit(512 << 20)
from omlx.patches.deepseek_v41 import ffn_fuse, hc_fuse, language, og_fused  # noqa: E402
from omlx.patches.deepseek_v41.config import ModelConfig  # noqa: E402
from omlx.patches.deepseek_v41.loading import _load_shard, set_module  # noqa: E402
from omlx.patches.deepseek_v41.quantization import QuantizedProjection  # noqa: E402

HOME = Path.home()
MODEL = HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
LAYERS = (20, 21, 22)
OUT_PATH = Path(os.environ.get('FB_OUT', str(HOME/'llm/ds41/ffn/bench.jsonl')))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
_out = OUT_PATH.open('a')
TESTS = os.environ.get('FB_TESTS', 'check,bench').split(',')


def emit(**r):
    line = json.dumps(r)
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


def load():
    raw = json.loads((MODEL/'config.json').read_text())
    c = ModelConfig.from_dict(raw)
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    blocks = []
    for i in LAYERS:
        pre = f'language_model.layers.{i}.'
        keys = [k for k in wm if k.startswith(pre + 'ffn.') or k.startswith(pre + 'hc_ffn_') or k == pre + 'ffn_norm.weight']
        vals = {}
        for f in sorted({wm[k] for k in keys}):
            v = _load_shard(MODEL/f)
            vals.update({k[len(pre):]: v[k] for k in keys if k in v})
            del v
        m = language.MoE(c)
        for name in ('experts.w1', 'experts.w2', 'experts.w3', 'shared_experts.w1', 'shared_experts.w2', 'shared_experts.w3'):
            spec = specs[pre + 'ffn.' + name]
            set_module(m, name, QuantizedProjection(vals['ffn.' + name + '.weight'], vals['ffn.' + name + '.scales'], **spec))
        m.load_weights([(k[4:], v) for k, v in vals.items() if k.startswith('ffn.gate.')], strict=False)
        norm = language.RMSNorm(c.dim, c.norm_eps)
        norm.weight = vals['ffn_norm.weight']
        blk = types.SimpleNamespace(_config=c, ffn=m, ffn_norm=norm, hc_ffn_fn=vals['hc_ffn_fn'],
                                    hc_ffn_base=vals['hc_ffn_base'], hc_ffn_scale=vals['hc_ffn_scale'])
        mx.eval(m.parameters(), norm.weight, blk.hc_ffn_fn, blk.hc_ffn_base, blk.hc_ffn_scale)
        blocks.append(blk)
        mx.clear_cache()
    return c, blocks


def reference(blk, h, pre, bounds=None):
    c = blk._config
    mix, x = hc_fuse.project_pre_norm(h, pre, blk.hc_ffn_fn, blk.ffn_norm.weight, c.norm_eps, blk.ffn_norm.eps)
    f = blk.ffn(x, None) if bounds is None else og_fused._moe(blk.ffn, x, bounds)
    return hc_fuse.post_mix(f, h, mix, blk.hc_ffn_scale, blk.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters)


def fused(blk, h, pre, bounds=None):
    return ffn_fuse.ffn_forward(blk, h, pre, bounds)


def inputs():
    """Real boundary rows (snap-8k) plus hc-slot permutations of them: 20 realistic rows."""
    d = mx.load(str(HOME/'llm/ds41/og-speed/snap-8k/step.safetensors'))
    h, pre = d['h'], d['pre']
    hs, ps = [h], [pre]
    for perm in ([1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]):
        hs.append(h[:, :, perm, :]); ps.append(pre[:, :, perm])
    return mx.concatenate(hs, 1), mx.concatenate(ps, 1)


def bounds_of(spec):
    rows = [int(x) for x in spec.split('+')]
    out, b = [], 0
    for n in rows:
        out.append((b, b + n)); b += n
    return out, b


def equal(a, b):
    return all(bool(mx.array_equal(x, y).item()) for x, y in zip(a, b))


def check(c, blocks, H, P):
    cases = [(r, None) for r in (1, 2, 3, 4, 5)]
    cases += [bounds_of(s)[::-1] for s in ('2+2', '3+3', '5+5', '2+5', '4+3', '5+5+5', '4+5+3+2')]
    mx.random.seed(7)
    for rows, bounds in cases:
        results = []
        for label, (h0, p0) in (('real', (H, P)), ('real_off5', (H[:, 5:], P[:, 5:])),
                                ('scaled', ((H * 8).astype(mx.bfloat16), P)),
                                ('random', ((mx.random.normal(H.shape) * 0.3).astype(mx.bfloat16), P))):
            h, p = h0[:, :rows], p0[:, :rows]
            ok, n = True, 0
            # Chain the three layers twice so each layer sees evolving real-scale inputs.
            ha, pa, hb, pb = h, p, h, p
            for step in range(6):
                blk = blocks[step % 3]
                ha, pa = reference(blk, ha, pa, bounds)
                hb, pb = fused(blk, hb, pb, bounds)
                mx.eval(ha, pa, hb, pb)
                same = equal([ha, pa], [hb, pb])
                n += 1
                if not same:
                    ok = False
                    diff = float(mx.abs(ha.astype(mx.float32) - hb.astype(mx.float32)).max().item())
                    emit(test='check_fail', rows=rows, bounds=bounds, input=label, step=step, max_abs=diff)
                    hb, pb = ha, pa
            results.append(dict(input=label, steps=n, bitwise=ok))
        emit(test='check', rows=rows, bounds=bounds, bitwise=all(r['bitwise'] for r in results), cases=results)


def ab(f_ref, f_new, reps=int(os.environ.get('FB_REPS', '11')), warm=3):
    t = {'ref': [], 'new': []}
    for rep in range(reps + warm):
        for name in (('ref', 'new') if rep % 2 == 0 else ('new', 'ref')):
            fn = f_ref if name == 'ref' else f_new
            mx.synchronize()
            s = time.perf_counter()
            mx.eval(fn())
            if rep >= warm:
                t[name].append((time.perf_counter() - s) * 1000)
    ref, new = statistics.median(t['ref']), statistics.median(t['new'])
    return dict(ref_ms=round(ref, 3), new_ms=round(new, 3), saved_ms=round(ref - new, 3),
                pair_saved=sorted(round(a - b, 3) for a, b in zip(t['ref'], t['new'])))


def bench(c, blocks, H, P, calls=60):
    specs = [(str(r), r, None) for r in (1, 2, 3, 4, 5)]
    for s in os.environ.get('FB_FUSED', '2+2,3+3,5+5').split(','):
        b, n = bounds_of(s)
        specs.append((s, n, b))
    for label, rows, bounds in specs:
        h0, p0 = H[:, :rows], P[:, :rows]

        def chain(fn):
            def run():
                h, p = h0, p0
                for k in range(calls):
                    h, p = fn(blocks[k % 3], h, p, bounds)
                return h, p
            return run
        r = ab(chain(reference), chain(fused))
        emit(test='bench', rows=label, calls=calls, per_layer_ref_us=round(r['ref_ms'] / calls * 1000, 1),
             per_layer_new_us=round(r['new_ms'] / calls * 1000, 1),
             per_forward_saved_ms=round(r['saved_ms'] / calls * 20, 3), **r)


def prof(c, blocks, H, P, calls=60):
    """Per-epoch ablation of ffn_forward (each variant a 60-layer serialized chain)."""
    from omlx.patches.deepseek_v41 import decode_fusions
    for rows in [int(r) for r in os.environ.get('FB_PROF_ROWS', '1,5').split(',')]:
        h0, p0 = H[:, :rows], P[:, :rows]
        # Fixed per-layer routing/activations from one real pass (so ablated chains keep real unions).
        fixed = []
        h, p = h0, p0
        for k in range(3):
            blk = blocks[k]
            mix, xf, xq = ffn_fuse.pre_norm_q(h, p, blk.hc_ffn_fn, blk.ffn_norm.weight, c.norm_eps, blk.ffn_norm.eps)
            ys, raw = ffn_fuse.shared_router(blk.ffn, xq.reshape(rows, -1), xf.reshape(rows, -1))
            ids, w = decode_fusions.router(raw.reshape(*xf.shape[:-1], -1), blk.ffn.gate.bias, c.route_scale)
            yr = ffn_fuse.routed_up(blk.ffn, xq.reshape(rows, -1), ids.reshape(-1), w.reshape(-1))
            r, sh = ffn_fuse.down(blk.ffn, yr, ys, ids.reshape(-1))
            mx.eval(xq, xf, ys, raw, ids, w, yr, r, sh, mix)
            fixed.append(dict(xq=xq.reshape(rows, -1), xf=xf.reshape(rows, -1), ys=ys, ids=ids.reshape(-1),
                              w=w.reshape(-1), yr=yr, r=r, sh=sh, mix=mix, topk=ids.shape[-1],
                              union=len(set(np.array(ids).reshape(-1).tolist()))))
            h, p = ffn_fuse.post_combine(r, sh, h, mix, blk.hc_ffn_scale, blk.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters, 6)
        eps = mx.array(1e-30, mx.bfloat16)

        def variant(name):
            def run():
                h, p = h0, p0
                acc = None
                for k in range(calls):
                    blk, f = blocks[k % 3], fixed[k % 3]
                    m = blk.ffn
                    if name == 'full':
                        h, p = ffn_fuse.ffn_forward(blk, h, p)
                    elif name == 'no_moe':  # pre_norm_q + shared/router + top6 + post (routed/shared outputs fixed)
                        mix, xf, xq = ffn_fuse.pre_norm_q(h, p, blk.hc_ffn_fn, blk.ffn_norm.weight, c.norm_eps, blk.ffn_norm.eps)
                        ys, raw = ffn_fuse.shared_router(m, xq.reshape(rows, -1), xf.reshape(rows, -1))
                        ids, w = decode_fusions.router(raw.reshape(*xf.shape[:-1], -1), m.gate.bias, c.route_scale)
                        h, p = ffn_fuse.post_combine(f['r'], f['sh'] + (ids[..., :1].astype(mx.bfloat16) * eps).reshape(rows, 1) + ys[:, :1] * eps,
                                                     h, mix, blk.hc_ffn_scale, blk.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters, 6)
                    elif name == 'hc_only':  # pre_norm_q + post
                        mix, xf, xq = ffn_fuse.pre_norm_q(h, p, blk.hc_ffn_fn, blk.ffn_norm.weight, c.norm_eps, blk.ffn_norm.eps)
                        h, p = ffn_fuse.post_combine(f['r'], f['sh'] + xq[:, 0, :1] * eps, h, mix, blk.hc_ffn_scale,
                                                     blk.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters, 6)
                    elif name == 'moe_only':  # routed up + down, fixed inputs, chained through xq
                        xq = f['xq'] if acc is None else f['xq'] + acc
                        yr = ffn_fuse.routed_up(m, xq, f['ids'], f['w'])
                        r, sh = ffn_fuse.down(m, yr, f['ys'], f['ids'])
                        acc = (r[:rows, :] * eps)
                    elif name == 'moe_nodep':  # up and down in one barrier epoch (down reads the fixed yr)
                        xq = f['xq'] if acc is None else f['xq'] + acc
                        yr = ffn_fuse.routed_up(m, xq, f['ids'], f['w'])
                        r, sh = ffn_fuse.down(m, f['yr'] if acc is None else f['yr'] + acc[:1, :1], f['ys'], f['ids'])
                        acc = yr[:rows, :1] * eps + r[:rows, :1] * eps
                    elif name == 'routed_up_only':
                        xq = f['xq'] if acc is None else f['xq'] + acc
                        yr = ffn_fuse.routed_up(m, xq, f['ids'], f['w'])
                        acc = yr[:rows, :1] * eps
                    elif name == 'down_only':
                        yr = f['yr'] if acc is None else f['yr'] + acc
                        r, sh = ffn_fuse.down(m, yr, f['ys'], f['ids'])
                        acc = r[:1, :1] * eps
                    elif name == 'shared_router_only':
                        xq = f['xq'] if acc is None else f['xq'] + acc
                        ys, raw = ffn_fuse.shared_router(m, xq, f['xf'])
                        acc = ys[:, :1] * eps + raw[:, :1].astype(mx.bfloat16) * eps
                    elif name in ('sr_shared', 'sr_router'):
                        xq = f['xq'] if acc is None else f['xq'] + acc
                        sh_, gate = m.shared_experts, m.gate
                        n_, nr = sh_.w1.weight.shape[0], gate.weight.shape[0]
                        rtg = nr // 16 if rows == 1 else nr // 8
                        tgs = rtg if name == 'sr_router' else rtg + n_ // 32
                        ys, raw = ffn_fuse._shared_router_kernel()(
                            inputs=[xq, sh_.w1.weight, sh_.w1.scales, sh_.w3.weight, sh_.w3.scales,
                                    ffn_fuse._limit(sh_._limit), gate.weight, f['xf']],
                            template=[("T", xq.dtype), ("K", xq.shape[-1]), ("N", n_), ("M", rows), ("NR", nr),
                                      ("RTG", 0 if name == 'sr_shared' else rtg)],
                            grid=(32 * (n_ // 32 if name == 'sr_shared' else rtg), 16, 1), threadgroup=(32, 16, 1),
                            output_shapes=[(rows, n_), (rows, nr)], output_dtypes=[xq.dtype, mx.float32])
                        acc = ys[:, :1] * eps + raw[:, :1].astype(mx.bfloat16) * eps
                    elif name == 'glue_only':  # the chaining ops alone
                        xq = f['xq'] if acc is None else f['xq'] + acc
                        acc = xq[:, :1] * eps
                return (h, p) if acc is None else acc
            return run
        names = os.environ.get('FB_PROF_NAMES', 'full,no_moe,hc_only,moe_only,moe_nodep,routed_up_only,down_only,shared_router_only,glue_only').split(',')
        t = {n: [] for n in names}
        for rep in range(12):
            for n in (names if rep % 2 == 0 else names[::-1]):
                fn = variant(n)
                mx.synchronize()
                s0 = time.perf_counter()
                mx.eval(fn())
                if rep >= 2:
                    t[n].append((time.perf_counter() - s0) * 1000)
        med = {n: round(statistics.median(v) / calls * 1000, 1) for n, v in t.items()}
        emit(test='prof', rows=rows, unions=[f['union'] for f in fixed], per_layer_us=med)


if __name__ == '__main__':
    t = time.time()
    c, blocks = load()
    emit(event='loaded', s=round(time.time() - t, 1), active_GB=round(mx.get_active_memory() / 1e9, 2))
    H, P = inputs()
    if 'check' in TESTS:
        check(c, blocks, H, P)
    if 'bench' in TESTS:
        bench(c, blocks, H, P)
    if 'prof' in TESTS:
        prof(c, blocks, H, P)
    emit(event='done', peak_GB=round(mx.get_peak_memory() / 1e9, 2))
