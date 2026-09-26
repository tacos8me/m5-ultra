"""Routed-MoE microbench + bitwise check, 3 real layers' experts (layers 20-22 ffn only, ~21 GB, no gpu.lock;
run only while production is idle).

Modes: prod = f56f7ffa limits (fused pair kernels for <= 5 rows, gather_qmm above); pair = fused pair kernels
for <= 16 rows (DS41_MOE_FUSED_ROWS). Per-layer us = 60 serialized expert calls (rotating the 3 layers, forced
routing at the measured real per-layer unions) per eval, incl. SwiGLU quant and the serializing glue.
MOE_BENCH_OUT: jsonl path. MOE_BENCH_ROWS: comma list (default 1,2,3,4,5,6,8,10). MOE_BENCH_TESTS: check,bench.
"""
import json, os, statistics, sys, time
from pathlib import Path

TREE = Path(os.environ.get('DS41_TREE', str(Path.home()/'src/wt/ds41-moe')))
sys.path.insert(0, str(TREE))
for key, value in dict(MLX_ENABLE_TF32='0', DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1',
                       DS41_NATIVE_DECODE='1').items():
    os.environ.setdefault(key, value)
import mlx.core as mx
import numpy as np

mx.set_memory_limit(40 << 30)
mx.set_cache_limit(512 << 20)
from omlx.patches.deepseek_v41 import language, moe_decode
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.loading import _load_shard, set_module
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, quantize_activation

MODEL = Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
LAYERS = (20, 21, 22)
OUT = open(os.environ.get('MOE_BENCH_OUT', str(Path.home()/'llm/ds41/moe/bench.jsonl')), 'a')
EXPERT = 18.80e6
BW = 1.17e12
# Measured real per-layer unions (REPORT.md, 8K; k=10 = two 5-row streams).
UNION = {1: 6, 2: 10, 3: 14, 4: 17, 5: 20, 6: 25, 8: 29, 10: 37}
UNION.update({int(k): int(v) for k, v in (kv.split(':') for kv in os.environ.get('MOE_BENCH_UNION', '').split(',') if kv)})


assert language.glm_fast.has_symbol("deepseek_v41_grouped_expert"), "native ext missing: copy build artifacts"
_calls = {}
_gate_up = moe_decode.gate_up


def _counted(*a, **kw):
    _calls['fused'] = _calls.get('fused', 0) + 1
    return _gate_up(*a, **kw)


moe_decode.gate_up = _counted


def emit(r):
    line = json.dumps(r); OUT.write(line + '\n'); OUT.flush(); print(line, flush=True)


def load():
    raw = json.loads((MODEL/'config.json').read_text())
    c = ModelConfig.from_dict(raw)
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    mods = {}
    for i in LAYERS:
        pre = f'language_model.layers.{i}.'
        keys = [k for k in wm if k.startswith(pre + 'ffn.') or k == pre + 'ffn_norm.weight']
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
        mx.eval(m.parameters())
        mods[i] = (m, vals['ffn_norm.weight'])
        mx.clear_cache()
    return c, mods


def real_inputs(norm_w):
    h = mx.load(str(Path.home()/'llm/ds41/og-speed/snap-8k/step.safetensors'))['h'][0]  # (5, 4, 5120)
    v = h.transpose(1, 0, 2).reshape(-1, h.shape[-1]).astype(mx.float32)  # stream-major: 20 real vectors
    v = v * mx.rsqrt((v * v).mean(-1, keepdims=True) + 1e-6)
    return (v * norm_w.astype(mx.float32)).astype(mx.bfloat16)


def experts_out(m, x, idx, w):
    q = quantize_activation(x)
    return m.experts(q[..., None, None, :], idx, w, input_quantized=True, max_grouped_tokens=16).squeeze(-2)


def set_mode(mode):
    moe_decode.MAX_ROWS = {'prod': 5, 'pair': 16}[mode]


def check(c, mods, rows_list):
    for i, (m, nw) in mods.items():
        xs = real_inputs(nw)
        for rows in rows_list:
            x = xs[:rows][None]
            idx, w = m.gate(x, None)
            sets = [('real', idx, w)]
            rng = np.random.default_rng(rows)
            for u in sorted({UNION[rows], max(6, UNION[rows] // 2)}):
                pool = rng.choice(384, u, replace=False)
                fi = mx.array(np.array([[int(pool[(6 * j + t) % u]) for t in range(6)] for j in range(rows)], np.uint32)[None])
                sets.append((f'forced{u}', fi, w))
            for label, idx, w in sets:
                set_mode('prod'); a = experts_out(m, x, idx, w); fa = m(x, None) if label == 'real' else None
                set_mode('pair'); b = experts_out(m, x, idx, w); fb = m(x, None) if label == 'real' else None
                mx.eval(a, b, *(t for t in (fa, fb) if t is not None))
                d = (a.astype(mx.float32) - b.astype(mx.float32)).abs()
                rec = dict(test='check', layer=i, rows=rows, routing=label,
                           union=len(set(np.array(idx).reshape(-1).tolist())),
                           bitwise=bool(mx.array_equal(a, b).item()), max_abs=float(d.max().item()),
                           max_rel=float((d / (a.astype(mx.float32).abs() + 1e-30)).max().item()))
                if fa is not None:
                    rec['moe_out_bitwise'] = bool(mx.array_equal(fa, fb).item())
                    rec['moe_out_max_abs'] = float((fa.astype(mx.float32) - fb.astype(mx.float32)).abs().max().item())
                rec['kernel_calls'] = dict(_calls); _calls.clear()
                emit(rec)


def bench(c, mods, rows_list, modes, calls=60, reps=int(os.environ.get('MOE_BENCH_REPS', '6'))):
    layers = [mods[i][0] for i in LAYERS]
    rng = np.random.default_rng(0)
    for rows in rows_list:
        u = UNION[rows]
        x = mx.random.normal((1, rows, c.dim)).astype(mx.bfloat16)
        q = quantize_activation(x)
        w = mx.full((1, rows, 6), 1 / 6, mx.float32)
        idx_sets = []
        for _ in range(calls):
            pool = rng.choice(384, u, replace=False)
            idx_sets.append(mx.array(np.array([[int(pool[(6 * j + t) % u]) for t in range(6)] for j in range(rows)], np.uint32)[None]))
        mx.eval(idx_sets)
        res = {}
        for mode in modes:
            set_mode(mode)

            def run():
                h = q
                for k, ids in enumerate(idx_sets):
                    y = layers[k % 3].experts(h[..., None, None, :], ids, w, input_quantized=True, max_grouped_tokens=16).squeeze(-2)
                    h = quantize_activation(y.sum(-2).astype(mx.bfloat16))
                return h
            for _ in range(2):
                mx.eval(run())
            ts = []
            for _ in range(reps):
                t = time.perf_counter(); mx.eval(run()); ts.append(time.perf_counter() - t)
            per = statistics.median(ts) / calls
            res[mode] = per
            floor = u * EXPERT / BW
            emit(dict(test='bench', mode=mode, rows=rows, union=u, per_layer_us=round(per * 1e6, 1),
                      floor_us=round(floor * 1e6, 1), pct_floor=round(100 * floor / per, 1), kernel_calls=dict(_calls)))
            _calls.clear()
        set_mode('pair')


if __name__ == '__main__':
    rows_list = [int(r) for r in os.environ.get('MOE_BENCH_ROWS', '1,2,3,4,5,6,8,10').split(',')]
    t = time.time()
    c, mods = load()
    print('loaded', round(time.time() - t, 1), 's', round(mx.get_active_memory() / 2**30, 1), 'GiB', flush=True)
    if 'check' in os.environ.get('MOE_BENCH_TESTS', 'check,bench'):
        check(c, mods, rows_list)
    if 'bench' in os.environ.get('MOE_BENCH_TESTS', 'check,bench'):
        bench(c, mods, rows_list, os.environ.get('MOE_BENCH_MODES', 'prod,pair').split(','))
