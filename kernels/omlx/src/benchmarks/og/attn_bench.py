"""Attention-sublayer microbench for the ds41 Mac half (speed + bitwise A/B).

Loads only the attention weights of layers 20-39 (~3.5 GB, all 20 distinct layers,
so no SLC reuse), builds synthetic 8K caches and times the 20-layer attention chain
(projections, indexer, sparse attention) at 1..5 rows. Runs without gpu.lock; keep
production idle. Usage: python attn_bench.py [label]; ATTN_TESTS=chain,ab,abl,proj,parts,index,graph
ATTN_DUMP=path.npz stores every layer output for a bitwise comparison run.
"""
import json
import os
from pathlib import Path
import statistics
import sys
import time

TREE = Path(os.environ.get('DS41_TREE', str(Path.home()/'src/wt/ds41-attn')))
sys.path.insert(0, str(TREE))
for key, value in dict(MLX_ENABLE_TF32='0', DS41_NATIVE_VERIFY='0', DS41_MHC='1', DS41_GROWTH='1',
                       DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1', DS41_NATIVE_DECODE='1').items():
    os.environ.setdefault(key, value)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

mx.set_memory_limit(16 << 30)
mx.set_cache_limit(1 << 30)

from omlx.patches.deepseek_v41 import pipe_decoder, language  # noqa: E402
from omlx.patches.deepseek_v41.config import ModelConfig  # noqa: E402
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, pack_activation  # noqa: E402
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache  # noqa: E402
from omlx.patches.deepseek_v41.encoder_replay import empty_slot  # noqa: E402
from omlx.patches.deepseek_v41.loading import set_module  # noqa: E402

MODEL = Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
LABEL = sys.argv[1] if len(sys.argv) > 1 else 'attn'
TESTS = os.environ.get('ATTN_TESTS', 'chain').split(',')
ROWS = [int(x) for x in os.environ.get('ATTN_ROWS', '1,2,3,4,5').split(',')]
CTX = int(os.environ.get('ATTN_CTX', '8192'))
REPS = int(os.environ.get('ATTN_REPS', '15'))
OUT = Path(os.environ.get('ATTN_OUT', str(Path.home()/'llm/ds41/attn'/(LABEL + '.jsonl'))))
OUT.parent.mkdir(parents=True, exist_ok=True)
_out = OUT.open('a')


def emit(rec):
    line = json.dumps(dict(label=LABEL, **rec))
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


def load():
    raw = json.loads((MODEL/'config.json').read_text())
    config = ModelConfig.from_dict(raw)
    config.ced_prefill = True
    model = pipe_decoder.DecoderContainer(config)
    mapping = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    keep = tuple(f'language_model.layers.{i}.attn.' for i in range(20, 40))
    keep += tuple(f'language_model.layers.{i}.attn_norm.' for i in range(20, 40))
    for filename in sorted({f for k, f in mapping.items() if k.startswith(keep)}):
        values = {k: v for k, v in mx.load(str(MODEL/filename)).items() if k.startswith(keep)}
        for name, spec in specs.items():
            if name + '.weight' in values:
                set_module(model, name, QuantizedProjection(values[name + '.weight'], values[name + '.scales'], **spec))
        model.load_weights(list(values.items()), strict=False)
        mx.eval(values)
        del values
    lm = model.language_model
    return lm, config


def packed(n, width, bits=8, group=32, e4m3=False, seed=0):
    mx.random.seed(seed)
    out = pack_activation(mx.random.normal((1, n, width)).astype(mx.bfloat16), bits, group, e4m3)
    mx.eval(out)
    return out


def make_cache(c, n, rows):
    """Caches at offset n; layer 20 carries n + rows global rows (the box increments already appended)."""
    cache = []
    for i in range(40):
        item = DeepseekV41Cache(1 if i == 20 else 0)
        item.cache = [mx.array([n], mx.int32)] + [None] * 6
        item.left_padding = item.lengths = None
        if i >= 20:
            item.cache[1] = packed(min(n, c.window_size), c.head_dim, seed=i)
        cache.append(item)
    cache[20].cache[2] = packed(n + rows, c.head_dim, 4, 16, True, seed=100)
    cache[20].cache[3] = packed(n + rows, c.index_head_dim, 4, seed=101)
    return cache


@mx.compile
def _next(x0, out):
    return x0 + out * 0.01


def chain(lm, c, cache, x0, start, rows, keep=None, gate=None):
    shared = {}
    if gate is not None:
        x0 = x0 + (gate * 0).astype(x0.dtype)
    x = x0
    outs = []
    for i in range(20, 40):
        a = lm.layers[i].attn
        window = cache[i][1]
        out = a(x, cache[i], shared, start, prebuilt_end=start + rows if i == 20 else None)
        cache[i][1] = window  # keep the window fixed across repetitions
        if keep is not None:
            keep.append(out)
        x = _next(x0, out)
    return x


_SPIN = mx.fast.metal_kernel(
    name='attn_bench_spin', input_names=['n'], output_names=['y'],
    source="""
    float acc = 0.0f;
    for (int i = 0; i < n[0]; ++i) acc = metal::fma(acc, 0.999f, 1.0f);
    y[0] = acc;
    """)
SPIN_N = int(os.environ.get('ATTN_SPIN', '3000000'))
GPU_ONLY = os.environ.get('ATTN_GPU', '1') == '1'


def spin():
    return _SPIN(inputs=[mx.array([SPIN_N], mx.int32)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                 output_shapes=[(1,)], output_dtypes=[mx.float32])[0]


def timed(fn, reps=REPS, warm=3):
    """Median eval ms. ATTN_GPU=1: the graph queues behind a long spin kernel, so the
    host encodes everything first and the time left is GPU time (spin time subtracted)."""
    def run(with_fn):
        ts = []
        for r in range(warm + reps):
            gate = spin() if GPU_ONLY else None
            out = fn(gate) if with_fn else gate
            t = time.perf_counter()
            mx.eval(out)
            if r >= warm:
                ts.append(time.perf_counter() - t)
        return statistics.median(ts), min(ts)
    med, best = run(True)
    if GPU_ONLY:
        base, base_best = run(False)
        LAST['spin_ms'] = round(1000 * base, 3)
        med, best = med - base, best - base_best
    return round(1000 * med, 3), round(1000 * best, 3)


LAST = {}


def main():
    t0 = time.time()
    lm, c = load()
    emit(dict(event='loaded', s=round(time.time() - t0, 1), active_gib=round(mx.get_active_memory() / 2**30, 2)))
    dump = os.environ.get('ATTN_DUMP')
    saved = {}
    for rows in ROWS:
        start = CTX
        cache = make_cache(c, CTX, rows)
        mx.random.seed(7 + rows)
        x0 = (mx.random.normal((1, rows, c.dim)) * 0.5).astype(mx.bfloat16)
        mx.eval(x0)
        if 'chain' in TESTS:
            med, best = timed(lambda g: chain(lm, c, cache, x0, start, rows, gate=g))
            emit(dict(test='chain', rows=rows, ctx=CTX, eval_ms=med, best_ms=best, **LAST))
        if 'ab' in TESTS:
            from omlx.patches.deepseek_v41 import attn_fusions, kernels, fast_qmv

            def config(new):
                attn_fusions.ENABLED = new
                kernels.DS41_ATTN_SHARED = new
                kernels.DS41_MERGE_WIDE = new
                attn_fusions.DECODE_SELECT = new
                fast_qmv.MIN_ROWS = 3 if new else 4
            best = {False: [], True: []}
            for rnd in range(int(os.environ.get('ATTN_AB_ROUNDS', '8'))):
                for new in (False, True):
                    config(new)
                    med, b = timed(lambda g: chain(lm, c, cache, x0, start, rows, gate=g), reps=6, warm=2)
                    best[new].append(b)
            config(True)
            old_ms, new_ms = min(best[False]), min(best[True])
            emit(dict(test='ab', rows=rows, ctx=CTX, old_ms=round(old_ms, 3), new_ms=round(new_ms, 3),
                      saved_ms=round(old_ms - new_ms, 3), old_med=round(statistics.median(best[False]), 3),
                      new_med=round(statistics.median(best[True]), 3)))
        if 'index' in TESTS:
            from omlx.patches.deepseek_v41 import attn_fusions, kernels, fast_qmv
            from omlx.patches.deepseek_v41.quantization import quantize_activation
            qr0 = (mx.random.normal((1, rows, c.q_lora_rank)) * 0.3).astype(mx.bfloat16)
            mx.eval(qr0)
            for flag in (False, True):
                attn_fusions.ENABLED = flag
                attn_fusions.DECODE_SELECT = flag
                base_shared = {}
                # layer 20 once to get candidates/index_k for the scorer layers
                a20 = lm.layers[20].attn
                base_shared['qr8'] = (qr0, quantize_activation(qr0))
                base_shared['idx20'] = a20.indexer(x0, qr0, None, cache[20], base_shared, start, 1,
                                                   latent_start=start + rows, prebuilt=True)
                mx.eval(base_shared['idx20'], base_shared['candidates'])
                for layer in (20, 24):
                    ind = lm.layers[layer].attn.indexer

                    def run(g, ind=ind, layer=layer):
                        sh = dict(base_shared)
                        qr = qr0 if g is None else qr0 + (g * 0).astype(qr0.dtype)
                        out = None
                        for _ in range(10):
                            if flag:
                                sh['qr8'] = (qr, quantize_activation(qr))
                            out = ind(x0, qr, None, cache[20] if layer == 20 else cache[layer], sh, start, 1,
                                      latent_start=start + rows, prebuilt=True)
                            qr = qr0 + (out.reshape(-1)[:1] * 0).astype(qr0.dtype)
                        return qr
                    med, best = timed(run, reps=8)
                    emit(dict(test='index', layer=layer, rows=rows, new=flag, per_call_us=round(best * 100, 1),
                              med_us=round(med * 100, 1)))
            attn_fusions.ENABLED = True
            attn_fusions.DECODE_SELECT = True
        if 'graph' in TESTS:
            import collections, re, tempfile
            for layers in ((20,), (21,), (24,)):
                shared = {}
                x = x0
                for i in range(20, layers[0] + 1):
                    a = lm.layers[i].attn
                    window = cache[i][1]
                    prev = x
                    x = a(x, cache[i], shared, start, prebuilt_end=start + rows if i == 20 else None)
                    cache[i][1] = window
                    if i < layers[0]:
                        mx.eval(x, *[v for v in shared.values() if isinstance(v, mx.array)])
                        x = x0
                path = tempfile.mktemp(suffix='.dot')
                mx.export_to_dot(path, x)
                text = Path(path).read_text()
                labels = collections.Counter(re.findall(r'label ="([^"]+)"', text))
                views = {'Reshape', 'Squeeze', 'ExpandDims', 'Broadcast', 'Transpose', 'Flatten', 'Unflatten',
                         'AsStrided', 'View', 'StopGradient', 'Depends'}
                emit(dict(test='graph', layer=layers[0], rows=rows, non_view=sum(v for k, v in labels.items() if k not in views),
                          top=dict(labels.most_common(40))))
                mx.eval(x)
        if 'abl' in TESTS:
            real_sparse, real_index = language.packed_sparse_attention, language.Indexer.__call__
            fixed = mx.broadcast_to(mx.arange(512, dtype=mx.int32)[None, None] * 8, (1, rows, 512))
            fixed_blocks = mx.broadcast_to(mx.arange(1024, dtype=mx.int32)[None, None], (1, rows, 1024))
            mx.eval(fixed, fixed_blocks)

            def no_index(self, x, qr, latent, cache, shared, start, ratio, latent_start=None, prebuilt=False):
                shared['index_k'] = cache[3] if cache[3] is not None else shared.get('index_k')
                shared['candidates'] = fixed_blocks
                return fixed + 0 * qr[..., :1].astype(mx.int32)

            def no_sparse(q, window, pooled, wi, ci, sink, scale, **kw):
                return q + 0 * ci[..., :1, None].astype(q.dtype)
            for name, (sp, ix) in dict(no_index=(real_sparse, no_index), no_sparse=(no_sparse, real_index),
                                       no_both=(no_sparse, no_index)).items():
                language.packed_sparse_attention, language.Indexer.__call__ = sp, ix
                try:
                    med, best = timed(lambda g: chain(lm, c, cache, x0, start, rows, gate=g))
                finally:
                    language.packed_sparse_attention, language.Indexer.__call__ = real_sparse, real_index
                emit(dict(test='abl', abl=name, rows=rows, ctx=CTX, eval_ms=med))
        if dump:
            keep = []
            chain(lm, c, cache, x0, start, rows, keep)
            mx.eval(keep)
            for i, o in enumerate(keep):
                saved[f'r{rows}_l{20 + i}'] = np.array(o.astype(mx.float32))
        if 'proj' in TESTS:
            def proj(g):
                x = x0 if g is None else x0 + (g * 0).astype(x0.dtype)
                for i in range(20, 40):
                    a = lm.layers[i].attn
                    query, kvi = a._input_projections(x)
                    qr = a.q_norm(query)
                    qb = a.wq_b(qr)
                    g = qb.reshape(1, rows, c.o_groups, -1)[..., :4096]
                    w = a.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
                    from omlx.patches.deepseek_v41 import decode_fusions as df
                    pr = df.grouped_gemv(g, w) if df.grouped_gemv_supported(g, w) else mx.einsum('bsgd,grd->bsgr', g, w)
                    x = _next(x0, a.wo_b(pr.flatten(-2)) + kvi[..., :1] * 0)
                return x
            med, best = timed(proj)
            emit(dict(test='proj', rows=rows, eval_ms=med, best_ms=best))
        if 'parts' in TESTS:
            from omlx.patches.deepseek_v41 import decode_fusions as df
            from omlx.patches.deepseek_v41.quantization import quantize_activation
            q8 = quantize_activation(x0)
            qr = mx.random.normal((1, rows, c.q_lora_rank)).astype(mx.bfloat16)
            og = mx.random.normal((1, rows, c.o_groups, 4096)).astype(mx.bfloat16)
            ob = mx.random.normal((1, rows, 8192)).astype(mx.bfloat16)
            mx.eval(q8, qr, og, ob)
            parts = {
                'wq_a': (lambda a: a.wq_a.project_quantized(q8), 1280 * 5120 * (1 + 1 / 32)),
                'wkv': (lambda a: a.wkv.project_quantized(q8), 512 * 5120 * (1 + 1 / 32)),
                'wq_b': (lambda a: a.wq_b(qr), 32768 * 1280 * (1 + 1 / 32)),
                'wo_a': (lambda a: (df.grouped_gemv(og, a.wo_a.weight.reshape(8, 1024, -1))
                                    if df.grouped_gemv_supported(og, a.wo_a.weight.reshape(8, 1024, -1))
                                    else mx.einsum('bsgd,grd->bsgr', og, a.wo_a.weight.reshape(8, 1024, -1))),
                          8 * 1024 * 4096 * 2),
                'wo_b': (lambda a: a.wo_b(ob), 5120 * 8192 * (1 + 1 / 32)),
            }
            def glue(y, prev):
                return y + (prev.reshape(-1)[:1] * 0).astype(y.dtype)
            inputs = {'wq_a': q8, 'wkv': q8, 'wq_b': qr, 'wo_a': og, 'wo_b': ob}
            for name, (fn, nbytes) in [('glue', (None, 0))] + list(parts.items()):
                def run(g, fn=fn, name=name):
                    prev = g if g is not None else mx.zeros((1,))
                    for i in range(20, 40):
                        if fn is None:
                            prev = glue(q8, prev)
                            continue
                        a = lm.layers[i].attn
                        # Rebind the input through one glue op so calls stay serialized.
                        x_in = glue(inputs[name], prev)
                        if name in ('wq_a', 'wkv', 'wo_b', 'wq_b'):
                            prev = {'wq_a': a.wq_a, 'wkv': a.wkv, 'wo_b': a.wo_b, 'wq_b': a.wq_b}[name].project_quantized(x_in)
                        else:
                            prev = fn(a) if False else (df.grouped_gemv(x_in, a.wo_a.weight.reshape(8, 1024, -1))
                                                        if df.grouped_gemv_supported(x_in, a.wo_a.weight.reshape(8, 1024, -1))
                                                        else mx.einsum('bsgd,grd->bsgr', x_in, a.wo_a.weight.reshape(8, 1024, -1)))
                    return prev
                med, best = timed(run)
                if name == 'glue':
                    glue_ms = med
                    emit(dict(test='part', part=name, rows=rows, per_layer_us=round(med * 50, 2)))
                    continue
                net = med - glue_ms
                emit(dict(test='part', part=name, rows=rows, per_layer_us=round(net * 50, 2),
                          gbps=round(20 * nbytes / (net * 1e-3) / 1e9, 1)))
    if dump:
        np.savez(dump, **saved)
        emit(dict(event='dumped', path=dump, n=len(saved)))


if __name__ == '__main__':
    main()
