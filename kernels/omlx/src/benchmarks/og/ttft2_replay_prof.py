"""ds41-ttft2: Mac tail replay profiler + bitwise identity check on a PARTIAL decoder half (layers 20/24/25
aliased + head/norm/embed/DSpark, ~31 GB). Run only under ttft2_guard.py (production idle, nothing else heavy).

Inputs: real box states from ttft2_fetch_state.py (~/llm/ds41/ttft2/states/state-<N>.pkl). A case "N@S" imports
state S re-targeted to an N-token prompt: layer-20 rows tiled from the real ones, the real 256-row tail, tokens
tiled (manifest/digests recomputed). Replay = the production DecoderHalf.import_state path.

RP_CASES   comma list of N@S (default: 8217@8217,8290@8290,8600@8600,131162@8290,131500@8600,524400@8600)
RP_TESTS   time,layers,parts,digest (default time,digest)
RP_REPS    timing reps per case (default 5)
RP_LABEL   output label (~/llm/ds41/ttft2/<label>.jsonl); digests in the same file (kind=digest)
"""
import hashlib, json, os, pickle, statistics, sys, time
from pathlib import Path
HOME = Path.home()
TREE = os.environ.get('DS41_TREE', str(HOME/'src/wt/ds41-ttft2'))
sys.path.insert(0, TREE)
for key, value in dict(MLX_ENABLE_TF32='0', DS41_NATIVE_VERIFY='0', DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1',
                       DS41_INDEX_NAX='1', DS41_SPARSE='1', DS41_NATIVE_DECODE='1', DS41_MTP_COST_POLICY='1',
                       DS41_COPY_DRAFT='1', DS41_EXTRA_DRAFT='1', DS41_PREFIX_CACHE_GIB='0',
                       DS41_BATCH_VERIFY='0').items():
    os.environ.setdefault(key, value)
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
mx.set_memory_limit(40 << 30)
mx.set_cache_limit(512 << 20)
from omlx.patches.deepseek_v41 import pipe_decoder, language, encoder_replay, woa_compact  # noqa: E402
from omlx.patches.deepseek_v41.config import ModelConfig  # noqa: E402
from omlx.patches.deepseek_v41.loading import _load_shard, set_module  # noqa: E402
from omlx.patches.deepseek_v41.quantization import QuantizedProjection  # noqa: E402
from omlx.patches.deepseek_v41.handoff import token_digest  # noqa: E402

MODEL_DIR = HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
STATES = HOME/'llm/ds41/ttft2/states'
LABEL = os.environ.get('RP_LABEL', 'rp')
OUT = (HOME/'llm/ds41/ttft2'/(LABEL + '.jsonl')).open('a')
TESTS = os.environ.get('RP_TESTS', 'time,digest').split(',')
REPS = int(os.environ.get('RP_REPS', '5'))
CASES = os.environ.get('RP_CASES', '8217@8217,8290@8290,8600@8600,131162@8290,131500@8600,524400@8600').split(',')
KIND = {20: 20, **{i: 24 for i in (24, 28, 32, 36)}}
T0 = time.time()


def emit(rec):
    rec = dict(t=round(time.time() - T0, 2), label=LABEL, tree=TREE, **rec)
    OUT.write(json.dumps(rec) + '\n'); OUT.flush(); print(json.dumps(rec), flush=True)


def load():
    raw = json.loads((MODEL_DIR/'config.json').read_text())
    config = ModelConfig.from_dict(raw)
    config.ced_prefill = True
    model = pipe_decoder.DecoderContainer(config)
    mapping = json.loads((MODEL_DIR/'model.safetensors.index.json').read_text())['weight_map']
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    keep = tuple(f'language_model.{p}.' for p in ('layers.20', 'layers.24', 'layers.25', 'head', 'norm', 'embed', 'mtp'))
    for filename in sorted({f for k, f in mapping.items() if k.startswith(keep)}):
        values = {k: v for k, v in _load_shard(MODEL_DIR/filename).items() if k.startswith(keep)}
        for name, spec in {n: s for n, s in specs.items() if n + '.weight' in values}.items():
            set_module(model, name, QuantizedProjection(values[name + '.weight'], values[name + '.scales'], **spec))
        model.load_weights(list(values.items()), strict=False)
        mx.eval(values)
        del values
        mx.clear_cache()
    lm = model.language_model
    for i in range(20, 40):
        if i not in (20, 24, 25):
            lm.layers[i] = lm.layers[KIND.get(i, 25)]
    model.eval()
    woa_compact.install(lm)
    emit(dict(event='loaded', active_gib=round(mx.get_active_memory() / 2**30, 2)))
    return lm


def case_state(spec):
    """(tokens, tensors in wire layout, manifest, identity) for 'N@S'."""
    n, src = (int(x) for x in spec.split('@'))
    s = pickle.loads((STATES/f'state-{src}.pkl').read_bytes())
    tokens0, tensors0, man = s['tokens'], s['tensors'], json.loads(json.dumps(s['manifest']))
    tensors = {k: [d, list(sh), b] for k, (d, sh, b) in tensors0.items()}
    if n != src:
        tokens = (tokens0 * (-(-n // len(tokens0))))[:n]
        for slot, width in ((2, 288), (3, 68)):
            rows = np.frombuffer(tensors0[f'layer.20.slot.{slot}'][2], np.uint8).reshape(-1, width)
            reps = -(-(n - 1) // rows.shape[0])
            tensors[f'layer.20.slot.{slot}'] = ['U8', [1, n - 1, width], np.tile(rows, (reps, 1))[: n - 1].tobytes()]
        tensors['tokens'] = ['U32', [n], np.asarray(tokens, np.uint32).tobytes()]
        for i in range(21):
            tensors[f'layer.{i}.slot.0'] = ['I32', [1], np.asarray([n - 1], np.int32).tobytes()]
        man['prompt_tokens'] = n
        man['token_sha256'] = token_digest(tokens)
        man['tail']['first_position'] = n - 1 - man['tail']['rows']
        man['bytes'] = sum(len(v[2]) for v in tensors.values())
    else:
        tokens = tokens0
    wire = {k: (d, sh, memoryview(b), len(b)) for k, (d, sh, b) in tensors.items()}
    return tokens, wire, man, s['identity']


def run_import(lm, tokens, wire, man, identity, marks=None):
    kw = {'marks': marks} if marks is not None else {}
    cache, rows = lm.import_state(wire, man, tokens, identity=identity, **kw)
    mx.eval([x for item in cache for x in item.cache if x is not None] + list(rows))
    return cache


def digest(lm, tokens, cache):
    """Same coverage as og_model._log_digest (layers 20-39 all slots except 20's 2/3, + DSpark prime keys)."""
    h = hashlib.sha256(np.asarray(tokens, np.uint32).tobytes())
    per = {}
    for i, item in enumerate(cache[20:], 20):
        hl = hashlib.sha256()
        for slot, x in enumerate(item.cache):
            if i == 20 and slot in (2, 3):
                continue
            if x is not None and x.size:
                b = np.array(x.view(mx.uint8) if x.dtype != mx.uint8 else x).tobytes()
                h.update(b); hl.update(b)
        per[i] = hl.hexdigest()[:8]
    ctx = getattr(cache[0], '_omlx_mtp_prime_ctx', None)
    for stage in (ctx.caches if ctx is not None else ()):
        if stage.keys is not None:
            b = np.array(stage.keys.view(mx.uint8)).tobytes()
            h.update(b)
            per.setdefault('prime', hashlib.sha256()).update(b) if False else None
    prime = hashlib.sha256(b''.join(np.array(s.keys.view(mx.uint8)).tobytes() for s in (ctx.caches if ctx else ())
                                    if s.keys is not None)).hexdigest()[:8]
    return h.hexdigest()[:16], per, prime


def drop(lm, cache):
    ctx = getattr(cache[0], '_omlx_mtp_prime_ctx', None)
    if ctx is not None:
        delattr(cache[0], '_omlx_mtp_prime_ctx')
    del cache
    mx.clear_cache()


def test_time(lm, spec, st):
    tokens, wire, man, ident = st
    plan = encoder_replay.segments(len(tokens) - 1, encoder_replay.CHUNK, lm._config.window_size)
    recs = []
    for rep in range(REPS + 1):
        marks = {}
        t0 = time.time()
        cache = run_import(lm, tokens, wire, man, ident, marks)
        t1 = time.time()
        seg = []
        prev = marks.get('og.import_arrays', t0)
        for k in range(len(plan)):
            a, b = marks.get('og.replay_layers%d' % k), marks.get('og.replay_prime%d' % k)
            seg.append(round(1000 * (a - prev), 1)); seg.append(round(1000 * (b - a), 1)); prev = b
        recs.append(dict(total=1000 * (t1 - t0), arrays=1000 * (marks['og.import_arrays'] - t0), seg=seg,
                         tail=round(1000 * (t1 - prev), 1)))
        drop(lm, cache)
        time.sleep(0.05)
    recs = recs[1:]
    emit(dict(kind='time', case=spec, plan=plan, total_ms=round(statistics.median(r['total'] for r in recs), 1),
              all=[round(r['total'], 1) for r in recs], arrays_ms=round(statistics.median(r['arrays'] for r in recs), 1),
              seg_layers_prime_ms=[round(statistics.median(r['seg'][j] for r in recs), 1) for j in range(len(recs[0]['seg']))],
              tail_ms=round(statistics.median(r['tail'] for r in recs), 1)))


def test_digest(lm, spec, st):
    tokens, wire, man, ident = st
    cache = run_import(lm, tokens, wire, man, ident)
    d, per, prime = digest(lm, tokens, cache)
    emit(dict(kind='digest', case=spec, state=d, prime=prime, layers=per))
    drop(lm, cache)


PARTS = {}


def instrument():
    """Per-layer / per-part wall time with a sync around each (attribution only; inflates totals)."""
    blk, attn, moe = language.Block.__call__, language.Attention.__call__, language.MoE.__call__

    def timed(fn, key):
        def inner(self, *a, **k):
            mx.eval([x for x in a if isinstance(x, mx.array)])
            t = time.perf_counter()
            out = fn(self, *a, **k)
            mx.eval(out)
            PARTS.setdefault(key(self, a, k), []).append(time.perf_counter() - t)
            return out
        return inner
    language.Attention.__call__ = timed(attn, lambda s, a, k: ('attn', s._layer, a[0].shape[1]))
    language.MoE.__call__ = timed(moe, lambda s, a, k: ('moe', a[0].shape[1]))
    language.Block.__call__ = timed(blk, lambda s, a, k: ('block', a[0].shape[1], k.get('hc_rows'), k.get('ced_tail')))
    return lambda: (setattr(language.Block, '__call__', blk), setattr(language.Attention, '__call__', attn),
                    setattr(language.MoE, '__call__', moe))


def test_parts(lm, spec, st):
    tokens, wire, man, ident = st
    PARTS.clear()
    undo = instrument()
    try:
        for rep in range(3):
            if rep == 1:
                PARTS.clear()
            cache = run_import(lm, tokens, wire, man, ident)
            drop(lm, cache)
    finally:
        undo()
    emit(dict(kind='parts', case=spec, parts={repr(k): dict(n=len(v) // 2, med_ms=round(1000 * statistics.median(v), 2),
                                                           sum_ms=round(1000 * sum(v) / 2, 1)) for k, v in PARTS.items()}))


def main():
    lm = load()
    for spec in CASES:
        st = case_state(spec)
        run_import(lm, *st[:3], st[3]) if False else None
        # warm (kernels, pools)
        drop(lm, run_import(lm, st[0], st[1], st[2], st[3]))
        for name in TESTS:
            globals()['test_' + name](lm, spec, st)
    emit(dict(event='done', peak_gib=round(mx.get_peak_memory() / 2**30, 2)))


main()
