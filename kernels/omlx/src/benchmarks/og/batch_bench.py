"""ds41-batch partial-load bench: fused multi-request Mac verify (og_fused) vs separate forward_boundary calls.

Loads layers 20/24/25 + head + norm (~28 GB; other layers alias a loaded block of the same kind, as
host_bench.py), opens N real box sessions (8K prompts, <= 4), then:
  numerics: per request, logits / DSpark hidden / every layer 20-39 cache slot and verify window after
            the fused pass vs its own forward_boundary (bitwise), several row mixes.
  timing:   separate (per request forward + eval, production c2 order) vs fused, medians.
Run only while production is idle (prof_idle.py), no gpu.lock.
  BB_TESTS=numerics,timing BB_STREAMS=4 BB_CTX=8192
"""
import json
import os
from pathlib import Path
import statistics
import sys
import time

HOME = Path.home()
TREE = Path(os.environ.get('DS41_TREE', str(HOME/'src/wt/ds41-batch')))
sys.path.insert(0, str(TREE))
for key, value in dict(MLX_ENABLE_TF32='0', DS41_NATIVE_VERIFY='0', DS41_MHC='1', DS41_GROWTH='1',
                       DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1', DS41_NATIVE_DECODE='1').items():
    os.environ.setdefault(key, value)

import mlx.core as mx  # noqa: E402

GIB = 1 << 30
mx.set_memory_limit(40 * GIB)
mx.set_cache_limit(512 << 20)

from omlx.patches.deepseek_v41 import pipe_decoder  # noqa: E402
from omlx.patches.deepseek_v41.config import ModelConfig  # noqa: E402
from omlx.patches.deepseek_v41.loading import _load_shard, set_module  # noqa: E402
from omlx.patches.deepseek_v41.quantization import QuantizedProjection  # noqa: E402
from omlx.patches.deepseek_v41.pipe_wire import EncoderSession, mlx_step  # noqa: E402

MODEL_DIR = HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
REPS = int(os.environ.get('BB_REPS', '15'))
TESTS = os.environ.get('BB_TESTS', 'numerics,timing').split(',')
N = int(os.environ.get('BB_STREAMS', '4'))
CTX = int(os.environ.get('BB_CTX', '8192'))
OUT = Path(os.environ.get('BB_OUT', str(HOME/'llm/ds41/batch/bench.jsonl')))
OUT.parent.mkdir(parents=True, exist_ok=True)
_out = OUT.open('a')
KIND = {20: 20, **{i: 24 for i in (24, 28, 32, 36)}}


def emit(record):
    line = json.dumps(record)
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


def load():
    raw = json.loads((MODEL_DIR/'config.json').read_text())
    config = ModelConfig.from_dict(raw)
    config.ced_prefill = True
    model = pipe_decoder.DecoderContainer(config)
    mapping = json.loads((MODEL_DIR/'model.safetensors.index.json').read_text())['weight_map']
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    keep = ('language_model.layers.20.', 'language_model.layers.24.', 'language_model.layers.25.',
            'language_model.head.', 'language_model.norm.', 'language_model.mtp.', 'language_model.embed.')
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
    lm.eval()
    return lm


def prompts():
    if CTX <= 8192:
        base = json.loads((HOME/'llm/ds41/split-wire/ids-131072.json').read_text())
        return [base[k * 8192:(k + 1) * 8192][:CTX] for k in range(N)]
    base = json.loads((HOME/'llm/ds41/split-wire/ids-524288.json').read_text())
    return [base[k * CTX:(k + 1) * CTX] for k in range(N)]


class Stream:
    def __init__(self, lm, tokens, tag):
        self.lm, self.tokens = lm, list(tokens)
        self.enc = EncoderSession()
        t = time.perf_counter()
        tensors, manifest = self.enc.open(self.tokens, tag, cache=True, state='lean', stream=True)
        self.cache, _ = lm.import_state(tensors, manifest, self.tokens, identity=self.enc.identity)
        mx.eval([x for item in self.cache for x in item.cache if x is not None])
        self.start = self.cache[0].size()
        self.open_s = time.perf_counter() - t
        self.snap = snapshot(self.cache)
        self.bounds = {}

    def boundary(self, rows):
        if rows not in self.bounds:
            ids = [self.tokens[-1]] + self.tokens[100:100 + rows - 1]
            self.enc.send_step(ids, self.start)
            raw, _ = self.enc.recv_step()
            self.bounds[rows] = mlx_step(raw, rows)
            mx.eval(list(self.bounds[rows].values()))
        return self.bounds[rows]

    def reset(self):
        restore(self.cache, self.snap)
        self.cache[0]._pipe1_verify = None


def snapshot(cache):
    return [(list(x.cache), x.left_padding, x.lengths) for x in cache]


def restore(cache, snap):
    for item, (lst, lp, ln) in zip(cache, snap):
        item.cache = list(lst)
        item.left_padding, item.lengths = lp, ln


def state_arrays(cache):
    """Every Mac-side array a later step reads: layer 20-39 slots + verify windows."""
    out = []
    stash = cache[0]._pipe1_verify
    for i in range(20, 40):
        out += [x for x in cache[i].cache if x is not None]
        out.append(stash[3][i].get('window'))
    return [x for x in out if x is not None]


def separate(lm, streams, rows, evaluate=True):
    res = []
    for s, n in zip(streams, rows):
        s.reset()
        logits, hidden = lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True)
        if evaluate:
            mx.eval(logits, hidden)
        res.append((logits, hidden))
    return res


def fused(lm, streams, rows):
    items = []
    for s, n in zip(streams, rows):
        s.reset()
        items.append(dict(**s.boundary(n), cache=s.cache, start=s.start))
    return lm.forward_boundaries(items)


def test_numerics(lm, streams):
    from omlx.patches.deepseek_v41 import og_fused
    for rows in ((5, 5), (2, 5), (3, 4), (2, 2), (5, 5, 5), (4, 5, 3, 2)):
        if len(rows) > len(streams):
            continue
        group = streams[:len(rows)]
        assert og_fused.eligible(lm, list(rows))
        ref = []
        for s, (logits, hidden) in zip(group, separate(lm, group, rows)):
            st = state_arrays(s.cache)
            mx.eval(st)
            ref.append((logits, hidden, st))
        out = fused(lm, group, rows)
        rec = dict(test='numerics', rows=list(rows), streams=[])
        for s, (logits, hidden), (l0, h0, st0) in zip(group, out, ref):
            st = state_arrays(s.cache)
            mx.eval(logits, hidden, st)
            same = [bool(mx.array_equal(a, b).item()) for a, b in zip(st0, st)]
            diff = float(mx.abs(logits - l0).max().item())
            rec['streams'].append(dict(logits_equal=bool(mx.array_equal(logits, l0).item()), logits_maxdiff=diff,
                                       hidden_equal=bool(mx.array_equal(hidden, h0).item()),
                                       state_equal=f'{sum(same)}/{len(same)}', n_state=len(st) == len(st0)))
        rec['bitwise'] = all(x['logits_equal'] and x['hidden_equal'] and x['state_equal'].split('/')[0] == x['state_equal'].split('/')[1]
                             for x in rec['streams'])
        emit(rec)


def test_draft(lm, streams):
    """DSpark: dspark_forward_batch per request == its own dspark_forward (logits + context caches, bitwise)."""
    import copy
    base = []
    for s in streams:
        s.reset()
        logits, hidden = lm.forward_boundary(**s.boundary(5), cache=s.cache, start=s.start, verify=True)
        mx.eval(logits, hidden)
        s.reset()
        cache = lm.make_mtp_cache()
        lm.dspark_append_context(hidden, cache)  # some committed history first
        mx.eval([x for c in cache for x in (c.keys,) if x is not None])
        base.append((hidden[:, :3], cache, mx.array([[s.tokens[-1]]], mx.uint32)))
    for widths in ((4, 4), (2, 2), (3, 3), (2, 2, 2)):
        if len(widths) > len(base):
            continue
        group = base[:len(widths)]
        solo, solo_caches = [], []
        for hidden, cache, anchor in group:
            c = [copy.copy(x) for x in cache]
            logits, _ = lm.dspark_forward(hidden, anchor, c, draft_length=widths[0])
            mx.eval(logits)
            solo.append(logits)
            solo_caches.append(c)
        caches = [[copy.copy(x) for x in cache] for _, cache, _ in group]
        out = lm.dspark_forward_batch([g[0] for g in group], [g[2] for g in group], caches, list(widths))
        assert out is not None, widths
        mx.eval(out)
        eq = [bool(mx.array_equal(a, b).item()) for a, b in zip(solo, out)]
        keys = [all(bool(mx.array_equal(x.keys, y.keys).item()) and x.offset == y.offset for x, y in zip(c1, c2))
                for c1, c2 in zip(solo_caches, caches)]
        emit(dict(test='draft', widths=list(widths), logits_equal=eq, caches_equal=keys))
        solo_ms = timed(lambda: [lm.dspark_forward(h, a, [copy.copy(x) for x in c], draft_length=widths[0])[0]
                                 for h, c, a in group])
        batch_ms = timed(lambda: lm.dspark_forward_batch([g[0] for g in group], [g[2] for g in group],
                                                         [[copy.copy(x) for x in g[1]] for g in group], list(widths)))
        emit(dict(test='draft_timing', widths=list(widths), separate_ms=solo_ms, batched_ms=batch_ms))


def timed(fn, reps=REPS, warm=3):
    ts = []
    for r in range(warm + reps):
        mx.synchronize()
        t = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append(time.perf_counter() - t)
    return round(1000 * statistics.median(ts[warm:]), 2)


def test_timing(lm, streams):
    for rows in ((5,), (5, 5), (4, 4), (3, 3), (5, 5, 5), (5, 5, 5, 5)):
        if len(rows) > len(streams):
            continue
        group = streams[:len(rows)]
        for s, n in zip(group, rows):
            s.boundary(n)
        sep = timed(lambda: [x for pair in separate(lm, group, rows) for x in pair])
        rec = dict(test='timing', ctx=CTX, rows=list(rows), separate_ms=sep)
        if len(rows) > 1:
            fu = timed(lambda: [x for pair in fused(lm, group, rows) for x in pair])
            rec.update(fused_ms=fu, saving_pct=round(100 * (1 - fu / sep), 1))
        emit(rec)


def main():
    t = time.perf_counter()
    lm = load()
    emit(dict(event='loaded', s=round(time.perf_counter() - t, 1), active_gib=round(mx.get_active_memory() / GIB, 2)))
    streams = []
    try:
        for k, tokens in enumerate(prompts()):
            streams.append(Stream(lm, tokens, f'batch-bench-{k}'))
        emit(dict(event='opened', n=len(streams), ctx=CTX, open_s=[round(s.open_s, 2) for s in streams]))
        if 'numerics' in TESTS:
            test_numerics(lm, streams)
        if 'draft' in TESTS:
            test_draft(lm, streams)
        if 'timing' in TESTS:
            test_timing(lm, streams)
    finally:
        for s in streams:
            s.enc.close()


if __name__ == '__main__':
    main()
