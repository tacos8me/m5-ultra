"""ds41-host partial-load bench: hc fusion numerics + Mac forward timing (speed lane, never serves).

Loads layers 20/24/25 + head + norm (~28 GB; the other layers alias a loaded block of the same kind, as
benchmarks/og/prof_mac.py partial mode), opens one lean 8K box session for real boundaries, then:
  numerics: every Block output (h, pre) and the logits, DS41_HC_FUSE off vs on, rows 1..5 (bitwise).
  timing:   forward_boundary build/wait/total, off vs on (fused hc + early submit), rows 1..5.
  hcmicro:  20 chained hc stages, unfused vs fused (per-layer us, bitwise check).
  host:     host timeline of one forward (Python vs time inside async_eval, first submit).
  gap:      forward after a 10 ms GPU idle: none / sleep / spin / spin + GPU keep-warm.
Run only while production is idle (benchmarks/og/prof_idle.py), no gpu.lock.
"""
import json
import os
from pathlib import Path
import statistics
import sys
import time

HOME = Path.home()
TREE = Path(os.environ.get('DS41_TREE', str(HOME/'src/wt/ds41-host')))
sys.path.insert(0, str(TREE))
for key, value in dict(MLX_ENABLE_TF32='0', DS41_NATIVE_VERIFY='0', DS41_MHC='1', DS41_GROWTH='1',
                       DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1', DS41_NATIVE_DECODE='1').items():
    os.environ.setdefault(key, value)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

GIB = 1 << 30
mx.set_memory_limit(40 * GIB)
mx.set_cache_limit(512 << 20)

from omlx.patches.deepseek_v41 import pipe_decoder, language, hc_fuse, pipe_wire  # noqa: E402
from omlx.patches.deepseek_v41.config import ModelConfig  # noqa: E402
from omlx.patches.deepseek_v41.loading import _load_shard, set_module  # noqa: E402
from omlx.patches.deepseek_v41.quantization import QuantizedProjection  # noqa: E402

MODEL_DIR = HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
REPS = int(os.environ.get('HB_REPS', '15'))
TESTS = os.environ.get('HB_TESTS', 'numerics,timing').split(',')
OUT = Path(os.environ.get('HB_OUT', str(HOME/'llm/ds41/prof/host.jsonl')))
_out = OUT.open('a')


def emit(record):
    line = json.dumps(record)
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


KIND = {20: 20, **{i: 24 for i in (24, 28, 32, 36)}}


def load():
    raw = json.loads((MODEL_DIR/'config.json').read_text())
    config = ModelConfig.from_dict(raw)
    config.ced_prefill = True
    model = pipe_decoder.DecoderContainer(config)
    mapping = json.loads((MODEL_DIR/'model.safetensors.index.json').read_text())['weight_map']
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    keep = ('language_model.layers.20.', 'language_model.layers.24.', 'language_model.layers.25.',
            'language_model.head.', 'language_model.norm.')
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


def snapshot(cache):
    return [(list(x.cache), x.left_padding, x.lengths) for x in cache]


def restore(cache, snap):
    for item, (lst, lp, ln) in zip(cache, snap):
        item.cache = list(lst)
        item.left_padding, item.lengths = lp, ln


CAPTURE = [None]
_block_call = language.Block.__call__


ENTRY = []


def _capture_call(self, *a, **kw):
    ENTRY.append(time.perf_counter())
    out = _block_call(self, *a, **kw)
    if CAPTURE[0] is not None:
        CAPTURE[0].append(out)
    return out


language.Block.__call__ = _capture_call


def forward(lm, cache, b, start):
    logits, hidden = lm.forward_boundary(**b, cache=cache, start=start, verify=True)
    return logits, hidden


def test_numerics(lm, cache, bounds, start):
    snap = snapshot(cache)
    for rows, b in bounds.items():
        outs = {}
        for fuse, early in ((False, False), (True, False), (True, True)):
            hc_fuse.DS41_HC_FUSE = fuse
            hc_fuse.EARLY_SUBMIT = early
            restore(cache, snap)
            CAPTURE[0] = []
            logits, hidden = forward(lm, cache, b, start)
            layers = CAPTURE[0]
            CAPTURE[0] = None
            mx.eval(logits, hidden, layers)
            outs[fuse] = (logits, hidden, layers)
            cache[0]._pipe1_verify = None
        (l0, d0, y0), (l1, d1, y1) = outs[False], outs[True]
        worst = dict(h_abs=0.0, h_rel=0.0, pre_abs=0.0)
        diff_layers = []
        for i, ((h0, p0), (h1, p1)) in enumerate(zip(y0, y1)):
            same = bool(mx.array_equal(h0, h1).item()) and bool(mx.array_equal(p0, p1).item())
            if not same:
                a0, a1 = h0.astype(mx.float32), h1.astype(mx.float32)
                d = mx.abs(a0 - a1)
                worst['h_abs'] = max(worst['h_abs'], float(d.max().item()))
                worst['h_rel'] = max(worst['h_rel'], float((d / (mx.abs(a0) + 1e-6)).max().item()))
                worst['pre_abs'] = max(worst['pre_abs'], float(mx.abs(p0 - p1).max().item()))
                diff_layers.append(20 + i)
        emit(dict(test='numerics', rows=rows, layers=len(y0), bitwise_layers=len(y0) - len(diff_layers),
                  diff_layers=diff_layers, logits_equal=bool(mx.array_equal(l0, l1).item()),
                  hidden_equal=bool(mx.array_equal(d0, d1).item()),
                  argmax_equal=bool(mx.array_equal(mx.argmax(l0, -1), mx.argmax(l1, -1)).item()), **worst))
    restore(cache, snap)


def time_forward(lm, cache, b, start, reps):
    snap = snapshot(cache)
    recs = []
    for i in range(reps + 3):
        restore(cache, snap)
        t0 = time.perf_counter()
        logits, hidden = forward(lm, cache, b, start)
        out = mx.argmax(logits[0], -1)
        t1 = time.perf_counter()
        mx.eval(out, hidden)
        t2 = time.perf_counter()
        cache[0]._pipe1_verify = None
        if i >= 3:
            recs.append((t1 - t0, t2 - t1, t2 - t0))
    restore(cache, snap)
    med = lambda j: round(1000 * statistics.median(r[j] for r in recs), 3)
    return dict(build_ms=med(0), wait_ms=med(1), total_ms=med(2),
                total_p10_ms=round(1000 * sorted(r[2] for r in recs)[len(recs) // 10], 3))


def test_host(lm, cache, bounds, start):
    """Host timeline of one forward: Python build between submits vs time inside each async_eval."""
    real = mx.async_eval
    marks = []

    def timed(*a):
        t = time.perf_counter()
        real(*a)
        marks.append((t, time.perf_counter()))
    mx.async_eval = timed
    try:
        for fuse, early in ((False, False), (True, False), (True, True)):
            hc_fuse.DS41_HC_FUSE = fuse
            hc_fuse.EARLY_SUBMIT = early
            for rows in (1, 5):
                b = bounds[rows]
                snap = snapshot(cache)
                runs = []
                for i in range(14):
                    restore(cache, snap)
                    marks.clear()
                    ENTRY.clear()
                    t0 = time.perf_counter()
                    logits, hidden = forward(lm, cache, b, start)
                    out = mx.argmax(logits[0], -1)
                    t1 = time.perf_counter()
                    mx.eval(out, hidden)
                    t2 = time.perf_counter()
                    cache[0]._pipe1_verify = None
                    inside = sum(e - s for s, e in marks)
                    runs.append(dict(prelude_ms=1000 * (ENTRY[0] - t0), l20_python_ms=1000 * (marks[0][0] - ENTRY[0]), first_submit_ms=1000 * (marks[0][0] - t0), first_return_ms=1000 * (marks[0][1] - t0),
                                     in_async_ms=1000 * inside, python_ms=1000 * (t1 - t0 - inside),
                                     per_submit_ms=[round(1000 * (e - s), 3) for s, e in marks],
                                     wait_ms=1000 * (t2 - t1), total_ms=1000 * (t2 - t0)))
                restore(cache, snap)
                r = runs[-1]
                emit(dict(test='host', fuse=fuse, early=early, rows=rows, **{k: round(statistics.median(x[k] for x in runs[2:]), 3)
                          for k in r if k != 'per_submit_ms'}, per_submit_ms=r['per_submit_ms']))
    finally:
        mx.async_eval = real


def hc_chain(block, c, h, pre, fused, steps=20):
    for _ in range(steps):
        if fused:
            mix_a, x = hc_fuse.project_pre_norm(h, pre, block.hc_attn_fn, block.attn_norm.weight, c.norm_eps, block.attn_norm.eps)
            h2, ap = hc_fuse.post_mix(x, h, mix_a, block.hc_attn_scale, block.hc_attn_base, c.hc_eps, c.hc_sinkhorn_iters)
            mix_f, xf = hc_fuse.project_pre_norm(h2, ap, block.hc_ffn_fn, block.ffn_norm.weight, c.norm_eps, block.ffn_norm.eps)
            h, pre = hc_fuse.post_mix(xf, h2, mix_f, block.hc_ffn_scale, block.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters)
        else:
            ap, ao, ac = language.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base, c)
            x = language.hc_pre_norm(h, pre, block.attn_norm.weight, block.attn_norm.eps)
            h2 = language.hc_post(x, h, ao, ac)
            fp, fo, fc = language.hc_mixes(h2, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base, c)
            xf = language.hc_pre_norm(h2, ap, block.ffn_norm.weight, block.ffn_norm.eps)
            h, pre = language.hc_post(xf, h2, fo, fc), fp
    return h, pre


def test_hcmicro(lm, bounds):
    """20 chained hc stages (attn + ffn sublayer, the sublayer itself replaced by its normed input)."""
    c = lm._config
    block = lm.layers[25]
    for rows in (1, 5):
        h0, p0 = bounds[rows]['h'], bounds[rows]['pre']
        outs = {}
        for fused in (False, True, False, True):
            ts = []
            for i in range(12):
                t = time.perf_counter()
                out = hc_chain(block, c, h0, p0, fused)
                mx.eval(out)
                ts.append(time.perf_counter() - t)
            outs[fused] = out
            emit(dict(test='hcmicro', rows=rows, fused=fused, ms_per_layer=round(1000 * statistics.median(ts[2:]) / 20, 4)))
        emit(dict(test='hcmicro_equal', rows=rows, h=bool(mx.array_equal(outs[False][0], outs[True][0]).item()),
                  pre=bool(mx.array_equal(outs[False][1], outs[True][1]).item())))


WARM = {}
WARM_STREAM = mx.new_stream(mx.gpu)


def idle(kind, seconds):
    end = time.perf_counter() + seconds
    if kind == 'sleep':
        time.sleep(seconds)
    elif kind == 'spin':
        while time.perf_counter() < end:
            pass
    elif kind.startswith('warm'):
        # spin + keep the GPU busy with small independent kernels (never waited on).
        # warm<size>i<interval_us>[s]: s = on a separate stream
        spec = kind[4:]
        stream = spec.endswith('s')
        spec = spec.rstrip('s')
        size, _, interval = spec.partition('i')
        size = int(size or 1 << 20)
        interval = float(interval or 500) / 1e6
        x = WARM.get(size)
        if x is None:
            x = WARM[size] = mx.zeros((size,), mx.float32)
            mx.eval(x)
        nxt = 0.0
        while time.perf_counter() < end:
            now = time.perf_counter()
            if now >= nxt:
                if stream:
                    with mx.stream(WARM_STREAM):
                        mx.async_eval(x + 1.0)
                else:
                    mx.async_eval(x + 1.0)
                nxt = now + interval


def test_gap(lm, cache, bounds, start):
    kinds = os.environ.get('HB_GAPS', 'none,sleep,spin,warm').split(',')
    gap = float(os.environ.get('HB_GAP_MS', '10')) / 1000
    for rows in (1, 5):
        b = bounds[rows]
        snap = snapshot(cache)
        for kind in kinds:
            recs = []
            for i in range(REPS + 3):
                restore(cache, snap)
                if kind != 'none':
                    idle(kind, gap)
                t0 = time.perf_counter()
                logits, hidden = forward(lm, cache, b, start)
                out = mx.argmax(logits[0], -1)
                mx.eval(out, hidden)
                t2 = time.perf_counter()
                cache[0]._pipe1_verify = None
                if i >= 3:
                    recs.append(t2 - t0)
            emit(dict(test='gap', rows=rows, kind=kind, gap_ms=gap * 1000, total_ms=round(1000 * statistics.median(recs), 3)))
        restore(cache, snap)


def test_timing(lm, cache, bounds, start):
    variants = os.environ.get('HB_VARIANTS', 'off,on,off,on').split(',')
    for v in variants:
        hc_fuse.DS41_HC_FUSE = hc_fuse.EARLY_SUBMIT = v == 'on'
        for rows, b in bounds.items():
            emit(dict(test='fwd', variant=v, rows=rows, **time_forward(lm, cache, b, start, REPS)))


def main():
    t = time.perf_counter()
    lm = load()
    emit(dict(event='loaded', s=round(time.perf_counter() - t, 1), active_gib=round(mx.get_active_memory() / GIB, 2)))
    tokens = json.loads((HOME/'llm/ds41/split-wire/ids-8192.json').read_text())
    enc = pipe_wire.EncoderSession()
    try:
        tensors, manifest = enc.open(tokens, 'host-bench', cache=True, state='lean', stream=True)
        cache, _ = lm.import_state(tensors, manifest, tokens, identity=enc.identity)
        mx.eval([x for item in cache for x in item.cache if x is not None])
        del tensors
        start = cache[0].size()
        bounds = {}
        for rows in (1, 2, 3, 4, 5):
            ids = [tokens[-1]] + tokens[100:100 + rows - 1]
            enc.send_step(ids, start)
            raw, timing = enc.recv_step()
            bounds[rows] = pipe_wire.mlx_step(raw, rows)
        emit(dict(event='session', start=start, active_gib=round(mx.get_active_memory() / GIB, 2)))
    finally:
        enc.close()
    if 'numerics' in TESTS:
        test_numerics(lm, cache, bounds, start)
    if 'hcmicro' in TESTS:
        test_hcmicro(lm, bounds)
    if 'gap' in TESTS:
        test_gap(lm, cache, bounds, start)
    if 'host' in TESTS:
        test_host(lm, cache, bounds, start)
    if 'timing' in TESTS:
        test_timing(lm, cache, bounds, start)
    emit(dict(event='done', peak_gib=round(mx.get_peak_memory() / GIB, 2)))


if __name__ == '__main__':
    main()
