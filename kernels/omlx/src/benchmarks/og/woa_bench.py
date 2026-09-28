"""Compact wo_a (woa_compact): identity and timing. Run through idle_guard.py only (production idle).

WB_MODE=forward  partial load (layers 20/24/25 aliased over 20-39 + head + DSpark, <= 35 GB), 2 real
                 8K box sessions. Flag off vs on, bitwise: logits, DSpark hidden, every layer 20-39 cache
                 slot + verify window for widths 1-5; fused pairs; DSpark single and batched drafts.
                 Saves the real wo_a inputs (widths 2-5, 20 layer positions + 3 DSpark stages) for
                 WB_MODE=layers. Then forward / pair / draft timing A/B (alternating, 3 warm + 9 kept).
WB_MODE=layers   no box, ~3 GB: every wo_a in the checkpoint (20 layers + 3 DSpark stages): GPU codes ==
                 numpy reference, compact == grouped_gemv == einsum on the saved real inputs and random
                 inputs; encode time; a 20-call chain over the 20 distinct weights vs 3 rotated
                 (WB_CHAIN_ROWS=2,3,4,5; wider rows = fused shapes, random input).
  python benchmarks/og/idle_guard.py $PY -u benchmarks/og/woa_bench.py
"""
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['DS41_TREE'] = str(ROOT)
os.environ.setdefault('MLX_ENABLE_TF32', '0')
MODE = os.environ.get('WB_MODE', 'forward')
OUT_DIR = Path.home()/'llm/ds41/woa'
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUTS = OUT_DIR/'real-inputs.safetensors'
_out = (OUT_DIR/f'{MODE}.jsonl').open('a')


def emit(**r):
    line = json.dumps(r)
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


def ab(fn_off, fn_on, reps=12, warm=3):
    """Alternating flag-off / flag-on wall times (ms, including graph build), medians of retained pairs."""
    from omlx.patches.deepseek_v41 import woa_compact
    import mlx.core as mx
    times = {'off': [], 'on': []}
    for rep in range(reps):
        for name in (['off', 'on'] if rep % 2 == 0 else ['on', 'off']):
            woa_compact.ENABLED = name == 'on'
            mx.synchronize()
            t = time.perf_counter()
            mx.eval(fn_off() if name == 'off' else fn_on())
            if rep >= warm:
                times[name].append((time.perf_counter() - t) * 1000)
    woa_compact.ENABLED = True
    off, on = statistics.median(times['off']), statistics.median(times['on'])
    return dict(off_ms=round(off, 3), on_ms=round(on, 3), saved_ms=round(off - on, 3),
                pair_saved_ms=sorted(round(a - b, 3) for a, b in zip(times['off'], times['on'])))


def forward_mode():
    os.environ['BB_STREAMS'] = '2'
    os.environ['BB_OUT'] = str(OUT_DIR/'bb.jsonl')
    import batch_bench as B
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import woa_compact
    mx.set_memory_limit(34_000_000_000)
    mx.set_cache_limit(128 << 20)
    t = time.perf_counter()
    lm = B.load()
    emit(event='loaded', s=round(time.perf_counter() - t, 1), active_GB=mx.get_active_memory() / 1e9)
    t = time.perf_counter()
    summary = woa_compact.install(lm)
    emit(event='install', s=round(time.perf_counter() - t, 3), **{k: v for k, v in summary.items() if k != 'event'})
    have = {i: woa_compact.codes_of(lm.layers[i].attn.wo_a) is not None for i in range(20, 40)}
    have.update({f'mtp{i}': woa_compact.codes_of(s.attn.wo_a) is not None for i, s in enumerate(lm.mtp)})
    assert all(have.values()), have

    calls = [0]
    kernel = woa_compact._gemv()

    def counted(*a, **k):
        calls[0] += 1
        return kernel(*a, **k)
    woa_compact._gemv = lambda: counted
    captured, capture = {}, [None]
    compact_gemv = woa_compact.grouped_gemv

    def capturing(linear, grouped, weight):
        if capture[0] is not None:
            key = f'{capture[0]}.w{grouped.shape[1]}.{len([k for k in captured if k.startswith(capture[0] + f".w{grouped.shape[1]}.")])}'
            captured[key] = grouped
        return compact_gemv(linear, grouped, weight)
    woa_compact.grouped_gemv = capturing

    def run(flag, fn):
        woa_compact.ENABLED = flag
        calls[0] = 0
        out = fn()
        mx.eval(out)
        return out, calls[0]

    def equal(a, b):
        return [bool(mx.array_equal(x, y).item()) for x, y in zip(a, b)] + [len(a) == len(b)]

    streams = []
    try:
        for i, toks in enumerate(B.prompts()):
            streams.append(B.Stream(lm, toks, f'woa-{i}'))
            for n in (1, 2, 3, 4, 5):
                streams[-1].boundary(n)
        emit(event='sessions', n=len(streams), open_s=[round(s.open_s, 2) for s in streams])

        # Singleton verify, widths 1-5, both streams.
        for k, s in enumerate(streams):
            for n in (1, 2, 3, 4, 5):
                res = {}
                for flag in (False, True):
                    def fwd():
                        s.reset()
                        capture[0] = f'target.s{k}' if flag and k == 0 else None
                        out = lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True)
                        st = B.state_arrays(s.cache)
                        return [*out, *st]
                    res[flag] = run(flag, fwd)
                capture[0] = None
                eq = equal(res[False][0], res[True][0])
                emit(test='forward_identity', stream=k, rows=n, arrays=len(res[True][0]),
                     equal=f'{sum(eq[:-1])}/{len(eq) - 1}', bitwise=all(eq), compact_calls=res[True][1])

        # Fused pairs (og_fused keeps the BF16 kernel).
        for rows in ((2, 2), (3, 3), (5, 5), (2, 5), (4, 3)):
            res = {}
            for flag in (False, True):
                def fused():
                    out = B.fused(lm, streams, rows)
                    return [x for pair in out for x in pair] + [x for s in streams for x in B.state_arrays(s.cache)]
                res[flag] = run(flag, fused)
            eq = equal(res[False][0], res[True][0])
            emit(test='fused_identity', rows=list(rows), equal=f'{sum(eq[:-1])}/{len(eq) - 1}', bitwise=all(eq),
                 compact_calls=res[True][1])

        # DSpark: single-stream drafts use the compact kernel, batched drafts (og_fused pairs) do not.
        base = []
        for s in streams:
            s.reset()
            logits, hidden = lm.forward_boundary(**s.boundary(5), cache=s.cache, start=s.start, verify=True)
            mx.eval(logits, hidden)
            s.reset()
            cache = lm.make_mtp_cache()
            lm.dspark_append_context(hidden, cache)
            mx.eval([c.keys for c in cache])
            base.append((hidden[:, :3], cache, mx.array([[s.tokens[-1]]], mx.uint32)))
        for width in (2, 3, 4):
            h, cache, anchor = base[0]
            res = {}
            for flag in (False, True):
                def draft():
                    c = [copy.copy(x) for x in cache]
                    capture[0] = 'draft' if flag and width == 4 else None
                    logits, hid = lm.dspark_forward(h, anchor, c, draft_length=width)
                    return [logits, hid] + [x.keys for x in c]
                res[flag] = run(flag, draft)
            capture[0] = None
            eq = equal(res[False][0], res[True][0])
            emit(test='draft_identity', width=width, equal=f'{sum(eq[:-1])}/{len(eq) - 1}', bitwise=all(eq),
                 compact_calls=res[True][1])
        for widths in ((4, 4), (2, 2)):
            res = {}
            for flag in (False, True):
                def batch():
                    cs = [[copy.copy(x) for x in b[1]] for b in base]
                    out = lm.dspark_forward_batch([b[0] for b in base], [b[2] for b in base], cs, list(widths))
                    return list(out) + [x.keys for c in cs for x in c]
                res[flag] = run(flag, batch)
            eq = equal(res[False][0], res[True][0])
            emit(test='draft_batch_identity', widths=list(widths), equal=f'{sum(eq[:-1])}/{len(eq) - 1}',
                 bitwise=all(eq), compact_calls=res[True][1])

        mx.save_safetensors(str(INPUTS), captured)
        emit(event='saved_inputs', n=len(captured), path=str(INPUTS))
        woa_compact.grouped_gemv = compact_gemv

        # Timing A/B: the flag switches between the BF16 and the compact kernel inside one process.
        s = streams[0]

        def fwd(n):
            def f():
                s.reset()
                return lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True)
            return f
        for n in (1, 2, 3, 4, 5):
            emit(test='forward_timing', rows=n, **ab(fwd(n), fwd(n)))
        for rows in ((3, 3), (5, 5)):
            f = lambda: [x for pair in B.fused(lm, streams, rows) for x in pair]
            emit(test='fused_timing', rows=list(rows), **ab(f, f))
        h, cache, anchor = base[0]
        f = lambda: lm.dspark_forward(h, anchor, [copy.copy(x) for x in cache], draft_length=4)[0]
        emit(test='draft_timing', width=4, **ab(f, f))
        emit(event='done', peak_GB=mx.get_peak_memory() / 1e9)
    finally:
        for s in streams:
            s.enc.close()


def layers_mode():
    import mlx.core as mx
    import numpy as np
    from omlx.patches.deepseek_v41 import decode_fusions as df, woa_compact
    from woa_scan import MODEL, read_bf16_bits, reference
    mx.set_memory_limit(8 << 30)
    mx.set_cache_limit(128 << 20)
    real = mx.load(str(INPUTS))
    mx.eval(real)
    by_width = {}
    for key, x in sorted(real.items()):
        by_width.setdefault(x.shape[1], []).append(x)
    emit(event='inputs', widths={k: len(v) for k, v in by_width.items()})
    mx.random.seed(41)
    rand = [(mx.random.normal((1, m, 8, 4096)) * scale).astype(mx.bfloat16)
            for m in (2, 3, 4, 5) for scale in (1.0, 64.0, 1 / 64)]
    mx.eval(rand)
    mapping = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    keys = sorted((k for k in mapping if k.endswith('attn.wo_a.weight')),
                  key=lambda k: (('mtp' in k), int(k.split('.')[2])))
    distinct = []
    for key in keys:
        bits = read_bf16_bits(MODEL/mapping[key], key)
        ref_codes, ref_base, _, _ = reference(bits)
        w = mx.array(bits).view(mx.bfloat16)
        mx.eval(w)
        mx.synchronize()
        t = time.perf_counter()
        codes = woa_compact.encode(w)
        mx.synchronize()
        encode_ms = (time.perf_counter() - t) * 1000
        codes_equal = codes.base == ref_base and bool(np.array_equal(np.array(codes.codes), ref_codes))
        linear = type('L', (), {})()
        linear.__dict__['weight'] = w
        linear.__dict__[woa_compact._ATTR] = codes
        w3 = w.reshape(8, 1024, 4096)
        n = bad = 0
        for x in [x for v in by_width.values() for x in v] + rand:
            a = df.grouped_gemv(x, w3)
            b = woa_compact.grouped_gemv(linear, x, w3)
            c = mx.einsum('bsgd,grd->bsgr', x, w3)
            mx.eval(a, b, c)
            n += 1
            bad += not (bool(mx.array_equal(a, b).item()) and bool(mx.array_equal(a, c).item()))
        emit(test='layer_identity', key=key, base=codes.base, escape_rate=codes.escape_rate,
             codes_equal_numpy=codes_equal, inputs=n, mismatches=bad, encode_ms=round(encode_ms, 2))
        if '.layers.' in key:
            distinct.append((w3, linear))
        else:
            del w, codes, linear
        mx.clear_cache()

    # 20 dependent calls: the 20 distinct layer weights, and 3 of them rotated (the partial harness).
    # Widths above 5 only occur fused (og_fused pairs, batched drafts): random input, kernel-only.
    for m in [int(r) for r in os.environ.get('WB_CHAIN_ROWS', '2,3,4,5').split(',')]:
        x = by_width[m][0] if m in by_width else mx.random.normal((1, m, 8, 4096)).astype(mx.bfloat16)
        if m not in by_width:
            w3, linear = distinct[0]
            a, b = df.grouped_gemv(x, w3), woa_compact.grouped_gemv(linear, x, w3)
            emit(test='wide_identity', rows=m, bitwise=bool(mx.array_equal(a, b).item()))
        for label, sets in (('20_distinct', distinct), ('3_rotated', [distinct[0], distinct[4], distinct[5]])):
            def chain(compact):
                xx = x
                for j in range(20):
                    w3, linear = sets[j % len(sets)]
                    y = woa_compact.grouped_gemv(linear, xx, w3) if compact else df.grouped_gemv(xx, w3)
                    xx = x + y[..., :1].astype(x.dtype) * mx.array(1e-4, x.dtype)
                return xx
            emit(test='chain_timing', rows=m, weights=label, **ab(lambda: chain(False), lambda: chain(True)))
    emit(event='done', peak_GB=mx.get_peak_memory() / 1e9)


if __name__ == '__main__':
    forward_mode() if MODE == 'forward' else layers_mode()
