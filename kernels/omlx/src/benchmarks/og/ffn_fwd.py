"""ffn_fuse in the real-boundary partial forward: identity + timing. Run through idle_guard.py only.

Partial load (layers 20/24/25 aliased over 20-39 + head + DSpark, <= 35 GB), 2 real 8K box sessions.
  identity: DS41_FFN_FUSE off vs on, bitwise: logits, DSpark hidden, every layer 20-39 cache slot + verify
            window, widths 1-5 per stream; og_fused pairs.
  depth:    (flag on) row-prefix invariance across verify widths 2-5: the box boundary rows, the Mac logits
            rows and, after rollback_boundary(keep), every committed cache array and the next step's logits
            are bitwise the same whichever width verified them (depth choices cannot change outputs).
  timing:   forward widths 1-5 and fused pairs, flag off/on alternating (3 warm + FW_REPS kept).
FW_TESTS=identity,depth,timing  FW_REPS=12
"""
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
OUT_DIR = Path.home()/'llm/ds41/ffn'
OUT_DIR.mkdir(parents=True, exist_ok=True)
_out = (OUT_DIR/'forward.jsonl').open('a')
TESTS = os.environ.get('FW_TESTS', 'identity,depth,timing').split(',')


def emit(**r):
    line = json.dumps(r)
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


_KCOUNT = {}


def _count_kernels():
    import mlx.core as mx
    make = mx.fast.metal_kernel

    def wrapped(*a, **k):
        kern = make(*a, **k)
        name = k.get('name', a[0] if a else '?')

        def call(*ca, **ck):
            _KCOUNT[name] = _KCOUNT.get(name, 0) + 1
            return kern(*ca, **ck)
        return call
    mx.fast.metal_kernel = wrapped


def main():
    if 'kcount' in TESTS or 'kdraft' in TESTS:
        _count_kernels()
    os.environ['BB_STREAMS'] = '2'
    os.environ['BB_OUT'] = str(OUT_DIR/'bb.jsonl')
    import batch_bench as B
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import attn_in, attn_out, ffn_fuse, woa_compact
    mx.set_memory_limit(34_000_000_000)
    mx.set_cache_limit(128 << 20)
    t = time.perf_counter()
    lm = B.load()
    emit(event='loaded', s=round(time.perf_counter() - t, 1), active_GB=round(mx.get_active_memory() / 1e9, 2))
    if os.environ.get('FW_WOA', '1') == '1':  # production parity: og_model.load installs the compact wo_a
        summary = woa_compact.install(lm)
        emit(event='woa_compact', encoded=summary['encoded'])

    calls = [0]
    fwd_fn = ffn_fuse.ffn_forward

    def counted(*a, **k):
        calls[0] += 1
        return fwd_fn(*a, **k)
    ffn_fuse.ffn_forward = counted

    def setflags(flag):
        # False = production path; True = ffn_fuse + attn_in + attn_out; 'ffn' = ffn_fuse only.
        ffn_fuse.ENABLED = bool(flag)
        attn_in.ENABLED = flag in (True, 'noout')
        attn_out.ENABLED = flag is True

    def run(flag, fn):
        setflags(flag)
        calls[0] = 0
        out = fn()
        mx.eval(out)
        return out, calls[0]

    def equal(a, b):
        return [bool(mx.array_equal(x, y).item()) for x, y in zip(a, b)] + [len(a) == len(b)]

    def cache_arrays(cache):
        return [x for i in range(20, 40) for x in cache[i].cache if x is not None]

    streams = []
    try:
        for i, toks in enumerate(B.prompts()):
            streams.append(B.Stream(lm, toks, f'ffn-{i}'))
            for n in (1, 2, 3, 4, 5):
                streams[-1].boundary(n)
        emit(event='sessions', n=len(streams), open_s=[round(s.open_s, 2) for s in streams])

        if 'identity' in TESTS:
            for k, s in enumerate(streams):
                for n in (1, 2, 3, 4, 5):
                    res = {}
                    for flag in (False, True):
                        def fwd():
                            s.reset()
                            out = lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True)
                            return [*out, *B.state_arrays(s.cache)]
                        res[flag] = run(flag, fwd)
                    eq = equal(res[False][0], res[True][0])
                    emit(test='forward_identity', stream=k, rows=n, arrays=len(res[True][0]),
                         equal=f'{sum(eq[:-1])}/{len(eq) - 1}', bitwise=all(eq), fused_calls=res[True][1])
            for rows in ((2, 2), (3, 3), (5, 5), (2, 5), (4, 3)):
                res = {}
                for flag in (False, True):
                    def fused():
                        out = B.fused(lm, streams, rows)
                        return [x for pair in out for x in pair] + [x for s in streams for x in B.state_arrays(s.cache)]
                    res[flag] = run(flag, fused)
                eq = equal(res[False][0], res[True][0])
                emit(test='fused_identity', rows=list(rows), equal=f'{sum(eq[:-1])}/{len(eq) - 1}', bitwise=all(eq),
                     fused_calls=res[True][1])

        if 'draft' in TESTS:
            import copy
            base = []
            setflags(False)
            for s_ in streams:
                s_.reset()
                logits, hidden = lm.forward_boundary(**s_.boundary(5), cache=s_.cache, start=s_.start, verify=True)
                mx.eval(logits, hidden)
                s_.reset()
                cache = lm.make_mtp_cache()
                lm.dspark_append_context(hidden, cache)
                mx.eval([c.keys for c in cache])
                base.append((hidden[:, :3], cache, mx.array([[s_.tokens[-1]]], mx.uint32)))
            for width in (2, 3, 4):
                h, cache, anchor = base[0]
                res = {}
                for flag in (False, True):
                    def draft():
                        c = [copy.copy(x) for x in cache]
                        logits, hid = lm.dspark_forward(h, anchor, c, draft_length=width)
                        return [logits, hid] + [x.keys for x in c]
                    res[flag] = run(flag, draft)
                eq = equal(res[False][0], res[True][0])
                emit(test='draft_identity', width=width, equal=f'{sum(eq[:-1])}/{len(eq) - 1}', bitwise=all(eq))
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
                     bitwise=all(eq))
            draft_base = base
        if 'depth' in TESTS:
            depth_flag = os.environ.get('FW_DEPTH_FLAGS', 'on') == 'on'
            setflags(depth_flag)
            for k, s in enumerate(streams):
                ref = {}
                s.reset()
                b5 = s.boundary(5)
                logits5, _ = lm.forward_boundary(**b5, cache=s.cache, start=s.start, verify=True)
                mx.eval(logits5)
                s.reset()
                for n in (2, 3, 4, 5):
                    bn = s.boundary(n)
                    rows_same = all(bool(mx.array_equal(bn[key][:, :n], b5[key][:, :n]).item())
                                    for key in bn if bn[key] is not None and bn[key].ndim >= 2)
                    for keep in range(1, n + 1):
                        s.reset()
                        logits, _ = lm.forward_boundary(**bn, cache=s.cache, start=s.start, verify=True)
                        lm.rollback_boundary(s.cache, keep)
                        state = cache_arrays(s.cache)
                        # One more step on the committed prefix: the next verify must not depend on the width.
                        nxt = lm.forward_boundary(**{key: (v[:, keep:keep + 1] if v is not None and v.ndim >= 2 else v)
                                                     for key, v in b5.items()},
                                                  cache=s.cache, start=s.start + keep, verify=True)[0] if keep < 5 else None
                        mx.eval(logits, state, *([nxt] if nxt is not None else []))
                        key = keep
                        if key not in ref:
                            ref[key] = (logits[:, :keep], state, nxt, n)
                            continue
                        l0, st0, nx0, n0 = ref[key]
                        same_logits = bool(mx.array_equal(logits[:, :keep], l0).item())
                        eq = equal(st0, state)
                        same_next = nxt is None or bool(mx.array_equal(nxt, nx0).item())
                        emit(test='depth_invariance', flags='on' if depth_flag else 'off', stream=k, width=n, ref_width=n0, keep=keep,
                             boundary_rows_equal=rows_same, logits_equal=same_logits,
                             state_equal=f'{sum(eq[:-1])}/{len(eq) - 1}', next_step_equal=same_next,
                             bitwise=same_logits and all(eq) and same_next and rows_same)
                s.reset()
                s.cache[0]._pipe1_verify = None

        if 'dot' in TESTS:
            from omlx.patches.deepseek_v41 import hc_fuse
            saved = (lm._async_every, hc_fuse.EARLY_SUBMIT)
            lm._async_every, hc_fuse.EARLY_SUBMIT = 0, False
            for flag, tag in ((False, 'off'), (True, 'on')):
                setflags(flag)
                for n in (1, 5):
                    s = streams[0]
                    s.reset()
                    logits, hidden = lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True)
                    path = OUT_DIR/f'graph-{tag}-L{n}.dot'
                    mx.export_to_dot(str(path), logits, hidden, *B.state_arrays(s.cache))
                    mx.eval(logits, hidden)
                    emit(test='dot', flag=tag, rows=n, path=str(path))
            s.reset()
            lm._async_every, hc_fuse.EARLY_SUBMIT = saved
        if 'kdraft' in TESTS:
            import copy
            s = streams[0]
            s.reset()
            logits, hidden = lm.forward_boundary(**s.boundary(5), cache=s.cache, start=s.start, verify=True)
            mx.eval(logits, hidden)
            s.reset()
            cache = lm.make_mtp_cache()
            lm.dspark_append_context(hidden, cache)
            mx.eval([c.keys for c in cache])
            h, anchor = hidden[:, :3], mx.array([[s.tokens[-1]]], mx.uint32)
            for width in (4,):
                _KCOUNT.clear()
                out = lm.dspark_forward(h, anchor, [copy.copy(x) for x in cache], draft_length=width)
                mx.eval(out)
                emit(test='kdraft', width=width, total=sum(_KCOUNT.values()),
                     kernels=dict(sorted(_KCOUNT.items(), key=lambda kv: -kv[1])))
                path = OUT_DIR/f'graph-draft-w{width}.dot'
                from omlx.patches.deepseek_v41 import dspark as dsp
                saved = {k: getattr(dsp, k) for k in dir(dsp) if k.endswith('ASYNC') and isinstance(getattr(dsp, k), bool)}
                for k in saved:
                    setattr(dsp, k, False)
                out = lm.dspark_forward(h, anchor, [copy.copy(x) for x in cache], draft_length=width)
                mx.export_to_dot(str(path), *[o for o in out if o is not None])
                mx.eval(out)
                for k, v in saved.items():
                    setattr(dsp, k, v)
                emit(test='draft_async_flags', flags=list(saved))
                times = []
                for rep in range(15):
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval(lm.dspark_forward(h, anchor, [copy.copy(x) for x in cache], draft_length=width)[0])
                    times.append((time.perf_counter() - t0) * 1000)
                emit(test='draft_timing', width=width, ms=round(statistics.median(times[3:]), 3))
        if 'kcount' in TESTS:
            for flag, tag in ((False, 'off'), (True, 'on')):
                setflags(flag)
                for n in (5, 2):
                    s = streams[0]
                    s.reset()
                    _KCOUNT.clear()
                    mx.eval(lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True))
                    emit(test='kcount', flag=tag, rows=n, total=sum(_KCOUNT.values()),
                         kernels=dict(sorted(_KCOUNT.items(), key=lambda kv: -kv[1])))
            s.reset()
        if 'timing' in TESTS:
            reps, warm = int(os.environ.get('FW_REPS', '12')), 3

            def ab(fn):
                order = [False, 'ffn', 'noout', True]
                times = {k: [] for k in order}
                for rep in range(reps + warm):
                    for flag in (order if rep % 2 == 0 else order[::-1]):
                        setflags(flag)
                        mx.synchronize()
                        t0 = time.perf_counter()
                        mx.eval(fn())
                        if rep >= warm:
                            times[flag].append((time.perf_counter() - t0) * 1000)
                setflags(True)
                off, ffn, noout, on = (statistics.median(times[k]) for k in order)
                return dict(off_ms=round(off, 3), ffn_ms=round(ffn, 3), noout_ms=round(noout, 3), on_ms=round(on, 3),
                            saved_ffn_ms=round(off - ffn, 3), saved_ms=round(off - on, 3),
                            pair_saved=sorted(round(a - b, 3) for a, b in zip(times[False], times[True])))
            s = streams[0]
            for n in (1, 2, 3, 4, 5):
                def f(n=n):
                    s.reset()
                    return lm.forward_boundary(**s.boundary(n), cache=s.cache, start=s.start, verify=True)
                emit(test='forward_timing', rows=n, **ab(f))
            for rows in ((3, 3), (5, 5)):
                emit(test='fused_timing', rows=list(rows),
                     **ab(lambda rows=rows: [x for pair in B.fused(lm, streams, rows) for x in pair]))
            if 'draft' in TESTS:
                import copy
                h, cache, anchor = draft_base[0]
                emit(test='draft_timing', width=4, **ab(
                    lambda: lm.dspark_forward(h, anchor, [copy.copy(x) for x in cache], draft_length=4)[0]))
                emit(test='draft_batch_timing', widths=[4, 4], **ab(lambda: lm.dspark_forward_batch(
                    [b[0] for b in draft_base], [b[2] for b in draft_base],
                    [[copy.copy(x) for x in b[1]] for b in draft_base], [4, 4])))
        emit(event='done', peak_GB=round(mx.get_peak_memory() / 1e9, 2))
    finally:
        for s in streams:
            s.enc.close()


if __name__ == '__main__':
    main()
