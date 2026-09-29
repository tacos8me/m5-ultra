"""Cold (first-use) vs warm cost of the c4 shapes in a fresh process. Run through idle_guard.py only.

Partial load (layers 20/24/25 aliased over 20-39 + head + DSpark, <= 35 GB), 2 real 8K box sessions, production
flags (woa_compact, ffn_fuse, attn_in). Every custom kernel is specialised on its row count (template M), so each
new verify/draft shape builds new Metal pipelines on first use. The og_server warmup only runs singleton requests:
the first c4 traffic after a restart pays the fused-pair / batched-draft builds. This measures that bill.
  CC_REPS=5 (warm reps per shape)   CC_WARM=1: run the warm-up helper first, then the same sequence (should be warm)
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
OUT = Path.home()/'llm/ds41/c4/cold.jsonl'
OUT.parent.mkdir(parents=True, exist_ok=True)
_out = OUT.open('a')
REPS = int(os.environ.get('CC_REPS', '5'))
BUILDS = set()


def emit(**r):
    line = json.dumps(r)
    _out.write(line + '\n'); _out.flush(); print(line, flush=True)


def count_builds():
    import mlx.core as mx
    make = mx.fast.metal_kernel

    def wrapped(*a, **k):
        kern = make(*a, **k)
        name = k.get('name', a[0] if a else '?')

        def call(*ca, **ck):
            key = (name, repr(ck.get('template')))
            BUILDS.add(key)
            return kern(*ca, **ck)
        return call
    mx.fast.metal_kernel = wrapped


def main():
    count_builds()
    os.environ['BB_STREAMS'] = '2'
    os.environ['BB_OUT'] = str(OUT.parent/'bb.jsonl')
    import batch_bench as B
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import woa_compact
    mx.set_memory_limit(34_000_000_000)
    mx.set_cache_limit(128 << 20)
    t = time.perf_counter()
    lm = B.load()
    summary = woa_compact.install(lm)
    emit(event='loaded', s=round(time.perf_counter() - t, 1), active_GB=round(mx.get_active_memory() / 1e9, 2),
         woa_encoded=summary['encoded'])
    streams = []
    try:
        for i, toks in enumerate(B.prompts()):
            streams.append(B.Stream(lm, toks, f'c4cold-{i}'))
            for n in (1, 2, 3, 4, 5):
                streams[-1].boundary(n)
        emit(event='sessions', n=len(streams), open_s=[round(s.open_s, 2) for s in streams])

        def measure(label, fn, **meta):
            before = len(BUILDS)
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(fn())
            cold = (time.perf_counter() - t0) * 1000
            new = len(BUILDS) - before
            warm = []
            for _ in range(REPS):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fn())
                warm.append((time.perf_counter() - t0) * 1000)
            w = statistics.median(warm)
            emit(test=label, cold_ms=round(cold, 2), warm_ms=round(w, 3), extra_ms=round(cold - w, 2),
                 new_kernel_builds=new, **meta)
            return cold - w

        s0 = streams[0]
        # production drafts come from real hidden rows; one 5-row verify hidden per stream
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
        emit(event='primed', builds_so_far=len(BUILDS))
        if os.environ.get('CC_WARM') == '1':
            from omlx.patches.deepseek_v41 import c4_warm
            t0 = time.perf_counter()
            info = c4_warm.warm(lm)
            emit(event='warm_helper', s=round(time.perf_counter() - t0, 2), builds_so_far=len(BUILDS), **info)
        total = {'single': 0.0, 'fused': 0.0, 'draft': 0.0, 'draft_batch': 0.0}
        for n in (1, 2, 3, 4, 5):
            def single(n=n):
                s0.reset()
                return lm.forward_boundary(**s0.boundary(n), cache=s0.cache, start=s0.start, verify=True)
            total['single'] += measure('single', single, rows=n)
        combos = [(a, b) for a in (2, 3, 4, 5) for b in (2, 3, 4, 5)]
        for rows in combos:
            total['fused'] += measure('fused', lambda rows=rows: [x for p in B.fused(lm, streams, rows) for x in p],
                                      rows=list(rows))
        h, cache, anchor = base[0]
        for width in (1, 2, 3, 4):
            total['draft'] += measure('draft', lambda width=width: lm.dspark_forward(
                h, anchor, [copy.copy(x) for x in cache], draft_length=width)[0], width=width)
        for width in (2, 3, 4):
            total['draft_batch'] += measure('draft_batch', lambda width=width: lm.dspark_forward_batch(
                [b[0] for b in base], [b[2] for b in base], [[copy.copy(x) for x in b[1]] for b in base],
                [width, width]), widths=[width, width])
        emit(event='done', extra_ms_total={k: round(v, 1) for k, v in total.items()}, builds=len(BUILDS),
             peak_GB=round(mx.get_peak_memory() / 1e9, 2))
    finally:
        for s in streams:
            s.enc.close()


if __name__ == '__main__':
    main()
