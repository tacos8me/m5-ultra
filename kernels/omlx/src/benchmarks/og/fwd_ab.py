"""Full Mac-half forward (partial load: layers 20/24/25 aliased to 20 layers + head) A/B of the attention
fusions, interleaved in one process, min of rounds. Uses the ds41-prof harness (prof_mac.py) for loading,
synthetic 8K caches/boundaries and the production forward_boundary. Speed only; production must be idle."""
import os
import sys
from pathlib import Path

HOME = Path.home()
os.environ.setdefault('DS41_TREE', str(HOME/'src/wt/ds41-attn'))
os.environ.setdefault('PROF_OUT', str(HOME/'llm/ds41/attn/fwd_ab.jsonl'))
os.environ.setdefault('PROF_LABEL', 'fwd-ab')
os.environ.setdefault('PROF_LEASE_S', '1200')
sys.path.insert(0, str(Path(os.environ.get('PROF_HARNESS', str(HOME/'src/wt/ds41-prof/benchmarks/og')))))
import prof_mac as P  # noqa: E402
import mlx.core as mx  # noqa: E402
from omlx.patches.deepseek_v41 import attn_fusions, kernels, fast_qmv  # noqa: E402


def config(new):
    attn_fusions.ENABLED = new
    attn_fusions.DECODE_SELECT = new
    kernels.DS41_ATTN_SHARED = new
    kernels.DS41_MERGE_WIDE = new
    fast_qmv.MIN_ROWS = 3 if new else 4


_SPIN = mx.fast.metal_kernel(
    name='fwd_ab_spin', input_names=['n'], output_names=['y'],
    source="""
    float acc = 0.0f;
    for (int i = 0; i < n[0]; ++i) acc = metal::fma(acc, 0.999f, 1.0f);
    y[0] = acc;
    """)
SPIN_N = int(os.environ.get('AB_SPIN', '6000000'))


def spin():
    return _SPIN(inputs=[mx.array([SPIN_N], mx.int32)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                 output_shapes=[(1,)], output_dtypes=[mx.float32])[0]


def gpu_forward(lm, cache, b, rows, reps=5):
    """GPU ms of one forward: it queues behind a spin kernel longer than the host build."""
    import time
    def once(with_fwd):
        g = spin()
        t0 = time.perf_counter()
        if with_fwd:
            bb = dict(b, h=b['h'] + (g * 0).astype(b['h'].dtype))
            start = cache[0].size()
            logits, hidden = lm.forward_boundary(**bb, cache=cache, start=start, verify=True)
            out = [mx.argmax(logits[0], -1), hidden]
        else:
            out = [g]
        mx.eval(*out)
        t = time.perf_counter() - t0
        if with_fwd:
            lm.rollback_boundary(cache, 1)
        return t
    fwd = min(once(True) for _ in range(reps))
    base = min(once(False) for _ in range(reps))
    return 1000 * (fwd - base)


def main():
    lm = P.load('partial')
    c = lm._config
    ctx = int(os.environ.get('AB_CTX', '8192'))
    cache = P.synth_cache(c, ctx)
    bound = P.synth_boundary(c, 5)
    rounds = int(os.environ.get('AB_ROUNDS', '6'))
    for rows in [int(r) for r in os.environ.get('AB_ROWS', '1,2,3,4,5').split(',')]:
        b = bound if rows == 5 else {k: v[:, :rows] for k, v in bound.items()}
        res = {False: [], True: []}
        gpu = {False: [], True: []}
        for _ in range(rounds):
            for new in (False, True):
                config(new)
                r = P.time_forward(lm, cache, b, rows, reps=6, warm=2)
                res[new].append(r)
                gpu[new].append(gpu_forward(lm, cache, b, rows))
        config(True)
        best = {k: min(x['total_ms'] for x in v) for k, v in res.items()}
        wait = {k: min(x['wait_ms'] for x in v) for k, v in res.items()}
        build = {k: min(x['build_ms'] for x in v) for k, v in res.items()}
        P.emit(dict(test='fwd_ab', context=ctx, rows=rows, old_ms=best[False], new_ms=best[True],
                    saved_ms=round(best[False] - best[True], 3), old_build=build[False], new_build=build[True],
                    old_wait=wait[False], new_wait=wait[True],
                    old_gpu=round(min(gpu[False]), 3), new_gpu=round(min(gpu[True]), 3),
                    gpu_saved=round(min(gpu[False]) - min(gpu[True]), 3)))


if __name__ == '__main__':
    main()
