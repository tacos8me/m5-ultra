"""MXFP8 draft head (DS41_DRAFT_HEAD=mxfp8): bitwise + timing gate on the real 129280x5120 head. GPU, WINDOW ONLY.

Loads only language_model.head.weight (BF16, 1.32 GB) from the og checkpoint, quantizes it to MXFP8 exactly
as dspark.install_draft_head does (+0.68 GB), and for M = 1..8 query rows checks:
  bitwise    og_fused._rows (fast_qmv rows kernel = the draft-head path) == mx.quantized_matmul(mode='mxfp8')
             at M = 2..5 (required: MLX's qmv_wide, same lanes/order). M = 1 is reported only: MLX's M=1 path
             is qmv_fast (fast_qmv._fast_kernel is its replica, also reported); cost-policy drafts are 4 rows
             single and 8 rows per fused pair, never 1.
  invariant  the rows kernel's row q is bitwise the same at every M > q and for a 4+4 split (single draft
             M=4 == its half of a batched pair draft M=8). Required.
  e2e        dspark.draft_logits / dspark.batch_logits with DS41_DRAFT_HEAD=mxfp8: cost policy -> the rows
             kernel, no cost policy -> the BF16 head, batched (mixed policies) == single. Required.
  f32        the same rows kernel with FP32 in/out (unrounded logits) rounds to the BF16 result (informational).
  timing     GPU ms per dependent launch at M = 1..8: BF16 head (project_logits single path, _head_rows
             batched path), MXFP8 rows, the full MXFP8 draft path, MXFP8 rows FP32 in/out, MLX qmm mxfp8.
             Watch M=8 against M=4 (register-spill risk at the pair shape).

Needs gpu.lock (refuses otherwise; HB_NO_LOCK=1 = idle_guard mode alongside an idle production worker):
  cd ~/src/wt/ds41-next4 && GPU_LOCK_WAIT=120 ~/llm/bin/gpu-exec \
      ~/llm/.venv-ds41-omlx-tiles/bin/python -u benchmarks/og/head_mxfp8_bench.py
Output: JSON lines on stdout and in ~/llm/ds41/next4/head_mxfp8_bench.jsonl; exit 1 if a required check fails.
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DS41_TREE'] = str(ROOT)
os.environ.setdefault('MLX_ENABLE_TF32', '0')
os.environ['DS41_DRAFT_HEAD'] = 'mxfp8'  # dspark reads it at import
HOME = Path.home()
LOCK = HOME/'llm/locks/gpu.lock'
MODEL = Path(os.environ.get('HB_MODEL', str(HOME/'llm/ds41/og/models/ds41-og')))
OUT = Path(os.environ.get('HB_OUT', str(HOME/'llm/ds41/next4/head_mxfp8_bench.jsonl')))
CALLS, REPS, WARM = 10, 15, 3
FAIL = []


def emit(**record):
    line = json.dumps(record)
    print(line, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('a') as out:
        out.write(line + '\n')


def holds_lock():
    if os.environ.get('HB_NO_LOCK') == '1':
        return True
    holders = subprocess.run(['lsof', '-t', str(LOCK)], capture_output=True, text=True).stdout.split()
    return str(os.getpid()) in holders or str(os.getppid()) in holders


if not holds_lock():
    sys.exit(f'head_mxfp8_bench: run under ~/llm/bin/gpu-exec (this process must hold {LOCK})')

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.set_memory_limit(24 << 30)
mx.set_cache_limit(1 << 30)
from omlx.patches.deepseek_v41 import dspark, fast_qmv, head, og_fused  # noqa: E402


def mismatches(a, b):
    """Differing elements, compared as raw bits."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    view = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}.get(a.dtype)
    a, b = mx.contiguous(a), mx.contiguous(b)
    if view is not None:
        a, b = a.view(view), b.view(view)
    return int((np.array(a) != np.array(b)).sum())


def require(name, bad, **extra):
    ok = bad == 0
    if not ok:
        FAIL.append(name)
    emit(check=name, ok=ok, mismatches=bad, **extra)


def load_head():
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    key = next(k for k in index if k.endswith('head.weight') and '.mtp.' not in k and 'markov' not in k)
    weight = mx.load(str(MODEL/index[key]))[key]  # lazy: only this tensor is read
    mx.eval(weight)
    return key, weight


def gpu_ms(step, x):
    """Median GPU ms per launch: CALLS dependent launches minus one, per extra launch."""
    def run(n):
        times = []
        for rep in range(REPS + WARM):
            y, outs = x, []
            mx.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                o = step(y)
                outs.append(o)
                y = x + (o[..., :1, :1] * 0).astype(x.dtype)  # the next launch waits for this one
            mx.eval(outs)
            if rep >= WARM:
                times.append(time.perf_counter() - t)
        return statistics.median(times)
    return round((run(CALLS) - run(1)) / (CALLS - 1) * 1e3, 4)


def main():
    assert mx.default_device() == mx.gpu, 'needs the GPU'
    t = time.perf_counter()
    key, weight = load_head()
    n, k = weight.shape
    emit(event='loaded', key=key, shape=[n, k], dtype=str(weight.dtype), s=round(time.perf_counter() - t, 2))
    if (n, k) != (129280, 5120) or weight.dtype != mx.bfloat16:
        FAIL.append('head_shape')
    model = SimpleNamespace(head=SimpleNamespace(weight=weight))
    summary = dspark.install_draft_head(model)  # prints the same draft_head line the og worker logs
    emit(event='quantized', **{k: v for k, v in (summary or {}).items() if k != 'event'})
    q = dspark._draft_head(model)
    rng = [mx.random.normal((1, 8, k), key=mx.random.key(seed)).astype(mx.bfloat16) * scale
           for seed, scale in ((1, 1.0), (2, 4.0), (3, 0.25))]
    mx.eval(rng)

    # bitwise vs MLX (M <= 5), M-invariance (M = 1..8), 4+4 split
    for s, x8 in enumerate(rng):
        full = og_fused._rows(q, x8)
        for m in range(1, 9):
            xm = x8[:, :m]
            rows = og_fused._rows(q, xm)
            require(f'invariant_M{m}_set{s}', mismatches(rows, full[:, :m]), M=m)
            f32 = og_fused._rows(q, xm.astype(mx.float32))
            emit(check=f'f32_rounds_to_rows_M{m}_set{s}', info=True,
                 mismatches=mismatches(f32.astype(mx.bfloat16), rows))
            if m <= 5:
                ref = mx.quantized_matmul(xm, q.weight, q.scales, transpose=True, group_size=32, bits=8, mode='mxfp8')
                bad = mismatches(rows, ref)
                if m == 1:
                    emit(check=f'vs_mlx_M1_set{s}', info=True, rows_kernel_mismatches=bad,
                         fast_m1_kernel_mismatches=mismatches(fast_qmv.mxfp8_qmv(xm, q.weight, q.scales), ref))
                else:
                    require(f'vs_mlx_M{m}_set{s}', bad, M=m)
        split = mx.concatenate([og_fused._rows(q, x8[:, :4]), og_fused._rows(q, x8[:, 4:])], 1)
        require(f'split_4+4_set{s}', mismatches(split, full))

    # the draft path itself (single, batched, mixed policies)
    x8 = rng[0]
    a, b = x8[:, :4], x8[:, 4:]
    require('draft_mxfp8_is_rows', mismatches(dspark.draft_logits(model, a, True),
                                              og_fused._rows(q, a).astype(mx.float32)))
    require('draft_bf16_is_head', mismatches(dspark.draft_logits(model, a, False), og_fused._head_rows(weight, a)))
    for policy in ([True, True], [True, False], [False, True], [False, False]):
        batched = dspark.batch_logits(model, [a, b], 4, policy)
        single = [dspark.draft_logits(model, x, p) for x, p in zip((a, b), policy)]
        require(f'batched_is_single_{policy}', sum(mismatches(u, v) for u, v in zip(batched, single)))

    # timing
    wbytes = {'bf16': n * k * 2, 'mxfp8': n * k + n * k // 32}
    variants = {
        'bf16_project': (lambda y: head.project_logits(y, model.head), 'bf16', 5),  # single path (M<=5 kernels)
        'bf16_head_rows': (lambda y: og_fused._head_rows(weight, y), 'bf16', 8),  # batched path
        'mxfp8_rows': (lambda y: og_fused._rows(q, y), 'mxfp8', 8),
        'mxfp8_draft': (lambda y: dspark.draft_logits(model, y, True), 'mxfp8', 8),  # rows + FP32 cast
        'mxfp8_rows_f32io': (lambda y: og_fused._rows(q, y.astype(mx.float32)), 'mxfp8', 8),
        'mlx_qmm_mxfp8': (lambda y: mx.quantized_matmul(y, q.weight, q.scales, transpose=True, group_size=32,
                                                        bits=8, mode='mxfp8'), 'mxfp8', 8),
    }
    table = {}
    for m in range(1, 9):
        x = rng[0][:, :m]
        for name, (step, kind, max_m) in variants.items():
            if m > max_m:
                continue
            ms = gpu_ms(step, x)
            table.setdefault(name, {})[m] = ms
            emit(timing=name, M=m, ms=ms, gb_s=round(wbytes[kind] / ms / 1e6, 1) if ms > 0 else None)
    t = table
    emit(summary='timing',
         single_M4_saved_ms=round(t['bf16_project'][4] - t['mxfp8_draft'][4], 4),
         pair_M8_saved_ms=round(t['bf16_head_rows'][8] - t['mxfp8_draft'][8], 4),
         mxfp8_M8_over_M4=round(t['mxfp8_rows'][8] / t['mxfp8_rows'][4], 3),
         bf16_M8_over_M4=round(t['bf16_head_rows'][8] / t['bf16_head_rows'][4], 3),
         expected='verdict: BF16 head ~1.18 ms at M=4, MXFP8 ~0.78 ms (-0.40); pair M=8 -0.55 ms')
    emit(summary='gates', ok=not FAIL, failed=FAIL)
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
