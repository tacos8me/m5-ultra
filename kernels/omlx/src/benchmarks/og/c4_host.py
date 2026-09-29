"""host_server.py (partial decoder half, real box sessions) + a per-scheduler-step timeline of the fused path.

  DS41_TREE=<tree> DS41_OG_HOME=... python c4_host.py --host 127.0.0.1 --port 12699
Every C4T_EVERY (default 50) scheduler steps that ran og verify work it prints one JSON line with the mean
per-step ms of: step (Scheduler.step), gap (engine loop between steps: output processing / detokenize /
streaming), batch_next (_mtp_batch_next), advance (fused_batch.advance), verify (mtp_verify_requests: box
wait + Mac fused/single forward + eval), recv (box wait inside it), fwd_build (forward_boundaries or
forward_boundary graph build), chain (per-request accept + commit + queued draft plan), draft_jobs (batched
DSpark + presend), draft_single (_dspark_next_drafts), emit (_emit_ragged_responses), rows/groups per step.
Output text is meaningless (aliased layers); timing structure is production's.
"""
import collections
import functools
import json
import os
from pathlib import Path
import runpy
import sys
import time

HERE = Path(__file__).resolve().parent
EVERY = int(os.environ.get('C4T_EVERY', '50'))
ACC = collections.defaultdict(float)
CNT = collections.defaultdict(int)
STATE = dict(steps=0, last_end=None, verify_steps=0)
DEPTH = collections.Counter()


def _install_trace():
    from omlx.patches.deepseek_v41 import og_model, pipe_wire, pipe_decoder
    from omlx.patches.mlx_lm_mtp import batch_generator as bg, fused_batch
    from omlx import scheduler as sched

    def wrap(owner, name, label):
        fn = getattr(owner, name)

        @functools.wraps(fn)
        def inner(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                ACC[label] += time.perf_counter() - t
                CNT[label] += 1
        setattr(owner, name, inner)

    wrap(pipe_wire.EncoderSession, 'recv_step', 'recv')
    wrap(pipe_decoder.DecoderHalf, 'forward_boundary', 'fwd_build')
    wrap(pipe_decoder.DecoderHalf, 'forward_boundaries', 'fwd_build')
    wrap(og_model.OgLanguageModel, 'mtp_verify_requests', 'verify')
    wrap(og_model.OgLanguageModel, 'mtp_draft_jobs', 'draft_jobs')
    wrap(bg, '_dspark_next_drafts', 'draft_single')
    wrap(bg, '_emit_ragged_responses', 'emit')
    wrap(bg, '_mtp_batch_next', 'batch_next')
    wrap(fused_batch, 'advance', 'advance')
    wrap(bg, '_run_verify_cycle', 'single_cycle')
    wrap(bg, '_run_verify_cycle_chain', 'chain')
    wrap(bg, '_make_row_batch', 'row_batch')
    wrap(bg, '_replace_cache_rows', 'replace_rows')
    wrap(bg, '_dspark_prepare', 'draft_prepare')
    wrap(bg, '_dspark_finish', 'draft_finish')
    wrap(bg, '_mtp_head_trim_to', 'head_trim')
    wrap(og_model.OgLanguageModel, 'mtp_partial_rollback', 'rollback')
    wrap(og_model, 'presend', 'presend')
    wrap(pipe_wire.EncoderSession, 'ensure_step_safe', 'ensure')
    from omlx.patches.mlx_lm_mtp import batched_head
    wrap(batched_head, 'flush', 'head_flush')
    import mlx.core as mx
    wrap(mx, 'eval', 'mx.eval')
    try:
        from omlx.patches.deepseek_v41 import draft_sources
        for name in ('create',):
            wrap(draft_sources, name, 'extra_create')
    except Exception:
        pass
    fn_single = og_model.OgLanguageModel._forward

    @functools.wraps(fn_single)
    def single_fwd(self, *a, **k):
        t = time.perf_counter()
        try:
            return fn_single(self, *a, **k)
        finally:
            ACC['single_verify'] += time.perf_counter() - t
            CNT['single_verify'] += 1
    og_model.OgLanguageModel._forward = single_fwd
    choose = og_model.PipelineDepthController.choose_cost_depth

    def traced_choose(self, probabilities, context):
        d = choose(self, probabilities, context)
        fused = self.fused is not None and self.fused()
        DEPTH[('fused' if fused else 'c1') + str(d)] += 1
        return d
    og_model.PipelineDepthController.choose_cost_depth = traced_choose
    step = sched.Scheduler.step

    @functools.wraps(step)
    def traced_step(self):
        t = time.perf_counter()
        before = CNT['verify'] + CNT['single_verify']
        try:
            return step(self)
        finally:
            e = time.perf_counter()
            if CNT['verify'] + CNT['single_verify'] > before:
                if STATE['last_end'] is not None and t - STATE['last_end'] < 0.5:
                    ACC['gap'] += t - STATE['last_end']
                    CNT['gap'] += 1
                ACC['step'] += e - t
                STATE['verify_steps'] += 1
                if STATE['verify_steps'] % EVERY == 0:
                    n = EVERY
                    builds = len(BUILDS) - STATE.get('builds', 0)
                    STATE['builds'] = len(BUILDS)
                    print(json.dumps(dict(event='c4trace', steps=n, sessions=len(og_model.SESSIONS),
                                          new_kernel_builds=builds, kernel_builds=len(BUILDS),
                                          per_step_ms={k: round(1000 * v / n, 3) for k, v in sorted(ACC.items())},
                                          calls_per_step={k: round(v / n, 2) for k, v in sorted(CNT.items())},
                                          depth=dict(DEPTH), stats={k: og_model.STATS[k] for k in (
                                              'fused_calls', 'fused_steps', 'draft_batches', 'steps', 'rows')})),
                          flush=True)
                    ACC.clear(); CNT.clear(); DEPTH.clear()
            STATE['last_end'] = e
    sched.Scheduler.step = traced_step


sys.path.insert(0, os.environ['DS41_TREE'])
# host_server.py's (= og_server.py's) import-time flags, set before og_model is imported here.
for key, value in dict(MLX_ENABLE_TF32='0', OMLX_BONJOUR='0', OMLX_DISCOVERY='0', DS41_NATIVE_VERIFY='0',
                       DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1',
                       DS41_NATIVE_DECODE='1', DS41_MTP_COST_POLICY='1', DS41_COPY_DRAFT='1',
                       DS41_EXTRA_DRAFT='1', DS41_PREFIX_CACHE_GIB='0', DS41_BATCH_VERIFY='0').items():
    os.environ.setdefault(key, value)
import mlx.core as _mx  # noqa: E402
BUILDS = set()
_make_kernel = _mx.fast.metal_kernel


def _counted_kernel(*a, **k):
    kern = _make_kernel(*a, **k)
    name = k.get('name', a[0] if a else '?')

    def call(*ca, **ck):
        BUILDS.add((name, repr(ck.get('template'))))
        return kern(*ca, **ck)
    return call


_mx.fast.metal_kernel = _counted_kernel  # first-use pipeline builds (one per name + template) per window
from omlx.patches.deepseek_v41 import og_model  # noqa: E402
_install = og_model.install


def install(*a, **k):
    out = _install(*a, **k)
    _install_trace()
    return out


og_model.install = install
sys.argv[0] = str(HERE/'host_server.py')
runpy.run_path(sys.argv[0], run_name='__main__')
