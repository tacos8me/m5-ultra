"""og worker over a PARTIAL decoder half (layers 20/24/25 aliased + head + embed + DSpark, ~31 GB): speed A/B only.

Runs the unchanged og_serve/og_server.py (omlx scheduler, DSpark loop, presend, box sessions) with
og_model.load_decoder replaced by the prof_mac partial loader. Output text is meaningless (aliased
layers); timing per step is what it measures. No gpu.lock: only while production is idle.
  DS41_TREE=<tree> DS41_OG_HOME=~/llm/ds41/host/og python host_server.py --host 127.0.0.1 --port 12699
HS_TRACE=1 prints per-og-step host time of the wrapped calls every 100 steps; HS_SWITCH sets the GIL switch
interval. benchmarks/og/host_ab.sh runs one A/B leg (zsh: start it with BG_NICE off, or it runs at nice 5).
"""
import json
import os
from pathlib import Path
import runpy
import sys

HOME = Path.home()
TREE = os.environ['DS41_TREE']
sys.path.insert(0, TREE)
os.environ.setdefault('DS41_WARMUP', '0')
# og_server.py's defaults, before og_model (and language) are imported below: flags such as DS41_MHC are
# read at import time, so setting them only in og_server.py (run after this import) left them off here.
for key, value in dict(MLX_ENABLE_TF32='0', OMLX_BONJOUR='0', OMLX_DISCOVERY='0', DS41_NATIVE_VERIFY='0',
                       DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1',
                       DS41_NATIVE_DECODE='1', DS41_MTP_COST_POLICY='1', DS41_COPY_DRAFT='1',
                       DS41_EXTRA_DRAFT='1', DS41_PREFIX_CACHE_GIB='0', DS41_BATCH_VERIFY='0').items():
    os.environ.setdefault(key, value)
os.environ.setdefault('DS41_OG_CONCURRENCY', '1')
KIND = {20: 20, **{i: 24 for i in (24, 28, 32, 36)}}


def partial_decoder(path, cls=None):
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import pipe_decoder
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.loading import _load_shard, set_module
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection
    path = Path(path)
    raw = json.loads((path/'config.json').read_text())
    config = ModelConfig.from_dict(raw)
    config.ced_prefill = True
    model = pipe_decoder.DecoderContainer(config, cls)
    mapping = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    keep = tuple(f'language_model.{p}.' for p in ('layers.20', 'layers.24', 'layers.25', 'head', 'norm', 'embed', 'mtp'))
    for filename in sorted({f for k, f in mapping.items() if k.startswith(keep)}):
        values = {k: v for k, v in _load_shard(path/filename).items() if k.startswith(keep)}
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
    print(json.dumps(dict(event='partial_loaded', active_gib=round(mx.get_active_memory() / 2**30, 2))), flush=True)
    return lm


from omlx.patches.deepseek_v41 import og_model  # noqa: E402
og_model.load_decoder = partial_decoder
if os.environ.get('HS_FIXED_DEPTH') == '1':
    # Aliased layers make garbage text, so DSpark acceptance (and with it the verify depth) collapses.
    # Keep every cycle at the configured depth (5 rows, production averages ~4.4) so Mac/box work per
    # cycle is production-like; compare cycles/s, not tokens.
    from omlx.patches.deepseek_v41.pipe_session import PipelineDepthController
    PipelineDepthController.observe = lambda self, *a, **k: None
    PipelineDepthController.choose_cost_depth = lambda self, *a, **k: self.max_depth
if os.environ.get('HS_DRAFTS') == '1':
    # Draft fingerprint per request (sha256 over every draft block it verified), printed when the
    # request finishes, keyed by its first generated tokens: compare trees/concurrency on fixed prompts.
    import functools, hashlib
    from omlx.patches.mlx_lm_mtp import batch_generator as _bg
    _drafts, _log_stats, _fp = _bg._dspark_next_drafts, _bg._log_mtp_stats, {}

    @functools.wraps(_drafts)
    def _drafts_fp(gen_batch, state, *a, **k):
        out = _drafts(gen_batch, state, *a, **k)
        rec = _fp.setdefault(id(state.stats), [hashlib.sha256(), 0, None])
        if rec[2] is None and len(gen_batch.tokens[0]) >= 4:
            rec[2] = list(gen_batch.tokens[0][:4])
        rec[0].update(repr(state.drafts.tolist()).encode())
        rec[1] += 1
        return out

    _jobs = _bg.dspark_draft_jobs

    @functools.wraps(_jobs)
    def _jobs_fp(jobs, *a, **k):
        out = _jobs(jobs, *a, **k)
        for gen_batch, state, *_ in jobs:  # batched drafting (og_model.mtp_draft_jobs) bypasses _dspark_next_drafts
            rec = _fp.setdefault(id(state.stats), [hashlib.sha256(), 0, None])
            if rec[2] is None and len(gen_batch.tokens[0]) >= 4:
                rec[2] = list(gen_batch.tokens[0][:4])
            rec[0].update(repr(state.drafts.tolist()).encode())
            rec[1] += 1
        return out
    _bg.dspark_draft_jobs = _jobs_fp

    @functools.wraps(_log_stats)
    def _log_stats_fp(uid, stats, *a, **k):
        rec = _fp.pop(id(stats), None)
        if rec is not None:
            print(json.dumps(dict(event='drafts', key=rec[2], blocks=rec[1], sha=rec[0].hexdigest()[:16])), flush=True)
        return _log_stats(uid, stats, *a, **k)
    _bg._dspark_next_drafts, _bg._log_mtp_stats = _drafts_fp, _log_stats_fp
if os.environ.get('HS_SWITCH'):
    sys.setswitchinterval(float(os.environ['HS_SWITCH']))

if os.environ.get('HS_TRACE') == '1':
    # Per-step host timeline: summed duration of each wrapped call per og step, every 100 steps.
    import collections, functools, threading, time
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import pipe_wire, pipe_decoder
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx import scheduler as sched
    ACC = collections.defaultdict(float)
    CNT = collections.defaultdict(int)
    LAST = {}

    def wrap(owner, name, label):
        fn = getattr(owner, name)

        @functools.wraps(fn)
        def inner(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                e = time.perf_counter()
                ACC[label] += e - t
                CNT[label] += 1
                if label == 'sched.step':
                    prev = LAST.get('end')
                    if prev is not None:
                        ACC['between_steps'] += t - prev
                    LAST['end'] = e
                if label == 'remote' and CNT['remote'] % 100 == 0:
                    n = 100
                    print(json.dumps(dict(event='trace', per_step_ms={k: round(1000 * v / n, 3) for k, v in sorted(ACC.items())},
                                          calls_per_step={k: round(v / n, 2) for k, v in sorted(CNT.items())})), flush=True)
                    ACC.clear(); CNT.clear()
        setattr(owner, name, inner)

    wrap(og_model.OgLanguageModel, '_remote', 'remote')
    wrap(pipe_wire.EncoderSession, 'recv_step', 'recv')
    wrap(pipe_decoder.DecoderHalf, 'forward_boundary', 'fwd_build')
    wrap(og_model.OgLanguageModel, 'mtp_partial_rollback', 'rollback')
    wrap(bg, '_dspark_next_drafts', 'dspark_drafts')
    wrap(og_model, 'presend', 'presend')
    wrap(sched.Scheduler, 'step', 'sched.step')
    wrap(mx, 'eval', 'mx.eval')
    wrap(mx, 'async_eval', 'mx.async_eval')
    _install = og_model.install

    def install(*a, **k):
        out = _install(*a, **k)
        wrap(bg, '_run_verify_cycle_chain', 'chain')
        return out
    og_model.install = install

sys.argv[0] = str(Path(TREE)/'og_serve/og_server.py')
runpy.run_path(sys.argv[0], run_name='__main__')
