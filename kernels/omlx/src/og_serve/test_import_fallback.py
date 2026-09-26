"""CPU-only tests: a failed box-state import retries with a plain full OPEN, and never reaches a local prefill.

Loads the real og_model / og_failover / pipe_wire with MLX and the heavy decoder modules stubbed
(no Metal), against og_serve/fakebox.py. Run: python og_serve/test_import_fallback.py
"""
import importlib.util
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
PKG = Path(os.environ.get('OG_PKG', str(HERE.parent/'omlx/patches/deepseek_v41')))
sys.path.insert(0, str(HERE))
os.environ.setdefault('DS41_OG_RESUME_WAIT_S', '2')
os.environ.setdefault('DS41_OG_RESTART_WAIT_S', '2')
from fakebox import FakeBox  # noqa: E402

PORT = int(os.environ.get('FAKEBOX_PORT', '12198'))


def stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


class Arr:
    def __init__(self, value=None):
        self.value = value


mx = stub('mlx.core', array=lambda v, dtype=None: Arr(v), int64='int64', bfloat16='bf16', eval=lambda *a, **k: None)
stub('mlx', core=mx, nn=None)
nn = stub('mlx.nn', Module=type('Module', (), {'__init__': lambda self: None}))
sys.modules['mlx'].nn = nn


class Store:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1

    def lookup(self, tokens, key):
        return None, 0

    def record(self, *args):
        pass


STORE = Store()
pkg = stub('ogtest')
pkg.__path__ = []
stub('ogtest.growth')
stub('ogtest.og_cache', from_env=lambda: STORE, numerics_key=lambda m: 'k')
stub('ogtest.cache', DeepseekV41Cache=object)
stub('ogtest.pipe_decoder', DecoderHalf=type('DecoderHalf', (), {}), load_decoder=lambda *a, **k: None)


def open_remote(encoder, tokens, request_id='', **options):
    OPENS.append(options)
    tensors, manifest = encoder.open(tokens, request_id)
    return tensors, manifest, 0.0


OPENS = []
stub('ogtest.pipe_session', PipelineDepthController=object, open_remote=open_remote)


def load(name):
    spec = importlib.util.spec_from_file_location(f'ogtest.{name}', PKG/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'ogtest.{name}'] = mod
    spec.loader.exec_module(mod)
    setattr(pkg, name, mod)
    return mod


wire = load('pipe_wire')
failover = load('og_failover')
load('og_images')
og = load('og_model')


class Item:
    def __init__(self):
        self.cache = []
        self.slots = {}

    def __setitem__(self, key, value):
        self.slots[key] = value

    def __getitem__(self, key):
        return self.slots.get(key)


class LM:
    layers, _config = [], None

    def __init__(self, state_ok=True, prefill_ok=True):
        self.state_ok, self.prefill_ok, self.calls = state_ok, prefill_ok, []

    def import_state(self, tensors, manifest, tokens, *, identity, base_rows=None):
        self.calls.append('state')
        if not self.state_ok:
            raise KeyError('layer.20.slot.2')
        return [Item() for _ in range(40)], []

    def import_prefill(self, arrays, manifest, tokens, *, identity):
        self.calls.append('prefill')
        if not self.prefill_ok:
            raise ValueError('bad state')
        return [Item() for _ in range(40)]


class Request:
    def __init__(self, rid, tokens):
        self.request_id, self.prompt_token_ids, self.num_prompt_tokens = rid, tokens, len(tokens)
        self.vlm_inputs_embeds = None


class Sched:
    def __init__(self, lm):
        self.model = lm
        self._prefix_cache_prepared = set()


def run(manager, lm, rid):
    req = Request(rid, list(range(100, 140)))
    sched = Sched(lm)
    assert manager.should_defer(sched, req) is True
    manager.jobs[rid].join(10)
    assert manager.should_defer(sched, req) is False, 'job should be done'
    return req, sched, manager.prepare(sched, req)


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f': {detail}'), flush=True)
    return bool(cond)


def main():
    box = FakeBox(PORT)
    manager = og.OgPrefill('127.0.0.1', PORT)
    results = []

    lm = LM()
    req, sched, ok = run(manager, lm, 'r-ok')
    results.append(check('cached import works as before', ok and lm.calls == ['state'] and req.prompt_cache is not None
                         and og.STATS['import_fallbacks'] == 0, lm.calls))
    og.close_request('r-ok')

    lm = LM(state_ok=False)
    OPENS.clear()
    req, sched, ok = run(manager, lm, 'r-retry')
    results.append(check('failed import -> plain full OPEN -> import_prefill, request proceeds',
                         ok and lm.calls == ['state', 'prefill'] and req.prompt_cache is not None
                         and req.request_id in sched._prefix_cache_prepared and og.STATS['import_fallbacks'] == 1,
                         (ok, lm.calls, dict(og.STATS))))
    results.append(check('the retry OPEN asks for no cache, no delta, full state', OPENS[-1] == {}, OPENS))
    results.append(check('the Mac row store is cleared after a failed import', STORE.cleared == 1, STORE.cleared))
    results.append(check('session registered for the retried request', 'r-retry' in og.REQUESTS, og.REQUESTS))
    og.close_request('r-retry')

    lm = LM(state_ok=False, prefill_ok=False)
    req, sched, ok = run(manager, lm, 'r-fail')
    marker = failover.RESUMABLE.get('r-fail')
    results.append(check('both imports fail -> prepare False, request marked, resumable marker',
                         ok is False and getattr(req, '_ds41_og_failed', None) and marker and marker['kind'] == 'box'
                         and marker['output'] == [], (ok, getattr(req, '_ds41_og_failed', None), marker)))

    class FakeScheduler:
        def _do_external_prefill(self, request, *args, **kwargs):
            raise AssertionError('local prefill ran')

        def _step_prefill_chunk(self, state, *args, **kwargs):
            raise AssertionError('local chunk ran')
    failover.guard_local_prefill(FakeScheduler)
    s = FakeScheduler()
    msgs = []
    for fn, arg in ((s._do_external_prefill, req), (s._step_prefill_chunk, types.SimpleNamespace(request=req))):
        try:
            fn(arg)
        except RuntimeError as exc:
            msgs.append(str(exc))
        except AssertionError as exc:
            msgs.append('LOCAL:' + str(exc))
    results.append(check('guard: no local prefill, a clean RuntimeError for that request',
                         len(msgs) == 2 and all(m.startswith('ds41-og: box state import failed') for m in msgs), msgs))

    lm = LM(state_ok=False)
    manager2 = og.OgPrefill('127.0.0.1', PORT)
    req2 = Request('r-refused', list(range(100, 140)))
    sched2 = Sched(lm)
    manager2.should_defer(sched2, req2)
    manager2.jobs['r-refused'].join(10)
    box.err_opens = True  # the cached OPEN went through; the retry OPEN is refused
    ok = manager2.prepare(sched2, req2)
    box.err_opens = False
    results.append(check('box refuses the retry OPEN -> clean failure with marker, no local prefill',
                         ok is False and getattr(req2, '_ds41_og_failed', None)
                         and failover.RESUMABLE.get('r-refused', {}).get('kind') == 'box', (ok, lm.calls)))
    box.down()
    ok = all(results)
    print('ALL PASS' if ok else 'SOME FAILED', f'({sum(results)}/{len(results)})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
