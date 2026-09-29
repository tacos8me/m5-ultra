"""host_server.py (partial worker) + extra first-token stamps (x.*) for the ds41-ttft2 first-token path study.
TT2_STAMPS=1: step_end (Scheduler.step returned), burst_end (EngineCore._step_burst returned), put (first
collector.put on the event loop), each once per traced request after og.first_response."""
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if os.environ.get('TT2_STAMPS') == '1':
    sys.path.insert(0, os.environ['DS41_TREE'])
    from omlx.patches.deepseek_v41 import fe_trace
    from omlx.scheduler import Scheduler
    from omlx.engine_core import EngineCore
    from omlx.output_collector import RequestOutputCollector

    def pending(key):
        with fe_trace._lock:
            items = list(fe_trace.BY_ID.values())
        return [t for t in items if 'og.first_response' in t and key not in t]

    step = Scheduler.step

    def step_stamped(self):
        out = step(self)
        now = time.time()
        for t in pending('x.step_end'):
            t['x.step_end'] = now
        return out
    Scheduler.step = step_stamped
    burst = EngineCore._step_burst

    def burst_stamped(self):
        out = burst(self)
        now = time.time()
        for t in pending('x.burst_end'):
            t['x.burst_end'] = now
        return out
    EngineCore._step_burst = burst_stamped
    put = RequestOutputCollector.put

    def put_stamped(self, output):
        rid = getattr(output, 'request_id', None)
        t = fe_trace.for_request(rid)
        if t is not None and 'x.put' not in t and 'og.first_response' in t:
            t['x.put'] = time.time()
        return put(self, output)
    RequestOutputCollector.put = put_stamped
if os.environ.get('TT2_STEPS') == '1':
    sys.path.insert(0, os.environ['DS41_TREE'])
    import json as _json
    from omlx.scheduler import Scheduler as _S2
    _st = _S2.step

    def _step_logged(self):
        t = time.time()
        w = len(self.waiting)
        out = _st(self)
        if w or len(self.waiting):
            print('TT2_STEP ' + _json.dumps(dict(t0=round(t, 4), dt=round(1000 * (time.time() - t), 1), running=len(self.running),
                                                 waiting=w, after=len(self.waiting))), flush=True)
        return out
    _S2.step = _step_logged
if os.environ.get('TT2_PROF') == '1':
    sys.path.insert(0, os.environ['DS41_TREE'])
    import cProfile, io, pstats
    from omlx.scheduler import Scheduler as _S
    _pbr = _S._process_batch_responses
    _seen = set()

    def _pbr_prof(self, responses):
        first = [r for r in responses if getattr(r, 'uid', None) not in _seen]
        for r in responses:
            _seen.add(getattr(r, 'uid', None))
        if not first:
            return _pbr(self, responses)
        prof = cProfile.Profile()
        t = time.perf_counter()
        out = prof.runcall(_pbr, self, responses)
        dt = time.perf_counter() - t
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats('cumulative').print_stats(25)
        print('TT2_PROF first-response process_batch_responses %.1f ms\n%s' % (1000 * dt, buf.getvalue()), flush=True)
        return out
    _S._process_batch_responses = _pbr_prof
sys.argv[0] = str(HERE/'host_server.py')
runpy.run_path(sys.argv[0], run_name='__main__')
