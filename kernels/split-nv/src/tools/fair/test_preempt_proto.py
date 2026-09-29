"""CPU test of the preemption protocol (split_nv.preempt + Front.gpu_loop/run_now parking) with two fake ranks.

Rank 0 is the real Front (gpu_loop, run_now, _take_parked) over a fake engine whose prefill chunk "forward" calls
preempt.point() at 21 layer entries x 2 halves; rank 1 is a replica fed through a real multiprocessing Pipe, with its own
copy of the preempt module. A decoder thread STEPs in a loop while 6 preemptible chunks run.
Checks: both ranks execute the same sequence (layers and steps at the same points), every STEP completes exactly once,
steps parked too late for a point run right after the chunk, the decode share cap holds, and step latency drops from
~a chunk to ~a layer. usage: CUDA_VISIBLE_DEVICES= python tools/fair/test_preempt_proto.py"""
import importlib.util
import multiprocessing
import os
import queue
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOOKS = os.path.join(HERE, "..", "..", "hooks")
sys.path.insert(0, HOOKS)
os.environ["SPLIT_NV_DIR"] = "/tmp/split-nv-fair-test"
os.makedirs("/tmp/split-nv-fair-test", exist_ok=True)
from split_nv import front as F  # noqa: E402


def load_preempt(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HOOKS, "split_nv", "preempt.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.run_inline = lambda engine, cmd: engine.execute(cmd)
    return m


LAYER_S, STEP_S = 0.0015, 0.0006


class FakeSteps:
    class Slot:
        graph = True

    def __init__(self, engine):
        self.engine = engine

    def pick(self, L):
        if L > 5:
            raise ValueError
        return FakeSteps.Slot()


class FakeEngine:
    def __init__(self, mod, rank0):
        self.mod, self.rank0 = mod, rank0
        self.log = []
        self.use_graph = True
        self.steps = FakeSteps(self)
        self.sessions = {}
        self.preempt_peers = None
        self.preempt_take = None
        self.last_preempt = None

    def execute(self, cmd):
        if cmd[0] == "prefill_chunk":
            pre = len(cmd) > 5 and cmd[5]
            win = self.mod.open_window(self, self.rank0, self.preempt_peers, self.preempt_take) if pre else None
            if win:
                win.lag = 0
            try:
                for layer in range(21):
                    for half in "AB":
                        self.mod.point()
                        self.log.append(("L", cmd[1], layer, half))
                        time.sleep(LAYER_S / 2)
            finally:
                if win:
                    self.mod.close_window()
            return {}, 0.0
        if cmd[0] == "step":
            self.log.append(("S", cmd[1], cmd[2]))
            time.sleep(STEP_S)
            return {"payload": b"", "L": len(cmd[3])}, STEP_S
        return None


def run(preempt_on, share=0.5, chunks=6):
    m0, m1 = load_preempt("preempt_r0"), load_preempt("preempt_r1")
    m0.SHARE = m1.SHARE = F.preempt.SHARE = share
    e0, e1 = FakeEngine(m0, True), FakeEngine(m1, False)
    r, w = multiprocessing.Pipe(duplex=False)
    e0.preempt_peers, e1.preempt_peers = [w], r
    f = F.Front.__new__(F.Front)
    f.engine, f.peers = e0, [w]
    f.jobs, f.job_seq, f.sid_lock = queue.PriorityQueue(), 0, threading.Lock()
    f.gpu_lock, f.step_cv, f.step_waiters = threading.Lock(), threading.Condition(), 0
    f.current = None
    f.preempt_on = preempt_on
    f.pq_lock, f.pq_open, f.pq = threading.Lock(), False, []
    f.pstats = {"chunks": 0, "inline_steps": 0, "after_chunk_steps": 0}
    e0.preempt_take = f._take_parked
    stop = threading.Event()

    def rank1():
        while True:
            cmd = r.recv()
            if cmd is None:
                return
            e1.execute(cmd)

    threading.Thread(target=rank1, daemon=True).start()
    threading.Thread(target=f.gpu_loop, daemon=True).start()
    lat = []

    def decoder():
        k = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            job = f.run_now(("step", 7, k, [1, 2, 3]))
            job.wait()
            lat.append(time.perf_counter() - t0)
            k += 1
            time.sleep(0.002)

    d = threading.Thread(target=decoder)
    d.start()
    time.sleep(0.02)
    t0 = time.perf_counter()
    jobs = [f.submit_async(("prefill_chunk", 3, [0] * 8192, None, 0, preempt_on), priority=1) for _ in range(chunks)]
    for j in jobs:
        j.wait()
    pf_s = time.perf_counter() - t0
    stop.set()
    d.join()
    w.send(None)
    time.sleep(0.05)
    steps0 = [x[2] for x in e0.log if x[0] == "S"]
    same = e0.log == e1.log
    once = steps0 == list(range(len(lat)))
    return same, once, lat, pf_s, dict(f.pstats), e0.log


ok = True
for pre in (False, True):
    same, once, lat, pf_s, st, log = run(pre)
    print(f"preempt={pre}: ranks identical {same}, every step once in order {once}, steps {len(lat)}, "
          f"step latency median {statistics.median(lat) * 1e3:.1f} ms max {max(lat) * 1e3:.1f} ms, prefill {pf_s * 1e3:.0f} ms, {st}")
    ok &= same and once
    if pre:
        ok &= statistics.median(lat) < 0.5 * 42 * LAYER_S / 2 and st["inline_steps"] > 0
# share cap: steps take at most ~share of a chunk
same, once, lat, pf_s, st, log = run(True, share=0.1)
chunk_time = 21 * LAYER_S
inline = st["inline_steps"]
print(f"share 0.1: ranks identical {same}, steps {len(lat)}, inline {inline}, after-chunk {st['after_chunk_steps']}, prefill {pf_s * 1e3:.0f} ms")
ok &= same and once and inline * STEP_S <= 0.1 * 6 * chunk_time * 1.5 + 6 * STEP_S
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
