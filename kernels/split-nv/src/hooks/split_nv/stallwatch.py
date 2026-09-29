"""Host-stall attribution for decode gaps during long prefills (SPLIT_NV_STALLWATCH=1, default off; logging only).

A STEP parked on a preemptible chunk runs at the next decoder-layer point of the prefill's host thread, so anything that
stops that thread -- or rank 1's, which blocks at every point for rank 0's message -- stops decoding too, and no amount
of preemption granularity helps. This module names such stalls:
  * gc: every CPython collection that takes >= SPLIT_NV_STALLWATCH_GC_MS (20) is logged with its generation and the
    tracked-object counts (a full collection walks every container object of the process);
  * host: a daemon thread ticks every 5 ms; a tick late by >= SPLIT_NV_STALLWATCH_MS (100) means this process ran no
    Python for that long (GIL held by one long C call, a collection, page-fault/direct-reclaim, SIGSTOP...). It logs
    the stall with the GPU job and preemption point current at the time and the cgroup's memory-pressure `full`
    delta (time every task of the container was stalled on memory) over the stall.
SPLIT_NV_GC_FREEZE=1: gc.freeze() once warm-up is done (Engine command "gc_freeze", every rank): the model, graphs,
tokenizer map and prefix-cache index move to the permanent generation and are never scanned again -- SGLang's own
scheduler does the same (sglang.srt.utils.freeze_gc). Byte-neutral: no tensor or kernel is touched.
"""
import gc
import os
import threading
import time

TICK_S = 0.005
PSI = "/sys/fs/cgroup/memory.pressure"  # the container's own cgroup (cgroup v2 namespace)

stats = {"gc_slow": 0, "gc_max_ms": 0.0, "stalls": 0, "stall_max_ms": 0.0, "frozen": None}
_ctx = None  # callable() -> short string: what this rank was doing (GPU job, preemption point)
_gc_t0 = {}


def enabled():
    return os.environ.get("SPLIT_NV_STALLWATCH", "0") == "1"


def _log(msg):
    print(f"[engine] stallwatch: {msg}", flush=True)


def psi_full_us(path=PSI):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("full"):
                    return int(line.rsplit("total=", 1)[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _on_gc(phase, info):
    gen = info.get("generation")
    if phase == "start":
        _gc_t0[gen] = time.perf_counter()
        return
    t0 = _gc_t0.pop(gen, None)
    if t0 is None:
        return
    ms = (time.perf_counter() - t0) * 1e3
    if ms >= float(os.environ.get("SPLIT_NV_STALLWATCH_GC_MS", "20")):
        stats["gc_slow"] += 1
        stats["gc_max_ms"] = max(stats["gc_max_ms"], round(ms, 1))
        _log(f"gc gen{gen} {ms:.0f} ms, collected {info.get('collected')}, counts {gc.get_count()}, "
             f"frozen {gc.get_freeze_count()}")


class Ticker(threading.Thread):
    def __init__(self, limit_s, psi_path=PSI, tick_s=TICK_S):
        super().__init__(daemon=True, name="stallwatch")
        self.limit_s, self.psi_path, self.tick_s = limit_s, psi_path, tick_s
        self.stop = threading.Event()

    def run(self):
        last, psi = time.perf_counter(), psi_full_us(self.psi_path)
        while not self.stop.wait(self.tick_s):
            now = time.perf_counter()
            late = now - last - self.tick_s
            psi_now = psi_full_us(self.psi_path)
            if late >= self.limit_s:
                ms = late * 1e3
                stats["stalls"] += 1
                stats["stall_max_ms"] = max(stats["stall_max_ms"], round(ms, 1))
                mem = f", memory-full {(psi_now - psi) / 1e3:.0f} ms" if psi is not None and psi_now is not None else ""
                ctx = ""
                if _ctx is not None:
                    try:
                        ctx = f", {_ctx()}"
                    except Exception as e:  # noqa: BLE001
                        ctx = f", ctx error {e!r}"
                _log(f"host stall {ms:.0f} ms{mem}{ctx}")
            last, psi = now, psi_now


def install(ctx=None):
    """Start the watch on this rank (idempotent). ctx() describes what the rank is doing when a stall is logged."""
    global _ctx
    _ctx = ctx
    if _on_gc not in gc.callbacks:
        gc.callbacks.append(_on_gc)
    t = Ticker(float(os.environ.get("SPLIT_NV_STALLWATCH_MS", "100")) / 1e3)
    t.start()
    return t


def freeze():
    """gc.freeze() after warm-up (SPLIT_NV_GC_FREEZE=1); returns the number of objects moved out of collection."""
    if os.environ.get("SPLIT_NV_GC_FREEZE", "0") != "1":
        return 0
    gc.collect()
    gc.freeze()
    stats["frozen"] = gc.get_freeze_count()
    _log(f"gc.freeze(): {stats['frozen']} objects in the permanent generation")
    return stats["frozen"]


def summary():
    return dict(stats, enabled=enabled())


__all__ = ["enabled", "install", "freeze", "summary", "psi_full_us", "Ticker", "stats"]
