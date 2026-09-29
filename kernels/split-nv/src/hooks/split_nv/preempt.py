"""Layer-granular preemption of prefill chunks by decode STEPs (SPLIT_NV_PREEMPT=1, default off).

Today a STEP that arrives while a prefill chunk runs waits for the whole chunk; to bound that wait the front end cuts
chunks into SPLIT_NV_SHARE_CHUNK-row pieces whenever another session decodes (2048 rows: ~120 ms per piece at any
depth, unsplit, so the decoder gets one step per piece and the prefill loses the overlapped 8K chunk).

With this module the front end keeps full (overlapped) 8K chunks and marks them preemptible while a session decodes.
Every decoder-layer entry of such a chunk (both pf_overlap halves) is a yield point. On rank 0 a point
  1. records a CUDA event and waits for the event LAG points back, so the GPU never has more than ~LAG segments of this
     chunk queued (a step enqueued now runs after at most that much prefill work);
  2. takes the STEPs the front end has parked (graph widths only, within the decode time share of this chunk);
  3. sends ("pp", k, [step commands]) to every other rank -- one message per point, also when empty;
  4. runs those steps inline on the current thread and hands the results to the waiting connection threads.
Other ranks block at each point for rank 0's message and run the same steps there, so every rank issues the same
kernels and collectives in the same order. Uncontended chunks have no points at all (no messages, no syncs).

Numerics: a step runs between two layers of the prefill on the same stream, after the prefill's in-flight all-reduces
(fence()). It writes only its own session's KV slots, its graph's static buffers, its Engram history row and the og-moe
valid-row count, which it resets to "all rows" behind its graph exactly as between chunks. The attention backend the
graphs were captured with is also half A's prefill backend, so its instance attributes are saved before and restored
after the step, together with the process-global forward state pf_overlap already carries between halves. The
prefill's arithmetic is unchanged; the step's arithmetic is the graph replay it is between chunks.
"""
import os
import time

import torch

LAG = int(os.environ.get("SPLIT_NV_PREEMPT_LAG", "2"))
SHARE = float(os.environ.get("SPLIT_NV_PREEMPT_SHARE", "0.5"))  # max fraction of a chunk's wall time spent in steps

_win = None  # the preemptible chunk running on this rank (one at a time)


def enabled():
    return os.environ.get("SPLIT_NV_PREEMPT", "0") == "1"


class Window:
    def __init__(self, engine, rank0, peers, take, lag=LAG):
        self.engine = engine
        self.rank0 = rank0
        self.peers = peers  # rank 0: connections to the other ranks; others: the connection from rank 0
        self.take = take  # rank 0: callable(window) -> list of parked step jobs (front end)
        self.lag = lag
        self.k = 0
        self.events = []
        self.t0 = time.perf_counter()
        self.step_s = 0.0
        self.steps = 0
        self.busy = False
        self.t_point = None  # host time of the last point (rank-local): head = open -> first point, gap = point -> point
        self.head_s = None
        self.max_gap_s = 0.0
        self.max_lag_s = 0.0  # longest wait for the GPU LAG points back (prefill work a step would queue behind)

    def point(self):
        if self.busy:  # never nested (an inline step's own forward, if it had points)
            return
        now = time.perf_counter()
        if self.t_point is None:
            self.head_s = now - self.t0
        else:
            self.max_gap_s = max(self.max_gap_s, now - self.t_point)
        k = self.k
        self.k += 1
        if not self.rank0:
            msg = self.peers.recv()
            if not (isinstance(msg, tuple) and msg[0] == "pp" and msg[1] == k):
                raise RuntimeError(f"preempt: expected point {k}, got {msg!r:.200}")
            for cmd in msg[2]:
                self._run(cmd)
            self.t_point = time.perf_counter()
            return
        if self.lag > 0:
            ev = torch.cuda.Event()
            ev.record()
            self.events.append(ev)
            if len(self.events) > self.lag:
                t = time.perf_counter()
                self.events.pop(0).synchronize()
                self.max_lag_s = max(self.max_lag_s, time.perf_counter() - t)
        jobs = self.take(self)
        cmds = [j.cmd for j in jobs]
        for conn in self.peers:
            conn.send(("pp", k, cmds))
        try:
            self._run_jobs(jobs)
        finally:
            self.t_point = time.perf_counter()

    def _run_jobs(self, jobs):
        for i, job in enumerate(jobs):
            job.t_get = job.t_bcast = time.perf_counter()
            try:
                job.result = self._run(job.cmd)
            except Exception as e:  # noqa: BLE001
                import traceback
                job.error = f"{e}\n{traceback.format_exc()}"
                for rest in jobs[i + 1:]:  # never leave a connection thread waiting
                    rest.error = f"preempted chunk failed: {e}"
                    rest.done.set()
                raise
            finally:
                job.t_exec = time.perf_counter()
                job.done.set()

    def _run(self, cmd):
        t0 = time.perf_counter()
        self.busy = True
        try:
            return run_inline(self.engine, cmd)
        finally:
            self.busy = False
            self.step_s += time.perf_counter() - t0
            self.steps += 1

    def share_ok(self, share=None):
        return self.step_s <= (SHARE if share is None else share) * (time.perf_counter() - self.t0)


def point():
    w = _win
    if w is not None:
        w.point()


def open_window(engine, rank0, peers, take):
    global _win
    lag = LAG
    if rank0:  # runtime A/B (perf flags preempt_lag, preempt_share); only rank 0 throttles or decides
        from split_nv.perf_flags import flag
        lag = int(flag("preempt_lag", LAG))
    _win = Window(engine, rank0, peers, take, lag)
    return _win


def close_window():
    global _win
    w, _win = _win, None
    return w


def run_inline(engine, cmd):
    """Execute a STEP inside a running prefill forward, leaving the prefill's host-side state as it found it."""
    from sglang.srt.model_executor import forward_context as fc
    from split_nv import pf_overlap

    be = engine.steps.be
    saved = dict(vars(be))
    glob = pf_overlap._save_globals()
    prev_ctx = fc.set_forward_context(None)
    mr_be = engine.mr.attn_backend
    pf_overlap.fence()
    try:
        return engine.execute(cmd)
    finally:
        for name, value in saved.items():
            setattr(be, name, value)
        engine.mr.attn_backend = mr_be
        fc.set_forward_context(prev_ctx)
        pf_overlap._restore_globals(glob)


__all__ = ["enabled", "point", "open_window", "close_window", "run_inline", "Window", "LAG", "SHARE"]
