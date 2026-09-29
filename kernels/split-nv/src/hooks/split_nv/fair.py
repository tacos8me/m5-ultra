"""Prefill admission (the front end's prefill gate): FIFO, plus an optional short-prompt bypass.

Every prefill (and anything else that needs the prefill lock: cache clear, dev exec) enters the gate. Entrants run one at
a time in arrival order, holding the re-entrant prefill lock as before. With bypass_tokens > 0
(SPLIT_NV_BYPASS_TOKENS, default 0 = off), a running prefill that is longer than that checks the gate between its chunks
(prefill_yield): if a waiting prompt needs at most bypass_tokens new tokens, the long prefill releases the lock entirely
(Condition.wait on the lock), the short one runs to completion (reply and snapshot included), and the long one resumes.
Shorts are served in arrival order; a long prefill stays suspended for at most `share` of its own elapsed time.

The engine keeps per-session capture state while prefills interleave (Engine._cap_use), and the front end reserves
the suspended prefills' remaining KV in admission, so interleaving is invisible to either prompt's bytes.
"""
import contextlib
import threading
import time


class Ticket:
    __slots__ = ("new_tokens", "t_enter", "t_run", "suspended_s", "bypass")

    def __init__(self, new_tokens):
        self.new_tokens = new_tokens  # None: not eligible for a bypass
        self.t_enter = time.monotonic()
        self.t_run = None
        self.suspended_s = 0.0
        self.bypass = False


class PrefillGate:
    def __init__(self, lock, bypass_tokens=0, share=0.5):
        self.lock = lock  # the front end's re-entrant prefill lock
        self.rcv = threading.Condition(lock)  # a suspended prefill waits here with the lock fully released
        self.mu = threading.Lock()
        self.cv = threading.Condition(self.mu)  # entrants wait here for their turn (without the prefill lock)
        self.bypass_tokens = bypass_tokens
        self.share = share
        self.fifo = []
        self.active = None
        self.bypass = None
        self.suspended = []
        self.tickets = {}  # thread ident -> (ticket, depth)
        self.stats = {"entered": 0, "bypasses": 0, "suspended_s": 0.0}

    def limit(self):
        """bypass_tokens, overridable at runtime by the perf flag bypass_tokens (0 = off) once enabled by env."""
        if self.bypass_tokens <= 0:
            return 0
        from split_nv.perf_flags import flag
        return int(flag("bypass_tokens", self.bypass_tokens))

    def small(self, t):
        lim = self.limit()
        return lim > 0 and t.new_tokens is not None and t.new_tokens <= lim

    def _may_run(self, t):
        if self.bypass is t:
            return True
        return self.active is None and self.bypass is None and not self.suspended and self.fifo and self.fifo[0] is t

    @contextlib.contextmanager
    def enter(self, new_tokens=None):
        """Hold the gate (re-entrant per thread). new_tokens: new tokens this prefill will compute (None = unknown)."""
        me = threading.get_ident()
        held = self.tickets.get(me)
        if held is not None:
            self.tickets[me] = (held[0], held[1] + 1)
            try:
                yield held[0]
            finally:
                t, d = self.tickets[me]
                self.tickets[me] = (t, d - 1)
            return
        t = Ticket(new_tokens)
        with self.mu:
            self.fifo.append(t)
            self.cv.wait_for(lambda: self._may_run(t))
            self.fifo.remove(t)
            t.bypass = self.bypass is t
            self.active = t
            t.t_run = time.monotonic()
            self.stats["entered"] += 1
            self.stats["bypasses"] += t.bypass
        self.lock.acquire()
        self.tickets[me] = (t, 1)
        try:
            yield t
        finally:
            del self.tickets[me]
            with self.mu:
                if self.active is t:
                    self.active = None
                if self.bypass is t:
                    self.bypass = None
                self.cv.notify_all()
            with self.rcv:  # wake a suspended prefill (we still hold the lock here)
                self.rcv.notify_all()
            self.lock.release()

    def prefill_yield(self):
        """Called by the running prefill between its chunks (no chunk of its own queued). Runs waiting short prompts
        first when the bypass is on and this prefill is not itself short. Returns the seconds spent suspended."""
        if self.limit() <= 0:
            return 0.0
        me = threading.get_ident()
        held = self.tickets.get(me)
        if held is None:
            return 0.0
        t = held[0]
        if t.bypass or self.small(t):
            return 0.0
        waited = 0.0
        while True:
            with self.mu:
                cand = next((w for w in self.fifo if self.small(w)), None)
                elapsed = time.monotonic() - t.t_run
                if cand is None or t.suspended_s > self.share * elapsed:
                    return waited
                self.bypass = cand
                self.active = None
                self.suspended.append(t)
                self.cv.notify_all()
            t0 = time.monotonic()
            with self.rcv:
                self.rcv.wait_for(lambda: self.bypass is None and self.active is None and self.suspended[-1] is t)
            dt = time.monotonic() - t0
            with self.mu:
                self.suspended.remove(t)
                self.active = t
                t.suspended_s += dt
                self.stats["suspended_s"] += dt
            waited += dt

    def summary(self):
        with self.mu:
            return {"waiting": len(self.fifo), "suspended": len(self.suspended), "bypass_tokens": self.bypass_tokens,
                    **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stats.items()}}
