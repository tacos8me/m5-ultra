"""CPU test of the prefill gate (split_nv.fair) through the real Front.prefill with a fake GPU, plus the engine's
per-session capture swap (Engine._cap_use).

  * FIFO when the bypass is off: a short prompt behind a long one finishes after it.
  * bypass on: the short one runs between the long one's chunks and finishes first; both prompts' row parts
    (on_rows order and bytes) and chunk plans equal their solo runs.
  * share cap: a stream of shorts keeps the long one suspended for at most `share` of its elapsed time.
  * re-entrancy (an outer gate.enter around prefill, as cache=1 OPENs do) and cache-clear through the gate.
  * Engine._cap_use: interleaved begin/chunk/end of two sessions keep their capture fields apart.
usage: CUDA_VISIBLE_DEVICES= python tools/fair/test_gate.py"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hooks"))
os.environ["SPLIT_NV_DIR"] = "/tmp/split-nv-fair-test"
os.makedirs("/tmp/split-nv-fair-test", exist_ok=True)
import torch  # noqa: E402
from split_nv import front as F, engine as EM  # noqa: E402
from split_nv.fair import PrefillGate  # noqa: E402

SOURCE = {2: 2, 8: 2, 14: 2, 20: 1}
GPU = threading.Lock()
MS_PER_KROW = 3.0  # fake GPU: 3 ms per 1024 rows


def rows_for(sid, a, e):
    out = {}
    for L, r in SOURCE.items():
        lo, hi = a // r, e // r
        base = torch.arange(lo, hi, dtype=torch.int64) * 7 + sid * 1000 + L
        out[L] = (base.repeat_interleave(288).view(-1, 288).to(torch.uint8), base.repeat_interleave(68).view(-1, 68).to(torch.uint8))
    return out


class FJob:
    def __init__(self, fn):
        self.ev = threading.Event()
        self.fn = fn
        self.r = None
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        with GPU:
            self.r = self.fn()
        self.ev.set()

    def wait(self):
        self.ev.wait()
        return self.r


def make(bypass, share=0.5):
    f = F.Front.__new__(F.Front)
    f.prefill_lock = threading.RLock()
    f.gate = PrefillGate(f.prefill_lock, bypass, share)
    f.identity, f.token_map = "t", None
    f.share_chunk = 2048
    f.pf_need = {}
    f.preempt_on = False
    f.trim_min = 0
    f.min_resume = 1024
    f.contended = lambda sid: False
    f.admit = lambda n1, sid=None: None
    f.log = []
    pos = {}

    def submit_async(cmd, priority=0):
        sid = cmd[1]

        def fn():
            a = pos.get(sid, 0)
            pos[sid] = a + len(cmd[2])
            f.log.append((sid, a, pos[sid]))
            time.sleep(len(cmd[2]) / 1024 * MS_PER_KROW / 1000)
            return rows_for(sid, a, pos[sid]), 0.0
        return FJob(fn)

    f.submit_async = submit_async
    f.submit = lambda cmd, priority=0: {"prefill_end": {"final": 1}}.get(cmd[0])
    return f


def prefill(f, sid, n1, out, outer=False):
    seen = []
    t0 = time.monotonic()
    if outer:
        with f.gate.enter(n1):
            _, _, info = f.prefill(sid, list(range(n1 + 1)), use_cache=False, on_rows=lambda p: seen.append(
                {L: (tuple(v[0].shape), int(v[0].sum()), int(v[1].sum())) for L, v in p.items()}))
    else:
        _, _, info = f.prefill(sid, list(range(n1 + 1)), use_cache=False, on_rows=lambda p: seen.append(
            {L: (tuple(v[0].shape), int(v[0].sum()), int(v[1].sum())) for L, v in p.items()}))
    out[sid] = (seen, time.monotonic() - t0, time.monotonic(), info.get("suspended_s", 0.0))


F.assemble_parts = lambda *a, **k: ({}, {})


def scenario(bypass, shorts=((1, 0.02, 8192),), long_n=65536, outer=False, share=0.5):
    f = make(bypass, share)
    out = {}
    th = [threading.Thread(target=prefill, args=(f, 100, long_n, out))]
    th[0].start()
    t0 = time.monotonic()
    for sid, delay, n in shorts:
        time.sleep(max(0.0, t0 + delay - time.monotonic()))
        t = threading.Thread(target=prefill, args=(f, sid, n, out, outer))
        t.start()
        th.append(t)
    for t in th:
        t.join()
    return f, out


def solo(sid, n):
    f = make(0)
    out = {}
    prefill(f, sid, n, out)
    return out[sid][0]


ok = True
ref_long, ref_short = solo(100, 65536), solo(1, 8192)
# 1. FIFO (bypass off)
f, out = scenario(0)
fifo_short_after = out[1][2] > out[100][2]
print(f"FIFO: long {out[100][1] * 1e3:.0f} ms, short {out[1][1] * 1e3:.0f} ms (finished after long: {fifo_short_after})")
ok &= fifo_short_after and out[1][0] == ref_short and out[100][0] == ref_long
# 2. bypass
for outer in (False, True):
    f, out = scenario(16384, outer=outer)
    first = out[1][2] < out[100][2]
    same = out[1][0] == ref_short and out[100][0] == ref_long
    print(f"bypass (outer enter {outer}): long {out[100][1] * 1e3:.0f} ms (suspended {out[100][3] * 1e3:.0f}), "
          f"short {out[1][1] * 1e3:.0f} ms; short first {first}; rows identical to solo {same}; gate {f.gate.summary()}")
    ok &= first and same and f.gate.stats["bypasses"] == 1
# 3. share cap under a stream of shorts
shorts = tuple((i, 0.01 + 0.012 * i, 4096) for i in range(1, 30))
f, out = scenario(16384, shorts=shorts, share=0.3)
el = out[100][1]
sus = out[100][3]
print(f"stream of 29 shorts, share 0.3: long {el * 1e3:.0f} ms, suspended {sus * 1e3:.0f} ms ({sus / el:.2f}); "
      f"bypasses {f.gate.stats['bypasses']}")
ok &= sus <= 0.3 * el + 0.05 and out[100][0] == ref_long and all(out[i][0] == solo(i, 4096) for i in range(1, 30, 7))
# 4. a long prompt never bypasses; another long waits FIFO
f, out = scenario(16384, shorts=((2, 0.02, 32768),))
print(f"long behind long: second finished after first: {out[2][2] > out[100][2]}")
ok &= out[2][2] > out[100][2] and f.gate.stats["bypasses"] == 0


# 5. Engine._cap_use
class Cap:
    enabled = True

    def reset(self):
        for fld in EM.CAP_FIELDS:
            setattr(self, fld, {"tokens": [], "ntok": 0, "rows": {}, "swa": {}, "tail": {}, "chunk_rows": {}, "chunks": []}.get(fld))


e = EM.Engine.__new__(EM.Engine)
e.cap = Cap()
e.cap_swap, e.cap_owner, e.cap_saved = True, None, {}
e.sessions = {}


def begin(sid):
    e.sessions[sid] = object()
    e._cap_use(sid)
    e.cap.reset()


def chunk(sid, n):
    e._cap_use(sid)
    e.cap.tokens.append(n)
    e.cap.ntok += n
    e.cap.rows[sid] = e.cap.rows.get(sid, 0) + n


def end(sid):
    e._cap_use(sid)
    r = (list(e.cap.tokens), e.cap.ntok, dict(e.cap.rows))
    e._cap_done(sid)
    return r


begin(1); chunk(1, 8192); chunk(1, 8192)
begin(2); chunk(2, 100)
r2 = end(2)
chunk(1, 5)
r1 = end(1)
print("cap swap:", r1, r2)
ok &= r1 == ([8192, 8192, 5], 16389, {1: 16389}) and r2 == ([100], 100, {2: 100}) and not e.cap_saved
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
