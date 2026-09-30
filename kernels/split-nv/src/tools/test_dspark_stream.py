"""CPU test of the drafter's side-stream ordering (split_nv.dspark_box.BoxDrafter launch_draft / collect / drain and
engine.cmd_stepd), with the real methods and instrumented stand-ins for the CUDA stream, event and graph primitives.

Checks, as a log of (operation, stream):
- a graphed draft replays on the side stream, after the side stream waited for the main stream (the last step, a
  preempted prefill's queued segments), and its staging copies and completion event are on the side stream too;
- the step bookkeeping of cmd_stepd runs between launch and collect on the main stream;
- collect synchronizes the draft's event and orders the main stream after the side stream before the step;
- with the live flag dspark_side_stream off, without a graph for the width, or without a side stream: main stream only;
- rows beyond one block are appended on the main stream first, then the side stream waits for them;
- a failing prepare drains the draft (main ordered after it) before the error propagates.
usage: CUDA_VISIBLE_DEVICES= python tools/test_dspark_stream.py
"""
import contextlib
import json
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TMP = tempfile.mkdtemp(prefix="dspark-stream-test-")
os.environ["SPLIT_NV_DIR"] = TMP
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))
import torch  # noqa: E402

from split_nv import dspark_box as DB  # noqa: E402
from split_nv import dspark_wire as W  # noqa: E402
from split_nv import engine as EM  # noqa: E402
from split_nv import perf_flags  # noqa: E402

RESULTS = []
LOG = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {str(detail)[:600]}"), flush=True)


class Stream:
    def __init__(self, name):
        self.name = name

    def wait_stream(self, other):
        LOG.append(("wait", self.name, other.name))


MAIN = Stream("main")
CUR = [MAIN]


def current_stream():
    return CUR[-1]


@contextlib.contextmanager
def stream(s):
    CUR.append(s)
    try:
        yield
    finally:
        CUR.pop()


class Event:
    def record(self):
        self.on = CUR[-1].name
        LOG.append(("record", self.on))

    def synchronize(self):
        LOG.append(("sync", self.on))


class Graph:
    def __init__(self, name):
        self.name = name

    def replay(self):
        LOG.append(("replay:" + self.name, CUR[-1].name))


class LoggedTensor:
    """A staging destination: records the stream its H2D copy was queued on."""

    def __init__(self, name, t):
        self.name, self.t = name, t

    def copy_(self, src, non_blocking=False):
        LOG.append(("h2d:" + self.name, CUR[-1].name))
        self.t.copy_(src)


torch.cuda.current_stream = current_stream
torch.cuda.stream = stream
torch.cuda.Event = Event


def set_flags(d):
    with open(os.path.join(TMP, "box-perf-flags.json"), "w") as f:
        json.dump(d, f)
    perf_flags._state["t"] = -1e9


def drafter(side=True, graphs=(4, "append")):
    D = DB.BoxDrafter.__new__(DB.BoxDrafter)
    D.side = Stream("side") if side else None
    D.on_side = None
    D.staged = None
    D.graphs = {k: Graph(str(k)) for k in graphs}
    D.noise = 7
    D.h_meta = torch.zeros(5, DB.BLOCK_MAX, dtype=torch.int64)
    D.h_taps = torch.zeros(DB.BLOCK_MAX, DB.TAP, dtype=torch.bfloat16)
    D.taps = LoggedTensor("taps", torch.zeros(DB.BLOCK_MAX, DB.TAP, dtype=torch.bfloat16))
    D.meta = LoggedTensor("meta", torch.zeros(5, DB.BLOCK_MAX, dtype=torch.int64))
    D.h_toks = torch.arange(10, 10 + DB.BLOCK_MAX)
    D.h_probs = torch.full((DB.BLOCK_MAX,), 0.5)
    D.stats = {"drafts": 0, "appends": 0, "rings": 0}
    D.full = lambda W_: LOG.append(("eager_full", CUR[-1].name))
    D.append = lambda: LOG.append(("eager_append", CUR[-1].name))
    return D


def taps(n):
    return bytes(n * W.TAP_ROW_BYTES)


def ops(kind):
    return [x for x in LOG if x[0].startswith(kind)]


def main():
    set_flags({})
    # 1. side stream: wait, staging, replay, event on side; collect syncs and orders main after side
    D = drafter()
    LOG.clear()
    D.launch_draft(0, 2000, 5, 1997, 3, taps(3), 4)
    LOG.append(("prepare_copy", CUR[-1].name))
    toks, probs = D.collect(4)
    seq = [x for x in LOG]
    i_wait = seq.index(("wait", "side", "main"))
    i_rep = seq.index(("replay:4", "side"))
    check("side: the side stream waits for main before anything of the draft", i_wait == 0, seq)
    check("side: staging copies and the replay are on the side stream, after the wait",
          ("h2d:taps", "side") in seq and ("h2d:meta", "side") in seq and i_wait < i_rep, seq)
    check("side: the completion event is recorded on the side stream after the replay",
          seq.index(("record", "side")) > i_rep, seq)
    check("side: the host bookkeeping ran on main between launch and collect",
          seq.index(("prepare_copy", "main")) > seq.index(("record", "side")) and seq.index(("prepare_copy", "main")) < seq.index(("sync", "side")), seq)
    check("side: collect syncs the event, then orders main after side", seq[-2:] == [("sync", "side"), ("wait", "main", "side")], seq)
    check("side: drafts returned from the pinned outputs", toks == [10, 11, 12, 13] and probs == [0.5] * 4, (toks, probs))
    check("side: nothing left on the side stream after collect", D.on_side is None)
    LOG.clear()
    D.collect(4)
    check("collect twice: no second main-after-side ordering", ("wait", "main", "side") not in LOG, LOG)

    # 2. live flag off -> main stream only
    set_flags({"dspark_side_stream": False})
    LOG.clear()
    D.launch_draft(0, 2003, 5, 2000, 3, taps(3), 4)
    D.collect(4)
    check("flag off: replay and event on main, no stream waits", ("replay:4", "main") in LOG and not ops("wait"), LOG)
    set_flags({})

    # 3. no graph for the width -> eager on main
    D2 = drafter(graphs=("append",))
    LOG.clear()
    D2.launch_draft(0, 2000, 5, 1997, 3, taps(3), 4)
    D2.collect(4)
    check("no graph: eager draft on main, no stream waits", ("eager_full", "main") in LOG and not ops("wait"), LOG)

    # 4. no side stream (SPLIT_NV_DSPARK_STREAM=main)
    D3 = drafter(side=False)
    LOG.clear()
    D3.launch_draft(0, 2000, 5, 1997, 3, taps(3), 4)
    D3.collect(4)
    check("stream=main: replay on main, no stream waits", ("replay:4", "main") in LOG and not ops("wait"), LOG)

    # 5. more rows than one block: appended on main first, then the side stream waits for them
    LOG.clear()
    D.launch_draft(0, 2008, 5, 2000, 8, taps(8), 4)
    D.collect(4)
    seq = list(LOG)
    check("8 rows: 3 appended on main, then side waits, then 5 staged with the draft on side",
          seq.index(("replay:append", "main")) < seq.index(("wait", "side", "main")) < seq.index(("replay:4", "side")), seq)
    check("8 rows: appends counted", D.stats["appends"] >= 8, D.stats)

    # 6. engine.cmd_stepd: prepare between launch and collect; a failing prepare drains the draft
    class Sess:
        pending = (1990, [1, 2, 3, 4, 5])
        length = 1995

    class Steps:
        def __init__(self, fail):
            self.fail = fail

        def pick(self, L):
            class S:
                W = 5
            return S()

        def prepare(self, sess, keep, upto):
            LOG.append(("prepare", CUR[-1].name))
            if self.fail:
                raise RuntimeError("KV pool exhausted")

        def run(self, sess, keep, ids, use_graph=True):
            LOG.append(("step", CUR[-1].name))
            return {"payload": b"", "L": len(ids)}, 0.0

    class Eng:
        pass

    for fail in (False, True):
        e = Eng()
        e.sessions = {1: Sess()}
        e.dspark = drafter()
        e.steps = Steps(fail)
        e.use_graph = True
        LOG.clear()
        costs = [24.1, 26.1, 28.2, 30.8]
        try:
            out, _, info = EM.Engine.cmd_stepd(e, 1, 0, 1993, 5, 1990, 3, taps(3), True, 4, 4, costs, [], (1990, 5))
            err = None
        except RuntimeError as ex:
            err, info = ex, None
        seq = list(LOG)
        if not fail:
            check("cmd_stepd: launch (side) -> prepare (main) -> collect -> step (main)",
                  seq.index(("replay:4", "side")) < seq.index(("prepare", "main")) < seq.index(("wait", "main", "side"))
                  < seq.index(("step", "main")) and info["drafted"] and len(info["split_ms"]) == 3, (seq, info))
        else:
            check("cmd_stepd: a failing prepare drains the draft before the error propagates",
                  err is not None and seq[-2:] == [("sync", "side"), ("wait", "main", "side")] and ("step", "main") not in seq,
                  (err, seq))
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED", f"({sum(RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
