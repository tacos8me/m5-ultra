"""CPU test of the decode-gap attribution tools (no GPU, no engine).

  * split_nv.stallwatch: the ticker logs a host stall when one long C call holds the GIL; the gc callback logs a slow
    full collection; gc.freeze (SPLIT_NV_GC_FREEZE=1) takes a large heap out of later full collections; the cgroup
    memory-pressure reader parses the PSI format.
  * fair_gate: a GIL-holding copy in the gate's own process (what parsing the 1M STAT blob does) shows up as a decode
    "step gap" with in-process decoder threads and not with decoder processes; tail() counts and ranks the gaps.
usage: CUDA_VISIBLE_DEVICES= python tools/fair/test_stallwatch.py"""
import argparse
import gc
import io
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hooks"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from split_nv import stallwatch as SW  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")


def hold_gil(seconds):
    """One C call that keeps the GIL (bytes(bytearray) of a large buffer, like og_client.parse_safetensors)."""
    buf = bytearray(64 << 20)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        bytes(buf) * 1  # ~10-20 ms per copy with the GIL held; the loop between copies gives it back briefly
        n += 1
    big = bytearray(int(1.2e9 * seconds))  # plus one single long hold: a ~seconds*1.2 GB copy
    t1 = time.perf_counter()
    bytes(big)
    return time.perf_counter() - t1


# 1. host stall: one long GIL-holding copy is reported by the ticker with its length
os.environ["SPLIT_NV_STALLWATCH_MS"] = "100"
out = io.StringIO()
with redirect_stdout(out):
    t = SW.install(lambda: "test ctx")
    time.sleep(0.05)
    held = hold_gil(0.3)
    time.sleep(0.05)
logged = [l for l in out.getvalue().splitlines() if "host stall" in l]
check("stall logged for a long GIL hold", SW.stats["stalls"] >= 1 and logged and "test ctx" in logged[-1],
      f"(single hold {held * 1e3:.0f} ms, max stall {SW.stats['stall_max_ms']} ms: {logged[-1:] or out.getvalue()[-200:]})")
t.stop.set()

# 2. gc: a full collection over a large heap is logged; after gc.freeze it no longer walks that heap
heap = [(i, [i]) for i in range(1_500_000)]  # 3M container objects
out = io.StringIO()
with redirect_stdout(out):
    t0 = time.perf_counter()
    gc.collect()
    full_ms = (time.perf_counter() - t0) * 1e3
check("slow full collection logged", SW.stats["gc_slow"] >= 1 and "gc gen2" in out.getvalue(),
      f"({full_ms:.0f} ms over {len(gc.get_objects()) / 1e6:.1f}M objects)")
os.environ["SPLIT_NV_GC_FREEZE"] = "1"
with redirect_stdout(io.StringIO()):
    frozen = SW.freeze()
t0 = time.perf_counter()
gc.collect()
frozen_ms = (time.perf_counter() - t0) * 1e3
gc.unfreeze()
check("gc.freeze takes the heap out of full collections", frozen > 3_000_000 and frozen_ms < full_ms / 5,
      f"(frozen {frozen}, full collection {full_ms:.0f} -> {frozen_ms:.1f} ms)")
del heap
gc.collect()

# 3. PSI reader
with tempfile.NamedTemporaryFile("w", suffix=".pressure", delete=False) as f:
    f.write("some avg10=0.00 avg60=0.00 avg300=0.00 total=31676163\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=31675672\n")
check("memory.pressure full total parsed", SW.psi_full_us(f.name) == 31675672)
os.unlink(f.name)
check("missing memory.pressure -> None", SW.psi_full_us("/nonexistent/memory.pressure") is None)

# 4. fair_gate: the gate's own GIL holds are not decode gaps once decoders run as processes
import fair_gate as FG  # noqa: E402


class FakeSession:
    def __init__(self, host, port):
        pass

    def open(self, tokens, state="none"):
        return None

    def step(self, keep, ids):
        time.sleep(0.004)  # a fast box
        return b"payload", 0.004

    def close(self):
        pass


FG.Session = FakeSession
args = argparse.Namespace(host="x", port=0)
tokens = list(range(9000))
res = {}
for inproc in (True, False):
    d = (FG.Decoder if inproc else FG.DecoderProc)(args, tokens, 8193, 4, 0.02)
    d.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    bytes(bytearray(1200 << 20))  # one GIL-held 1.2 GB copy: the gate parsing a 1M STAT blob
    time.sleep(0.3)
    t1 = time.monotonic()
    d.close()
    res[inproc] = FG.tail(d.samples, t0 - 0.3, t1 + 0.3)
check("in-process decoders see the gate's GIL hold as a step gap", res[True]["during_over_100ms"] >= 1, str(res[True]))
check("decoder processes do not", res[False]["during_over_100ms"] == 0, str(res[False]))
check("tail() ranks the top gaps with their offsets", len(res[True]["top"]) == 5 and
      res[True]["top"][0][0] == max(x[0] for x in res[True]["top"]), str(res[True]["top"][:2]))

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
