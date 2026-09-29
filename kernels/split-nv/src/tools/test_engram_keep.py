"""CPU-only test of split_nv.engram_keep against the real SGLang _HostTable (no GPU: cudaHostRegister is stubbed).

Runs inside a GPU-less container from the production image, in a PRIVATE IPC namespace so no host segment is seen:
  docker run --rm --network none --ipc=private --entrypoint python3 -e CUDA_VISIBLE_DEVICES= \\
    -e PYTHONPATH=/home/ian/split-nv/hooks -v <this tree>:/home/ian/split-nv:ro sglang-dsv41-split:6152b54 \\
    /home/ian/split-nv/tools/test_engram_keep.py
Two TP ranks are threads sharing a barrier and a broadcast slot, like the gloo CPU group.
  1. keep off: install() patches nothing and drops kept segments nobody has attached;
  2. first start: segment created, both halves filled by SGLang's fill_from_source, COMPLETE only after the barrier,
     registered on both ranks, bytes == source;
  3. restart: segment reused (no fill), registration on a side thread started at construction, bytes == source;
  4. refill (fresh segment, old one removed) when the source changed, the fill never completed, a block differs,
     or the table size changed;
  5. a failed async registration raises from finish_load.
"""
import os
import sys
import tempfile
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SGLANG_DSV41_ENGRAM_PINNED_RESERVE_GIB"] = "0"
os.environ["SPLIT_NV_ENGRAM_KEEP_KEY_BASE"] = "0x7E570000"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))

from sglang.srt.layers import engram as E  # noqa: E402

from split_nv import engram_keep as K  # noqa: E402

NAME = "sglang_engram_1"
KEY = K.KEY_BASE + 1


class Group:
    """Per-rank handle on a 2-rank group: barrier + broadcast_object(src=0) in call order."""

    def __init__(self, shared, rank):
        self.shared, self.rank_in_group, self.world_size, self.n = shared, rank, 2, 0

    def barrier(self):
        self.shared["barrier"].wait(timeout=30)

    def broadcast_object(self, obj=None, src=0):
        i, self.n = self.n, self.n + 1
        with self.shared["cv"]:
            if self.rank_in_group == src:
                self.shared["slots"][i] = obj
                self.shared["cv"].notify_all()
            else:
                self.shared["cv"].wait_for(lambda: i in self.shared["slots"], timeout=30)
            return self.shared["slots"][i]


def groups():
    shared = {"barrier": threading.Barrier(2), "cv": threading.Condition(), "slots": {}}
    return [Group(shared, r) for r in range(2)]


def on_ranks(fn):
    out, errs = [None, None], []

    def run(r):
        try:
            out[r] = fn(r)
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]
    return out


REG, FILLS, FAIL = [], [], []
orig_fill = E._HostTable.fill_from_source


def fake_register(self):
    if FAIL:
        raise RuntimeError("register failed")
    time.sleep(0.05)
    REG.append(threading.current_thread().name)
    self.registered, self.register_seconds = True, 0.05


def spy_fill(self, threads, reserve):
    FILLS.append(self.group.rank_in_group)
    return orig_fill(self, threads, reserve)


def start(src, nbytes):
    """One engine start: construct + finish_load on both ranks; returns the two tables."""
    gs = groups()
    tables = on_ranks(lambda r: E._HostTable("shared", nbytes, NAME, gs[r], True, source_path=src))
    on_ranks(lambda r: tables[r].finish_load("layer 1"))
    return tables


def stop(tables):
    for t in tables:  # process exit
        K.detach(t.keep.addr)


def expect(tables, data, reused, why=None):
    for t in tables:
        assert t.keep.reused == reused, (t.keep.reused, t.keep.why)
        assert why is None or why in t.keep.why, t.keep.why
        assert t.registered
        assert bytes(t.bytes.numpy()) == data
    segs = [s for k, s in K.segments().items() if k == KEY]
    assert len(segs) == 1 and segs[0]["size"] == K.seg_bytes(len(data)), K.segments()
    addr = K.attach(segs[0]["shmid"], readonly=True)
    try:
        assert K.header_at(addr, len(data)).state == K.COMPLETE
    finally:
        K.detach(addr)


def main():
    E._HostTable._register = fake_register
    E._HostTable.fill_from_source = spy_fill
    os.environ.pop("SPLIT_NV_ENGRAM_KEEP", None)
    orig_init = E._HostTable.__init__
    stale, busy = K.create(KEY, 1 << 20), K.create(KEY + 13, 1 << 20)
    addr = K.attach(busy)
    assert K.install() is False and E._HostTable.__init__ is orig_init
    segs = K.segments()
    assert KEY not in segs and segs[KEY + 13]["shmid"] == busy, segs  # unused dropped, attached one left alone
    K.detach(addr)
    K.remove(busy)
    print("ok  keep off: nothing patched; unused kept segments dropped, attached ones left")
    os.environ["SPLIT_NV_ENGRAM_KEEP"] = "1"
    assert K.install() is True and E._HostTable.__init__ is not orig_init

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, f"{NAME}.bin")
        nbytes = 3 * (1 << 20) + 123
        data = os.urandom(nbytes)
        with open(src, "wb") as f:
            f.write(data)

        REG.clear(), FILLS.clear()
        tables = start(src, nbytes)
        expect(tables, data, reused=False, why="no segment")
        assert sorted(FILLS) == [0, 1] and len(REG) == 2 and all(not n.startswith("engram-register") for n in REG)
        stop(tables)
        print("ok  first start: created, both halves filled, COMPLETE after the barrier, registered, bytes == source")

        REG.clear(), FILLS.clear()
        tables = start(src, nbytes)
        expect(tables, data, reused=True, why="blocks match")
        assert FILLS == [] and len(REG) == 2 and all(n.startswith("engram-register") for n in REG), (FILLS, REG)
        stop(tables)
        print("ok  restart: reused without a fill, registered on side threads")

        shmid_before = K.segments()[KEY]["shmid"]
        os.utime(src, ns=(time.time_ns(), time.time_ns() + 10**9))
        FILLS.clear()
        tables = start(src, nbytes)
        expect(tables, data, reused=False, why="source file changed")
        assert sorted(FILLS) == [0, 1] and K.segments()[KEY]["shmid"] != shmid_before
        stop(tables)
        print("ok  source changed: old segment removed, fresh one filled")

        seg = K.segments()[KEY]
        addr = K.attach(seg["shmid"])
        K.header_at(addr, nbytes).state = K.FILLING  # a start that died mid-fill
        K.detach(addr)
        tables = start(src, nbytes)
        expect(tables, data, reused=False, why="fill never completed")
        stop(tables)
        print("ok  interrupted fill: refilled")

        seg = K.segments()[KEY]
        addr = K.attach(seg["shmid"])
        ctypes_off = addr + 2 * K.PAGE + 7
        import ctypes
        ctypes.c_ubyte.from_address(ctypes_off).value ^= 0xFF
        K.detach(addr)
        os.environ["SPLIT_NV_ENGRAM_KEEP_SAMPLES"] = "full"
        tables = start(src, nbytes)
        expect(tables, data, reused=False, why="block 2 differs")
        stop(tables)
        os.environ.pop("SPLIT_NV_ENGRAM_KEEP_SAMPLES")
        print("ok  corrupted block: detected (SAMPLES=full) and refilled")

        small = nbytes - 5000
        with open(src, "wb") as f:
            f.write(data[:small])
        tables = start(src, small)
        expect(tables, data[:small], reused=False, why="size")
        stop(tables)
        print("ok  table size changed: new segment")

        FAIL.append(1)
        try:
            start(src, small)
        except RuntimeError as e:
            assert "register failed" in str(e)
        else:
            raise AssertionError("a failed async registration must raise from finish_load")
        finally:
            FAIL.clear()
        print("ok  failed async registration raises from finish_load")

        for k, s in K.segments().items():
            if k == KEY:
                K.remove(s["shmid"])
    print("all passed")


if __name__ == "__main__":
    main()
