"""CPU-only checks of the fast-restart changes (no GPU: CUDA_VISIBLE_DEVICES is emptied first).

  1. prefix_cache.IndexLoader builds the same index as the synchronous PrefixIndex (entries, blocks, bytes, dropped
     incomplete entries, orphan cleanup), in the background, and re-raises a failed build from get();
  2. Front.__init__ takes the loader and does not build a second index;
  3. teardown: front.hard_exit signals the parent (SIGUSR2) before exiting with its code; engine.kill_ranks SIGKILLs
     only live ranks; engine.exit_code keeps rank 0's code and maps killed ranks;
  4. tools/prefetch_weights.py reads every tensor of the view except the Engram embedding tables;
usage: python3 tools/test_fast_restart.py
"""
import inspect
import json
import os
import sys
import tempfile
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))

import numpy as np  # noqa: E402

from split_nv import prefix_cache as PC  # noqa: E402

NUM = "og-test"


def make_cache(root):
    d = os.path.join(root, NUM)
    os.makedirs(d)

    def w(name, n=64):
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"x" * n)

    for bid in ("b1", "b2", "b9"):  # b9 is referenced by nobody: deleted at load
        for r in (0, 1):
            w(f"blk-{bid}.r{r}", 1000)
        w(f"rows-{bid}", 300)
    for key, P, blocks in (("e1", 8200, ["b1"]), ("e2", 16400, ["b1", "b2"]), ("e3", 10, [])):
        np.save(os.path.join(d, f"tok-{key}.npy"), np.arange(P, dtype=np.uint64))
        with open(os.path.join(d, f"idx-{key}.json"), "w") as f:
            json.dump({"key": key, "P": P, "blocks": blocks, "capture": True}, f)
        w(f"rows-tail-{key}", 50)
        for r in (0, 1):
            w(f"ent-{key}.r{r}", 200)
    os.unlink(os.path.join(d, "ent-e3.r1"))  # incomplete: dropped at load
    w("ent-e1.r0.tmp123", 10)  # interrupted write: deleted at load
    return d


def test_loader_matches_sync():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        make_cache(a)
        make_cache(b)
        os.environ["SPLIT_NV_CACHE_DIR"] = a
        sync = PC.PrefixIndex(NUM, 1 << 30)
        os.environ["SPLIT_NV_CACHE_DIR"] = b
        loader = PC.IndexLoader(NUM, 1 << 30)
        idx, waited = loader.get()
        s1, s2 = sync.summary(), idx.summary()
        s1.pop("root"), s2.pop("root")
        assert s1 == s2, (s1, s2)
        assert s2["loaded"] == 2 and s2["dropped_at_load"] == 1 and s2["blocks"] == 2, s2
        assert sorted(os.listdir(os.path.join(a, NUM))) == sorted(os.listdir(os.path.join(b, NUM)))
        assert not any(".tmp" in f or "b9" in f for f in os.listdir(os.path.join(b, NUM)))
        assert loader.seconds >= 0 and waited >= 0
        assert idx.lookup(list(range(16400)) + [7, 8]).key == "e2"
    print("ok  IndexLoader == PrefixIndex (entries, blocks, bytes, drops, orphan cleanup, lookup)")


def test_loader_runs_in_background_and_reraises():
    orig = PC.PrefixIndex

    class Slow:
        def __init__(self, numerics, budget):
            time.sleep(0.3)
            raise OSError("boom")

    PC.PrefixIndex = Slow
    try:
        t0 = time.monotonic()
        loader = PC.IndexLoader(NUM, 1)
        assert time.monotonic() - t0 < 0.1, "constructor must not wait for the build"
        try:
            loader.get()
        except OSError as e:
            assert str(e) == "boom"
        else:
            raise AssertionError("get() must re-raise the build error")
    finally:
        PC.PrefixIndex = orig
    print("ok  IndexLoader starts without blocking and re-raises a failed build")


def test_budget_env():
    for k in ("SPLIT_NV_CACHE_GIB", "SPLIT_NV_CACHE_GB"):
        os.environ.pop(k, None)
    assert PC.budget_bytes() == 64 << 30
    os.environ["SPLIT_NV_CACHE_GB"] = "96"
    assert PC.budget_bytes() == 96 << 30
    os.environ["SPLIT_NV_CACHE_GIB"] = "1.5"
    assert PC.budget_bytes() == 3 << 29
    print("ok  budget_bytes (SPLIT_NV_CACHE_GIB > SPLIT_NV_CACHE_GB > 64)")


def test_front_uses_loader():
    import split_nv.front as F

    src = inspect.getsource(F.Front.__init__)
    assert "cache_loader" in inspect.signature(F.Front.__init__).parameters
    assert src.count("PrefixIndex(") == 1 and "cache_loader.get()" in src
    import split_nv.engine as E

    rm = inspect.getsource(E.rank_main)
    assert rm.index("IndexLoader(") < rm.index("Engine(server_args"), "the index must start before the model load"
    assert "Front(engine, server_args, cache_loader)" in rm
    print("ok  rank 0 starts the IndexLoader before Engine() and hands it to Front")


def test_hard_exit_signals_parent():
    import signal

    import split_nv.front as F

    got = []
    old = signal.signal(signal.SIGUSR2, lambda *_: got.append(1))
    try:
        pid = os.fork()
        if pid == 0:
            F.hard_exit(3)
        _, status = os.waitpid(pid, 0)
        for _ in range(100):
            if got:
                break
            time.sleep(0.01)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 3, status
        assert got, "parent never got SIGUSR2"
        os.environ["SPLIT_NV_FAST_EXIT"] = "0"
        got.clear()
        pid = os.fork()
        if pid == 0:
            F.hard_exit(0)
        _, status = os.waitpid(pid, 0)
        time.sleep(0.1)
        assert os.WEXITSTATUS(status) == 0 and not got, "SPLIT_NV_FAST_EXIT=0 must not signal"
    finally:
        os.environ.pop("SPLIT_NV_FAST_EXIT", None)
        signal.signal(signal.SIGUSR2, old)
    print("ok  hard_exit: SIGUSR2 to the parent, then os._exit(code); SPLIT_NV_FAST_EXIT=0 = plain exit")


def _sleeper(code):
    if code is not None:
        os._exit(code)
    time.sleep(60)


def test_kill_ranks_and_exit_code():
    import multiprocessing

    import split_nv.engine as E

    ctx = multiprocessing.get_context("fork")
    done = ctx.Process(target=_sleeper, args=(1,))
    live = ctx.Process(target=_sleeper, args=(None,))
    done.start()
    live.start()
    done.join()
    t0 = time.monotonic()
    E.kill_ranks([done, live], "test")
    live.join(5)
    assert not live.is_alive() and live.exitcode == -9 and time.monotonic() - t0 < 2
    assert E.exit_code([done.exitcode, live.exitcode]) == 1
    assert E.exit_code([0, -9]) == 0  # drain: rank 0 exited 0, rank 1 killed by us
    assert E.exit_code([3, -9]) == 3  # watchdog
    assert E.exit_code([-9, 4]) == 4  # rank 1 fatal CUDA error, rank 0 killed
    assert E.exit_code([-9, -9]) == 1  # drain budget exceeded
    print("ok  kill_ranks SIGKILLs live ranks only; exit_code keeps rank 0's code")


def _st(path, tensors):
    import struct

    header, off, blobs = {}, 0, []
    for name, nbytes in tensors:
        header[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [off, off + nbytes]}
        blobs.append(bytes([len(blobs) + 1]) * nbytes)
        off += nbytes
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return 8 + len(h)


def test_prefetch_ranges():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import prefetch_weights as PW

    with tempfile.TemporaryDirectory() as d:
        b1 = _st(os.path.join(d, "a.safetensors"), [("layers.0.attn.w", 5000), ("layers.1.engram.embed.weight", 9000),
                                                     ("layers.1.engram.embed.scale", 300), ("layers.1.engram.q_weight", 70)])
        b2 = _st(os.path.join(d, "b.safetensors"), [("layers.14.engram.embed.weight", 9000)])
        b3 = _st(os.path.join(d, "small.safetensors"), [("layers.1.engram.wkv.weight", 40)])
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"layers.0.attn.w": "a.safetensors", "layers.1.engram.embed.weight": "a.safetensors",
                                      "layers.1.engram.embed.scale": "a.safetensors",
                                      "layers.1.engram.q_weight": "a.safetensors",
                                      "layers.14.engram.embed.weight": "b.safetensors"}}, f)
        PW.MERGE_GAP = 0
        r = {os.path.basename(k): v for k, v in PW.ranges(d).items()}
        assert "b.safetensors" not in r, r  # nothing but an Engram table
        assert r["a.safetensors"] == [(0, b1), (b1, b1 + 5000), (b1 + 14300, b1 + 14370)], r["a.safetensors"]
        assert r["small.safetensors"] == [(0, b3), (b3, b3 + 40)], r
        done = {"bytes": 0, "lock": threading.Lock()}
        PW.read_file(os.path.join(d, "a.safetensors"), r["a.safetensors"], False, done)
        assert done["bytes"] == b1 + 5070, done
    print("ok  prefetch ranges: Engram embed tables skipped, other tensors and headers read")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all passed")
