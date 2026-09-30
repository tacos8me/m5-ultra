"""CPU test of rank-failstop step 1 (no GPU, no engine): a snapshot write that failed on either rank never registers a
prefix-cache entry, is counted, and never raises or hangs.

  1. Front.save_snapshot over a real PrefixIndex (temp dir) and a fake engine: with every file whole, the grid
     entries and the prompt-end entry are registered; with rank 1's entry file missing, a block file truncated, or
     per-rank sizes that must match but differ, exactly the affected entries are skipped (counted in cache stats
     skipped_incomplete, files removed, unreferenced blocks forgotten), nothing raises, and a restart's index load
     sees only the registered entries (dropped_at_load 0);
  2. RankStore: failed background writes are counted (n_errors), the last 8 kept, flush() does not raise;
  3. /health rank_errors: rank 0 live, the other ranks as of the last barrier;
  4. Engine.cmd_barrier in two real processes (gloo): rank 1's flush raises, both ranks still meet in the collective
     and rank 0 gets every rank's [write errors, failed commands]. Control: the old order (flush, then barrier)
     leaves rank 0 blocked until the collective times out (the 300 s watchdog hang in production).
usage: python3 tools/test_rank_failstop.py
"""
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hooks"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

GRID = 8192


def fake_sglang_distributed(group):
    for nm in ("sglang", "sglang.srt"):
        sys.modules.setdefault(nm, types.ModuleType(nm))
    d = types.ModuleType("sglang.srt.distributed")
    d.get_tp_group = lambda: group
    sys.modules["sglang.srt.distributed"] = d
    sys.modules["sglang.srt"].distributed = d


def barrier_rank(rank, init, mode, q):
    """One engine rank: rank 1's snapshot flush raises. mode new = Engine.cmd_barrier; old = flush, then barrier."""
    from datetime import timedelta

    import torch.distributed as dist

    sys.path.insert(0, os.path.join(HERE, "..", "hooks"))
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=6))
    fake_sglang_distributed(types.SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD))
    from split_nv import engine as EM

    def flush():
        if rank == 1:
            raise OSError("injected: rank 1 flush failed")

    E = EM.Engine.__new__(EM.Engine)
    E.tp_rank, E.job_failures = rank, 3 * rank
    E.store = types.SimpleNamespace(n_errors=rank, flush=flush)
    t0 = time.monotonic()
    try:
        if mode == "new":
            out = E.cmd_barrier()
        else:
            E.store.flush()
            dist.barrier()
            out = None
        q.put((rank, "returned", out, round(time.monotonic() - t0, 1)))
    except Exception as e:  # noqa: BLE001
        q.put((rank, "raised", repr(e)[:80], round(time.monotonic() - t0, 1)))
        time.sleep(9)  # like rank_main: the rank lives on (it only logs a failed command)


def two_ranks(mode):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    init = "file://" + os.path.join(tempfile.mkdtemp(prefix="split-nv-failstop-"), "store")
    ps = [ctx.Process(target=barrier_rank, args=(r, init, mode, q)) for r in (0, 1)]
    for p in ps:
        p.start()
    res = {}
    for _ in ps:
        r = q.get(timeout=60)
        res[r[0]] = r[1:]
    for p in ps:
        p.join(30)
        if p.is_alive():
            p.kill()
    return res


def main():
    fails = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    # ---- 4 first (spawned processes, before this process starts threads) ------------------------------------------
    new = two_ranks("new")
    check(new.get(0, ("",))[0] == "returned" and new[0][1] == [[0, 0], [2, 3]] and new.get(1, ("",))[0] == "returned"
          and new[0][2] < 3, f"cmd_barrier with rank 1's flush raising: both ranks return, counts gathered ({new})")
    old = two_ranks("old")
    check(old.get(1, ("",))[0] == "raised" and old.get(0, ("", "", 0))[2] >= 5,
          f"control, flush before barrier: rank 0 blocked until the collective timed out ({old})")

    root = tempfile.mkdtemp(prefix="split-nv-failstop-cache-")
    os.environ["SPLIT_NV_CACHE_DIR"] = root
    import torch

    from split_nv import front as F
    from split_nv import prefix_cache as PC

    # ---- 1. save_snapshot -----------------------------------------------------------------------------------------
    def rank_tensors(extra=False, n=64):
        t = {"kv0": torch.arange(n * 8, dtype=torch.int32).view(n, 8)}
        if extra:
            t["cap.h.0"] = torch.ones(3, 5)
        return t

    def scenario(drop_final_r1=False, truncate_block=None, grid_size_mismatch=False, rank1_errors=0):
        cache = PC.PrefixIndex("t", 1 << 40)
        cache.clear()
        fr = F.Front.__new__(F.Front)
        fr.cache = cache
        fr.gate = types.SimpleNamespace(suspended=False)
        fr.engine = types.SimpleNamespace(job_failures=0, store=types.SimpleNamespace(n_errors=0, errors=[]))
        fr.rank_counts = fr.rank_counts_t = None
        cmds = []
        P = 2 * GRID + 100
        ids = list(range(1000, 1000 + P))
        b1, b2 = cache.new_id("b"), cache.new_id("b")
        g1, g2 = cache.new_id("g"), cache.new_id("g")
        for bid in (b1, b2):  # written by cmd_prefill_chunk on each rank (blocks) and rank 0's front (rows)
            for r in (0, 1):
                PC._atomic_save(rank_tensors(), cache.p(f"blk-{bid}.r{r}"))
            PC._atomic_save(rank_tensors(n=4), cache.p(f"rows-{bid}"))
        for gkey in (g1, g2):
            for r in (0, 1):
                PC._atomic_save(rank_tensors(n=32 if (grid_size_mismatch and r == 1 and gkey == g2) else 16),
                                cache.p(f"ent-{gkey}.r{r}"))
            PC._atomic_save(rank_tensors(n=2), cache.p(f"rows-tail-{gkey}"))
        if truncate_block is not None:
            f = cache.p(f"blk-{(b1, b2)[truncate_block]}.r1")
            with open(f, "r+b") as fh:
                fh.truncate(os.path.getsize(f) - 100)

        def submit(cmd, priority=0):
            cmds.append(cmd[0])
            if cmd[0] == "snapshot":
                PC._atomic_save(rank_tensors(extra=True), cache.p(f"ent-{cmd[2]}.r0"))
                if not drop_final_r1:
                    PC._atomic_save(rank_tensors(), cache.p(f"ent-{cmd[2]}.r1"))
                return 0.01
            if cmd[0] == "barrier":
                return [[0, 0], [rank1_errors, 0]]
            raise AssertionError(cmd)

        fr.submit = submit
        info = {"blocks": [b1, b2], "new_blocks": [b1, b2], "rows": F.Rows(),
                "grid_entries": [(g1, GRID, [b1]), (g2, 2 * GRID, [b1, b2])]}
        err = None
        try:
            snap = fr.save_snapshot(7, ids, info)
        except Exception as e:  # noqa: BLE001
            err, snap = e, None
        reg = sorted(cache.entries)
        final = [k for k in reg if k.startswith("e")]
        leftovers = sorted(os.path.basename(f) for f in os.listdir(cache.root))
        return dict(err=err, snap=snap, grid=[k for k in (g1, g2) if k in cache.entries], final=final, cmds=cmds,
                    skipped=cache.stats["skipped_incomplete"], blocks=sorted(cache.block_refs), b=(b1, b2), g=(g1, g2),
                    files=leftovers, front=fr, cache=cache, ids=ids)

    s = scenario()
    check(s["err"] is None and s["grid"] == list(s["g"]) and len(s["final"]) == 1 and s["skipped"] == 0
          and s["snap"] and s["cmds"] == ["snapshot", "barrier"], f"all files whole: 2 grid + 1 prompt-end entry registered ({s['grid']}, {s['final']})")

    s = scenario(drop_final_r1=True, rank1_errors=1)
    final_files = [f for f in s["files"] if f.startswith(("ent-e", "rows-tail-e", "idx-e", "tok-e"))]
    check(s["err"] is None and s["snap"] is None and s["grid"] == list(s["g"]) and not s["final"] and s["skipped"] == 1
          and not final_files and s["blocks"] == sorted(s["b"]),
          f"rank 1's prompt-end file missing: entry skipped + counted, its files removed, grid entries kept ({final_files})")
    check(s["cache"].lookup(s["ids"] + list(range(300))).key == s["g"][1], "a resume uses the longest registered (grid) entry")
    re = s["front"].rank_errors()
    check(re["write_errors"] == [0, 1] and re["failed_commands"] == [0, 0] and re["peers_as_of_s"] is not None,
          f"/health rank_errors after the barrier: {re}")

    s = scenario(truncate_block=1)
    b1, b2 = s["b"]
    check(s["err"] is None and s["grid"] == [s["g"][0]] and not s["final"] and s["skipped"] == 2
          and s["blocks"] == [b1] and not any(f.startswith(f"blk-{b2}") or f == f"rows-{b2}" for f in s["files"]),
          f"block b2.r1 truncated: the two entries using it skipped, b2 forgotten, the 8K grid entry kept ({s['grid']}, {s['blocks']})")

    s = scenario(grid_size_mismatch=True)
    check(s["err"] is None and s["grid"] == [s["g"][0]] and len(s["final"]) == 1 and s["skipped"] == 1,
          "grid entry whose rank files differ in size: skipped; the prompt-end entry (own files whole) kept")

    reload = PC.PrefixIndex("t", 1 << 40)
    check(reload.stats["dropped_at_load"] == 0 and sorted(reload.entries) == sorted(s["cache"].entries),
          f"restart index load: registered entries only, nothing dropped ({reload.stats})")
    check(PC.whole_file(os.path.join(root, "nope")) is None, "whole_file: missing file")

    # ---- 2. RankStore write errors ----------------------------------------------------------------------------------
    st = PC.RankStore(types.SimpleNamespace(tp_rank=1), "t")
    for i in range(11):
        st.put(f"sub/dir-{i}/x", {"a": torch.zeros(2)})  # the directory does not exist: the write fails
    st.flush()
    st.put("ok", {"a": torch.zeros(2)})
    st.flush()
    check(st.n_errors == 11 and len(st.errors) == PC.MAX_ERRORS_KEPT and "dir-10" in st.errors[-1]
          and PC.whole_file(st.path("ok")) is not None, f"RankStore counts failed writes ({st.n_errors}, kept {len(st.errors)})")

    # ---- 3. rank_errors before any barrier --------------------------------------------------------------------------
    fr = F.Front.__new__(F.Front)
    fr.engine = types.SimpleNamespace(job_failures=2, store=st)
    re = fr.rank_errors()
    check(re["write_errors"] == [11] and re["failed_commands"] == [2] and re["peers_as_of_s"] is None
          and "dir-10" in re["last_write_error"], f"rank_errors before a barrier: {re}")
    fr2 = F.Front.__new__(F.Front)
    fr2.engine = types.SimpleNamespace()
    check(fr2.rank_errors()["write_errors"] == [0], "rank_errors with a bare engine (test fixtures)")

    print("ALL PASS" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
