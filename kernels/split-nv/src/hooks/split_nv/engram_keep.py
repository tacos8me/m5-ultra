"""Keep the pinned Engram host tables across engine restarts (SPLIT_NV_ENGRAM_KEEP=1, default off).

Production (SGLANG_DSV41_ENGRAM_PINNED=copy) copies both Engram tables (layers 1 and 14, 94.4 GiB each) from
/engram into a fresh memfd on every start: 46-55 s of preadv on the crash path, plus the page-cache and swap
pressure that makes the prefix index load slow afterwards. The memfd dies with the container, so every restart
pays again.

With keep on, each table is a SysV shared-memory segment in the host IPC namespace (the container runs with
--ipc=host): shmem like the memfd, so cudaHostRegister pins it the same way, but it outlives the container. The
first start after a boot fills it with SGLang's own fill_from_source (same bytes as today). Later starts attach,
check it against the source file, and only cudaHostRegister it. With SPLIT_NV_ENGRAM_ASYNC_REGISTER=1 (the default)
that registration runs on a side thread from model construction, so it overlaps the checkpoint load and the
post-load work (~35 s). The bytes the GPU reads are identical, so numerics do not change.

Segment layout: the table bytes, padded to a page, then one header page (Header) recording the source identity
(size, mtime_ns, inode) and the state (FILLING while a fill runs, COMPLETE after every rank's fill and the barrier).
A segment is reused only when it is COMPLETE, matches the source identity and the table size, and SAMPLES random
4 KiB blocks (plus the first and last) match the source file. Anything else means the segment is dropped and
refilled. Keys are KEY_BASE + layer id, so there is at most one segment per table and nothing leaks across fills.

The segments are SHM_LOCKed (no swap between a stop and the next registration) and hold ~189 GiB of host RAM while
the engine is stopped. Free them with tools/engram_keep.py drop (as
root in the host IPC namespace; see that tool).
"""

import ctypes
import os
import random
import re
import threading
import time

PAGE = 4096
MAGIC = 0x4B45455052414D47  # "GMARPEEK"
VERSION = 1
FILLING, COMPLETE = 1, 2
KEY_BASE = int(os.environ.get("SPLIT_NV_ENGRAM_KEEP_KEY_BASE", "0x53444B00"), 0)
IPC_CREAT, IPC_EXCL, IPC_RMID, SHM_RDONLY, SHM_LOCK = 0o1000, 0o2000, 0, 0o10000, 11

_libc = ctypes.CDLL(None, use_errno=True)
_libc.shmget.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_int)
_libc.shmget.restype = ctypes.c_int
_libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
_libc.shmat.restype = ctypes.c_void_p
_libc.shmdt.argtypes = (ctypes.c_void_p,)
_libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


class Header(ctypes.Structure):
    _fields_ = [("magic", ctypes.c_uint64), ("version", ctypes.c_uint32), ("state", ctypes.c_uint32),
                ("nbytes", ctypes.c_uint64), ("src_size", ctypes.c_uint64), ("src_mtime_ns", ctypes.c_uint64),
                ("src_ino", ctypes.c_uint64), ("filled_at", ctypes.c_double), ("fill_seconds", ctypes.c_double),
                ("reuses", ctypes.c_uint64)]


def enabled():
    return os.environ.get("SPLIT_NV_ENGRAM_KEEP") == "1"


def seg_bytes(nbytes):
    return (nbytes + PAGE - 1) // PAGE * PAGE + PAGE


def key_for(name):
    m = re.search(r"(\d+)$", name)
    if not m:
        raise ValueError(f"engram keep: no layer id in table name {name!r}")
    return KEY_BASE + int(m.group(1))


def source_identity(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns, st.st_ino


def _err(what):
    e = ctypes.get_errno()
    return OSError(e, f"engram keep: {what}: {os.strerror(e)}")


def segments():
    """{key: dict(shmid, size, nattch, cpid)} of this IPC namespace (/proc/sysvipc/shm)."""
    out = {}
    with open("/proc/sysvipc/shm") as f:
        cols = f.readline().split()
        for line in f:
            v = dict(zip(cols, line.split()))
            out[int(v["key"])] = {"shmid": int(v["shmid"]), "size": int(v["size"]), "nattch": int(v["nattch"]),
                                  "cpid": int(v["cpid"])}
    return out


def attach(shmid, readonly=False):
    addr = _libc.shmat(shmid, None, SHM_RDONLY if readonly else 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise _err(f"shmat({shmid})")
    return addr


def detach(addr):
    _libc.shmdt(ctypes.c_void_p(addr))


def remove(shmid):
    if _libc.shmctl(shmid, IPC_RMID, None) != 0:
        raise _err(f"shmctl({shmid}, IPC_RMID)")


def lock(shmid):
    """SHM_LOCK: keep the segment out of swap while no engine has it registered (between a stop and the next start's
    cudaHostRegister). Needs RLIMIT_MEMLOCK >= the size (run_engine.sh: --ulimit memlock=-1) or CAP_IPC_LOCK."""
    if _libc.shmctl(shmid, SHM_LOCK, None) != 0:
        return os.strerror(ctypes.get_errno())
    return "locked"


def header_at(addr, nbytes):
    return Header.from_address(addr + seg_bytes(nbytes) - PAGE)


def create(key, nbytes):
    shmid = _libc.shmget(key, seg_bytes(nbytes), IPC_CREAT | IPC_EXCL | 0o600)
    if shmid < 0:
        raise _err(f"shmget(0x{key:x}, {seg_bytes(nbytes)})")
    return shmid


def sample_check(addr, path, nbytes, samples, seed=None):
    """First, last and `samples` random 4 KiB blocks of the segment equal the source file. 'full' compares all."""
    nblk = (nbytes + PAGE - 1) // PAGE
    if samples == "full":
        blocks = range(nblk)
    else:
        rng = random.Random(seed)
        blocks = sorted({0, nblk - 1, *(rng.randrange(nblk) for _ in range(int(samples)))})
    fd = os.open(path, os.O_RDONLY)
    try:
        for b in blocks:
            off = b * PAGE
            n = min(PAGE, nbytes - off)
            if os.pread(fd, n, off) != ctypes.string_at(addr + off, n):
                return False, b
    finally:
        os.close(fd)
    return True, len(blocks)


class Kept:
    """This rank's view of one kept table: segment id, address, whether it was reused, and the async registration."""

    def __init__(self, shmid, addr, reused, why, check_s):
        self.shmid, self.addr, self.reused, self.why, self.check_s = shmid, addr, reused, why, check_s
        self.thread = None
        self.reg_error = None
        self.reg_wait = 0.0


def open_table(nbytes, name, group, source_path):
    """Rank 0 decides reuse vs (re)fill and broadcasts it; every rank then attaches the same segment."""
    key = key_for(name)
    decision = None
    if group.rank_in_group == 0:
        t0 = time.perf_counter()
        seg = segments().get(key)
        ident = source_identity(source_path)
        reused, why, shmid = False, "no segment", None
        if seg is not None:
            shmid = seg["shmid"]
            if seg["size"] != seg_bytes(nbytes):
                why = f"size {seg['size']} != {seg_bytes(nbytes)}"
            else:
                addr = attach(shmid)
                h = header_at(addr, nbytes)
                if (h.magic, h.version, h.nbytes) != (MAGIC, VERSION, nbytes):
                    why = "no valid header"
                elif (h.src_size, h.src_mtime_ns, h.src_ino) != ident:
                    why = "source file changed"
                elif h.state != COMPLETE:
                    why = "fill never completed"
                else:
                    ok, info = sample_check(addr, source_path, nbytes,
                                            os.environ.get("SPLIT_NV_ENGRAM_KEEP_SAMPLES", "64"))
                    reused, why = ok, (f"{info} blocks match the source" if ok else f"block {info} differs from the source")
                if reused:
                    h.reuses += 1
                detach(addr)
            if not reused:
                remove(shmid)  # destroyed once the last attached process (if any) detaches
        if not reused:
            shmid = create(key, nbytes)
            addr = attach(shmid)
            h = header_at(addr, nbytes)
            h.magic, h.version, h.state, h.nbytes = MAGIC, VERSION, FILLING, nbytes
            h.src_size, h.src_mtime_ns, h.src_ino = ident
            detach(addr)
        why += f", SHM_LOCK {lock(shmid)}"
        decision = (shmid, reused, why, time.perf_counter() - t0)
    shmid, reused, why, check_s = group.broadcast_object(decision, src=0)
    return Kept(shmid, attach(shmid), reused, why, check_s)


def _start_register(table, kept):
    import torch

    dev = torch.cuda.current_device() if torch.cuda.is_available() else None

    def run():
        try:
            if dev is not None:
                torch.cuda.set_device(dev)  # else the thread would register under (and create) a context on GPU 0
            table._register()
        except BaseException as e:  # noqa: BLE001  (re-raised by finish_load)
            kept.reg_error = e

    kept.thread = threading.Thread(target=run, name=f"engram-register-{kept.shmid}", daemon=True)
    kept.thread.start()


def drop_unused():
    """Keep off: remove kept segments nobody has attached. Otherwise turning keep off would leave 189 GiB pinned
    beside the memfd this start is about to fill, and the fill's MemAvailable check (or the host) would fail."""
    dropped = []
    try:
        segs = segments()
    except OSError:
        return dropped
    for key, s in segs.items():
        if KEY_BASE <= key < KEY_BASE + 256 and s["nattch"] == 0:
            try:
                remove(s["shmid"])
                dropped.append(key - KEY_BASE)
            except OSError:
                pass  # the other rank got there first, or not ours to remove
    if dropped:
        print(f"[split-nv] engram keep is off: dropped unused kept segments of layers {sorted(dropped)}", flush=True)
    return dropped


def install():
    """Patch SGLang's _HostTable (sglang.srt.layers.engram) for the pinned-copy layout. No-op unless enabled."""
    if not enabled():
        drop_unused()
        return False
    import torch
    from sglang.srt.layers import engram as E

    if getattr(E._HostTable, "_split_nv_keep", False):
        return True
    orig_init, orig_finish = E._HostTable.__init__, E._HostTable.finish_load
    logger = E.logger

    def __init__(self, layout, nbytes, name, group, pin, *, backing_dir="", source_path=None):
        if not (layout == "shared" and source_path and not backing_dir and pin):
            return orig_init(self, layout, nbytes, name, group, pin, backing_dir=backing_dir, source_path=source_path)
        self.layout, self.nbytes, self.group = layout, nbytes, group
        self.dirty = self.registered = self.reused = False
        self.backing_path, self.fd, self.fill_stats = None, None, None
        self.source_path = source_path
        self.pin_deferred = True
        self.keep = open_table(nbytes, name, group, source_path)
        self.mm = (ctypes.c_ubyte * nbytes).from_address(self.keep.addr)
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        if self.keep.reused and os.environ.get("SPLIT_NV_ENGRAM_ASYNC_REGISTER", "1") == "1":
            _start_register(self, self.keep)

    def finish_load(self, label=""):
        kept = getattr(self, "keep", None)
        if kept is None:
            return orig_finish(self, label)
        t0 = time.perf_counter()
        if kept.reused:
            if kept.thread is not None:
                kept.thread.join()
                kept.reg_wait = time.perf_counter() - t0
                if kept.reg_error is not None:
                    raise kept.reg_error
            else:
                self._register()
            self.group.barrier()
            logger.info("engram host table %s: kept segment shmid %d reused (%s, checked in %.2f s), "
                        "cudaHostRegister %.1f s (%s, waited %.1f s), pinned", label, kept.shmid, kept.why, kept.check_s,
                        getattr(self, "register_seconds", 0.0), "async" if kept.thread is not None else "sync",
                        kept.reg_wait)
            return
        self.fill_from_source(E.envs.SGLANG_DSV41_ENGRAM_PINNED_THREADS.get(),
                              E.envs.SGLANG_DSV41_ENGRAM_PINNED_RESERVE_GIB.get() << 30)
        self.group.barrier()  # every rank's half is in the segment
        if self.group.rank_in_group == 0:
            h = header_at(kept.addr, self.nbytes)
            h.filled_at, h.fill_seconds = time.time(), self.fill_stats["seconds"]
            h.state = COMPLETE
        self._register()
        st = self.fill_stats
        logger.info("engram host table %s: kept segment shmid %d filled (%s): this rank %.1f GiB in %.1f s "
                    "(%.1f GiB/s, %d threads), cudaHostRegister %.1f s, pinned", label, kept.shmid, kept.why,
                    st["bytes"] / 2**30, st["seconds"], st["bytes"] / 2**30 / max(st["seconds"], 1e-9), st["threads"],
                    getattr(self, "register_seconds", 0.0))

    E._HostTable.__init__ = __init__
    E._HostTable.finish_load = finish_load
    E._HostTable._split_nv_keep = True
    print("[split-nv] engram keep installed (SysV segments survive restarts)", flush=True)
    return True
