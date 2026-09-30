"""Status / removal of the kept Engram segments (split_nv.engram_keep, SPLIT_NV_ENGRAM_KEEP=1).

The segments belong to root (created inside the engine container) and live in the host IPC namespace, so run this
as root there, e.g. from the engine image:
  docker run --rm --ulimit core=0 --ipc=host --network none --entrypoint python3 -e PYTHONPATH=/home/ian/split-nv/hooks \\
    -v /home/ian/split-nv-deploy:/home/ian/split-nv:ro -v /home/ian/models/dsv41-engram:/engram:ro \\
    sglang-dsv41-split:6152b54 /home/ian/split-nv/tools/engram_keep.py status|verify|drop [--force]
(`ipcs -m` as any user lists them too: keys 0x53444b01 and 0x53444b0e, ~94.4 GiB each.)
  status  key, shmid, size, attached processes, header state / source identity / fill time / reuse count
  verify  compare every byte of each COMPLETE segment with /engram/sglang_engram_<layer>.bin (read-only; ~94 GiB
          of NVMe reads per table, fine while the engine runs). Exit 1 on any difference.
  drop    IPC_RMID every kept segment (frees ~189 GiB once no process has it attached). Refused while the engine
          has them attached unless --force (the engine keeps working; the next start refills).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))
from split_nv import engram_keep as K  # noqa: E402


def kept():
    return {k: s for k, s in K.segments().items() if K.KEY_BASE <= k < K.KEY_BASE + 256}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("status", "verify", "drop"))
    ap.add_argument("--engram-dir", default="/engram")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    segs = kept()
    if not segs:
        print("no kept Engram segments")
        return 0
    bad = 0
    for key, s in sorted(segs.items()):
        line = f"layer {key - K.KEY_BASE}: key 0x{key:x} shmid {s['shmid']} {s['size'] / 2**30:.1f} GiB nattch {s['nattch']}"
        if a.cmd == "status":
            try:
                addr = K.attach(s["shmid"], readonly=True)
                h = K.Header.from_address(addr + s["size"] - K.PAGE)
                state = {K.FILLING: "FILLING", K.COMPLETE: "COMPLETE"}.get(h.state, f"state {h.state}")
                line += (f" {state} src size {h.src_size} mtime_ns {h.src_mtime_ns} ino {h.src_ino}"
                         f" filled {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h.filled_at)) if h.filled_at else '-'}"
                         f" in {h.fill_seconds:.1f}s, reused {h.reuses}x")
                K.detach(addr)
            except OSError as e:
                line += f" (header unreadable: {e.strerror}; run as root)"
            print(line)
            continue
        if a.cmd == "verify":
            bad += not verify(key - K.KEY_BASE, s, a.engram_dir, line)
            continue
        if s["nattch"] and not a.force:
            print(line + ": attached (engine running?); not dropped without --force")
            continue
        K.remove(s["shmid"])
        print(line + ": dropped")
    return 1 if bad else 0


def verify(layer, s, engram_dir, line, chunk=64 << 20):
    import ctypes

    addr = K.attach(s["shmid"], readonly=True)
    try:
        h = K.Header.from_address(addr + s["size"] - K.PAGE)
        path = os.path.join(engram_dir, f"sglang_engram_{layer}.bin")
        if h.state != K.COMPLETE or os.path.getsize(path) != h.nbytes:
            print(f"{line}: not COMPLETE or {path} size differs")
            return False
        t0 = time.monotonic()
        fd = os.open(path, os.O_RDONLY)
        try:
            for off in range(0, h.nbytes, chunk):
                n = min(chunk, h.nbytes - off)
                if os.pread(fd, n, off) != ctypes.string_at(addr + off, n):
                    print(f"{line}: DIFFERS in [{off}, {off + n})")
                    return False
        finally:
            os.close(fd)
        print(f"{line}: all {h.nbytes} bytes equal {path} ({time.monotonic() - t0:.0f}s)")
        return True
    finally:
        K.detach(addr)


if __name__ == "__main__":
    sys.exit(main())
