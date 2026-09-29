"""Read the checkpoint tensors the engine will load into the page cache, while the container is still starting.

run_engine.sh starts this on the host just before `docker run` (SPLIT_NV_PREFETCH=1, the default). The first
~16-27 s of a start are imports and process spawn, when the NVMe sits idle. The weight load then faults ~148 GiB
of layer 0-20 / embed / head / vision tensors in through mmap at ~4.4 GB/s effective (shards 12-15 s, then another
~20 s of async loaders). Nothing is cached between restarts: the Engram fill and the cache mirror evict it. The
Engram embedding tables (layers.N.engram.embed.*, 189 GiB in the last two shards) are skipped: the engine fills
those from /engram, or reuses its kept segments.

Standard library only (host python3). Page cache only: no effect on what the engine computes.
usage: prefetch_weights.py <model view dir> [--threads 12] [--max-files N] [--drop-after]
  --drop-after  posix_fadvise(DONTNEED) the ranges again afterwards (measurement only: leaves the cache as found)
Measured on the box while serving (--max-files 6 --threads 6 --drop-after): 29.7 GiB in 5.2 s, 6.1 GB/s.
"""
import argparse
import json
import os
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

SKIP = re.compile(r"^layers\.\d+\.engram\.embed\.(weight|scale)$")
CHUNK = 16 << 20
MERGE_GAP = 1 << 20  # ranges closer than this are read as one


def ranges(view):
    """{real file path: [(start, end)] merged byte ranges of the tensors the engine loads}."""
    idx = json.load(open(os.path.join(view, "model.safetensors.index.json")))["weight_map"]
    files = sorted(set(idx.values()))
    for f in os.listdir(view):
        if f.endswith(".safetensors") and f not in files:
            files.append(f)  # e.g. engram-small.safetensors (loaded, not in the index)
    out = {}
    for f in files:
        path = os.path.realpath(os.path.join(view, f))
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        base = 8 + n
        spans = sorted((base + v["data_offsets"][0], base + v["data_offsets"][1]) for k, v in header.items()
                       if k != "__metadata__" and idx.get(k, f) == f and not SKIP.match(k))
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + MERGE_GAP:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if merged:
            out[path] = [(0, base)] + [tuple(r) for r in merged]
    return out


def read_file(path, spans, drop, done):
    buf = bytearray(CHUNK)
    mv = memoryview(buf)
    fd = os.open(path, os.O_RDONLY)
    try:
        for a, b in spans:
            off = a
            while off < b:
                got = os.preadv(fd, [mv[:min(CHUNK, b - off)]], off)
                if got <= 0:
                    break
                off += got
                with done["lock"]:
                    done["bytes"] += got
            if drop:
                os.posix_fadvise(fd, a, b - a, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--drop-after", action="store_true")
    a = ap.parse_args()
    t0 = time.monotonic()
    try:
        work = ranges(a.view)
    except (OSError, ValueError, KeyError) as e:
        print(f"[prefetch] skipped: {e}", flush=True)
        return 0
    items = sorted(work.items())
    if a.max_files:
        items = items[:a.max_files]
    total = sum(b - s for _, spans in items for s, b in spans)
    done = {"bytes": 0, "lock": threading.Lock()}
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        for f in [pool.submit(read_file, p, spans, a.drop_after, done) for p, spans in items]:
            try:
                f.result()
            except OSError as e:
                print(f"[prefetch] {e}", flush=True)
    dt = time.monotonic() - t0
    print(f"[prefetch] {done['bytes'] / 2**30:.1f} of {total / 2**30:.1f} GiB of checkpoint tensors from {len(items)} "
          f"files in {dt:.1f}s ({done['bytes'] / 1e9 / max(dt, 1e-9):.1f} GB/s, {a.threads} threads)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
