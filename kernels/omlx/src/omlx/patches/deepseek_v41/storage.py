# SPDX-License-Identifier: Apache-2.0
"""Selected-row Engram I/O, following Qwen4-Exp's _SafeTensorMMap pattern.

DeepSeek uses per-row/per-32-channel scales instead of Qwen's shared scale.
Copies leave the mapping under the lock, so close cannot invalidate a live view.
Resident tables retain their packed bytes; prefetch workers only copy CPU rows.
"""

import ctypes
import fcntl
import hashlib
import json
import logging
import math
import mmap
import os
import shutil
import struct
import subprocess
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Lock, RLock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .hot_rows import HotRows

RESIDENT_READ_BYTES = 8 * 1024 * 1024
# safetensors dtype tag -> numpy transport dtype (bf16 travels as raw uint16).
SAFETENSORS_NUMPY_DTYPES = {
    "BF16": "<u2",
    "F16": "<f2",
    "F32": "<f4",
    "U32": "<u4",
    "U8": "u1",
    "I8": "i1",
    "F8_E4M3": "u1",
    "F8_E8M0": "u1",
    "F8_E4M3FN": "u1",
    "F8_E8M0FNU": "u1",
}
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
PAGE_PREFETCH_MIN_ROWS = 128
PAGE_IO_WORKERS = 48
_PAGE_IO_POOL = ThreadPoolExecutor(
    max_workers=PAGE_IO_WORKERS, thread_name_prefix="v41-page-io"
)
logger = logging.getLogger(__name__)

# Prefill-sized gathers (hundreds of thousands of random 256-byte rows per
# 8K chunk) bypass the mmap: deduplicated, sorted rows are read by a native
# pthread pool with direct 4 KiB positional reads (F_NOCACHE), so no clean
# file pages accumulate in RSS or the unified cache and no Python runs per row.
# Decode-sized gathers keep the mmap path, where warm pages cost microseconds.
NATIVE_MIN_ROWS = int(os.environ.get("DS41_ENGRAM_NATIVE_MIN_ROWS", "256"))
NATIVE_THREADS = int(os.environ.get("DS41_ENGRAM_IO_THREADS", "64"))
# Decode gathers (L x 24 rows per table) used to fault the mmap one page at a
# time on the forward's critical path (~6 ms per verify step on new text).
# They now use the native pool as well, through a cached descriptor so warm
# rows still cost only a copy.
DECODE_NATIVE = os.environ.get("DS41_ENGRAM_DECODE_NATIVE", "1") == "1"
DECODE_THREADS = int(os.environ.get("DS41_ENGRAM_DECODE_THREADS", "16"))
# Retain only the first prefill chunk's compact, immutable SSD rows. Later
# chunks already have GPU compute to hide their reads behind. Two released
# model tables use at most 128 MiB total; 0 restores uncached row reads.
WARM_ROWS_BYTES = int(
    max(0, min(64, float(os.environ.get("DS41_ENGRAM_WARM_ROWS_MIB", "64"))))
    * 1024**2
)
# Two released tables: at most 2.5 GiB total, including cache metadata.
# Set 0 to retain the original first-chunk snapshot behavior.
HOT_ROWS_BYTES = int(
    max(0, min(1280, float(os.environ.get("DS41_ENGRAM_HOT_ROWS_MIB", "1280"))))
    * 1024**2
)
NATIVE_ALIGN = 4096
_F_RDAHEAD = getattr(fcntl, "F_RDAHEAD", 45)
_native_lib = None
_native_lock = Lock()


def _native_gather():
    """Build (once, cached by source hash) and load the native row reader."""
    global _native_lib
    with _native_lock:
        if _native_lib is not None:
            return _native_lib or None
        _native_lib = False
        if os.environ.get("DS41_ENGRAM_NATIVE_IO", "1") != "1":
            return None
        try:
            source = Path(__file__).with_name("engram_io.c")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            root = Path(
                os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
            ) / "omlx"
            library = root / f"ds41_engram_io_{digest}.dylib"
            if not library.exists():
                root.mkdir(parents=True, exist_ok=True)
                compiler = shutil.which("xcrun")
                command = [compiler, "clang"] if compiler else [shutil.which("cc") or "cc"]
                partial = library.with_name(f"{library.name}.{os.getpid()}.tmp")
                subprocess.run(
                    [*command, "-O3", "-shared", "-fPIC", "-o", str(partial), str(source)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                os.replace(partial, library)
            lib = ctypes.CDLL(str(library))
            lib.ds41_engram_gather.argtypes = [
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int64,
            ]
            lib.ds41_engram_gather.restype = ctypes.c_int
            _native_lib = lib
        except Exception as error:  # noqa: BLE001 -- the mmap path stays correct
            logger.warning(
                "DeepSeek V4.1 native Engram row reader unavailable (%s); "
                "using mmap gathers",
                error,
            )
        return _native_lib or None


def _resident_buffer(shape, dtype):
    """Share packed Metal storage with CPU gathers, without a host copy."""
    types = {
        np.dtype("<u4"): mx.uint32,
        np.dtype("<u2"): mx.uint16,
        np.dtype("u1"): mx.uint8,
        np.dtype("i1"): mx.int8,
        np.dtype("<f4"): mx.float32,
        np.dtype("<f2"): mx.float16,
    }
    # Materialize on the GPU before exposing a NumPy view. Otherwise zeros
    # remains a scalar broadcast and NumPy materializes unwired CPU storage.
    value = mx.contiguous(mx.zeros(shape, dtype=types[dtype]))
    mx.eval(value)
    # Submit after allocation so newly created residency sets are attached.
    corner = tuple(slice(0, 1) for _ in shape)
    mx.eval(mx.sum(value[corner].astype(mx.float32)))
    mx.synchronize()
    return np.asarray(value)


class TensorFile:
    def __init__(self, path):
        self._lock = RLock()
        self._path = Path(path)
        self._direct_fd = None
        self._cached_fd = None
        self._io = Condition(Lock())
        self._io_active = 0
        self._file = Path(path).open("rb")  # noqa: SIM115 -- owned until close()
        try:
            length = struct.unpack("<Q", self._file.read(8))[0]
            self.header = json.loads(self._file.read(length))
            self._start = length + 8
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._file_size = os.fstat(self._file.fileno()).st_size
            self._seen_pages = None
            self._last_rearm = 0.0
            # Engram hashes select sparse rows throughout the table. Whole-tensor
            # resident reads use readinto below and do not use this mapping.
            if hasattr(self._mapping, "madvise"):
                self._mapping.madvise(mmap.MADV_RANDOM)
        except Exception:
            mapping = getattr(self, "_mapping", None)
            if mapping is not None:
                mapping.close()
            self._file.close()
            raise

    def read(self, key, rows=None, *, metal_backed=False):
        with self._lock:
            if self._mapping is None:
                raise RuntimeError("Engram tensor file is closed")
            entry = self.header[key]
            dtype = entry["dtype"]
            if dtype not in SAFETENSORS_NUMPY_DTYPES:
                raise ValueError(f"Unsupported source tensor dtype: {dtype}")
            dt = np.dtype(SAFETENSORS_NUMPY_DTYPES[dtype])
            start, end = entry["data_offsets"]
            if end - start != math.prod(entry["shape"]) * dt.itemsize:
                raise ValueError(f"Invalid tensor byte length: {key}")
            view = np.ndarray(
                entry["shape"],
                dtype=dt,
                buffer=self._mapping,
                offset=self._start + start,
            )
            if rows is None:
                # Avoid faulting a whole mmap alongside its resident copy.
                # Read directly into the destination with bounded I/O requests.
                result = (
                    _resident_buffer(view.shape, dt)
                    if metal_backed and view.size
                    else np.empty(view.shape, dtype=dt)
                )
                if not result.size:
                    return result, dtype
                target = memoryview(result).cast("B")
                # Resident tensors already have a full host-memory copy. Avoid
                # retaining a second copy in the macOS unified file cache.
                nocache = getattr(fcntl, "F_NOCACHE", None)
                if nocache is not None:
                    fcntl.fcntl(self._file.fileno(), nocache, 1)
                try:
                    self._file.seek(self._start + start)
                    offset = 0
                    while offset < len(target):
                        stop = min(offset + RESIDENT_READ_BYTES, len(target))
                        count = self._file.readinto(target[offset:stop])
                        if not count:
                            raise ValueError(f"Truncated tensor data: {key}")
                        offset += count
                finally:
                    if nocache is not None:
                        fcntl.fcntl(self._file.fileno(), nocache, 0)
                return result, dtype
            if rows is not None:
                rows = np.asarray(rows, dtype=np.intp)
                if rows.size and (rows.min() < 0 or rows.max() >= view.shape[0]):
                    raise IndexError("Engram row outside table")
                gather_start = None
                row_bytes = (end - start) // view.shape[0] if view.shape[0] else 0
                if (
                    rows.size > PAGE_PREFETCH_MIN_ROWS
                    and 0 < row_bytes <= PAGE_SIZE
                    and self._prefetch_pages(rows, self._start + start, row_bytes)
                ):
                    gather_start = time.perf_counter()
                copied = view[rows]  # Advanced indexing already copies the rows.
                if gather_start is not None:
                    elapsed = time.perf_counter() - gather_start
                    now = time.monotonic()
                    # As in Qwen4-Exp, a slow warm gather can indicate eviction.
                    # This only changes I/O scheduling, never the selected rows.
                    if (
                        elapsed > 0.0005 + 2e-6 * rows.size
                        and now - self._last_rearm >= 60
                    ):
                        self._seen_pages = None
                        self._last_rearm = now
                return copied, dtype
            return np.array(view, copy=True), dtype

    def gather_rows(self, key, rows, cached=False):
        """Copy sorted unique ``rows`` of ``key`` with native direct reads.

        Returns None when the native reader is unavailable. Reads run without
        the file lock (and without the GIL); close() waits for them instead.
        ``cached`` reads decode-sized requests through the unified buffer
        cache (unaligned row reads, fewer threads) so repeated rows stay warm.
        """
        lib = _native_gather()
        if lib is None:
            return None
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        with self._lock:
            if self._mapping is None:
                raise RuntimeError("Engram tensor file is closed")
            entry = self.header[key]
            dtype = entry["dtype"]
            if dtype not in SAFETENSORS_NUMPY_DTYPES:
                raise ValueError(f"Unsupported source tensor dtype: {dtype}")
            dt = np.dtype(SAFETENSORS_NUMPY_DTYPES[dtype])
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            if end - start != math.prod(shape) * dt.itemsize:
                raise ValueError(f"Invalid tensor byte length: {key}")
            if rows.size and (rows[0] < 0 or rows[-1] >= shape[0]):
                raise IndexError("Engram row outside table")
            attribute = "_cached_fd" if cached else "_direct_fd"
            if getattr(self, attribute) is None:
                fd = os.open(self._path, os.O_RDONLY)
                try:
                    if not cached:
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    fcntl.fcntl(fd, _F_RDAHEAD, 0)
                except OSError:
                    os.close(fd)
                    raise
                setattr(self, attribute, fd)
            fd = getattr(self, attribute)
            with self._io:
                self._io_active += 1
        try:
            out = np.empty((rows.size, *shape[1:]), dtype=dt)
            row_bytes = out.itemsize * math.prod(shape[1:])
            code = lib.ds41_engram_gather(
                fd,
                self._start + start,
                row_bytes,
                rows.ctypes.data,
                rows.size,
                out.ctypes.data,
                DECODE_THREADS if cached else NATIVE_THREADS,
                0 if cached else NATIVE_ALIGN,
            )
            if code:
                raise OSError(code, f"Engram row read failed: {os.strerror(code)}")
            return out, dtype
        finally:
            with self._io:
                self._io_active -= 1
                self._io.notify_all()

    def _prefetch_pages(self, rows, base, row_bytes):
        """Read unseen pages concurrently before gathering from the mmap."""
        if self._seen_pages is None:
            self._seen_pages = np.zeros(
                (self._file_size + PAGE_SIZE - 1) // PAGE_SIZE, dtype=np.uint8
            )
        offsets = base + rows.reshape(-1) * row_bytes
        pages = np.unique(
            np.concatenate(
                (offsets // PAGE_SIZE, (offsets + row_bytes - 1) // PAGE_SIZE)
            )
        )
        fresh = pages[self._seen_pages[pages] == 0]
        if not fresh.size:
            return True
        fd = self._file.fileno()

        def touch(group):
            for page in group:
                offset = int(page) * PAGE_SIZE
                remaining = min(PAGE_SIZE, self._file_size - offset)
                while remaining:
                    data = os.pread(fd, remaining, offset)
                    if not data:
                        raise ValueError("Truncated Engram page")
                    offset += len(data)
                    remaining -= len(data)

        # Bound the queue as well as the active reads. The caller holds _lock;
        # drain every worker, including on error, before close can release fd.
        futures = []
        try:
            for group in np.array_split(fresh, min(PAGE_IO_WORKERS, fresh.size)):
                futures.append(_PAGE_IO_POOL.submit(touch, group))
        finally:
            wait(futures)
        for future in futures:
            future.result()
        self._seen_pages[fresh] = 1
        return False

    def close(self):
        with self._lock:
            self._seen_pages = None
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            with self._io:
                while self._io_active:
                    self._io.wait()
            for attribute in ("_direct_fd", "_cached_fd"):
                fd = getattr(self, attribute)
                if fd is not None:
                    os.close(fd)
                    setattr(self, attribute, None)
            self._file.close()


def decode_array(raw, dtype):
    if dtype == "BF16":
        return mx.array((raw.astype(np.uint32) << 16).view(np.float32)).astype(
            mx.bfloat16
        )
    if dtype.startswith("F8_E4M3"):
        return mx.from_fp8(mx.array(raw), dtype=mx.float32)
    if dtype.startswith("F8_E8M0"):
        if np.any(raw == 255):
            raise ValueError("NaN E8M0 scale in checkpoint")
        return mx.array(np.exp2(raw.astype(np.float32) - 127))
    return mx.array(raw)


class _Window:
    """Gathered rows for consecutive hash positions.

    ``rows`` is [positions, K] (K = n-gram heads per token); the future yields
    raw bytes per tensor, aligned with ``rows.reshape(-1)``.
    """

    __slots__ = ("rows", "future", "claimed", "handed_off")

    def __init__(self, rows, future):
        self.rows = rows
        self.future = future
        self.claimed = 0  # positions [0, claimed) were used by a forward
        self.handed_off = None  # positions >= this were copied into a newer window

    @property
    def done(self):
        return self.claimed >= (
            self.rows.shape[0] if self.handed_off is None else self.handed_off
        )


class _PendingGathers:
    """Windows gathered ahead of use for one table, oldest first."""

    def __init__(self):
        self.items = []
        self.lock = Lock()
        self.stats = dict.fromkeys(
            ("read_ahead", "reused", "claimed", "read_at_claim"), 0
        )


class _WarmRows:
    """One immutable sorted row snapshot, outside the MLX parameter tree."""

    def __init__(self):
        self.lock = Lock()
        self.snapshot = None

    def remember(self, rows, data):
        row_bytes = 8 + sum(raw[0].nbytes for raw, _ in data)
        count = min(len(rows), WARM_ROWS_BYTES // row_bytes)
        if count <= 0:
            return
        # Own the bounded slice; retaining a view would retain its whole base.
        snapshot = (
            rows if count == len(rows) else rows[:count].copy(),
            tuple(
                (raw if count == len(rows) else raw[:count].copy(), dtype)
                for raw, dtype in data
            ),
        )
        with self.lock:
            self.snapshot = snapshot


class DiskEngramEmbedding(nn.Module):
    def __init__(
        self,
        path,
        weight_key,
        scale_key,
        scale_path=None,
        *,
        bias_key=None,
        bits=None,
        group_size=32,
    ):
        super().__init__()
        self._lock = RLock()
        self._weights = TensorFile(path)
        try:
            self._scales = (
                self._weights
                if scale_path is None or Path(scale_path) == Path(path)
                else TensorFile(scale_path)
            )
        except Exception:
            self._weights.close()
            raise
        self._weight_key, self._scale_key = weight_key, scale_key
        self._bias_key, self._bits, self._group_size = bias_key, bits, group_size
        if (bits is None) != (bias_key is None) or (
            bits is not None and scale_key is None
        ):
            self._weights.close()
            self._scales.close()
            raise ValueError("Affine Engram requires bits, scales and biases")
        self._closed = False
        self._resident = None
        # Gathers started ahead of use; a plain object, not a Module child.
        self._queue = _PendingGathers()
        self._warm_rows = _WarmRows()
        self._hot_rows = HotRows(HOT_ROWS_BYTES)

    def make_resident(self):
        """Keep packed tensors in Metal-managed RAM with shared CPU views."""
        with self._lock:
            resident = {
                self._weight_key: self._weights.read(
                    self._weight_key, metal_backed=True
                ),
            }
            if self._scale_key is not None:
                resident[self._scale_key] = self._scales.read(
                    self._scale_key, metal_backed=True
                )
            if self._bias_key is not None:
                resident[self._bias_key] = self._scales.read(
                    self._bias_key, metal_backed=True
                )
            self._resident = resident
            self._weights.close()
            if self._scales is not self._weights:
                self._scales.close()

    def selected_bytes(self, rows):
        total = 0
        for reader, key in (
            (self._weights, self._weight_key),
            (self._scales, self._scale_key),
            (self._scales, self._bias_key),
        ):
            if key is not None:
                entry = reader.header[key]
                start, end = entry["data_offsets"]
                total += (end - start) // entry["shape"][0] * rows
        return total

    def _read_rows(self, host):
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram embedding is closed")
            result = []
            for reader, key in (
                (self._weights, self._weight_key),
                (self._scales, self._scale_key),
                (self._scales, self._bias_key),
            ):
                if key is None:
                    continue
                if self._resident is None:
                    result.append(reader.read(key, host.reshape(-1)))
                else:
                    raw, dtype = self._resident[key]
                    rows = host.reshape(-1)
                    if rows.size and (rows.min() < 0 or rows.max() >= raw.shape[0]):
                        raise IndexError("Engram row outside table")
                    result.append((raw[rows], dtype))
            return result

    def gather(self, host, *, warm_rows=False):
        """Raw row bytes for ``host`` ids, as ``_read_rows`` returns them.

        Large requests are deduplicated and sorted, read natively, and expanded
        back to request order, so every returned byte equals the mmap gather.
        """
        flat = np.asarray(host, dtype=np.int64).reshape(-1)
        cached = flat.size < NATIVE_MIN_ROWS
        if self._resident is not None or (cached and not DECODE_NATIVE) or not flat.size:
            return self._read_rows(flat)
        rows, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
        # The table is immutable and IDs, not prompt positions, are the key.
        # Decode keeps its existing tiny cached-pread path. Snapshot references
        # are immutable, so current/lookahead lanes may read concurrently.
        with self._warm_rows.lock:
            snapshot = (
                self._warm_rows.snapshot if WARM_ROWS_BYTES and not cached and not HOT_ROWS_BYTES else None
            )
        hits = np.zeros(rows.size, dtype=bool)
        hot = HOT_ROWS_BYTES and not cached
        if hot:
            hits, hot_data = self._hot_rows.lookup(rows)
        if snapshot is not None:
            saved, data = snapshot
            at = np.minimum(np.searchsorted(saved, rows), len(saved) - 1)
            hits = saved[at] == rows
        missing = rows[~hits]
        sources = [
            (reader, key)
            for reader, key in (
                (self._weights, self._weight_key),
                (self._scales, self._scale_key),
                (self._scales, self._bias_key),
            )
            if key is not None
        ]
        if self._closed:
            raise RuntimeError("Engram embedding is closed")
        if cached and len(sources) > 1:
            # Decode reads are SSD-latency bound: fetch the weight and scale
            # rows concurrently (the native reader releases the GIL).
            futures = [
                _PAGE_IO_POOL.submit(reader.gather_rows, key, missing, cached=True)
                for reader, key in sources
            ]
            gathered = [future.result() for future in futures]
        else:
            gathered = []
            for reader, key in sources:
                gathered.append(reader.gather_rows(key, missing, cached=cached))
                if gathered[-1] is None:
                    break
        if any(got is None for got in gathered):
            return self._read_rows(flat)
        if np.any(hits):
            merged = []
            for (raw, dtype), (saved_raw, saved_dtype) in zip(
                gathered, hot_data if hot else data
            ):
                assert dtype == saved_dtype
                full = np.empty((len(rows), *raw.shape[1:]), dtype=raw.dtype)
                full[hits] = saved_raw if hot else saved_raw[at[hits]]
                full[~hits] = raw
                merged.append((full, dtype))
            gathered = merged
        if hot:
            self._hot_rows.remember(rows, counts, gathered)
        elif warm_rows and WARM_ROWS_BYTES and not cached:
            self._warm_rows.remember(rows, gathered)
        return [(raw[inverse], dtype) for raw, dtype in gathered]

    def _locate(self, host):
        """Queued window holding the longest run of ``host``'s leading positions.

        Scheduler chunks shrink after being announced and the next
        announcement starts where the actual chunk ended, so ``host`` usually
        begins inside a queued window, at offset 0 or where the last claim
        stopped. Other offsets are found from anchor positions (the first,
        and the first past the n-gram history, which a wrong history guess
        cannot change) and verified over the whole overlap. Returns
        (window, offset, length) or None; caller holds the queue lock.
        """
        m, heads = host.shape
        if not m:
            return None
        best, best_score = None, 0.0
        for window in self._queue.items:
            rows = window.rows
            n = rows.shape[0]
            if rows.shape[1] != heads:
                continue
            offsets = [0, window.claimed]
            for anchor in dict.fromkeys((0, min(3, m - 1))):
                hits = np.flatnonzero((rows == host[anchor]).all(1)) - anchor
                offsets += hits[hits >= 0][:32].tolist()
            for offset in dict.fromkeys(offsets):
                if offset >= n:
                    continue
                length = min(m, n - offset)
                if length * 4 < n and not window.future.done():
                    continue  # don't wait on a large read for a small slice of it
                probe = slice(min(3, length - 1), min(length, 19))
                if 2 * np.count_nonzero(
                    rows[offset + probe.start : offset + probe.stop] == host[probe]
                ) <= (probe.stop - probe.start) * heads:
                    continue  # a repeated n-gram, not this window's position
                same = np.count_nonzero(rows[offset : offset + length] == host[:length])
                if same * 2 > length * heads and same > best_score:
                    best, best_score = (window, offset, length), same
        return best

    def pending_bytes(self):
        with self._queue.lock:
            return sum(self.selected_bytes(w.rows.size) for w in self._queue.items)

    def add_pending(self, window, limit=6):
        with self._queue.lock:
            self._queue.items.append(window)
            stale = self._queue.items[:-limit]
            del self._queue.items[:-limit]
        for old in stale:
            old.future.cancel()

    def plan(self, host, lane, *, max_bytes=None, warm_rows=False):
        """Queue a gather of ``host`` on ``lane``, reusing queued windows.

        Leading positions already queued are copied from that window when it
        completes; only the remainder is read.
        """
        rows = np.array(host, dtype=np.int64).reshape(-1, host.shape[-1] if host.ndim else 1)
        with self._queue.lock:
            found = self._locate(rows)
            if found is not None and found[2] == rows.shape[0]:
                return False
            if max_bytes is not None:
                while self._queue.items and (
                    sum(self.selected_bytes(w.rows.size) for w in self._queue.items)
                    + self.selected_bytes(rows.size)
                    > max_bytes
                ):
                    self._queue.items.pop(0).future.cancel()
            if found is None:
                source, offset, length = None, 0, 0
            else:
                source, offset, length = found
                if offset + length == source.rows.shape[0]:
                    source.handed_off = offset
                rows = np.concatenate([source.rows[offset : offset + length], rows[length:]])
            stats = self._queue.stats

        def job():
            kept = None
            if source is not None:
                try:
                    kept = source.future.result()
                except CancelledError:  # evicted before it ran: read it all
                    pass
            if kept is None:
                stats["read_ahead"] += rows.size
                return self.gather(rows, warm_rows=warm_rows)
            stats["reused"] += length * rows.shape[1]
            span = slice(offset * rows.shape[1], (offset + length) * rows.shape[1])
            kept = [(raw[span], dtype) for raw, dtype in kept]
            if length == rows.shape[0]:
                return [(raw.copy(), dtype) for raw, dtype in kept]
            stats["read_ahead"] += rows[length:].size
            own = self.gather(rows[length:])
            return [
                (np.concatenate([a, b]), dtype) for (a, dtype), (b, _) in zip(kept, own)
            ]

        self.add_pending(_Window(rows, lane.submit(job)))
        return True

    def discard_pending(self):
        with self._queue.lock:
            pending, self._queue.items = self._queue.items, []
        for window in pending:
            if not window.future.cancel():
                try:
                    window.future.result()
                except Exception:  # noqa: BLE001 -- the result is discarded
                    pass

    def _take_pending(self, host):
        """Assemble ``host``'s rows from queued windows; read what is missing.

        Row bytes depend only on the row id, so the result equals a
        synchronous read whatever was planned: positions no window covers and
        rows planned with a different n-gram history are read here.
        """
        rows = host.reshape(-1, host.shape[-1] if host.ndim else 1)
        m, heads = rows.shape
        parts, planned, covered = [], [], 0
        while covered < m:
            with self._queue.lock:
                found = self._locate(rows[covered:])
            if found is None:
                break
            window, offset, length = found
            try:
                gathered = window.future.result()
            except CancelledError:  # evicted before it ran
                with self._queue.lock:
                    self._queue.items = [w for w in self._queue.items if w is not window]
                continue
            span = slice(offset * heads, (offset + length) * heads)
            parts.append([(raw[span], dtype) for raw, dtype in gathered])
            planned.append(window.rows[offset : offset + length])
            with self._queue.lock:
                window.claimed = max(window.claimed, offset + length)
                self._queue.items = [w for w in self._queue.items if not w.done]
            covered += length
        if not parts:
            return None
        stats = self._queue.stats
        stats["claimed"] += covered * heads
        if covered < m:
            parts.append(self.gather(rows[covered:]))
            planned.append(rows[covered:])
            stats["read_at_claim"] += (m - covered) * heads
        data = [
            (np.concatenate([part[k][0] for part in parts]), parts[0][k][1])
            for k in range(len(parts[0]))
        ]
        wrong = np.flatnonzero(np.concatenate(planned).reshape(-1) != rows.reshape(-1))
        if wrong.size:
            stats["read_at_claim"] += wrong.size
            fixed = self.gather(rows.reshape(-1)[wrong])
            for (raw, _), (patch, _) in zip(data, fixed):
                raw[wrong] = patch
        return data

    def __call__(self, indices):
        host = np.asarray(indices).astype(np.int64)
        data = self._take_pending(host)
        if data is None:
            data = self.gather(host)
        values = decode_array(*data[0])
        if self._bits is not None:
            values = mx.dequantize(
                values,
                decode_array(*data[1]),
                decode_array(*data[2]),
                bits=self._bits,
                group_size=self._group_size,
                mode="affine",
            )
        elif self._scale_key is not None:
            scales = decode_array(*data[1])
            values = (
                values.reshape(values.shape[0], -1, 32) * scales[..., None]
            ).reshape(values.shape)
        return values.reshape(*host.shape, values.shape[-1]).astype(mx.bfloat16)

    def close(self):
        self.discard_pending()
        with self._warm_rows.lock:
            self._warm_rows.snapshot = None
        self._hot_rows.clear()
        with self._lock:
            self._closed = True
            self._resident = None
            self._weights.close()
            if self._scales is not self._weights:
                self._scales.close()


# Host bytes one table may hold in queued windows when a lookahead is queued
# (an 8192-token chunk is ~52 MB per table); the oldest windows go first.
LOOKAHEAD_MAX_BYTES = int(os.environ.get("DS41_ENGRAM_LOOKAHEAD_MIB", "256")) << 20


class EngramPrefetch:
    """Engram row gathers started before the rows are needed.

    ``submit`` queues a gather on one of two single-thread lanes: lookahead
    (the next prefill chunk, announced by the scheduler while the current
    chunk runs on the GPU) and current (rows for the forward in progress), so a
    long lookahead read never delays a decode step. Each lane is FIFO, so the
    first Engram layer's rows are read before later layers'. Windows stay
    queued on the embedding across forwards; a forward takes any prefix or
    range of them (the scheduler may shrink a chunk after announcing it), and
    a new announcement copies the unconsumed tail instead of re-reading it.
    """

    def __init__(self):
        self._lanes = {
            name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"v41-engram-{name}")
            for name in ("current", "lookahead")
        }
        self._embeds = []
        self._closed = False

    def submit(self, embed, ids, *, lookahead=False, warm_rows=False):
        if self._closed:
            raise RuntimeError("Engram prefetch is closed")
        if not isinstance(embed, DiskEngramEmbedding) or embed._resident is not None:
            return
        host = np.asarray(ids, dtype=np.int64)
        lane = self._lanes["lookahead" if lookahead else "current"]
        embed.plan(
            host,
            lane,
            max_bytes=LOOKAHEAD_MAX_BYTES if lookahead else None,
            warm_rows=warm_rows,
        )
        if embed not in self._embeds:
            self._embeds.append(embed)

    def drain(self):
        for embed in self._embeds:
            embed.discard_pending()

    @contextmanager
    def forward(self):
        try:
            yield self
        except BaseException:
            self.drain()
            raise

    def close(self):
        self._closed = True
        try:
            self.drain()
        finally:
            for lane in self._lanes.values():
                lane.shutdown(wait=True, cancel_futures=True)
