# SPDX-License-Identifier: Apache-2.0
"""Bounded, process-persistent admission cache for immutable Engram row bytes."""
from threading import Lock

import numpy as np


class HotRows:
    """Frequency admission with direct-mapped slots and full row-ID validation.

    Only CPU byte arrays are retained, outside the model parameter tree. The
    sketch decides admission, never row identity. Returned hits own their bytes
    so another prefetch lane can replace slots while SSD misses are in flight.
    ``budget`` includes payload, tags and the frequency sketch.
    """

    def __init__(self, budget):
        self.budget = max(0, int(budget))
        self.lock = Lock()
        self.tags = None
        self.data = None
        self.freq = None
        self.calls = 0
        self.hits = self.queries = self.admitted = 0

    @staticmethod
    def _hash(rows):
        value = rows.astype(np.uint64)
        value = (value ^ (value >> np.uint64(16))) * np.uint64(0x9E3779B185EBCA87)
        return value ^ (value >> np.uint64(32))

    @property
    def nbytes(self):
        with self.lock:
            return 0 if self.tags is None else (
                self.tags.nbytes + self.freq.nbytes
                + sum(raw.nbytes for raw, _ in self.data)
            )

    def lookup(self, rows):
        with self.lock:
            self.queries += len(rows)
            if self.tags is None:
                return np.zeros(len(rows), dtype=bool), None
            slots = rows % len(self.tags)
            hits = (rows >= 0) & (self.tags[slots] == rows)
            self.hits += int(hits.sum())
            return hits, tuple((raw[slots[hits]], dtype) for raw, dtype in self.data)

    def remember(self, rows, counts, data):
        if not len(rows) or not self.budget:
            return
        with self.lock:
            if self.tags is None:
                # At most 8 MiB per table, and small test budgets stay bounded.
                sketch_size = min(1 << 22, max(1, self.budget // 128))
                row_bytes = 8 + sum(raw[0].nbytes for raw, _ in data)
                capacity = (self.budget - sketch_size * 2) // row_bytes
                if capacity <= 0:
                    return
                self.freq = np.zeros(sketch_size, dtype=np.uint16)
                self.tags = np.full(capacity, -1, dtype=np.int64)
                self.data = tuple(
                    (np.empty((capacity, *raw.shape[1:]), dtype=raw.dtype), dtype)
                    for raw, dtype in data
                )
            self.calls += 1
            if self.calls % 256 == 0:
                self.freq >>= 1
            buckets = (self._hash(rows) % len(self.freq)).astype(np.intp)
            # Aggregate sketch collisions before saturating; uint16 never wraps.
            order = np.argsort(buckets)
            start = np.r_[0, np.flatnonzero(np.diff(buckets[order])) + 1]
            unique = buckets[order[start]]
            increment = np.add.reduceat(counts[order].astype(np.int64), start)
            self.freq[unique] = np.minimum(65535, self.freq[unique].astype(np.int64) + increment)
            scores = self.freq[buckets]
            slots = rows % len(self.tags)
            # One winner per slot, highest observed frequency; ties stable by ID.
            order = np.lexsort((rows, scores, slots))
            keep = np.r_[np.diff(slots[order]) != 0, True]
            chosen = order[keep]
            dest = slots[chosen]
            old = self.tags[dest]
            old_scores = self.freq[(self._hash(old) % len(self.freq)).astype(np.intp)]
            admit = (old < 0) | ((old != rows[chosen]) & (scores[chosen] > old_scores))
            chosen, dest = chosen[admit], dest[admit]
            for (target, target_dtype), (raw, dtype) in zip(self.data, data):
                assert target_dtype == dtype
                target[dest] = raw[chosen]
            self.tags[dest] = rows[chosen]
            self.admitted += len(chosen)

    def clear(self):
        with self.lock:
            self.tags = self.data = self.freq = None
