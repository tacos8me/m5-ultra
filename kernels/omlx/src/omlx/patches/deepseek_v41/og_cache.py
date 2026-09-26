# SPDX-License-Identifier: MIT
"""ds41-og prefix reuse, Mac side.

The box keeps exact snapshots of layers 0-19 (OPEN ``cache``) and returns the
same state bytes a fresh prefill would. The Mac's layers 20-39 need no
snapshot: after an OPEN they are a function of layer 20's global KV / index
rows and the replayed tail only. This store keeps those layer-20 rows
(356 B/token) per token prefix, so a resumed OPEN can ask for the rows past
the longest stored common prefix only (``delta_from``). Row r depends on
tokens[:r+1] (and on the content of any image among them) and the box
arithmetic, so rows of a stored prefix are reusable by any prompt sharing that
prefix under the same numerics key. Prefixes are compared on uint64 keys
(og_images.prompt_keys): token ids, and image-content keys inside image spans,
so a different image never matches.

Entries are immutable MLX arrays (the import's own arrays; decode appends go
to separate growth buffers). Lookups run on network threads, records on the
engine thread: every access holds the lock. Byte budget, whole-entry LRU.
"""

from collections import OrderedDict
from dataclasses import dataclass
import os
import threading

import numpy as np

ROW_BYTES = 288 + 68


@dataclass(eq=False)
class Entry:
    tokens: np.ndarray
    kv: object
    index: object
    key: tuple
    nbytes: int


def common_prefix(a, b):
    n = min(a.size, b.size)
    if not n:
        return 0
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


class RowStore:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.stats = dict(lookups=0, hits=0, hit_rows=0, stores=0, evictions=0, dropped_prefixes=0)

    def lookup(self, tokens, key=None):
        """(entry, P): the stored entry sharing the longest prefix with tokens (uint64 prefix keys)."""
        ids = np.asarray(tokens, np.uint64)
        with self.lock:
            self.stats['lookups'] += 1
            best, length = None, 0
            for entry in self.entries.values():
                if key is not None and entry.key != key:
                    continue
                n = common_prefix(ids, entry.tokens)
                if n > length:
                    best, length = entry, n
            if best is not None:
                self.entries.move_to_end(id(best))
                self.stats['hits'] += 1
                self.stats['hit_rows'] += length
            return best, length

    def record(self, tokens, kv, index, key):
        """Keep rows [0, len(tokens)) for this exact prefix (uint64 prefix keys)."""
        ids = np.asarray(tokens, np.uint64)
        n = ids.size
        if kv.shape[1] < n or index.shape[1] < n:
            raise ValueError('rows do not cover the tokens')
        nbytes = n * (ROW_BYTES + 8)
        if not n or nbytes > self.max_bytes:
            return False
        if kv.shape[1] != n:
            kv, index = kv[:, :n], index[:, :n]
        with self.lock:
            for k, entry in list(self.entries.items()):
                if entry.key != key:
                    # Another box numerics version: its rows can never be combined with these.
                    self._remove(k)
                    continue
                shared = common_prefix(ids, entry.tokens)
                if shared == entry.tokens.size:
                    self._remove(k)          # a prefix of the new entry: redundant
                    self.stats['dropped_prefixes'] += 1
                elif shared == n:
                    self.entries.move_to_end(k)
                    return False             # already covered by a longer entry
            while self.entries and self.bytes + nbytes > self.max_bytes:
                self._remove(next(iter(self.entries)))
                self.stats['evictions'] += 1
            entry = Entry(ids.copy(), kv, index, key, nbytes)
            self.entries[id(entry)] = entry
            self.bytes += nbytes
            self.stats['stores'] += 1
            return True

    def _remove(self, k):
        entry = self.entries.pop(k)
        self.bytes -= entry.nbytes

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.bytes = 0

    def summary(self):
        with self.lock:
            return dict(self.stats, entries=len(self.entries), bytes=self.bytes, max_bytes=self.max_bytes,
                        tokens=[int(e.tokens.size) for e in self.entries.values()])


def numerics_key(manifest):
    """Rows are reusable only under the same box identity and arithmetic."""
    return (manifest.get('identity'), manifest.get('numerics'))


def from_env():
    gib = float(os.environ.get('DS41_OG_CACHE_GIB', '4'))
    return RowStore(gib * 2**30) if gib > 0 else None
