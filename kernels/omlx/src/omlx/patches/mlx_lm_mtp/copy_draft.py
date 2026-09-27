# SPDX-License-Identifier: MIT
"""Context-copy drafts for the DSpark verify loop.

Agent and code turns repeat long spans of the prompt or of earlier output
verbatim. A hashed n-gram index over prompt + committed tokens finds the
earlier span whose tail matches the newest committed tokens, and the tokens
that followed it become the draft (4 rows by default, so the verify keeps the
served L<=5 arithmetic). The target verify accepts or rejects those rows
exactly as it does DSpark rows. The policy reacts to acceptance
only, never to time, so a request's draft sequence is deterministic.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

ENABLED = os.environ.get("DS41_COPY_DRAFT", "1") == "1"
NGRAM = max(2, int(os.environ.get("DS41_COPY_NGRAM", "4")))
# Verify rows stay at L<=5 (the served k=4 numeric path): at most 4 copy tokens.
MIN_DRAFT = max(1, int(os.environ.get("DS41_COPY_MIN", "4")))
MAX_DRAFT = max(MIN_DRAFT, int(os.environ.get("DS41_COPY_MAX", "4")))
# A rejected lock raises the required match length; successful copies relax it.
MAX_MATCH = max(NGRAM, int(os.environ.get("DS41_COPY_MAX_MATCH", "16")))
CANDIDATES = 16
MATCH_CAP = 64
_P = 1000003
_MASK = (1 << 64) - 1


def _prompt_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """uint64 hash of every n-gram, indexed by its start position."""
    t = tokens.astype(np.uint64)
    h = np.zeros(len(t) - n + 1, np.uint64)
    p = np.uint64(_P)
    for i in range(n):
        h = h * p + t[i : len(t) - n + 1 + i]
    return h


def _hash(values) -> int:
    h = 0
    for x in values:
        h = (h * _P + int(x)) & _MASK
    return h


class PromptIndex:
    """The prompt half of a CopyIndex, built ahead of time (ds41-og: while the box prefills).

    ``CopyIndex.from_prompt(PromptIndex(prompt), extra)`` has exactly the state
    ``CopyIndex(prompt + extra)`` builds, but only hashes and inserts the
    ``len(extra)`` new n-grams on the critical path. The list conversion runs in
    chunks so other threads get the GIL between them; hashing and the sort run
    in numpy.
    """

    CHUNK = 1 << 16

    def __init__(self, tokens: List[int], extra: int = 1):
        n = len(tokens)
        total = n + extra
        self.buf = np.zeros(max(2 * total, total + 8192), np.int64)
        for lo in range(0, n, self.CHUNK):
            hi = min(n, lo + self.CHUNK)
            self.buf[lo:hi] = tokens[lo:hi]
        self.n, self.extra = n, extra
        self.order = self.sorted = None
        if n >= NGRAM:
            h = _prompt_hashes(self.buf[:n], NGRAM)
            self.order = np.argsort(h, kind="stable").astype(np.int64)
            self.sorted = h[self.order]

    def matches(self, tokens: List[int], n: int) -> bool:
        """Cheap identity check against ``tokens[:n]`` (length, head and tail)."""
        if n != self.n or len(tokens) < n:
            return False
        k = min(n, 256)
        return (
            self.buf[:k].tolist() == list(tokens[:k])
            and self.buf[n - k : n].tolist() == list(tokens[n - k : n])
        )


class CopyIndex:
    """n-gram index over the token stream plus the acceptance-only draft policy."""

    def __init__(self, tokens: List[int]):
        n = len(tokens)
        self._buf = np.zeros(max(2 * n, n + 8192), np.int64)
        self._buf[:n] = tokens
        self.n = n
        if n >= NGRAM:
            h = _prompt_hashes(self._buf[:n], NGRAM)
            self._order = np.argsort(h, kind="stable").astype(np.int64)
            self._sorted = h[self._order]
        else:
            self._order = np.zeros(0, np.int64)
            self._sorted = np.zeros(0, np.uint64)
        self._init_policy()

    @classmethod
    def from_prompt(cls, pre: PromptIndex, extra: List[int]) -> Optional["CopyIndex"]:
        """``CopyIndex(prompt + extra)`` from a PromptIndex of ``prompt``, or None if it does not apply.

        The stable argsort orders equal hashes by start position, and every new
        n-gram starts after all prompt n-grams, so each one goes right after its
        equal keys: the same ``_order``/``_sorted`` arrays, bit for bit. Consumes
        ``pre`` (its buffer becomes this index's buffer).
        """
        m = len(extra)
        if pre.order is None or m != pre.extra or pre.buf is None:
            return None
        self = cls.__new__(cls)
        base, n = pre.n, pre.n + m
        buf, pre.buf = pre.buf, None
        buf[base:n] = extra
        order, keys = pre.order, pre.sorted
        new = _prompt_hashes(buf[base + 1 - NGRAM : n], NGRAM)
        for j, key in enumerate(new):
            at = int(np.searchsorted(keys, key, "right"))
            keys = np.insert(keys, at, key)
            order = np.insert(order, at, base + 1 - NGRAM + j)
        self._buf, self.n, self._order, self._sorted = buf, n, order, keys
        self._init_policy()
        return self

    def _init_policy(self) -> None:
        self._extra: dict[int, list[int]] = {}
        self.cur = MIN_DRAFT
        self.min_match = NGRAM
        self.expected: Optional[int] = None
        self.pending: Optional[tuple[int, int]] = None

    def append(self, ids: List[int]) -> None:
        m = len(ids)
        if self.n + m > len(self._buf):
            grown = np.zeros(2 * (self.n + m), np.int64)
            grown[: self.n] = self._buf[: self.n]
            self._buf = grown
        self._buf[self.n : self.n + m] = ids
        for p in range(self.n, self.n + m):
            if p + 1 >= NGRAM:
                key = _hash(self._buf[p + 1 - NGRAM : p + 1])
                self._extra.setdefault(key, []).append(p + 1)
        self.n += m

    def _candidates(self, key: int) -> List[int]:
        n = self.n
        found: list[int] = []
        lo = int(np.searchsorted(self._sorted, np.uint64(key), "left"))
        hi = int(np.searchsorted(self._sorted, np.uint64(key), "right"))
        if hi > lo:
            starts = self._order[lo:hi]
            if len(starts) > CANDIDATES:
                starts = np.sort(starts)[-CANDIDATES:]
            found.extend(int(s) + NGRAM for s in starts)
        extra = self._extra.get(key)
        if extra:
            found.extend(extra[-CANDIDATES:])
        if self.expected is not None:
            found.append(self.expected)
        return sorted({c for c in found if NGRAM <= c < n})

    def propose(self, budget: int) -> Optional[List[int]]:
        """Tokens that followed the best earlier match of the current tail, or None."""
        n = self.n
        if not ENABLED or budget < 1 or n < NGRAM + 1:
            return None
        cands = self._candidates(_hash(self._buf[n - NGRAM : n]))
        if not cands:
            return None
        c = np.array(cands, np.int64)
        j = np.arange(MATCH_CAP)
        src = c[:, None] - 1 - j[None, :]
        valid = src >= 0
        eq = np.zeros(src.shape, bool)
        dst = np.broadcast_to(n - 1 - j[None, :], src.shape)
        eq[valid] = self._buf[src[valid]] == self._buf[dst[valid]]
        match = np.cumprod(eq, axis=1).sum(axis=1)
        best = int(match.max())
        if best < self.min_match:
            return None
        ties = [cands[i] for i in np.flatnonzero(match == best)]
        source = self.expected if self.expected in ties else ties[-1]
        k = min(self.cur, MAX_DRAFT, budget, n - source)
        if k < 1:
            return None
        self.pending = (source, k)
        return self._buf[source : source + k].tolist()

    def observe(self, accepted: int) -> None:
        if self.pending is None:
            return
        source, k = self.pending
        self.pending = None
        if accepted >= k:
            self.cur = min(MAX_DRAFT, 2 * self.cur)
            self.expected = source + k + 1
            self.min_match = max(NGRAM, self.min_match - 2)
        else:
            self.cur = MIN_DRAFT
            self.expected = None
            if accepted < 2:
                self.min_match = min(MAX_MATCH, self.min_match + 4)

