# SPDX-License-Identifier: MIT
"""Exact, bounded, process-local checkpoints of the CED prefill program.

Only complete 8192-token chunks are reusable. Decode states and partial chunk
states are never admitted: their arithmetic and decoder replay differ from a
cold prefill of a longer prompt. Each checkpoint includes the decoder window,
compressed KV, index keys, compressor tails, Engram history and DSpark prime
ring. Model identity is implicit in ownership by one loaded language model.

No SSD writes or background MLX work. The engine thread owns all operations.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
import os
import struct
import time

import mlx.core as mx

from .cache import DeepseekV41Cache
from ..mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache, _DSparkPrimeContext

logger = logging.getLogger(__name__)
CHUNK = 8192


_copy_kernel = None


def compact_copy(value):
    """Own exactly the live bytes; cache views can retain much larger buffers."""
    global _copy_kernel
    if value is None:
        return None
    if not value.size:
        # An empty compressor tail may still be a view of a whole chunk.
        return mx.zeros(value.shape, value.dtype)
    if _copy_kernel is None:
        _copy_kernel = mx.fast.metal_kernel(
            name="ds41_prefix_snapshot_bytes",
            input_names=["src", "count"],
            output_names=["dst"],
            source="uint i = thread_position_in_grid.x; if (i < count[0]) dst[i] = src[i];",
        )
    raw = mx.contiguous(value).view(mx.uint8)
    copied = _copy_kernel(
        inputs=[raw, mx.array([raw.size], mx.uint32)],
        grid=(raw.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[raw.shape],
        output_dtypes=[mx.uint8],
    )[0]
    return copied.view(value.dtype).reshape(value.shape)


def clone_cache(cache, *, compact=False):
    """New Python/MLX handles; subsequent writes use MLX copy-on-write.

    Do not retain growth buffers, speculative rollback stashes or request
    metadata. These are mutable accelerators, not the committed model state.
    """
    copy = compact_copy if compact else lambda x: None if x is None else mx.array(x)
    result = []
    for item in cache:
        new = DeepseekV41Cache(item.compress_ratio)
        new.cache = [copy(x) for x in item.cache]
        new.left_padding = copy(item.left_padding)
        new.lengths = copy(item.lengths)
        result.append(new)
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    if ctx is not None:
        stages = []
        for stage in ctx.caches:
            copied = DSparkContextCache(stage.max_size)
            copied.offset = stage.offset
            copied.keys = copy(stage.keys)
            stages.append(copied)
        result[0]._omlx_mtp_prime_ctx = _DSparkPrimeContext(
            stages, ctx.expected_target_offset
        )
    return result


def cache_arrays(cache):
    arrays = [x for c in cache for x in c.cache if x is not None]
    arrays += [x for c in cache for x in (c.left_padding, c.lengths) if x is not None]
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    if ctx is not None:
        arrays += [c.keys for c in ctx.caches if c.keys is not None]
    return arrays


@dataclass
class Plan:
    tokens: bytes
    keys: dict
    next_position: int
    valid: bool = True


@dataclass
class Entry:
    tokens: bytes
    cache: list
    size: int


class ExactPrefixCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.stores = self.evictions = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def prepare(self, tokens, step=CHUNK):
        # Leave the final prompt token for the ordinary generation kickoff.
        if not self.max_bytes or step != CHUNK or len(tokens) <= CHUNK:
            return None, 0, None
        packed = struct.pack("<%dI" % len(tokens), *tokens)
        digest = hashlib.sha256(b"ds41-ced-8192-v1")
        keys = {}
        for end in range(CHUNK, len(tokens), CHUNK):
            digest.update(packed[(end - CHUNK) * 4 : end * 4])
            keys[end] = digest.copy().digest()
        for end, key in reversed(list(keys.items())):
            entry = self.entries.get(key)
            # Hashes are an index only: compare every token before reuse.
            if entry is not None and entry.tokens == packed[: end * 4]:
                t0 = time.perf_counter()
                cache = clone_cache(entry.cache)
                self.entries.move_to_end(key)
                self.hits += 1
                logger.info(
                    "DS41 exact prefix hit: tokens=%d restore_ms=%.3f hot_MiB=%.1f",
                    end,
                    (time.perf_counter() - t0) * 1000,
                    self.bytes / 2**20,
                )
                return cache, end, Plan(packed, keys, end)
        return None, 0, Plan(packed, keys, 0)

    def record(self, plan, cache, start, count):
        if plan is None or not plan.valid:
            return
        if count != CHUNK or start != plan.next_position or start % CHUNK:
            plan.valid = False
            return
        end = start + count
        plan.next_position = end
        key = plan.keys.get(end)
        if key is None:
            plan.valid = False
            return
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        arrays = cache_arrays(cache)
        # Include metadata/tokens and array views conservatively (shared buffers
        # may be double-counted). Evict BEFORE retaining another checkpoint.
        size = sum(x.nbytes for x in arrays) + end * 4
        if size > self.max_bytes:
            return
        while self.entries and self.bytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.size
            del old
            self.evictions += 1
        mx.eval(arrays)
        snapshot = clone_cache(cache, compact=True)
        mx.eval(cache_arrays(snapshot))
        self.entries[key] = Entry(plan.tokens[: end * 4], snapshot, size)
        self.bytes += size
        self.stores += 1


def for_model(model):
    if not model._config.ced_prefill:
        return None
    limit = float(os.environ.get("DS41_PREFIX_CACHE_GIB", "0"))
    if limit <= 0:
        return None
    # Keep checkpoints outside nn.Module's parameter tree.
    signature = (
        bool(getattr(model, "_omlx_dspark_decode_enabled", False)),
        getattr(model, "_omlx_mtp_depth", None),
    )
    manager = model.__dict__.get("_exact_prefix_cache")
    if manager is None or getattr(manager, "signature", None) != signature:
        manager = ExactPrefixCache(min(limit, 3.0) * 2**30)
        manager.signature = signature
        object.__setattr__(model, "_exact_prefix_cache", manager)
    return manager
