# SPDX-License-Identifier: MIT
"""Turn a ``ds41-encoder-state-v1`` state (encoder layers 0-20 + the residual
stream entering layer 20 for the last rows) into the full post-prefill state
by replaying the decoder half on the Mac with the served CED chunk geometry.

The served prefill runs 8192-token chunks; layers >= 20 see only the last 128
rows of a chunk longer than 128 tokens (windows dropped first), and every row
of a shorter final chunk (windows kept).  So the replay is one or two segments
over the supplied tail rows: the previous long chunk's 128-row tail (when the
final chunk is short) and the final chunk.  Layer 20's global KV/index rows
already exist (prebuilt_end); its attention-side hyper-connection mix is
computed with the kernel variant the full chunk would have used (hc_rows).
DSpark target hidden states (layers 37/38/39) are captured per segment exactly
like the served prompt capture, which rebuilds the prime ring.
"""

from collections import OrderedDict
import os
import time

import mlx.core as mx

from .cache import DeepseekV41Cache
from .handoff import token_digest
from .language import pack_activation
from ..mlx_lm_mtp.deepseek_v4_dspark import _PRIME_CTX_ATTR, _DSparkPrimeContext, capture_prompt

FORMAT = "ds41-encoder-state-v1"
CHUNK = 8192
# Host syncs in the replay: 1 = wait for every layer; N > 1 = submit each layer asynchronously and
# wait every N layers (same kernels on the same inputs, fewer idle gaps). Default 20 = one wait per
# segment (layers 20-39); ds41-fe window C/D and ds41-ttft: identical outputs and state digests.
EVAL_EVERY = max(1, int(os.environ.get("DS41_OG_REPLAY_EVAL_EVERY", "20")))
# Two-segment replays (prompts just past an 8K boundary): run the short final chunk one layer behind the
# long chunk's tail on a second GPU stream (same ops, same inputs; bitwise the sequential schedule).
OVERLAP = os.environ.get("DS41_OG_REPLAY_OVERLAP", "1") == "1"
# Reuse of a two-segment replay's first segment (the finished long chunk's 128-row tail): its result is a
# function of that tail's layer-20 input rows, layer 20's global rows before it, the token ids and the Mac
# code, so a later prompt with the same token prefix through that chunk (a follow-up turn whose final chunk
# is still short) takes the stored windows and DSpark context instead of recomputing them. Checked on use:
# box numerics key, token ids [0, b) and the tail rows bitwise. Entries (LRU); 0 = off.
SEG_CACHE = max(0, int(os.environ.get("DS41_OG_SEG_CACHE", "8")))
SEG_STATS = dict(seg0_hits=0, seg0_misses=0, seg0_stores=0, seg0_rejects=0)
_SEG0 = OrderedDict()


def segments(prefilled, chunk=CHUNK, window=128):
    """Replay plan for ``prefilled`` = N-1 prompt tokens.

    Returns [(row_start, row_end, chunk_len)] in execution order; rows are
    absolute positions, chunk_len the length of the chunk the rows belong to.
    """
    if prefilled <= 0:
        return []
    n_full, rem = divmod(prefilled, chunk)
    last = rem or chunk
    last_start = prefilled - last
    if last > window:
        return [(prefilled - window, prefilled, last)]
    plan = []
    if last_start:
        plan.append((last_start - window, last_start, chunk))
    plan.append((last_start, prefilled, last))
    return plan


def empty_slot(c, slot, dtype):
    if slot == 6:
        return mx.zeros((1, 0), mx.int64)
    width = c.index_head_dim if slot == 3 else c.head_dim
    empty = mx.zeros((1, 0, width), dtype)
    if slot == 1:
        return pack_activation(empty)
    if slot == 2:
        return pack_activation(empty, 4, 16, True)
    if slot == 3:
        return pack_activation(empty, 4)
    return empty


def build_cache(lm, arrays, manifest, tokens, *, identity):
    """Validate, rebuild the 21 encoder caches, replay layers 20..39, prime DSpark."""
    prefilled = len(tokens) - 1
    if (manifest["format"] != FORMAT or manifest["identity"] != identity
            or manifest["prompt_tokens"] != len(tokens)
            or manifest["token_sha256"] != token_digest(tokens)
            or arrays["tokens"].tolist() != list(tokens)):
        raise ValueError("Encoder state format, identity or prompt mismatch")
    if sum(x.nbytes for x in arrays.values()) != manifest["bytes"]:
        raise ValueError("Invalid encoder state byte count")
    c = lm._config
    mid = c.n_layers // 2
    if manifest.get("encoder_layers", len(manifest["layers"])) != mid + 1 or len(manifest["layers"]) != mid + 1:
        raise ValueError("Encoder state must carry layers 0..%d" % mid)
    tail = manifest["tail"]
    rows, first = int(tail["rows"]), int(tail["first_position"])
    if tail.get("layer", mid) != mid or first + rows != prefilled or rows < min(prefilled, 2 * c.window_size):
        raise ValueError("Encoder tail does not cover the replay rows")
    hidden, pre = arrays[tail["hidden"]], arrays[tail["pre"]]
    if hidden.shape != (1, rows, c.hc_mult, c.dim) or pre.shape != (1, rows, c.hc_mult):
        raise ValueError("Encoder tail shapes mismatch")

    def get(name):
        return arrays[name] if name is not None else None

    cache = []
    for i, layer in enumerate(manifest["layers"]):
        ratio = c.compress_ratios[i] if i in c.kv_source_layers else 0
        if layer["compress_ratio"] != ratio or len(layer["slots"]) != 7:
            raise ValueError("Encoder layer %d layout mismatch" % i)
        item = DeepseekV41Cache(ratio)
        item.cache = [get(name) for name in layer["slots"]]
        item.left_padding = get(layer.get("left_padding"))
        item.lengths = get(layer.get("lengths"))
        if item.size() != prefilled:
            raise ValueError("Encoder layer %d offset mismatch" % i)
        cache.append(item)
    for i in range(mid + 1, c.n_layers):
        item = DeepseekV41Cache(0)
        item.cache = [mx.array([prefilled], mx.int32)] + [None] * 6
        item.left_padding = item.lengths = None
        cache.append(item)
    replay(lm, cache, hidden, pre, first, tokens)
    return cache, manifest


def replay(lm, cache, hidden, pre, first, tokens, marks=None, seg_key=None):
    """Run layers mid..n-1 over the tail rows following the served geometry.

    ``marks`` (tracing only): a dict that receives wall-clock stamps per segment.
    ``seg_key``: the box numerics key of this state; enables the first-segment store (SEG_CACHE).
    """
    for _ in replay_steps(lm, cache, hidden, pre, first, tokens, marks, seg_key):
        pass
    return cache


def replay_steps(lm, cache, hidden, pre, first, tokens, marks=None, seg_key=None):
    """replay() as a generator: yields the row count after each submitted layer (1 per layer and segment),
    with no stream context held across a yield, and finishes with every cache array evaluated. The driver may
    run other work between steps (og_model admission slices); the ops and their order are unchanged."""
    c = lm._config
    prefilled = len(tokens) - 1
    ids = mx.array(tokens[:prefilled], mx.uint32)[None]
    prime = bool(getattr(lm, "_omlx_dspark_decode_enabled", False))
    # Image positions route through bias_vl (inference/model.py image_mask); text-only rows keep image_mask=None.
    image_id = getattr(c, "image_token_id", None)
    images = image_id is not None and bool(mx.any(ids == image_id).item())

    def image_mask(lo, hi):
        if not images:
            return None
        mask = ids[:, lo:hi] == image_id
        return mask if bool(mx.any(mask).item()) else None
    segs = [_Segment(seg, a, b, chunk_len, hidden, pre, first, c.window_size)
            for seg, (a, b, chunk_len) in enumerate(segments(prefilled, CHUNK, c.window_size))]
    two = len(segs) == 2 and segs[0].long and not segs[1].long
    store = SEG_CACHE and seg_key is not None and two and not images
    if store and _seg0_restore(lm, cache, segs[0], seg_key, ids, prime, marks):
        segs = segs[1:]
    elif store:
        segs[0].record = []
    if OVERLAP and len(segs) == 2 and not segs[1].long:
        yield from _replay_overlapped(lm, cache, hidden, pre, first, segs, ids, prime, image_mask, marks)
    else:
        for sg in segs:
            if sg.long:
                for i in range(c.n_layers // 2, c.n_layers):
                    cache[i][1] = None       # bounded replay skipped rows: stale window
            for i in range(c.n_layers // 2, c.n_layers):
                _layer(lm, cache, sg, i, hidden, pre, first, image_mask)
                if EVAL_EVERY == 1 or (i - c.n_layers // 2 + 1) % EVAL_EVERY == 0 or i == c.n_layers - 1:
                    mx.eval(sg.h, sg.p)
                else:
                    mx.async_eval(sg.h, sg.p)
                    yield sg.b - sg.a
            _finish(lm, cache, sg, ids, prime, marks)
    if store and segs[0].record is not None:
        _seg0_record(lm, cache, segs[0], seg_key, ids, prime)
    mx.eval([x for item in cache for x in item.cache if x is not None])
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    if ctx is not None:
        mx.eval([s.keys for s in ctx.caches if s.keys is not None])


class _Segment:
    """One replay segment: rows [a, b) of a chunk of chunk_len tokens, and its running h/pre."""

    def __init__(self, seg, a, b, chunk_len, hidden, pre, first, window):
        self.seg, self.a, self.b, self.chunk_len = seg, a, b, chunk_len
        self.long = chunk_len > window
        self.chunk_start = b - chunk_len
        self.h = self.h0 = hidden[:, a - first : b - first]
        self.p = self.p0 = pre[:, a - first : b - first]
        self.shared, self.captured = {}, {}
        self.record = None   # first-segment store: per layer (i, slots 1..6) after this segment's layer i
        self.prime = None    # and the DSpark context right after this segment's capture


def _layer(lm, cache, sg, i, hidden, pre, first, image_mask):
    """Layer i of segment sg (the served bounded-replay arithmetic; one call per layer and segment)."""
    c = lm._config
    mid = c.n_layers // 2
    layer = lm.layers[i]
    if getattr(lm, "_omlx_dspark_decode_enabled", False) and i in c.dspark_target_layer_ids:
        sg.captured[i] = mx.mean(sg.h, axis=2)
    a, b = sg.a, sg.b
    if i == mid and sg.long and sg.chunk_start >= first:
        # The whole chunk's layer-20 input is available: run the served
        # bounded-replay path itself (chunk-wide hyper-connection mixes,
        # queries on the tail), with the global rows already prebuilt.
        cs = sg.chunk_start
        sg.h, sg.p = layer(
            hidden[:, cs - first : b - first], pre[:, cs - first : b - first],
            cache[i], sg.shared, cs, image_mask(cs, b), ced_tail=c.window_size, prebuilt_end=b,
        )
    else:
        # Tail rows only; the chunk-wide mixes of layer 20 are per-row
        # kernels whose variant follows the chunk length (hc_rows).
        sg.h, sg.p = layer(
            sg.h, sg.p, cache[i], sg.shared, a, image_mask(a, b),
            prebuilt_end=b if i == mid else None,
            hc_rows=sg.chunk_len if (sg.long and i == mid) else None,
        )
    cache[i][0] = mx.array([b], mx.int32)
    for slot in range(1, 7):
        if cache[i][slot] is None:
            cache[i][slot] = empty_slot(c, slot, sg.h.dtype)
    if sg.record is not None:
        sg.record.append((i, [cache[i][slot] for slot in range(1, 7)]))


def _finish(lm, cache, sg, ids, prime, marks):
    """After a segment's last layer: layers 0-19 offsets, trace mark, DSpark prompt capture."""
    c = lm._config
    for i in range(c.n_layers // 2):
        cache[i][0] = mx.array([sg.b], mx.int32)
    if marks is not None:
        marks.setdefault("og.replay_layers%d" % sg.seg, time.time())
    if prime:
        aux = mx.concatenate([sg.captured[i] for i in c.dspark_target_layer_ids], axis=-1)
        capture_prompt(lm, ids[:, sg.a:sg.b], aux, cache)
        if sg.record is not None:
            ctx = getattr(_prime_owner(lm, cache), _PRIME_CTX_ATTR, None)
            sg.prime = None if ctx is None else (ctx.expected_target_offset,
                                                 [(x.max_size, x.offset, x.keys) for x in ctx.caches])
        if marks is not None:
            ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
            mx.eval([s.keys for s in ctx.caches if s.keys is not None] if ctx is not None else [])
            marks.setdefault("og.replay_prime%d" % sg.seg, time.time())


_SIDE = []


def _side_stream():
    if not _SIDE:
        _SIDE.append(mx.new_stream(mx.default_device()))
    return _SIDE[0]


def _replay_overlapped(lm, cache, hidden, pre, first, segs, ids, prime, image_mask, marks):
    """Two segments (a long chunk's 128-row tail + the short final chunk): segment 1's layer i runs on a
    side GPU stream right after segment 0's layer i (the window it reads), so it overlaps segment 0's
    layer i+1. Each segment issues exactly the ops of the sequential schedule on the same inputs; only
    the stream and the submission order differ, so the state is bitwise the sequential one. A generator
    (see replay_steps); it never yields inside the side-stream context."""
    c = lm._config
    mid = c.n_layers // 2
    s0, s1 = segs
    main, side = mx.default_stream(mx.default_device()), _side_stream()
    if s0.long:
        for i in range(mid, c.n_layers):
            cache[i][1] = None
    for i in range(mid, c.n_layers):
        with mx.stream(main):
            _layer(lm, cache, s0, i, hidden, pre, first, image_mask)
            mx.async_eval(s0.h, s0.p)
            if i == c.n_layers - 1:
                # Segment 0 done: its offsets (read by the DSpark capture) and capture, as in the
                # sequential schedule, before segment 1's last layer sets the final offsets.
                _finish(lm, cache, s0, ids, prime, marks)
        yield s0.b - s0.a
        with mx.stream(side):
            _layer(lm, cache, s1, i, hidden, pre, first, image_mask)
            mx.async_eval(s1.h, s1.p)
        yield s1.b - s1.a
    with mx.stream(main):
        mx.eval(s1.h, s1.p)
        _finish(lm, cache, s1, ids, prime, marks)


def _prime_owner(lm, cache):
    return cache[0] if getattr(lm, "_omlx_dspark_cache_owned", False) else lm


class _Seg0:
    def __init__(self, key, ids, h, p, slots, prime):
        self.key, self.ids, self.h, self.p, self.slots, self.prime = key, ids, h, p, slots, prime


def _bits_equal(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    view = mx.uint16 if x.dtype in (mx.bfloat16, mx.float16) else (mx.uint32 if x.dtype == mx.float32 else x.dtype)
    return bool(mx.array_equal(x.view(view), y.view(view)).item())


def _seg0_record(lm, cache, sg, seg_key, ids, prime):
    """Keep a computed first segment (windows/slots of layers 20-39 after it, its DSpark context)."""
    c = lm._config
    if len(sg.record) != c.n_layers - c.n_layers // 2 or (prime and sg.prime is None):
        return
    key = (seg_key, sg.a, sg.b, sg.chunk_len, bool(prime))
    entry = _Seg0(key, ids[:, : sg.b], sg.h0, sg.p0, dict(sg.record), sg.prime if prime else None)
    for k, old in list(_SEG0.items()):
        if old.key == key and _bits_equal(old.ids, entry.ids) and _bits_equal(old.h, entry.h):
            del _SEG0[k]  # the same inputs again: keep the newest
    _SEG0[id(entry)] = entry
    while len(_SEG0) > SEG_CACHE:
        _SEG0.popitem(last=False)
    SEG_STATS["seg0_stores"] += 1


def _seg0_restore(lm, cache, sg, seg_key, ids, prime, marks):
    """Install a stored first segment whose inputs equal this one's bitwise; False if none."""
    c = lm._config
    key = (seg_key, sg.a, sg.b, sg.chunk_len, bool(prime))
    candidates = [(k, e) for k, e in _SEG0.items() if e.key == key]
    prefix = ids[:, : sg.b]
    same = [(k, e) for k, e in candidates if _bits_equal(e.ids, prefix)]
    if not same:
        SEG_STATS["seg0_misses"] += 1
        return False
    k, entry = same[-1]
    if not (_bits_equal(entry.h, sg.h0) and _bits_equal(entry.p, sg.p0)):
        # Same token prefix, other layer-20 input rows (box arithmetic changed): never reuse.
        SEG_STATS["seg0_rejects"] += 1
        return False
    _SEG0.move_to_end(k)
    mid = c.n_layers // 2
    for i in range(mid, c.n_layers):
        slots = entry.slots[i]
        for slot in range(1, 7):
            if i == mid and slot in (2, 3):
                continue  # layer 20's global rows are this request's own (same values through b)
            cache[i][slot] = slots[slot - 1]
        cache[i][0] = mx.array([sg.b], mx.int32)
    for i in range(mid):
        cache[i][0] = mx.array([sg.b], mx.int32)
    if prime:
        expected, stages = entry.prime
        ctx = _DSparkPrimeContext(caches=lm.make_mtp_cache(), expected_target_offset=expected)
        for item, (max_size, offset, keys) in zip(ctx.caches, stages):
            if item.max_size != max_size:
                raise ValueError("DSpark context geometry changed")
            item.offset, item.keys = offset, keys
        setattr(_prime_owner(lm, cache), _PRIME_CTX_ATTR, ctx)
    SEG_STATS["seg0_hits"] += 1
    if marks is not None:
        marks.setdefault("og.replay_layers0", time.time())
        marks.setdefault("og.replay_prime0", time.time())
    return True
