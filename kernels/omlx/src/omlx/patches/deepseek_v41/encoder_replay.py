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

import mlx.core as mx

from .cache import DeepseekV41Cache
from .handoff import token_digest
from .language import pack_activation
from ..mlx_lm_mtp.deepseek_v4_dspark import capture_prompt

FORMAT = "ds41-encoder-state-v1"
CHUNK = 8192


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


def replay(lm, cache, hidden, pre, first, tokens):
    """Run layers mid..n-1 over the tail rows following the served geometry."""
    c = lm._config
    mid = c.n_layers // 2
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
    for a, b, chunk_len in segments(prefilled, CHUNK, c.window_size):
        long = chunk_len > c.window_size
        chunk_start = b - chunk_len
        h = hidden[:, a - first : b - first]
        p = pre[:, a - first : b - first]
        shared = {}
        captured = {}
        if long:
            for i in range(mid, c.n_layers):
                cache[i][1] = None       # bounded replay skipped rows: stale window
        for i in range(mid, c.n_layers):
            layer = lm.layers[i]
            if prime and i in c.dspark_target_layer_ids:
                captured[i] = mx.mean(h, axis=2)
            if i == mid and long and chunk_start >= first:
                # The whole chunk's layer-20 input is available: run the served
                # bounded-replay path itself (chunk-wide hyper-connection mixes,
                # queries on the tail), with the global rows already prebuilt.
                h, p = layer(
                    hidden[:, chunk_start - first : b - first], pre[:, chunk_start - first : b - first],
                    cache[i], shared, chunk_start, image_mask(chunk_start, b), ced_tail=c.window_size, prebuilt_end=b,
                )
            else:
                # Tail rows only; the chunk-wide mixes of layer 20 are per-row
                # kernels whose variant follows the chunk length (hc_rows).
                h, p = layer(
                    h, p, cache[i], shared, a, image_mask(a, b),
                    prebuilt_end=b if i == mid else None,
                    hc_rows=chunk_len if (long and i == mid) else None,
                )
            mx.eval(h, p)
            cache[i][0] = mx.array([b], mx.int32)
            for slot in range(1, 7):
                if cache[i][slot] is None:
                    cache[i][slot] = empty_slot(c, slot, h.dtype)
        for i in range(mid):
            cache[i][0] = mx.array([b], mx.int32)
        if prime:
            aux = mx.concatenate([captured[i] for i in c.dspark_target_layer_ids], axis=-1)
            capture_prompt(lm, ids[:, a:b], aux, cache)
    mx.eval([x for item in cache for x in item.cache if x is not None])
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    if ctx is not None:
        mx.eval([s.keys for s in ctx.caches if s.keys is not None])
    return cache
