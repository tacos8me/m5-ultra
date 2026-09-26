# SPDX-License-Identifier: MIT
"""Versioned, lossless DS41 c1 post-prefill handoff (before the final token).

This format includes the already replayed decoder window and DSpark prime ring.
It is a continuation snapshot, not an arbitrary reusable prefix. No pickle,
growth buffers, rollback stashes, weights or process-global state are serialized.
All MLX operations must run on the model's owning engine thread/stream.
"""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import mlx.core as mx

from .cache import DeepseekV41Cache
from ..mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache, _DSparkPrimeContext

FORMAT = "ds41-post-prefill-v1"
MAX_BYTES = 2 * 2**30


def token_digest(tokens):
    return hashlib.sha256(struct.pack("<%dI" % len(tokens), *tokens)).hexdigest()


def export_state(path, cache, tokens, *, identity):
    """Atomically export a complete prompt's N-1 state, preserving raw dtypes."""
    if len(cache) != 40 or len(tokens) < 2:
        raise ValueError("Expected DS41 c1 prompt state")
    arrays = {"tokens": mx.array(tokens, mx.uint32)}
    layers = []

    def put(name, value):
        if value is not None:
            arrays[name] = value
            return name
        return None

    for i, item in enumerate(cache):
        if item.size() != len(tokens) - 1:
            raise ValueError("Handoff must precede the final prompt token")
        layers.append({
            "compress_ratio": item.compress_ratio,
            "slots": [put(f"layer.{i}.slot.{j}", x) for j, x in enumerate(item.cache)],
            "left_padding": put(f"layer.{i}.left_padding", item.left_padding),
            "lengths": put(f"layer.{i}.lengths", item.lengths),
        })
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    prime = None
    if ctx is not None:
        prime = {"expected_target_offset": ctx.expected_target_offset, "stages": [
            {"max_size": s.max_size, "offset": s.offset,
             "keys": put(f"prime.{i}", s.keys)} for i, s in enumerate(ctx.caches)
        ]}
    size = sum(x.nbytes for x in arrays.values())
    if size > MAX_BYTES:
        raise ValueError("Handoff exceeds 2 GiB limit")
    manifest = dict(format=FORMAT, identity=identity, token_sha256=token_digest(tokens),
                    prompt_tokens=len(tokens), bytes=size, layers=layers, prime=prime)
    mx.eval(list(arrays.values()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".handoff-", suffix=".safetensors", dir=path.parent)
    os.close(fd)
    try:
        mx.save_safetensors(tmp, arrays, metadata={"manifest": json.dumps(manifest)})
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return manifest


def build_cache(arrays, manifest, tokens, *, identity):
    """Validate identity and exact token sequence, then rebuild the cache objects."""
    if (manifest["format"] != FORMAT or manifest["identity"] != identity
            or manifest["prompt_tokens"] != len(tokens)
            or manifest["token_sha256"] != token_digest(tokens)
            or arrays["tokens"].tolist() != list(tokens)):
        raise ValueError("Handoff format, identity or prompt mismatch")
    if len(manifest["layers"]) != 40:
        raise ValueError("Invalid handoff layer count")
    if sum(x.nbytes for x in arrays.values()) != manifest["bytes"]:
        raise ValueError("Invalid handoff byte count")

    def get(name):
        return arrays[name] if name is not None else None

    cache = []
    for layer in manifest["layers"]:
        if len(layer["slots"]) != 7:
            raise ValueError("Invalid cache slots")
        item = DeepseekV41Cache(layer["compress_ratio"])
        item.cache = [get(name) for name in layer["slots"]]
        item.left_padding = get(layer["left_padding"])
        item.lengths = get(layer["lengths"])
        if item.size() != len(tokens) - 1:
            raise ValueError("Invalid handoff offset")
        cache.append(item)
    if manifest["prime"] is not None:
        prime = manifest["prime"]
        stages = []
        for stage in prime["stages"]:
            item = DSparkContextCache(stage["max_size"])
            item.offset = stage["offset"]
            item.keys = get(stage["keys"])
            stages.append(item)
        cache[0]._omlx_mtp_prime_ctx = _DSparkPrimeContext(stages, prime["expected_target_offset"])
    return cache, manifest


def import_state(path, tokens, *, identity):
    """Load an exported file and admit it through build_cache."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES + 2**20:
        raise ValueError("Handoff exceeds size limit")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    return build_cache(arrays, json.loads(metadata["manifest"]), tokens, identity=identity)
