"""Image spans in an OPEN, and the prefix-cache keys that make a cached state depend on image content.

An image occupies a span of the prompt where every token id is `image_token_id` (reference layout
[IMAGE_START] + ([IMAGE] * nw + [NEWLINE]) * nh + [IMAGE_END], nh/nw = ceil(vit grid / 3)); the ids alone
cannot tell two images apart. Cache keys are uint64: the token id at text positions, and
(1 << 63) | first 63 bits of sha256(image digest || offset as uint32 LE) at image positions, so any
common-prefix match stops exactly where image content differs. The Mac (og_cache / pipe_wire) computes the
same keys with the same function (test vector in tools/test_imagekeys.py).
"""

import hashlib
import math
import struct

import numpy as np

PATCH = 14
DOWNSAMPLE = 3
PATCH_BYTES = 3 * PATCH * PATCH * 2  # one bf16 patch
IMAGE_BIT = 1 << 63


def num_image_tokens(vit_h, vit_w):
    nh, nw = math.ceil(vit_h / DOWNSAMPLE), math.ceil(vit_w / DOWNSAMPLE)
    return nh * (nw + 1) + 2


def image_digest(vit_h, vit_w, patch_bytes):
    """Content digest of what the box consumes: the vit grid and the bf16 patch bytes."""
    h = hashlib.sha256(struct.pack("<II", vit_h, vit_w))
    h.update(patch_bytes)
    return h.hexdigest()


def span_keys(digest_hex, length):
    d = bytes.fromhex(digest_hex)
    out = np.empty(length, dtype=np.uint64)
    for k in range(length):
        v = int.from_bytes(hashlib.sha256(d + struct.pack("<I", k)).digest()[:8], "little")
        out[k] = (v & (IMAGE_BIT - 1)) | IMAGE_BIT
    return out


def prompt_keys(tokens, images):
    """uint64 keys for `tokens` (any length); images: iterable of (start, length, digest_hex)."""
    keys = np.asarray(tokens, dtype=np.uint64).copy()
    for start, length, digest in images:
        if start >= len(keys):
            continue
        k = span_keys(digest, length)[: len(keys) - start]
        keys[start:start + len(k)] = k
    return keys


def key_tokens(keys, image_token_id):
    """Token ids back from keys (image positions carry image_token_id)."""
    keys = np.asarray(keys, dtype=np.uint64)
    return np.where(keys >= np.uint64(IMAGE_BIT), np.uint64(image_token_id), keys).astype(np.int64)


def prefix_digest(tokens, keys, P, has_images):
    """delta_from check: sha256 of uint32 tokens[:P] (text-only OPENs, unchanged) or of uint64 LE keys[:P]."""
    if has_images:
        return hashlib.sha256(np.asarray(keys[:P], dtype="<u8").tobytes()).hexdigest()
    return hashlib.sha256(struct.pack("<%dI" % P, *tokens[:P])).hexdigest()


def parse_images(header, tokens, payload_tail, image_token_id):
    """Validate OPEN images against the prompt; returns [(start, length, vit_h, vit_w, digest, patch_bytes)].

    Spans lie inside prompt[:-1], do not overlap, carry image_token_id everywhere, and every image_token_id of
    the prompt lies in a span (routing and Engram treat that id as an image position)."""
    specs = header.get("images") or []
    n1 = len(tokens) - 1
    out, off, last_end = [], 0, 0
    for spec in sorted(specs, key=lambda s: int(s["start"])):
        start, (vh, vw) = int(spec["start"]), (int(spec["grid"][0]), int(spec["grid"][1]))
        length = num_image_tokens(vh, vw)
        if int(spec.get("length", length)) != length or vh < 1 or vw < 1:
            raise ValueError(f"image at {start}: grid {vh}x{vw} needs a {length}-token span")
        if start < last_end or start + length > n1:
            raise ValueError(f"image span {start}+{length} overlaps or leaves prompt[:-1] ({n1})")
        nbytes = vh * vw * PATCH_BYTES
        data = payload_tail[off:off + nbytes]
        if len(data) != nbytes:
            raise ValueError("image payload shorter than its grids")
        off += nbytes
        digest = image_digest(vh, vw, data)
        if spec.get("sha256") not in (None, digest):
            raise ValueError(f"image at {start}: sha256 mismatch")
        if any(t != image_token_id for t in tokens[start:start + length]):
            raise ValueError(f"image span {start}+{length} does not carry image_token_id")
        out.append((start, length, vh, vw, digest, bytes(data)))
        last_end = start + length
    if off != len(payload_tail):
        raise ValueError("OPEN payload has bytes past the images")
    covered = sum(length for _, length, *_ in out)
    if sum(1 for t in tokens if t == image_token_id) != covered:
        raise ValueError("image_token_id outside a declared image span")
    return out
