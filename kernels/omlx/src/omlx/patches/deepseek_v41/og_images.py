# SPDX-License-Identifier: MIT
"""ds41-og image input: the processor's patches go to the box, which runs the ViT.

An image occupies a span of the prompt where every token id is ``image_token_id``
(the reference layout [IMAGE_START] + ([IMAGE] * nw + [NEWLINE]) * nh + [IMAGE_END]).
The box receives the bf16 patches of each image (OPEN ``images``), runs the vision
tower and aligner, and merges the rows at those positions; routing uses ``bias_vl``
there on both halves (the Mac's tail replay passes the image mask).

Cache keys: the token ids alone cannot tell two images apart, so the prefix keys
(box snapshots, Mac row store, ``delta_from`` digests) are uint64: the token id at
text positions and (1 << 63) | 63 bits of sha256(image digest || offset) inside an
image span. Same algorithm and test vector as the box (split_nv/imagekeys.py).

omlx computes a request's ``vlm_inputs_embeds`` through ``get_input_embeddings``
before scheduling; og needs no Mac-side embeddings (there is no local prefill), so
it returns a tiny placeholder and registers the image payload under it. The
admission hook takes the payload back from the request's placeholder.
"""

import hashlib
import math
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

IMAGE_BIT = 1 << 63
DOWNSAMPLE = 3
PATCH_VALUES = 3 * 14 * 14


@dataclass(frozen=True)
class Image:
    start: int
    length: int
    vit_h: int
    vit_w: int
    data: bytes  # bf16 patches [vit_h * vit_w, 3, 14, 14], row-major
    digest: str


def num_image_tokens(vit_h, vit_w):
    nh, nw = math.ceil(vit_h / DOWNSAMPLE), math.ceil(vit_w / DOWNSAMPLE)
    return nh * (nw + 1) + 2


def image_digest(vit_h, vit_w, data):
    h = hashlib.sha256(struct.pack('<II', vit_h, vit_w))
    h.update(data)
    return h.hexdigest()


def span_keys(digest, length):
    d = bytes.fromhex(digest)
    out = np.empty(length, dtype=np.uint64)
    for k in range(length):
        v = int.from_bytes(hashlib.sha256(d + struct.pack('<I', k)).digest()[:8], 'little')
        out[k] = (v & (IMAGE_BIT - 1)) | IMAGE_BIT
    return out


def prompt_keys(tokens, images):
    """uint64 prefix keys of `tokens`; images: iterable of Image (spans past the end are cut)."""
    keys = np.asarray(tokens, dtype=np.uint64).copy()
    for im in images or ():
        if im.start >= len(keys):
            continue
        k = span_keys(im.digest, im.length)[: len(keys) - im.start]
        keys[im.start:im.start + len(k)] = k
    return keys


def keys_digest(keys, n):
    return hashlib.sha256(np.asarray(keys[:n], dtype='<u8').tobytes()).hexdigest()


def build(input_ids, pixel_values, image_token_id, grids, spans, types):
    """Validate the processor's image layout against the prompt and return [Image]."""
    import mlx.core as mx

    ids = np.asarray(input_ids).reshape(-1)
    if not (len(grids) == len(spans) == len(types)):
        raise ValueError('Image metadata counts differ')
    images, offset = [], 0
    for (vit_h, vit_w), (start, length), kinds in zip(grids, spans, types):
        vit_h, vit_w, start, length = int(vit_h), int(vit_w), int(start), int(length)
        nh, nw = math.ceil(vit_h / DOWNSAMPLE), math.ceil(vit_w / DOWNSAMPLE)
        if list(kinds) != [0] + ([1] * nw + [2]) * nh + [3] or length != num_image_tokens(vit_h, vit_w):
            raise ValueError('Image layout does not match its patch grid')
        if not np.all(ids[start:start + length] == image_token_id) or start + length > ids.size:
            raise ValueError('Image span does not cover image token ids')
        count = vit_h * vit_w
        patches = pixel_values[offset:offset + count]
        offset += count
        if patches.shape[0] != count or patches.size != count * PATCH_VALUES:
            raise ValueError('Image patches do not match their grid')
        data = np.array(patches.astype(mx.bfloat16).reshape(count, PATCH_VALUES).view(mx.uint16)).tobytes()
        images.append(Image(start, length, vit_h, vit_w, data, image_digest(vit_h, vit_w, data)))
    if offset != pixel_values.shape[0]:
        raise ValueError('Unused image patches')
    if int(np.count_nonzero(ids == image_token_id)) != sum(im.length for im in images):
        raise ValueError('image_token_id outside an image span')
    return images


def wire(images):
    """OPEN header specs and payload tail."""
    specs = [{'start': im.start, 'length': im.length, 'grid': [im.vit_h, im.vit_w], 'sha256': im.digest} for im in images]
    return specs, b''.join(im.data for im in images)


class Registry:
    """Image payloads by the identity of the placeholder embeddings omlx stores on the request."""

    def __init__(self, ttl_s=900):
        self.items, self.lock, self.ttl = {}, threading.Lock(), ttl_s

    def put(self, placeholder, images):
        now = time.monotonic()
        with self.lock:
            for key in [k for k, (_, _, t) in self.items.items() if now - t > self.ttl]:
                del self.items[key]
            self.items[id(placeholder)] = (placeholder, images, now)

    def take(self, placeholder):
        with self.lock:
            item = self.items.pop(id(placeholder), None)
        if item is None or item[0] is not placeholder:
            return None
        return item[1]


REGISTRY = Registry()
