# SPDX-License-Identifier: MIT
"""Incremental prompt tokenization for DeepSeek V4.1 (exactly equal to a full encode).

A Hugging Face fast tokenizer first cuts its input at every added-token match
(leftmost-longest, non-overlapping) and then normalizes, pre-tokenizes and
BPE-encodes each remaining piece on its own. Cutting the prompt at occurrences of
a *delimiter* (an added token that no other added token can overlap, contain or
extend, and that cannot overlap itself) therefore splits it at places where the
full encode also has a hard boundary, so

    encode(prompt) == concat(encode(piece) for piece in pieces)

bit for bit. The chat template puts a delimiter (<｜User｜>, <｜Assistant｜>,
</think>, EOS, ...) around every message, so turn N+1 of a conversation re-uses
the ids of every piece turn N already tokenized and encodes only the new text.

The safety conditions are checked against the tokenizer's full added vocabulary
when the encoder is built; a delimiter that fails them is not used. The class
also refuses to install (and the caller keeps the plain encode) when the
tokenizer's normalizer is not the identity, when added tokens strip whitespace,
when its post-processor adds tokens, or when a self-check on sample text
disagrees with the full encode.

Cache: piece text -> token ids (array('I')), LRU bounded by DS41_FE_ENCODE_CACHE_MB
(default 256, counting 4 bytes/char of key + 4 bytes/id). Pieces shorter than
MIN_PIECE characters are encoded but not cached. Misses are encoded in one
encode_batch call (parallel in the Rust tokenizer).
"""

from array import array
from collections import OrderedDict
import logging
import os
import re
import threading
import time

from . import fe_trace

logger = logging.getLogger(__name__)

# Template tokens that bracket messages (the renderer emits these); others are ignored.
DEFAULT_DELIMITERS = (
    '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>', '<｜System｜>',
    '<｜latest_reminder｜>', '<think>', '</think>',
)
ACTIVE = [True]  # runtime switch (A/B runs through og_server's trace-mode /og/fe endpoint)
MIN_PIECE = int(os.environ.get('DS41_FE_ENCODE_MIN_PIECE', '64'))
ENCODERS = []  # installed encoders (for /og/stats)
STATS = dict(calls=0, chars=0, pieces=0, hits=0, hit_chars=0, misses=0, miss_chars=0, encode_s=0.0, fallbacks=0)


def safe_delimiters(added, candidates):
    """The candidates whose every occurrence the full encode also matches as that exact token.

    Leftmost-longest, non-overlapping matching over the added vocabulary matches
    every occurrence of D as D (and nothing straddles it) when D cannot overlap
    itself and, for every other added token T: T does not contain D (no longer
    match covering D), no proper suffix of T is a proper prefix of D (no match
    straddling D's start) and no proper suffix of D is a proper prefix of T (no
    match straddling D's end). A T lying strictly inside D is harmless: the scan
    reaches D's start first and takes the longest match there, which is D.
    """
    added = {a for a in added if a}
    safe = []
    for d in candidates:
        n = len(d)
        if d not in added or any(d[i:] == d[:n - i] for i in range(1, n)):
            continue
        ok = True
        for t in added:
            if t == d:
                continue
            m = len(t)
            if (d in t or any(m - i < n and d.startswith(t[i:]) for i in range(1, m))
                    or any(m > n - i and t.startswith(d[i:]) for i in range(1, n))):
                ok = False
                break
        if ok:
            safe.append(d)
    return safe


def private_backend(backend):
    """A copy of a Rust tokenizer that nothing else configures.

    Hugging Face __call__ (e.g. mlx_vlm's text path, padding=True) leaves padding and
    truncation enabled on the tokenizer's own backend, and encode_batch would then pad
    every piece to the longest one. Misses are therefore encoded on a private clone with
    both off.
    """
    from tokenizers import Tokenizer

    clone = Tokenizer.from_str(backend.to_str())
    clone.no_padding()
    clone.no_truncation()
    return clone


class SegmentEncoder:
    """encode(text) == the fast tokenizer's encode(text, add_special_tokens=False), incrementally."""

    def __init__(self, backend, delimiters=DEFAULT_DELIMITERS, max_bytes=None):
        backend = private_backend(backend)
        self.backend = backend  # tokenizers.Tokenizer (private: no padding/truncation)
        tokens = list(backend.get_added_tokens_decoder().values())
        if any(t.lstrip or t.rstrip or t.single_word for t in tokens):
            raise ValueError('added tokens with lstrip/rstrip/single_word')
        added = [t.content for t in tokens]
        self.delimiters = safe_delimiters(added, delimiters)
        if not self.delimiters:
            raise ValueError('no safe delimiter tokens')
        self.ids = {d: backend.token_to_id(d) for d in self.delimiters}
        if any(v is None for v in self.ids.values()):
            raise ValueError('delimiter without a token id')
        self.pattern = re.compile('|'.join(re.escape(d) for d in sorted(self.delimiters, key=len, reverse=True)))
        mb = float(os.environ.get('DS41_FE_ENCODE_CACHE_MB', '256')) if max_bytes is None else max_bytes / 2**20
        self.max_bytes = int(mb * 2**20)
        self.cache = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def __deepcopy__(self, memo):
        return self  # one shared piece cache for every copy of the tokenizer (ids do not depend on the copy)

    def __copy__(self):
        return self

    @staticmethod
    def _plain(backend, texts):
        if len(texts) == 1:
            return [backend.encode(texts[0], add_special_tokens=False).ids]
        return [e.ids for e in backend.encode_batch(texts, add_special_tokens=False)]

    def encode(self, text, backend=None):
        """Token ids of text (a list); misses are encoded on `backend` (a private_backend clone)."""
        return self.encode_array(text, backend).tolist()

    def encode_array(self, text, backend=None):
        """Token ids of text as array('I')."""
        t0 = time.perf_counter()
        pieces, pos = [], 0  # str (text piece) or int (delimiter id)
        for m in self.pattern.finditer(text):
            if m.start() > pos:
                pieces.append(text[pos:m.start()])
            pieces.append(self.ids[m.group()])
            pos = m.end()
        if pos < len(text):
            pieces.append(text[pos:])
        found, missing = {}, []
        with self.lock:
            for piece in pieces:
                if isinstance(piece, int) or piece in found:
                    continue
                ids = self.cache.get(piece) if len(piece) >= MIN_PIECE else None
                if ids is None:
                    found[piece] = None
                    missing.append(piece)
                else:
                    self.cache.move_to_end(piece)
                    found[piece] = ids
                    STATS['hits'] += 1
                    STATS['hit_chars'] += len(piece)
        if missing:
            for piece, ids in zip(missing, self._plain(backend or self.backend, missing)):
                found[piece] = array('I', ids)
            STATS['misses'] += len(missing)
            STATS['miss_chars'] += sum(map(len, missing))
            with self.lock:
                for piece in missing:
                    if len(piece) >= MIN_PIECE:
                        self._store(piece, found[piece])
        out = array('I')
        for piece in pieces:
            if isinstance(piece, int):
                out.append(piece)
            else:
                out.extend(found[piece])
        STATS['calls'] += 1
        STATS['chars'] += len(text)
        STATS['pieces'] += len(pieces)
        spent = time.perf_counter() - t0
        STATS['encode_s'] += spent
        fe_trace.add('encode_s', spent)
        return out

    def _store(self, piece, ids):
        if piece in self.cache:
            self.cache.move_to_end(piece)
            return
        size = 4 * len(piece) + 4 * len(ids) + 200
        if size > self.max_bytes:
            return
        while self.cache and self.bytes + size > self.max_bytes:
            old, old_ids = self.cache.popitem(last=False)
            self.bytes -= 4 * len(old) + 4 * len(old_ids) + 200
        self.cache[piece] = ids
        self.bytes += size

    def summary(self):
        with self.lock:
            return dict(STATS, entries=len(self.cache), bytes=self.bytes, max_bytes=self.max_bytes,
                        delimiters=len(self.delimiters))


# Canonically equivalent and compatibility forms, full-width letters and odd whitespace: an identity normalizer keeps them.
PROBE = 'A \u00c5 A\u030a e\u0301 \u00e9 \ufb01 \uff21 \u3000 \u00a0 \t x'
SAMPLE = ('<｜begin▁of▁sentence｜><｜System｜>sys  \n<｜User｜>hi 123456 中文<｜Assistant｜></think>ok '
          '<｜end▁of▁sentence｜><｜User｜>  <think>x</think>\n\n<｜Assistant｜><think>')


class Encode:
    """tokenizer.encode replacement bound to one tokenizer instance.

    omlx deep-copies the tokenizer so the event loop and the MLX thread each own a Rust
    backend ("Already borrowed" otherwise); each Encode owns a private backend clone
    (private_backend), a deep copy binds to the new tokenizer (found in the deepcopy memo)
    with a clone of its own, and all of them share the piece cache.
    """

    def __init__(self, tokenizer, encoder):
        self.tokenizer, self.encoder = tokenizer, encoder
        self._backend = None  # created on first use (a deep copy is bound before its tokenizer is filled in)

    @property
    def backend(self):
        if self._backend is None:
            self._backend = private_backend(self.tokenizer._tokenizer)
        return self._backend

    def __deepcopy__(self, memo):
        return Encode(memo.get(id(self.tokenizer), self.tokenizer), self.encoder)

    def __call__(self, text, *args, add_special_tokens=True, **kwargs):
        tokenizer = self.tokenizer
        if isinstance(text, str) and not args and not kwargs and ACTIVE[0]:
            return self.encoder.encode(text, self.backend)
        STATS['fallbacks'] += 1
        return type(tokenizer).encode(tokenizer, text, *args, add_special_tokens=add_special_tokens, **kwargs)


def install_prepare_inputs(processor_type):
    """Text-only mlx_vlm.prepare_inputs for `processor_type` through the incremental encoder.

    For a prompt without images, audio or video, mlx_vlm tokenizes with the tokenizer's
    __call__ (a full Rust encode plus a Python flatten into an int32 [1, N] array: 0.55 s
    at 512K tokens), not with tokenizer.encode. This returns the same dict, the same
    int32 [1, N] input_ids (encode(prompt, add_special_tokens=False): one prompt, so
    padding is a no-op) and the same all-ones int32 mask, from the piece cache.
    Anything else goes to the original.
    """
    import mlx.core as mx
    import mlx_vlm.utils as vlm_utils
    import numpy as np

    original = vlm_utils.prepare_inputs
    if getattr(original, '_ds41_fast', False):
        return

    def prepare_inputs(processor, *args, **kwargs):
        prompts = kwargs.get('prompts')
        tokenizer = getattr(processor, 'tokenizer', None)
        encode = getattr(tokenizer, '__dict__', {}).get('encode')
        if (not isinstance(encode, Encode) or args or not ACTIVE[0] or not isinstance(processor, processor_type)
                or set(kwargs) - {'images', 'audio', 'videos', 'prompts'}
                or any(kwargs.get(k) is not None and (not hasattr(kwargs[k], '__len__') or len(kwargs[k]))
                       for k in ('images', 'audio', 'videos'))
                or not isinstance(prompts, list) or len(prompts) != 1 or not isinstance(prompts[0], str)):
            return original(processor, *args, **kwargs)
        if tokenizer.pad_token is None:  # as the original (padding=True)
            tokenizer.pad_token = tokenizer.eos_token
        ids = np.frombuffer(encode.encoder.encode_array(prompts[0], encode.backend), np.uint32).astype(np.int32)
        return {'input_ids': mx.array(ids[None]), 'attention_mask': mx.ones((1, ids.size), mx.int32)}

    prepare_inputs._ds41_fast = True
    prepare_inputs.__wrapped__ = original
    vlm_utils.prepare_inputs = prepare_inputs


def install(tokenizer):
    """Route tokenizer.encode(str) through a SegmentEncoder when provably equivalent; returns it or None."""
    if os.environ.get('DS41_FE_ENCODE_CACHE', '1') != '1':
        return None
    backend = getattr(tokenizer, '_tokenizer', None)
    if backend is None or getattr(tokenizer, '_ds41_segment_encoder', None) is not None:
        return getattr(tokenizer, '_ds41_segment_encoder', None)
    try:
        if backend.normalizer is not None and backend.normalizer.normalize_str(PROBE) != PROBE:
            raise ValueError('tokenizer normalizer is not the identity')
        encoder = SegmentEncoder(backend)
        plain = tokenizer.encode
        for text in (SAMPLE, SAMPLE * 3, 'x', ''):
            full = plain(text, add_special_tokens=False)
            if encoder.encode(text) != full:
                raise ValueError('self-check mismatch')
            special = plain(text, add_special_tokens=True)
            if special != full:
                raise ValueError('add_special_tokens changes the ids')
    except Exception as exc:  # noqa: BLE001 -- keep the plain encode
        logger.warning('ds41 incremental tokenization disabled: %s', exc)
        return None
    encoder.cache.clear()
    encoder.bytes = 0
    tokenizer.encode = Encode(tokenizer, encoder)
    tokenizer._ds41_segment_encoder = encoder
    ENCODERS.append(encoder)
    logger.info('ds41 incremental tokenization on: %d delimiters, cache %d MB', len(encoder.delimiters),
                encoder.max_bytes >> 20)
    return encoder
