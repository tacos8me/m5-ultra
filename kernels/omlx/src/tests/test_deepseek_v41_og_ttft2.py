# SPDX-License-Identifier: MIT
"""ds41-ttft2: replay schedules (sequential / two-stream overlap / resumable steps), the first-segment store,
admission slices and the reused streaming detokenizer. CPU; fake layers whose outputs depend on the windows
the replay keeps across segments, so a wrong window, order or restore changes the state."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from omlx.patches.deepseek_v41 import encoder_replay as er  # noqa: E402
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache  # noqa: E402
from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache  # noqa: E402

D = 16


class FakeLayer:
    """h' = f(h, window): the window this layer keeps is part of every later row's input (like attention)."""

    def __init__(self, i):
        self.i = i

    def __call__(self, h, pre, cache, shared, start, image_mask, ced_tail=None, prebuilt_end=None, hc_rows=None):
        new = (h[:, :, 0, :8].astype(mx.float32) * (self.i + 1) + start).astype(mx.bfloat16)
        old = cache[1]
        kv = new if old is None or old.shape[1] == 0 else mx.concatenate([old, new], 1)
        cache[1] = kv[:, -128:]
        ctx = mx.mean(kv.astype(mx.float32), axis=1, keepdims=True)[:, :, None, :]
        h = (h.astype(mx.float32) * 0.5 + mx.tile(ctx, (1, 1, 4, D // 8)) + (hc_rows or 0) * 1e-3).astype(mx.bfloat16)
        return h, pre * 0.9 + self.i


class FakeLM:
    _omlx_dspark_cache_owned = True

    def __init__(self, prime=True):
        self._config = SimpleNamespace(n_layers=40, window_size=128, dspark_target_layer_ids=(37, 38, 39),
                                       image_token_id=None, head_dim=512, index_head_dim=128)
        self.layers = [None] * 20 + [FakeLayer(i) for i in range(20, 40)]
        self._omlx_dspark_decode_enabled = prime

    def make_mtp_cache(self):
        return [DSparkContextCache(128) for _ in range(2)]

    def dspark_append_context(self, aux, caches, start_offset=None):
        for k, item in enumerate(caches):
            item.append((aux[:, None, :, :4] * (k + 1)).astype(mx.bfloat16), start_offset=start_offset)


def fresh_cache(n1):
    cache = []
    for i in range(40):
        item = DeepseekV41Cache(1 if i == 20 else 0)
        item.cache = [mx.array([n1], mx.int32)] + [None] * 6
        if i == 20:
            item.cache[2] = mx.zeros((1, n1, 288), mx.uint8)
            item.cache[3] = mx.zeros((1, n1, 68), mx.uint8)
        cache.append(item)
    return cache


def tail(tokens, seed):
    """The box's 256-row tail for tokens: rows are a function of (token prefix, seed)."""
    n1 = len(tokens) - 1
    rows = min(n1, 256)
    first = n1 - rows
    # row p is a function of (seed, p, tokens[p]): a later prompt with the same prefix gets the same rows
    pos = np.arange(first, n1, dtype=np.float32)[None, :, None, None]
    lane = np.arange(4 * D, dtype=np.float32).reshape(1, 1, 4, D)
    tok = np.asarray(tokens[first:n1], np.float32)[None, :, None, None]
    hidden = np.sin(pos * 0.37 + lane * 0.11 + seed) + tok * 1e-3
    pre = np.full((1, rows, 4), 0.5, np.float32)
    return mx.array(hidden).astype(mx.bfloat16), mx.array(pre), first


def run(lm, tokens, seed=0, seg_key=("id", "num"), steps=False, interleave=None):
    hidden, pre, first = tail(tokens, seed)
    cache = fresh_cache(len(tokens) - 1)
    if steps:
        gen = er.replay_steps(lm, cache, hidden, pre, first, tokens, None, seg_key)
        n = 0
        for rows in gen:
            n += 1
            assert rows in (128, (len(tokens) - 1) % 8192 or 8192) or rows > 0
            if interleave is not None:
                interleave()
    else:
        er.replay(lm, cache, hidden, pre, first, tokens, seg_key=seg_key)
    return cache


def state(cache):
    out = []
    for item in cache:
        for x in item.cache:
            out.append(None if x is None else np.array(x.view(mx.uint8) if x.dtype != mx.uint8 else x).tobytes())
    ctx = getattr(cache[0], "_omlx_mtp_prime_ctx", None)
    for item in (ctx.caches if ctx is not None else ()):
        out.append((item.offset, np.array(item.keys.view(mx.uint16)).tobytes()))
    return out


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    er._SEG0.clear()
    for k in er.SEG_STATS:
        er.SEG_STATS[k] = 0
    monkeypatch.setattr(er, "SEG_CACHE", 8)
    monkeypatch.setattr(er, "OVERLAP", True)
    yield


@pytest.mark.parametrize("n", [40, 300, 8193, 8217, 8320, 8321, 8600, 16384 + 50, 16384 + 129])
def test_overlap_and_steps_equal_sequential(monkeypatch, n):
    lm = FakeLM()
    tokens = [(7 * i + 3) % 1000 for i in range(n)]
    monkeypatch.setattr(er, "SEG_CACHE", 0)
    monkeypatch.setattr(er, "OVERLAP", False)
    ref = state(run(lm, tokens))
    monkeypatch.setattr(er, "OVERLAP", True)
    assert state(run(lm, tokens)) == ref
    noise = []
    assert state(run(lm, tokens, steps=True, interleave=lambda: noise.append(mx.eval(mx.ones((64,)) * 2)))) == ref


def test_first_segment_store_hit_equals_recompute(monkeypatch):
    lm = FakeLM()
    base = [(11 * i) % 997 for i in range(8192 + 40)]
    turn2 = base[:8192] + [(5 * i) % 991 for i in range(90)]        # same chunk 0, new short final chunk
    run(lm, base)
    assert er.SEG_STATS["seg0_stores"] == 1 and er.SEG_STATS["seg0_misses"] == 1
    hit = state(run(lm, turn2))
    assert er.SEG_STATS["seg0_hits"] == 1
    monkeypatch.setattr(er, "SEG_CACHE", 0)
    assert hit == state(run(lm, turn2))
    monkeypatch.setattr(er, "OVERLAP", False)
    assert hit == state(run(lm, turn2))


def test_first_segment_store_rejects_changed_inputs(monkeypatch):
    lm = FakeLM()
    base = [(11 * i) % 997 for i in range(8192 + 40)]
    turn2 = base[:8192] + [3] * 60
    run(lm, base, seed=0)
    got = state(run(lm, turn2, seed=1))                       # other tail rows (box arithmetic changed)
    assert er.SEG_STATS["seg0_rejects"] == 1 and er.SEG_STATS["seg0_hits"] == 0
    other = [1] + base[1:8192] + [3] * 60                     # other token prefix
    got2 = state(run(lm, other))
    assert er.SEG_STATS["seg0_hits"] == 0
    got3 = state(run(lm, turn2, seg_key=("id", "other")))     # other numerics key
    assert er.SEG_STATS["seg0_hits"] == 0
    monkeypatch.setattr(er, "SEG_CACHE", 0)
    assert got == state(run(lm, turn2, seed=1))
    assert got2 == state(run(lm, other))
    assert got3 == state(run(lm, turn2))


def test_first_segment_store_without_prime_and_lru(monkeypatch):
    lm = FakeLM(prime=False)
    prompts = [[(k * 13 + i) % 911 for i in range(8192 + 20)] for k in range(4)]
    monkeypatch.setattr(er, "SEG_CACHE", 2)
    for p in prompts:
        run(lm, p)
    assert len(er._SEG0) == 2
    t = prompts[3][:8192] + [9] * 30
    got = state(run(lm, t))
    assert er.SEG_STATS["seg0_hits"] == 1
    monkeypatch.setattr(er, "SEG_CACHE", 0)
    assert got == state(run(lm, t))


def test_single_segment_and_no_key_never_store():
    lm = FakeLM()
    run(lm, list(range(8600)))
    run(lm, list(range(8192 + 30)), seg_key=None)
    assert er.SEG_STATS["seg0_stores"] == 0 and not er._SEG0


# ---------------------------------------------------------------- admission slices (og_model)
def test_admission_slices(monkeypatch):
    from omlx.patches.deepseek_v41 import og_model
    monkeypatch.setattr(og_model, "ADMIT_SLICE_MS", 30.0)
    lm = FakeLM()
    tokens = [(3 * i) % 500 for i in range(8600)]
    hidden, pre, first = tail(tokens, 0)

    def import_state_steps(tensors, manifest, toks, *, identity, base_rows=None, marks=None):
        cache = fresh_cache(len(toks) - 1)
        yield from er.replay_steps(lm, cache, hidden, pre, first, toks, None, None)
        return cache, ("r2", "r3")
    lm.import_state_steps = import_state_steps
    manager = og_model.OgPrefill("127.0.0.1", 1)
    job = SimpleNamespace(done=SimpleNamespace(is_set=lambda: True), error=None, result=({}, {}, 0.0),
                          encoder=SimpleNamespace(identity="x"), base_rows=None, replay=None, imported=None,
                          replay_error=None, replay_start=None, replay_marks=None)
    manager.jobs["r"] = job
    sched = SimpleNamespace(running=[object()], _stream=mx.default_stream(mx.cpu), model=lm)
    request = SimpleNamespace(request_id="r", prompt_token_ids=tokens)
    slices = 0
    while manager.should_defer(sched, request):
        slices += 1
        assert sched._ds41_opened is True
        sched._ds41_opened = False
    # 20 layers x 8.5 ms at 128 rows in 30 ms slices
    assert 4 <= slices <= 8 and job.imported is not None and job.replay is None
    ref = fresh_cache(len(tokens) - 1)
    er.replay(lm, ref, hidden, pre, first, tokens)
    assert state(job.imported[0]) == state(ref)
    # nobody decoding: the rest runs at once
    job2 = SimpleNamespace(**{**job.__dict__, "replay": None, "imported": None})
    manager.jobs["q"] = job2
    idle = SimpleNamespace(running=[], _stream=mx.default_stream(mx.cpu), model=lm)
    assert manager.should_defer(idle, SimpleNamespace(request_id="q", prompt_token_ids=tokens)) is False
    assert job2.imported is None  # prepare() imports it in one piece, as before
    # a failing import is handed to prepare()'s fallback path
    def broken(*a, **k):
        yield 128
        raise ValueError("bad state")
    lm.import_state_steps = broken
    job3 = SimpleNamespace(**{**job.__dict__, "replay": None, "imported": None})
    manager.jobs["e"] = job3
    request3 = SimpleNamespace(request_id="e", prompt_token_ids=tokens)
    while manager.should_defer(sched, request3):
        pass
    assert isinstance(job3.replay_error, ValueError) and job3.imported is None


# ---------------------------------------------------------------- streaming detokenizer reuse
def _tokenizer():
    from pathlib import Path
    path = Path.home() / "models/DeepSeek-V4.1-Flash-pipe1-mlx"
    if not (path / "tokenizer.json").exists():
        pytest.skip("DS41 tokenizer not present")
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(str(path)), path


def test_detokenizer_reuse_same_stream(monkeypatch):
    from omlx.patches.deepseek_v41 import output_parser as op
    tok, path = _tokenizer()
    rng = random.Random(0)
    texts = ["plain ascii text, with punctuation!", "多字节字符和 emoji 🙂🙃 mixed", "  leading spaces\n\ttabs",
             "<｜DSML｜function_calls>x</｜DSML｜function_calls>", "ünïcödé " * 20]
    streams = [tok.encode(t, add_special_tokens=False) for t in texts]
    streams += [[rng.randrange(len(tok)) for _ in range(rng.randrange(1, 80))] for _ in range(40)]

    def outputs(reuse):
        monkeypatch.setattr(op, "REUSE", reuse)
        op._TEMPLATES.clear()
        out = []
        sessions = [op.DeepSeekV41OutputParserSession(tok, path) for _ in streams]
        for s, ids in zip(sessions, streams):  # interleaved sessions share the template's token map only
            det = s._detokenizer
            parts = []
            for t in ids:
                det.add_token(t)
                parts.append(det.last_segment)
            det.finalize()
            parts.append(det.last_segment)
            out.append((parts, det.text, list(det.tokens)))
        return out, sessions

    fresh, _ = outputs(False)
    reused, sessions = outputs(True)
    assert fresh == reused
    assert len(op._TEMPLATES) == 1
    maps = {id(s._detokenizer.tokenmap) for s in sessions}
    assert len(maps) == 1 and len({id(s._detokenizer) for s in sessions}) == len(sessions)
