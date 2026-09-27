# SPDX-License-Identifier: MIT
"""ds41-ttft: first token before the MTP post-init, prebuilt copy-draft prompt index."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.mlx_lm_mtp import copy_draft


def _state(index):
    return (
        index.n,
        index._buf[: index.n].tolist(),
        len(index._buf),
        index._order.tolist(),
        index._sorted.tolist(),
        index._order.dtype,
        index._sorted.dtype,
    )


def _drive(index, rng, steps=40):
    """Proposals, observations and appends: the policy's full observable behaviour."""
    out = []
    for _ in range(steps):
        out.append(index.propose(rng.randint(0, 6)))
        index.observe(rng.randint(0, 5))
        index.append([rng.randint(0, 3) for _ in range(rng.randint(1, 5))])
    return out


@pytest.mark.parametrize("vocab", [2, 3, 7, 50000])
def test_prompt_index_matches_constructor(vocab):
    rng = random.Random(vocab)
    lengths = [copy_draft.NGRAM, copy_draft.NGRAM + 1, 17, 300, 5000, 70000]
    for n in lengths:
        prompt = [rng.randrange(vocab) for _ in range(n)]
        main = rng.randrange(vocab)
        ref = copy_draft.CopyIndex(prompt + [main])
        pre = copy_draft.PromptIndex(prompt)
        assert pre.matches(prompt + [main], n)
        got = copy_draft.CopyIndex.from_prompt(pre, [main])
        assert got is not None
        assert _state(got) == _state(ref)
        a, b = random.Random(n), random.Random(n)
        assert _drive(got, a) == _drive(ref, b)
        assert _state(got) == _state(ref)
        assert pre.buf is None  # consumed


def test_prompt_index_refuses_short_or_mismatched():
    short = copy_draft.PromptIndex([1, 2])
    assert copy_draft.CopyIndex.from_prompt(short, [3]) is None
    pre = copy_draft.PromptIndex(list(range(1000)))
    assert not pre.matches(list(range(999)), 999)
    other = list(range(1000))
    other[-1] = 7
    assert not pre.matches(other, 1000)
    other = list(range(1000))
    other[0] = 7
    assert not pre.matches(other, 1000)
    assert pre.matches(list(range(1000)) + [5], 1000)
    assert copy_draft.CopyIndex.from_prompt(pre, [1, 2]) is None  # built for one extra token


# ---------------------------------------------------------------------------
# First token before the post-init (tiny qwen3_5 MTP model, real GenerationBatch)
# ---------------------------------------------------------------------------

TINY = {
    "model_type": "qwen3_5", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256, "linear_num_value_heads": 2,
    "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 3, "full_attention_interval": 2, "tie_word_embeddings": True,
    "rms_norm_eps": 1e-5, "head_dim": 32, "rope_theta": 1000.0, "partial_rotary_factor": 0.5,
    "max_position_embeddings": 256, "mtp_num_hidden_layers": 1,
}


@pytest.fixture()
def tiny():
    try:
        from omlx.patches.mlx_lm_mtp import batch_generator, is_mtp_active, qwen35_model, set_mtp_active
    except ImportError:
        pytest.skip("omlx.patches.mlx_lm_mtp not importable")
    if not qwen35_model.apply() or not batch_generator.apply():
        pytest.skip("mlx_lm MTP patches refused to apply")
    previous_device, previous = mx.default_device(), is_mtp_active()
    mx.set_default_device(mx.cpu)
    set_mtp_active(True)
    try:
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        mx.random.seed(3)
        model = TextModel(TextModelArgs.from_dict(TINY))
        mx.eval(model.parameters())
        yield model
    finally:
        set_mtp_active(previous)
        mx.set_default_device(previous_device)


def _generate(model, prompt, *, early, max_tokens=24, stop=None):
    """Tokens per next() call and the backbone forwards before the first emission."""
    from mlx_lm.generate import GenerationBatch, StopSequences
    from mlx_lm.models.cache import make_prompt_cache

    model._omlx_mtp_early_first = early
    model._omlx_mtp_preserve_requests = True
    from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

    cache = make_prompt_cache(model)
    model(mx.array(prompt[:-1])[None], cache=cache)
    mx.eval([c.state for c in cache])
    batched = []
    for c in cache:  # the per-row cache types BatchGenerator hands a GenerationBatch
        if isinstance(c, KVCache):
            c = BatchKVCache.merge([c])
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array([0])
        batched.append(c)
    cache = batched
    calls = []
    original = type(model).__call__

    def counted(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    type(model).__call__ = counted
    try:
        batch = GenerationBatch(
            model, [0], mx.array(prompt[-1:]), cache, [list(prompt[:-1])], None,
            lambda lp: mx.argmax(lp, axis=-1), [[]], [StopSequences(stop)], [max_tokens],
        )
        steps, forwards_before_first, start = [], None, len(calls)
        while batch.uids:
            out = batch.next()
            if forwards_before_first is None and out:
                forwards_before_first = len(calls) - start
            steps.append([(r.token, r.finish_reason) for r in out])
    finally:
        type(model).__call__ = original
        del model._omlx_mtp_early_first
    return steps, forwards_before_first


def _flat(steps):
    return [t for step in steps for t in step]


def test_early_first_token_same_stream(tiny):
    prompt = list(range(5, 40))
    ref, ref_forwards = _generate(tiny, prompt, early=False)
    got, got_forwards = _generate(tiny, prompt, early=True)
    assert _flat(got) == _flat(ref)
    assert got[0] == ref[0]  # one token per call on the singleton path, the same first token
    assert got_forwards == ref_forwards - 1  # emitted before the post-init forward


@pytest.mark.parametrize("max_tokens", [1, 2, 3])
def test_early_first_token_length_limits(tiny, max_tokens):
    prompt = list(range(9, 30))
    ref, _ = _generate(tiny, prompt, early=False, max_tokens=max_tokens)
    got, _ = _generate(tiny, prompt, early=True, max_tokens=max_tokens)
    assert _flat(got) == _flat(ref)
    assert _flat(got)[-1][1] == "length"


def test_early_first_token_stop_on_first(tiny):
    prompt = list(range(9, 30))
    ref, _ = _generate(tiny, prompt, early=False)
    first = ref[0][0][0]
    ref_stop, _ = _generate(tiny, prompt, early=False, stop=[[first]])
    got_stop, _ = _generate(tiny, prompt, early=True, stop=[[first]])
    assert _flat(got_stop) == _flat(ref_stop) == [(first, "stop")]
    second = _flat(ref)[1][0]
    ref_stop, _ = _generate(tiny, prompt, early=False, stop=[[second]])
    got_stop, _ = _generate(tiny, prompt, early=True, stop=[[second]])
    assert _flat(got_stop) == _flat(ref_stop)


def test_standard_step_refused_after_early_emission(tiny):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    batch = SimpleNamespace(uids=[4], _omlx_mtp_early={4: 11})
    with pytest.raises(RuntimeError):
        bg._refuse_standard_after_early(batch)
    bg._refuse_standard_after_early(SimpleNamespace(uids=[5], _omlx_mtp_early={4: 11}))


def test_extend_carries_early_markers():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    host = SimpleNamespace(uids=[1], _omlx_mtp_early={1: 7})
    donor = SimpleNamespace(uids=[2], _omlx_mtp_early={2: 9})

    def extend(batch, other):
        batch.uids += other.uids

    bg._extend_preserving_mtp(extend, host, donor)
    assert host._omlx_mtp_early == {1: 7, 2: 9} and host.uids == [1, 2]
    assert bg._early_emitted(host, 2) and not bg._early_emitted(host, 3)
