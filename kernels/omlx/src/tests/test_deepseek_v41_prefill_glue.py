"""Chunk-shared rope tables and one-layer-in-flight prefill are bit-exact."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v41 import language

ROPE = SimpleNamespace(
    rope_head_dim=64,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    original_seq_len=65536,
    beta_fast=32,
    beta_slow=1,
    rope_factor=16,
)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("start,step", [(0, 1), (122880, 1), (61440, 2), (7, 4)])
def test_rope_range_matches_rope(compressed, inverse, start, step):
    mx.random.seed(11)
    language._ROPE_TABLES.clear()
    for x in (
        mx.random.normal((1, 37, 8, 512)).astype(mx.bfloat16),
        mx.random.normal((1, 37, 512)).astype(mx.bfloat16),
        mx.random.normal((1, 8, 37, 512)).astype(mx.bfloat16).transpose(0, 2, 1, 3),
    ):
        positions = mx.arange(start, start + 37) * step
        expected = language.rope(x, positions, ROPE, compressed, inverse)
        for _ in range(2):  # built, then served from the chunk cache
            actual = language.rope_range(x, start, 37, ROPE, compressed, inverse, step=step)
            np.testing.assert_array_equal(
                np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
            )
    assert len(language._ROPE_TABLES) == 1


def _state(cache):
    return [np.array(v) for item in cache for _, v in tree_flatten(item.state) if isinstance(v, mx.array)]


def test_prefill_pipeline_is_bitwise(monkeypatch):
    from test_deepseek_v41 import load_reference_weights, tiny

    model = language.LanguageModel(tiny())
    load_reference_weights(model)
    ids = mx.array(np.random.default_rng(3).integers(0, 64, (1, 300)))
    results = []
    for pipeline in (False, True):
        monkeypatch.setattr(language, "DS41_PREFILL_PIPELINE", pipeline)
        cache = model.make_cache()
        logits = model(ids, cache=cache)
        results.append((np.array(logits), _state(cache)))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    for a, b in zip(results[0][1], results[1][1]):
        np.testing.assert_array_equal(a, b)
