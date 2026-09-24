"""Batch lifecycle tests; full-checkpoint HTTP greedy gates live in benchmarks/ds41/concurrency."""

import copy
import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import tiny, load_reference_weights
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_layer_major_ragged_decode_and_filter(monkeypatch, batch):
    model = LanguageModel(tiny(max_batch_size=batch))
    load_reference_weights(model)
    caches = []
    for row in range(batch):
        cache = model.make_cache()
        mx.eval(model(mx.array([[3 + row] * (3 + row)]), cache=cache))
        caches.append(cache)
    merged = [DeepseekV41Cache.merge(parts) for parts in zip(*caches)]
    reference = copy.deepcopy(merged)
    # Cross multiple compressor boundaries with independent row histories.
    for step in range(7):
        ids = mx.array([[12 + row + step] for row in range(batch)])
        monkeypatch.setenv("DS41_BATCH_DECODE", "0")
        expected = model(ids, cache=reference)
        mx.eval(expected)
        monkeypatch.setenv("DS41_BATCH_DECODE", "1")
        actual = model(ids, cache=merged)
        mx.eval(actual)
        np.testing.assert_array_equal(mx.argmax(actual, -1), mx.argmax(expected, -1))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        for a, b in zip(merged, reference):
            np.testing.assert_array_equal(a.offset, b.offset)
            for slot, (av, bv) in enumerate(zip(a.cache[1:], b.cache[1:]), 1):
                assert av.shape == bv.shape
                if slot == 6:
                    np.testing.assert_array_equal(av, bv)
                elif av.dtype != mx.uint8:
                    np.testing.assert_allclose(av, bv, atol=1e-5, rtol=1e-5)
                # Packed FP8 bytes can straddle a rounding boundary in this
                # tiny FP32 fixture. Continuation logits/greedy IDs above
                # check their effect; real BF16 gates use the full checkpoint.
    # A completion removes rows; a one-row remainder must use normal decode.
    for a, b in zip(merged, reference):
        a.filter(mx.array([batch - 1]))
        b.filter(mx.array([batch - 1]))
    monkeypatch.setenv("DS41_BATCH_DECODE", "0")
    expected = model(mx.array([[30]]), cache=reference)
    monkeypatch.setenv("DS41_BATCH_DECODE", "1")
    actual = model(mx.array([[30]]), cache=merged)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_explicit_extract_offset_preserves_layout():
    cache = DeepseekV41Cache(2)
    cache.cache = [
        mx.array([5, 9]),
        mx.zeros((2, 8, 4)),
        mx.zeros((2, 4, 4)),
        mx.zeros((2, 4, 4)),
        mx.zeros((2, 1, 4)),
        mx.zeros((2, 1, 4)),
        mx.array([[1, 2], [3, 4]]),
    ]
    for row, offset in enumerate([5, 9]):
        for a, b in zip(
            cache.extract(row).cache, cache.extract(row, offset=offset).cache
        ):
            np.testing.assert_array_equal(a, b)
