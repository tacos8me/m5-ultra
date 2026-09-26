"""Decode at depth: ping-pong KV growth, single-tile decode selection and multi-row index scoring
are storage/scheduling changes only; every value must match the original paths bitwise."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import growth, kernels
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.quantization import pack_activation, quantize_activation


def _rows(n, width=68, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, 256, (1, n, width), dtype=np.uint8))


@pytest.fixture
def growth_on(monkeypatch):
    monkeypatch.setattr(growth, "ENABLED", True)
    monkeypatch.setattr(growth, "PINGPONG", True)


def test_pingpong_append_truncate_extract_match_concatenation(growth_on):
    rng = np.random.default_rng(1)
    cache = DeepseekV41Cache(2)
    cache[0] = mx.array([0], mx.int32)
    cache[2] = _rows(0)
    reference = np.zeros((1, 0, 68), np.uint8)
    buffers = set()
    step = 0
    # A prefill-sized append, then decode/verify appends with DSpark-style rollbacks.
    for length in [5000] + [int(x) for x in rng.integers(1, 6, 300)]:
        start = reference.shape[1]
        values = _rows(length, seed=step + 10)
        row = cache.extract(0)
        assert row[2] is cache[2]
        view = growth.append(row, 2, row[2][:, :start], values, start)
        row[2] = view
        cache.cache = row.cache
        reference = np.concatenate([reference, np.array(values)], 1)
        cache[0] = mx.array([reference.shape[1] * 2], mx.int32)
        np.testing.assert_array_equal(np.array(cache[2]), reference)
        if length <= growth.PINGPONG_ROWS:
            buffers.add(id(cache._ds41_buffers[2]["buffer"]))
        if rng.uniform() < 0.5:
            keep = start + int(rng.integers(1, length + 1))
            growth.truncate(cache, 2, keep)
            reference = reference[:, :keep]
            cache[0] = mx.array([keep * 2], mx.int32)
            np.testing.assert_array_equal(np.array(cache[2]), reference)
        step += 1
    # After warm-up the chain alternates between two buffers (plus reseeds on growth).
    assert len(buffers) < 12


def test_replaced_slot_reseeds_without_touching_old_views(growth_on):
    cache = DeepseekV41Cache(1)
    cache[0] = mx.array([0], mx.int32)
    cache[3] = _rows(0)
    first = growth.append(cache, 3, cache[3], _rows(10, seed=1), 0)
    cache[3] = first
    second = growth.append(cache, 3, cache[3], _rows(3, seed=2), 10)
    cache[3] = second
    kept = np.array(second)
    # An external replacement (prefix restore) must not reuse the chain buffers in place.
    cache[3] = mx.array(np.array(second)[:, :8])
    third = growth.append(cache, 3, cache[3], _rows(2, seed=3), 8)
    np.testing.assert_array_equal(np.array(second), kept)
    np.testing.assert_array_equal(
        np.array(third), np.concatenate([kept[:, :8], np.array(_rows(2, seed=3))], 1)
    )


def test_batch_one_extract_keeps_arrays_and_registry():
    cache = DeepseekV41Cache(2)
    cache.cache = [mx.array([6], mx.int32), _rows(6, 528), _rows(3, 288), _rows(3), _rows(0, 512), _rows(0, 512), mx.zeros((1, 3), mx.int64)]
    row = cache.extract(0)
    assert all(a is b for a, b in zip(row.cache[1:], cache.cache[1:]))
    assert row._ds41_buffers is cache._ds41_buffers
    row = cache.extract(0, offset=6)
    assert all(a is b for a, b in zip(row.cache[1:], cache.cache[1:]))
    assert int(row[0].item()) == 6


def _keys(n, seed, extreme=False):
    mx.random.seed(seed)
    x = mx.random.normal((1, n, 128)).astype(mx.bfloat16) * (1 + 3 * mx.random.uniform(shape=(1, n, 1)))
    keys = np.array(pack_activation(x, bits=4))
    if extreme:
        rng = np.random.default_rng(seed)
        # Scale bytes at the ldexp fallback edges (e <= 2, e = 255) and near them.
        pick = rng.integers(0, n, n // 4)
        keys[0, pick, 64 + rng.integers(0, 4, pick.size)] = rng.choice([1, 2, 3, 4, 250, 253, 254, 255], pick.size)
    return mx.array(keys)


@pytest.mark.parametrize("n,ratio", [(1, 2), (31, 1), (33, 2), (4000, 2), (8195, 1), (70001, 1)])
@pytest.mark.parametrize("length", [2, 3, 5, 8])
@pytest.mark.parametrize("extreme", [False, True])
def test_multi_row_index_scores_match_mma_bitwise(monkeypatch, n, ratio, length, extreme):
    keys = _keys(n, n + length, extreme)
    mx.random.seed(length)
    q = quantize_activation(mx.random.normal((1, length, 32, 128)).astype(mx.bfloat16), bits=4)
    w = mx.random.normal((1, length, 32)).astype(mx.float32)
    for start in {max(0, n * ratio - length + 1), max(0, n * ratio // 2), n * ratio + 3}:
        monkeypatch.setattr(kernels, "DS41_INDEX_ROWS", False)
        ref = np.array(kernels.packed_index_scores(q, keys, w, start, ratio))
        monkeypatch.setattr(kernels, "DS41_INDEX_ROWS", True)
        out = np.array(kernels.packed_index_scores(q, keys, w, start, ratio))
        np.testing.assert_array_equal(out.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("length", [1, 2, 5])
@pytest.mark.parametrize("blocks", [0, 64])
def test_single_tile_decode_selection_matches_chunked_scan(length, blocks):
    n, ratio = 20000, 1
    keys = _keys(n, 5)
    mx.random.seed(9)
    q = quantize_activation(mx.random.normal((1, length, 32, 128)).astype(mx.bfloat16), bits=4)
    w = mx.random.normal((1, length, 32)).astype(mx.float32)
    start = n * ratio - length + 1
    single = kernels.packed_index_topk(q, keys, w, start, ratio, 512, block_count=blocks, block_size=8)
    chunked = kernels.packed_index_topk(q, keys, w, start, ratio, 512, block_count=blocks, block_size=8, chunk_size=4096)
    for a, b in zip(single, chunked):
        np.testing.assert_array_equal(np.array(a), np.array(b))


def _row_cache(n, seed):
    cache = DeepseekV41Cache(2)
    cache.cache = [
        mx.array([2 * n], mx.int32),
        _rows(min(2 * n, 128), 528, seed),
        _rows(n, 288, seed + 1),
        _rows(n, 68, seed + 2),
        None,
        None,
        mx.array([[seed, seed + 1, seed + 2]], mx.int64),
    ]
    return cache


def test_batched_rows_keep_their_storage_and_match_padded_concatenation():
    a, b, c = _row_cache(5, 1), _row_cache(9, 11), _row_cache(7, 21)
    batch = DeepseekV41Cache.merge([a, b])
    assert batch.batch_size == 2
    for row, src in enumerate((a, b)):
        out = batch.extract(row, offset=int(src[0].item()))
        assert all(x is y for x, y in zip(out.cache[1:], src.cache[1:]))
        assert out._ds41_buffers is src._registry()
    # The lazily batched arrays still hold the padded concatenation.
    for slot in (1, 2, 3):
        ref = np.zeros((2, 9 if slot > 1 else 18, a[slot].shape[2]), np.uint8)
        ref[0, : a[slot].shape[1]] = np.array(a[slot])[0]
        ref[1, : b[slot].shape[1]] = np.array(b[slot])[0]
        np.testing.assert_array_equal(np.array(batch[slot]), ref)
    batch.extend(c)
    out = batch.extract(2, offset=14)
    assert out[2] is c[2]
    batch.filter([1])
    assert batch.batch_size == 1 and batch[3] is b[3]
    assert batch.extract(0)._ds41_buffers is b._registry()


def test_replaced_batched_slot_falls_back_to_slices():
    a, b = _row_cache(5, 1), _row_cache(9, 11)
    batch = DeepseekV41Cache.merge([a, b])
    batch[2] = mx.array(np.array(batch[2]))
    out = batch.extract(1, offset=18)
    assert out[2] is not b[2]
    np.testing.assert_array_equal(np.array(out[2]), np.array(b[2]))


def _tile_selection(scores, k):
    values, ids = kernels._tile_topk(scores, k)
    return mx.sort(mx.where(values > -float("inf"), ids, -1), axis=-1)


@pytest.mark.parametrize("n", [100, 512, 700, 20000, 131072 + 17, 524288])
@pytest.mark.parametrize("length", [1, 2, 5])
def test_radix_decode_selection_matches_tile_sort(n, length):
    from omlx.patches.deepseek_v41 import decode_topk

    rng = np.random.default_rng(n + length)
    x = rng.normal(0, 1, (1, length, n)).astype(np.float32)
    cases = {
        "relu": np.maximum(x, 0) * rng.uniform(0.5, 2, (1, length, 1)).astype(np.float32),
        "normal": x,
        "ties": np.round(x * 4) / 4,
        "masked": np.where(np.arange(n)[None, None, :] >= n - 3 + np.arange(length)[None, :, None], -np.inf, x),
        "signed_zero": np.where(rng.uniform(size=x.shape) < 0.5, np.float32(-0.0), np.float32(0.0)) + (x > 2.5) * x,
        "equal": np.ones_like(x),
        "wide": x * 1e6,
    }
    for name, values in cases.items():
        scores = mx.array(values.astype(np.float32))
        np.testing.assert_array_equal(
            np.array(decode_topk.selection(scores, 512)), np.array(_tile_selection(scores, 512)), err_msg=name
        )
