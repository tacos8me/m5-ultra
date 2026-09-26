"""First-chunk row reuse stays exact across misses, eviction and I/O lanes."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from test_deepseek_v41_engram_io import table
from omlx.patches.deepseek_v41 import storage


@pytest.fixture(autouse=True)
def legacy_cache(monkeypatch):
    monkeypatch.setattr(storage, "HOT_ROWS_BYTES", 0)


def test_warm_rows_hits_misses_and_bound(table, monkeypatch):
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    monkeypatch.setattr(storage, "WARM_ROWS_BYTES", 4096)
    if storage._native_gather() is None:
        pytest.skip("native reader unavailable")
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")
    try:
        first = np.arange(100, dtype=np.int64)
        embed.gather(first, warm_rows=True)
        snapshot = embed._warm_rows.snapshot
        keys, data = snapshot
        assert keys.nbytes + sum(raw.nbytes for raw, _ in data) <= 4096
        assert len(keys) < len(first)
        calls = []
        original = embed._weights.gather_rows

        def read(key, rows, **kwargs):
            calls.extend(rows.tolist())
            return original(key, rows, **kwargs)

        monkeypatch.setattr(embed._weights, "gather_rows", read)
        query = np.array([2, 1, 2, 98, 99, 101], dtype=np.int64)
        result = embed.gather(query)
        assert embed._warm_rows.snapshot is snapshot
        assert 1 not in calls and 2 not in calls and 101 in calls
        for (actual, dtype), (expected, expected_dtype) in zip(
            result, embed._read_rows(query)
        ):
            assert dtype == expected_dtype
            np.testing.assert_array_equal(actual, expected)
        embed.gather(np.arange(300, 400, dtype=np.int64), warm_rows=True)
        assert int(embed._warm_rows.snapshot[0][0]) == 300
        for (actual, _), (expected, _) in zip(
            embed.gather(query), embed._read_rows(query)
        ):
            np.testing.assert_array_equal(actual, expected)
    finally:
        embed.close()
    assert embed._warm_rows.snapshot is None


def test_warm_rows_concurrent_replacement_and_decode_bypass(table, monkeypatch):
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 256)
    monkeypatch.setattr(storage, "WARM_ROWS_BYTES", 4096)
    embed = storage.DiskEngramEmbedding(table, "weight", "scale")

    def run(seed):
        ids = np.random.default_rng(seed).integers(0, 5000, 500)
        result = embed.gather(ids, warm_rows=True)
        for (actual, dtype), (expected, expected_dtype) in zip(
            result, embed._read_rows(ids)
        ):
            assert dtype == expected_dtype
            np.testing.assert_array_equal(actual, expected)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, range(8)))
        previous = embed._warm_rows.snapshot
        embed.gather(np.array([1, 2, 3]), warm_rows=True)
        assert embed._warm_rows.snapshot is previous
        assert "_warm_rows" not in embed and not embed.children()
    finally:
        embed.close()
