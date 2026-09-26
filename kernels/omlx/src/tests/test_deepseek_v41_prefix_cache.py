"""Exact CED cache ownership, geometry and eviction (small arrays only)."""

import numpy as np
import mlx.core as mx
from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.prefix_cache import ExactPrefixCache, compact_copy
from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import (
    DSparkContextCache,
    _DSparkPrimeContext,
)


def sample(end=8192):
    c = DeepseekV41Cache(2)
    c.cache = [
        mx.array([end]),
        mx.arange(24).reshape(1, 8, 3),
        mx.zeros((1, 4, 4), mx.uint32),
        mx.zeros((1, 4, 4), mx.uint32),
        mx.zeros((1, 0, 4)),
        mx.zeros((1, 0, 4)),
        mx.array([[1, 2]], mx.int64),
    ]
    stage = DSparkContextCache(128)
    stage.keys = mx.arange(24).reshape(1, 1, 8, 3)
    stage.offset = end
    c._omlx_mtp_prime_ctx = _DSparkPrimeContext([stage], end)
    return [c]


def test_snapshot_and_two_readers_own_python_and_array_state():
    cache = sample()
    tokens = list(range(17000))
    manager = ExactPrefixCache(2**20)
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, cache, 0, 8192)
    a, n, _ = manager.prepare(tokens)
    b, _, _ = manager.prepare(tokens)
    assert n == 8192
    cache[0][1][:] = 99
    cache[0]._omlx_mtp_prime_ctx.caches[0].keys[:] = 88
    a[0][1][:] = 77
    a[0]._omlx_mtp_prime_ctx.caches[0].keys[:] = 66
    a[0]._omlx_mtp_prime_ctx.caches[0].offset = 9
    assert b[0][1][0, 0, 0].item() == 0
    assert b[0]._omlx_mtp_prime_ctx.caches[0].keys[0, 0, 0, 0].item() == 0
    assert b[0]._omlx_mtp_prime_ctx.caches[0].offset == 8192


def test_partial_chunk_and_changed_geometry_never_create_checkpoint():
    manager = ExactPrefixCache(2**20)
    tokens = list(range(17000))
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 256)
    manager.record(plan, sample(), 256, 8192)
    assert not manager.entries and not plan.valid
    assert manager.prepare(tokens, 4096) == (None, 0, None)


def test_longest_match_divergence_and_leave_final_token():
    manager = ExactPrefixCache(2**20)
    tokens = list(range(17000))
    _, _, plan = manager.prepare(tokens)
    manager.record(plan, sample(), 0, 8192)
    manager.record(plan, sample(16384), 8192, 8192)
    assert manager.prepare(tokens)[1] == 16384
    assert manager.prepare(tokens[:8192])[1] == 0
    assert manager.prepare(tokens[:16384])[1] == 8192
    changed = tokens.copy()
    changed[9000] = 123
    assert manager.prepare(changed)[1] == 8192
    changed[100] = 456
    assert manager.prepare(changed)[1] == 0


def test_eviction_hard_budget_and_disabled_cache():
    manager = ExactPrefixCache(100000)
    tokens = list(range(33000))
    _, _, plan = manager.prepare(tokens)
    for start in range(0, 32768, 8192):
        manager.record(plan, sample(start + 8192), start, 8192)
        assert manager.bytes <= manager.max_bytes
    assert manager.evictions > 0
    assert manager.prepare(tokens)[1] == 24576  # 32K snapshot exceeds budget
    manager.clear()
    assert manager.bytes == 0 and not manager.entries
    manager.max_bytes = 0
    assert manager.prepare(tokens) == (None, 0, None)


def test_compact_copy_preserves_raw_float_payloads_and_slice():
    bits = np.array(
        [0, 0x80000000, 0x7FC00001, 0xFFC00002, 0x7F800000, 1, 0xFFFFFFFF],
        dtype=np.uint32,
    )
    copied = compact_copy(mx.array(bits).view(mx.float32))
    assert np.array_equal(np.asarray(copied.view(mx.uint32)), bits)
    x = mx.arange(10000)
    y = compact_copy(x[512:1024])
    mx.eval(y)
    x[:] = 0
    assert y.tolist() == list(range(512, 1024))


def test_model_ownership_and_draft_mode_change(monkeypatch):
    from types import SimpleNamespace
    from omlx.patches.deepseek_v41.prefix_cache import for_model

    monkeypatch.setenv("DS41_PREFIX_CACHE_GIB", "2")
    first = SimpleNamespace(
        _config=SimpleNamespace(ced_prefill=True),
        _omlx_dspark_decode_enabled=True,
        _omlx_mtp_depth=4,
    )
    second = SimpleNamespace(
        _config=SimpleNamespace(ced_prefill=True),
        _omlx_dspark_decode_enabled=True,
        _omlx_mtp_depth=4,
    )
    manager = for_model(first)
    assert for_model(first) is manager
    assert for_model(second) is not manager
    first._omlx_mtp_depth = 3
    assert for_model(first) is not manager
    first._config.ced_prefill = False
    assert for_model(first) is None
    monkeypatch.setenv("DS41_PREFIX_CACHE_GIB", "0")
    assert for_model(second) is None
