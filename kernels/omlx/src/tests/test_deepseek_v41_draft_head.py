# SPDX-License-Identifier: MIT
"""DS41_DRAFT_HEAD routing (CPU, fake head kernels): which head serves which draft block.

The real kernels' bitwise properties (fast_qmv rows == mx.quantized_matmul mxfp8 at M<=5, M-invariance
at M=1..8 on the 129280x5120 head) are checked on the GPU by benchmarks/og/head_mxfp8_bench.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.deepseek_v41 import dspark, og_fused  # noqa: E402
from omlx.patches.mlx_lm_mtp import batch_generator as bg  # noqa: E402

N, K = 16, 96


@pytest.fixture(autouse=True)
def cpu():
    # Tiny fake arrays only: run them on the CPU whatever the suite's default device is.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _per_row(x, n, offset):
    """Per-row fake head: row q -> sum(x[q]) * (j + offset) for vocab j (independent of the row count)."""
    s = x.astype(mx.float32).sum(-1, keepdims=True)
    return s * (mx.arange(n, dtype=mx.float32) + offset)


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def rows(p, x):  # og_fused._rows: (1, m, K) -> (1, m, N) in x.dtype
        assert x.ndim == 3 and x.shape[0] == 1 and x.dtype == mx.bfloat16
        calls.append(("mxfp8", x.shape[1]))
        return _per_row(x, p.weight.shape[0], 1).astype(x.dtype)

    def head_rows(weight, x):  # og_fused._head_rows (BF16 rows kernel)
        calls.append(("bf16_rows", x.shape[1]))
        return _per_row(x, weight.shape[0], 3)

    def project(x, head):  # head.project_logits (BF16 single path)
        calls.append(("bf16", x.size // x.shape[-1]))
        return _per_row(x, head.weight.shape[0], 3)

    q = SimpleNamespace(weight=mx.zeros((N, K // 4), mx.uint32), scales=mx.zeros((N, K // 32), mx.uint8))
    monkeypatch.setattr(og_fused, "_rows", rows)
    monkeypatch.setattr(og_fused, "_head_rows", head_rows)
    monkeypatch.setattr(dspark, "project_logits", project)
    monkeypatch.setattr(dspark, "_draft_head", lambda model: q)
    monkeypatch.setattr(dspark.mx, "default_device", lambda: dspark.mx.gpu)  # the head kernels need the GPU
    monkeypatch.setattr(dspark, "DRAFT_HEAD", "mxfp8")
    stats = {}
    dspark.bind(stats)
    model = SimpleNamespace(head=SimpleNamespace(weight=mx.zeros((N, K), mx.bfloat16)))
    return model, calls, stats


def _x(seed, width):
    return mx.random.normal((1, width, K), key=mx.random.key(seed)).astype(mx.bfloat16)


def test_default_is_bf16_even_on_the_cost_policy_path(fake, monkeypatch):
    model, calls, stats = fake
    monkeypatch.setattr(dspark, "DRAFT_HEAD", "bf16")
    out = dspark.draft_logits(model, _x(0, 4), cost_policy=True)
    assert calls == [("bf16", 4)] and out.dtype == mx.float32
    assert stats["draft_head_bf16_rows"] == 4 and stats["draft_head_mxfp8_rows"] == 0
    assert dspark.install_draft_head(model) is None


def test_mxfp8_only_on_the_cost_policy_path(fake):
    model, calls, stats = fake
    dspark.draft_logits(model, _x(0, 3), cost_policy=False)  # below 1024: acceptance-only, BF16
    out = dspark.draft_logits(model, _x(1, 4), cost_policy=True)
    assert calls == [("bf16", 3), ("mxfp8", 4)]
    assert out.shape == (1, 4, N) and out.dtype == mx.float32
    assert stats["draft_head_bf16_rows"] == 3 and stats["draft_head_mxfp8_rows"] == 4


def test_mxfp8_needs_a_bf16_head_and_the_gpu(fake, monkeypatch):
    model, calls, _ = fake
    quantized = SimpleNamespace(weight=model.head.weight, bits=4)
    assert not dspark.use_mxfp8(SimpleNamespace(head=quantized), True)
    monkeypatch.setattr(dspark.mx, "default_device", lambda: dspark.mx.cpu)
    assert not dspark.use_mxfp8(model, True)


@pytest.mark.parametrize("policy", [[True, True], [False, False], [True, False], [False, True]])
def test_batched_head_is_each_requests_single_head(fake, policy):
    """proposal_forward_batch's head gives every request exactly its proposal_forward logits."""
    model, calls, _ = fake
    xs = [_x(10 + k, 4) for k in range(len(policy))]
    batched = dspark.batch_logits(model, xs, 4, policy)
    launches = list(calls)
    calls.clear()
    single = [dspark.draft_logits(model, x, p) for x, p in zip(xs, policy)]
    for b, s in zip(batched, single):
        assert b.dtype == s.dtype == mx.float32
        assert np.array_equal(np.array(b), np.array(s))
    # one launch per head over all its requests' rows
    assert sorted(launches) == sorted(
        [("mxfp8", 4 * sum(policy))] * any(policy) + [("bf16_rows", 4 * (len(policy) - sum(policy)))] * (not all(policy)))


def test_batch_supported_no_longer_rejects_mxfp8(monkeypatch):
    monkeypatch.setattr(dspark, "DRAFT_HEAD", "mxfp8")
    monkeypatch.setattr(dspark, "DRAFT_BATCH", True)
    monkeypatch.setattr(dspark, "_stages_ok", lambda model: True)
    monkeypatch.setattr(dspark.mx, "default_device", lambda: dspark.mx.gpu)
    assert dspark.batch_supported(None, [4, 4])
    assert not dspark.batch_supported(None, [4, 4, 4])  # 12 rows > 8


def test_generator_passes_cost_policy_only_to_hosts_that_take_it(monkeypatch):
    class Host:
        _omlx_dspark_cost_head = True

        def __init__(self):
            self.seen = []

        def dspark_forward_batch(self, hiddens, anchors, caches, widths, cost_policy=None):
            self.seen.append(("batch", list(widths), cost_policy))
            return [mx.zeros((1, w, N)) for w in widths]

        def dspark_forward(self, hidden, anchor, cache, *, draft_length=None, cost_policy=False):
            self.seen.append(("single", draft_length, cost_policy))
            return mx.zeros((1, draft_length, N)), None

    class Legacy(Host):  # e.g. the V4 host: no cost_policy keyword
        _omlx_dspark_cost_head = False

        def dspark_forward_batch(self, hiddens, anchors, caches, widths):
            self.seen.append(("batch", list(widths)))
            return None

        def dspark_forward(self, hidden, anchor, cache, *, draft_length=None):
            self.seen.append(("single", draft_length))
            return mx.zeros((1, draft_length, N)), None

    def state():
        return SimpleNamespace(head_clone=True, controller=None, stats=SimpleNamespace(mtp_head_ms=0.0),
                               mtp_cache=[], hist_offset=0)

    for cls in (Host, Legacy):
        host = cls()
        policies = [True, False]
        plans = iter([(host, 4, mx.zeros((1, 1)), p) for p in policies])
        monkeypatch.setattr(bg, "_dspark_prepare", lambda *a: next(plans))
        monkeypatch.setattr(bg, "_dspark_finish", lambda *a: None)
        jobs = [(None, state(), None, mx.zeros((2,)), None) for _ in policies]
        bg.dspark_draft_jobs(jobs, [None, None])
        if cls is Host:
            assert host.seen == [("batch", [4, 4], [True, False])]
        else:  # batch refused -> one call per request, no unknown keyword
            assert host.seen == [("batch", [4, 4]), ("single", 4), ("single", 4)]
    assert bg._dspark_head_kwargs(object(), True) == {}
