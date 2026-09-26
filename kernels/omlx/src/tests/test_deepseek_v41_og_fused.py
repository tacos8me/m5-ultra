"""ds41-og fused verify: grouping policy (og_model.mtp_verify_groups) and og_fused.eligible bounds.

The bitwise per-request identity of the fused pass needs the real weights and box boundaries:
benchmarks/og/batch_bench.py (numerics) and benchmarks/og/batch_identity.py (served, temperature 0).
"""
import pytest

from omlx.patches.deepseek_v41 import og_fused, og_model


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    # Other tests may leave the default device on the CPU; eligible() requires the GPU.
    monkeypatch.setattr(og_fused.mx, "default_device", lambda: og_fused.mx.gpu)


@pytest.fixture
def groups(monkeypatch):
    monkeypatch.setattr(og_fused, "ENABLED", True)
    monkeypatch.setattr(og_fused, "_static_ok", lambda lm: True)
    monkeypatch.setattr(og_model, "FUSE_MIN", 3)
    return lambda lengths: og_model.OgLanguageModel.mtp_verify_groups(object(), lengths)


def test_pairs_from_three_requests(groups):
    assert groups([5, 5]) is None  # c2 keeps per-request verify (box hidden by the interleave)
    assert groups([5, 5, 5]) == [[0, 1], [2]]
    assert groups([5, 4, 3, 2]) == [[0, 1], [2, 3]]


def test_one_row_requests_stay_alone(groups):
    assert groups([1, 5, 5]) == [[0], [1, 2]]
    assert groups([5, 1, 5, 1]) == [[1], [0, 2], [3]]
    assert groups([1, 1, 5]) is None


def test_disabled(groups, monkeypatch):
    monkeypatch.setattr(og_fused, "ENABLED", False)
    assert groups([5, 5, 5, 5]) is None


def test_eligible_bounds(monkeypatch):
    monkeypatch.setattr(og_fused, "_static_ok", lambda lm: True)
    monkeypatch.setattr(og_fused, "ENABLED", True)
    assert og_fused.eligible(None, [5, 5])
    assert not og_fused.eligible(None, [5])
    assert not og_fused.eligible(None, [1, 5])
    assert not og_fused.eligible(None, [5, 5, 5, 5])  # 20 rows > MAX_ROWS
