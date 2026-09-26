"""The cost policy must preserve the served short-context and verify-shape contract."""
import pytest
from omlx.patches.deepseek_v41.mtp import AcceptanceDepthController


def test_disabled_policy_keeps_acceptance_rule(monkeypatch):
    monkeypatch.delenv('DS41_MTP_COST_POLICY', raising=False)
    policy = AcceptanceDepthController(4)
    policy.observe(4, 0, 999)
    assert policy.choose_cost_depth([1.0] * 4, 1048576) == 1


@pytest.mark.parametrize('context', [0, 1000, 1023])
def test_short_context_keeps_served_width(monkeypatch, context):
    monkeypatch.setenv('DS41_MTP_COST_POLICY', '1')
    policy = AcceptanceDepthController(4)
    policy.observe(4, 1, 0)
    assert policy.choose_cost_depth([1.0] * 4, context) == 2


@pytest.mark.parametrize('context', [1024, 8192, 131072, 524288, 1048576])
def test_confidence_extremes_and_exact_shape_bounds(monkeypatch, context):
    monkeypatch.setenv('DS41_MTP_COST_POLICY', '1')
    for depth in range(1, 5):
        policy = AcceptanceDepthController(depth)
        assert policy.choose_cost_depth([1.0] * depth, context) == depth
        assert policy.choose_cost_depth([0.0] * depth, context) == 1
        assert policy.choose_cost_depth([1.0] + [0.0] * (depth - 1), context) == 1
    assert not AcceptanceDepthController(5).cost_policy


def test_runtime_timing_does_not_change_selection(monkeypatch):
    monkeypatch.setenv('DS41_MTP_COST_POLICY', '1')
    first, second = AcceptanceDepthController(4), AcceptanceDepthController(4)
    first.observe(4, 0, 0.001)
    second.observe(4, 3, 1000000)
    ps = [0.95, 0.9, 0.7, 0.1]
    assert first.choose_cost_depth(ps, 131072) == second.choose_cost_depth(ps, 131072)
