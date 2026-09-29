"""Depth controller C(L) tables: pipeline vs fused-pair regime, env overrides."""
import pytest

from omlx.patches.deepseek_v41 import pipe_session
from omlx.patches.deepseek_v41.pipe_session import PipelineDepthController, _pipeline_costs


@pytest.fixture(autouse=True)
def _cost_policy(monkeypatch):
    monkeypatch.setenv("DS41_MTP_COST_POLICY", "1")


def _best(probs, costs):
    cumulative, expected, best, best_u = 1.0, 1.0, 1, -1.0
    for i, p in enumerate(probs[:4]):
        cumulative *= p
        expected += cumulative
        if expected / costs[i] > best_u:
            best, best_u = i + 1, expected / costs[i]
    return best


def test_tables_are_sorted_and_positive():
    for table in (pipe_session._PIPE_COSTS, pipe_session._FUSED_COSTS):
        floors = [f for f, _ in table]
        assert floors == sorted(floors, reverse=True) and floors[-1] == 0
        assert all(len(c) == 4 and all(x > 0 for x in c) and list(c) == sorted(c) for _, c in table)


def test_regime_selects_table():
    probs = [0.9, 0.8, 0.7, 0.6]
    pipe = PipelineDepthController(4)
    fused_on = PipelineDepthController(4, fused=lambda: True)
    fused_off = PipelineDepthController(4, fused=lambda: False)
    c1 = dict(pipe_session._PIPE_COSTS)[0]
    c4 = dict(pipe_session._FUSED_COSTS)[0]
    assert pipe.choose_cost_depth(probs, 8192) == _best(probs, c1)
    assert fused_off.choose_cost_depth(probs, 8192) == _best(probs, c1)
    assert fused_on.choose_cost_depth(probs, 8192) == _best(probs, c4)
    # The steeper fused curve verifies fewer rows for this draft.
    assert _best(probs, c4) < _best(probs, c1)


def test_short_context_keeps_acceptance_rule():
    ctl = PipelineDepthController(4, fused=lambda: True)
    ctl.cur = 3
    assert ctl.choose_cost_depth([0.1, 0.1, 0.1, 0.1], 512) == 3


def test_env_override(monkeypatch):
    monkeypatch.setenv("DS41_PIPE_COSTS", "0:10,20,30,40;131072:11,21,31,41")
    assert _pipeline_costs() == ((131072, (11.0, 21.0, 31.0, 41.0)), (0, (10.0, 20.0, 30.0, 40.0)))
    monkeypatch.setenv("DS41_PIPE_COSTS_FUSED", "0:1,2,3,4")
    assert _pipeline_costs("DS41_PIPE_COSTS_FUSED", ()) == ((0, (1.0, 2.0, 3.0, 4.0)),)
    monkeypatch.setenv("DS41_PIPE_COSTS_FUSED", "0:1,2,3")
    with pytest.raises(ValueError):
        _pipeline_costs("DS41_PIPE_COSTS_FUSED", ())
