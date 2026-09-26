"""Fused MXFP4 routed-expert kernels stay bitwise the gather_qmm path for 1-16 rows."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v41 import moe_decode
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Expert
from omlx.patches.deepseek_v41.quantization import QuantizedProjection

pytestmark = pytest.mark.skipif(
    not fast.has_symbol("deepseek_v41_grouped_expert"),
    reason="Grouped expert extension is not built",
)


def make_expert(experts=8, dim=512, inter=2304):
    mx.random.seed(20)
    expert = Expert(ModelConfig(dim=dim, moe_inter_dim=inter, n_routed_experts=experts), True)
    for name, shape in (("w1", (inter, dim)), ("w3", (inter, dim)), ("w2", (dim, inter))):
        w, s = mx.quantize(mx.random.normal((experts, *shape)).astype(mx.bfloat16) * 0.05,
                           group_size=32, bits=4, mode="mxfp4")
        setattr(expert, name, QuantizedProjection(w, s, 4, "mxfp4"))
    return expert


@pytest.mark.parametrize("rows", [1, 5, 6, 8, 10, 16])
def test_fused_rows_match_gather(monkeypatch, rows):
    expert = make_expert()
    rng = np.random.default_rng(rows)
    # Rows share experts (the verify/batch case): 6 of 8 experts per row.
    ids = mx.array(np.stack([rng.choice(8, 6, replace=False) for _ in range(rows)])[None].astype(np.uint32))
    weights = mx.random.uniform(shape=(1, rows, 6))
    x = mx.random.normal((1, rows, 1, 1, 512)).astype(mx.bfloat16)
    calls = []
    gate_up = moe_decode.gate_up
    monkeypatch.setattr(moe_decode, "gate_up", lambda *a: calls.append(1) or gate_up(*a))
    fused = expert(x, ids, weights, max_grouped_tokens=16)
    mx.eval(fused)
    assert calls
    monkeypatch.setattr(moe_decode, "MAX_ROWS", 0)
    gathered = expert(x, ids, weights, max_grouped_tokens=16)
    mx.eval(gathered)
    assert len(calls) == 1
    np.testing.assert_array_equal(fused.astype(mx.float32), gathered.astype(mx.float32))
