"""Request rows retain routing, weighting, and shared-expert arithmetic."""

import mlx.core as mx
import numpy as np
import pytest
from omlx.patches.deepseek_v41.language import MoE
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.quantization import QuantizedProjection
from omlx.patches.deepseek_v41.routed_batch import forward


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_routed_batch_matches_independent_rows(batch):
    mx.random.seed(412)
    moe = MoE(
        ModelConfig(
            dim=128, moe_inter_dim=128, n_routed_experts=4, n_activated_experts=2
        )
    )
    moe.gate.weight = mx.random.normal((4, 128)) * 0.05
    for expert, switched in ((moe.experts, True), (moe.shared_experts, False)):
        bits = 3 if switched else 8
        for name in ("w1", "w3", "w2"):
            shape = (4, 128, 128) if switched else (128, 128)
            weight = (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(weight, bits=bits, group_size=128)
            setattr(
                expert,
                name,
                QuantizedProjection(
                    packed,
                    scales,
                    bits,
                    "affine",
                    biases=biases,
                    group_size=128,
                    quantize_input=True,
                ),
            )
    x = mx.random.normal((batch, 1, 128)).astype(mx.bfloat16)
    expected = mx.concatenate([moe(x[r : r + 1], None) for r in range(batch)], 0)
    actual = forward(moe, x)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
