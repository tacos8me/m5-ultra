"""Single-pass partial RoPE (bit-exact against the reference)."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import language


ROPE = SimpleNamespace(
    rope_head_dim=64,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    original_seq_len=65536,
    beta_fast=32,
    beta_slow=1,
    rope_factor=16,
)


def _reference(x, positions, compressed, inverse):
    c = ROPE
    params = (
        c.rope_head_dim,
        c.compress_rope_theta if compressed else c.rope_theta,
        c.original_seq_len,
        c.beta_fast,
        c.beta_slow,
        c.rope_factor,
    )
    return language._rope(x, positions, params, compressed, inverse)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_fast_partial_rope_matches_reference(dtype, compressed, inverse):
    mx.random.seed(3)
    start = 122880
    cases = [
        mx.random.normal((1, 37, 8, 512)),
        mx.random.normal((1, 1, 8, 512)),
        mx.random.normal((1, 37, 512)),
        mx.random.normal((1, 37, 4, 128)),
        mx.random.normal((2, 5, 3, 512)),
        mx.random.normal((1, 8, 37, 512)).transpose(0, 2, 1, 3),
    ]
    for x in cases:
        x = x.astype(dtype)
        for positions in (
            mx.arange(start, start + x.shape[1]),
            mx.arange(3, 3 + x.shape[1]) * 2,
        ):
            expected = _reference(x, positions, compressed, inverse)
            actual = language.rope(x, positions, ROPE, compressed, inverse)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            np.testing.assert_array_equal(
                np.array(actual.astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
            )
