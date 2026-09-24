"""The compacting radix top-k selects exactly the ids of the six-pass radix top-k."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import index_nax


def _reference(scores, k):
    b, l, n = scores.shape
    ids = index_nax._radix()(
        inputs=[scores, mx.array([k], mx.uint32)],
        grid=(256 * b * l, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, l, k)],
        output_dtypes=[mx.int32],
    )[0]
    return mx.sort(ids, axis=-1)


def _compact(scores, k):
    b, l, n = scores.shape
    ids = index_nax._radix_compact()(
        inputs=[scores, mx.array([k], mx.uint32)],
        grid=(256 * b * l, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, l, k)],
        output_dtypes=[mx.int32],
    )[0]
    return mx.sort(ids, axis=-1)


def _cases(rng, n):
    rows = 24
    x = rng.normal(0, 1, (1, rows, n)).astype(np.float32)
    ties = np.round(x * 4) / 4  # heavy exact ties around the threshold
    relu = np.maximum(x, 0) * rng.uniform(0.5, 2, (1, rows, 1)).astype(np.float32)  # many exact zeros
    masked = x.copy()
    masked[:, :, n // 3 :] = -np.inf  # causal mask
    equal = np.ones_like(x)  # a single giant threshold bin: fallback path
    signed_zero = np.where(rng.uniform(size=x.shape) < 0.5, np.float32(-0.0), np.float32(0.0))
    wide = (x * 1e6).astype(np.float32)
    return {"normal": x, "ties": ties, "relu": relu, "masked": masked, "equal": equal,
            "signed_zero": signed_zero, "wide": wide}


@pytest.mark.parametrize("n", [512, 600, 4097, 65536, 266240])
def test_compact_topk_matches_six_pass(n):
    rng = np.random.default_rng(n)
    k = 512
    for name, values in _cases(rng, n).items():
        scores = mx.array(values)
        np.testing.assert_array_equal(
            np.array(_compact(scores, k)), np.array(_reference(scores, k)), err_msg=name
        )
    assert np.array_equal(np.array(index_nax.topk(scores, k)), np.array(_reference(scores, k)))
