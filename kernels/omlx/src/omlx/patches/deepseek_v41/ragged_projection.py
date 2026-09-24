# SPDX-License-Identifier: MIT
"""Share projection weights without changing any request's GEMV reduction."""

import os
import mlx.core as mx
from . import fast_qmv
from .quantization import QuantizedProjection, quantize_activation


def project(projection, values, *, quantized=False):
    if os.environ.get("DS41_BATCH_PROJECTIONS", "1") == "0" or not (
        mx.default_device() == mx.gpu
        and isinstance(projection, QuantizedProjection)
        and fast_qmv._static_ok(projection)[1]
        and all(
            x.ndim == 3
            and x.shape[0] == 1
            and 1 <= x.shape[1] <= 5
            and x.dtype == mx.bfloat16
            for x in values
        )
    ):
        fn = projection.project_quantized if quantized else projection
        return [fn(x) for x in values]
    if projection.quantize_input and not quantized:
        values = [quantize_activation(x) for x in values]
    result = [None] * len(values)
    singles = [i for i, x in enumerate(values) if x.shape[1] == 1]
    wide = [i for i, x in enumerate(values) if x.shape[1] != 1]
    if singles:
        if len(singles) == 1:
            i = singles[0]
            result[i] = projection.project_quantized(values[i])
        else:
            x = mx.concatenate([values[i] for i in singles], 0)
            y = mx.quantized_matmul(
                x,
                projection.weight[None],
                projection.scales[None],
                group_size=32,
                bits=8,
                mode="mxfp8",
            )
            for r, i in enumerate(singles):
                result[i] = y[r : r + 1]
    if wide:
        x = mx.concatenate([values[i] for i in wide], 1)
        outputs = []
        k = x.shape[-1]
        n = projection.weight.shape[0]
        tile = int(os.environ.get("DS41_BATCH_QMV_TILE", "8"))
        rpl = int(os.environ.get("DS41_BATCH_QMV_ROWS", "1"))
        for begin in range(0, x.shape[1], tile):
            part = x[:, begin : begin + tile]
            m = part.shape[1]
            # Always the wide reduction here, including a one-token tail:
            # each source request selected qmv_wide at L=2..5.
            outputs.append(
                fast_qmv._kernel()(
                    inputs=[projection.weight, projection.scales, part],
                    template=[
                        ("T", part.dtype),
                        ("K", k),
                        ("N", n),
                        ("M", m),
                        ("R", rpl),
                    ],
                    grid=(32, (n + 4 * rpl - 1) // (4 * rpl) * 2, 1),
                    threadgroup=(32, 2, 1),
                    output_shapes=[(1, m, n)],
                    output_dtypes=[part.dtype],
                )[0]
            )
        y = mx.concatenate(outputs, 1)
        begin = 0
        for i in wide:
            length = values[i].shape[1]
            result[i] = y[:, begin : begin + length]
            begin += length
    return result
