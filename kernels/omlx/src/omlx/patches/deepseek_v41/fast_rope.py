# SPDX-License-Identifier: MIT
"""Single-pass partial RoPE: copy the unrotated prefix, rotate the paired tail.

The reference rotates x[..., -d:] in fp32, stacks the pair halves and then
concatenates the prefix back. Stack and concatenate do not fuse, so each call
made several strided copies plus fp32 intermediates, all fresh allocations
under the prefill's per-layer clear_cache (~19.5 ms per [8192, 64, 512] bf16
tensor on M5). This kernel reads every element once and writes one output
(~12 ms fresh, ~0.8 ms from a warm buffer cache), bit-identical to the
reference: same fp32 arithmetic, same cos/sin tables. A transposed input
(the attention output) is read through its strides, absorbing that copy.
"""

import mlx.core as mx

_SOURCE = """
    uint gid = thread_position_in_grid.x;
    constexpr uint PAIRS = D / 2;
    constexpr uint KEEP = (D - R) / 2;
    uint L = meta[0];
    uint H = meta[1];
    uint p = gid % PAIRS;
    uint row = gid / PAIRS;
    uint h = row % H;
    uint bl = row / H;
    uint l = bl % L;
    uint b = bl / L;
    int64_t step = x_strides[NDIM - 1];
    int64_t src = int64_t(b) * x_strides[0] + int64_t(l) * x_strides[1]
        + int64_t(2 * p) * step;
    if (NDIM == 4) {
        src += int64_t(h) * x_strides[2];
    }
    size_t dst = size_t(row) * D + 2 * p;
    if (p < KEEP) {
        out[dst] = x[src];
        out[dst + 1] = x[src + step];
        return;
    }
    uint j = l * (R / 2) + (p - KEEP);
    float c = cos_t[j];
    float s = sin_t[j];
    float a = static_cast<float>(x[src]);
    float b2 = static_cast<float>(x[src + step]);
    float a_c = a * c;
    float b_s = b2 * s;
    float a_s = a * s;
    float b_c = b2 * c;
    out[dst] = static_cast<T>(a_c - b_s);
    out[dst + 1] = static_cast<T>(a_s + b_c);
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="ds41_partial_rope",
            input_names=["x", "cos_t", "sin_t", "meta"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
    return _kernel


def supported(x, positions, rope_dim):
    return positions.ndim == 1 and supported_length(x, positions.shape[0], rope_dim)


def supported_length(x, length, rope_dim):
    return (
        mx.default_device() == mx.gpu
        and x.ndim in (3, 4)
        and length == x.shape[1]
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and 0 < rope_dim <= x.shape[-1]
        and rope_dim % 2 == 0
        and x.shape[-1] % 2 == 0
        and 0 < x.size // 2 < 2**32
    )


def partial_rope(x, cos, sin, rope_dim):
    """x: [B, L, (H,) D], any strides; cos/sin: row-contiguous [L, rope_dim // 2]
    float32 with the angle sign already applied. Returns a row-contiguous copy."""
    heads = x.shape[2] if x.ndim == 4 else 1
    return _get_kernel()(
        inputs=[x, cos, sin, mx.array([x.shape[1], heads], dtype=mx.uint32)],
        template=[("T", x.dtype), ("D", x.shape[-1]), ("R", rope_dim), ("NDIM", x.ndim)],
        grid=(x.size // 2, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
