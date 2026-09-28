"""Compact wo_a byte codes: bitwise equal to the BF16 grouped GEMV and einsum at widths 2-5."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import decode_fusions, woa_compact

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="Metal kernels")


def fp8_like(groups, rows, k, seed):
    """BF16 with the low four mantissa bits clear, plus zeros, far exponents and set low bits (escapes)."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((groups * rows, k)) * 0.02).astype(np.float32)
    bits = (w.view(np.uint32) >> 16).astype(np.uint16) & np.uint16(0xFFF0)
    flat = bits.reshape(-1)
    pick = rng.choice(flat.size, 64, replace=False)
    flat[pick[:16]] = 0
    flat[pick[16:32]] = (flat[pick[16:32]] & 0x807F) | (140 << 7)
    flat[pick[32:48]] = (flat[pick[32:48]] & 0x807F) | (90 << 7)
    flat[pick[48:]] |= 3
    return mx.array(bits).view(mx.bfloat16)


def linear_with(weight):
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = weight
    return linear


@pytest.mark.parametrize("k", [512, 4096])
@pytest.mark.parametrize("length", [2, 3, 4, 5])
def test_compact_matches_bf16_kernel_and_einsum(monkeypatch, k, length):
    monkeypatch.setattr(woa_compact, "ENABLED", True)
    groups, rows = 8, 128
    weight = fp8_like(groups, rows, k, seed=k + length)
    linear = linear_with(weight)
    codes = woa_compact.encode(weight)
    assert codes is not None and codes.escape_rate > 0
    linear.__dict__[woa_compact._ATTR] = codes
    w3 = weight.reshape(groups, rows, k)
    for scale in (1.0, 256.0, 1 / 256):
        x = (mx.random.normal((1, length, groups, k), key=mx.random.key(length)) * scale).astype(mx.bfloat16)
        assert decode_fusions.grouped_gemv_supported(x, w3)
        compact = woa_compact.grouped_gemv(linear, x, w3)
        reference = decode_fusions.grouped_gemv(x, w3)
        einsum = mx.einsum("bsgd,grd->bsgr", x, w3)
        assert mx.array_equal(compact, reference).item()
        assert mx.array_equal(compact, einsum).item()


def test_encode_rejects_escape_heavy_weight():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 1 << 16, (256, 512), dtype=np.uint16) & np.uint16(0x3FFF)
    assert woa_compact.encode(mx.array(bits).view(mx.bfloat16)) is None


def test_install_layers_mtp_aliases_and_fallbacks(monkeypatch):
    monkeypatch.setattr(woa_compact, "ENABLED", True)
    block = SimpleNamespace(attn=SimpleNamespace(wo_a=linear_with(fp8_like(8, 128, 512, 1))))
    stage = SimpleNamespace(attn=SimpleNamespace(wo_a=linear_with(fp8_like(8, 128, 512, 2))))
    fp32 = SimpleNamespace(attn=SimpleNamespace(wo_a=nn.Linear(512, 1024, bias=False)))
    model = SimpleNamespace(layers=[nn.Module(), block, block, fp32], mtp=[stage])
    summary = woa_compact.install(model)
    assert summary["encoded"] == 2 and summary["layers"]["layers.3"] is None
    assert woa_compact.codes_of(block.attn.wo_a) is not None
    assert woa_compact.codes_of(stage.attn.wo_a) is not None
    assert "_ds41_woa_codes" not in dict(block.attn.wo_a.parameters())

    x = mx.random.normal((1, 3, 8, 512)).astype(mx.bfloat16)
    w3 = block.attn.wo_a.weight.reshape(8, 128, 512)
    on = woa_compact.grouped_gemv(block.attn.wo_a, x, w3)
    monkeypatch.setattr(woa_compact, "ENABLED", False)
    off = woa_compact.grouped_gemv(block.attn.wo_a, x, w3)
    assert mx.array_equal(on, off).item()

    # A replaced weight never reads stale codes.
    block.attn.wo_a.weight = fp8_like(8, 128, 512, 9)
    assert woa_compact.codes_of(block.attn.wo_a) is None


def test_disabled_install_is_a_no_op(monkeypatch):
    monkeypatch.setattr(woa_compact, "ENABLED", False)
    block = SimpleNamespace(attn=SimpleNamespace(wo_a=linear_with(fp8_like(8, 128, 512, 4))))
    assert woa_compact.install(SimpleNamespace(layers=[block], mtp=None)) is None
    assert woa_compact.codes_of(block.attn.wo_a) is None
