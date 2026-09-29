"""ffn_fuse / attn_in stay bitwise the unfused DS41_MHC decode path (synthetic weights, checkpoint dims)."""
import types

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v41 import attn_in, ffn_fuse, hc_fuse, language, og_fused
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import MoE, RMSNorm
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, quantize_activation

pytestmark = [
    pytest.mark.skipif(mx.default_device() != mx.gpu, reason="Metal kernels"),
    pytest.mark.skipif(not fast.has_symbol("deepseek_v41_grouped_expert"), reason="native extension not built"),
]

D, I, E = 5120, 2304, 8


@pytest.fixture(autouse=True)
def _mhc(monkeypatch):
    # The served DS41_MHC decode kernels (router, mHC) are the reference path.
    monkeypatch.setattr(language, "DS41_MHC", True)
_rng = np.random.default_rng(41)


def _fp8(n, k):
    b = _rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    b = np.where((b & 0x7F) >= 0x78, b & 0xB7, b).astype(np.uint8)  # finite E4M3 codes
    return mx.array(b.view(np.uint32))


def _e8m0(*shape):
    return mx.array(_rng.integers(118, 126, size=shape, dtype=np.uint8))


def _fp4(*shape):
    return mx.array(_rng.integers(0, 2**32, size=shape, dtype=np.uint32))


@pytest.fixture(scope="module")
def block():
    mx.random.seed(41)
    c = ModelConfig(dim=D, moe_inter_dim=I, n_routed_experts=384, n_activated_experts=6)
    c.score_func, c.norm_topk_prob, c.route_scale, c.swiglu_limit = "sqrtsoftplus", True, 1.5, 10.0
    moe = MoE(c)
    # Routed weights for 8 experts; the gate bias keeps every top-6 among them.
    moe.experts.w1 = QuantizedProjection(_fp4(E, I, D // 8), _e8m0(E, I, D // 32), 4, "mxfp4")
    moe.experts.w3 = QuantizedProjection(_fp4(E, I, D // 8), _e8m0(E, I, D // 32), 4, "mxfp4")
    moe.experts.w2 = QuantizedProjection(_fp4(E, D, I // 8), _e8m0(E, D, I // 32), 4, "mxfp4")
    moe.shared_experts.w1 = QuantizedProjection(_fp8(I, D), _e8m0(I, D // 32), 8, "mxfp8")
    moe.shared_experts.w3 = QuantizedProjection(_fp8(I, D), _e8m0(I, D // 32), 8, "mxfp8")
    moe.shared_experts.w2 = QuantizedProjection(_fp8(D, I), _e8m0(D, I // 32), 8, "mxfp8")
    moe.gate.weight = (mx.random.normal((384, D)) * 0.02).astype(mx.bfloat16)
    moe.gate.bias = mx.concatenate([mx.full((E,), 50.0), mx.zeros((384 - E,))]).astype(mx.float32)
    norm = RMSNorm(D, 1e-20)
    norm.weight = (mx.random.normal((D,)) * 0.1 + 1).astype(mx.bfloat16)
    blk = types.SimpleNamespace(
        _config=c, ffn=moe, ffn_norm=norm,
        hc_ffn_fn=(mx.random.normal((24, 4 * D)) * 0.01).astype(mx.float32),
        hc_ffn_base=(mx.random.normal((24,)) * 0.1).astype(mx.float32),
        hc_ffn_scale=mx.array([0.5, 0.7, 0.9], mx.float32))
    mx.eval(moe.parameters(), norm.weight, blk.hc_ffn_fn, blk.hc_ffn_base)
    return blk


def _reference(blk, h, pre, bounds):
    c = blk._config
    mix, x = hc_fuse.project_pre_norm(h, pre, blk.hc_ffn_fn, blk.ffn_norm.weight, c.norm_eps, blk.ffn_norm.eps)
    f = blk.ffn(x, None) if bounds is None else og_fused._moe(blk.ffn, x, bounds)
    return hc_fuse.post_mix(f, h, mix, blk.hc_ffn_scale, blk.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters)


@pytest.mark.parametrize("rows,bounds", [(1, None), (2, None), (3, None), (5, None),
                                         (4, [(0, 2), (2, 4)]), (10, [(0, 5), (5, 10)]),
                                         (14, [(0, 4), (4, 9), (9, 12), (12, 14)])])
@pytest.mark.parametrize("scale", [1.0, 25.0])
def test_ffn_forward_bitwise(block, rows, bounds, scale):
    h = (mx.random.normal((1, rows, 4, D)) * scale).astype(mx.bfloat16)
    pre = mx.random.uniform(shape=(1, rows, 4)).astype(mx.float32)
    assert ffn_fuse.eligible(block, h, None, max_rows=5 if bounds is None else og_fused.MAX_ROWS)
    for _ in range(3):  # chained: later steps see realistic post-mHC inputs
        ref = _reference(block, h, pre, bounds)
        new = ffn_fuse.ffn_forward(block, h, pre, bounds)
        mx.eval(ref, new)
        for a, b in zip(ref, new):
            np.testing.assert_array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))
        h, pre = ref


def test_singleton_rows_above_five_stay_unfused(block):
    h = mx.zeros((1, 6, 4, D), mx.bfloat16)
    assert not ffn_fuse.eligible(block, h, None)


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8, 16])
def test_attn_input_bitwise(rows):
    wq_a = QuantizedProjection(_fp8(1280, D), _e8m0(1280, D // 32), 8, "mxfp8")
    wkv = QuantizedProjection(_fp8(512, D), _e8m0(512, D // 32), 8, "mxfp8")
    h = (mx.random.normal((1, rows, 4, D)) * 0.7).astype(mx.bfloat16)
    pre = mx.random.uniform(shape=(1, rows, 4)).astype(mx.float32)
    fn = (mx.random.normal((24, 4 * D)) * 0.01).astype(mx.float32)
    nw = (mx.random.normal((D,)) * 0.1 + 1).astype(mx.bfloat16)
    mix_a, x = hc_fuse.project_pre_norm(h, pre, fn, nw, 1e-6, 1e-20)
    mix_b, y, yq = attn_in.project_pre_norm_q(h, pre, fn, nw, 1e-6, 1e-20)
    xq = quantize_activation(x)
    if rows <= 5:  # the singleton path's own kernels (MLX quantized_matmul at 2 rows)
        ref = (wq_a.project_quantized(xq), wkv.project_quantized(xq))
    else:  # og_fused
        ref = (og_fused._rows(wq_a, xq), og_fused._rows(wkv, xq))
    new = attn_in.input_projections(yq, wq_a, wkv)
    mx.eval(mix_a, x, xq, mix_b, y, yq, ref, new)
    for a, b in [(mix_a, mix_b), (x, y), (xq, yq), *zip(ref, new)]:
        np.testing.assert_array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


@pytest.mark.parametrize("rows", [1, 2, 3, 5])
@pytest.mark.parametrize("compressed,start", [(False, 100), (True, 8191), (True, 524287)])
def test_wqb_rope_bitwise(monkeypatch, rows, compressed, start):
    monkeypatch.setattr(language, "DS41_FAST_ROPE", True)
    c = ModelConfig(dim=D, n_heads=4, head_dim=512, rope_head_dim=64, q_lora_rank=1280)
    wq_b = QuantizedProjection(_fp8(4 * 512, 1280), _e8m0(4 * 512, 1280 // 32), 8, "mxfp8")
    qr8 = quantize_activation((mx.random.normal((1, rows, 1280)) * 3).astype(mx.bfloat16))
    d = language.rope_params(c, compressed)[0]
    assert attn_in.wqb_rope_supported(wq_b, qr8, 4, 512, d)
    ref = language.rope_range(wq_b.project_quantized(qr8).reshape(1, rows, 4, 512), start, rows, c, compressed)
    new = attn_in.wqb_rope(qr8, wq_b, *language.rope_tables(start, rows, c, compressed), 4, 512)
    mx.eval(ref, new)
    np.testing.assert_array_equal(np.array(ref.astype(mx.float32)), np.array(new.astype(mx.float32)))
