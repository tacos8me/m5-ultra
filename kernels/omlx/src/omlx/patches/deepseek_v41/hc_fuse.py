# SPDX-License-Identifier: MIT
"""Fused mHC chains for short decode/verify rows (bitwise the DS41_MHC kernels).

A decode Block runs eight small hc kernels: per sublayer hc_project, the
Sinkhorn mix, hc_pre_norm and hc_post. Two kernels replace them, four per
Block instead of eight:

* ``project_pre_norm``: the 24 hc_project threadgroups of a row plus one
  threadgroup running hc_pre_norm on the same input (independent work, one
  launch).
* ``post_mix``: hc_post, each threadgroup first rebuilding its row's Sinkhorn
  mix from the raw projection in simdgroup 0; also writes the mix's ``pre``
  (the next pre_norm's input) instead of a separate mix launch.

Every per-element operation, its order and its fp-contraction/reassociation
mode is copied from decode_fusions (_project_pipelined, _mix_parallel) and
hyper_connection (pre_norm, post), so outputs are bit-identical. DS41_HC_FUSE=0
restores the unfused kernels.
"""
import os
from functools import cache

import mlx.core as mx

from . import decode_fusions

DS41_HC_FUSE = os.environ.get("DS41_HC_FUSE", "1") == "1"
# Submit the first Block's attention as soon as it is built, so the GPU starts
# while Python still builds that Block's MoE (scheduling only).
EARLY_SUBMIT = os.environ.get("DS41_HC_EARLY_SUBMIT", "1") == "1"

_HEADER = r"""
// decode_fusions._mix_parallel for one row, run by all 32 lanes of simdgroup 0
// (lanes 16-31 repeat lanes 0-15 as the second row of the original layout).
// Pointer parameters are templates: MLX passes small inputs in the constant address space.
template <int ITERS, typename PM, typename PS, typename PB>
inline void ds41_mix_row(PM mix, PS scale, PB base, float eps, uint r, uint lane, threadgroup float* tpre, threadgroup float* tpost,
                         threadgroup float* tcomb) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    uint first = lane & ~15u;
    uint t = lane % 16, i = t / 4, j = t % 4;
    if (lane < 4) {
        float p = mix[r*24+t]*scale[0]+base[t];
        float o = mix[r*24+4+t]*scale[1]+base[4+t];
        tpre[t] = 1.0f/(1.0f+exp(-p))+eps;
        tpost[t] = 2.0f/(1.0f+exp(-o));
    }
    float a = mix[r*24+8+t]*scale[2]+base[8+t];
    float m = -INFINITY;
    for (uint jj=0;jj<4;jj++) m = max(m, simd_shuffle(a, first+i*4+jj));
    a = exp(a-m);
    float sum = 0;
    for (uint jj=0;jj<4;jj++) sum += simd_shuffle(a, first+i*4+jj);
    a = a/sum+eps;
    for (uint it=0;it<ITERS;it++) {
        if (it>0) {
            float rs = 0;
            for (uint jj=0;jj<4;jj++) rs = simd_shuffle(a, first+i*4+jj)+rs;
            rs += eps;
            a /= rs;
        }
        float cs = 0;
        for (uint ii=0;ii<4;ii++) cs = simd_shuffle(a, first+ii*4+j)+cs;
        cs += eps;
        a /= cs;
    }
    if (lane < 16) tcomb[t] = a;
}

// hyper_connection pre_norm for one row (256 threads).
template <typename T, uint D, typename PX, typename PP, typename PW>
inline void ds41_pre_norm_row(PX x, PP pre, PW weight, float eps,
                              uint row, uint tid, uint lane, uint simd, threadgroup float* sums, device T* y) {
    #pragma clang fp contract(off)
    float values[(D + 255) / 256];
    float total = 0.0f;
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        float value = 0.0f;
        if (d < D) {
            for (uint i = 0; i < 4; ++i)
                value = float(x[(row * 4 + i) * D + d]) * pre[row * 4 + i] + value;
            value = float(T(value));
        }
        values[t] = value;
        total = total + value * value;
    }
    total = simd_sum(total);
    if (lane == 0) sums[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float sum = lane < 8 ? sums[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) sums[0] = rsqrt(sum / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        if (d < D) y[row * D + d] = T((values[t] * sums[0]) * float(weight[d]));
    }
}

// hyper_connection post for one element z of x.
template <typename T, uint D, typename PX, typename PR>
inline void ds41_post_elem(PX x, PR residual, threadgroup const float* post,
                           threadgroup const float* comb, uint z, device T* y) {
    #pragma clang fp contract(off)
    const uint row = z / D, d = z % D;
    float values[4];
    for (uint i = 0; i < 4; ++i)
        values[i] = float(residual[(row * 4 + i) * D + d]);
    const float value = float(x[z]);
    for (uint j = 0; j < 4; ++j) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i)
            sum = fma(comb[i * 4 + j], values[i], sum);
        y[(row * 4 + j) * D + d] = T(post[j] * value + sum);
    }
}
"""

_PROJECT_PRE_NORM = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint o = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = t % 32, sg = t / 32;
    threadgroup float dot[8], sq[8], sums[8];
    if (o == 24) {
        ds41_pre_norm_row<T, D>(x, pre, weight, eps[1], row, t, lane, sg, sums, y);
        return;
    }
    // decode_fusions._project_pipelined (DH = 4 * D)
    constexpr uint STEPS = (DH + 255) / 256;
    constexpr uint U = 8;
    float a = 0, b = 0;
    for (uint s0 = 0; s0 < STEPS; s0 += U) {
        float xv[U], wv[U];
        for (uint u = 0; u < U; ++u) {
            uint k = t + 256 * (s0 + u);
            bool ok = s0 + u < STEPS && k < DH;
            xv[u] = ok ? float(x[row*DH+k]) : 0.0f;
            wv[u] = ok ? float(fn[o*DH+k]) : 0.0f;
        }
        for (uint u = 0; u < U; ++u) {
            if (s0 + u < STEPS && t + 256 * (s0 + u) < DH) {
                a = fma(xv[u], wv[u], a);
                b = fma(xv[u], xv[u], b);
            }
        }
    }
    a = simd_sum(a); b = simd_sum(b);
    if (lane==0) {dot[sg]=a; sq[sg]=b;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0) {
        float v = simd_sum(lane<8 ? dot[lane] : 0.0f);
        float s = simd_sum(lane<8 ? sq[lane] : 0.0f);
        if (lane==0) mix[row*24+o] = v * rsqrt(s/float(DH)+eps[0]);
    }
"""

_POST_MIX = r"""
    const uint z = thread_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = (z - thread_position_in_threadgroup.x) / D;
    threadgroup float tpre[4], tpost[4], tcomb[16];
    if (sg == 0)
        ds41_mix_row<ITERS>(mix, scale, base, eps[0], row, lane, tpre, tpost, tcomb);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ds41_post_elem<T, D>(x, residual, tpost, tcomb, z, y);
    const uint d = z % D;
    if (d < 4) pre_out[row * 4 + d] = tpre[d];
"""


@cache
def _project_pre_norm():
    return mx.fast.metal_kernel(
        name="ds41_hc_project_pre_norm", input_names=["x", "pre", "fn", "weight", "eps"],
        output_names=["mix", "y"], header=_HEADER, source=_PROJECT_PRE_NORM)


@cache
def _post_mix():
    return mx.fast.metal_kernel(
        name="ds41_hc_post_mix", input_names=["x", "residual", "mix", "scale", "base", "eps"],
        output_names=["y", "pre_out"], header=_HEADER, source=_POST_MIX)


@cache
def _consts(*values):
    value = mx.array(values, mx.float32)
    mx.eval(value)
    return value


def project_pre_norm(h, pre, fn, weight, proj_eps, norm_eps):
    """(hc_project(h, fn, proj_eps), hc_pre_norm(h, pre, weight, norm_eps)) in one launch."""
    rows, d = h.size // (4 * h.shape[-1]), h.shape[-1]
    return _project_pre_norm()(
        inputs=[h, pre, fn, weight, _consts(float(proj_eps), float(norm_eps))],
        template=[("T", h.dtype), ("D", d), ("DH", 4 * d)],
        grid=(rows * 256, 25, 1), threadgroup=(256, 1, 1),
        output_shapes=[(*h.shape[:-2], 24), (*h.shape[:-2], d)],
        output_dtypes=[mx.float32, h.dtype])


def post_mix(x, residual, mix, scale, base, hc_eps, iters):
    """(hc_post(x, residual, post, comb), pre) with (pre, post, comb) = mix_sinkhorn(mix, ...)."""
    return _post_mix()(
        inputs=[x, residual, mix, scale, base, _consts(float(hc_eps))],
        template=[("T", x.dtype), ("D", x.shape[-1]), ("ITERS", max(1, iters))],
        grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[residual.shape, (*residual.shape[:-1],)],
        output_dtypes=[x.dtype, mx.float32])


def eligible(block, h, pre, verify_tile):
    """The DS41_MHC short-row path the fused kernels reproduce (language.hc_mixes/hc_pre_norm/hc_post)."""
    c = block._config
    return (
        DS41_HC_FUSE
        and decode_fusions.DS41_DECODE_KERNELS_V2
        and h.ndim == 4
        and h.shape[0] == 1
        and 1 <= h.shape[1] <= 8
        and not verify_tile
        and h.dtype == mx.bfloat16
        and h.shape[-2] == c.hc_mult == 4
        and h.shape[-1] % 256 == 0
        and pre.dtype == mx.float32
        and pre.shape == h.shape[:-1]
        and block.hc_attn_fn.dtype == mx.float32
        and block.hc_attn_fn.shape == (24, 4 * h.shape[-1])
        and block.hc_ffn_fn.dtype == mx.float32
        and block.hc_ffn_fn.shape == (24, 4 * h.shape[-1])
        and all(getattr(block, f"hc_{k}_{p}").dtype == mx.float32 for k in ("attn", "ffn") for p in ("scale", "base"))
        and block.attn_norm.weight.dtype == h.dtype
        and block.ffn_norm.weight.dtype == h.dtype
        and mx.default_device() == mx.gpu
    )


def block_forward(block, h, pre, cache, shared, start, image_mask, prebuilt_end=None):
    """Block.__call__ for short rows (no CED tail): four hc launches instead of eight."""
    from . import attn_in, ffn_fuse
    c = block._config
    if attn_in.eligible(block.attn, h):
        mix_a, x, xq = attn_in.project_pre_norm_q(h, pre, block.hc_attn_fn, block.attn_norm.weight, c.norm_eps,
                                                  block.attn_norm.eps)
        shared["attn_in_xq"] = (x, xq)
    else:
        mix_a, x = project_pre_norm(h, pre, block.hc_attn_fn, block.attn_norm.weight, c.norm_eps, block.attn_norm.eps)
    a = block.attn(x, cache, shared, start, prebuilt_end=prebuilt_end)
    if EARLY_SUBMIT and not shared.get("hc_submitted"):
        shared["hc_submitted"] = True
        mx.async_eval(a)
    h, ap = post_mix(a, h, mix_a, block.hc_attn_scale, block.hc_attn_base, c.hc_eps, c.hc_sinkhorn_iters)
    if ffn_fuse.eligible(block, h, image_mask):
        return ffn_fuse.ffn_forward(block, h, ap)
    mix_f, x = project_pre_norm(h, ap, block.hc_ffn_fn, block.ffn_norm.weight, c.norm_eps, block.ffn_norm.eps)
    f = block.ffn(x, image_mask)
    return post_mix(f, h, mix_f, block.hc_ffn_scale, block.hc_ffn_base, c.hc_eps, c.hc_sinkhorn_iters)
