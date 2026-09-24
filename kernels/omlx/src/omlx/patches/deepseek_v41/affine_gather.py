# SPDX-License-Identifier: MIT
"""Expert-sorted affine MoE with an explicit source-row gather.

Preserves oMLX's FP8 activations, pre-down router weighting, and FP32 combine.
Supports the exported q3/g128 expert geometry without native g64 assumptions.
"""
import mlx.core as mx
from .quantization import QuantizedProjection, quantize_activation
from .activation import quantize_swiglu_activation
from .routing import combine_sorted_experts

def eligible(moe, x):
    return (x.shape[1] >= 512 and x.dtype == mx.bfloat16 and mx.default_device() == mx.gpu
            and all(isinstance(p, QuantizedProjection) and p.mode == 'affine'
                    and p.group_size == 128 and p.bits == 3 and p.quantize_input
                    for p in (moe.experts.w1, moe.experts.w3, moe.experts.w2)))

def forward(moe, x, ids, weights):
    shape=ids.shape
    flat=ids.reshape(-1)
    order=mx.argsort(flat)
    inverse=mx.argsort(order)
    expert=flat[order].astype(mx.uint32)
    rows=(order//shape[-1]).astype(mx.uint32)
    source=quantize_activation(x).reshape(-1,1,x.shape[-1])
    def gather(p, value, lhs=None):
        return mx.gather_qmm(value,p.weight,p.scales,p.biases,
                            lhs_indices=lhs,rhs_indices=expert,transpose=True,
                            bits=p.bits,group_size=p.group_size,mode=p.mode,
                            sorted_indices=True)
    gate=gather(moe.experts.w1,source,rows)
    up=gather(moe.experts.w3,source,rows)
    hidden=quantize_swiglu_activation(gate,up,weights.reshape(-1)[order],x.dtype,moe.experts._limit)
    routed=gather(moe.experts.w2,hidden)
    shared=moe.shared_experts(x)
    combined=combine_sorted_experts(routed,inverse,shared)
    if combined is not None:
        return combined
    routed=routed[inverse].reshape(*shape,x.shape[-1])
    return (routed.astype(mx.float32).sum(-2)+shared.astype(mx.float32)).astype(x.dtype)
