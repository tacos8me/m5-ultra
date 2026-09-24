# Copyright © 2026 Apple Inc.

import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache, _BaseCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU, switch_glu_combine

# Fused sliding-window attention needs mx.fast.scaled_dot_product_attention
# with the window_size argument.
_SDPA_HAS_WINDOW = "window_size" in (mx.fast.scaled_dot_product_attention.__doc__ or "")

# Long prefill chunks run the 8-bit attention projections as a bf16 matmul on
# weights dequantized per call. The dequantized weights are the same values the
# quantized matmul uses; only the fp32 summation order changes.
_DQ_PREFILL_MIN_TOKENS = 2048

_DQ8_T_SOURCE = """
    // Dequantize a 64 x 64 tile of W[N, K] (8-bit affine, group size 64) and
    // write it transposed to out[K, N].
    threadgroup T tile[64][72];
    const uint n0 = threadgroup_position_in_grid.x * 64;
    const uint k0 = threadgroup_position_in_grid.y * 64;
    const uint t = thread_index_in_threadgroup;
    const uint r = t / 4;
    const uint c = (t % 4) * 16;
    {
      const uint n = n0 + r;
      const uint g = n * (K / 64) + k0 / 64;
      const float s = float(scales[g]);
      const float b = float(biases[g]);
      const uint4 p = *(const device uint4*)(
          (const device uint8_t*)w + size_t(n) * K + k0 + c);
      for (int i = 0; i < 16; i++) {
        const uint q = (p[i / 4] >> (8 * (i % 4))) & 0xffu;
        tile[c + i][r] = static_cast<T>(s * q + b);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device T* o = out + size_t(k0 + r) * N + n0 + c;
    for (int i = 0; i < 16; i++) {
      o[i] = tile[r][c + i];
    }
"""


@lru_cache
def _dq8_t_kernel():
    return mx.fast.metal_kernel(
        name="mimo_dequant8_transposed",
        input_names=["w", "scales", "biases"],
        output_names=["out"],
        source=_DQ8_T_SOURCE,
    )


def _prefill_linear(layer, x):
    """layer(x) for a long prefill chunk; see _DQ_PREFILL_MIN_TOKENS."""
    if not (
        x.ndim >= 2
        and x.shape[-2] >= _DQ_PREFILL_MIN_TOKENS
        and isinstance(layer, nn.QuantizedLinear)
        and layer.mode == "affine"
        and layer.bits == 8
        and layer.group_size == 64
        and "bias" not in layer
        and x.dtype == mx.bfloat16
        and layer.scales.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return layer(x)
    N, K = layer.weight.shape[0], layer.weight.shape[1] * 4
    if N % 64 or K % 64:
        return layer(x)
    wt = _dq8_t_kernel()(
        inputs=[layer.weight, layer.scales, layer.biases],
        template=[("T", x.dtype), ("N", N), ("K", K)],
        grid=(N // 64 * 256, K // 64, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(K, N)],
        output_dtypes=[x.dtype],
    )[0]
    return x @ wt


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    rope_theta: float
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    swa_v_head_dim: int
    swa_rope_theta: float
    sliding_window_size: int
    add_full_attention_sink_bias: bool
    add_swa_attention_sink_bias: bool
    hybrid_layer_pattern: List[int]
    moe_layer_freq: List[int]
    n_routed_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    topk_method: str
    partial_rotary_factor: float
    attention_bias: bool
    layernorm_epsilon: float
    max_position_embeddings: int
    routed_scaling_factor: Optional[float] = None
    attention_value_scale: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        n = self.num_hidden_layers
        if len(self.hybrid_layer_pattern) != n:
            raise ValueError("hybrid_layer_pattern length must match num_hidden_layers")
        if len(self.moe_layer_freq) != n:
            raise ValueError("moe_layer_freq length must match num_hidden_layers")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sliding_window: bool):
        super().__init__()
        dim = args.hidden_size
        self.is_sliding_window = is_sliding_window
        if is_sliding_window:
            self.n_heads = args.swa_num_attention_heads
            self.n_kv_heads = args.swa_num_key_value_heads
            head_dim = args.swa_head_dim
            v_head_dim = args.swa_v_head_dim
            rope_theta = args.swa_rope_theta
            has_sinks = args.add_swa_attention_sink_bias
        else:
            self.n_heads = args.num_attention_heads
            self.n_kv_heads = args.num_key_value_heads
            head_dim = args.head_dim
            v_head_dim = args.v_head_dim
            rope_theta = args.rope_theta
            has_sinks = args.add_full_attention_sink_bias

        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.window_size = args.sliding_window_size if is_sliding_window else None
        self.scale = head_dim**-0.5
        self.v_scale = args.attention_value_scale

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * v_head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * v_head_dim, dim, bias=False)

        self.attention_sink_bias = mx.zeros((self.n_heads,)) if has_sinks else None

        self.rope = initialize_rope(
            int(args.partial_rotary_factor * head_dim),
            base=rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = (
            _prefill_linear(self.q_proj, x)
            .reshape(B, L, self.n_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        keys = (
            _prefill_linear(self.k_proj, x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        values = (
            _prefill_linear(self.v_proj, x)
            .reshape(B, L, self.n_kv_heads, self.v_head_dim)
            .swapaxes(1, 2)
        )

        if self.v_scale is not None:
            values = values * self.v_scale

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.window_size and L > 1 and _SDPA_HAS_WINDOW:
            output = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=mask,
                sinks=self.attention_sink_bias,
                window_size=self.window_size,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
                sinks=self.attention_sink_bias,
            )
        return _prefill_linear(self.o_proj, output.swapaxes(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor
    return inds, scores


_SIGMOID_TOPK_SOURCE = """
    constexpr int PER = E / 32;
    uint tok = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    const device T* g = gates + size_t(tok) * E;
    constexpr float neg_inf = -metal::numeric_limits<float>::infinity();

    float orig[PER];
    float sel[PER];
    for (int i = 0; i < PER; i++) {
      int e = i * 32 + lane;
      float x = static_cast<float>(g[e]);
      float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
      orig[i] = (x < 0) ? y : 1.0f - y;
      sel[i] = orig[i] + static_cast<float>(bias[e]);
    }

    // K rounds of argmax; ties go to the lowest expert index.
    float w_out = 0.0f;
    uint e_out = 0;
    float w_sum = 0.0f;
    for (int k = 0; k < K; k++) {
      float best = neg_inf;
      int bi = 0;
      for (int i = 0; i < PER; i++) {
        if (sel[i] > best) {
          best = sel[i];
          bi = i;
        }
      }
      float gmax = simd_max(best);
      uint ew = simd_min((best == gmax) ? uint(bi * 32 + lane) : 0xffffffffu);
      uint owner = ew % 32;
      float w = 0.0f;
      for (int i = 0; i < PER; i++) {
        if (lane == owner && i == int(ew / 32)) {
          w = orig[i];
          sel[i] = neg_inf;
        }
      }
      w = simd_shuffle(w, ushort(owner));
      w_sum += w;
      if (int(lane) == k) {
        w_out = w;
        e_out = ew;
      }
    }
    if (int(lane) < K) {
      float w = NORM ? w_out / (w_sum + 1e-20f) : w_out;
      inds[size_t(tok) * K + lane] = e_out;
      scores[size_t(tok) * K + lane] = w * scale[0];
    }
"""


@lru_cache
def _sigmoid_topk_kernel():
    return mx.fast.metal_kernel(
        name="mimo_sigmoid_topk",
        input_names=["gates", "bias", "scale"],
        output_names=["inds", "scores"],
        source=_SIGMOID_TOPK_SOURCE,
    )


@lru_cache
def _sigmoid_topk(top_k, routed_scaling_factor, norm_topk_prob):
    """Fused sigmoid + bias + top-k + renormalize for n_group == 1 routing."""
    norm = top_k > 1 and norm_topk_prob
    scale = mx.array([routed_scaling_factor], mx.float32)

    @mx.custom_function
    def select(gates, bias):
        E = gates.shape[-1]
        return _sigmoid_topk_kernel()(
            inputs=[gates, bias, scale],
            template=[("T", gates.dtype), ("E", E), ("K", top_k), ("NORM", norm)],
            grid=(gates.size // E * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[gates.shape[:-1] + (top_k,)] * 2,
            output_dtypes=[mx.uint32, mx.float32],
        )

    @select.vjp
    def select_vjp(primals, cotangents, outputs):
        gates, bias = primals
        inds = outputs[0]

        def scores(g):
            s = mx.take_along_axis(mx.sigmoid(g.astype(mx.float32)), inds, -1)
            if norm:
                s = s / (s.sum(axis=-1, keepdims=True) + 1e-20)
            return s * routed_scaling_factor

        _, (dgates,) = mx.vjp(scores, [gates], [cotangents[1]])
        return dgates, mx.zeros_like(bias)

    return select


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.topk_method == "noaux_tc", "Unsupported topk method."
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor or 1.0
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))

    def __call__(self, x):
        gates = x @ self.weight.T
        n_experts = gates.shape[-1]
        if (
            self.n_group == 1
            and n_experts % 32 == 0
            and self.top_k <= 32
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            select = _sigmoid_topk(
                self.top_k, self.routed_scaling_factor, self.norm_topk_prob
            )
            return select(gates, self.e_score_correction_bias)
        return group_expert_select(
            gates,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = MoEGate(config)
        self.sharding_group = None

    def __call__(self, x, residual=None):
        """Return residual + MoE(x), or MoE(x) if residual is None."""
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, scores = self.gate(x)
        if self.sharding_group is None:
            return switch_glu_combine(self.switch_mlp, x, inds, scores, residual)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y if residual is None else residual + y


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_moe: bool, is_sliding_window: bool):
        super().__init__()
        self.self_attn = Attention(config, is_sliding_window)
        self.mlp = MoE(config) if is_moe else MLP(config)
        self.is_sliding_window = is_sliding_window
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        if isinstance(self.mlp, MoE):
            return self.mlp(self.post_attention_layernorm(h), residual=h)
        return h + self.mlp(self.post_attention_layernorm(h))


class MiMoV2Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        pattern = config.hybrid_layer_pattern
        moe_freq = config.moe_layer_freq
        self.layers = [
            DecoderLayer(
                config,
                is_moe=bool(moe_freq[idx]),
                is_sliding_window=bool(pattern[idx]),
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.sliding_window_size = config.sliding_window_size
        if config.num_nextn_predict_layers > 0:
            self.mtp = MTP(config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        norm: bool = True,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        local_layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(local_layers)

        swa_local = next(
            (i for i, l in enumerate(local_layers) if l.is_sliding_window), None
        )
        ga_local = next(
            (i for i, l in enumerate(local_layers) if not l.is_sliding_window), None
        )
        full_mask = (
            create_attention_mask(h, cache[ga_local]) if ga_local is not None else None
        )
        swa_mask = None
        if swa_local is not None:
            swa_cache = cache[swa_local]
            if (
                _SDPA_HAS_WINDOW
                and h.shape[1] > 1
                and (swa_cache is None or type(swa_cache) is RotatingKVCache)
            ):
                # Keys are in temporal order; the window is applied in SDPA.
                swa_mask = "causal"
            else:
                swa_mask = create_attention_mask(
                    h, swa_cache, window_size=self.sliding_window_size
                )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, pipeline_rank + 1)

        for layer, c in zip(local_layers, cache):
            mask = swa_mask if layer.is_sliding_window else full_mask
            h = layer(h, mask, cache=c)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h) if norm else h


class MTPLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dim, eps = config.hidden_size, config.layernorm_epsilon
        self.enorm = nn.RMSNorm(dim, eps=eps)
        self.hnorm = nn.RMSNorm(dim, eps=eps)
        self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.self_attn = Attention(config, is_sliding_window=True)
        self.pre_mlp_layernorm = nn.RMSNorm(dim, eps=eps)
        self.mlp = MLP(config)
        self.final_layernorm = nn.RMSNorm(dim, eps=eps)

    def __call__(self, embeds, hidden, mask, cache):
        x = mx.concatenate([self.enorm(embeds), self.hnorm(hidden)], axis=-1)
        x = self.eh_proj(x)
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.pre_mlp_layernorm(x))


class MTP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [MTPLayer(config) for _ in range(config.num_nextn_predict_layers)]


class _MTPLayerCache:
    """Sliding window KV state of one MTP layer. Slot ``p`` is the RoPE position."""

    def __init__(self, window: int, capacity: int):
        self.window = window
        self.capacity = capacity
        self.keys = None
        self.values = None
        self.start = 0
        self.offset = 0

    def reset(self, slot: int):
        self.keys = self.values = None
        self.start = self.offset = slot

    def truncate(self, slot: int):
        if slot >= self.offset:
            return
        if slot <= self.start or self.keys is None:
            self.reset(slot)
            return
        n = slot - self.start
        self.keys = self.keys[..., :n, :]
        self.values = self.values[..., :n, :]
        self.offset = slot

    def make_mask(self, L: int):
        if L == 1:
            return None
        q = mx.arange(self.offset, self.offset + L)[:, None]
        k = mx.arange(self.start, self.offset + L)[None]
        return (k <= q) & (k > q - self.window)

    def update_and_fetch(self, keys, values):
        L = keys.shape[2]
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.offset += L
        excess = keys.shape[2] - self.capacity
        if excess > 0:
            self.keys = keys[..., excess:, :]
            self.values = values[..., excess:, :]
            self.start += excess
        else:
            self.keys, self.values = keys, values
        if L == 1:
            return keys[..., -self.window :, :], values[..., -self.window :, :]
        return keys, values


class MTPCache(_BaseCache):
    """Draft cache of the MTP head.

    It keeps the tokens given to the draft model, the recent target hidden
    states (with the tokens the target read) and the KV state of each MTP
    layer. MTP layer ``k`` at slot ``p`` reads the target hidden state at
    position ``p - 1`` and token ``p + k``, and predicts token ``p + k + 1``.
    """

    def __init__(self, num_layers: int, window: int, capacity: Optional[int] = None):
        self.capacity = capacity or window + 64
        self.layers = [_MTPLayerCache(window, self.capacity) for _ in range(num_layers)]
        # A layer entry made with a longer lag than the layer was trained for.
        self.stale_slot = None
        self.target_cache = None
        self.aligned = False
        self.tokens = None
        self.tokens_start = 0
        self.n_tokens = 0
        self.hidden = None
        self.hidden_tokens = None
        self.hidden_start = 0
        self.hidden_valid = 0
        self.hidden_valid_lo = 0
        self.hidden_new = False

    @property
    def hidden_end(self):
        return self.hidden_start + (0 if self.hidden is None else self.hidden.shape[0])

    def add_tokens(self, tokens: mx.array):
        if self.tokens is None:
            self.tokens = tokens
            self.tokens_start = self.n_tokens
        else:
            self.tokens = mx.concatenate([self.tokens, tokens])
        self.n_tokens += tokens.size
        excess = self.tokens.size - 2 * self.capacity
        if excess > 0:
            self.tokens = self.tokens[excess:]
            self.tokens_start += excess

    def get_tokens(self, start: int, end: int):
        return self.tokens[start - self.tokens_start : end - self.tokens_start]

    def push_hidden(self, start: int, tokens: mx.array, hidden: mx.array):
        if not self.aligned:
            # A new draft cache can follow a target cache that already holds
            # tokens. The first target step starts at the first draft token.
            self.aligned = True
            self.n_tokens += start
            self.tokens_start += start
        if self.hidden is None or not (self.hidden_start <= start <= self.hidden_end):
            self.hidden = hidden
            self.hidden_tokens = tokens
            self.hidden_start = start
        else:
            keep = start - self.hidden_start
            self.hidden = mx.concatenate([self.hidden[:keep], hidden])
            self.hidden_tokens = mx.concatenate([self.hidden_tokens[:keep], tokens])
        excess = self.hidden.shape[0] - self.capacity
        if excess > 0:
            self.hidden = self.hidden[excess:]
            self.hidden_tokens = self.hidden_tokens[excess:]
            self.hidden_start += excess
        self.hidden_valid = min(self.hidden_valid, start)
        self.hidden_new = True

    def validate_hidden(self):
        """Accept the target hidden states read from the same tokens as the draft."""
        # Only check after a target step, when the tokens are already evaluated.
        if not self.hidden_new:
            return
        self.hidden_new = False
        start = max(self.hidden_valid, self.hidden_start, self.tokens_start)
        end = min(self.hidden_end, self.n_tokens)
        if start >= end:
            return
        if start > self.hidden_valid:
            self.hidden_valid_lo = start
        ht = self.hidden_tokens[start - self.hidden_start : end - self.hidden_start]
        n = 0
        for a, b in zip(ht.tolist(), self.get_tokens(start, end).tolist()):
            if a != b:
                break
            n += 1
        self.hidden_valid = start + n

    def truncate_layers(self):
        # Layer k at slot p is valid while token p + k and hidden p - 1 are valid.
        for k, c in enumerate(self.layers):
            c.truncate(min(self.n_tokens - k, self.hidden_valid + 1))

    @property
    def state(self):
        arrays = [self.tokens, self.hidden, self.hidden_tokens]
        for c in self.layers:
            arrays += [c.keys, c.values]
        return [a for a in arrays if a is not None]

    @state.setter
    def state(self, v):
        raise NotImplementedError("MTPCache does not support loading state.")

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.n_tokens, n)
        self.n_tokens -= n
        if self.tokens is not None:
            self.tokens = self.tokens[: max(self.n_tokens - self.tokens_start, 0)]
        self.hidden_valid = min(self.hidden_valid, self.n_tokens)
        self.truncate_layers()
        return n

    def size(self):
        return self.n_tokens

    def empty(self):
        return self.n_tokens == 0

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.state)


class MTPDraftModel(nn.Module):
    """Draft model made from the MTP head of a target ``Model``.

    Give it to ``speculative_generate_step`` as the draft model. While it is
    in use, the target sends its hidden states to the active ``MTPCache``.
    Draft step ``j`` of a round uses MTP layer ``j``. Steps after the last
    layer use the last layer with a longer lag.
    """

    def __init__(self, target: "Model"):
        super().__init__()
        self._target = weakref.ref(target)
        self.num_layers = len(target.model.mtp.layers)
        self.window = target.args.sliding_window_size

    def make_cache(self):
        return [MTPCache(self.num_layers, self.window)]

    def __call__(self, inputs: mx.array, cache: List[Any], **kwargs):
        target = self._target()
        c = cache[0]
        sink = target._mtp_sink
        if sink is None or sink() is not c:
            target._mtp_sink = weakref.ref(c)
            c.target_cache = None
        c.add_tokens(inputs.reshape(-1).astype(mx.uint32))
        c.validate_hidden()
        c.truncate_layers()
        if c.stale_slot is not None:
            c.layers[-1].truncate(c.stale_slot)
            c.stale_slot = None

        # The anchor slot pairs the last valid target hidden state with the next token.
        anchor = min(c.hidden_valid, c.n_tokens - 1)
        depth = c.n_tokens - 1 - anchor
        k = min(depth, self.num_layers - 1)
        cache_k = c.layers[k]
        if anchor < 1:
            return mx.zeros((1, 1, target.args.vocab_size))
        if depth > k:
            c.stale_slot = anchor
        if depth > k or cache_k.offset > anchor:
            cache_k.truncate(anchor)
        lo = max(c.hidden_valid_lo, c.hidden_start, c.tokens_start - depth - 1) + 1
        # MTP layers use sliding-window attention, so only the last `window`
        # positions reach the anchor; catching up on older ones (the whole
        # prompt after prefill) is wasted work.
        lo = max(lo, anchor - self.window + 1)
        first = anchor if depth > k else max(cache_k.offset, lo)
        if not lo <= first <= anchor:
            return mx.zeros((1, 1, target.args.vocab_size))
        if cache_k.offset != first:
            cache_k.reset(first)

        layer = target.model.mtp.layers[k]
        hidden = c.hidden[first - 1 - c.hidden_start : anchor - c.hidden_start]
        tokens = c.get_tokens(first + depth, anchor + depth + 1)
        embeds = target.model.embed_tokens(tokens)
        mask = cache_k.make_mask(anchor - first + 1)
        out = layer(embeds[None], hidden[None], mask, cache_k)
        out = layer.final_layernorm(out[:, -1:])
        if target.args.tie_word_embeddings:
            return target.model.embed_tokens.as_linear(out)
        return target.lm_head(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = MiMoV2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._mtp_sink = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if self._mtp_sink is None:
            out = self.model(inputs, cache, input_embeddings)
        else:
            out = self.model(inputs, cache, input_embeddings, norm=False)
            self._send_hidden(inputs, cache, out)
            out = self.model.norm(out)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def _send_hidden(self, inputs, cache, out):
        sink = self._mtp_sink()
        if sink is None or cache is None or inputs is None:
            return
        B, L = out.shape[:2]
        offset = cache[0].offset
        if B != 1 or inputs.shape[-1] != L or not isinstance(offset, int):
            return
        # Only send to the draft cache of the generation that uses this cache.
        if sink.target_cache is None:
            sink.target_cache = weakref.ref(cache[0])
        elif sink.target_cache() is not cache[0]:
            return
        sink.push_hidden(offset - L, inputs.reshape(-1), out[0])

    def make_draft_model(self):
        if getattr(self.model, "mtp", None) is None:
            raise ValueError("The model does not have MTP weights.")
        return MTPDraftModel(self)

    def sanitize(self, weights):
        skip_prefixes = (
            "visual.",
            "audio_encoder.",
            "speech_embeddings.",
        )
        weights = {k: v for k, v in weights.items() if not k.startswith(skip_prefixes)}
        if not any(k.startswith("model.mtp.") for k in weights):
            self.model.mtp = None

        BS = 128
        bf16 = mx.bfloat16

        def detect_tp():
            n_h = self.args.num_attention_heads
            n_kv = self.args.num_key_value_heads
            hd = self.args.head_dim
            vhd = self.args.v_head_dim
            for layer_idx in range(self.args.num_hidden_layers):
                if bool(self.args.hybrid_layer_pattern[layer_idx]):
                    continue
                qkv_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                scale_key = f"{qkv_key}_scale_inv"
                if qkv_key not in weights or scale_key not in weights:
                    continue
                actual = weights[qkv_key].shape[0]
                padded = weights[scale_key].shape[0] * BS
                for tp in (1, 2, 4, 8, 16, 32):
                    if n_h % tp or n_kv % tp:
                        continue
                    pr = (n_h // tp) * hd + (n_kv // tp) * (hd + vhd)
                    pr_padded = -(-pr // BS) * BS
                    if pr * tp == actual and pr_padded * tp == padded:
                        return tp
                raise ValueError(
                    f"unable to determine fused-qkv TP layout from layer {layer_idx} "
                    f"(actual={actual}, padded={padded})"
                )
            # Fused qkv checkpoints are split in num_key_value_heads chunks.
            return n_kv

        TP = detect_tp()

        def dequant_block(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=bf16)
            m, n = weight.shape
            pad_b, pad_r = (-m) % BS, (-n) % BS
            if pad_b or pad_r:
                weight = mx.pad(weight, ((0, pad_b), (0, pad_r)))
            weight = weight.reshape((m + pad_b) // BS, BS, (n + pad_r) // BS, BS)
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_b, n + pad_r
            )
            return weight[:m, :n].astype(bf16)

        def split_qkv(qkv_fp8, scale_inv, n_h, n_kv, hd, vhd):
            q_pr = (n_h // TP) * hd
            k_pr = (n_kv // TP) * hd
            v_pr = (n_kv // TP) * vhd
            actual_pr = q_pr + k_pr + v_pr
            padded_pr = -(-actual_pr // BS) * BS
            n = qkv_fp8.shape[-1]

            qkv = mx.from_fp8(qkv_fp8, dtype=bf16).reshape(TP, actual_pr, n)
            if padded_pr > actual_pr:
                qkv = mx.pad(qkv, ((0, 0), (0, padded_pr - actual_pr), (0, 0)))
            n_col_blocks = scale_inv.shape[1]
            pad_side = BS * n_col_blocks - n
            if pad_side > 0:
                qkv = mx.pad(qkv, ((0, 0), (0, 0), (0, pad_side)))

            blocked = qkv.reshape(TP * padded_pr // BS, BS, n_col_blocks, BS)
            qkv = (blocked * scale_inv[:, None, :, None]).reshape(
                TP, padded_pr, n_col_blocks * BS
            )[:, :actual_pr, :n]

            q = mx.contiguous(qkv[:, :q_pr, :]).reshape(TP * q_pr, n).astype(bf16)
            k = (
                mx.contiguous(qkv[:, q_pr : q_pr + k_pr, :])
                .reshape(TP * k_pr, n)
                .astype(bf16)
            )
            v = (
                mx.contiguous(qkv[:, q_pr + k_pr :, :])
                .reshape(TP * v_pr, n)
                .astype(bf16)
            )
            return q, k, v

        attn_layers = [
            (f"model.layers.{i}.self_attn", bool(self.args.hybrid_layer_pattern[i]))
            for i in range(self.args.num_hidden_layers)
        ]
        attn_layers += [
            (f"model.mtp.layers.{i}.self_attn", True)
            for i in range(self.args.num_nextn_predict_layers)
        ]
        for prefix, is_swa in attn_layers:
            qkv_key = f"{prefix}.qkv_proj.weight"
            scale_key = f"{qkv_key}_scale_inv"
            if qkv_key not in weights or scale_key not in weights:
                continue

            if is_swa:
                n_h = self.args.swa_num_attention_heads
                n_kv = self.args.swa_num_key_value_heads
                hd = self.args.swa_head_dim
                vhd = self.args.swa_v_head_dim
            else:
                n_h = self.args.num_attention_heads
                n_kv = self.args.num_key_value_heads
                hd = self.args.head_dim
                vhd = self.args.v_head_dim

            q, k, v = split_qkv(
                weights.pop(qkv_key), weights.pop(scale_key), n_h, n_kv, hd, vhd
            )
            weights[f"{prefix}.q_proj.weight"] = q
            weights[f"{prefix}.k_proj.weight"] = k
            weights[f"{prefix}.v_proj.weight"] = v

        scale_keys = [k for k in weights if k.endswith("weight_scale_inv")]
        for sk in scale_keys:
            wk = sk[: -len("_scale_inv")]
            weights[wk] = dequant_block(weights[wk], weights.pop(sk))

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ("gate_proj", "down_proj", "up_proj"):
                expert0 = f"{prefix}.experts.0.{proj}.weight"
                if expert0 not in weights:
                    continue
                stacked = mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                )
                scale0 = f"{prefix}.experts.0.{proj}.weight_scale"
                if scale0 in weights:
                    weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked.view(
                        mx.uint32
                    )
                    weights[f"{prefix}.switch_mlp.{proj}.scales"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.weight_scale")
                            for e in range(self.args.n_routed_experts)
                        ]
                    )
                else:
                    weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()
        for layer in self.model.layers:
            if layer is None:
                continue

            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            if attn.attention_sink_bias is not None:
                attn.attention_sink_bias = attn.attention_sink_bias[
                    R * attn.n_heads : (R + 1) * attn.n_heads
                ]

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size))
            else:
                caches.append(KVCache())
        return caches
