# Copyright © 2023 Apple Inc.

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


# Fused decode kernels for MXFP4 SwitchGLU experts (few tokens, top-k experts).
# The per-row dot products follow the structure of MLX's fp_qmv_fast.

_MXFP4_QDOT_HEADER = """
inline float mxfp4_val(uint8_t b) {
  half h = as_type<half>(ushort((b & 7) << 9));
  h *= 16384.0h;
  return float((b & 8) ? -h : h);
}

inline float e8m0_val(uint8_t s) {
  uint32_t out = (s == 0 ? 0x400000 : (static_cast<uint32_t>(s) << 23));
  return as_type<float>(out);
}

inline float mxfp4_qdot(
    const device uint8_t* w,
    const thread float* x,
    float scale) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < 4; i++) {
    accum +=
        (x[4 * i] * mxfp4_val(uint8_t(ws[i])) +
         x[4 * i + 1] * mxfp4_val(uint8_t(ws[i] >> 4)) +
         x[4 * i + 2] * mxfp4_val(uint8_t(ws[i] >> 8)) +
         x[4 * i + 3] * mxfp4_val(uint8_t(ws[i] >> 12)));
  }
  return scale * accum;
}

// Four output rows of one expert, reduced over the simdgroup.
inline void mxfp4_rows4(
    const device uint8_t* w,
    const device uint8_t* s,
    const device bfloat16_t* x,
    int in_dims,
    uint lane,
    thread float* result) {
  const int w_row = in_dims / 2;
  const int s_row = in_dims / 32;
  w += lane * 8;
  s += lane / 2;
  x += lane * 16;
  for (int k = 0; k < in_dims; k += 512) {
    float xt[16];
    for (int i = 0; i < 16; i++) {
      xt[i] = x[i];
    }
    for (int r = 0; r < 4; r++) {
      result[r] += mxfp4_qdot(w + r * w_row, xt, e8m0_val(s[r * s_row]));
    }
    w += 256;
    s += 16;
    x += 512;
  }
}
"""

_MXFP4_GATE_UP_SOURCE = """
    uint pair = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    int row = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup * 4;
    uint e = inds[pair];
    size_t w_off = (size_t(e) * N + row) * (IN / 2);
    size_t s_off = (size_t(e) * N + row) * (IN / 32);
    const device bfloat16_t* xp = x + size_t(pair / TOPK) * IN;
    float g[4] = {0, 0, 0, 0};
    float u[4] = {0, 0, 0, 0};
    mxfp4_rows4((const device uint8_t*)wg + w_off, sg + s_off, xp, IN, lane, g);
    mxfp4_rows4((const device uint8_t*)wu + w_off, su + s_off, xp, IN, lane, u);
    for (int r = 0; r < 4; r++) {
      bfloat16_t gb = bfloat16_t(simd_sum(g[r]));
      bfloat16_t ub = bfloat16_t(simd_sum(u[r]));
      if (lane == 0) {
        float gf = gb;
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf)));
        bfloat16_t sig = bfloat16_t((gf < 0) ? y : 1.0f - y);
        bfloat16_t act = bfloat16_t(gf * float(sig));
        out[size_t(pair) * N + row + r] = bfloat16_t(float(act) * float(ub));
      }
    }
"""

_MXFP4_DOWN_SOURCE = """
    uint tok = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint k = simdgroup_index_in_threadgroup;
    int row = threadgroup_position_in_grid.x * 4;
    uint pair = tok * TOPK + k;
    uint e = inds[pair];
    float r4[4] = {0, 0, 0, 0};
    mxfp4_rows4(
        (const device uint8_t*)wd + (size_t(e) * N + row) * (IN / 2),
        sd + (size_t(e) * N + row) * (IN / 32),
        h + size_t(pair) * IN,
        IN,
        lane,
        r4);
    threadgroup float partial[TOPK][4];
    float sc = scores[pair];
    for (int r = 0; r < 4; r++) {
      float y = float(bfloat16_t(simd_sum(r4[r])));
      if (lane == 0) {
        partial[k][r] = y * sc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0 && lane < 4) {
      float acc = 0;
      for (int j = 0; j < TOPK; j++) {
        acc += partial[j][lane];
      }
      size_t o = size_t(tok) * N + row + lane;
      bfloat16_t y = bfloat16_t(acc);
      out[o] = HAS_RES ? bfloat16_t(float(res[o]) + float(y)) : y;
    }
"""


@lru_cache
def _mxfp4_moe_kernels():
    gate_up = mx.fast.metal_kernel(
        name="mxfp4_moe_gate_up_swiglu",
        input_names=["x", "inds", "wg", "sg", "wu", "su"],
        output_names=["out"],
        header=_MXFP4_QDOT_HEADER,
        source=_MXFP4_GATE_UP_SOURCE,
    )
    down = mx.fast.metal_kernel(
        name="mxfp4_moe_down_combine",
        input_names=["h", "inds", "scores", "wd", "sd", "res"],
        output_names=["out"],
        header=_MXFP4_QDOT_HEADER,
        source=_MXFP4_DOWN_SOURCE,
    )
    return gate_up, down


def _mxfp4_fast_path_ok(switch_glu, x, residual):
    projs = (switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj)
    return (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.dtype == mx.bfloat16
        and (residual is None or residual.dtype == mx.bfloat16)
        and type(switch_glu.activation) is SwiGLU
        and all(
            isinstance(p, QuantizedSwitchLinear)
            and p.mode == "mxfp4"
            and "bias" not in p
            for p in projs
        )
    )


def mxfp4_switch_glu_decode(switch_glu, x, indices, scores, residual=None):
    """Return sum_k scores[..., k] * SwitchGLU(x)[..., k, :] (plus residual if
    given) with fused kernels, or None if the fast path does not apply."""
    projs = (switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj)
    if not _mxfp4_fast_path_ok(switch_glu, x, residual):
        return None
    top_k = indices.shape[-1]
    n_tok = indices.size // top_k
    hidden, inter = x.shape[-1], switch_glu.gate_proj.output_dims
    if n_tok > 4 or top_k > 32 or hidden % 512 or inter % 512:
        return None
    gate_up, down = _mxfp4_moe_kernels()
    inds = indices.reshape(-1).astype(mx.uint32)
    g, u, d = projs
    h = gate_up(
        inputs=[x.reshape(n_tok, hidden), inds, g.weight, g.scales, u.weight, u.scales],
        template=[("TOPK", top_k), ("N", inter), ("IN", hidden)],
        grid=(inter // 8 * 64, n_tok * top_k, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n_tok * top_k, inter)],
        output_dtypes=[x.dtype],
    )[0]
    has_res = residual is not None
    res = residual.reshape(n_tok, hidden) if has_res else mx.zeros((1,), x.dtype)
    y = down(
        inputs=[
            h,
            inds,
            scores.reshape(-1).astype(mx.float32),
            d.weight,
            d.scales,
            res,
        ],
        template=[("TOPK", top_k), ("N", hidden), ("IN", inter), ("HAS_RES", has_res)],
        grid=(hidden // 4 * 32 * top_k, n_tok, 1),
        threadgroup=(32 * top_k, 1, 1),
        output_shapes=[(n_tok, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(indices.shape[:-1] + (hidden,))


# Grouped decode kernels for more tokens: the (token, expert) pairs are grouped
# by expert so each selected expert's weights are read once for all of its
# tokens. Rounding follows the kernels above.

_MOE_GROUP_SOURCE = """
    // One threadgroup of E threads. Group g gets the g-th selected expert
    // (ascending id) and its pairs; unused groups get a zero count.
    uint e = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sgi = simdgroup_index_in_threadgroup;
    threadgroup atomic_int cnt[E];
    threadgroup atomic_int cur[E];
    threadgroup int tot_c[E / 32];
    threadgroup int tot_n[E / 32];
    threadgroup int tot_w[E / 32];
    atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = e; p < P; p += E) {
      atomic_fetch_add_explicit(&cnt[inds[p]], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
    int nz = c > 0 ? 1 : 0;
    int nw = (c + TT - 1) / TT;
    int pc = simd_prefix_exclusive_sum(c);
    int pn = simd_prefix_exclusive_sum(nz);
    int pw = simd_prefix_exclusive_sum(nw);
    if (lane == 31) {
      tot_c[sgi] = pc + c;
      tot_n[sgi] = pn + nz;
      tot_w[sgi] = pw + nw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int ng = 0;
    int n_items = 0;
    for (int i = 0; i < int(E / 32); i++) {
      if (i < int(sgi)) {
        pc += tot_c[i];
        pn += tot_n[i];
        pw += tot_w[i];
      }
      ng += tot_n[i];
      n_items += tot_w[i];
    }
    // Work item pw + j is pass j (tokens [j TT, j TT + TT)) of group pn.
    for (int j = 0; j < nw; j++) {
      items[pw + j] = (pn << 16) | j;
    }
    for (int i = n_items + e; i < NW; i += E) {
      items[i] = -1;
    }
    atomic_store_explicit(&cur[e], pc, memory_order_relaxed);
    if (nz) {
      gexp[pn] = e;
      gstart[pn] = pc;
      gcnt[pn] = c;
    }
    for (int g = ng + e; g < G; g += E) {
      gexp[g] = 0;
      gstart[g] = 0;
      gcnt[g] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The slot order within an expert does not change any result.
    for (int p = e; p < P; p += E) {
      int slot = atomic_fetch_add_explicit(&cur[inds[p]], 1, memory_order_relaxed);
      order[slot] = p;
    }
"""

_MXFP4_GROUPED_HEADER = """
inline float e8m0_val_x16384(uint8_t s) {
  uint32_t out = (s == 0 ? 0x400000 : (static_cast<uint32_t>(s) << 23));
  return as_type<float>(out) * 16384.0f;
}

// R rows of one expert times the first tn (<= TT) tokens of one group. Each
// lane covers 8 of every 256 input values. The x loads of all TT tokens are
// issued together (past the group's end they repeat its last token). Every
// lane ends with the simdgroup sum for i = t * R + r in own[i / 32] of lane
// i % 32. More than about 16 accumulators (TT * R) plus 8 TT x values cost
// the occupancy the loads need.
template <int TT, int IN, int R>
inline void mxfp4_rows_tokens(
    const device uint8_t* w,
    const device uint8_t* s,
    const thread uint* xrow,
    const device bfloat16_t* x,
    int tn,
    uint lane,
    thread float* own) {
  constexpr int w_row = IN / 2;
  constexpr int s_row = IN / 32;
  w += lane * 4;
  s += lane / 4;
  x += lane * 8;
  float acc[TT][R];
  #pragma unroll
  for (int t = 0; t < TT; t++) {
    #pragma unroll
    for (int r = 0; r < R; r++) {
      acc[t][r] = 0;
    }
  }
  for (int k = 0; k < IN; k += 256) {
    float2 xv[TT][4];
    #pragma unroll
    for (int t = 0; t < TT; t++) {
      metal::vec<bfloat16_t, 8> xb =
          *(const device metal::vec<bfloat16_t, 8>*)(x + xrow[t] + k);
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        xv[t][j] = float2(float(xb[j]), float(xb[j + 4]));
      }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
      uint32_t wr = *(const device uint32_t*)(w + r * w_row + k / 2);
      float sc = e8m0_val_x16384(s[r * s_row + k / 32]);
      // Values j and j + 4 as a half2, times 2^-14 (folded into sc).
      float2 wv[4];
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t q = wr >> (4 * j);
        wv[j] = float2(as_type<half2>(
            ((q & 0x00070007u) << 9) | ((q & 0x00080008u) << 12)));
      }
      #pragma unroll
      for (int t = 0; t < TT; t++) {
        if (t < tn) {
          float2 d = xv[t][0] * wv[0];
          d = fma(xv[t][1], wv[1], d);
          d = fma(xv[t][2], wv[2], d);
          d = fma(xv[t][3], wv[3], d);
          acc[t][r] = fma(sc, d.x + d.y, acc[t][r]);
        }
      }
    }
  }
  #pragma unroll
  for (int t = 0; t < TT; t++) {
    if (t < tn) {
      #pragma unroll
      for (int r = 0; r < R; r++) {
        float v = simd_sum(acc[t][r]);
        if (lane == uint((t * R + r) % 32)) {
          own[(t * R + r) / 32] = v;
        }
      }
    }
  }
}
"""

_MXFP4_GROUPED_GATE_UP_SOURCE = """
    // One pass (up to TT tokens of one group) per threadgroup row.
    int item = items[threadgroup_position_in_grid.y];
    if (item < 0) {
      return;
    }
    uint g = item >> 16;
    int t0 = (item & 0xffff) * TT;
    int tn = min(TT, gcnt[g] - t0);
    uint lane = thread_index_in_simdgroup;
    int row = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * R;
    size_t base = size_t(gexp[g]) * N + row;
    const device uint8_t* wgp = (const device uint8_t*)wg + base * (IN / 2);
    const device uint8_t* wup = (const device uint8_t*)wu + base * (IN / 2);
    const device uint8_t* sgp = sg + base * (IN / 32);
    const device uint8_t* sup = su + base * (IN / 32);
    int start = gstart[g] + t0;
    constexpr int NO = (TT * R + 31) / 32;
    uint pr[TT];
    uint xrow[TT];
    #pragma unroll
    for (int t = 0; t < TT; t++) {
      pr[t] = order[start + min(t, tn - 1)];
      xrow[t] = (pr[t] / TOPK) * IN;
    }
    float own[2][NO];
    mxfp4_rows_tokens<TT, IN, R>(wgp, sgp, xrow, x, tn, lane, own[0]);
    mxfp4_rows_tokens<TT, IN, R>(wup, sup, xrow, x, tn, lane, own[1]);
    #pragma unroll
    for (int i = 0; i < NO; i++) {
      int t = (i * 32 + lane) / R;
      if (i * 32 + int(lane) < TT * R && t < tn) {
        float gf = bfloat16_t(own[0][i]);
        bfloat16_t ub = bfloat16_t(own[1][i]);
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf)));
        bfloat16_t sig = bfloat16_t((gf < 0) ? y : 1.0f - y);
        bfloat16_t act = bfloat16_t(gf * float(sig));
        out[size_t(pr[t]) * N + row + (i * 32 + lane) % R] = bfloat16_t(float(act) * float(ub));
      }
    }
"""

_MXFP4_GROUPED_DOWN_SOURCE = """
    int item = items[threadgroup_position_in_grid.y];
    if (item < 0) {
      return;
    }
    uint g = item >> 16;
    int t0 = (item & 0xffff) * TT;
    int tn = min(TT, gcnt[g] - t0);
    uint lane = thread_index_in_simdgroup;
    int row = (threadgroup_position_in_grid.x * 2 + simdgroup_index_in_threadgroup) * R;
    size_t base = size_t(gexp[g]) * N + row;
    const device uint8_t* wp = (const device uint8_t*)wd + base * (IN / 2);
    const device uint8_t* sp = sd + base * (IN / 32);
    int start = gstart[g] + t0;
    constexpr int NO = (TT * R + 31) / 32;
    uint pr[TT];
    uint xrow[TT];
    #pragma unroll
    for (int t = 0; t < TT; t++) {
      pr[t] = order[start + min(t, tn - 1)];
      xrow[t] = pr[t] * IN;
    }
    float own[1][NO];
    mxfp4_rows_tokens<TT, IN, R>(wp, sp, xrow, h, tn, lane, own[0]);
    #pragma unroll
    for (int i = 0; i < NO; i++) {
      int t = (i * 32 + lane) / R;
      if (i * 32 + int(lane) < TT * R && t < tn) {
        out[size_t(pr[t]) * N + row + (i * 32 + lane) % R] = bfloat16_t(own[0][i]);
      }
    }
"""


_MOE_PAIR_COMBINE_SOURCE = """
    uint gid = thread_position_in_grid.x;
    constexpr int chunks = N / 8;
    uint t = gid / chunks;
    uint c = (gid % chunks) * 8;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < TOPK; k++) {
      uint p = t * TOPK + k;
      float s = scores[p];
      metal::vec<bfloat16_t, 8> v =
          *(const device metal::vec<bfloat16_t, 8>*)(y + size_t(p) * N + c);
      for (int j = 0; j < 8; j++) {
        acc[j] += float(v[j]) * s;
      }
    }
    metal::vec<bfloat16_t, 8> o;
    for (int j = 0; j < 8; j++) {
      bfloat16_t yj = bfloat16_t(acc[j]);
      o[j] = HAS_RES ? bfloat16_t(float(res[size_t(t) * N + c + j]) + float(yj)) : yj;
    }
    *(device metal::vec<bfloat16_t, 8>*)(out + size_t(t) * N + c) = o;
"""


@lru_cache
def _mxfp4_grouped_kernels():
    group = mx.fast.metal_kernel(
        name="moe_group_pairs",
        input_names=["inds"],
        output_names=["gexp", "gstart", "gcnt", "order", "items"],
        source=_MOE_GROUP_SOURCE,
    )
    gate_up = mx.fast.metal_kernel(
        name="mxfp4_moe_grouped_gate_up_swiglu",
        input_names=["x", "order", "items", "gexp", "gstart", "gcnt", "wg", "sg", "wu", "su"],
        output_names=["out"],
        header=_MXFP4_GROUPED_HEADER,
        source=_MXFP4_GROUPED_GATE_UP_SOURCE,
    )
    down = mx.fast.metal_kernel(
        name="mxfp4_moe_grouped_down",
        input_names=["h", "order", "items", "gexp", "gstart", "gcnt", "wd", "sd"],
        output_names=["out"],
        header=_MXFP4_GROUPED_HEADER,
        source=_MXFP4_GROUPED_DOWN_SOURCE,
    )
    combine = mx.fast.metal_kernel(
        name="moe_pair_combine",
        input_names=["y", "scores", "res"],
        output_names=["out"],
        source=_MOE_PAIR_COMBINE_SOURCE,
    )
    return group, gate_up, down, combine


# Above this many tokens the sorted gather_qmm path is faster.
_GROUPED_MAX_TOKENS = 48


def mxfp4_switch_glu_grouped(switch_glu, x, indices, scores, residual=None):
    """Like mxfp4_switch_glu_decode, but the (token, expert) pairs are grouped
    by expert so each selected expert is dequantized once per four of its
    tokens. Returns None if the fast path does not apply."""
    if not _mxfp4_fast_path_ok(switch_glu, x, residual):
        return None
    top_k = indices.shape[-1]
    n_pairs = indices.size
    n_tok = n_pairs // top_k
    hidden, inter = x.shape[-1], switch_glu.gate_proj.output_dims
    n_experts = switch_glu.gate_proj.num_experts
    if hidden % 256 or inter % 256 or n_experts % 32 or n_experts > 1024:
        return None
    # Tokens per pass and rows per simdgroup. Each pass of a group is its own
    # row of threadgroups, so the passes of a busy expert run in parallel.
    tt, rows = (2 if n_tok <= 3 else 4), 4
    n_groups = min(n_pairs, n_experts)
    n_items = n_groups + (n_pairs + tt - 1) // tt
    group, gate_up, down, combine = _mxfp4_grouped_kernels()
    gexp, gstart, gcnt, order, items = group(
        inputs=[indices.reshape(-1).astype(mx.uint32)],
        template=[("E", n_experts), ("P", n_pairs), ("G", n_groups), ("TT", tt), ("NW", n_items)],
        grid=(n_experts, 1, 1),
        threadgroup=(n_experts, 1, 1),
        output_shapes=[(n_groups,), (n_groups,), (n_groups,), (n_pairs,), (n_items,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.uint32, mx.int32],
    )
    g, u, d = (switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj)
    h = gate_up(
        inputs=[
            x.reshape(n_tok, hidden),
            order,
            items,
            gexp,
            gstart,
            gcnt,
            g.weight,
            g.scales,
            u.weight,
            u.scales,
        ],
        template=[("TOPK", top_k), ("N", inter), ("IN", hidden), ("TT", tt), ("R", rows)],
        grid=(inter // (2 * rows) * 64, n_items, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n_pairs, inter)],
        output_dtypes=[x.dtype],
    )[0]
    y = down(
        inputs=[h, order, items, gexp, gstart, gcnt, d.weight, d.scales],
        template=[("N", hidden), ("IN", inter), ("TT", tt), ("R", rows)],
        grid=(hidden // (2 * rows) * 64, n_items, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n_pairs, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    has_res = residual is not None
    res = residual.reshape(n_tok, hidden) if has_res else mx.zeros((1,), x.dtype)
    out = combine(
        inputs=[y, scores.reshape(-1).astype(mx.float32), res],
        template=[("TOPK", top_k), ("N", hidden), ("HAS_RES", has_res)],
        grid=(n_tok * hidden // 8, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_tok, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(indices.shape[:-1] + (hidden,))


_SORTED_COMBINE_SOURCE = """
    uint gid = thread_position_in_grid.x;
    constexpr int chunks = N / 8;
    uint t = gid / chunks;
    uint c = (gid % chunks) * 8;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < TOPK; k++) {
      uint r = inv_order[t * TOPK + k];
      float s = scores[t * TOPK + k];
      metal::vec<T, 8> v = *(const device metal::vec<T, 8>*)(y + size_t(r) * N + c);
      for (int j = 0; j < 8; j++) {
        acc[j] += float(v[j]) * s;
      }
    }
    metal::vec<T, 8> o;
    for (int j = 0; j < 8; j++) {
      T yj = static_cast<T>(acc[j]);
      o[j] = HAS_RES ? static_cast<T>(float(res[size_t(t) * N + c + j]) + float(yj)) : yj;
    }
    *(device metal::vec<T, 8>*)(out + size_t(t) * N + c) = o;
"""


# gather_qmm can read the x rows of sorted expert indices in the matmul kernel
# (no gathered copy of x) when this is in its docstring.
_GATHER_QMM_LHS_ROWS = "only ``rhs_indices`` needs to be sorted" in (
    mx.gather_qmm.__doc__ or ""
)


def _gathered_rows_matmul(proj, x, rows, idx):
    """proj(x[rows], idx) for sorted idx, without a gathered copy of x."""
    return mx.gather_qmm(
        x,
        proj["weight"],
        proj["scales"],
        proj.get("biases"),
        lhs_indices=rows,
        rhs_indices=idx,
        transpose=True,
        group_size=proj.group_size,
        bits=proj.bits,
        mode=proj.mode,
        sorted_indices=True,
    )


@lru_cache
def _sorted_combine_kernel():
    return mx.fast.metal_kernel(
        name="switch_sorted_combine",
        input_names=["y", "inv_order", "scores", "res"],
        output_names=["out"],
        source=_SORTED_COMBINE_SOURCE,
    )


def switch_glu_combine(switch_glu, x, indices, scores, residual=None):
    """Return sum_k scores[..., k] * SwitchGLU(x)[..., k, :] (plus residual if
    given). With many tokens the expert outputs stay in expert order and one
    kernel gathers, weights and sums them."""
    n_tok = indices.size // indices.shape[-1]
    if n_tok == 1:
        y = mxfp4_switch_glu_decode(switch_glu, x, indices, scores, residual)
    elif n_tok <= _GROUPED_MAX_TOKENS:
        y = mxfp4_switch_glu_grouped(switch_glu, x, indices, scores, residual)
    else:
        y = None
    if y is not None:
        return y

    top_k = indices.shape[-1]
    hidden = x.shape[-1]
    if not (
        indices.size >= 64
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.dtype in (mx.bfloat16, mx.float16)
        and (residual is None or residual.dtype == x.dtype)
        and type(switch_glu.activation) is SwiGLU
        and hidden % 8 == 0
    ):
        y = switch_glu(x, indices)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y if residual is None else residual + y

    indices = mx.stop_gradient(indices)
    n_tok = indices.size // top_k
    up_proj, gate_proj = switch_glu.up_proj, switch_glu.gate_proj
    if (
        _GATHER_QMM_LHS_ROWS
        and isinstance(up_proj, QuantizedSwitchLinear)
        and isinstance(gate_proj, QuantizedSwitchLinear)
        and up_proj.mode != "affine"
        and gate_proj.mode != "affine"
        and "bias" not in up_proj
        and "bias" not in gate_proj
    ):
        flat = indices.flatten()
        order = mx.argsort(flat)
        inv_order = mx.argsort(order)
        idx = flat[order]
        rows = (order // top_k).astype(mx.uint32)
        xt = x.reshape(n_tok, 1, hidden)
        h = switch_glu.activation(
            _gathered_rows_matmul(up_proj, xt, rows, idx),
            _gathered_rows_matmul(gate_proj, xt, rows, idx),
        )
    else:
        xs, idx, inv_order = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
        h = switch_glu.activation(
            up_proj(xs, idx, sorted_indices=True),
            gate_proj(xs, idx, sorted_indices=True),
        )
    ys = switch_glu.down_proj(h, idx, sorted_indices=True)
    has_res = residual is not None
    res = residual.reshape(n_tok, hidden) if has_res else mx.zeros((1,), x.dtype)
    out = _sorted_combine_kernel()(
        inputs=[ys.reshape(-1, hidden), inv_order, scores.astype(mx.float32), res],
        template=[("T", x.dtype), ("TOPK", top_k), ("N", hidden), ("HAS_RES", has_res)],
        grid=(n_tok * hidden // 8, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_tok, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(indices.shape[:-1] + (hidden,))
