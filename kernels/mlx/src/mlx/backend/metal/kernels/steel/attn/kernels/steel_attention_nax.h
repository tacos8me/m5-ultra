// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/attn/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
constant bool has_window [[function_constant(303)]];
constant bool row_packed [[function_constant(304)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) thread : scale(scale_) {}

  METAL_FUNC T apply(T x) const thread {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float,
    int BDV = BD>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Without a window, run causal Q blocks in reverse order, so the longest
  // blocks start first, and give the query heads of one KV head consecutive
  // threadgroups, so that they read the same keys at the same time.
  const bool causal_order = do_causal && !has_window;
  int qx = int(tid.x);
  int head = int(tid.y);
  if (causal_order) {
    const int g = params->gqa_factor;
    const int id = int(tid.x) + int(params->NQ) * int(tid.y);
    qx = (id / g) % int(params->NQ);
    head = (id / g / int(params->NQ)) * g + id % g;
  }
  const int qb = causal_order ? int(params->NQ) - 1 - qx : qx;

  // With packed rows the row strides are compile-time constants.
  const int q_ld = row_packed ? BD : int(params->Q_strides[2]);
  const int k_ld = row_packed ? BD : int(params->K_strides[2]);
  const int v_ld = row_packed ? BDV : int(params->V_strides[2]);
  const int o_ld = row_packed ? BDV : int(params->O_strides[2]);

  // Move to correct block
  ulong3 tidl{ulong(qb), ulong(head), tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * q_ld; // Sequence

  ulong kv_head_idx = head / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * o_ld; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // Value head dim frags
  constexpr int TDV = BDV / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TDV % 2 == 0, "P@V accumulates output fragments in pairs");
  using otile_t = NAXTile<AccumType, TQ, TDV>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * q_ld;

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (qb + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = qb * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Sliding window: row r sees keys c with r - c < window. Skip blocks
  // below the window of the first row, mask blocks below the last row.
  int kb_start = 0;
  int kb_lim_window = 0;
  if (has_window) {
    int q_min = qb * BQ + params->qL_off;
    int q_max = q_min + BQ - 1;
    kb_start = max(0, q_min - params->window + 1) / BK;
    kb_start = min(kb_start, kb_lim);
    kb_lim_window = (max(0, q_max - params->window + 1) + BK - 1) / BK;
    K += kb_start * BK * k_ld;
    V += kb_start * BK * v_ld;
  }

  const bool is_last_bq = qb == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // For bfloat16 with head dim 192 and no window, keep Q in registers and
  // rescale O only when a row max grows by more than 2^8. Together they are
  // faster than either alone. With float16 the late rescale adds error.
  const bool q_in_regs = (BD == 192) && is_same_v<T, bfloat> && !has_window;
  NAXTile<T, TQ, TD> Qreg;
  if (q_in_regs) {
    if (!align_Q && is_last_q) {
      Qreg.load_rows(Q, q_ld, lim_rows_q);
    } else {
      Qreg.load(Q, q_ld);
    }
  }

  // Loop over KV seq length
  for (int kb = kb_start; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        // Unrolling the head-dim loop by 4, rather than fully, potentially
        // lets the compiler interleave the next K-tile loads with the running
        // mma chain instead of hoisting all TD loads up front and is faster
        // for head dim 128.
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * kU * q_ld + id * kU;
          const int K_load_off = ik * kU * k_ld + id * kU;

          if (!q_in_regs) {
            if (!align_Q && is_last_q) {
              Qtile.load_rows(
                  Q + Q_load_off,
                  q_ld,
                  lim_rows_q - iq * kU);
            } else {
              Qtile.load(Q + Q_load_off, q_ld);
            }
          }

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off,
                k_ld,
                lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, k_ld);
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              q_in_regs ? Qreg.frag_at(iq, id) : Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = qb * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Mask out keys below the sliding window
    if (has_window && kb < kb_lim_window) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = qb * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r - c >= params->window) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = qb * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    if (q_in_regs) {
      // Keep the old max unless the row max grows by more than 2^8.
      bool rescale = false;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        const bool up = new_max[i] - max_score[i] > AccumType(8);
        new_max[i] = up ? new_max[i] : max_score[i];
        rescale |= up;
      }

      Stile.template row_bin_op<ExpSubOp>(new_max);

      if (simd_any(rescale)) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          factor[i] = fast::exp2(max_score[i] - new_max[i]);
          max_score[i] = new_max[i];
          sum_score[i] = sum_score[i] * factor[i];
        }
        Otile.template row_bin_op<MulOp>(factor);
      }

      Stile.template row_reduce<SumOp>(sum_score);
    } else {
      // exp(Si - rowmax(Si))
      Stile.template row_bin_op<ExpSubOp>(new_max);

      // Factor exp(rowmax(Si) - rowmax(Si-1))
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        factor[i] = fast::exp2(max_score[i] - new_max[i]);
        max_score[i] = new_max[i];
      }

      // Row Sum
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        sum_score[i] = sum_score[i] * factor[i];
      }

      Stile.template row_reduce<SumOp>(sum_score);

      // Update O
      Otile.template row_bin_op<MulOp>(factor);
    }

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDV; id += 2) {
        if constexpr (BDV == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * v_ld + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off,
                v_ld,
                lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, v_ld);
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * k_ld;
    V += BK * v_ld;
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * o_ld;

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, o_ld, lim_rows_q);
  } else {
    Otile.store(O, o_ld);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Head-dim split attention kernel
///////////////////////////////////////////////////////////////////////////////

// Variant of attention_nax for wide heads (bd = 256). There, the per-simdgroup
// accumulator working set of attention_nax (TD output fragments plus the S
// fragments) is what gates tensor-unit throughput, so this kernel splits the
// head dim across the WN = 2 simdgroups of the second warp dimension: each
// simdgroup of a pair owns one half of D for Q@K.T and one half of Dv for P@V,
// halving its accumulator set. The pair exchanges its partial Q@K.T sums
// through threadgroup memory, then both simdgroups run softmax redundantly on
// the full S tile (the row statistics are cheap) and each accumulates P@V for
// its own half of Dv.

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax_dsplit(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // The WM simdgroups along the first warp dimension split the Q sequence;
  // the WN simdgroups along the second split the head dim. The exchange
  // below reduces exactly one peer, so WN is fixed at 2.
  static_assert(WN == 2, "The head-dim split kernel needs WN == 2");
  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags over the full head dim
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TD % WN == 0, "The head dim must split evenly across WN");

  // HeadDim frags / columns owned by each of the WN simdgroups of a row group
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;

  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");
  static_assert(TK % 2 == 0, "S fragments are exchanged pair by pair");

  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;

  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * row_group;
  Q += tm * int(params->Q_strides[2]) + d_half * BDh;
  K += d_half * BDh;
  V += d_half * BDh;
  O += tm * int(params->O_strides[2]) + d_half * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Sliding window: row r sees keys c with r - c < window. Skip blocks
  // below the window of the first row, mask blocks below the last row.
  int kb_start = 0;
  int kb_lim_window = 0;
  if (has_window) {
    int q_min = tid.x * BQ + params->qL_off;
    int q_max = q_min + BQ - 1;
    kb_start = max(0, q_min - params->window + 1) / BK;
    kb_start = min(kb_start, kb_lim);
    kb_lim_window = (max(0, q_max - params->window + 1) + BK - 1) / BK;
    K += kb_start * BK * int(params->K_strides[2]);
    V += kb_start * BK * int(params->V_strides[2]);
  }

  const bool is_last_q = int(tid.x) == (params->NQ_aligned);
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;

  // One slot per (row group, half): a fragment pair in per-lane-linear
  // layout. Both halves share the fragment-to-lane mapping, so the
  // exchange needs no coordinate math.
  threadgroup AccumType s_xchg[WM][WN][2 * kEPF * 32];

  // Keep the simdgroup's Q half resident in registers for the whole KV
  // loop: TDh fragments of T are cheap next to the accumulators.
  NAXTile<T, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    const int Q_load_off = id * kU;
    if (!align_Q && is_last_q) {
      Qtiles[id].load_rows(
          Q + Q_load_off, int(params->Q_strides[2]), lim_rows_q);
    } else {
      Qtiles[id].load(Q + Q_load_off, int(params->Q_strides[2]));
    }
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Loop over KV seq length
  for (int kb = kb_start; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    stile_t Stile;
    Stile.clear();

    // S = Q @ K.T, this half of D only, exchanged pair by pair.
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Ktile.load_rows(
              K + K_load_off, int(params->K_strides[2]), lim_rows_k - ik * kU);
        } else {
          Ktile.load(K + K_load_off, int(params->K_strides[2]));
        }

        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }

      // Exchange the partial pair and reduce.
      threadgroup AccumType* slot = s_xchg[row_group][d_half];
      thread auto& s0 = Stile.frag_at(0, ik);
      thread auto& s1 = Stile.frag_at(0, ik + 1);
      const short base = short(simd_lane_id) * (2 * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + i] = s0[i];
        slot[base + kEPF + i] = s1[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup AccumType* peer = s_xchg[row_group][1 - d_half];
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s0[i] += peer[base + i];
        s1[i] += peer[base + kEPF + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + ii * stile_t::kFragRowsJump + sm;
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c) ? neg_inf : fg[loc];
          }
        }
      }
    }

    // Mask out keys below the sliding window
    if (has_window && kb < kb_lim_window) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r - c >= params->window) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + kU <= params->qL && base_col + BK <= params->kL) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      }
    }

    // Do softmax (redundantly per half; the row statistics are cheap)
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O = P @ V, this half of Dv only.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;

        const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Vtile.load_rows(
              V + V_load_off, int(params->V_strides[2]), lim_rows_k - ik * kU);
        } else {
          Vtile.load(V + V_load_off, int(params->V_strides[2]));
        }

        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    // Next block
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Split-KV decode kernel
///////////////////////////////////////////////////////////////////////////////

// A few queries per head (qL <= 8). Simdgroup i serves query i: the
// gqa_factor query heads of a KV head form the rows of one 16-row tile. All
// simdgroups of a threadgroup read the same chunk of the keys, so K and V
// come from memory once. Writes the unnormalized partial output with its row
// max and sum in the layout of sdpa_vector_2pass_2.

// clang-format off
template <
    typename T,
    int BK,
    int BD,
    int BDV,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(256)]] void attention_nax_decode(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant AttnDecodeParams* params [[buffer(6)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    const device bool* mask [[buffer(8), function_constant(has_mask)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) { // clang-format on

  constexpr short kU = 16;
  constexpr int TD = BD / kU;
  constexpr int TDV = BDV / kU;
  constexpr short TK = BK / kU;
  static_assert(TK % 2 == 0 && TDV % 2 == 0, "Fragments are used in pairs");

  const int G = params->gqa_factor;
  const int qL = params->qL;
  const int qi = simd_group_id;
  const int block = tid.x;
  const int kv_head = tid.y;
  const int batch = tid.z;
  const int q_head0 = kv_head * G;
  const int n_q_heads = G * int(tpg.y);

  Q += batch * params->Q_strides[0] + q_head0 * params->Q_strides[1] +
      qi * params->Q_strides[2];
  K += batch * params->K_strides[0] + kv_head * params->K_strides[1];
  V += batch * params->V_strides[0] + kv_head * params->V_strides[1];

  // Causal: query qi sees keys up to kL - qL + qi.
  const int k_lim = do_causal ? params->kL - qL + qi + 1 : params->kL;
  const int k_start = block * params->chunk;
  const int k_end = min(k_lim, k_start + params->chunk);
  K += k_start * params->K_strides[2];
  V += k_start * params->V_strides[2];

  using otile_t = NAXTile<AccumType, 1, TDV>;
  using stile_t = NAXTile<AccumType, 1, TK>;
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  NAXTile<T, 1, TD> Qtile;
  Qtile.load_rows(Q, int(params->Q_strides[1]), G);

  otile_t Otile;
  Otile.clear();

  const float scale2 = params->scale * 1.44269504089f;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  if (has_sinks && block == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int r = min(sm + i * stile_t::kFragRowsJump, G - 1);
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[q_head0 + r]);
      sum_score[i] = 1;
    }
  }

  for (int kb = k_start; kb < k_end; kb += BK) {
    const int k_rem = k_end - kb;
    const bool full = k_rem >= BK;

    // S = Q @ K.T
    stile_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_off = ik * kU * int(params->K_strides[2]) + id * kU;
        if (full) {
          Ktile.load(K + K_off, int(params->K_strides[2]));
        } else {
          Ktile.load_rows(
              K + K_off, int(params->K_strides[2]), k_rem - ik * kU);
        }
        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtile.frag_at(0, id),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    if (has_mask) {
      // A boolean mask shared by the query heads of this query.
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      const device bool* mrow =
          mask + batch * params->M_strides[0] + qi * params->M_strides[1];
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
          const int kc = ik * kU + sn + jj;
          const bool keep =
              kc < k_rem && mrow[(kb + kc) * params->M_strides[2]];
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = keep ? fg[loc] : neg_inf;
          }
        }
      }
    } else if (!full) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (ik * kU + sn + jj < k_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Online softmax
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDV; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const int V_off = ik * kU * int(params->V_strides[2]) + id * kU;
        if (full) {
          Vtile.load(V + V_off, int(params->V_strides[2]));
        } else {
          Vtile.load_rows(
              V + V_off, int(params->V_strides[2]), k_rem - ik * kU);
        }
        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Partials for sdpa_vector_2pass_2: row r of this tile is query head
  // q_head0 + r at query qi, stored at ((head * qL + qi) * blocks + block).
  const int blocks = params->blocks;
  const size_t row_base = (size_t(batch) * n_q_heads + q_head0) * qL + qi;
  device T* Ob = O + (row_base * blocks + block) * BDV;
  Otile.store_rows(Ob, qL * blocks * BDV, G);
  if (sn == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int r = sm + i * stile_t::kFragRowsJump;
      if (r < G) {
        const size_t idx = (row_base + size_t(r) * qL) * blocks + block;
        sums[idx] = sum_score[i];
        maxs[idx] = max_score[i] * M_LN2_F;
      }
    }
  }
}
