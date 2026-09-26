// og-moe: DeepSeek-V4.1 routed + shared MoE for the split-nv box (TP2: this rank holds the I = 1152 slice of every
// expert), sm_120.
//
// Arithmetic (numerics og-s4.4), identical for every row count and every kernel below:
//   * activations: official act_quant (e4m3, ue8m0 scale 2^ceil(log2(max(amax,1e-4)/448)) per 32, RN, satfinite)
//   * every GEMM element = fp32 sum, in ascending order, over aligned groups of four 32-wide k-blocks of the group's
//     block-scaled MMAs chained in the tensor core (first block from zero; e2m1/e4m3 x e4m3, ue8m0 scales). This is the
//     official kernel.py per-block promotion with a promotion interval of 4 (as DeepGEMM), ~1e-7 relative; every
//     kernel below issues the same MMA sequence per element (same operand roles, same k relabeling), and MMA results
//     do not depend on neighbouring rows/columns, so decode and prefill produce bit-identical rows.
//   * gate/up rounded to bf16, clamp (up both sides, gate from above, 10), silu(g)*u in fp32, times the route weight
//     BEFORE w2's activation quant (inference/model.py order), bf16, act_quant, w2 partial over this rank's K slice.
//   * each (token, expert) w2 partial is rounded to bf16 (the official code rounds every expert's w2 output to bf16);
//     per token: fp32 sum of the six routed partials in ascending expert id, then the shared partial, -> bf16; the
//     caller all-reduces the bf16 partials across the two ranks (as before).
// Weights are read in place: routed = FlashInfer SM120 MXFP4 layout (w13 rows [0,I) up, [I,2I) gate; e2m1 packed,
// low nibble = even k; ue8m0 scales 128x4-swizzled per expert), shared = the engine's FP8 e4m3 rows + 32x32 block scales.
// Fragments use one fixed k-relabeling inside each 32-block (thread t owns k 8t..8t+7) for A and B alike.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace og {

constexpr int H = 5120;
constexpr int I = 1152;
constexpr int KB1 = H / 32;  // 160 k-blocks for gate/up
constexpr int KB2 = I / 32;  // 36 k-blocks for down
constexpr int TOPK = 6;
constexpr int NPOS = TOPK + 1;  // D positions per token: 6 routed (ascending id) + shared
constexpr int NEXP = 384;
constexpr int MAXM = 8;
constexpr int MAXSLOT = MAXM * TOPK + 1;  // slot 0 = shared, 1.. = distinct routed experts ascending
constexpr int GU_TILES = I / 8;           // decode gate/up items per expert: 8 up + 8 gate rows (one m16 A tile)
constexpr int DN_TILES = H / 16;          // decode down items per expert: 16 w2 rows
constexpr float LIMIT = 10.f;

// ------------------------------------------------------------------------------------------------ device helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp16z(void* s, const void* g, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(s)), "l"(g), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp4z(void* s, const void* g, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(s)), "l"(g), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

__device__ __forceinline__ uint32_t swz(uint32_t n, uint32_t kb, uint32_t sf_cols) {
  // FlashInfer block_scale_interleave (128x4) byte offset of scale (row n, k-block kb) inside one expert.
  return (kb & 3) + (kb >> 2) * 512 + (n & 31) * 16 + ((n & 127) >> 5) * 4 + (n >> 7) * 128 * sf_cols;
}

__device__ __forceinline__ void expand_fp4x8(uint32_t w, uint32_t& lo, uint32_t& hi) {
  // 8 packed e2m1 nibbles (k 0..7, low nibble = even k) -> two words of 8-bit containers (value in bits [5:2]):
  // lo = k 0..3, hi = k 4..7.
  const uint32_t t = (w << 2) & 0x3C3C3C3Cu;  // even k
  const uint32_t u = (w >> 2) & 0x3C3C3C3Cu;  // odd k
  lo = __byte_perm(t, u, 0x5140);
  hi = __byte_perm(t, u, 0x7362);
}

// One block-scaled m16n8k32 MMA from a zero accumulator. A = weights (16 rows), B = tokens (8 columns).
// BID selects the byte of the packed 4-k-block scale words.
template <bool FP8W, int BID>
__device__ __forceinline__ void mma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  const float z = 0.f;
  if constexpr (FP8W) {
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {%12,%13}, {%14}, {%15,%16};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(z), "r"(sfa), "n"(BID), "n"(0), "r"(sfb),
          "n"(BID), "n"(0));
  } else {
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {%12,%13}, {%14}, {%15,%16};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(z), "r"(sfa), "n"(BID), "n"(0), "r"(sfb),
          "n"(BID), "n"(0));
  }
}

// Same MMA accumulating into d (the k-blocks of one aligned group of four are chained in the tensor core).
template <bool FP8W, int BID>
__device__ __forceinline__ void mmac(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  if constexpr (FP8W) {
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "n"(BID), "n"(0), "r"(sfb), "n"(BID), "n"(0));
  } else {
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "n"(BID), "n"(0), "r"(sfb), "n"(BID), "n"(0));
  }
}
// k-block c of an aligned group of four: c % 4 == 0 starts from zero, the others chain into d.
template <bool FP8W, int C4>
__device__ __forceinline__ void mma_g(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1, uint32_t sfa, uint32_t sfb) {
  if constexpr (C4 == 0) mma0<FP8W, 0>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
  else mmac<FP8W, C4>(d, a0, a1, a2, a3, b0, b1, sfa, sfb);
}

// Official act_quant scale exponent: s = 2^e, e = ceil(log2(max(amax, 1e-4) * (1/448))) (fast_log2_ceil bit trick).
__device__ __forceinline__ int quant_exp(float amax) {
  const float r = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
  const uint32_t b = __float_as_uint(r);
  return static_cast<int>((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) ? 1 : 0);
}
__device__ __forceinline__ float pow2_neg(int e) { return __uint_as_float(static_cast<uint32_t>(127 - e) << 23); }
__device__ __forceinline__ uint32_t pack_e4m3(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// inference/model.py Expert: gate/up bf16 -> fp32 clamp -> silu(gate) * up -> * route weight (fp32). Returns fp32
// (the caller rounds to bf16).
__device__ __forceinline__ float swiglu(float gate_acc, float up_acc, float w) {
  float g = bf16r(gate_acc);
  float u = bf16r(up_acc);
  u = fminf(fmaxf(u, -LIMIT), LIMIT);
  g = fminf(g, LIMIT);
  const float h = (g / (1.0f + expf(-g))) * u;
  return w * h;
}

// ------------------------------------------------------------------------------------------------ weights
struct Weights {
  const uint8_t* w13;     // [E][2I][H/2]
  const uint8_t* w13_sf;  // [E][2I*KB1] swizzled
  const uint8_t* w2;      // [E][H][I/2]
  const uint8_t* w2_sf;   // [E][H*KB2] swizzled
  const uint8_t* s13;     // shared gate_up [rows][H] e4m3
  const uint8_t* s13_sf;  // [rows/32][KB1]
  const uint8_t* s2;      // shared down [H][I] e4m3
  const uint8_t* s2_sf;   // [H/32][KB2]
  int s_up0, s_gate0;     // shared: first row of up / gate inside s13
};

// ------------------------------------------------------------------------------------------------ act quant
// One 32-group of bf16 -> 32 e4m3 bytes + exponent byte (e + 127).
__device__ __forceinline__ void quant_group32(const __nv_bfloat16* src, uint8_t* dst_q, uint8_t* dst_s) {
  float v[32];
  const uint4* s4 = reinterpret_cast<const uint4*>(src);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 u = s4[i];
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[i * 8 + j * 2] = __uint_as_float(w[j] << 16);
      v[i * 8 + j * 2 + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const int e = quant_exp(amax);
  const float inv = pow2_neg(e);
  uint4 q0, q1;
  q0.x = pack_e4m3(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
  q0.y = pack_e4m3(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv);
  q0.z = pack_e4m3(v[8] * inv, v[9] * inv, v[10] * inv, v[11] * inv);
  q0.w = pack_e4m3(v[12] * inv, v[13] * inv, v[14] * inv, v[15] * inv);
  q1.x = pack_e4m3(v[16] * inv, v[17] * inv, v[18] * inv, v[19] * inv);
  q1.y = pack_e4m3(v[20] * inv, v[21] * inv, v[22] * inv, v[23] * inv);
  q1.z = pack_e4m3(v[24] * inv, v[25] * inv, v[26] * inv, v[27] * inv);
  q1.w = pack_e4m3(v[28] * inv, v[29] * inv, v[30] * inv, v[31] * inv);
  reinterpret_cast<uint4*>(dst_q)[0] = q0;
  reinterpret_cast<uint4*>(dst_q)[1] = q1;
  *dst_s = static_cast<uint8_t>(e + 127);
}

// x [M][H] bf16 -> xq [M][H] e4m3, xs [M][KB1]. grid (M), block 160 (one thread per 32-group).
__global__ void __launch_bounds__(160) quant_x_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ xq,
                                                     uint8_t* __restrict__ xs) {
  const int r = blockIdx.x, gi = threadIdx.x;
  quant_group32(x + static_cast<size_t>(r) * H + gi * 32, xq + static_cast<size_t>(r) * H + gi * 32,
                xs + static_cast<size_t>(r) * KB1 + gi);
}

// ================================================================================================ decode (M <= 8)
struct DecodeWork {
  int m;       // valid tokens (rows >= m are padding: zero output)
  int nslot;   // 1 (shared) + distinct routed experts
  int queue;   // work counter
  int pad0;
  int slot_expert[MAXSLOT];
  int slot_pos[MAXSLOT][MAXM];    // D position of (slot, token) or -1
  float slot_w[MAXSLOT][MAXM];    // route weight (shared: 1)
  uint32_t slot_mask[MAXSLOT];    // tokens routed to the slot
  int gu_done[MAXSLOT];           // quantized 32-feature groups of the intermediate (KB2 = done)
  int grp_done[MAXSLOT][KB2];     // gate/up items finished per group (4 = complete)
  int c_done[DN_TILES];
};

// Routing (block 0) + x quant (blocks 1..M). ids/w: [M][6].
__global__ void __launch_bounds__(256) decode_prep(const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ ids,
                                                   const float* __restrict__ tw, const int32_t* __restrict__ valid,
                                                   int M, DecodeWork* __restrict__ wk, uint8_t* __restrict__ xq,
                                                   uint8_t* __restrict__ xs) {
  pdl_wait();  // x / ids come from the previous kernels
  pdl_trigger();
  // Valid rows: below *valid (the step runner's padding) and before the first row SGLang masked (ids -1).
  int m = min(M, *valid);
  for (int r = 0; r < m; ++r) {
    bool neg = false;
    for (int j = 0; j < TOPK; ++j) neg |= ids[r * TOPK + j] < 0;
    if (neg) { m = r; break; }
  }
  if (blockIdx.x > 0) {
    const int r = blockIdx.x - 1;
    if (r < m && threadIdx.x < KB1) {
      quant_group32(x + static_cast<size_t>(r) * H + threadIdx.x * 32, xq + static_cast<size_t>(r) * H + threadIdx.x * 32,
                    xs + static_cast<size_t>(r) * KB1 + threadIdx.x);
    }
    return;
  }
  __shared__ int s_e[MAXM * TOPK];
  __shared__ int s_first[MAXM * TOPK];
  __shared__ int s_slot[MAXM * TOPK];
  const int tid = threadIdx.x;
  const int np = m * TOPK;
  for (int i = tid; i < MAXSLOT * MAXM; i += blockDim.x) {
    (&wk->slot_pos[0][0])[i] = -1;
    (&wk->slot_w[0][0])[i] = 0.f;
  }
  for (int i = tid; i < MAXSLOT; i += blockDim.x) {
    wk->gu_done[i] = 0;
    wk->slot_mask[i] = 0u;
  }
  for (int i = tid; i < MAXSLOT * KB2; i += blockDim.x) (&wk->grp_done[0][0])[i] = 0;
  for (int i = tid; i < DN_TILES; i += blockDim.x) wk->c_done[i] = 0;
  if (tid < np) s_e[tid] = ids[tid];
  __syncthreads();
  if (tid < np) {
    const int e = s_e[tid];
    int first = 1;
    for (int j = 0; j < tid; ++j) first &= (s_e[j] != e);
    s_first[tid] = first;
  }
  __syncthreads();
  if (tid < np) {
    const int e = s_e[tid], t = tid / TOPK;
    int rank = 0, nd = 0;
    for (int j = 0; j < np; ++j) {
      rank += (s_first[j] && s_e[j] < e);
      nd += s_first[j];
    }
    int pos = 0;
    for (int j = 0; j < TOPK; ++j) pos += (s_e[t * TOPK + j] < e);
    const int slot = 1 + rank;
    s_slot[tid] = slot;
    if (s_first[tid]) wk->slot_expert[slot] = e;
    wk->slot_pos[slot][t] = pos;
    wk->slot_w[slot][t] = tw[tid];
    if (tid == 0) {
      wk->nslot = 1 + nd;
      wk->m = m;
      wk->queue = 0;
      wk->slot_expert[0] = -1;
    }
  }
  if (np == 0 && tid == 0) {
    wk->nslot = 1;
    wk->m = 0;
    wk->queue = 0;
    wk->slot_expert[0] = -1;
  }
  if (tid < m) {
    wk->slot_pos[0][tid] = TOPK;
    wk->slot_w[0][tid] = 1.f;
  }
  __syncthreads();
  if (tid < MAXSLOT) {
    uint32_t mask = 0;
    if (tid == 0) mask = (m >= 32) ? 0xffffffffu : ((1u << m) - 1u);
    for (int j = 0; j < np; ++j)
      if (s_slot[j] == tid) mask |= 1u << (j / TOPK);
    wk->slot_mask[tid] = mask;
  }
}

constexpr int D_WARPS = 4;
constexpr int D_NST = 4;          // cp.async stages per warp
constexpr int D_RS4 = 144;        // weight smem row stride FP4 (8 k-blocks x 16 B + pad)
constexpr int D_RS8 = 160;        // weight smem row stride FP8 (4 k-blocks x 32 B + pad)
constexpr int D_W = 16 * D_RS8;   // weight bytes per stage (max)
constexpr int D_RB4 = 8 * 32 + 32;  // token smem row stride FP4 stage (8 k-blocks)
constexpr int D_RB8 = 4 * 32 + 32;  // FP8 stage (4 k-blocks)
constexpr int D_B = MAXM * D_RB4;
constexpr int D_STAGE = D_W + D_B + 16 * 8 + MAXM * 8;  // + weight scales (2 words/row) + token scales
constexpr int D_WARP_SMEM = D_NST * D_STAGE;
constexpr int D_SMEM = D_WARPS * D_WARP_SMEM;

struct DecodeBufs {
  const uint8_t* xq;      // [MAXM][H]
  const uint8_t* xs;      // [MAXM][KB1]
  __nv_bfloat16* hbuf;    // [MAXSLOT][MAXM][I] bf16 intermediate
  uint8_t* hq;            // [MAXSLOT][MAXM][I] its act_quant
  uint8_t* hs;            // [MAXSLOT][MAXM][KB2]
  __nv_bfloat16* dbuf;    // [MAXM][NPOS][H]
  __nv_bfloat16* out;     // [M][H]
  int M;                  // output rows
};

// Stream one A tile (16 weight rows x nkb k-blocks) and the B tokens (bit set in tmask) through the warp's cp.async
// ring; promoted block MMAs into acc.
template <bool FP8W, typename RowPtr, typename SfPtr>
__device__ __forceinline__ void decode_stream(uint8_t* ring, int nkb, uint32_t tmask, RowPtr row_ptr, SfPtr sf_ptr,
                                              const uint8_t* bsrc, int bstride, const uint8_t* bsf, int bsf_stride,
                                              float (&acc)[4]) {
  constexpr int KBS = FP8W ? 4 : 8;  // k-blocks per stage
  constexpr int BPK = FP8W ? 32 : 16;
  constexpr int RS = FP8W ? D_RS8 : D_RS4;
  constexpr int RB = FP8W ? D_RB8 : D_RB4;
  constexpr int CPT = KBS * 2;  // 16-byte token chunks per stage
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nst = (nkb + KBS - 1) / KBS;
  auto issue = [&](int st) {
    uint8_t* dst = ring + (st % D_NST) * D_STAGE;
    const int kb0 = st * KBS;
    const int kbn = min(KBS, nkb - kb0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = lane + 32 * i, r = q >> 3, c = q & 7;
      if (c * 16 < kbn * BPK) cp16(dst + r * RS + c * 16, row_ptr(r) + kb0 * BPK + c * 16);
    }
#pragma unroll
    for (int i = 0; i < (MAXM * CPT) / 32; ++i) {
      const int q = lane + 32 * i, tok = q / CPT, c = q % CPT;
      if (((tmask >> tok) & 1u) && c < kbn * 2)
        cp16(dst + D_W + tok * RB + c * 16, bsrc + static_cast<size_t>(tok) * bstride + kb0 * 32 + c * 16);
    }
    {
      const int r = lane >> 1, wi = lane & 1;
      if (wi * 4 < kbn) cp4(dst + D_W + D_B + r * 8 + wi * 4, sf_ptr(r, kb0 + wi * 4));
      const int tok = lane >> 1;
      if (lane < 2 * MAXM && ((tmask >> tok) & 1u) && wi * 4 < kbn)
        cp4(dst + D_W + D_B + 128 + tok * 8 + wi * 4, bsf + static_cast<size_t>(tok) * bsf_stride + kb0 + wi * 4);
    }
    cp_commit();
  };
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] = 0.f;
#pragma unroll
  for (int p = 0; p < D_NST - 1; ++p) {
    if (p < nst) issue(p);
    else cp_commit();
  }
  const bool tv = (tmask >> g) & 1u;
  for (int st = 0; st < nst; ++st) {
    cp_wait<D_NST - 2>();
    __syncwarp();
    if (st + D_NST - 1 < nst) issue(st + D_NST - 1);
    else cp_commit();
    const uint8_t* src = ring + (st % D_NST) * D_STAGE;
    const uint32_t* wsf = reinterpret_cast<const uint32_t*>(src + D_W + D_B);
    const uint32_t* bsw = wsf + 32;
    const int kb0 = st * KBS;
    const int kbn = min(KBS, nkb - kb0);
    const int arow = g + (t & 1) * 8;
    float d[4];
#pragma unroll
    for (int c = 0; c < KBS; ++c) {
      if (c < kbn) {
        uint32_t a0, a1, a2, a3;
        if constexpr (FP8W) {
          const uint2 v0 = *reinterpret_cast<const uint2*>(src + g * RS + c * BPK + 8 * t);
          const uint2 v1 = *reinterpret_cast<const uint2*>(src + (g + 8) * RS + c * BPK + 8 * t);
          a0 = v0.x; a2 = v0.y; a1 = v1.x; a3 = v1.y;
        } else {
          expand_fp4x8(*reinterpret_cast<const uint32_t*>(src + g * RS + c * BPK + 4 * t), a0, a2);
          expand_fp4x8(*reinterpret_cast<const uint32_t*>(src + (g + 8) * RS + c * BPK + 4 * t), a1, a3);
        }
        const uint32_t sfa = wsf[arow * 2 + c / 4];
        const uint2 bv = *reinterpret_cast<const uint2*>(src + D_W + g * RB + c * 32 + 8 * t);
        const uint32_t b0 = tv ? bv.x : 0u, b1 = tv ? bv.y : 0u;
        const uint32_t sfb = tv ? bsw[g * 2 + c / 4] : 0x7F7F7F7Fu;
        switch (c & 3) {
          case 0: mma_g<FP8W, 0>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
          case 1: mma_g<FP8W, 1>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
          case 2: mma_g<FP8W, 2>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
          default: mma_g<FP8W, 3>(d, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
        }
        if ((c & 3) == 3) {
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] += d[i];
        }
      }
    }
  }
  cp_wait<0>();
  __syncwarp();
}

template <bool FP8W>
__device__ __forceinline__ void decode_gateup(uint8_t* ring, const Weights& W, const DecodeBufs& B,
                                              DecodeWork* wk, int s, int f) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int e = wk->slot_expert[s];
  const uint32_t tmask = wk->slot_mask[s];
  const uint8_t* wbase;
  const uint8_t* sfbase;
  size_t rstride;
  int up0, gate0;
  if constexpr (FP8W) {
    wbase = W.s13; sfbase = W.s13_sf; rstride = H; up0 = W.s_up0; gate0 = W.s_gate0;
  } else {
    wbase = W.w13 + static_cast<size_t>(e) * (2 * I) * (H / 2);
    sfbase = W.w13_sf + static_cast<size_t>(e) * (2 * I) * KB1;
    rstride = H / 2; up0 = 0; gate0 = I;
  }
  auto row_n = [&](int r) { return (r < 8 ? up0 : gate0) + f * 8 + (r & 7); };
  auto row_ptr = [&](int r) { return wbase + static_cast<size_t>(row_n(r)) * rstride; };
  auto sf_ptr = [&](int r, int kb) -> const uint8_t* {
    const int n = row_n(r);
    if constexpr (FP8W) return sfbase + (n >> 5) * KB1 + kb;
    else return sfbase + swz(n, kb, KB1);
  };
  float acc[4];
  decode_stream<FP8W>(ring, KB1, tmask, row_ptr, sf_ptr, B.xq, H, B.xs, KB1, acc);
  // acc: 0 = up(feature f*8+g, token 2t), 1 = up(2t+1), 2 = gate(2t), 3 = gate(2t+1)
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int tok = 2 * t + j;
    if ((tmask >> tok) & 1u) {
      const float h = swiglu(acc[2 + j], acc[j], wk->slot_w[s][tok]);
      B.hbuf[(static_cast<size_t>(s) * MAXM + tok) * I + f * 8 + g] = __float2bfloat16_rn(h);
    }
  }
  // The last of the four items of a 32-feature group quantizes it (official act_quant) for the down items.
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) last = (atomicAdd(&wk->grp_done[s][f >> 2], 1) == 3);
  last = __shfl_sync(0xffffffffu, last, 0);
  if (last) {
    __threadfence();
    const int q = f >> 2;
    for (int tok = 0; tok < MAXM; ++tok) {
      if (!((tmask >> tok) & 1u)) continue;
      const size_t row = static_cast<size_t>(s) * MAXM + tok;
      const float v = __bfloat162float(__ldcg(B.hbuf + row * I + q * 32 + lane));
      float amax = fabsf(v);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      const int ex = quant_exp(amax);
      B.hq[row * I + q * 32 + lane] =
          static_cast<uint8_t>(__nv_cvt_float_to_fp8(v * pow2_neg(ex), __NV_SATFINITE, __NV_E4M3));
      if (lane == 0) B.hs[row * KB2 + q] = static_cast<uint8_t>(ex + 127);
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) atomicAdd(&wk->gu_done[s], 1);
  }
}

template <bool FP8W>
__device__ __forceinline__ void decode_down(uint8_t* ring, const Weights& W, const DecodeBufs& B,
                                            const DecodeWork* wk, int s, int ct) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int e = wk->slot_expert[s];
  const uint32_t tmask = wk->slot_mask[s];
  const uint8_t* wbase;
  const uint8_t* sfbase;
  size_t rstride;
  if constexpr (FP8W) {
    wbase = W.s2; sfbase = W.s2_sf; rstride = I;
  } else {
    wbase = W.w2 + static_cast<size_t>(e) * H * (I / 2);
    sfbase = W.w2_sf + static_cast<size_t>(e) * H * KB2;
    rstride = I / 2;
  }
  auto row_ptr = [&](int r) { return wbase + static_cast<size_t>(ct * 16 + r) * rstride; };
  auto sf_ptr = [&](int r, int kb) -> const uint8_t* {
    const int n = ct * 16 + r;
    if constexpr (FP8W) return sfbase + (n >> 5) * KB2 + kb;
    else return sfbase + swz(n, kb, KB2);
  };
  const size_t row0 = static_cast<size_t>(s) * MAXM;
  float acc[4];
  decode_stream<FP8W>(ring, KB2, tmask, row_ptr, sf_ptr, B.hq + row0 * I, I, B.hs + row0 * KB2, KB2, acc);
  // acc: 0 = (row ct*16+g, token 2t), 1 = (.., 2t+1), 2 = (row +8, 2t), 3 = (row +8, 2t+1)
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int tok = 2 * t + j;
    if ((tmask >> tok) & 1u) {
      __nv_bfloat16* d = B.dbuf + (static_cast<size_t>(tok) * NPOS + wk->slot_pos[s][tok]) * H + ct * 16 + g;
      d[0] = __float2bfloat16_rn(acc[j]);
      d[8] = __float2bfloat16_rn(acc[2 + j]);
    }
  }
}

#ifdef OG_TRACE
__device__ unsigned long long* g_trace;  // per warp: [tk0, t_wait, (item << 40 | t0), t1, ...]
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#endif

__global__ void __launch_bounds__(D_WARPS * 32, 1) decode_main(Weights W, DecodeBufs B, DecodeWork* __restrict__ wk) {
  extern __shared__ __align__(16) uint8_t dsmem[];
#ifdef OG_TRACE
  const unsigned long long tk0 = gtime();
  unsigned long long* tr = g_trace + (static_cast<size_t>(blockIdx.x) * D_WARPS + (threadIdx.x >> 5)) * 130;
  int ntr = 0;
  if ((threadIdx.x & 31) == 0) tr[0] = tk0;
#endif
  pdl_wait();
  pdl_trigger();
#ifdef OG_TRACE
  if ((threadIdx.x & 31) == 0) tr[1] = gtime();
#endif
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint8_t* ring = dsmem + warp * D_WARP_SMEM;
  const int m = wk->m;
  const int nslot = wk->nslot;
  const int n_gu = nslot * GU_TILES;
  const int n_total = n_gu + nslot * DN_TILES;
  for (;;) {
    int item = 0;
    if (lane == 0) item = atomicAdd(&wk->queue, 1);
    item = __shfl_sync(0xffffffffu, item, 0);
    if (item >= n_total) break;
#ifdef OG_TRACE
    const unsigned long long ti0 = gtime();
#endif
    if (item < n_gu) {
      // gate/up: shared slot first (its FP8 items are twice as long), then routed ascending
      const int s = item / GU_TILES, f = item % GU_TILES;
      if (s == 0) decode_gateup<true>(ring, W, B, wk, s, f);
      else decode_gateup<false>(ring, W, B, wk, s, f);
    } else {
      // down: routed slots first, shared last
      const int j = item - n_gu;
      const int s = (j / DN_TILES + 1) % nslot, ct = j % DN_TILES;
      if (lane == 0) {
        while (*reinterpret_cast<volatile const int*>(&wk->gu_done[s]) < KB2) __nanosleep(32);
      }
      __syncwarp();
      __threadfence();
      if (s == 0) decode_down<true>(ring, W, B, wk, s, ct);
      else decode_down<false>(ring, W, B, wk, s, ct);
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) last = (atomicAdd(&wk->c_done[ct], 1) == nslot - 1);
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        __threadfence();
        for (int idx = lane; idx < B.M * 16; idx += 32) {
          const int tok = idx >> 4, c = ct * 16 + (idx & 15);
          float v = 0.f;
          if (tok < m) {
#pragma unroll
            for (int p = 0; p < NPOS; ++p)
              v += __bfloat162float(__ldcg(B.dbuf + (static_cast<size_t>(tok) * NPOS + p) * H + c));
          }
          B.out[static_cast<size_t>(tok) * H + c] = __float2bfloat16_rn(v);
        }
      }
    }
#ifdef OG_TRACE
    if (lane == 0 && ntr < 64) {
      tr[2 + 2 * ntr] = (static_cast<unsigned long long>(item) << 40) | (ti0 - tk0);
      tr[3 + 2 * ntr] = gtime() - tk0;
    }
    ++ntr;
#endif
  }
}

// ================================================================================================ prefill (any M)
// CTA tile: 128 weight rows x 128 tokens, 8 warps = 4 (32 weight rows: two m16 A tiles) x 2 (token n8 tiles
// jj = 2j + wt). gate/up: rows 0..63 up, 64..127 gate of the same 64 features (a warp holds up and gate of its 16
// features); down: 128 w2 rows. Stages of 4 k-blocks (one promotion group), XOR-swizzled unpadded smem.
#ifndef OG_P_TW
#define OG_P_TW 2
#endif
constexpr int P_TW = OG_P_TW;         // token warp groups (n8 tiles jj = P_TW*j + wt)
constexpr int P_THREADS = 128 * P_TW;  // 4 row-warps x P_TW token-warps
constexpr int P_ROWS = 128;
constexpr int P_TOK = 128;
constexpr int P_KBS = 4;
constexpr int P_NJ = 16 / P_TW;

template <bool FP8W>
struct PfCfg {
  static constexpr int BPK = FP8W ? 32 : 16;
  static constexpr int WRB = P_KBS * BPK;  // weight bytes per row per stage (64 / 128)
  static constexpr int NST = 3;
  static constexpr int W_BYTES = P_ROWS * WRB;
  static constexpr int X_BYTES = P_TOK * P_KBS * 32;
  static constexpr int STAGE = W_BYTES + X_BYTES + P_ROWS * 4 + P_TOK * 4;
  static constexpr int SMEM = NST * STAGE;
  static constexpr int WCH = WRB / 16;                      // 16-byte chunks per weight row
  static constexpr int WPT = P_ROWS * WCH / P_THREADS;      // weight chunks per thread per stage
  static constexpr int XPT = P_TOK * 8 / P_THREADS;         // token chunks per thread per stage
};
// byte offset of logical 16-byte chunk q of a row (swizzles keep the fragment reads bank-conflict free)
__device__ __forceinline__ int wchunk4(int r, int q) { return r * 64 + ((q ^ ((r >> 1) & 3)) << 4); }
__device__ __forceinline__ int chunk8(int r, int q) { return r * 128 + ((q ^ ((r & 3) << 1)) << 4); }

struct PfParams {
  Weights W;
  const uint8_t* xq;      // [M][H]
  const uint8_t* xs;      // [M][KB1]
  uint8_t* hq;            // [P][I]  (P = M * 7 sorted pairs)
  uint8_t* hsf;           // [P][KB2]
  __nv_bfloat16* dbuf;    // [M][NPOS][H]
  const int32_t* p_tok;   // [P] token of sorted pair
  const float* p_w;       // [P]
  const int32_t* p_pos;   // [P]
  const int32_t* tile_e;  // [max_tiles]: shared tiles first, then routed ascending
  const int32_t* tile_start;
  const int32_t* tile_cnt;
  const int32_t* n_tiles;
};

// One stage = one promotion group of 4 k-blocks: A tiles x = 0, 1 (rows R0, R1) x NJ token tiles jj = 2j + wt.
template <bool FP8W, int NJ>
__device__ __forceinline__ void pf_stage(const uint8_t* S, int R0, int R1, int wt, int g, int t,
                                         float (&acc)[2][P_NJ][4]) {
  using C = PfCfg<FP8W>;
  const uint8_t* Sw = S;
  const uint8_t* Sx = S + C::W_BYTES;
  const uint32_t* Swsf = reinterpret_cast<const uint32_t*>(Sx + C::X_BYTES);
  const uint32_t* Sxsf = Swsf + P_ROWS;
  const uint32_t sfa0 = Swsf[R0 + g + (t & 1) * 8];
  const uint32_t sfa1 = Swsf[R1 + g + (t & 1) * 8];
  uint32_t sfb[NJ];
#pragma unroll
  for (int j = 0; j < NJ; ++j) sfb[j] = Sxsf[(P_TW * j + wt) * 8 + g];
  float d[2][NJ][4];
#pragma unroll
  for (int c = 0; c < P_KBS; ++c) {
    uint32_t a[2][4];
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      const int R = (x ? R1 : R0) + g;
      if constexpr (FP8W) {
        const int q = 2 * c + (t >> 1), o = 8 * (t & 1);
        const uint2 v0 = *reinterpret_cast<const uint2*>(Sw + chunk8(R, q) + o);
        const uint2 v1 = *reinterpret_cast<const uint2*>(Sw + chunk8(R + 8, q) + o);
        a[x][0] = v0.x; a[x][2] = v0.y; a[x][1] = v1.x; a[x][3] = v1.y;
      } else {
        expand_fp4x8(*reinterpret_cast<const uint32_t*>(Sw + wchunk4(R, c) + 4 * t), a[x][0], a[x][2]);
        expand_fp4x8(*reinterpret_cast<const uint32_t*>(Sw + wchunk4(R + 8, c) + 4 * t), a[x][1], a[x][3]);
      }
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const uint2 bv =
          *reinterpret_cast<const uint2*>(Sx + chunk8((P_TW * j + wt) * 8 + g, 2 * c + (t >> 1)) + 8 * (t & 1));
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        const uint32_t sa = x ? sfa1 : sfa0;
        switch (c) {
          case 0: mma_g<FP8W, 0>(d[x][j], a[x][0], a[x][1], a[x][2], a[x][3], bv.x, bv.y, sa, sfb[j]); break;
          case 1: mma_g<FP8W, 1>(d[x][j], a[x][0], a[x][1], a[x][2], a[x][3], bv.x, bv.y, sa, sfb[j]); break;
          case 2: mma_g<FP8W, 2>(d[x][j], a[x][0], a[x][1], a[x][2], a[x][3], bv.x, bv.y, sa, sfb[j]); break;
          default: mma_g<FP8W, 3>(d[x][j], a[x][0], a[x][1], a[x][2], a[x][3], bv.x, bv.y, sa, sfb[j]); break;
        }
      }
    }
  }
#pragma unroll
  for (int x = 0; x < 2; ++x)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[x][j][i] += d[x][j][i];
}

// DOWN = false: gate/up for 64 features of one expert tile of <= 128 tokens; epilogue SwiGLU * w -> bf16 ->
// act_quant -> hq/hsf rows of the sorted pairs. DOWN = true: 128 w2 rows; epilogue fp32 D[token][pos][c].
template <bool FP8W, bool DOWN>
__global__ void __launch_bounds__(P_THREADS, 1) pf_gemm(PfParams p, int tile0) {
  using C = PfCfg<FP8W>;
  extern __shared__ __align__(16) uint8_t psmem[];
  const int tile = tile0 + blockIdx.y;
  if (tile >= *p.n_tiles) return;
  const int e = p.tile_e[tile];
  if ((e == NEXP) != FP8W) return;
  const int t0 = p.tile_start[tile], cnt = p.tile_cnt[tile];
  const int nv8 = (cnt + 7) >> 3;
  const int bx = blockIdx.x;  // gate/up: 64-feature tile (0..17); down: 128-row c tile (0..39)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wr = warp & 3, wt = warp >> 2;
  const int nvw = (nv8 + P_TW - 1 - wt) / P_TW;  // this warp's token tiles jj = P_TW*j + wt
  constexpr int NKB = DOWN ? KB2 : KB1;
  constexpr int NSTG = NKB / P_KBS;
  constexpr int SFC = DOWN ? KB2 : KB1;

  const uint8_t* wbase;
  const uint8_t* sfbase;
  size_t rstride;
  if constexpr (DOWN) {
    if constexpr (FP8W) { wbase = p.W.s2; sfbase = p.W.s2_sf; rstride = I; }
    else {
      wbase = p.W.w2 + static_cast<size_t>(e) * H * (I / 2);
      sfbase = p.W.w2_sf + static_cast<size_t>(e) * H * KB2;
      rstride = I / 2;
    }
  } else {
    if constexpr (FP8W) { wbase = p.W.s13; sfbase = p.W.s13_sf; rstride = H; }
    else {
      wbase = p.W.w13 + static_cast<size_t>(e) * (2 * I) * (H / 2);
      sfbase = p.W.w13_sf + static_cast<size_t>(e) * (2 * I) * KB1;
      rstride = H / 2;
    }
  }
  auto row_n = [&](int r) -> int {
    if constexpr (DOWN) return bx * 128 + r;
    else {
      const int up0 = FP8W ? p.W.s_up0 : 0, gate0 = FP8W ? p.W.s_gate0 : I;
      return (r < 64 ? up0 : gate0) + bx * 64 + (r & 63);
    }
  };
  // Per-thread copy sources (advanced by kb0 each stage) and smem destinations.
  const uint8_t* wsrc[C::WPT];
  int wdst[C::WPT];
#pragma unroll
  for (int i = 0; i < C::WPT; ++i) {
    const int q = tid + P_THREADS * i, r = q / C::WCH, c = q % C::WCH;
    wsrc[i] = wbase + static_cast<size_t>(row_n(r)) * rstride + c * 16;
    wdst[i] = FP8W ? chunk8(r, c) : wchunk4(r, c);
  }
  const int nload = nv8 * 8;  // token rows staged (both warp halves read them; rows >= cnt are zero-filled)
  const uint8_t* xsrc[C::XPT];
  int xdst[C::XPT];
  bool xval[C::XPT], xload[C::XPT];
#pragma unroll
  for (int i = 0; i < C::XPT; ++i) {
    const int q = tid + P_THREADS * i, r = q >> 3, c = q & 7;
    xval[i] = r < cnt;
    xload[i] = r < nload;
    const int rr = xval[i] ? r : 0;
    if constexpr (DOWN) xsrc[i] = p.hq + static_cast<size_t>(t0 + rr) * I + c * 16;
    else xsrc[i] = p.xq + static_cast<size_t>(xval[i] ? p.p_tok[t0 + rr] : 0) * H + c * 16;
    xdst[i] = C::W_BYTES + chunk8(r, c);
  }
  const uint8_t* ssrc = nullptr;  // scale words: threads 0..127 weight rows, 128..255 token rows
  int sdst = 0;
  bool sval = false, sload = false;
  if (tid >= P_ROWS + P_TOK) {
  } else if (tid < P_ROWS) {
    const int n = row_n(tid);
    ssrc = FP8W ? sfbase + (n >> 5) * SFC : sfbase + swz(n, 0, SFC);
    sdst = C::W_BYTES + C::X_BYTES + tid * 4;
    sval = sload = true;
  } else {
    const int r = tid - P_ROWS;
    sval = r < cnt;
    sload = r < nload;
    const int rr = sval ? r : 0;
    if constexpr (DOWN) ssrc = p.hsf + static_cast<size_t>(t0 + rr) * KB2;
    else ssrc = p.xs + static_cast<size_t>(sval ? p.p_tok[t0 + rr] : 0) * KB1;
    sdst = C::W_BYTES + C::X_BYTES + P_ROWS * 4 + r * 4;
  }
  const bool wscale = tid < P_ROWS;

  auto issue = [&](int st) {
    uint8_t* S = psmem + (st % C::NST) * C::STAGE;
    const int kb0 = st * P_KBS;
#pragma unroll
    for (int i = 0; i < C::WPT; ++i) cp16(S + wdst[i], wsrc[i] + kb0 * C::BPK);
#pragma unroll
    for (int i = 0; i < C::XPT; ++i)
      if (xload[i]) cp16z(S + xdst[i], xsrc[i] + kb0 * 32, xval[i]);
    if (sload) {
      // swizzled scales: 4 consecutive k-blocks are contiguous and groups of 4 are 512 bytes apart
      const uint8_t* src = (wscale && !FP8W) ? ssrc + (kb0 >> 2) * 512 : ssrc + kb0;
      cp4z(S + sdst, src, sval);
    }
    cp_commit();
  };

  const int R0 = DOWN ? wr * 32 : wr * 16;
  const int R1 = DOWN ? wr * 32 + 16 : 64 + wr * 16;
  float acc[2][P_NJ][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int j = 0; j < P_NJ; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[a][j][i] = 0.f;

#pragma unroll
  for (int s = 0; s < C::NST - 1; ++s) issue(s);
  for (int st = 0; st < NSTG; ++st) {
    cp_wait<C::NST - 2>();
    __syncthreads();
    if (st + C::NST - 1 < NSTG) issue(st + C::NST - 1);
    else cp_commit();
    const uint8_t* S = psmem + (st % C::NST) * C::STAGE;
    if constexpr (P_NJ == 8) {
      switch (nvw) {
        case 8: pf_stage<FP8W, 8>(S, R0, R1, wt, g, t, acc); break;
        case 7: pf_stage<FP8W, 7>(S, R0, R1, wt, g, t, acc); break;
        case 6: pf_stage<FP8W, 6>(S, R0, R1, wt, g, t, acc); break;
        case 5: pf_stage<FP8W, 5>(S, R0, R1, wt, g, t, acc); break;
        case 4: pf_stage<FP8W, 4>(S, R0, R1, wt, g, t, acc); break;
        case 3: pf_stage<FP8W, 3>(S, R0, R1, wt, g, t, acc); break;
        case 2: pf_stage<FP8W, 2>(S, R0, R1, wt, g, t, acc); break;
        case 1: pf_stage<FP8W, 1>(S, R0, R1, wt, g, t, acc); break;
        default: break;
      }
    } else {
      switch (nvw) {
        case 4: pf_stage<FP8W, 4>(S, R0, R1, wt, g, t, acc); break;
        case 3: pf_stage<FP8W, 3>(S, R0, R1, wt, g, t, acc); break;
        case 2: pf_stage<FP8W, 2>(S, R0, R1, wt, g, t, acc); break;
        case 1: pf_stage<FP8W, 1>(S, R0, R1, wt, g, t, acc); break;
        default: break;
      }
    }
  }
  cp_wait<0>();
  __syncthreads();

  if constexpr (DOWN) {
    // acc[x][j][i]: row bx*128 + (x ? R1 : R0) + g (+8 for i>=2), token row (P_TW*j+wt)*8 + 2t (+1 for odd i)
#pragma unroll
    for (int j = 0; j < P_NJ; ++j) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int r = (P_TW * j + wt) * 8 + 2 * t + k;
        if (r < cnt) {
          const int pi = t0 + r;
          __nv_bfloat16* dd = p.dbuf + (static_cast<size_t>(p.p_tok[pi]) * NPOS + p.p_pos[pi]) * H + bx * 128 + g;
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const int R = x ? R1 : R0;
            dd[R] = __float2bfloat16_rn(acc[x][j][k]);
            dd[R + 8] = __float2bfloat16_rn(acc[x][j][2 + k]);
          }
        }
      }
    }
  } else {
    // SwiGLU in registers (x=0 up, x=1 gate at the same (feature, token)), bf16 into smem [token][64 features]
    __nv_bfloat16* Hs = reinterpret_cast<__nv_bfloat16*>(psmem);
    constexpr int HRS = 64 + 8;
#pragma unroll
    for (int j = 0; j < P_NJ; ++j) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int r = (P_TW * j + wt) * 8 + 2 * t + k;
        if (r < cnt) {
          const float w = p.p_w[t0 + r];
          Hs[r * HRS + wr * 16 + g] = __float2bfloat16_rn(swiglu(acc[1][j][k], acc[0][j][k], w));
          Hs[r * HRS + wr * 16 + g + 8] = __float2bfloat16_rn(swiglu(acc[1][j][2 + k], acc[0][j][2 + k], w));
        }
      }
    }
    __syncthreads();
    // act_quant: (token, 32-feature group) tasks
    for (int task = tid; task < 2 * P_TOK; task += P_THREADS) {
      const int r = task >> 1, gr = task & 1;
      if (r < cnt) {
        const size_t row = static_cast<size_t>(t0 + r);
        quant_group32(Hs + r * HRS + gr * 32, p.hq + row * I + bx * 64 + gr * 32, p.hsf + row * KB2 + bx * 2 + gr);
      }
    }
  }
}

// out[t][c] = bf16(sum_{pos 0..6} D[t][pos][c]) in fp32, in order. grid (M, H / 2048), block 256 (8 columns/thread).
__global__ void __launch_bounds__(256) pf_combine(const __nv_bfloat16* __restrict__ dbuf,
                                                  __nv_bfloat16* __restrict__ out) {
  const int tok = blockIdx.x;
  const int c = (blockIdx.y * 256 + threadIdx.x) * 8;
  if (c >= H) return;
  float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int pp = 0; pp < NPOS; ++pp) {
    const uint4 d = __ldcs(reinterpret_cast<const uint4*>(dbuf + (static_cast<size_t>(tok) * NPOS + pp) * H + c));
    const uint32_t w[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[2 * i] += __uint_as_float(w[i] << 16);
      v[2 * i + 1] += __uint_as_float(w[i] & 0xFFFF0000u);
    }
  }
  uint4 o;
  uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 b = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    ow[i] = *reinterpret_cast<uint32_t*>(&b);
  }
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(tok) * H + c) = o;
}

// Prefill routing: pairs grouped by expert (order inside an expert is irrelevant: rows are independent), the shared
// expert (index NEXP) takes every token. Masked slots (id < 0, SGLang padding) become expert 0 with weight 0.
// counts must be zeroed ([NEXP + 1] ints) before pf_route_count.
__global__ void __launch_bounds__(256) pf_route_count(const int32_t* __restrict__ ids, int M, int32_t* __restrict__ counts,
                                                      int32_t* __restrict__ ppos) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= M * TOPK) return;
  const int t = p / TOPK;
  const int e = max(ids[p], 0);
  int pos = 0;
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    const int ej = max(ids[t * TOPK + j], 0);
    pos += (ej < e) || (ej == e && j < p % TOPK);  // ties only for masked slots
  }
  ppos[p] = pos;
  atomicAdd(&counts[e], 1);
}

// offsets[0..NEXP+1] (routed experts, then the shared expert with M tokens), cursors, and the tile list (shared tiles
// first, then routed experts ascending; each expert's pairs split into ceil(n/128) near-equal tiles). One block.
__global__ void __launch_bounds__(512) pf_route_scan(const int32_t* __restrict__ counts, int M,
                                                     int32_t* __restrict__ offsets, int32_t* __restrict__ cursor,
                                                     int32_t* __restrict__ tile_e, int32_t* __restrict__ tile_start,
                                                     int32_t* __restrict__ tile_cnt, int32_t* __restrict__ n_tiles) {
  __shared__ int s_off[NEXP + 2];
  __shared__ int s_nt[NEXP + 1];
  __shared__ int s_base[NEXP + 1];
  const int tid = threadIdx.x;
  if (tid == 0) {
    int o = 0;
    for (int e = 0; e < NEXP; ++e) { s_off[e] = o; o += counts[e]; }
    s_off[NEXP] = o;
    s_off[NEXP + 1] = o + M;
  }
  __syncthreads();
  if (tid < NEXP + 2) offsets[tid] = s_off[tid];
  if (tid <= NEXP) cursor[tid] = s_off[tid];
  auto expert_of = [](int i) { return i == 0 ? NEXP : i - 1; };  // list order: shared first
  if (tid <= NEXP) {
    const int e = expert_of(tid);
    s_nt[tid] = (s_off[e + 1] - s_off[e] + P_TOK - 1) / P_TOK;
  }
  __syncthreads();
  if (tid == 0) {
    int b = 0;
    for (int i = 0; i <= NEXP; ++i) { s_base[i] = b; b += s_nt[i]; }
    *n_tiles = b;
  }
  __syncthreads();
  if (tid <= NEXP) {
    const int e = expert_of(tid);
    const int o = s_off[e], n = s_off[e + 1] - o, nt = s_nt[tid];
    for (int k = 0; k < nt; ++k) {
      const int a = o + (k * n) / nt, b2 = o + ((k + 1) * n) / nt;
      tile_e[s_base[tid] + k] = e;
      tile_start[s_base[tid] + k] = a;
      tile_cnt[s_base[tid] + k] = b2 - a;
    }
  }
}

__global__ void __launch_bounds__(256) pf_route_scatter(const int32_t* __restrict__ ids, const float* __restrict__ w,
                                                        int M, const int32_t* __restrict__ ppos,
                                                        int32_t* __restrict__ cursor, int32_t* __restrict__ p_tok,
                                                        float* __restrict__ p_w, int32_t* __restrict__ p_pos) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < M * TOPK) {
    const int id = ids[p];
    const int slot = atomicAdd(&cursor[max(id, 0)], 1);
    p_tok[slot] = p / TOPK;
    p_w[slot] = id < 0 ? 0.f : w[p];
    p_pos[slot] = ppos[p];
  } else if (p < M * TOPK + M) {
    const int t = p - M * TOPK;
    const int slot = cursor[NEXP] + t;  // shared: fixed order, cursor[NEXP] = offsets[NEXP] (not advanced)
    p_tok[slot] = t;
    p_w[slot] = 1.f;
    p_pos[slot] = TOPK;
  }
}

// ------------------------------------------------------------------------------------------------ host
static Weights make_weights(const torch::Tensor& w13, const torch::Tensor& w13_sf, const torch::Tensor& w2,
                            const torch::Tensor& w2_sf, const torch::Tensor& s13, const torch::Tensor& s13_sf,
                            const torch::Tensor& s2, const torch::Tensor& s2_sf, int64_t s_up0, int64_t s_gate0) {
  TORCH_CHECK(w13.dim() == 3 && w13.size(1) == 2 * I && w13.size(2) == H / 2, "w13 shape");
  TORCH_CHECK(w13_sf.numel() == w13.size(0) * 2 * I * KB1, "w13_sf size");
  TORCH_CHECK(w2.dim() == 3 && w2.size(1) == H && w2.size(2) == I / 2, "w2 shape");
  TORCH_CHECK(w2_sf.numel() == w2.size(0) * H * KB2, "w2_sf size");
  TORCH_CHECK(s13.dim() == 2 && s13.size(1) == H && s13.size(0) >= 2 * I, "s13 shape");
  TORCH_CHECK(s13_sf.numel() == (s13.size(0) / 32) * KB1, "s13_sf size");
  TORCH_CHECK(s2.dim() == 2 && s2.size(0) == H && s2.size(1) == I, "s2 shape");
  TORCH_CHECK(s2_sf.numel() == (H / 32) * KB2, "s2_sf size");
  for (auto* x : {&w13, &w13_sf, &w2, &w2_sf, &s13, &s13_sf, &s2, &s2_sf})
    TORCH_CHECK(x->is_contiguous() && x->element_size() == 1 && (reinterpret_cast<uintptr_t>(x->data_ptr()) & 15) == 0,
                "weights must be contiguous 16-byte aligned 1-byte tensors");
  Weights W;
  W.w13 = static_cast<const uint8_t*>(w13.data_ptr());
  W.w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr());
  W.w2 = static_cast<const uint8_t*>(w2.data_ptr());
  W.w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr());
  W.s13 = static_cast<const uint8_t*>(s13.data_ptr());
  W.s13_sf = static_cast<const uint8_t*>(s13_sf.data_ptr());
  W.s2 = static_cast<const uint8_t*>(s2.data_ptr());
  W.s2_sf = static_cast<const uint8_t*>(s2_sf.data_ptr());
  W.s_up0 = static_cast<int>(s_up0);
  W.s_gate0 = static_cast<int>(s_gate0);
  return W;
}

template <typename K, typename... Args>
static void launch_pdl(K kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

void set_trace(torch::Tensor buf) {
#ifdef OG_TRACE
  unsigned long long* p = reinterpret_cast<unsigned long long*>(buf.data_ptr());
  C10_CUDA_CHECK(cudaMemcpyToSymbol(g_trace, &p, sizeof(p)));
#endif
}

// Workspace layout (256-byte aligned pieces): DecodeWork | xq | xs | hbuf | hq | hs | dbuf.
struct DecodeLayout {
  size_t work, xq, xs, hbuf, hq, hs, dbuf, total;
  DecodeLayout() {
    auto al = [](size_t v) { return (v + 255) & ~size_t(255); };
    work = 0;
    xq = al(work + sizeof(DecodeWork));
    xs = al(xq + MAXM * H);
    hbuf = al(xs + MAXM * KB1);
    hq = al(hbuf + 2ull * MAXSLOT * MAXM * I);
    hs = al(hq + 1ull * MAXSLOT * MAXM * I);
    dbuf = al(hs + 1ull * MAXSLOT * MAXM * KB2);
    total = al(dbuf + 2ull * MAXM * NPOS * H) + 256;
  }
};

int64_t decode_workspace_bytes() { return static_cast<int64_t>(DecodeLayout().total); }

// Decode: x [M<=8][H] bf16, ids [M][6] int32, tw [M][6] fp32, valid [1] int32 (rows >= valid are padding).
// ws: workspace (decode_workspace_bytes()). Returns bf16 [M][H] (this rank's partial).
torch::Tensor decode(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor valid, torch::Tensor ws,
                     torch::Tensor w13, torch::Tensor w13_sf, torch::Tensor w2, torch::Tensor w2_sf,
                     torch::Tensor s13, torch::Tensor s13_sf, torch::Tensor s2, torch::Tensor s2_sf, int64_t s_up0,
                     int64_t s_gate0, int64_t grid) {
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(M >= 1 && M <= MAXM, "decode M must be 1..8");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == H, "x");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "x alignment");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.size(0) == M && ids.size(1) == TOPK, "ids");
  TORCH_CHECK(tw.scalar_type() == at::kFloat && tw.is_contiguous() && tw.size(0) == M && tw.size(1) == TOPK, "tw");
  TORCH_CHECK(valid.scalar_type() == at::kInt && valid.numel() >= 1, "valid");
  const DecodeLayout L;
  TORCH_CHECK(ws.numel() >= static_cast<int64_t>(L.total), "workspace");
  const at::cuda::CUDAGuard guard(x.device());
  Weights W = make_weights(w13, w13_sf, w2, w2_sf, s13, s13_sf, s2, s2_sf, s_up0, s_gate0);
  uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(ws.data_ptr()) + 255) & ~uintptr_t(255));
  DecodeWork* wk = reinterpret_cast<DecodeWork*>(base + L.work);
  uint8_t* xq = base + L.xq;
  uint8_t* xs = base + L.xs;
  auto out = torch::empty({M, H}, x.options());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  launch_pdl(decode_prep, dim3(1 + M), dim3(256), 0, st, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
             static_cast<const int32_t*>(ids.data_ptr()), static_cast<const float*>(tw.data_ptr()),
             static_cast<const int32_t*>(valid.data_ptr()), M, wk, xq, xs);
  DecodeBufs B;
  B.xq = xq;
  B.xs = xs;
  B.hbuf = reinterpret_cast<__nv_bfloat16*>(base + L.hbuf);
  B.hq = base + L.hq;
  B.hs = base + L.hs;
  B.dbuf = reinterpret_cast<__nv_bfloat16*>(base + L.dbuf);
  B.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  B.M = M;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(decode_main, cudaFuncAttributeMaxDynamicSharedMemorySize, D_SMEM));
    attr_set = true;
  }
  launch_pdl(decode_main, dim3(static_cast<unsigned>(grid)), dim3(D_WARPS * 32), D_SMEM, st, W, B, wk);
  return out;
}

// Prefill: x [M][H] bf16, ids [M][6] int32, tw [M][6] fp32; ws: int8 workspace from prefill_workspace_bytes(M).
// The shared-expert GEMMs run on a side stream concurrently with the routed ones.
struct PfLayout {
  size_t xq, xs, counts, cursor, offsets, ppos, p_tok, p_w, p_pos, tiles, hq, hsf, dbuf, total;
  int max_tiles, n_shared, max_routed;
  explicit PfLayout(int M) {
    auto al = [](size_t v) { return (v + 255) & ~size_t(255); };
    const size_t P = static_cast<size_t>(M) * NPOS;
    n_shared = (M + P_TOK - 1) / P_TOK;
    max_routed = (M * TOPK + P_TOK - 1) / P_TOK + NEXP;
    max_tiles = n_shared + max_routed;
    xq = 0;
    xs = al(xq + static_cast<size_t>(M) * H);
    counts = al(xs + static_cast<size_t>(M) * KB1);
    cursor = al(counts + 4 * (NEXP + 1));
    offsets = al(cursor + 4 * (NEXP + 1));
    ppos = al(offsets + 4 * (NEXP + 2));
    p_tok = al(ppos + 4 * static_cast<size_t>(M) * TOPK);
    p_w = al(p_tok + 4 * P);
    p_pos = al(p_w + 4 * P);
    tiles = al(p_pos + 4 * P);
    hq = al(tiles + 4 * (3 * static_cast<size_t>(max_tiles) + 1));
    hsf = al(hq + P * I);
    dbuf = al(hsf + P * KB2);
    total = al(dbuf + 2 * static_cast<size_t>(M) * NPOS * H) + 256;
  }
};

int64_t prefill_workspace_bytes(int64_t M) { return static_cast<int64_t>(PfLayout(static_cast<int>(M)).total); }

torch::Tensor prefill(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor ws, torch::Tensor w13,
                      torch::Tensor w13_sf, torch::Tensor w2, torch::Tensor w2_sf, torch::Tensor s13,
                      torch::Tensor s13_sf, torch::Tensor s2, torch::Tensor s2_sf, int64_t s_up0, int64_t s_gate0) {
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(M >= 1, "prefill M");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == H, "x");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "x alignment");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.size(0) == M && ids.size(1) == TOPK, "ids");
  TORCH_CHECK(tw.scalar_type() == at::kFloat && tw.is_contiguous() && tw.size(0) == M && tw.size(1) == TOPK, "tw");
  const PfLayout L(M);
  TORCH_CHECK(ws.numel() >= static_cast<int64_t>(L.total), "prefill workspace");
  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  Weights W = make_weights(w13, w13_sf, w2, w2_sf, s13, s13_sf, s2, s2_sf, s_up0, s_gate0);
  uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(ws.data_ptr()) + 255) & ~uintptr_t(255));
  auto I32 = [&](size_t off) { return reinterpret_cast<int32_t*>(base + off); };
  const int32_t* idp = static_cast<const int32_t*>(ids.data_ptr());
  const float* twp = static_cast<const float*>(tw.data_ptr());
  const int np = M * TOPK;
  int32_t* tb = I32(L.tiles);
  const int mt = L.max_tiles;
  C10_CUDA_CHECK(cudaMemsetAsync(base + L.counts, 0, 4 * (NEXP + 1), st));
  pf_route_count<<<(np + 255) / 256, 256, 0, st>>>(idp, M, I32(L.counts), I32(L.ppos));
  pf_route_scan<<<1, 512, 0, st>>>(I32(L.counts), M, I32(L.offsets), I32(L.cursor), tb, tb + mt, tb + 2 * mt,
                                   tb + 3 * mt);
  pf_route_scatter<<<(np + M + 255) / 256, 256, 0, st>>>(idp, twp, M, I32(L.ppos), I32(L.cursor), I32(L.p_tok),
                                                          reinterpret_cast<float*>(base + L.p_w), I32(L.p_pos));
  quant_x_kernel<<<M, 160, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), base + L.xq, base + L.xs);
  PfParams pp;
  pp.W = W;
  pp.xq = base + L.xq;
  pp.xs = base + L.xs;
  pp.hq = base + L.hq;
  pp.hsf = base + L.hsf;
  pp.dbuf = reinterpret_cast<__nv_bfloat16*>(base + L.dbuf);
  pp.p_tok = I32(L.p_tok);
  pp.p_w = reinterpret_cast<const float*>(base + L.p_w);
  pp.p_pos = I32(L.p_pos);
  pp.tile_e = tb;
  pp.tile_start = tb + mt;
  pp.tile_cnt = tb + 2 * mt;
  pp.n_tiles = tb + 3 * mt;
  static bool init = false;
  static cudaStream_t side;
  static cudaEvent_t ev_fork, ev_join;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pf_gemm<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, PfCfg<false>::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(pf_gemm<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, PfCfg<true>::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(pf_gemm<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, PfCfg<false>::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(pf_gemm<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, PfCfg<true>::SMEM));
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
    init = true;
  }
  // shared expert (FP8) on the side stream, routed experts (FP4) here; D positions are disjoint until the combine
  C10_CUDA_CHECK(cudaEventRecord(ev_fork, st));
  C10_CUDA_CHECK(cudaStreamWaitEvent(side, ev_fork, 0));
  pf_gemm<true, false><<<dim3(I / 64, L.n_shared), P_THREADS, PfCfg<true>::SMEM, side>>>(pp, 0);
  pf_gemm<true, true><<<dim3(H / 128, L.n_shared), P_THREADS, PfCfg<true>::SMEM, side>>>(pp, 0);
  C10_CUDA_CHECK(cudaEventRecord(ev_join, side));
  pf_gemm<false, false><<<dim3(I / 64, L.max_routed), P_THREADS, PfCfg<false>::SMEM, st>>>(pp, L.n_shared);
  pf_gemm<false, true><<<dim3(H / 128, L.max_routed), P_THREADS, PfCfg<false>::SMEM, st>>>(pp, L.n_shared);
  C10_CUDA_CHECK(cudaStreamWaitEvent(st, ev_join, 0));
  auto out = torch::empty({M, H}, x.options());
  pf_combine<<<dim3(M, H / 2048 + (H % 2048 ? 1 : 0)), 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(base + L.dbuf), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

void quant_x(torch::Tensor x, torch::Tensor xq, torch::Tensor xs) {
  const at::cuda::CUDAGuard guard(x.device());
  quant_x_kernel<<<x.size(0), 160, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<uint8_t*>(xq.data_ptr()),
      static_cast<uint8_t*>(xs.data_ptr()));
}

}  // namespace og

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &og::decode);
  m.def("decode_workspace_bytes", &og::decode_workspace_bytes);
  m.def("prefill", &og::prefill);
  m.def("prefill_workspace_bytes", &og::prefill_workspace_bytes);
  m.def("quant_x", &og::quant_x);
  m.def("set_trace", &og::set_trace);
}
