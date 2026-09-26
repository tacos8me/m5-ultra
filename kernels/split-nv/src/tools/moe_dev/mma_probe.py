"""Probe the SM120 block-scaled MMA (mxf8f6f4, e2m1 x e4m3, ue8m0, m16n8k32) numerics and throughput.

1. One k-block from a zero accumulator vs the exact scaled dot product (float64): is it RN(exact)?
2. Row/column independence: the same element computed with different companion rows/columns.
3. Chained accumulation (acc through many k-blocks) vs fp32 promotion (block from zero, then fp32 add).
4. Throughput: MMAs per SM per clock.
"""
import torch
from torch.utils.cpp_extension import load_inline

SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// A = weights e2m1 (8-bit containers, value << 2), 16 rows; B = activations e4m3, 8 columns (tokens).
__device__ __forceinline__ void mma_w_a(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                         uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "n"(0), "n"(0), "r"(sfb), "n"(0),
        "n"(0));
}

__device__ __forceinline__ void mma_e4_e4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                          uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "n"(0), "n"(0), "r"(sfb), "n"(0),
        "n"(0));
}

// Canonical fragments. A: u8 [16][32*nkb] containers; B: u8 [8][32*nkb] (column-major: token rows);
// sfa u8 [16][nkb]; sfb u8 [8][nkb]. mode 0: chained acc; mode 1: promotion (zero C per block, fp32 add).
// perm: 0 canonical k layout; 1 "thread t owns k 8t..8t+7" (same relabeling for A and B).
__global__ void probe(const uint8_t* A, const uint8_t* B, const uint8_t* sfa, const uint8_t* sfb, float* D,
                      int nkb, int mode, int perm) {
  const int lane = threadIdx.x, g = lane / 4, t = lane % 4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kb = 0; kb < nkb; ++kb) {
    const int K = 32 * nkb;
    auto ld4 = [&](const uint8_t* base, int row, int k0) {
      uint32_t v = 0;
      for (int i = 0; i < 4; ++i) v |= uint32_t(base[row * K + kb * 32 + k0 + i]) << (8 * i);
      return v;
    };
    int k_lo = perm ? 8 * t : 4 * t, k_hi = perm ? 8 * t + 4 : 4 * t + 16;
    uint32_t a[4] = {ld4(A, g, k_lo), ld4(A, g + 8, k_lo), ld4(A, g, k_hi), ld4(A, g + 8, k_hi)};
    uint32_t b0 = ld4(B, g, k_lo), b1 = ld4(B, g, k_hi);
    uint32_t sa = sfa[(g + (t & 1) * 8) * nkb + kb], sb = sfb[g * nkb + kb];
    if (mode == 2) {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      mma_e4_e4(d, a, b0, b1, sa, sb);
      for (int i = 0; i < 4; ++i) acc[i] += d[i];
    } else if (mode == 0) {
      mma_w_a(acc, a, b0, b1, sa, sb);
    } else {
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      mma_w_a(d, a, b0, b1, sa, sb);
      for (int i = 0; i < 4; ++i) acc[i] += d[i];
    }
  }
  D[g * 8 + 2 * t] = acc[0];
  D[g * 8 + 2 * t + 1] = acc[1];
  D[(g + 8) * 8 + 2 * t] = acc[2];
  D[(g + 8) * 8 + 2 * t + 1] = acc[3];
}

__global__ void tput(float* out, int iters) {
  uint32_t a[4] = {0x08080808u ^ threadIdx.x, 0x04040404u, 0x08080808u, 0x0c0c0c0cu};
  float acc[4][4] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) mma_w_a(acc[j], a, 0x38383838u + j, 0x30303030u, 127u, 127u);
  }
  float s = 0;
  for (int j = 0; j < 4; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

torch::Tensor run_probe(torch::Tensor A, torch::Tensor B, torch::Tensor sfa, torch::Tensor sfb, int64_t mode,
                        int64_t perm) {
  int nkb = A.size(1) / 32;
  auto D = torch::zeros({16, 8}, A.options().dtype(torch::kFloat32));
  probe<<<1, 32>>>(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(), sfb.data_ptr<uint8_t>(),
                   D.data_ptr<float>(), nkb, (int)mode, (int)perm);
  return D;
}

double run_tput(int64_t blocks, int64_t warps, int64_t iters) {
  auto out = torch::empty({blocks * warps * 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  tput<<<blocks, warps * 32>>>(out.data_ptr<float>(), iters);
  cudaEventRecord(e0);
  tput<<<blocks, warps * 32>>>(out.data_ptr<float>(), iters);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms;
}
'''
CPP = 'torch::Tensor run_probe(torch::Tensor A, torch::Tensor B, torch::Tensor sfa, torch::Tensor sfb, int64_t mode, int64_t perm);\n' \
      'double run_tput(int64_t blocks, int64_t warps, int64_t iters);'
mod = load_inline('mma_probe', CPP, SRC, functions=['run_probe', 'run_tput'], verbose=False,
                  extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a', '-std=c++17'])

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
e4m3_vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)
ok_e4m3 = torch.isfinite(e4m3_vals).nonzero().flatten().to(torch.uint8)


def rand_case(nkb, wide):
    wn = torch.randint(0, 16, (16, 32 * nkb), dtype=torch.uint8)
    xb = ok_e4m3[torch.randint(0, len(ok_e4m3), (8, 32 * nkb))]
    if not wide:  # moderate exponents
        xv = torch.randn(8, 32 * nkb).clamp(-40, 40).to(torch.float8_e4m3fn)
        xb = xv.view(torch.uint8)
    sa = torch.randint(120, 134, (16, nkb), dtype=torch.uint8)
    sb = torch.randint(120, 134, (8, nkb), dtype=torch.uint8)
    return wn, xb, sa, sb


def exact(wn, xb, sa, sb, nkb, mode):
    w = FP4[wn.long()]
    x = xb.view(torch.float8_e4m3fn).to(torch.float64)
    out = torch.zeros(16, 8, dtype=torch.float32)
    for kb in range(nkb):
        blk = (w[:, kb * 32:(kb + 1) * 32] @ x[:, kb * 32:(kb + 1) * 32].T)
        sc = torch.ldexp(torch.ones(16, 8, dtype=torch.float64),
                         (sa[:, kb].long()[:, None] - 127) + (sb[:, kb].long()[None, :] - 127))
        d = (blk * sc).to(torch.float32)  # RN once
        out = (out + d) if mode == 1 else out + d  # fp32 adds (promotion arithmetic)
    return out


def gpu(wn, xb, sa, sb, mode, perm=0):
    return mod.run_probe((wn << 2).cuda(), xb.cuda(), sa.cuda(), sb.cuda(), mode, perm).cpu()


torch.manual_seed(0)
for wide in (False, True):
    n_eq = n_tot = 0
    n_eq_p = 0
    maxrel = 0
    for trial in range(50):
        wn, xb, sa, sb = rand_case(1, wide)
        e = exact(wn, xb, sa, sb, 1, 1)
        g0 = gpu(wn, xb, sa, sb, 1, 0)
        g1 = gpu(wn, xb, sa, sb, 1, 1)
        n_eq += (g0 == e).sum().item()
        n_eq_p += (g1 == g0).sum().item()
        n_tot += e.numel()
        maxrel = max(maxrel, ((g0.double() - e.double()).abs() / e.double().abs().clamp_min(1e-30)).max().item())
    print(f'[1] one k-block from zero, wide={wide}: == RN(exact) {n_eq}/{n_tot}, max rel {maxrel:.2e}; '
          f'permuted-k == canonical {n_eq_p}/{n_tot}')

for trial_wide in (False, True):
    n_eq = n_tot = n_p = 0
    for trial in range(50):
        _, xb, sa, sb = rand_case(1, trial_wide)
        _, wb, _, _ = rand_case(1, True)
        wb = torch.cat([wb, rand_case(1, trial_wide)[1]])  # 16 rows e4m3
        w = wb.view(torch.float8_e4m3fn).to(torch.float64)
        x = xb.view(torch.float8_e4m3fn).to(torch.float64)
        sc = torch.ldexp(torch.ones(16, 8, dtype=torch.float64), (sa[:, 0].long()[:, None] - 127) + (sb[:, 0].long()[None, :] - 127))
        e = ((w @ x.T) * sc).to(torch.float32)
        g0 = mod.run_probe(wb.cuda(), xb.cuda(), sa.cuda(), sb.cuda(), 2, 0).cpu()
        g1 = mod.run_probe(wb.cuda(), xb.cuda(), sa.cuda(), sb.cuda(), 2, 1).cpu()
        n_eq += (g0 == e).sum().item(); n_p += (g1 == g0).sum().item(); n_tot += e.numel()
    print(f'[1b] e4m3 x e4m3 block from zero, wide={trial_wide}: == RN(exact) {n_eq}/{n_tot}; permuted == canonical {n_p}/{n_tot}')

# 2. row/column independence
wn, xb, sa, sb = rand_case(8, True)
base = gpu(wn, xb, sa, sb, 0)
wn2, xb2, sa2, sb2 = rand_case(8, True)
wn2[3], sa2[3] = wn[3], sa[3]
xb2[5], sb2[5] = xb[5], sb[5]
other = gpu(wn2, xb2, sa2, sb2, 0)
print('[2] element (3,5) with different companions equal:', bool(base[3, 5] == other[3, 5]))
ok = 0
for trial in range(30):
    wn, xb, sa, sb = rand_case(4, trial % 2 == 0)
    base = gpu(wn, xb, sa, sb, 0)
    wn2, xb2, sa2, sb2 = rand_case(4, True)
    r, c = trial % 16, trial % 8
    wn2[r], sa2[r] = wn[r], sa[r]
    xb2[c], sb2[c] = xb[c], sb[c]
    other = gpu(wn2, xb2, sa2, sb2, 0)
    ok += int(base[r, c] == other[r, c])
print('[2b] chained 4-block MMA: element equal with different companions', ok, '/ 30')

# 3. chained vs promotion over 160 k-blocks
for wide in (False, True):
    wn, xb, sa, sb = rand_case(160, wide)
    ch = gpu(wn, xb, sa, sb, 0)
    pr = gpu(wn, xb, sa, sb, 1)
    ex = exact(wn, xb, sa, sb, 160, 1)
    exd = torch.zeros(16, 8, dtype=torch.float64)
    w = FP4[wn.long()]
    x = xb.view(torch.float8_e4m3fn).to(torch.float64)
    for kb in range(160):
        sc = torch.ldexp(torch.ones(16, 8, dtype=torch.float64), (sa[:, kb].long()[:, None] - 127) + (sb[:, kb].long()[None, :] - 127))
        exd += (w[:, kb * 32:(kb + 1) * 32] @ x[:, kb * 32:(kb + 1) * 32].T) * sc
    rel = lambda a: ((a.double() - exd).norm() / exd.norm()).item()
    print(f'[3] K=5120 wide={wide}: promotion == emulated promotion {(pr == ex).sum().item()}/128; '
          f'relRMS vs exact: chained {rel(ch):.2e}, promotion {rel(pr):.2e}; chained==promotion {(ch == pr).sum().item()}/128')

# 4. throughput
props = torch.cuda.get_device_properties(0)
for blocks_per_sm, warps in ((1, 4), (2, 4), (1, 8), (2, 8)):
    blocks = props.multi_processor_count * blocks_per_sm
    iters = 20000
    ms = mod.run_tput(blocks, warps, iters)
    mmas = blocks * warps * iters * 4
    print(f'[4] {blocks_per_sm} blk/SM x {warps} warps: {mmas * 16 * 8 * 32 * 2 / ms / 1e9:.0f} TFLOPS '
          f'({mmas / props.multi_processor_count / (ms * 1e-3) / 2.4e9:.2f} MMA/SM/clk @2.4GHz)')
