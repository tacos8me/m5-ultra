import torch
from torch.utils.cpp_extension import load_inline
SRC = r'''
#include <torch/extension.h>
#include <cstdint>
__device__ __forceinline__ void mma0(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  const float z = 0.f;
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {%12,%13}, {%14}, {%15,%16};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(z), "r"(sa), "n"(0), "n"(0), "r"(sb), "n"(0), "n"(0));
}
__device__ __forceinline__ void mmac(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "n"(0), "n"(0), "r"(sb), "n"(0), "n"(0));
}
template <int MODE, int NA>
__global__ void k(float* out, int iters, uint32_t seed) {
  uint32_t a0 = 0x08080808u ^ threadIdx.x ^ seed, a1 = 0x04040404u, a2 = 0x08080808u, a3 = 0x0c0c0c0cu;
  float acc[NA][4] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < NA; ++j) {
      if (MODE == 0) mmac(acc[j], a0, a1, a2, a3, 0x38383838u + j, 0x30303030u, 127u, 127u);
      else {
        float d[4];
        mma0(d, a0, a1, a2, a3, 0x38383838u + j, 0x30303030u, 127u, 127u);
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[j][q] += d[q];
      }
    }
    if (MODE == 2) { a0 = (a0 << 2) & 0x3C3C3C3Cu; a1 = __byte_perm(a0, a1, 0x5140); a2 = (a2 >> 2) & 0x3C3C3C3Cu; a3 = __byte_perm(a2, a3, 0x7362); }
  }
  float s = 0;
  for (int j = 0; j < NA; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
double run(int64_t mode, int64_t na, int64_t blocks, int64_t warps, int64_t iters) {
  auto out = torch::empty({blocks * warps * 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto launch = [&]() {
    if (mode == 0 && na == 8) k<0, 8><<<blocks, warps * 32>>>(out.data_ptr<float>(), iters, 1);
    if (mode == 1 && na == 8) k<1, 8><<<blocks, warps * 32>>>(out.data_ptr<float>(), iters, 1);
    if (mode == 1 && na == 16) k<1, 16><<<blocks, warps * 32>>>(out.data_ptr<float>(), iters, 1);
    if (mode == 2 && na == 8) k<2, 8><<<blocks, warps * 32>>>(out.data_ptr<float>(), iters, 1);
  };
  launch();
  cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms;
}
'''
mod = load_inline('mma_tput2', 'double run(int64_t mode, int64_t na, int64_t blocks, int64_t warps, int64_t iters);', SRC,
                  functions=['run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a', '--fmad=false'])
sms = torch.cuda.get_device_properties(0).multi_processor_count
for mode, na, name in ((0, 8, 'chained acc'), (1, 8, 'block from zero + 4 FADD'), (1, 16, 'from zero + FADD, 16 indep'), (2, 8, 'from zero + FADD + expand ALU')):
    for warps in (4, 8, 16):
        iters = 4000
        ms = mod.run(mode, na, sms, warps, iters)
        mmas = sms * warps * iters * na
        print(f'{name:32s} warps/SM {warps:2d}: {mmas * 8192 / ms / 1e9:6.0f} TFLOPS ({mmas / sms / (ms * 1e-3) / 1e9:.3f} G MMA/SM/s)')
