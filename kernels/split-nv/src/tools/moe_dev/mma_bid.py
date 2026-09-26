"""Does byte-id selector c pick byte c of a packed 4-kb scale word (per providing thread)?"""
import torch
from torch.utils.cpp_extension import load_inline
SRC = r'''
#include <torch/extension.h>
#include <cstdint>
template <int BID>
__device__ __forceinline__ void mma_bid(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "n"(BID), "n"(0), "r"(sfb), "n"(BID), "n"(0));
}
// 4 k-blocks; scales packed: sfa word per row = bytes kb0..kb3; one block MMA per kb from zero, promoted.
__global__ void k(const uint8_t* A, const uint8_t* B, const uint32_t* sfa, const uint32_t* sfb, float* D, int packed) {
  int lane = threadIdx.x, g = lane / 4, t = lane % 4;
  float acc[4] = {0, 0, 0, 0};
  const int K = 128;
  uint32_t wa = sfa[g + (t & 1) * 8], wb = sfb[g];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    auto ld4 = [&](const uint8_t* base, int row, int k0) { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(base[row * K + kb * 32 + k0 + i]) << (8 * i); return v; };
    uint32_t a[4] = {ld4(A, g, 8 * t), ld4(A, g + 8, 8 * t), ld4(A, g, 8 * t + 4), ld4(A, g + 8, 8 * t + 4)};
    uint32_t b0 = ld4(B, g, 8 * t), b1 = ld4(B, g, 8 * t + 4);
    float d[4] = {0, 0, 0, 0};
    if (packed) {
      if (kb == 0) mma_bid<0>(d, a, b0, b1, wa, wb);
      if (kb == 1) mma_bid<1>(d, a, b0, b1, wa, wb);
      if (kb == 2) mma_bid<2>(d, a, b0, b1, wa, wb);
      if (kb == 3) mma_bid<3>(d, a, b0, b1, wa, wb);
    } else {
      mma_bid<0>(d, a, b0, b1, (wa >> (8 * kb)) & 0xFF, (wb >> (8 * kb)) & 0xFF);
    }
    for (int i = 0; i < 4; ++i) acc[i] += d[i];
  }
  D[g * 8 + 2 * t] = acc[0]; D[g * 8 + 2 * t + 1] = acc[1]; D[(g + 8) * 8 + 2 * t] = acc[2]; D[(g + 8) * 8 + 2 * t + 1] = acc[3];
}
torch::Tensor run(torch::Tensor A, torch::Tensor B, torch::Tensor sa, torch::Tensor sb, int64_t packed) {
  auto D = torch::zeros({16, 8}, A.options().dtype(torch::kFloat32));
  k<<<1, 32>>>(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), (const uint32_t*)sa.data_ptr(), (const uint32_t*)sb.data_ptr(), D.data_ptr<float>(), (int)packed);
  return D;
}
'''
mod = load_inline('mma_bid', 'torch::Tensor run(torch::Tensor A, torch::Tensor B, torch::Tensor sa, torch::Tensor sb, int64_t packed);', SRC,
                  functions=['run'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a'])
torch.manual_seed(1)
ok = 0
for trial in range(20):
    A = (torch.randint(0, 16, (16, 128), dtype=torch.uint8) << 2).cuda()
    B = torch.randn(8, 128).clamp(-40, 40).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    sa = torch.randint(118, 136, (16, 4), dtype=torch.uint8).cuda()
    sb = torch.randint(118, 136, (8, 4), dtype=torch.uint8).cuda()
    r0 = mod.run(A, B, sa, sb, 0)
    r1 = mod.run(A, B, sa, sb, 1)
    ok += int(torch.equal(r0, r1))
print('packed byte-id == unpacked:', ok, '/ 20')
