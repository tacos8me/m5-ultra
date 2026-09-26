"""Streaming pattern microbench: 1504 warps, each streams items of 16 rows x 2560 B via a 3-stage cp.async ring.
mode 0: rows strided (row stride 2560, 128 B per row per stage) = current decode layout
mode 1: item-contiguous (each stage = 2 KB contiguous)"""
import torch
from torch.utils.cpp_extension import load_inline
SRC = r'''
#include <torch/extension.h>
#include <cstdint>
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__global__ void __launch_bounds__(128) k(const uint8_t* base, int n_items, int mode, int nst_ring, int* queue, float* sink) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint8_t* ring = sm + warp * 4 * 2304;
  float acc = 0.f;
  for (;;) {
    int item = 0;
    if (lane == 0) item = atomicAdd(queue, 1);
    item = __shfl_sync(0xffffffffu, item, 0);
    if (item >= n_items) break;
    const uint8_t* ib = base + (size_t)item * 40960;
    const int nst = 20;
    auto issue = [&](int st) {
      uint8_t* dst = ring + (st % nst_ring) * 2304;
      for (int i = 0; i < 4; ++i) {
        int q = lane + 32 * i, r = q >> 3, c = q & 7;
        const uint8_t* src = mode == 0 ? ib + r * 2560 + st * 128 + c * 16 : ib + st * 2048 + r * 128 + c * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(su(dst + r * 144 + c * 16)), "l"(src));
      }
      asm volatile("cp.async.commit_group;\n" ::);
    };
    for (int p = 0; p < nst_ring - 1; ++p) issue(p);
    for (int st = 0; st < nst; ++st) {
      asm volatile("cp.async.wait_group 2;\n" ::);
      __syncwarp();
      if (st + nst_ring - 1 < nst) issue(st + nst_ring - 1); else asm volatile("cp.async.commit_group;\n" ::);
      const uint8_t* s = ring + (st % nst_ring) * 2304;
      acc += *reinterpret_cast<const float*>(s + (lane >> 2) * 144 + (lane & 3) * 4);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncwarp();
  }
  if (acc == 12345.f) sink[0] = acc;
}
double run(torch::Tensor buf, int64_t mode, int64_t grid) {
  int n_items = buf.numel() / 40960;
  auto q = torch::zeros({1}, buf.options().dtype(torch::kInt32));
  auto sink = torch::zeros({1}, buf.options().dtype(torch::kFloat32));
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4 * 2304);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<grid, 128, 4 * 4 * 2304>>>(buf.data_ptr<uint8_t>(), n_items, mode, 4, q.data_ptr<int>(), sink.data_ptr<float>());
  q.zero_();
  cudaEventRecord(e0);
  k<<<grid, 128, 4 * 4 * 2304>>>(buf.data_ptr<uint8_t>(), n_items, mode, 4, q.data_ptr<int>(), sink.data_ptr<float>());
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms;
}
'''
mod = load_inline('bwpat', 'double run(torch::Tensor buf, int64_t mode, int64_t grid);', SRC, functions=['run'],
                  extra_cuda_cflags=['-O3', '-gencode=arch=compute_120a,code=sm_120a'])
buf = torch.randint(0, 255, (1200 * 2**20,), dtype=torch.uint8, device='cuda')
buf = buf[: (buf.numel() // 40960) * 40960]
sms = torch.cuda.get_device_properties(0).multi_processor_count
for grid_mult in (1, 2):
    for mode in (0, 1):
        best = min(mod.run(buf, mode, sms * grid_mult) for _ in range(3))
        print(f'grid {grid_mult}x{sms} blocks x4 warps, mode {mode} ({"row-strided" if mode == 0 else "contiguous"}): {buf.numel() / best / 1e9:.3f} TB/s')
