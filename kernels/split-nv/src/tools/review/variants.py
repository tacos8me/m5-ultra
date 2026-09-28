"""og_moe.cu variants for the ds41 self-review verification (built from the box-perf source by exact text edits).

base     unchanged
nocomb   decode combine does no dbuf loads (writes zeros) -- measurement only, NOT a candidate
lay8     decode dbuf [tok][c][8 pos] (one 16-byte load per output element; same fp32 summation order p = 0..6)
dnst2    decode D_NST 4 -> 2 and __launch_bounds__(128, 2): 2 CTAs/SM fit in the 100 KB/SM (run with grid 2 x SMs)
pfexact  prefill routed GEMM grid.y from a host override (set_grid_routed) instead of max_routed
pfnst2   prefill cp.async stages 3 -> 2 (sensitivity)
pfnst4   prefill stages 3 -> 4 for FP4 (the reviewer's proposal; FP8 stays 3: 4 x 33792 B cannot fit)
"""
import os

from torch.utils.cpp_extension import load

SRC = '/work/hooks/split_nv/og_moe/og_moe.cu'
FLAGS = ['-O3', '-std=c++17', '--fmad=false', '-gencode=arch=compute_120a,code=sm_120a', '-lineinfo']
OUT = os.environ.get('REVIEW_SRC', '/review/src')


def _sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


COMB_OLD = """#pragma unroll
            for (int p = 0; p < NPOS; ++p)
              v += __bfloat162float(__ldcg(B.dbuf + (static_cast<size_t>(tok) * NPOS + p) * H + c));"""
STORE_OLD = """      __nv_bfloat16* d = B.dbuf + (static_cast<size_t>(tok) * NPOS + wk->slot_pos[s][tok]) * H + ct * 16 + g;
      d[0] = __float2bfloat16_rn(acc[j]);
      d[8] = __float2bfloat16_rn(acc[2 + j]);"""


def edit(name, s):
    if name == 'base':
        return s
    if name == 'nocomb':
        return _sub(s, COMB_OLD, '            v += 0.f;')
    if name == 'lay8':
        s = _sub(s, 'total = al(dbuf + 2ull * MAXM * NPOS * H) + 256;', 'total = al(dbuf + 2ull * MAXM * 8 * H) + 256;')
        s = _sub(s, STORE_OLD, """      __nv_bfloat16* d = B.dbuf + ((static_cast<size_t>(tok) * H + ct * 16 + g) << 3) + wk->slot_pos[s][tok];
      d[0] = __float2bfloat16_rn(acc[j]);
      d[64] = __float2bfloat16_rn(acc[2 + j]);""")
        return _sub(s, COMB_OLD, """            const uint4 q4 = __ldcg(reinterpret_cast<const uint4*>(B.dbuf + ((static_cast<size_t>(tok) * H + c) << 3)));
            const uint32_t w4[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int p = 0; p < NPOS; ++p)
              v += __uint_as_float((p & 1) ? (w4[p >> 1] & 0xFFFF0000u) : (w4[p >> 1] << 16));""")
    if name == 'dnst2':
        s = _sub(s, 'constexpr int D_NST = 4;', 'constexpr int D_NST = 2;')
        return _sub(s, '__launch_bounds__(D_WARPS * 32, 1) decode_main', '__launch_bounds__(D_WARPS * 32, 2) decode_main')
    if name == 'pfexact':
        s = _sub(s, 'int64_t prefill_workspace_bytes(int64_t M)',
                 'static int g_grid_routed = 0;\nvoid set_grid_routed(int64_t n) { g_grid_routed = static_cast<int>(n); }\n\n'
                 'int64_t prefill_workspace_bytes(int64_t M)')
        s = _sub(s, 'dim3(I / 64, L.max_routed)', 'dim3(I / 64, g_grid_routed > 0 ? g_grid_routed : L.max_routed)')
        s = _sub(s, 'dim3(H / 128, L.max_routed)', 'dim3(H / 128, g_grid_routed > 0 ? g_grid_routed : L.max_routed)')
        return _sub(s, '  m.def("quant_x", &og::quant_x);',
                    '  m.def("quant_x", &og::quant_x);\n  m.def("set_grid_routed", &og::set_grid_routed);')
    if name == 'pfnst2':
        return _sub(s, 'static constexpr int NST = 3;', 'static constexpr int NST = 2;')
    if name == 'pfnst4':
        return _sub(s, 'static constexpr int NST = 3;', 'static constexpr int NST = FP8W ? 3 : 4;')
    raise KeyError(name)


def build(name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'og_{name}.cu')
    src = edit(name, open(SRC).read())
    if not os.path.exists(path) or open(path).read() != src:
        open(path, 'w').write(src)
    bdir = os.path.join('/review/build', name)
    os.makedirs(bdir, exist_ok=True)
    return load(name=f'og_rev_{name}', sources=[path], extra_cuda_cflags=FLAGS, build_directory=bdir, verbose=False)
