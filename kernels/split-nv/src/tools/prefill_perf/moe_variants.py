"""og_moe.cu prefill variants for the fused combine (task A), built from the deployed source by exact text edits.

base      unchanged (down GEMM writes dbuf [M][7][H] bf16, pf_combine sums positions 0..6 in fp32 -> bf16)
lastw     no pf_combine: every down-GEMM CTA (routed FP4 and shared FP8 alike), after writing its dbuf tile, bumps a
          per-(token, 128-column tile) counter; the CTA that brings it to 7 sums that token's 7 positions for its 128
          columns (same fp32 order p = 0..6 from 0.f, same bf16 rounding) straight into out. dbuf keeps its dtype and
          layout; counters are zeroed per call.
lastw_col lastw with the routed down GEMM grid transposed (tile index fastest): all tiles of one column band run
          together, so a token's 7 partials of that band are written close in time (L2-resident when combined).
"""
import os

from torch.utils.cpp_extension import load

SRC = "/work/hooks/split_nv/og_moe/og_moe.cu"
FLAGS = ["-O3", "-std=c++17", "--fmad=false", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo"]
OUT = os.environ.get("MOE_VAR_SRC", "/pf/build/moe_var")


def _sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:100], s.count(old))
    return s.replace(old, new)


def edit(name, s):
    if name == "base":
        return s
    # params: counters + out
    s = _sub(s, """  const int32_t* n_tiles;
};

// One stage""", """  const int32_t* n_tiles;
  int32_t* cnt;           // [M][H / 128] contributions seen per (token, column tile)
  __nv_bfloat16* out;     // [M][H]
};

// One stage""")
    if name == "lastw_col":
        s = _sub(s, """  const int tile = tile0 + blockIdx.y;
  if (tile >= *p.n_tiles) return;""", """  const int tile = tile0 + ((DOWN && !FP8W) ? blockIdx.x : blockIdx.y);
  if (tile >= *p.n_tiles) return;""")
        s = _sub(s, """  const int bx = blockIdx.x;  // gate/up: 64-feature tile (0..17); down: 128-row c tile (0..39)""",
                 """  const int bx = (DOWN && !FP8W) ? blockIdx.y : blockIdx.x;  // gate/up: 64-feature tile; down: c tile""")
    # epilogue: after the dbuf stores, count and let the last contributor combine
    s = _sub(s, """          for (int x = 0; x < 2; ++x) {
            const int R = x ? R1 : R0;
            dd[R] = __float2bfloat16_rn(acc[x][j][k]);
            dd[R + 8] = __float2bfloat16_rn(acc[x][j][2 + k]);
          }
        }
      }
    }
  } else {""", """          for (int x = 0; x < 2; ++x) {
            const int R = x ? R1 : R0;
            dd[R] = __float2bfloat16_rn(acc[x][j][k]);
            dd[R + 8] = __float2bfloat16_rn(acc[x][j][2 + k]);
          }
        }
      }
    }
    // fused combine: the last of a token's 7 contributors to this column tile sums them (pf_combine's arithmetic)
    // (dynamic smem is free again: the FP8 variant already uses the whole per-block limit, no static smem allowed)
    int* s_nlast = reinterpret_cast<int*>(psmem);
    int* s_last = s_nlast + 4;
    __threadfence();
    __syncthreads();
    if (tid == 0) *s_nlast = 0;
    __syncthreads();
    for (int r = tid; r < cnt; r += P_THREADS) {
      const int tok = p.p_tok[t0 + r];
      if (atomicAdd(p.cnt + static_cast<size_t>(tok) * (H / 128) + bx, 1) == NPOS - 1) s_last[atomicAdd(s_nlast, 1)] = tok;
    }
    __syncthreads();
    __threadfence();
    const int nl = *s_nlast;
    for (int task = tid; task < nl * 16; task += P_THREADS) {
      const int tok = s_last[task >> 4];
      const int c = bx * 128 + (task & 15) * 8;
      float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int pp = 0; pp < NPOS; ++pp) {
        const uint4 d = __ldcg(reinterpret_cast<const uint4*>(p.dbuf + (static_cast<size_t>(tok) * NPOS + pp) * H + c));
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
      *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(tok) * H + c) = o;
    }
  } else {""")
    # layout: counters after dbuf
    s = _sub(s, """  size_t xq, xs, counts, cursor, offsets, ppos, p_tok, p_w, p_pos, tiles, hq, hsf, dbuf, total;
  int max_tiles, n_shared, max_routed;""", """  size_t xq, xs, counts, cursor, offsets, ppos, p_tok, p_w, p_pos, tiles, hq, hsf, dbuf, cntb, total;
  int max_tiles, n_shared, max_routed;""")
    s = _sub(s, """    total = al(dbuf + 2 * static_cast<size_t>(M) * NPOS * H) + 256;
  }
};

int64_t prefill_workspace_bytes""", """    cntb = al(dbuf + 2 * static_cast<size_t>(M) * NPOS * H);
    total = al(cntb + 4 * static_cast<size_t>(M) * (H / 128)) + 256;
  }
};

int64_t prefill_workspace_bytes""")
    s = _sub(s, """  C10_CUDA_CHECK(cudaMemsetAsync(base + L.counts, 0, 4 * (NEXP + 1), st));""",
             """  C10_CUDA_CHECK(cudaMemsetAsync(base + L.counts, 0, 4 * (NEXP + 1), st));
  C10_CUDA_CHECK(cudaMemsetAsync(base + L.cntb, 0, 4 * static_cast<size_t>(M) * (H / 128), st));
  auto out = torch::empty({M, H}, x.options());""")
    s = _sub(s, """  pp.n_tiles = tb + 3 * mt;
  static bool init = false;""", """  pp.n_tiles = tb + 3 * mt;
  pp.cnt = I32(L.cntb);
  pp.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  static bool init = false;""")
    if name == "lastw_col":
        s = _sub(s, """  pf_gemm<false, true><<<dim3(H / 128, L.max_routed), P_THREADS, PfCfg<false>::SMEM, st>>>(pp, L.n_shared);""",
                 """  pf_gemm<false, true><<<dim3(L.max_routed, H / 128), P_THREADS, PfCfg<false>::SMEM, st>>>(pp, L.n_shared);""")
    s = _sub(s, """  C10_CUDA_CHECK(cudaStreamWaitEvent(st, ev_join, 0));
  auto out = torch::empty({M, H}, x.options());
  pf_combine<<<dim3(M, H / 2048 + (H % 2048 ? 1 : 0)), 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(base + L.dbuf), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));""",
             """  C10_CUDA_CHECK(cudaStreamWaitEvent(st, ev_join, 0));""")
    return s


def build(name):
    os.makedirs(OUT, exist_ok=True)
    src = open(SRC).read()
    path = os.path.join(OUT, f"og_{name}.cu")
    txt = edit(name, src)
    if not os.path.exists(path) or open(path).read() != txt:
        open(path, "w").write(txt)
    bdir = os.path.join(OUT, name)
    os.makedirs(bdir, exist_ok=True)
    return load(name=f"og_var_{name}", sources=[path], extra_cuda_cflags=FLAGS, build_directory=bdir, verbose=False)
