# SPDX-License-Identifier: MIT
"""Error-minimizing affine3/g128 export with the standard MLX packed layout.

Search clipping seeds, refine scale/bias by least squares, and retain the
original group's quantization when no candidate improves reconstruction MSE.
This is weight-error minimization, not activation-calibrated quantization.
"""

import mlx.core as mx
from functools import cache


@cache
def kernel():
    return mx.fast.metal_kernel(
        name="ds41_affine3_lsq",
        input_names=["x", "q0", "s0", "b0"],
        output_names=["q", "s", "b"],
        source=r"""
 const uint g=thread_position_in_grid.x/32, lane=thread_index_in_simdgroup;
 if(g>=GROUPS)return;
 float v[4];float mn=INFINITY,mxv=-INFINITY,sx=0;
 for(uint j=0;j<4;j++){v[j]=float(x[g*128+j*32+lane]);mn=min(mn,v[j]);mxv=max(mxv,v[j]);sx+=v[j];}
 mn=simd_min(mn);mxv=simd_max(mxv);sx=simd_sum(sx);
 float best_s=float(s0[g]),best_b=float(b0[g]);
 float err=0;
 for(uint j=0;j<4;j++){
  uint idx=j*32+lane, bit=idx*3, word=bit/32, shift=bit%32;
  uint code=q0[g*12+word]>>shift;
  if(shift>29)code|=q0[g*12+word+1]<<(32-shift);
  float d=v[j]-(float(code&7)*best_s+best_b);err+=d*d;
 }
 float best=simd_sum(err); bool improved=false;
 for(uint seed=0;seed<7;seed++){
  float factor=.55f+.10f*seed;
  float sc=max((mxv-mn)*factor/7,1e-10f),bi=(mxv+mn)*.5f-3.5f*sc;
  for(uint it=0;it<5;it++){
   sc=float(bfloat(sc));bi=float(bfloat(bi));
   float sq=0,sqq=0,sqx=0,se=0;
   for(uint j=0;j<4;j++){
    float c=clamp(rint((v[j]-bi)/sc),0.f,7.f);
    sq+=c;sqq+=c*c;sqx+=c*v[j];float d=v[j]-(c*sc+bi);se+=d*d;
   }
   sq=simd_sum(sq);sqq=simd_sum(sqq);sqx=simd_sum(sqx);se=simd_sum(se);
   if(se<best){best=se;best_s=sc;best_b=bi;improved=true;}
   float denom=sqq-sq*sq/128;
   if(denom<1e-8f)break;
   sc=max((sqx-sq*sx/128)/denom,1e-10f);bi=(sx-sc*sq)/128;
  }
 }
 uint codes[4];for(uint j=0;j<4;j++)codes[j]=uint(clamp(rint((v[j]-best_b)/best_s),0.f,7.f));
 // Every lane computes one packed word; shuffles execute uniformly.
 uint result=0;
 for(uint j=0;j<12;j++){
  uint idx=(lane*32)/3+j;
  uint source_lane=idx%32,source_vec=min(idx/32,3u);
  uint c0=simd_shuffle(codes[0],source_lane),c1=simd_shuffle(codes[1],source_lane),c2=simd_shuffle(codes[2],source_lane),c3=simd_shuffle(codes[3],source_lane);
  uint code=source_vec==0?c0:source_vec==1?c1:source_vec==2?c2:c3;
  int offset=int(idx*3)-int(lane*32);
  if(idx<128 && offset<32)result|=offset<0?code>>(-offset):code<<offset;
 }
 if(lane<12)q[g*12+lane]=improved?result:q0[g*12+lane];
 if(lane==0){s[g]=bfloat(best_s);b[g]=bfloat(best_b);}
""",
    )


def quantize(w, group_size=128, bits=3):
    assert group_size == 128 and bits == 3 and w.shape[-1] % 128 == 0
    if w.dtype != mx.bfloat16:
        raise ValueError("LSQ export requires BF16 source weights")
    q, s, b = mx.quantize(w, group_size=128, bits=3)
    n = w.size // 128
    return kernel()(
        inputs=[w, q, s, b],
        template=[("GROUPS", n)],
        grid=(n * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape, s.shape, b.shape],
        output_dtypes=[mx.uint32, w.dtype, w.dtype],
    )
