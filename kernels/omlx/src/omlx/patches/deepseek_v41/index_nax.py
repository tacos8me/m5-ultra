# SPDX-License-Identifier: MIT
"""M5 integer NAX index scoring and exact radix selection.

The integer path accepts FP4-exact floating-point values after the model's fake
quantization. Group scales are retained separately; no extra quantization is used.
"""

import os
from functools import cache

import mlx.core as mx

DS41_TOPK_COMPACT = os.environ.get("DS41_TOPK_COMPACT", "1") == "1"

_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using ds41_d2 = dextents<int32_t, 2>;
"""


@cache
def _pack():
    return mx.fast.metal_kernel(
        name="ds41_index_int8_pack",
        input_names=["x", "meta"],
        output_names=["packed", "scales"],
        source=r"""
    uint row=threadgroup_position_in_grid.x, batch=threadgroup_position_in_grid.y;
    uint group=threadgroup_position_in_grid.z,lane=thread_index_in_simdgroup;
    uint rows=x_shape[1],PAD=meta[0];
    float value=row<rows?float(x[(size_t(batch)*rows+row)*128+group*32+lane]):0.0f;
    float maximum=simd_max(abs(value));
    int exponent=0;frexp(maximum,exponent);
    float scale=maximum>0?ldexp(1.0f,exponent-4):1.0f;
    packed[((size_t(batch)*4+group)*PAD+row)*32+lane]=char(rint(value/scale));
    if(lane==0)scales[(size_t(batch)*4+group)*PAD+row]=scale;
    """,
    )


def pack(x, pad=None):
    b, n, d = x.shape
    assert d == 128
    pad = n if pad is None else pad
    return _pack()(
        inputs=[x, mx.array([pad], mx.uint32)],
        grid=(32 * pad, b, 4),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, 4, pad, 32), (b, 4, pad)],
        output_dtypes=[mx.int8, mx.float32],
    )


@cache
def _pack_full():
    return mx.fast.metal_kernel(
        name="ds41_index_int8_full_pack",
        input_names=["x", "meta"],
        output_names=["packed", "scales", "exact"],
        source=r"""
    uint row=threadgroup_position_in_grid.x,batch=threadgroup_position_in_grid.y,lane=thread_index_in_simdgroup;
    uint rows=x_shape[1],PAD=meta[0];float v[4],maximum=0;
    for(uint g=0;g<4;++g){v[g]=row<rows?float(x[(size_t(batch)*rows+row)*128+g*32+lane]):0;maximum=max(maximum,abs(v[g]));}
    maximum=simd_max(maximum);int exponent=0;frexp(maximum,exponent);
    float scale=maximum>0?ldexp(1.0f,exponent-7):1.0f;uint bad=0;
    for(uint g=0;g<4;++g){float z=rint(v[g]/scale);bad|=(z*scale!=v[g] || abs(z)>127);packed[(size_t(batch)*PAD+row)*128+g*32+lane]=char(clamp(z,-127.0f,127.0f));}
    bad=simd_max(bad);
    if(lane==0){scales[size_t(batch)*PAD+row]=scale;exact[size_t(batch)*PAD+row]=!bad;}
    """,
    )


def pack_full(x, pad=None):
    b, n, d = x.shape
    assert d == 128
    pad = n if pad is None else pad
    return _pack_full()(
        inputs=[x, mx.array([pad], mx.uint32)],
        grid=(32 * pad, b, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, pad, 128), (b, pad), (b, pad)],
        output_dtypes=[mx.int8, mx.float32, mx.bool_],
    )


@cache
def _score():
    return mx.fast.metal_kernel(
        name="ds41_index_int8_nax",
        input_names=[
            "q",
            "keys",
            "qs",
            "ks",
            "qfull",
            "kfull",
            "qscale",
            "kscale",
            "qexact",
            "kexact",
            "weights",
            "lens",
            "candidates",
            "meta",
        ],
        output_names=["scores"],
        header=_HEADER,
        source=r"""
    uint tile=threadgroup_position_in_grid.x,row=threadgroup_position_in_grid.y;
    uint L=meta[2];
    uint tid=thread_index_in_threadgroup,batch=row/L,query=row%L;
    uint n=meta[0],padded=meta[1],first=tile*64;
    threadgroup uint bad[128];
    bad[tid]=tid<32?!qexact[row*32+tid]:(tid<96?!kexact[batch*padded+first+tid-32]:0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){uint any=0;for(uint i=0;i<96;++i)any|=bad[i];bad[0]=any;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool full=bad[0]==0;
    constexpr auto desc=matmul2d_descriptor(32,64,static_cast<int>(dynamic_extent),false,true,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    using Tensor=tensor<device int8_t,ds41_d2,tensor_inline>;
    // Separate 32-element scale groups retain every FP4 value exactly.
    Tensor qa((device int8_t*)q+size_t(row)*4*32*32,ds41_d2(32,32));
    Tensor kb((device int8_t*)keys+size_t(batch)*4*padded*32+first*32,ds41_d2(32,64));
    auto acc=op.template get_destination_cooperative_tensor<decltype(qa),decltype(kb),int>();
    float values[32];
    for(uint i=0;i<acc.get_capacity();++i)values[i]=0;
    for(uint g=0;g<(full?1u:4u);++g) {
      Tensor a(full?(device int8_t*)qfull+size_t(row)*32*128:(device int8_t*)q+(size_t(row)*4+g)*32*32,ds41_d2(full?128:32,32));
      Tensor b(full?(device int8_t*)kfull+(size_t(batch)*padded+first)*128:(device int8_t*)keys+((size_t(batch)*4+g)*padded+first)*32,ds41_d2(full?128:32,64));
      op.run(a,b,acc);
      for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
        auto ij=acc.get_multidimensional_index(i);uint head=ij[1],key=first+ij[0];
        values[i]+=float(acc[i])*(full?qscale[row*32+head]*kscale[batch*padded+key]:qs[(size_t(row)*4+g)*32+head]*ks[(size_t(batch)*4+g)*padded+key]);
      }
    }
    threadgroup float dots[32*64];
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
      auto ij=acc.get_multidimensional_index(i);dots[ij[1]*64+ij[0]]=values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<64 && first+tid<n) {
      uint key=first+tid;float total=0;
      for(uint h=0;h<32;++h)total+=max(dots[h*64+tid],0.0f)*weights[row*32+h];
      bool valid=key<uint(lens[query]) && (!HAS_CAND || candidates[size_t(row)*n+key]);
      scores[size_t(row)*n+key]=valid?total:-INFINITY;
    }
    """,
    )


def scores(q, keys, weights, lens, candidates=None, packed_keys=None):
    b, l, h, d = q.shape
    assert (h, d) == (32, 128)
    n = keys.shape[1]
    padded = (n + 63) // 64 * 64
    if n == 0:
        return mx.zeros((b, l, 0), mx.float32)
    qi, qs = pack(q.reshape(b * l, 32, 128))
    qf, qscale, qexact = pack_full(q.reshape(b * l, 32, 128))
    kf, kscale, kexact = pack_full(keys, padded)
    ki, ks = pack(keys, padded) if packed_keys is None else packed_keys
    return _score()(
        inputs=[
            qi,
            ki,
            qs,
            ks,
            qf,
            kf,
            qscale,
            kscale,
            qexact,
            kexact,
            weights,
            lens,
            candidates if candidates is not None else mx.zeros((1,), mx.bool_),
            mx.array([n, padded, l], mx.int32),
        ],
        template=[("HAS_CAND", candidates is not None)],
        grid=(128 * (padded // 64), b * l, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, l, n)],
        output_dtypes=[mx.float32],
    )[0]


@cache
def _radix():
    return mx.fast.metal_kernel(
        name="ds41_index_radix_topk",
        input_names=["scores", "meta"],
        output_names=["ids"],
        header=r"""
    inline uint score_key(float x) {uint u=as_type<uint>(x == 0.0f ? 0.0f : x);return (u&0x80000000u)?~u:(u^0x80000000u);}
    """,
        source=r"""
    uint tid=thread_index_in_threadgroup,row=threadgroup_position_in_grid.x;
    uint n=scores_shape[2],K=meta[0];
    threadgroup atomic_uint hist[256];
    threadgroup uint prefix,remaining,greater[256],equal[256];
    if(tid==0){prefix=0;remaining=K;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int shift=24;shift>=0;shift-=8) {
      atomic_store_explicit(hist+tid,0,memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint i=tid;i<n;i+=256) {
        uint code=score_key(scores[size_t(row)*n+i]);
        if(shift==24 || (code >> (shift+8))==prefix)
          atomic_fetch_add_explicit(hist+((code>>shift)&255u),1,memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(tid==0)for(int bin=255;bin>=0;--bin) {
        uint count=atomic_load_explicit(hist+bin,memory_order_relaxed);
        if(remaining>count)remaining-=count;
        else {prefix=(prefix<<8)|uint(bin);break;}
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint first=(n*tid)/256,last=(n*(tid+1))/256,ng=0,ne=0;
    for(uint i=first;i<last;++i) {
      uint code=score_key(scores[size_t(row)*n+i]);ng+=code>prefix;ne+=code==prefix;
    }
    greater[tid]=ng;equal[tid]=ne;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
      uint g=0,e=0;for(uint t=0;t<256;++t) {uint a=greater[t],b=equal[t];greater[t]=g;equal[t]=e;g+=a;e+=b;}
      remaining=g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g=greater[tid],e=equal[tid];
    for(uint i=first;i<last;++i) {
      uint code=score_key(scores[size_t(row)*n+i]);
      if(code>prefix)ids[size_t(row)*K+g++]=i;
      else if(code==prefix) {if(remaining+e<K)ids[size_t(row)*K+remaining+e]=i;++e;}
    }
    """,
    )


@cache
def _radix_compact():
    """Exact radix top-k reading each score row twice instead of six times.

    Pass 1 histograms the top 12 bits of the order-preserving key. Pass 2
    writes every id above the threshold bin and collects the (few) threshold
    bin members, which are sorted in threadgroup memory by (key desc, index
    asc). That is the selected set of `_radix`: all keys above the k-th key,
    then equal keys by ascending index. A threshold bin larger than CAP falls
    back to `_radix`'s four 8-bit passes.
    """
    return mx.fast.metal_kernel(
        name="ds41_index_radix_compact_topk",
        input_names=["scores", "meta"],
        output_names=["ids"],
        header=r"""
    inline uint score_key(float x) {uint u=as_type<uint>(x == 0.0f ? 0.0f : x);return (u&0x80000000u)?~u:(u^0x80000000u);}
    """,
        source=r"""
    constexpr uint CAP=2048, BINS=4096;
    uint tid=thread_index_in_threadgroup,row=threadgroup_position_in_grid.x;
    uint n=scores_shape[2],K=meta[0];
    const device float* s=scores+size_t(row)*n;
    device int* out=ids+size_t(row)*K;
    threadgroup ulong buf[CAP];
    threadgroup atomic_uint* hist=(threadgroup atomic_uint*)buf;
    threadgroup atomic_uint counters[2];
    threadgroup uint part[256];
    threadgroup uint sh_bin,sh_above,sh_need;
    for(uint b=tid;b<BINS;b+=256)atomic_store_explicit(hist+b,0,memory_order_relaxed);
    if(tid<2)atomic_store_explicit(counters+tid,0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<n;i+=256)
      atomic_fetch_add_explicit(hist+(score_key(s[i])>>20),1,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Thread t owns bins [BINS-16(t+1), BINS-16t), scanned from the top.
    uint local=0;
    for(uint j=0;j<16;++j)local+=atomic_load_explicit(hist+(BINS-1-(tid*16+j)),memory_order_relaxed);
    part[tid]=local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
      uint acc=0,t=0;
      for(;t<255;++t){if(acc+part[t]>=K)break;acc+=part[t];}
      for(uint j=0;j<16;++j) {
        uint bin=BINS-1-(t*16+j),c=atomic_load_explicit(hist+bin,memory_order_relaxed);
        if(acc+c>=K || j==15){sh_bin=bin;sh_above=acc;sh_need=K-acc;break;}
        acc+=c;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bin=sh_bin;
    for(uint i=tid;i<n;i+=256) {
      uint code=score_key(s[i]),hb=code>>20;
      if(hb>bin)out[atomic_fetch_add_explicit(counters,1,memory_order_relaxed)]=int(i);
      else if(hb==bin) {
        uint c=atomic_fetch_add_explicit(counters+1,1,memory_order_relaxed);
        if(c<CAP)buf[c]=(ulong(code)<<32)|ulong(0xFFFFFFFFu-i);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    uint count=atomic_load_explicit(counters+1,memory_order_relaxed);
    if(count<=CAP) {
      uint P=1;while(P<count)P<<=1;
      for(uint i=count+tid;i<P;i+=256)buf[i]=0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Bitonic sort, descending: key desc, then index asc.
      for(uint k=2;k<=P;k<<=1)for(uint j=k>>1;j>0;j>>=1) {
        for(uint i=tid;i<P;i+=256) {
          uint l=i^j;
          if(l>i) {
            ulong a=buf[i],b=buf[l];
            bool desc=(i&k)==0;
            if(desc?(a<b):(a>b)){buf[i]=b;buf[l]=a;}
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for(uint r=tid;r<sh_need;r+=256)out[sh_above+r]=int(0xFFFFFFFFu-uint(buf[r]&0xFFFFFFFFul));
      return;
    }
    // Threshold bin too large to gather: the original four 8-bit passes.
    threadgroup uint prefix,remaining,greater[256],equal[256];
    threadgroup atomic_uint* h8=hist;
    if(tid==0){prefix=0;remaining=K;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int shift=24;shift>=0;shift-=8) {
      atomic_store_explicit(h8+tid,0,memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint i=tid;i<n;i+=256) {
        uint code=score_key(s[i]);
        if(shift==24 || (code >> (shift+8))==prefix)
          atomic_fetch_add_explicit(h8+((code>>shift)&255u),1,memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(tid==0)for(int b=255;b>=0;--b) {
        uint c=atomic_load_explicit(h8+b,memory_order_relaxed);
        if(remaining>c)remaining-=c;
        else {prefix=(prefix<<8)|uint(b);break;}
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint first=(n*tid)/256,last=(n*(tid+1))/256,ng=0,ne=0;
    for(uint i=first;i<last;++i) {
      uint code=score_key(s[i]);ng+=code>prefix;ne+=code==prefix;
    }
    greater[tid]=ng;equal[tid]=ne;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) {
      uint g=0,e=0;for(uint t=0;t<256;++t) {uint a=greater[t],b=equal[t];greater[t]=g;equal[t]=e;g+=a;e+=b;}
      remaining=g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g=greater[tid],e=equal[tid];
    for(uint i=first;i<last;++i) {
      uint code=score_key(s[i]);
      if(code>prefix)out[g++]=int(i);
      else if(code==prefix) {if(remaining+e<K)out[remaining+e]=int(i);++e;}
    }
    """,
    )


def topk(scores, k):
    assert scores.ndim == 3 and 0 < k <= scores.shape[-1]
    b, l, n = scores.shape
    ids = (_radix_compact() if DS41_TOPK_COMPACT else _radix())(
        inputs=[scores, mx.array([k], mx.uint32)],
        grid=(256 * b * l, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, l, k)],
        output_dtypes=[mx.int32],
    )[0]
    return mx.sort(ids, axis=-1)


@cache
def _packed_score():
    return mx.fast.metal_kernel(
        name="ds41_packed_index_nax",
        input_names=[
            "q",
            "qs",
            "qfull",
            "qscale",
            "qexact",
            "keys",
            "weights",
            "candidates",
            "meta",
        ],
        output_names=["scores"],
        header=_HEADER,
        source=r"""
    uint tile=threadgroup_position_in_grid.x,query=threadgroup_position_in_grid.y;
    uint tid=thread_index_in_threadgroup,first=tile*64;
    uint n=meta[0],width=meta[1],start=meta[2],ratio=meta[3],key_start=meta[4];
    threadgroup int key_rows[64],emin[64];threadgroup uint bad[128];
    if(tid<32)bad[tid]=!qexact[query*32+tid];
    if(tid<64) {
      uint pos=first+tid;int row=pos<width?(HAS_CAND?candidates[query*width+pos]:int(pos)):-1;
      if(row<0 || row>=int(n) || row+int(key_start)>=int((start+query+1)/ratio))row=-1;
      key_rows[tid]=row;int lo=127,hi=127;
      if(row>=0){lo=255;hi=0;for(uint g=0;g<4;++g){int e=keys[size_t(row)*68+64+g];lo=min(lo,e);hi=max(hi,e);}}
      emin[tid]=lo;bad[tid+32]=hi-lo>3;
    }
    // q flags [0,32) and key flags [32,96) are written by disjoint logical rows.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){uint any=0;for(uint i=0;i<96;++i)any|=bad[i];bad[0]=any;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool full=bad[0]==0;uint inner=full?128:32;
    threadgroup int8_t tile_keys[64*128];
    threadgroup float tile_scales[64],dots[32*64];
    constexpr auto desc=matmul2d_descriptor(32,64,static_cast<int>(dynamic_extent),false,true,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    using QTensor=tensor<device int8_t,ds41_d2,tensor_inline>;
    using KTensor=tensor<threadgroup int8_t,ds41_d2,tensor_inline>;
    QTensor proto_q((device int8_t*)q,ds41_d2(32,32));KTensor proto_k(tile_keys,ds41_d2(32,64));
    auto acc=op.template get_destination_cooperative_tensor<decltype(proto_q),decltype(proto_k),int>();
    float values[32];for(uint i=0;i<acc.get_capacity();++i)values[i]=0;
    constexpr int levels[8]={0,1,2,3,4,6,8,12};
    for(uint group=0;group<(full?1u:4u);++group) {
      for(uint i=tid;i<64*inner;i+=128) {
        uint row=i/inner,d=i%inner,g=full?d/32:group;uint channel=full?d:group*32+d;
        int source=key_rows[row];int value=0;
        if(source>=0){size_t base=size_t(source)*68;uchar code=(keys[base+channel/2]>>((channel%2)*4))&15;
          value=levels[code&7]*(code&8?-1:1);
          if(full)value*=1 << (int(keys[base+64+g])-emin[row]);}
        tile_keys[i]=char(value);
      }
      if(tid<64){int source=key_rows[tid];int e=full?emin[tid]:(source>=0?int(keys[size_t(source)*68+64+group]):128);tile_scales[tid]=ldexp(1.0f,e-128);}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      QTensor a(full?(device int8_t*)qfull+size_t(query)*32*128:(device int8_t*)q+(size_t(query)*4+group)*32*32,ds41_d2(inner,32));
      KTensor b(tile_keys,ds41_d2(inner,64));op.run(a,b,acc);
      for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
        auto ij=acc.get_multidimensional_index(i);uint head=ij[1],col=ij[0];
        values[i]+=float(acc[i])*(full?qscale[query*32+head]:qs[(size_t(query)*4+group)*32+head])*tile_scales[col];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)){auto ij=acc.get_multidimensional_index(i);dots[ij[1]*64+ij[0]]=values[i];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<64 && first+tid<width) {
      float total=0;for(uint h=0;h<32;++h)total+=max(dots[h*64+tid],0.0f)*weights[query*32+h];
      scores[size_t(query)*width+first+tid]=key_rows[tid]>=0?total:-INFINITY;
    }
    """,
    )


def packed_scores(q, keys, weights, start, ratio, candidates=None, *, key_start=0):
    """NAX score/ReLU/head reduction with packed FP4 keys gathered in the kernel."""
    from .kernels import packed_index_scores as reference

    if q.shape[0] != 1 or q.shape[2:] != (32, 128):
        return reference(
            q, keys, weights, start, ratio, candidates, key_start=key_start
        )
    n = keys.shape[1]
    length = q.shape[1]
    width = n if candidates is None else candidates.shape[-1]
    if not n or not width:
        return mx.full((1, length, width), -mx.inf, mx.float32)
    qi, qs = pack(q.reshape(length, 32, 128))
    qf, qscale, qexact = pack_full(q.reshape(length, 32, 128))
    return _packed_score()(
        inputs=[
            qi,
            qs,
            qf,
            qscale,
            qexact,
            keys,
            weights,
            candidates if candidates is not None else mx.zeros((1,), mx.int32),
            mx.array([n, width, start, ratio, key_start], mx.uint32),
        ],
        template=[("HAS_CAND", candidates is not None)],
        grid=(128 * ((width + 63) // 64), length, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, length, width)],
        output_dtypes=[mx.float32],
    )[0]
