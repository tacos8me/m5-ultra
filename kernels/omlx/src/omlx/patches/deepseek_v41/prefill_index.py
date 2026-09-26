# SPDX-License-Identifier: MIT
"""Bounded prefill index scoring on NAX, followed by exact radix selection."""
import os
from functools import cache
from collections import deque
import mlx.core as mx
from .index_nax import topk
from .quantization import unpack_activation

# Dense score tile geometry. 0: the original kernel (two queries x one 64-key tile per
# threadgroup, key tiles as the fast dispatch axis). 1: the same per-element NAX
# arithmetic and the same sequential ReLU/weight head sum, but QROWS queries x KCOLS keys
# per threadgroup with query groups as the fast dispatch axis, so a key tile stays
# cache-resident across the queries of a group instead of being re-streamed from
# memory for every pair. The per-key score values are bitwise those of geometry 0.
DS41_PREFILL_INDEX_TILE = int(os.environ.get('DS41_PREFILL_INDEX_TILE', '1'))
# Two bounded score groups overlap GPU submission with host preparation.
DS41_PREFILL_INDEX_PIPELINE = max(1, min(2, int(os.environ.get('DS41_PREFILL_INDEX_PIPELINE', '2'))))
TILE_QROWS = int(os.environ.get('DS41_PREFILL_INDEX_QROWS', '2'))
TILE_KCOLS = int(os.environ.get('DS41_PREFILL_INDEX_KCOLS', '128'))

_HEADER = r'''
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using D2 = dextents<int32_t,2>;
'''


@cache
def _dense_score(paired):
 return mx.fast.metal_kernel(
  name='ds41_prefill_index_dense'+str(int(paired)),
  input_names=['q','keys','weights','candidates','meta'],output_names=['scores'],
  header=_HEADER,source=r'''
 uint tid=thread_index_in_threadgroup;
 uint first=threadgroup_position_in_grid.x*64;
 uint query=threadgroup_position_in_grid.y*QROWS;
 uint length=meta[0],n=meta[1],width=meta[2],start=meta[3],ratio=meta[4],key_start=meta[5];
 threadgroup float dots[QROWS*32*64];
 constexpr auto desc=matmul2d_descriptor(QROWS*32,64,static_cast<int>(dynamic_extent),false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 using A=tensor<device bfloat,D2,tensor_inline>;
 using B=tensor<device bfloat,D2,tensor_inline>;
 A a((device bfloat*)q+size_t(query)*32*128,D2(128,QROWS*32));
 B b((device bfloat*)keys+size_t(first)*128,D2(128,64));
 auto acc=op.template get_destination_cooperative_tensor<A,B,float>();
 op.run(a,b,acc);
 for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
   auto ij=acc.get_multidimensional_index(i);
   dots[ij[1]*64+ij[0]]=acc[i];
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<QROWS*64) {
   uint qi=query+tid/64,col=tid%64,pos=first+col;
   if(qi<length && pos<width) {
     int row=HAS_CAND?candidates[qi*width+pos]:int(pos);
     float sum=0;
     for(uint h=0;h<32;++h)sum+=max(dots[((tid/64)*32+h)*64+col],0.f)*weights[qi*32+h];
     bool valid=row>=0 && row<int(n) && row+int(key_start)<int((start+qi+1)/ratio);
     scores[size_t(qi)*width+pos]=valid?sum:-INFINITY;
   }
 }
''')


def dense_scores(q,dense,weights,start,ratio,width):
 length=q.shape[1];paired=length%2==0
 return _dense_score(paired)(inputs=[q,dense,weights.astype(mx.float32),mx.zeros((1,),mx.int32),mx.array([length,width,width,start,ratio,0],mx.uint32)],template=[('QROWS',2 if paired else 1),('HAS_CAND',False)],grid=(128*((width+63)//64),length//(2 if paired else 1),1),threadgroup=(128,1,1),output_shapes=[(1,length,width)],output_dtypes=[mx.float32])[0]


@cache
def _dense_score_tiled():
 # The head sum runs in PASSES slices of 32/PASSES heads so the staged dot products stay
 # within 16 KiB of threadgroup memory; each query's sum still adds heads 0..31 in order.
 return mx.fast.metal_kernel(
  name='ds41_prefill_index_dense_tiled',
  input_names=['q','keys','weights','meta'],output_names=['scores'],
  header=_HEADER,source=r'''
 uint tid=thread_index_in_threadgroup;
 uint first=threadgroup_position_in_grid.y*KCOLS;
 uint query=threadgroup_position_in_grid.x*QROWS;
 uint length=meta[0],n=meta[1],width=meta[2],start=meta[3],ratio=meta[4],key_start=meta[5];
 constexpr uint HP=32/PASSES;
 threadgroup float dots[QROWS*HP*KCOLS];
 constexpr auto desc=matmul2d_descriptor(QROWS*32,KCOLS,static_cast<int>(dynamic_extent),false,true,false);
 matmul2d<desc,execution_simdgroups<4>> op;
 using A=tensor<device bfloat,D2,tensor_inline>;
 using B=tensor<device bfloat,D2,tensor_inline>;
 A a((device bfloat*)q+size_t(query)*32*128,D2(128,QROWS*32));
 B b((device bfloat*)keys+size_t(first)*128,D2(128,KCOLS));
 auto acc=op.template get_destination_cooperative_tensor<A,B,float>();
 op.run(a,b,acc);
 constexpr uint P=QROWS*KCOLS/128;
 float sum[P];
 for(uint p=0;p<P;++p)sum[p]=0;
 for(uint pass=0;pass<PASSES;++pass){
   if(pass)threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
     auto ij=acc.get_multidimensional_index(i);
     uint r=ij[1],h=r%32;
     if(h/HP==pass)dots[((r/32)*HP+(h%HP))*KCOLS+ij[0]]=acc[i];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint p=0;p<P;++p){
     uint t=tid+p*128,qq=t/KCOLS,col=t%KCOLS,qi=query+qq;
     if(qi<length)for(uint h=0;h<HP;++h)sum[p]+=max(dots[(qq*HP+h)*KCOLS+col],0.f)*weights[qi*32+pass*HP+h];
   }
 }
 for(uint p=0;p<P;++p){
   uint t=tid+p*128,qq=t/KCOLS,col=t%KCOLS,qi=query+qq,pos=first+col;
   if(qi<length && pos<width) {
     int row=int(pos);
     bool valid=row>=0 && row<int(n) && row+int(key_start)<int((start+qi+1)/ratio);
     scores[size_t(qi)*width+pos]=valid?sum[p]:-INFINITY;
   }
 }
''')


def _tile_geometry(length):
 if not DS41_PREFILL_INDEX_TILE or length%TILE_QROWS:
  return None
 return TILE_QROWS,TILE_KCOLS,max(1,TILE_QROWS*TILE_KCOLS//128)


def dense_scores_tiled(q,dense,weights,start,ratio,width):
 length=q.shape[1]
 qrows,kcols,passes=_tile_geometry(length)
 assert dense.shape[1]%kcols==0
 return _dense_score_tiled()(inputs=[q,dense,weights.astype(mx.float32),mx.array([length,width,width,start,ratio,0],mx.uint32)],template=[('QROWS',qrows),('KCOLS',kcols),('PASSES',passes)],grid=(128*(length//qrows),(width+kcols-1)//kcols,1),threadgroup=(128,1,1),output_shapes=[(1,length,width)],output_dtypes=[mx.float32])[0]


def select(q,keys,weights,start,ratio,count,*,score_fn=None,query_chunk=1024):
 from . import kernels
 width=keys.shape[1]
 # At most 256MiB of scores per group, including contexts beyond this benchmark.
 query_chunk=min(query_chunk,max(1,(256*1024**2)//max(4*width,1)))
 # A budget quotient can be odd. Round down so full groups retain the exact
 # paired-query tile instead of silently falling back for this whole layer.
 if DS41_PREFILL_INDEX_TILE and query_chunk>=TILE_QROWS:
  query_chunk-=query_chunk%TILE_QROWS
 dense=None
 geometry=_tile_geometry(query_chunk) if kernels.DS41_PREFILL_INDEX==2 else None
 if kernels.DS41_PREFILL_INDEX==2:
  pad=geometry[1] if geometry else 64
  dense=mx.pad(unpack_activation(keys,bits=4),[(0,0),(0,-width%pad),(0,0)])
  mx.eval(dense)
 results=[];pending=deque()
 for begin in range(0,q.shape[1],query_chunk):
  end=min(begin+query_chunk,q.shape[1])
  if dense is None:
   values=score_fn(q[:,begin:end],keys,weights[:,begin:end],start+begin,ratio)
  elif geometry and (end-begin)%geometry[0]==0:
   values=dense_scores_tiled(q[:,begin:end],dense,weights[:,begin:end],start+begin,ratio,width)
  else:
   values=dense_scores(q[:,begin:end],dense,weights[:,begin:end],start+begin,ratio,width)
  ids=topk(values,count)
  out=mx.sort(mx.where(mx.take_along_axis(values,ids,-1)>-float('inf'),ids,-1),axis=-1)
  mx.async_eval(out);pending.append(out);results.append(out)
  if len(pending)>=DS41_PREFILL_INDEX_PIPELINE:mx.eval(pending.popleft())
 for out in pending:mx.eval(out)
 return mx.concatenate(results,1)
