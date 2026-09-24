# SPDX-License-Identifier: MIT
"""Bounded prefill index scoring on NAX, followed by exact radix selection."""
from functools import cache
import mlx.core as mx
from .index_nax import topk
from .quantization import unpack_activation

@cache
def _dense_score(paired):
 return mx.fast.metal_kernel(
  name='ds41_prefill_index_dense'+str(int(paired)),
  input_names=['q','keys','weights','candidates','meta'],output_names=['scores'],
  header=r'''
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using D2 = dextents<int32_t,2>;
''',source=r'''
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


def select(q,keys,weights,start,ratio,count,*,score_fn=None,query_chunk=1024):
 from . import kernels
 width=keys.shape[1]
 # At most 256MiB of scores per group, including contexts beyond this benchmark.
 query_chunk=min(query_chunk,max(1,(256*1024**2)//max(4*width,1)))
 dense=None
 if kernels.DS41_PREFILL_INDEX==2:
  dense=mx.pad(unpack_activation(keys,bits=4),[(0,0),(0,-width%64),(0,0)])
  mx.eval(dense)
 results=[];pending=None
 for begin in range(0,q.shape[1],query_chunk):
  end=min(begin+query_chunk,q.shape[1])
  values=(dense_scores(q[:,begin:end],dense,weights[:,begin:end],start+begin,ratio,width)
          if dense is not None else score_fn(q[:,begin:end],keys,weights[:,begin:end],start+begin,ratio))
  ids=topk(values,count)
  out=mx.sort(mx.where(mx.take_along_axis(values,ids,-1)>-float('inf'),ids,-1),axis=-1)
  if pending is not None:mx.eval(pending)
  mx.async_eval(out);pending=out;results.append(out)
 if pending is not None:mx.eval(pending)
 return mx.concatenate(results,1)
