import sys, collections, re, numpy as np
sys.path.insert(0,'/mnt/nvme-1/split-nv-ops/box-perf/prof')
import nsys_an as N
def pbucket(tr,k,st):
    nm=tr.name(k); p="/".join(st)
    if 'AllReduce' in nm or 'cross_device' in nm: return 'allreduce (NCCL)'
    if nm.startswith('pf_') or nm=='quant_x_kernel' or ('m:mlp' in p and 'gate' not in p and '_linear' not in nm and 'router' not in nm): return 'moe (og_moe prefill)'
    if 'm:mlp' in p: return 'moe router'
    if nm.startswith('sparse_mla'): return 'attention kernel'
    if 'engram_gather' in nm: return 'engram host gather'
    if 'engram' in p: return 'engram other (wkv GEMM, gate)'
    if nm=='Kernel2': return 'wo_a (bf16 einsum)'
    if nm.startswith(('_hc_','mhc_','_mhc')) or 'mhc' in p: return 'mHC'
    if re.search('mqa_logits|topk|radix|Topk',nm) or 'index_topk' in p or 'idx.' in p: return 'indexer'
    if 'low_ratio' in p or 'cmp.' in p or 'compressor' in p: return 'compressor'
    if nm.startswith('kernel_cutlass_kernel_flashinf') and ('wq_b' in p or 'wo_b' in p or 'wqkv_a' in p): return 'dense FP8 GEMM (wqkv_a, wq_b, wo_b)'
    if 'layernorm' in p or 'q_norm' in p or 'rope' in nm: return 'norm/rope'
    return 'other'
db, rx = sys.argv[1], sys.argv[2]
tr=N.Trace(db)
for dev in (0,1):
    steps=N.step_kernels(tr,'prefill_chunk',dev,'',None,rx)
    n=len(steps); acc=collections.defaultdict(float); cpy=0; walls=[]; busy=[]
    for s,e,text,ks,cps,sets in steps:
        walls.append((e-s)/1e6)
        busy.append(N.union_busy([(k[0][0],k[0][1]) for k in ks]+[(m[0],m[1]) for m in cps+sets])/1e6)
        for k,st in ks: acc[pbucket(tr,k,st)]+=(k[1]-k[0])/1e6
        cpy+=sum(m[1]-m[0] for m in cps)/1e6
    W=np.median(walls)
    print(f"dev{dev}: n={n} wall {W:.1f} ms busy {np.median(busy):.1f} idle {W-np.median(busy):.1f}; memcpy {cpy/n:.1f} ms")
    for b,v in sorted(acc.items(),key=lambda x:-x[1]): print(f"   {b:38s} {v/n:7.1f} ms {100*v/n/W:5.1f}%")
