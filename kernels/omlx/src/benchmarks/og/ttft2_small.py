"""ds41-ttft2: routed-expert rows-per-expert histogram at replay row counts + small-block hybrid (scalar chain
for experts with <= S rows, steel v0 for the rest): bitwise vs v0 and time."""
import os, sys, time, json, statistics, importlib.util, collections
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
spec = importlib.util.spec_from_file_location('moe_small', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_moe_small.py'))
ms_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms_)
lm = load()
CAP = {}
blk = language.Block.__call__
def cap(self, h, pre, cache, shared, start, image_mask, **k):
    CAP.setdefault(h.shape[1], []).append((self, h, pre))
    return blk(self, h, pre, cache, shared, start, image_mask, **k)
language.Block.__call__ = cap
for case in ('8217@8217', '8600@8600'):
    st = case_state(case)
    drop(lm, run_import(lm, *st[:3], st[3]))
language.Block.__call__ = blk
blocks = [lm.layers[i] for i in (20, 24, 25)]
KREP = 9
def timeit(fn):
    ts = []
    for rep in range(4):
        mx.synchronize(); t = time.perf_counter()
        mx.eval([fn(blocks[k % 3]) for k in range(KREP)])
        ts.append((time.perf_counter() - t) / KREP)
    return round(1000 * statistics.median(ts[1:]), 3)
hist = collections.Counter()
for li in range(len(CAP[128])):
    b, h, pre = CAP[128][li]
    x = language.hc_pre_norm(h, pre, b.ffn_norm.weight, b.ffn_norm.eps)
    idx, _ = b.ffn.gate(x, None)
    for e, n in collections.Counter(idx.reshape(-1).tolist()).items():
        hist[min(n, 20)] += 1
emit(dict(kind='hist', rows_per_expert=dict(sorted(hist.items())), layers=len(CAP[128])))
for li in (5, 12, 25):
    b2, h, pre = CAP[128][li]
    b2 = blocks[2]
    x = language.hc_pre_norm(h, pre, b2.ffn_norm.weight, b2.ffn_norm.eps)
    idx, wts = b2.ffn.gate(x, None)
    xq = language.quantize_activation(x)
    flat = idx.reshape(-1); order = mx.argsort(flat)
    fo, wo = flat[order], wts.reshape(-1)[order]
    selected = xq.reshape(-1, x.shape[-1])[order // 6][:, None, :]
    e = b2.ffn.experts
    meta, count = _build_mxfp4_blocks(fo, 384, 8)
    mx.eval(selected, fo, wo, meta, count)
    ref = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, e.w1.weight, e.w1.scales, e.w3.weight, e.w3.scales, meta, count, 0)
    y = language.quantize_paired_swiglu_activation(ref, wo, mx.bfloat16, e._limit)
    ref2 = e.w2.project_quantized(y, fo, True, (meta, count, 0))
    mx.eval(ref, y, ref2)
    t_ref = timeit(lambda b: glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, b.ffn.experts.w1.weight, b.ffn.experts.w1.scales, b.ffn.experts.w3.weight, b.ffn.experts.w3.scales, meta, count, 0))
    t_ref2 = timeit(lambda b: b.ffn.experts.w2.project_quantized(y, fo, True, (meta, count, 0)))
    for S in (1, 2, 3, 4):
        msm, csm, mbg, cbg = ms_.split(meta, count, S)
        mask = ms_.mask(msm, csm, fo.size)
        mx.eval(msm, csm, mbg, cbg, mask)
        def hyb(b, sel=selected):
            ee = b.ffn.experts
            big = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(sel, ee.w1.weight, ee.w1.scales, ee.w3.weight, ee.w3.scales, mbg, cbg.astype(mx.int32), 0)
            sm = ms_.small(sel, [(ee.w1.weight, ee.w1.scales), (ee.w3.weight, ee.w3.scales)], msm, csm, S)
            return mx.where(mask[:, None, None].astype(mx.bool_), sm, big)
        def hyb2(b):
            ee = b.ffn.experts
            big = ee.w2.project_quantized(y, fo, True, (mbg, cbg.astype(mx.int32), 0))
            sm = ms_.small(y, [(ee.w2.weight, ee.w2.scales)], msm, csm, S)
            return mx.where(mask[:, None, None].astype(mx.bool_), sm, big)
        o1, o2 = hyb(b2), hyb2(b2)
        mx.eval(o1, o2)
        bad1 = int(mx.sum(o1.view(mx.uint16) != ref.view(mx.uint16)).item())
        bad2 = int(mx.sum(o2.view(mx.uint16) != ref2.view(mx.uint16)).item())
        emit(dict(kind='small', layer=li, S=S, small_blocks=int(csm.item()), big_blocks=int(cbg.item()), blocks=int(count.item()),
                  mismatch=[bad1, bad2], gate_up_ms=timeit(hyb), down_ms=timeit(hyb2), ref_gate_up_ms=t_ref, ref_down_ms=t_ref2))
