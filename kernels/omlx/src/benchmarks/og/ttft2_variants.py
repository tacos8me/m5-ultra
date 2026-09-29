"""ds41-ttft2: MXFP4 block-kernel variants 0-4 at replay row counts: time + bitwise vs the served variant."""
import os, sys, time, json, statistics
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
lm = load()
st = case_state('8217@8217')
CAP = {}
blk = language.Block.__call__
def cap(self, h, pre, cache, shared, start, image_mask, **k):
    CAP.setdefault(h.shape[1], []).append((self, h, pre))
    return blk(self, h, pre, cache, shared, start, image_mask, **k)
language.Block.__call__ = cap
drop(lm, run_import(lm, *st[:3], st[3]))
language.Block.__call__ = blk
blocks = [lm.layers[i] for i in (20, 24, 25)]
BM = {0: 8, 1: 16, 2: 32, 3: 16, 4: 32}
K = 9
for rows, li in ((128, 5), (128, 12), (256, 5)):
    if rows == 256:
        _, h, pre = CAP[128][5]; _, h2, pre2 = CAP[128][6]
        h, pre = mx.concatenate([h, h2], 1), mx.concatenate([pre, pre2], 1)
    else:
        _, h, pre = CAP[rows][li]
    b2 = blocks[2]
    x = language.hc_pre_norm(h, pre, b2.ffn_norm.weight, b2.ffn_norm.eps)
    idx, wts = b2.ffn.gate(x, None)
    xq = language.quantize_activation(x)
    flat = idx.reshape(-1); order = mx.argsort(flat)
    selected = xq.reshape(-1, x.shape[-1])[order // 6][:, None, :]
    fo, wo = flat[order], wts.reshape(-1)[order]
    mx.eval(selected, fo, wo)
    union = len(set(fo.tolist()))
    e = b2.ffn.experts
    ref = None
    for v in (1, 0, 2, 3, 4):
        meta, count = _build_mxfp4_blocks(fo, 384, BM[v]); mx.eval(meta, count)
        f = lambda b: glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
            selected, b.ffn.experts.w1.weight, b.ffn.experts.w1.scales, b.ffn.experts.w3.weight, b.ffn.experts.w3.scales, meta, count, v)
        pair = f(b2)
        y = language.quantize_paired_swiglu_activation(pair, wo, mx.bfloat16, e._limit)
        down = e.w2.project_quantized(y, fo, True, (meta, count, v))
        mx.eval(pair, down)
        if ref is None:
            ref = (pair, down)
        same = (bool(mx.all(pair.view(mx.uint16) == ref[0].view(mx.uint16)).item()),
                bool(mx.all(down.view(mx.uint16) == ref[1].view(mx.uint16)).item()))
        ts, ts2 = [], []
        for rep in range(4):
            mx.synchronize(); t = time.perf_counter()
            mx.eval([f(blocks[k % 3]) for k in range(K)])
            ts.append((time.perf_counter() - t) / K)
            mx.synchronize(); t = time.perf_counter()
            mx.eval([blocks[k % 3].ffn.experts.w2.project_quantized(y, fo, True, (meta, count, v)) for k in range(K)])
            ts2.append((time.perf_counter() - t) / K)
        emit(dict(kind='variant', rows=rows, layer=li, union=union, variant=v, bitwise_pair_down=same,
                  gate_up_ms=round(1000 * statistics.median(ts[1:]), 3), down_ms=round(1000 * statistics.median(ts2[1:]), 3)))
