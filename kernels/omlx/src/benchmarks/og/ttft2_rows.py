"""ds41-ttft2: moe_rows prototype vs the served block kernels: bitwise (real + adversarial) and time."""
import os, sys, time, json, statistics, importlib.util
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
spec = importlib.util.spec_from_file_location('moe_rows', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_moe_rows.py'))
mr = importlib.util.module_from_spec(spec); spec.loader.exec_module(mr)
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
KREP = 9
def timeit(fn):
    ts = []
    for rep in range(4):
        mx.synchronize(); t = time.perf_counter()
        mx.eval([fn(blocks[k % 3]) for k in range(KREP)])
        ts.append((time.perf_counter() - t) / KREP)
    return round(1000 * statistics.median(ts[1:]), 3)
CONFIGS = [(int(a), int(b), int(c)) for a, b, c in (s.split(':') for s in os.environ.get('RP_CFG', '8:128:128,16:128:128,8:256:128,8:64:128,4:128:128').split(','))]
for li in (5, 12):
    _, h, pre = CAP[128][li]
    b2 = blocks[2]
    x = language.hc_pre_norm(h, pre, b2.ffn_norm.weight, b2.ffn_norm.eps)
    idx, wts = b2.ffn.gate(x, None)
    xq = language.quantize_activation(x)
    flat = idx.reshape(-1); order = mx.argsort(flat)
    fo, wo = flat[order], wts.reshape(-1)[order]
    for adv in (False, True):
        selected = xq.reshape(-1, x.shape[-1])[order // 6][:, None, :]
        if adv:
            mx.random.seed(li)
            selected = (mx.random.normal(selected.shape) * mx.power(2.0, mx.random.randint(-12, 12, selected.shape).astype(mx.float32))).astype(mx.bfloat16)
        mx.eval(selected, fo, wo)
        e = b2.ffn.experts
        meta, count = _build_mxfp4_blocks(fo, 384, 16); mx.eval(meta, count)
        ref = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, e.w1.weight, e.w1.scales, e.w3.weight, e.w3.scales, meta, count, 1)
        y = language.quantize_paired_swiglu_activation(ref, wo, mx.bfloat16, e._limit)
        if adv:
            y = (mx.random.normal(y.shape) * mx.power(2.0, mx.random.randint(-12, 12, y.shape).astype(mx.float32))).astype(mx.bfloat16)
        ref2 = e.w2.project_quantized(y, fo, True, (meta, count, 1))
        mx.eval(ref, y, ref2)
        t_ref = timeit(lambda b: glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, b.ffn.experts.w1.weight, b.ffn.experts.w1.scales, b.ffn.experts.w3.weight, b.ffn.experts.w3.scales, meta, count, 1))
        t_ref2 = timeit(lambda b: b.ffn.experts.w2.project_quantized(y, fo, True, (meta, count, 1)))
        for R, TN, KC in CONFIGS:
            m2, c2 = _build_mxfp4_blocks(fo, 384, R); mx.eval(m2, c2)
            out = mr.run(selected, [(e.w1.weight, e.w1.scales), (e.w3.weight, e.w3.scales)], m2, c2, rows_per_block=R, tn=TN, kc=KC)
            out2 = mr.run(y, [(e.w2.weight, e.w2.scales)], m2, c2, rows_per_block=R, tn=TN, kc=KC)
            mx.eval(out, out2)
            bad = int(mx.sum(out.view(mx.uint16) != ref.view(mx.uint16)).item())
            bad2 = int(mx.sum(out2.view(mx.uint16) != ref2.view(mx.uint16)).item())
            t1 = timeit(lambda b: mr.run(selected, [(b.ffn.experts.w1.weight, b.ffn.experts.w1.scales), (b.ffn.experts.w3.weight, b.ffn.experts.w3.scales)], m2, c2, rows_per_block=R, tn=TN, kc=KC))
            t2 = timeit(lambda b: mr.run(y, [(b.ffn.experts.w2.weight, b.ffn.experts.w2.scales)], m2, c2, rows_per_block=R, tn=TN, kc=KC))
            emit(dict(kind='rows', layer=li, adv=adv, R=R, TN=TN, KC=KC, blocks=int(c2.item()), mismatch_gate_up=bad, mismatch_down=bad2,
                      gate_up_ms=t1, down_ms=t2, ref_gate_up_ms=t_ref, ref_down_ms=t_ref2))
