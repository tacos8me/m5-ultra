"""ds41-ttft2: micro-profile of one replay layer's pieces at 128 / 24 rows (chains of K launches, 3 distinct
loaded layers rotated so no weight stays in SLC). Partial load; run under ttft2_guard.py."""
import os, sys, time, json, statistics
sys.argv = [sys.argv[0]]
os.environ['RP_TESTS'] = 'none'
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
from omlx.patches.deepseek_v41 import routing
from omlx.patches.deepseek_v4.switch_layers import _block_config, _build_mxfp4_blocks
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
lm = load()
st = case_state(os.environ.get('RP_CASE', '8217@8217'))
CAP = {}
blk, att = language.Block.__call__, language.Attention.__call__
def cap(self, h, pre, cache, shared, start, image_mask, **k):
    CAP.setdefault(('blk', h.shape[1]), []).append((self, h, pre, dict(k)))
    return blk(self, h, pre, cache, shared, start, image_mask, **k)
def capa(self, x, cache, shared, start, *a, **k):
    CAP.setdefault(('attn', x.shape[1]), []).append((self, x))
    return att(self, x, cache, shared, start, *a, **k)
language.Block.__call__, language.Attention.__call__ = cap, capa
drop(lm, run_import(lm, *st[:3], st[3]))
language.Block.__call__, language.Attention.__call__ = blk, att
blocks = [lm.layers[i] for i in (20, 24, 25)]
K = 9

def chain(name, rows, fn):
    ts = []
    for rep in range(4):
        outs = []
        mx.synchronize()
        t = time.perf_counter()
        for k in range(K):
            o = fn(blocks[k % 3])
            outs.append(o)
        mx.eval(outs)
        ts.append((time.perf_counter() - t) / K)
    emit(dict(kind='micro', rows=rows, op=name, ms=round(1000 * statistics.median(ts[1:]), 3)))

for rows in (128, 24):
    _, h, pre, _ = CAP[('blk', rows)][5]
    x = language.hc_pre_norm(h, pre, blocks[2].ffn_norm.weight, blocks[2].ffn_norm.eps)
    mx.eval(x)
    q = language.quantize_activation
    chain('block_total', rows, lambda b: b(h, pre, *make_cache_dummy(b)) if False else b.ffn(x, None))
    chain('moe.gate', rows, lambda b: b.ffn.gate(x, None))
    chain('quantize_act', rows, lambda b: q(x))
    chain('shared_experts', rows, lambda b: b.ffn.shared_experts(q(x), input_quantized=True))
    idx, wts = blocks[2].ffn.gate(x, None); mx.eval(idx, wts)
    xq = q(x); mx.eval(xq)
    if rows >= 32:
        flat = idx.reshape(-1); order = mx.argsort(flat); inverse = mx.argsort(order)
        selected = xq.reshape(-1, x.shape[-1])[order // 6][:, None, :]
        fo, wo = flat[order], wts.reshape(-1)[order]
        mx.eval(selected, fo, wo, inverse)
        chain('sort+gather', rows, lambda b: xq.reshape(-1, x.shape[-1])[mx.argsort(flat) // 6])
        chain('routed_experts', rows, lambda b: b.ffn.experts(selected, fo, wo, sorted_indices=True, input_quantized=True))
        e = blocks[2].ffn.experts
        bm, variant = _block_config(fo.size, 'mxfp4')
        chain('block_plan', rows, lambda b: _build_mxfp4_blocks(fo, 384, bm)[0])
        meta, count = _build_mxfp4_blocks(fo, 384, bm); mx.eval(meta, count)
        chain('pair_gate_up', rows, lambda b: glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
            selected, b.ffn.experts.w1.weight, b.ffn.experts.w1.scales, b.ffn.experts.w3.weight, b.ffn.experts.w3.scales,
            meta, count, variant))
        pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, e.w1.weight, e.w1.scales, e.w3.weight, e.w3.scales, meta, count, variant)
        y = language.quantize_paired_swiglu_activation(pair, wo, mx.bfloat16, e._limit); mx.eval(y)
        chain('swiglu_q', rows, lambda b: language.quantize_paired_swiglu_activation(pair, wo, mx.bfloat16, e._limit))
        chain('w2_blocks', rows, lambda b: b.ffn.experts.w2.project_quantized(y, fo, True, (meta, count, variant)))
        routed = e(selected, fo, wo, sorted_indices=True, input_quantized=True)
        shared = blocks[2].ffn.shared_experts(xq, input_quantized=True); mx.eval(routed, shared)
        chain('combine', rows, lambda b: routing.combine_sorted_experts(routed, inverse, shared))
    else:
        chain('routed_experts', rows, lambda b: b.ffn.experts(xq[..., None, None, :], idx, wts, input_quantized=True))
    # hc pieces
    c = lm._config
    chain('hc_mixes', rows, lambda b: language.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, c)[2])
    chain('hc_pre_norm', rows, lambda b: language.hc_pre_norm(h, pre, b.attn_norm.weight, b.attn_norm.eps))
    chain('hc_post', rows, lambda b: language.hc_post(x, h, pre, mx.broadcast_to(pre[..., None], (*pre.shape, 4))))
    # attention projections
    _, xa = CAP[('attn', rows)][5]
    chain('attn.in_proj', rows, lambda b: b.attn._input_projections(xa)[0])
    qa, kva = blocks[2].attn._input_projections(xa); mx.eval(qa)
    chain('attn.q_norm+wq_b', rows, lambda b: b.attn.wq_b(b.attn.q_norm(qa)))
    o = mx.random.normal((1, rows, c.n_heads * c.head_dim)).astype(mx.bfloat16)
    chain('attn.wo_a', rows, lambda b: mx.einsum('bsgd,grd->bsgr', o.reshape(1, rows, c.o_groups, -1), b.attn.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)))
    oa = mx.random.normal((1, rows, c.o_groups * c.o_lora_rank)).astype(mx.bfloat16)
    chain('attn.wo_b', rows, lambda b: b.attn.wo_b(oa))
emit(dict(event='done', types={k: str(getattr(blocks[2].attn, k).weight.dtype) + str(getattr(blocks[2].attn, k).weight.shape) for k in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b')}))
