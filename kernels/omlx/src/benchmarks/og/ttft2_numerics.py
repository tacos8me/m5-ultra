"""ds41-ttft2: are FFN/hc pieces row-count invariant (bitwise)? Partial load; captured real layer inputs."""
import os, sys, time, json
sys.argv = [sys.argv[0]]
os.environ['RP_TESTS'] = 'none'
os.environ['RP_CASES'] = ''
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
lm = load()
st = case_state('8217@8217')
CAP = {}
blk = language.Block.__call__
count = [0]
def cap(self, h, pre, cache, shared, start, image_mask, **k):
    count[0] += 1
    CAP.setdefault(h.shape[1], []).append((self, h, pre))
    return blk(self, h, pre, cache, shared, start, image_mask, **k)
language.Block.__call__ = cap
drop(lm, run_import(lm, *st[:3], st[3]))
language.Block.__call__ = blk
c = lm._config
def eq(a, b):
    a, b = mx.array(a), mx.array(b)
    if a.shape != b.shape: return 'shape'
    return bool(mx.all(a.view(mx.uint8) == b.view(mx.uint8)).item()) if a.dtype == b.dtype else 'dtype'
res = {}
for li in (1, 5, 10):   # captured layers 21, 25, 30 (0 = layer 20)
    b0, h0, p0 = CAP[128][li]
    b1, h1, p1 = CAP[24][li]
    assert b0 is b1
    blk_ = b0
    x0 = language.hc_pre_norm(h0, p0, blk_.ffn_norm.weight, blk_.ffn_norm.eps)
    x1 = language.hc_pre_norm(h1, p1, blk_.ffn_norm.weight, blk_.ffn_norm.eps)
    hh = mx.concatenate([h0, h1], 1); pp = mx.concatenate([p0, p1], 1)
    xx = language.hc_pre_norm(hh, pp, blk_.ffn_norm.weight, blk_.ffn_norm.eps)
    r = dict(pre_norm=eq(xx, mx.concatenate([x0, x1], 1)))
    g0, g1, gg = blk_.ffn.gate(x0, None), blk_.ffn.gate(x1, None), blk_.ffn.gate(xx, None)
    r['gate_idx'] = eq(gg[0], mx.concatenate([g0[0], g1[0]], 1)); r['gate_w'] = eq(gg[1], mx.concatenate([g0[1], g1[1]], 1))
    m0, m1, mm = blk_.ffn(x0, None), blk_.ffn(x1, None), blk_.ffn(xx, None)
    r['moe_152_vs_128'] = eq(mm[:, :128], m0); r['moe_152_vs_24'] = eq(mm[:, 128:], m1)
    # sorted path forced for 24 rows by concatenating twice? use 32-row pad: rows of x1 + first 8 of x0
    s0 = blk_.ffn.shared_experts
    q = language.quantize_activation
    sh0, sh1, shh = s0(q(x0), input_quantized=True), s0(q(x1), input_quantized=True), s0(q(xx), input_quantized=True)
    r['shared_152'] = eq(shh, mx.concatenate([sh0, sh1], 1))
    for n in (8, 16, 32, 64):
        sn = s0(q(xx[:, :n]), input_quantized=True)
        r[f'shared_{n}_vs_152'] = eq(sn, shh[:, :n])
    mix0 = language.hc_mixes(h0, blk_.hc_ffn_fn, blk_.hc_ffn_scale, blk_.hc_ffn_base, c)
    mix1 = language.hc_mixes(h1, blk_.hc_ffn_fn, blk_.hc_ffn_scale, blk_.hc_ffn_base, c)
    mixx = language.hc_mixes(hh, blk_.hc_ffn_fn, blk_.hc_ffn_scale, blk_.hc_ffn_base, c)
    r['hc_mixes'] = [eq(mixx[j], mx.concatenate([mix0[j], mix1[j]], 1)) for j in range(3)]
    # attention-side input projections
    at = blk_.attn
    a0, a1, aa = at._input_projections(x0), at._input_projections(x1), at._input_projections(xx)
    r['wq_a'] = eq(aa[0], mx.concatenate([a0[0], a1[0]], 1)); r['wkv'] = eq(aa[1], mx.concatenate([a0[1], a1[1]], 1))
    res[li + 20] = r
    print(json.dumps(dict(layer=li + 20, **{k: v for k, v in r.items()})), flush=True)
