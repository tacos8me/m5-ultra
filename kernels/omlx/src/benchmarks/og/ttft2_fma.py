"""ds41-ttft2: is the MXFP4 block kernel's per-element arithmetic a plain sequential fp32 chain over k?"""
import os, sys, time, json
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
b2 = lm.layers[25]
_, h, pre = CAP[128][5]
x = language.hc_pre_norm(h, pre, b2.ffn_norm.weight, b2.ffn_norm.eps)
idx, wts = b2.ffn.gate(x, None)
xq = language.quantize_activation(x)
flat = idx.reshape(-1); order = mx.argsort(flat)
selected = xq.reshape(-1, x.shape[-1])[order // 6][:, None, :]
fo = flat[order].astype(mx.uint32)
e = b2.ffn.experts
meta, count = _build_mxfp4_blocks(flat[order], 384, 16)
pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, e.w1.weight, e.w1.scales, e.w3.weight, e.w3.scales, meta, count, 1)
mx.eval(pair, selected, fo)
if os.environ.get('ADV', '1') == '1':
    mx.random.seed(7)
    adv = (mx.random.normal(selected.shape) * mx.power(2.0, mx.random.randint(-12, 12, selected.shape).astype(mx.float32))).astype(mx.bfloat16)
    selected = adv
    pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(selected, e.w1.weight, e.w1.scales, e.w3.weight, e.w3.scales, meta, count, 1)
    mx.eval(pair, selected)
N, K = e.w1.weight.shape[1], e.w1.weight.shape[2] * 8
print('shapes', selected.shape, e.w1.weight.shape, e.w1.scales.shape, pair.shape, N, K, flush=True)
HDR = r"""
constant float LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e8(uint8_t b) { uint16_t o = (b == 0 ? 0x40 : (uint16_t(b) << 7)); return float(as_type<bfloat16_t>(o)); }
"""
def src_for(mode):
    body = r"""
    uint n = thread_position_in_grid.x, r = thread_position_in_grid.y;
    if (n >= NN || r >= MM) return;
    uint ex = ids[r];
    const device uint8_t* wr = (const device uint8_t*)w + (size_t(ex) * NN + n) * (KK / 2);
    const device uint8_t* sr = sc + (size_t(ex) * NN + n) * (KK / 32);
    const device bfloat16_t* xr = xs + size_t(r) * KK;
    float acc = 0.0f;
    for (uint k0 = 0; k0 < KK; k0 += 8) {
      float p[8];
      for (int j = 0; j < 8; j++) {
        uint k = k0 + j;
        float s = e8(sr[k / 32]);
        uint8_t byte = wr[k / 2];
        uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 15);
        float wv = float(static_cast<bfloat16_t>(s * LUT[nib]));
        p[j] = float(xr[k]) * wv;
      }
      MODE
    }
    out[size_t(r) * NN + n] = static_cast<bfloat16_t>(acc);
    """
    modes = {
        'seq': 'for (int j = 0; j < 8; j++) acc = fma(1.0f, p[j], acc);',
        'chunk_seq': 'float t = p[0]; for (int j = 1; j < 8; j++) t = t + p[j]; acc = acc + t;',
        'chunk_tree': 'float t = ((p[0]+p[1])+(p[2]+p[3]))+((p[4]+p[5])+(p[6]+p[7])); acc = acc + t;',
        'chunk_pairs_seq': 'float t = (p[0]+p[1]); t = t + (p[2]+p[3]); t = t + (p[4]+p[5]); t = t + (p[6]+p[7]); acc = acc + t;',
        'acc_first_tree': 'acc = ((acc + p[0]) + p[1]) + ((p[2]+p[3]) + ((p[4]+p[5])+(p[6]+p[7])));',
        'half_seq': 'float a = (p[0]+p[1])+(p[2]+p[3]); float b = (p[4]+p[5])+(p[6]+p[7]); acc = (acc + a) + b;',
        'quad_acc': 'acc = acc + ((p[0]+p[1])+(p[2]+p[3])); acc = acc + ((p[4]+p[5])+(p[6]+p[7]));',
        'pair_acc': 'acc = acc + (p[0]+p[1]); acc = acc + (p[2]+p[3]); acc = acc + (p[4]+p[5]); acc = acc + (p[6]+p[7]);',
    }
    return body.replace('MODE', modes[mode]).replace('NN', str(N)).replace('KK', str(K)).replace('MM', str(selected.shape[0]))
ref = pair[:, 0, :N]
rows = selected.shape[0]
for mode in ('seq', 'chunk_seq', 'chunk_tree', 'chunk_pairs_seq', 'acc_first_tree', 'half_seq', 'quad_acc', 'pair_acc'):
    kern = mx.fast.metal_kernel(name='ttft2_fma_' + mode, input_names=['xs', 'w', 'sc', 'ids'], output_names=['out'],
                                source=src_for(mode), header=HDR)
    out = kern(inputs=[selected.reshape(rows, K), e.w1.weight, e.w1.scales, fo], grid=(N, rows, 1), threadgroup=(64, 1, 1),
               output_shapes=[(rows, N)], output_dtypes=[mx.bfloat16])[0]
    mx.eval(out)
    diff = (out.view(mx.uint16) != ref.view(mx.uint16))
    nd = int(mx.sum(diff).item())
    maxrel = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item())
    print(json.dumps(dict(mode=mode, mismatches=nd, of=rows * N, max_abs=maxrel)), flush=True)
