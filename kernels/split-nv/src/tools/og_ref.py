"""CPU emulation of the official DS-V4.1 inference arithmetic (inference/model.py + kernel.py).

Used to attribute CUDA-vs-Mac drift: each component is recomputed from the same input with the
official quantization points (FP8 act quant g32 UE8M0, FP8 window KV g32 incl. RoPE tail, BF16 Q,
BF16 attention probabilities, route weights before w2's act quant) and compared with both traces.
"""
import json
import math
from functools import lru_cache

import torch
from safetensors import safe_open

CK = '/home/ian/models/DeepSeek-V4.1-Flash-original'
_MAP = json.load(open(f'{CK}/model.safetensors.index.json'))['weight_map']
CFG = json.load(open(f'{CK}/config.json'))
CFG = CFG.get('text_config', CFG)
EPS = CFG['rms_norm_eps']
F32 = torch.float32
BF16 = torch.bfloat16


_DT = {'F8_E8M0': torch.float8_e8m0fnu, 'F8_E4M3': torch.float8_e4m3fn, 'I8': torch.int8, 'U8': torch.uint8,
       'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16, 'I32': torch.int32, 'I64': torch.int64}
_HDR = {}


def W(name):
    """Header-parsing safetensors reader (the venv's safetensors predates F8_E8M0)."""
    path = f'{CK}/{_MAP[name]}'
    if path not in _HDR:
        with open(path, 'rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            _HDR[path] = (8 + n, json.loads(f.read(n)))
    base, hdr = _HDR[path]
    e = hdr[name]
    a, b = e['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + a)
        buf = bytearray(f.read(b - a))
    return torch.frombuffer(buf, dtype=_DT[e['dtype']]).reshape(e['shape'])


def e8m0(s):
    return s.to(F32) if s.dtype != torch.uint8 else torch.ldexp(torch.ones_like(s, dtype=F32), s.to(torch.int32) - 127)


@lru_cache(64)
def fp8_weight(name):
    """Dequantized [out, in] FP32 weight of an FP8 32x32-block linear (bf16 weights pass through)."""
    w = W(name + '.weight')
    if w.dtype == BF16 or w.dtype == F32:
        return w.to(F32)
    s = e8m0(W(name + '.scale'))
    return w.to(F32) * s.repeat_interleave(32, 0)[:w.shape[0]].repeat_interleave(32, 1)[:, :w.shape[1]]


FP4_LEVELS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=F32)


def fp4_weight(name):
    w = W(name + '.weight').view(torch.uint8)
    codes = torch.stack((w & 15, w >> 4), -1).flatten(-2).long()
    s = e8m0(W(name + '.scale'))
    return FP4_LEVELS[codes] * s.repeat_interleave(32, 1)


def pow2_scale(amax, limit):
    return torch.exp2(torch.ceil(torch.log2(amax / limit)))


def act_quant(x, g=32, floor=1e-4):
    """Official act_quant(inplace): FP8 E4M3 with UE8M0 group scales; returns FP32 dequantized values."""
    shape = x.shape
    x = x.to(F32).reshape(*shape[:-1], -1, g)
    s = pow2_scale(x.abs().amax(-1, keepdim=True).clamp_min(floor), 448.0)
    q = (x / s).clamp(-448, 448).to(torch.float8_e4m3fn).to(F32)
    return (q * s).reshape(shape)


def _fp4_round(v):
    a = v.abs()
    q = torch.zeros_like(a)
    for thr, val, inc in [(0.25, .5, False), (0.75, 1., True), (1.25, 1.5, False), (1.75, 2., True),
                          (2.5, 3., False), (3.5, 4., True), (5., 6., False)]:
        q = torch.where(a >= thr if inc else a > thr, torch.full_like(a, val), q)
    return torch.sign(v) * q


def fp4_quant(x, g, e4m3_scale):
    shape = x.shape
    x = x.to(F32).reshape(*shape[:-1], -1, g)
    amax = x.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(6 * 2.0 ** -9) / 6).to(torch.float8_e4m3fn).to(F32)
    else:
        s = pow2_scale(amax.clamp_min(6 * 2.0 ** -126), 6.0)
    return (_fp4_round((x / s).clamp(-6, 6)) * s).reshape(shape)


def linear(x, name):
    """Official linear(): FP8/FP4 weights take FP8 g32 activations; output rounded to BF16."""
    w = fp8_weight(name)
    xin = act_quant(x) if W(name + '.weight').dtype == torch.float8_e4m3fn else x.to(F32)
    return (xin.double() @ w.double().T).to(BF16)


def rms(x, weight):
    x = x.to(F32)
    return (weight.to(F32) * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS))).to(BF16)


@lru_cache(4)
def freqs(compressed):
    rd = CFG['qk_rope_head_dim']
    if compressed:
        base, orig = CFG['compress_rope_theta'], CFG['rope_scaling']['original_max_position_embeddings']
    else:
        base, orig = CFG['rope_theta'], 0
    f = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=F32) / rd))
    if orig > 0:
        rs = CFG['rope_scaling']
        cd = lambda r: rd * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(cd(rs['beta_fast'])), 0), min(math.ceil(cd(rs['beta_slow'])), rd - 1)
        ramp = ((torch.arange(rd // 2, dtype=F32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        f = f / rs['factor'] * ramp + f * (1 - ramp)
    return f


def rope(x, pos, compressed, inverse=False):
    """x [..., T, (H,) D] bf16; rotates the last 64 dims as adjacent complex pairs; returns bf16."""
    rd = CFG['qk_rope_head_dim']
    ang = torch.outer(pos.to(F32), freqs(compressed))
    fc = torch.polar(torch.ones_like(ang), ang)
    if inverse:
        fc = fc.conj()
    y = x.clone()
    tail = torch.view_as_complex(x[..., -rd:].to(F32).unflatten(-1, (-1, 2)).contiguous())
    fc = fc.view(*fc.shape[:1], *([1] * (tail.ndim - 2)), fc.shape[-1])
    y[..., -rd:] = torch.view_as_real(tail * fc).flatten(-2).to(x.dtype)
    return y


def sparse_attn(q, kv, sink, scale):
    """q [H,D] bf16, kv [N,D] bf16 values, sink [H]; official online-softmax arithmetic (BF16 P)."""
    s = (q.double() @ kv.double().T).float() * scale
    mx = s.amax(-1, keepdim=True)
    p = torch.exp(s - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink[:, None] - mx)
    o = (p.to(BF16).double() @ kv.double()) / denom.double()
    return o.to(BF16)
