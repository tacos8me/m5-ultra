"""Real DS-V4.1 layer weights in the layout the live engine keeps in VRAM (FlashInfer SM120 MXFP4 experts).

w13 [E, 2I, H/2] u8: rows [0, I) = w3 (up), [I, 2I) = w1 (gate), this TP rank's I = 1152 slice; packed e2m1, low nibble
= even k. w13_sf [E, 2I, H/32] ue8m0 bytes, 128x4-swizzled per expert (block_scale_interleave). w2 [E, H, I/2] u8 (this
rank's K slice), w2_sf [E, H, I/32] swizzled. Shared expert: FP8 e4m3 [2I, H] (gate_up rows: up first? see below) with
32x32 block scales.
"""
import json
import os

import torch

CK = os.environ.get("OG_CKPT", "/ckpt")
H, I_FULL, TP = 5120, 2304, 2
I = I_FULL // TP
_MAP = None
_HDR = {}
_DT = {'F8_E8M0': torch.uint8, 'F8_E4M3': torch.uint8, 'I8': torch.uint8, 'U8': torch.uint8, 'BF16': torch.bfloat16,
       'F32': torch.float32}


def _map():
    global _MAP
    if _MAP is None:
        _MAP = json.load(open(f'{CK}/model.safetensors.index.json'))['weight_map']
    return _MAP


def raw(name):
    path = f'{CK}/{_map()[name]}'
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


def swizzle_sf(sf):
    """[.., N, K/32] u8 -> FlashInfer 128x4 swizzled bytes (same shape), N % 128 == 0, (K/32) % 4 == 0."""
    *lead, n, kb = sf.shape
    assert n % 128 == 0 and kb % 4 == 0
    x = sf.reshape(-1, n // 128, 4, 32, kb // 4, 4)  # [B, nB, q=(n>>5)&3, n&31, kbB, kb&3]
    x = x.permute(0, 1, 4, 3, 2, 5)  # [B, nB, kbB, n&31, q, kb&3]
    return x.contiguous().reshape(*lead, n, kb)


def expert_host(layer, e, rank):
    p = f'layers.{layer}.ffn.experts.{e}.'
    sl = slice(rank * I, (rank + 1) * I)
    w1, w3 = raw(p + 'w1.weight')[sl], raw(p + 'w3.weight')[sl]
    s1, s3 = raw(p + 'w1.scale')[sl], raw(p + 'w3.scale')[sl]
    w2 = raw(p + 'w2.weight')[:, rank * I // 2:(rank + 1) * I // 2]
    s2 = raw(p + 'w2.scale')[:, rank * I // 32:(rank + 1) * I // 32]
    return torch.cat([w3, w1]), torch.cat([s3, s1]), w2.contiguous(), s2.contiguous()


def load_experts(layer, experts, rank=0, device='cuda'):
    """Returns dict of device tensors for the given expert ids (in that order = local slot order)."""
    n = len(experts)
    w13 = torch.empty((n, 2 * I, H // 2), dtype=torch.uint8)
    s13 = torch.empty((n, 2 * I, H // 32), dtype=torch.uint8)
    w2 = torch.empty((n, H, I // 2), dtype=torch.uint8)
    s2 = torch.empty((n, H, I // 32), dtype=torch.uint8)
    for j, e in enumerate(experts):
        a, b, c, d = expert_host(layer, e, rank)
        w13[j], s13[j], w2[j], s2[j] = a, b, c, d
    return dict(w13=w13.to(device), w13_sf=swizzle_sf(s13).to(device), w2=w2.to(device), w2_sf=swizzle_sf(s2).to(device),
                w13_sf_lin=s13.to(device), w2_sf_lin=s2.to(device))


def shared_host(layer, rank=0):
    p = f'layers.{layer}.ffn.shared_experts.'
    sl = slice(rank * I, (rank + 1) * I)
    bs = slice(rank * I // 32, (rank + 1) * I // 32)
    w1, w3 = raw(p + 'w1.weight')[sl], raw(p + 'w3.weight')[sl]
    s1, s3 = raw(p + 'w1.scale')[bs], raw(p + 'w3.scale')[bs]
    w2 = raw(p + 'w2.weight')[:, sl].contiguous()
    s2 = raw(p + 'w2.scale')[:, bs].contiguous()
    return dict(w1=w1, w3=w3, s1=s1, s3=s3, w2=w2, s2=s2)


def gate(layer):
    return raw(f'layers.{layer}.ffn.gate.weight').float(), raw(f'layers.{layer}.ffn.gate.bias').float()


def route(x, layer, topk=6, scale=1.5):
    """Official Gate on fp32 logits of bf16 x (CPU or GPU)."""
    gw, gb = gate(layer)
    gw, gb = gw.to(x.device), gb.to(x.device)
    s = torch.nn.functional.softplus(x.float() @ gw.T).sqrt()
    idx = (s + gb).topk(topk, -1)[1]
    w = s.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * scale
    return idx, w
