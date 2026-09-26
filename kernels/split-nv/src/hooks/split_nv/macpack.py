"""Torch port of the Mac oMLX DS41 packed-activation formats and rope.

Mirrors ~/src/wt/ds41-serve-c2/omlx/patches/deepseek_v41/quantization.py
(pack_activation / unpack_activation / round_fp8) and language.py (_rope,
_rope_freq, norm). Pure torch, fp32 arithmetic, works on CPU and CUDA.
"""

import math

import torch

FP8_LIMIT = 448.0
FP4_LIMIT = 6.0
FP4_LEVELS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
# (threshold, value_code) with RNE tie handling: even code wins on ties.
_FP4_THRESHOLDS = [
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
]


_LEVELS = {}


def _levels(device):
    key = str(device)
    if key not in _LEVELS:
        _LEVELS[key] = FP4_LEVELS.to(device)
    return _LEVELS[key]


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """Exact fp32 powers of two from an integer-valued fp32 exponent >= -126."""
    return torch.ldexp(torch.ones_like(exponent), exponent.to(torch.int32))


def round_fp8(x: torch.Tensor) -> torch.Tensor:
    """E4M3FN round-to-nearest-even with finite saturation (Mac round_fp8). Scalar constants only: CUDA-graph safe."""
    x = x.float()
    a = torch.clamp(x.abs(), max=FP8_LIMIT)
    exponent = torch.floor(torch.log2(torch.clamp(a, min=2.0**-9)))
    step = _pow2(torch.clamp(exponent - 3, min=-9))
    return torch.sign(x) * torch.clamp(torch.round(a / step) * step, max=FP8_LIMIT)


def pack_activation(x: torch.Tensor, bits: int = 8, group_size: int = 32, e4m3_scale: bool = False) -> torch.Tensor:
    """Row layout: value bytes then one scale byte per group (uint8)."""
    if bits not in (4, 8) or x.shape[-1] % group_size:
        raise ValueError("Invalid packed activation geometry")
    shape = x.shape
    grouped = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    limit = FP8_LIMIT if bits == 8 else FP4_LIMIT
    minimum = limit * (2.0**-9 if e4m3_scale else 2.0**-126)
    amax = torch.clamp(grouped.abs().amax(-1), min=minimum)
    if e4m3_scale:
        scale = round_fp8(amax / limit)
        scale_bytes = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        exponent = torch.clamp(torch.ceil(torch.log2(amax / limit)), min=-126)
        scale = _pow2(exponent)
        scale_bytes = (exponent + 127).to(torch.uint8)
    scaled = torch.clamp(grouped / scale[..., None], -limit, limit).reshape(shape)
    if bits == 8:
        values = round_fp8(scaled).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        a = scaled.abs()
        code = torch.zeros(a.shape, dtype=torch.uint8, device=x.device)
        for i, (threshold, inclusive) in enumerate(_FP4_THRESHOLDS, 1):
            hit = a >= threshold if inclusive else a > threshold
            code = torch.where(hit, i, code).to(torch.uint8)
        code = code | ((scaled < 0).to(torch.uint8) << 3)
        values = code[..., ::2] | (code[..., 1::2] << 4)
    return torch.cat([values, scale_bytes], -1)


def unpack_activation(packed: torch.Tensor, bits: int = 8, group_size: int = 32, e4m3_scale: bool = False,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    groups = packed.shape[-1] // (group_size * bits // 8 + 1)
    width = groups * group_size
    nbytes = width * bits // 8
    values, scales = packed[..., :nbytes], packed[..., nbytes:]
    if bits == 8:
        values = values.contiguous().view(torch.float8_e4m3fn).float()
    else:
        codes = torch.stack([values & 15, values >> 4], -1).reshape(*packed.shape[:-1], width)
        levels = _levels(packed.device)
        values = levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    if e4m3_scale:
        scales = scales.contiguous().view(torch.float8_e4m3fn).float()
    else:
        scales = _pow2(scales.to(torch.float32) - 127)
    out = (values.reshape(*packed.shape[:-1], groups, group_size) * scales[..., None]).reshape(*packed.shape[:-1], width)
    return out.to(dtype)


def rope_freq(d: int, base: float, original_seq_len: int, beta_fast: float, beta_slow: float, factor: float,
              compressed: bool, device=None) -> torch.Tensor:
    freq = 1.0 / torch.pow(torch.tensor(base, dtype=torch.float32, device=device),
                           torch.arange(0, d, 2, dtype=torch.float32, device=device) / d)
    if compressed and original_seq_len:
        def correction(rotations):
            return d * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), d - 1)
        ramp = torch.clamp((torch.arange(d // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freq = freq / factor * (1 - smooth) + freq * smooth
    return freq


class RopeParams:
    """Layer rope selection as in the Mac/reference: compressed (YaRN, theta 160000)
    for every layer with compress_ratio > 0, plain theta 10000 otherwise."""

    def __init__(self, text_config: dict):
        self.d = int(text_config["qk_rope_head_dim"])
        scaling = text_config.get("rope_scaling") or {}
        self.main = (self.d, float(text_config["rope_theta"]), 0, 32, 1, 1.0)
        self.compressed = (
            self.d,
            float(text_config["compress_rope_theta"]),
            int(scaling.get("original_max_position_embeddings", 0)),
            float(scaling.get("beta_fast", 32)),
            float(scaling.get("beta_slow", 1)),
            float(scaling.get("factor", 1.0)),
        )
        self._cache = {}

    def freq(self, compressed: bool, device) -> torch.Tensor:
        key = (compressed, str(device))
        if key not in self._cache:
            p = self.compressed if compressed else self.main
            self._cache[key] = rope_freq(p[0], p[1], p[2], p[3], p[4], p[5], compressed, device=device)
        return self._cache[key]


def apply_rope(x: torch.Tensor, positions: torch.Tensor, freq: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Mac _rope: rotate the last len(freq)*2 dims in fp32 as adjacent pairs, cast back to x.dtype."""
    d = freq.shape[0] * 2
    angles = positions.to(torch.float32)[:, None] * freq[None, :]
    if inverse:
        angles = -angles
    while angles.ndim < x.ndim:
        angles = angles.unsqueeze(-2)
    tail = x[..., -d:].float().reshape(*x.shape[:-1], d // 2, 2)
    a, b = tail[..., 0], tail[..., 1]
    cos, sin = torch.cos(angles), torch.sin(angles)
    rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).reshape(*x.shape[:-1], d)
    return torch.cat([x[..., :-d], rotated.to(x.dtype)], -1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def pack_swa_row(rotated: torch.Tensor) -> torch.Tensor:
    return pack_activation(rotated, 8, 32, False)


def pack_ckv_row(rotated_latent: torch.Tensor) -> torch.Tensor:
    return pack_activation(rotated_latent, 4, 16, True)


def pack_idxk_row(rotated_key: torch.Tensor) -> torch.Tensor:
    return pack_activation(rotated_key, 4, 32, False)
