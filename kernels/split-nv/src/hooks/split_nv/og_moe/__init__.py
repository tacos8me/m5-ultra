"""og-moe: fused DS-V4.1 routed + shared MoE (numerics og-s4.4). See og_moe.cu for the arithmetic.

moe(x, ids, w, lw) -> bf16 [M, H]: this TP rank's partial (routed experts over its I slice + shared expert slice);
the caller all-reduces it. M <= 8 runs the decode kernel (weights read once for all rows), larger M the prefill GEMMs;
both give bit-identical rows.
"""
import hashlib
import os

import torch

H, I, TOPK, NEXP = 5120, 1152, 6, 384
MAX_DECODE = 8
_EXT = None
_WS = {}
VALID = {}  # device -> int32[1]: rows >= VALID of a decode call are padding (the step runner sets it around a step)


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'og_moe.cu')
        flags = ['-O3', '-std=c++17', '--fmad=false', '-gencode=arch=compute_120a,code=sm_120a', '-lineinfo']
        if os.environ.get('OG_MOE_TRACE'):
            flags.append('-DOG_TRACE')
        if os.environ.get('OG_MOE_PTW'):
            flags.append('-DOG_P_TW=' + os.environ['OG_MOE_PTW'])
        key = hashlib.sha256(open(src, 'rb').read() + ' '.join(flags).encode()).hexdigest()[:16]
        root = os.environ.get('OG_MOE_BUILD', os.path.expanduser('~/.cache/sglang/og_moe'))
        build = os.path.join(root, key)
        name = f'og_moe_{key}'
        so = os.path.join(build, f'{name}.so')
        if os.path.exists(so) and os.environ.get('OG_MOE_REBUILD') != '1':
            # Prebuilt for exactly this source + flags (the key): import it without the JIT (no ninja timestamp
            # checks against a freshly checked-out source tree).
            import importlib.util
            spec = importlib.util.spec_from_file_location(name, so)
            _EXT = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_EXT)
        else:
            os.makedirs(build, exist_ok=True)
            _EXT = load(name=name, sources=[src], extra_cuda_cflags=flags, build_directory=build, verbose=False)
    return _EXT


def valid_rows(device):
    v = VALID.get(device)
    if v is None:
        v = VALID[device] = torch.full((1,), 1 << 30, dtype=torch.int32, device=device)
    return v


def _decode_ws(device):
    ws = _WS.get(device)
    if ws is None:
        ws = _WS[device] = torch.empty(ext().decode_workspace_bytes(), dtype=torch.uint8, device=device)
    return ws


class LayerWeights:
    """Byte views of one layer's expert tensors as the engine holds them (no copies of the big tensors)."""

    def __init__(self, w13, w13_sf, w2, w2_sf, s13, s13_sf, s2, s2_sf, s_up0, s_gate0):
        u8 = lambda t: t.contiguous().view(torch.uint8)
        self.args = (u8(w13), u8(w13_sf), u8(w2), u8(w2_sf), u8(s13), u8(s13_sf), u8(s2), u8(s2_sf),
                     int(s_up0), int(s_gate0))


def _grid(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def decode(x, ids, w, lw, valid=None):
    return ext().decode(x, ids, w, valid if valid is not None else valid_rows(x.device), _decode_ws(x.device),
                        *lw.args, _grid(x.device))


def prefill(x, ids, w, lw):
    """Any M: routing, GEMMs and combine on the GPU (no host sync); one workspace allocation."""
    ws = torch.empty(ext().prefill_workspace_bytes(x.shape[0]), dtype=torch.uint8, device=x.device)
    return ext().prefill(x, ids, w, ws, *lw.args)


def moe(x, ids, w, lw, valid=None):
    """x bf16 [M, H] contiguous; ids int32 [M, 6]; w fp32 [M, 6] (route weights incl. routed_scaling_factor)."""
    if x.shape[0] <= MAX_DECODE:
        return decode(x, ids, w, lw, valid)
    return prefill(x, ids, w, lw)
