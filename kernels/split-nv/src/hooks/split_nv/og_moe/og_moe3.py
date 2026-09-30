"""og-moe's fused decode MoE kernel rebuilt for the DSpark drafter stages (128 routed experts, top-3; the target
layers are 384 / top-6). Same source as hooks/split_nv/og_moe/og_moe.cu with the two compile-time constants changed:
the decode path (M <= 8) is generic in TOPK, NEXP only sizes the prefill path's tables. Expert tensors are the
SGLang-loaded ones used in place, with og-moe's own layout checks against the checkpoint bytes (mtp.N.ffn.*).

  ext()                      build (or load the prebuilt .so for exactly this source + flags) -- no GPU needed
  stage_weights(mlp, s, rank, ckpt_dir) -> split_nv.og_moe.LayerWeights
  moe(x, ids, w, lw)         bf16 [M<=8, 5120] -> this rank's partial (routed + shared); the caller all-reduces
"""
import hashlib
import importlib.util
import os

import torch

TOPK, NEXP = 3, 128
FLAGS = ['-O3', '-std=c++17', '--fmad=false', '-gencode=arch=compute_120a,code=sm_120a', '-lineinfo']
_EXT = None
_WS = {}
_VALID = {}


def source():
    from split_nv import og_moe as base
    src = open(os.path.join(os.path.dirname(os.path.abspath(base.__file__)), 'og_moe.cu')).read()
    for old, new in (('constexpr int TOPK = 6;', f'constexpr int TOPK = {TOPK};'),
                     ('constexpr int NEXP = 384;', f'constexpr int NEXP = {NEXP};')):
        if src.count(old) != 1:
            raise RuntimeError(f'og_moe.cu changed: {old!r} not found exactly once')
        src = src.replace(old, new)
    return src


def ext():
    global _EXT
    if _EXT is None:
        src = source()
        key = hashlib.sha256(src.encode() + ' '.join(FLAGS).encode()).hexdigest()[:16]
        # engine: /root/.cache/sglang is the persistent JIT dir (/mnt/nvme-1/dsv41-sglang-jit); prebuild it there
        build = os.path.join(os.environ.get('OG_MOE3_BUILD', os.path.expanduser('~/.cache/sglang/og_moe3')), key)
        name = f'og_moe3_{key}'
        so = os.path.join(build, f'{name}.so')
        if os.path.exists(so):
            spec = importlib.util.spec_from_file_location(name, so)
            _EXT = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_EXT)
        else:
            from torch.utils.cpp_extension import load
            os.makedirs(build, exist_ok=True)
            cu = os.path.join(build, 'og_moe3.cu')
            with open(cu, 'w') as f:
                f.write(src)
            _EXT = load(name=name, sources=[cu], extra_cuda_cflags=FLAGS, build_directory=build, verbose=False)
    return _EXT


class _StageCk:
    """og_moe.install._layer_weights reads 'layers.{id}.ffn.*' rows; the drafter's live under 'mtp.{stage}.ffn.*'."""

    def __init__(self, ck):
        self.ck = ck

    def rows(self, name, r0, r1):
        if not name.startswith('layers.'):
            raise KeyError(name)
        return self.ck.rows('mtp.' + name[len('layers.'):], r0, r1)


def stage_weights(mlp, stage, rank, ckpt_dir):
    from split_nv.og_moe import install as OI
    return OI._layer_weights(mlp, stage, rank, _StageCk(OI._Ckpt(ckpt_dir)), n_experts=NEXP)


def valid_rows(device):
    v = _VALID.get(device)
    if v is None:
        v = _VALID[device] = torch.full((1,), 1 << 30, dtype=torch.int32, device=device)
    return v


def workspace(device):
    ws = _WS.get(device)
    if ws is None:
        ws = _WS[device] = torch.empty(ext().decode_workspace_bytes(), dtype=torch.uint8, device=device)
    return ws


def moe(x, ids, w, lw):
    """x bf16 [M, 5120] contiguous (M <= 8); ids int32 [M, 3]; w fp32 [M, 3] incl. routed_scaling_factor."""
    if x.shape[0] > 8:
        raise ValueError('og_moe3 is decode-only (M <= 8)')
    dev = x.device
    return ext().decode(x, ids, w, valid_rows(dev), workspace(dev), *lw.args,
                        torch.cuda.get_device_properties(dev).multi_processor_count)
