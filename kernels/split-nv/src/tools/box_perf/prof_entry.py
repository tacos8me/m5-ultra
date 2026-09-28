"""Profiling entry for the split-nv engine under nsys: NVTX ranges only, no numerical change.

Run as `python3 -m prof_entry <engine args>` (PYTHONPATH=/home/ian/split-nv/hooks:/prof). The multiprocessing spawn
children re-import this module as __mp_main__, so the patches below apply on every rank. Ranges are host-side only:
inside CUDA graph capture they are recorded by nsys `--cuda-graph-trace=node:nvtx-precapture` and projected onto the
replayed graph nodes; eager prefill gets them live.
"""
import functools
import inspect
import os

import torch

NV = torch.cuda.nvtx


def _wrap(fn, label):
    if getattr(fn, "_prof_wrapped", False):
        return fn

    @functools.wraps(fn)
    def w(*a, **k):
        NV.range_push(label(a) if callable(label) else label)
        try:
            return fn(*a, **k)
        finally:
            NV.range_pop()

    w._prof_wrapped = True
    return w


def _wrap_class(cls, names=None, prefix="", skip=()):
    n = 0
    for name, raw in list(vars(cls).items()):
        if name.startswith("__") or name in skip or (names is not None and name not in names):
            continue
        if not inspect.isfunction(raw):  # leaves staticmethod/classmethod/property alone
            continue
        if inspect.isgeneratorfunction(raw):
            continue
        setattr(cls, name, _wrap(raw, f"{prefix}{name}"))
        n += 1
    return n


def _module_hooks(model):
    """NVTX around every submodule call below the decoder layers (depth <= 3) and the model's direct children."""
    n = 0

    def pre(label):
        def h(m, a, kw=None):
            NV.range_push(label)
        return h

    def post(m, a, o):
        NV.range_pop()

    for name, m in model.named_modules():
        parts = name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            depth = len(parts) - i - 2  # 0 = the layer itself
            if depth < 1 or depth > 3:
                continue
            label = "m:" + ".".join(parts[i + 2:])
        elif name.count(".") == 1 and name.startswith("model."):
            label = "m:" + parts[-1]
        else:
            continue
        if isinstance(m, torch.nn.ModuleList):
            continue
        m.register_forward_pre_hook(pre(label))
        m.register_forward_hook(post)
        n += 1
    return n


def install_nvtx(engine):
    from sglang.srt.layers.attention import deepseek_v4_backend as B
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse as S
    from sglang.srt.models import deepseek_v4 as M
    from split_nv import steprunner as SR

    L = M.DeepseekV4DecoderLayer
    L.forward_hc_pre_from_prev = _wrap(L.forward_hc_pre_from_prev, lambda a: f"L{a[0].layer_id}")
    for name, lab in (("_hc_mix_and_combine", "mhc_pre"), ("hc_post", "mhc_post"), ("_run_moe_ffn_dp_sync", "moe_ffn")):
        if name in vars(L):
            setattr(L, name, _wrap(vars(L)[name], lab))
    M.DeepseekV4Model.forward = _wrap(M.DeepseekV4Model.forward, "model")
    for name in ("engram_fill_decode_pregather", "engram_setup_decode_pregather"):
        if name in vars(M.DeepseekV4Model):
            setattr(M.DeepseekV4Model, name, _wrap(vars(M.DeepseekV4Model)[name], name))
    nb = _wrap_class(B.DeepseekV4AttnBackend, prefix="be.", skip=("rows", "real_rows", "put", "copy_"))
    na = _wrap_class(M.MQALayer, prefix="attn.", skip=("maybe_use_decode_attn_tp", "_local_attn_sink"))
    ni = _wrap_class(S.DeepseekV41Indexer, prefix="idx.") + _wrap_class(S.DeepseekV41Compressor, prefix="cmp.")
    ns = _wrap_class(SR.StepRunner, names=("run", "commit_pending", "evict_swa", "ensure_alloc", "_load", "replay",
                                           "run_eager"), prefix="sr.")
    nm = _module_hooks(engine.mr.model)
    print(f"[prof] NVTX installed: backend {nb}, attn {na}, idx/cmp {ni}, steprunner {ns}, module hooks {nm}", flush=True)


def _patch():
    from split_nv import engine as E
    from split_nv import front as F

    orig_init = E.Engine.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        install_nvtx(self)

    E.Engine.__init__ = init

    def label(a):
        cmd = a[1]
        kind = cmd[0]
        if kind == "step":
            return f"cmd:step sid={cmd[1]} keep={cmd[2]} L={len(cmd[3])}"
        if kind in ("prefill_chunk",):
            return f"cmd:prefill_chunk n={len(cmd[2])}"
        if kind == "prefill":
            return f"cmd:prefill n={len(cmd[2])}"
        return f"cmd:{kind}"

    E.Engine.execute = _wrap(E.Engine.execute, label)

    orig_submit = F.Front.submit_async

    def submit_async(self, cmd, priority=0):
        # /prof/eager present: steps run eagerly (live NVTX per module) instead of replaying the graph
        if cmd[0] == "step" and len(cmd) == 4 and os.path.exists("/prof/eager"):
            cmd = (*cmd, False)
        return orig_submit(self, cmd, priority)

    F.Front.submit_async = submit_async


_patch()

if __name__ == "__main__":
    from split_nv.engine import main

    main()
