"""og-box numerics development helpers, run inside the engine through the dev exec hook.

Every rank executes the same calls; only rank 0 records and writes files.
"""
import hashlib
import json
import os

import numpy as np
import torch

from split_nv import numerics

TOKENS = "/home/ian/split-nv/ref/ids-8192.json"
OUT = "/dev/shm/split-nv/og"


def core_hook(engine):
    from sglang.srt.layers.attention import deepseek_v4_backend as B

    cap = engine.cap
    if getattr(B.DeepseekV4AttnBackend, "_og_core", False):
        return
    orig = B.DeepseekV4AttnBackend.forward

    def forward(self, q, k, v, layer, forward_batch, *a, **kw):
        o = orig(self, q, k, v, layer, forward_batch, *a, **kw)
        if cap.enabled and cap.trace is not None:
            numerics.record(cap, f"{layer.layer_id}.attn.core", o[:, :32].clone())
        return o

    B.DeepseekV4AttnBackend.forward = forward
    B.DeepseekV4AttnBackend._og_core = True


def run(engine, name, n=8213):
    """Prefill the 8K fixture prefix with the trace on; save trace + raw state under OUT/name.*"""
    ids = json.load(open(TOKENS))[:n]
    cap = engine.cap
    engine.cmd_prefill(990, ids, dump=True)
    engine.cmd_close(990)
    if not cap.enabled:
        return None
    os.makedirs(OUT, exist_ok=True)
    digest = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
    os.replace(f"/dev/shm/split-nv/raw-{digest}.safetensors", f"{OUT}/{name}.raw.safetensors")
    if n == 8213:
        os.replace("/dev/shm/split-nv/cuda-trace-8213.pt", f"{OUT}/{name}.pt")
    return {"name": name, "prefill_s": sum(c[1] for c in cap.chunks)}


def router_noise(scale):
    """Chaos-floor probe: multiply router logits by (1 + scale * N(0,1)); scale 0 restores."""
    from sglang.srt.models.deepseek_v2 import MoEGate
    from split_nv.fixed_linear import linear

    if scale == 0:
        MoEGate.forward = lambda self, x, *a, **kw: linear(x, self.weight)
        return
    gen = torch.Generator(device="cuda").manual_seed(1234)

    def fwd(self, x, *a, **kw):
        out = linear(x, self.weight)
        return out * (1 + scale * torch.randn(out.shape, device=out.device, generator=gen))

    MoEGate.forward = fwd


def run_chunked(engine, name, chunk, n=8213):
    """Same as run() but with a different prefill chunk size (reduction-shape perturbation)."""
    from split_nv import engine as E

    old = E.CHUNK
    E.CHUNK = chunk
    try:
        return run(engine, name, n)
    finally:
        E.CHUNK = old


def attn_ulp_noise(engine, layers, on=True):
    """Chaos-floor probe: flip the BF16 last bit of half the attention-output elements in `layers`."""
    from sglang.srt.models import deepseek_v4 as M

    cls = M.MQALayer
    if not hasattr(cls, "_og_orig_forward"):
        cls._og_orig_forward = cls.forward
    if not on:
        cls.forward = cls._og_orig_forward
        return
    gen = torch.Generator(device="cuda").manual_seed(99)
    orig = cls._og_orig_forward

    def forward(self, x, positions, forward_batch, x_quant=None):
        o = orig(self, x, positions, forward_batch, x_quant=x_quant)
        if self.layer_id in layers:
            bits = o.view(torch.int16)
            flip = torch.randint(0, 2, o.shape, device=o.device, generator=gen, dtype=torch.int16)
            o = (bits ^ flip).view(torch.bfloat16)
        return o

    cls.forward = forward


def idx_hook(engine):
    """Record the prefill indexer's selected logical compressed positions per layer (ascending, -1 pad)."""
    from sglang.srt.layers.attention import deepseek_v4_backend as B

    cap = engine.cap
    if getattr(B.DeepseekV4AttnBackend, "_og_idx", False):
        return
    orig = B.DeepseekV4AttnBackend._low_ratio_index_topk_dense

    def dense(self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu):
        orig(self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu)
        if cap.enabled and cap.trace is not None:
            raw = self.forward_metadata.core_metadata.sparse_raw_indices(layer.compress_ratio)
            if raw is not None:
                numerics.record(cap, f"{layer.layer_id}.attn.idx", raw[: pos.shape[0], : layer.indexer.index_topk].clone())

    B.DeepseekV4AttnBackend._low_ratio_index_topk_dense = dense
    B.DeepseekV4AttnBackend._og_idx = True


def run_trace(engine, name, rows=136, n=8213):
    """Prefill n fixture tokens with every trace key recorded for the last `rows` positions."""
    import time as _t

    ids = json.load(open(TOKENS))[:n]
    cap = engine.cap
    sess = engine.__class__.__mro__[0]
    from split_nv.engine import Session, CHUNK

    s = Session(991)
    s.req = engine._new_req(991)
    engine.sessions[991] = s
    cap.reset()
    cap.trace_start = n - rows
    cap.trace = {}
    cap.t_start = _t.perf_counter()
    for i in range(0, n, CHUNK):
        chunk = ids[i:i + CHUNK]
        cap.tokens.append(torch.tensor(chunk, dtype=torch.int64))
        cap.ntok += len(chunk)
        engine._extend(s, chunk)
        torch.cuda.synchronize()
    s.alloc_len = s.req.kv.kv_allocated_len
    snap = numerics.snapshot(cap) if cap.enabled else None
    cap.trace = None
    engine.cmd_close(991)
    if snap is not None:
        os.makedirs(OUT, exist_ok=True)
        torch.save(snap, f"{OUT}/{name}.pt")
    return {"name": name}


def attn_mode(engine, mode):
    """Switch the sparse-MLA kernel: 'og' (official arithmetic, split_nv.og_attn) or 'flashinfer' (consistent.py)."""
    import importlib
    from sglang.kernels.ops.attention import flash_mla_sm120 as A
    from split_nv import og_attn

    if not hasattr(A, "_og_saved"):
        A._og_saved = A.flash_mla_with_kvcache_sm120
    if mode == "og":
        importlib.reload(og_attn)
        og_attn.install()
    else:
        A.flash_mla_with_kvcache_sm120 = A._og_saved


def attn_heads_slice(engine, on=True):
    """Steps (<=64 rows): run FlashInfer on the real local heads only (32 under TP2), not the 64 padded
    heads. Prefill (>64 rows) already drops the padding, so per-head arithmetic is unchanged."""
    from sglang.kernels.ops.attention import flash_mla_sm120 as A

    if not hasattr(A, "_og_saved"):
        A._og_saved = A.flash_mla_with_kvcache_sm120
    if not on:
        A.flash_mla_with_kvcache_sm120 = A._og_saved
        return
    padded_wrapper = A._og_saved
    heads = engine.mr.model.model.layers[0].self_attn.n_local_heads

    def attention(*args, **kw):
        q = kw["q"]
        if q.shape[0] <= 64 and q.shape[-2] > heads:
            kw["q"] = q[..., :heads, :].contiguous()
            kw["attn_sink"] = kw["attn_sink"][:heads]
        return padded_wrapper(*args, **kw)

    A.flash_mla_with_kvcache_sm120 = attention


def mxfp8_b12x(engine, on=True):
    """Route FlashInfer MXFP8 dense GEMMs from the CUTLASS backend to b12x (bit-identical, faster at small M)."""
    from sglang.srt.layers.quantization import fp8_utils as F

    if not hasattr(F, "_og_mm"):
        F._og_mm = F.flashinfer_mm_mxfp8
    if not on:
        F.flashinfer_mm_mxfp8 = F._og_mm
        return
    orig = F._og_mm
    ok = {}

    def mm(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False, backend="auto"):
        key = (q_input.shape[1], weight_t.shape[1])
        if backend == "cutlass" and ok.get(key, True):
            try:
                out = orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                           use_8x4_sf_layout=use_8x4_sf_layout, backend="b12x")
                ok[key] = True
                return out
            except Exception:  # noqa: BLE001  (shape outside b12x requirements)
                ok[key] = False
        return orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                    use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)

    F.flashinfer_mm_mxfp8 = mm
    F._og_ok = ok
