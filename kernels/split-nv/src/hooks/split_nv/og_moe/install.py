"""Install og-moe (numerics og-s4.4) into the split-nv engine: DeepseekV2MoE.forward keeps SGLang's router, top-k and
all-reduce, and runs the routed + shared experts with og_moe.moe. Every rank calls install(engine) after the model is
loaded; the expert tensors are used in place, the layouts are checked against the checkpoint bytes first.
"""
import json
import os

import torch

from . import H, I, NEXP, TOPK, LayerWeights, ext, moe

ORIG_FORWARD = None


class _Ckpt:
    """Header-parsing safetensors row reader (works for F8_E8M0 / F8_E4M3 / packed FP4 bytes)."""

    def __init__(self, model_path):
        self.root = model_path
        self.map = json.load(open(os.path.join(model_path, 'model.safetensors.index.json')))['weight_map']
        self.hdr = {}

    def rows(self, name, r0, r1):
        path = os.path.realpath(os.path.join(self.root, self.map[name]))
        if path not in self.hdr:
            with open(path, 'rb') as f:
                n = int.from_bytes(f.read(8), 'little')
                self.hdr[path] = (8 + n, json.loads(f.read(n)))
        base, hdr = self.hdr[path]
        e = hdr[name]
        a, _ = e['data_offsets']
        shape = e['shape']
        rb = 1
        for d in shape[1:]:
            rb *= d
        with open(path, 'rb') as f:
            f.seek(base + a + r0 * rb)
            buf = bytearray(f.read((r1 - r0) * rb))
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(r1 - r0, rb)


def _swz(n, kb, sf_cols):
    return (kb & 3) + (kb >> 2) * 512 + (n & 31) * 16 + ((n & 127) >> 5) * 4 + (n >> 7) * 128 * sf_cols


def _layer_weights(mlp, layer_id, rank, ck, n_experts=NEXP):
    ex = mlp.experts
    w13, w13_sf = ex.w13_weight.data, ex.w13_weight_scale_inv.data
    w2, w2_sf = ex.w2_weight.data, ex.w2_weight_scale_inv.data
    E = w13.shape[0]
    assert E == n_experts and tuple(w13.shape) == (E, 2 * I, H // 2), f'layer {layer_id}: w13 {tuple(w13.shape)}'
    assert tuple(w2.shape) == (E, H, I // 2), f'layer {layer_id}: w2 {tuple(w2.shape)}'
    assert w13_sf.numel() == E * 2 * I * (H // 32) and w2_sf.numel() == E * H * (I // 32)
    assert not getattr(mlp, '_shared_expert_tp1', False), 'shared expert must be TP-sharded'
    se = mlp.shared_experts
    s13, s2 = se.gate_up_proj.weight.data, se.down_proj.weight.data
    assert s13.dtype == torch.float8_e4m3fn and tuple(s13.shape) == (2 * I, H), f'shared gate_up {s13.dtype} {tuple(s13.shape)}'
    assert s2.dtype == torch.float8_e4m3fn and tuple(s2.shape) == (H, I), f'shared down {s2.dtype} {tuple(s2.shape)}'
    p = f'layers.{layer_id}.ffn.'
    # routed: rows [0, I) = up (w3), [I, 2I) = gate (w1) of this rank's slice; w2 = this rank's K slice; scales swizzled
    for e in (0, E // 2 - 1, E - 1):
        up = ck.rows(p + f'experts.{e}.w3.weight', rank * I, rank * I + 1)[0]
        gate = ck.rows(p + f'experts.{e}.w1.weight', rank * I + 5, rank * I + 6)[0]
        down = ck.rows(p + f'experts.{e}.w2.weight', 7, 8)[0][rank * I // 2:(rank + 1) * I // 2]
        assert torch.equal(w13[e, 0].view(torch.uint8).cpu(), up), f'layer {layer_id} expert {e}: w13 up row'
        assert torch.equal(w13[e, I + 5].view(torch.uint8).cpu(), gate), f'layer {layer_id} expert {e}: w13 gate row'
        assert torch.equal(w2[e, 7].view(torch.uint8).cpu(), down), f'layer {layer_id} expert {e}: w2 row'
        s_up = ck.rows(p + f'experts.{e}.w3.scale', rank * I + 3, rank * I + 4)[0]
        s_dn = ck.rows(p + f'experts.{e}.w2.scale', 130, 131)[0][rank * I // 32:(rank + 1) * I // 32]
        f13 = w13_sf[e].reshape(-1).view(torch.uint8)
        f2 = w2_sf[e].reshape(-1).view(torch.uint8)
        i13 = torch.tensor([_swz(3, kb, H // 32) for kb in range(H // 32)], device=f13.device)
        i2 = torch.tensor([_swz(130, kb, I // 32) for kb in range(I // 32)], device=f2.device)
        got_up, got_dn = f13[i13].cpu(), f2[i2].cpu()
        assert torch.equal(got_up, s_up), f'layer {layer_id} expert {e}: w13 scale layout'
        assert torch.equal(got_dn, s_dn), f'layer {layer_id} expert {e}: w2 scale layout'
    # shared: find which half of gate_up is up/gate; 32x32 block scales from the checkpoint (this rank's slice)
    sp = p + 'shared_experts.'
    w1r = ck.rows(sp + 'w1.weight', rank * I, rank * I + 1)[0]
    w3r = ck.rows(sp + 'w3.weight', rank * I, rank * I + 1)[0]
    r0, rI = s13[0].view(torch.uint8).cpu(), s13[I].view(torch.uint8).cpu()
    if torch.equal(r0, w1r) and torch.equal(rI, w3r):
        s_gate0, s_up0 = 0, I
    elif torch.equal(r0, w3r) and torch.equal(rI, w1r):
        s_gate0, s_up0 = I, 0
    else:
        raise AssertionError(f'layer {layer_id}: shared gate_up rows match neither order')
    assert torch.equal(s2[9].view(torch.uint8).cpu(), ck.rows(sp + 'w2.weight', 9, 10)[0][rank * I:(rank + 1) * I]), \
        f'layer {layer_id}: shared down row'
    bs = I // 32
    s1 = ck.rows(sp + 'w1.scale', rank * bs, (rank + 1) * bs)
    s3 = ck.rows(sp + 'w3.scale', rank * bs, (rank + 1) * bs)
    s13_sf = (torch.cat([s1, s3]) if s_gate0 == 0 else torch.cat([s3, s1])).contiguous()
    s2_sf = ck.rows(sp + 'w2.scale', 0, H // 32)[:, rank * bs:(rank + 1) * bs].contiguous()
    dev = w13.device
    lw = LayerWeights(w13, w13_sf, w2, w2_sf, s13, s13_sf.to(dev), s2, s2_sf.to(dev), s_up0, s_gate0)
    lw.scaled = bool(getattr(ex, 'should_fuse_routed_scaling_factor_in_topk', False))
    lw.layer_id = layer_id
    return lw


def og_forward(self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None, input_ids=None,
               input_ids_global=None, skip_shared_experts=False):
    lw = self.__dict__.get('_og_lw')
    if lw is None or skip_shared_experts or hidden_states.shape[0] == 0:
        return ORIG_FORWARD(self, hidden_states, forward_batch, gemm_output_zero_allocator, input_ids,
                            input_ids_global, skip_shared_experts)
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce
    from sglang.srt.models import deepseek_v2 as D2

    num_token_non_padded = forward_batch.num_token_non_padded if forward_batch is not None else None
    router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
    if self.gate.e_score_correction_bias_vl is not None:
        topk_output = D2.vision_topk(self, router_logits, input_ids_global, num_token_non_padded=num_token_non_padded)
    else:
        topk_output = self.topk(hidden_states, router_logits, num_token_non_padded=num_token_non_padded,
                                expert_location_dispatch_info=None)
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    w = topk_output.topk_weights.float()
    if not lw.scaled:
        w = w * self.routed_scaling_factor
    out = moe(hidden_states.contiguous(), topk_output.topk_ids.to(torch.int32).contiguous(), w.contiguous(), lw)
    if self.tp_size > 1 and not should_skip_post_experts_all_reduce(is_tp_path=True):
        out = tensor_model_parallel_all_reduce(out)
    return out


def install(engine, model_path=None):
    """Attach og-moe weights to every MoE layer of the loaded model and switch DeepseekV2MoE.forward."""
    global ORIG_FORWARD
    from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    assert get_tensor_model_parallel_world_size() == 2, 'og-moe expects TP2 (I = 1152 per rank)'
    rank = get_tensor_model_parallel_rank()
    model_path = model_path or engine.mr.model_config.model_path
    ck = _Ckpt(model_path)
    ext()  # build / load the kernels before any forward
    n = 0
    for i, layer in enumerate(engine.mr.model.model.layers):
        mlp = getattr(layer, 'mlp', None)
        if not isinstance(mlp, DeepseekV2MoE):
            continue
        assert not mlp._dsv41_prune_pending and mlp.num_fused_shared_experts == 0
        mlp._og_lw = _layer_weights(mlp, layer.layer_id, rank, ck)
        n += 1
    if ORIG_FORWARD is None:
        ORIG_FORWARD = DeepseekV2MoE.forward
        DeepseekV2MoE.forward = og_forward
    torch.cuda.synchronize()
    print(f'[og-moe] rank {rank}: installed on {n} MoE layers (numerics og-s4.4)', flush=True)
    return n
