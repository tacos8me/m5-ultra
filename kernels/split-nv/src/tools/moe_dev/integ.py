"""SGLang-level og-moe test (TP2 across both GPUs, small allocations): a real DeepseekV2MoE for one layer, expert
count cut to NE, filled from the checkpoint through the real weight loaders + process_weights_after_loading. Checks:
install() layout verification, og_forward vs the original FlashInfer forward (fidelity-level), og_forward vs the
standalone kernel (bitwise), eager vs CUDA-graph replay (bitwise), prefill rows == step rows through og_forward.
"""
import os
import sys

import torch
import torch.multiprocessing as mp

LAYER = int(os.environ.get('LAYER', '3'))
NE = int(os.environ.get('NE', '32'))
ENV = dict(SGLANG_SM120_FLASHMLA_BACKEND='flashinfer', SGLANG_FLASHINFER_MOE_FUSED_FINALIZE='0',
           SGLANG_OPT_USE_TOPK_V2='1', PYTHONPATH='/work/hooks', HF_HUB_OFFLINE='1')


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def worker(rank, port):
    os.environ.update(ENV)
    sys.path.insert(0, '/work/hooks')
    sys.path.insert(0, '/work/tools/moe_dev')
    torch.cuda.set_device(rank)
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.runtime_context import publish
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    sa = prepare_server_args(['--model-path', '/model', '--trust-remote-code', '--tp', '2', '--mem-fraction-static', '0.5',
                              '--context-length', '1048576', '--chunked-prefill-size', '8192',
                              '--enable-deepseek-v4-fp4-indexer', '--fp8-gemm-backend', 'flashinfer_cutlass',
                              '--disable-cuda-graph', '--disable-radix-cache'])
    publish(sa, role='scheduler')
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend='nccl',
                                 distributed_init_method=f'tcp://127.0.0.1:{port}')
    initialize_model_parallel(tensor_model_parallel_size=2)
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader.loader import _get_quantization_config
    mc = ModelConfig.from_server_args(sa)
    qc = _get_quantization_config(mc, LoadConfig())
    cfg = mc.hf_text_config
    cfg.n_routed_experts = NE
    from split_nv import hooks as _h  # noqa: F401  (the engine imports these first)
    from split_nv.consistent import install as consistent_install
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.utils import install_shared_experts_fusion_decision
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    install_shared_experts_fusion_decision(DeepseekV4ForCausalLM, mc.hf_config, qc)
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    with set_default_torch_dtype(mc.dtype), torch.device('cuda'):
        mlp = DeepseekV2MoE(config=cfg, layer_id=LAYER, quant_config=qc, prefix=f'model.layers.{LAYER}.mlp',
                            alt_stream=torch.cuda.Stream(), is_deepseek_v4=True)
    consistent_install()
    params = dict(mlp.named_parameters())

    # checkpoint -> params (the model's load_weights rules for this module)
    from safetensors import safe_open
    import json
    idx = json.load(open('/model/model.safetensors.index.json'))['weight_map']
    emap = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='gate_proj', ckpt_down_proj_name='down_proj',
                                               ckpt_up_proj_name='up_proj', num_experts=NE)
    stacked = [('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
    VL = os.environ.get('VL') == '1'
    names = [k for k in idx if k.startswith(f'layers.{LAYER}.ffn.') and (VL or 'bias_vl' not in k)]
    files = {}
    n_loaded = 0
    for name in names:
        if '.experts.' in name and int(name.split('.experts.')[1].split('.')[0]) >= NE:
            continue
        f = files.get(idx[name])
        if f is None:
            f = files[idx[name]] = safe_open(f'/model/{idx[name]}', framework='pt', device='cpu')
        w = f.get_tensor(name)
        n = name.replace(f'layers.{LAYER}.ffn.', '')
        n = n.replace('gate.bias_vl', 'gate.e_score_correction_bias_vl').replace('gate.bias', 'gate.e_score_correction_bias')
        n = n.replace('.w1.', '.gate_proj.').replace('.w2.', '.down_proj.').replace('.w3.', '.up_proj.')
        if n.endswith('.scale'):
            n = n[:-len('.scale')] + '.weight_scale_inv'
        if n in ('gate.weight', 'gate.e_score_correction_bias', 'gate.e_score_correction_bias_vl'):
            w = w[:NE]
        if n.startswith('shared_experts.'):
            for pn, wn, sid in stacked:
                if wn in n:
                    p = params[n.replace(wn, pn)]
                    p.weight_loader(p, w, sid)
                    break
            else:
                p = params[n]
                p.weight_loader(p, w)
        elif n.startswith('experts.'):
            for pn, wn, eid, sid in emap:
                if wn in n:
                    p = params[n.replace(wn, pn)]
                    p.weight_loader(p, w, n.replace(wn, pn), shard_id=sid, expert_id=eid)
                    break
            else:
                raise KeyError(n)
        else:
            p = params[n]
            getattr(p, 'weight_loader', lambda p_, w_: p_.data.copy_(w_))(p, w)
        n_loaded += 1
    with set_default_torch_dtype(mc.dtype):
        for _, module in mlp.named_modules():
            qm = getattr(module, 'quant_method', None)
            if qm is not None and hasattr(qm, 'process_weights_after_loading'):
                qm.process_weights_after_loading(module)
    torch.cuda.synchronize()
    print(f'[rank {rank}] loaded {n_loaded} tensors; w13 {tuple(mlp.experts.w13_weight.shape)} '
          f'{mlp.experts.w13_weight.dtype}, shared gate_up {tuple(mlp.shared_experts.gate_up_proj.weight.shape)} '
          f'{mlp.shared_experts.gate_up_proj.weight.dtype}', flush=True)

    from split_nv.og_moe import install as I
    ck = I._Ckpt('/model')
    lw = I._layer_weights(mlp, LAYER, rank, ck, n_experts=NE)
    tr = torch.load('/traces/e0-base.pt', map_location='cpu', weights_only=True)
    x8 = tr[f'{LAYER}.post_attention_layernorm'][1].cuda()
    torch.manual_seed(0)
    xb = (torch.randn(300, 5120, device='cuda') * x8.float().std()).to(torch.bfloat16)
    xb[137:142] = x8[:5]
    IMG = 129264
    ids8 = torch.tensor([IMG, 5, 7, IMG, 11, 13, 17, IMG], device='cuda')  # image rows route with bias_vl when VL
    idsb = torch.randint(0, 120000, (300,), device='cuda')
    idsb[137:142] = ids8[:5]
    idsb[::7] = IMG
    print(f'[rank {rank}] VL={VL} bias_vl={mlp.gate.e_score_correction_bias_vl is not None}', flush=True)
    def call(xx, ii):
        return mlp(xx, None, input_ids=ii, input_ids_global=ii)

    orig = DeepseekV2MoE.forward
    with torch.no_grad():
        fi5 = orig(mlp, x8[:5].contiguous(), None, input_ids=ids8[:5], input_ids_global=ids8[:5])
        fi300 = orig(mlp, xb, None, input_ids=idsb, input_ids_global=idsb)
        I.ORIG_FORWARD = orig
        DeepseekV2MoE.forward = I.og_forward
        mlp._og_lw = lw
        og5 = call(x8[:5].contiguous(), ids8[:5])
        og300 = call(xb, idsb)
        og5b = call(x8[:5].contiguous(), ids8[:5])
        # graph capture of the step path
        xs = x8[:5].contiguous().clone()
        from sglang.srt.distributed.parallel_state import graph_capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with graph_capture() as ctx:
            for _ in range(2):
                call(xs, ids8[:5])
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=ctx.stream):
                gout = call(xs, ids8[:5])
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        gsame = torch.equal(gout, og5)
        # step padding: W=2 graph with one valid row equals the 1-row result
        from split_nv import og_moe as OM
        v = OM.valid_rows(xs.device)
        one = call(x8[:1].contiguous(), ids8[:1])
        v.fill_(1)
        two = call(x8[:2].contiguous(), ids8[:2])
        v.fill_(1 << 30)
        pad_ok = torch.equal(two[:1], one)
    res = dict(rank=rank, og_vs_fi_5=rel(og5, fi5), og_vs_fi_300=rel(og300, fi300),
               step_eq_prefill=torch.equal(og300[137:142], og5), repeat=torch.equal(og5, og5b), graph=gsame,
               pad=pad_ok)
    print(f'[rank {rank}] {res}', flush=True)
    torch.distributed.barrier()
    if os.environ.get('TIME') == '1':
        timing(rank, mlp, orig, I, tr)


def timing(rank, mlp, orig, I, tr):
    """Per-layer MoE time, original (production path, dual-stream graph) vs og-moe, cold experts: 24 different
    inputs per graph (trace rows of all layers through this layer's gate)."""
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from split_nv import og_moe as OM
    xs_all = torch.cat([tr[f'{l}.post_attention_layernorm'][1] for l in range(20)]).cuda()  # 160 real rows
    ids_all = torch.randint(0, 120000, (xs_all.shape[0],), device='cuda')
    from split_nv.hooks import _install_b12x
    _install_b12x()  # the engine's MXFP8 dense GEMM backend (shared expert in the original path)
    from flashinfer.autotuner import autotune
    with torch.no_grad(), autotune(True):
        for M in (1, 2, 3, 4, 5, 8):
            orig(mlp, xs_all[:M].contiguous(), None, input_ids=ids_all[:M], input_ids_global=ids_all[:M])
    torch.cuda.synchronize()
    out = {}
    for M, valid in ((1, None), (2, 1), (2, None), (3, None), (4, None), (5, None), (8, None)):
        inputs = [(xs_all[(k * 7) % 150:(k * 7) % 150 + M].contiguous(), ids_all[:M].contiguous()) for k in range(24)]
        if rank == 0:
            nd = []
            for x_, i_ in inputs:
                lg = mlp.gate(x_)
                to = mlp.topk(x_, lg)
                nd.append(len(set(to.topk_ids[:(valid or M)].flatten().tolist())))
            print(f'M={M} distinct experts per call: mean {sum(nd) / len(nd):.1f}', flush=True)
        for name, fwd in (('orig', orig), ('og', I.og_forward)):
            DeepseekV2MoE.forward = fwd
            v = OM.valid_rows(xs_all.device)
            if name == 'og' and valid is not None:
                v.fill_(valid)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), model_capture_mode(), graph_capture() as ctx:
                for x_, i_ in inputs[:2]:
                    mlp(x_, None, input_ids=i_, input_ids_global=i_)
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=ctx.stream):
                    for x_, i_ in inputs:
                        mlp(x_, None, input_ids=i_, input_ids_global=i_)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            best = 1e9
            for _ in range(5):
                torch.distributed.barrier()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); g.replay(); e1.record(); e1.synchronize()
                best = min(best, e0.elapsed_time(e1) * 1e3 / len(inputs))
            if os.environ.get('PROF') == '1' and M == 5:
                from torch.profiler import profile, ProfilerActivity
                torch.distributed.barrier()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    g.replay()
                    torch.cuda.synchronize()
                torch.distributed.barrier()
            if os.environ.get('PROF') == '1' and M == 5 and rank == 0:
                evs = [e for e in prof.events() if e.device_type.name == 'CUDA']
                agg = {}
                for e in evs:
                    k = e.name[:60]
                    agg[k] = agg.get(k, 0) + e.device_time
                for k, v_ in sorted(agg.items(), key=lambda kv: -kv[1])[:14]:
                    print(f'PROF {name} {v_ / len(inputs):7.1f} us/layer  {k}', flush=True)
            v.fill_(1 << 30)
            out[(M, valid, name)] = best
            del g
        if rank == 0:
            o, n = out[(M, valid, 'orig')], out[(M, valid, 'og')]
            print(f'TIME M={M}{" valid=" + str(valid) if valid else ""}: orig {o:6.1f} us/layer, og {n:6.1f} us/layer, '
                  f'saves {o - n:5.1f} us/layer = {20 * (o - n) / 1000:.2f} ms/step (20 layers)', flush=True)
    # prefill-sized calls (eager, like the engine's prefill chunks)
    for M in (1024, 4096):
        x = (torch.randn(M, 5120, device='cuda') * xs_all.float().std()).to(torch.bfloat16)
        ii = torch.randint(0, 120000, (M,), device='cuda')
        res = {}
        for name, fwd in (('orig', orig), ('og', I.og_forward)):
            DeepseekV2MoE.forward = fwd
            with torch.no_grad():
                mlp(x, None, input_ids=ii, input_ids_global=ii)
                torch.cuda.synchronize()
                best = 1e9
                for _ in range(3):
                    torch.distributed.barrier()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); mlp(x, None, input_ids=ii, input_ids_global=ii); e1.record(); e1.synchronize()
                    best = min(best, e0.elapsed_time(e1))
            res[name] = best
        if rank == 0:
            print(f'TIME prefill M={M}: orig {res["orig"]:.3f} ms/layer, og {res["og"]:.3f} ms/layer', flush=True)
    # prefill-sized calls (eager, like the engine's prefill chunks)
    for M in (1024, 4096):
        x = (torch.randn(M, 5120, device='cuda') * xs_all.float().std()).to(torch.bfloat16)
        ii = torch.randint(0, 120000, (M,), device='cuda')
        res = {}
        for name, fwd in (('orig', orig), ('og', I.og_forward)):
            DeepseekV2MoE.forward = fwd
            with torch.no_grad():
                mlp(x, None, input_ids=ii, input_ids_global=ii)
                torch.cuda.synchronize()
                best = 1e9
                for _ in range(3):
                    torch.distributed.barrier()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); mlp(x, None, input_ids=ii, input_ids_global=ii); e1.record(); e1.synchronize()
                    best = min(best, e0.elapsed_time(e1))
            res[name] = best
        if rank == 0:
            print(f'TIME prefill M={M}: orig {res["orig"]:.3f} ms/layer, og {res["og"]:.3f} ms/layer', flush=True)
    DeepseekV2MoE.forward = I.og_forward


if __name__ == '__main__':
    mp.spawn(worker, args=(29711,), nprocs=2, join=True)
