"""CPU-only proof that drafter_bench.py gets through the drafter's weight load (no GPU; run by load_dryrun.sh).

Two gloo ranks (TP2, the production sharding) run the bench's own code on host memory:
  1. SGLang's parallel/MoE/quant bootstrap as the bench's Mini engine gets it (same ServerArgs; the MoE runner backend
     pinned to flashinfer_mxfp4 and the platform facts to SM120, which is what the engine resolves on the box:
     journal "moe_runner_backend=flashinfer_mxfp4, quant_method=Mxfp4FlashinferCutlassMoEMethod"), and the global
     expert-location map computed exactly as ModelRunner computes it for the view (n_routed_experts = NE);
  2. REPRO: SGLang's DSpark load_weights with that map -> expects the window's IndexError;
  3. FIX: drafter_bench.load_draft (draft_expert_location + load tracking) -> every parameter written, no unexpected
     checkpoint tensor, the quant methods the GPU run selects, per-rank shapes/bytes;
  4. og_moe3.stage_weights (unchanged: og-moe's layout asserts against the checkpoint bytes) on the loaded experts,
     after the SM120 post-load step it depends on (flashinfer block_scale_interleave of the E8M0 scales, CUDA-only)
     is emulated with og-moe's own swizzle formula. Weights are untouched by that step on SM120.
Not covered (needs CUDA): process_weights_after_loading itself, graph capture, timing.
"""
import json
import os
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import drafter_bench as B  # noqa: E402

PLATFORM = dict(is_sm120=True, is_blackwell=True, is_sm90=False, is_sm100=False, is_sm100_or_sm110=False,
                has_flashinfer=True)
OUT = os.environ.get("DRYRUN_OUT", "/dsb/out/load_dryrun")


def server_args_for_view():
    from sglang.srt.server_args import PortArgs, ServerArgs
    import argparse
    p = argparse.ArgumentParser()
    ServerArgs.add_cli_args(p)
    sa = ServerArgs.from_cli_args(p.parse_args([
        "--model-path", B.VIEW, "--trust-remote-code", "--tp", "2", "--mem-fraction-static", "0.94",
        "--context-length", "131072", "--max-total-tokens", "20480", "--max-running-requests", "1",
        "--chunked-prefill-size", "8192", "--enable-deepseek-v4-fp4-indexer", "--fp8-gemm-backend", "flashinfer_cutlass",
        "--disable-cuda-graph", "--disable-radix-cache", "--device", "cpu",
        "--moe-runner-backend", "flashinfer_mxfp4"]))
    sa.enable_multimodal = False
    from sglang.srt.runtime_context import override_platform
    with override_platform(**PLATFORM):  # the resolver checks FP4-indexer support against the platform
        sa.resolve_once()
    return sa, PortArgs.init_new(sa)


def block_scale_interleave_cpu(scale_u8):
    """flashinfer.block_scale_interleave (CUDA) on CPU for a 2-D [rows, cols] E8M0 byte matrix: zero-pad to
    128-row / 4-column tiles and apply the 128x4 interleave (og-moe's _swz formula)."""
    rows, cols = scale_u8.shape
    R, C = -(-rows // 128) * 128, -(-cols // 4) * 4
    pad = torch.zeros(R, C, dtype=torch.uint8)
    pad[:rows, :cols] = scale_u8.view(torch.uint8)
    return swizzle_like_sm120(pad.unsqueeze(0))[0].contiguous()


def swizzle_like_sm120(scale_u8):
    """[E, rows, cols] E8M0 bytes -> the 128x4 block-interleaved layout og-moe reads (og_moe/install.py _swz)."""
    E, rows, cols = scale_u8.shape
    n = torch.arange(rows).view(-1, 1)
    kb = torch.arange(cols).view(1, -1)
    dst = (kb & 3) + (kb >> 2) * 512 + (n & 31) * 16 + ((n & 127) >> 5) * 4 + (n >> 7) * 128 * cols
    out = torch.empty(E, rows * cols, dtype=torch.uint8)
    out[:, dst.reshape(-1)] = scale_u8.reshape(E, -1)
    return out.view(E, rows, cols)


def worker(rank, sa, pa, res_q):
    rep = {"rank": rank}
    try:
        from sglang.srt.runtime_context import override_platform, publish
        with override_platform(**PLATFORM):
            publish(sa, role="scheduler")
            from sglang.srt.layers.moe import initialize_moe_config
            from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
            from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
            initialize_moe_config()
            initialize_fp8_gemm_config()
            initialize_fp4_gemm_config()
            from sglang.srt.configs.model_config import ModelConfig
            from sglang.srt.distributed import bootstrap
            from sglang.srt.distributed.parallel_state_wrapper import ParallelState
            from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
            mc = ModelConfig.from_server_args(sa)
            at_r, at_s, adp_r, adp_s = compute_dp_attention_world_info(False, rank, 2, 1, 1)
            ps = ParallelState(tp_rank=rank, tp_size=2, pp_rank=0, pp_size=1, dp_rank=None, dp_size=1,
                               attn_tp_rank=at_r, attn_tp_size=at_s, attn_cp_rank=0, attn_cp_size=1, attn_dcp_rank=0,
                               attn_dcp_size=1, attn_dp_rank=adp_r, attn_dp_size=adp_s, moe_ep_rank=0, moe_ep_size=1,
                               moe_dp_rank=None, moe_dp_size=1, gpu_id=rank)
            # CPU shim: gloo groups only (PyNccl / custom all-reduce need a CUDA device; nothing here communicates)
            from sglang.srt.distributed import parallel_state as PSt
            _orig_group = PSt.init_model_parallel_group

            def _cpu_group(*a, **k):
                k.update(use_pynccl=False, use_custom_allreduce=False, use_mscclpp_allreduce=False,
                         use_torch_symm_mem_allreduce=False)
                return _orig_group(*a, **k)

            PSt.init_model_parallel_group = _cpu_group
            bootstrap.init_torch_distributed(server_args=sa, model_config=mc, device="cpu", ps=ps,
                                             dist_port=pa.nccl_port, is_draft_worker=False, local_omp_cpuid=None)
            from sglang.srt.configs.load_config import LoadConfig
            from sglang.srt.model_loader.loader import _get_quantization_config
            qc = _get_quantization_config(mc, LoadConfig())
            from sglang.srt.eplb.expert_location import (compute_initial_expert_location_metadata,
                                                         set_global_expert_location_metadata)
            meta = compute_initial_expert_location_metadata(model_config=mc, moe_ep_rank=0)
            set_global_expert_location_metadata(meta)
            rep["target_expert_map"] = {"layers": int(meta.num_layers), "logical_experts": int(meta.num_logical_experts)}
            from sglang.srt.layers.quantization import mxfp4_flashinfer_cutlass_moe as MX
            MX.is_flashinfer_available = lambda: True  # the CPU container has flashinfer but no CUDA device
            cfg = mc.hf_config
            # the GPU bench builds the target through SGLang's loader first, which installs the TARGET's
            # shared-experts-fusion decision (loader.py _initialize_model); do the same before the draft build
            from sglang.srt.layers.moe.utils import install_shared_experts_fusion_decision, is_shared_experts_fusion_disabled
            from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
            install_shared_experts_fusion_decision(DeepseekV4ForCausalLM, cfg, qc)
            rep["target_shared_fusion_disabled"] = is_shared_experts_fusion_disabled()
            rep["config"] = {k: getattr(cfg, k, None) for k in ("n_routed_experts", "dspark_n_routed_experts",
                                                                 "dspark_num_experts_per_tok", "hc_pre_from_prev_sublayer",
                                                                 "q_head_norm", "vocab_size")}
            cpu = torch.device("cpu")

            # ---- 2. REPRO: SGLang's DSpark loader with the target-sized map
            d = B.make_draft(cfg, qc, cpu)
            try:
                d.load_weights(B.mtp_weights())
                rep["repro"] = "no error (UNEXPECTED: the window failed here)"
            except IndexError as e:
                rep["repro"] = f"IndexError reproduced: {str(e)[:160]}"
            del d

            # ---- 3. FIX: the bench's own load path
            t0 = time.time()
            d = B.make_draft(cfg, qc, cpu)
            info = B.load_draft(d, cpu, postprocess=False)
            seen, snap = info["seen"], info["snap"]
            rep["load_s"] = round(time.time() - t0, 1)
            rep["checkpoint_params"] = len(snap["params"])

            # ---- 3b. SM120 post-load derivation of the FP8 linears through SGLang's own code
            # (Fp8LinearMethod._prepare_block_fp8_as_mxfp8 -> block_fp8_scale_to_mxfp8_e8m0 -> cutlass branch ->
            # copy_or_rebind_param(weight_scale_inv_swizzled)), with the CUDA-only interleave emulated; then the
            # bench's check_derived must find exactly these as derived and validate them against their sources
            try:
                import flashinfer
            except Exception:  # noqa: BLE001
                import types
                flashinfer = sys.modules["flashinfer"] = types.ModuleType("flashinfer")
            flashinfer.block_scale_interleave = block_scale_interleave_cpu
            from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
            n_lin = 0
            for mname, m in d.named_modules():
                qm = getattr(m, "quant_method", None)
                if type(qm).__name__ == "Fp8LinearMethod":
                    qm.mxfp8_dense_backend = Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS  # what SM120 resolves
                    qm.block_fp8_gemv_max_m = 0  # the GEMV prebuild is a CUDA JIT (off by default in the engine)
                    qm._prepare_block_fp8_as_mxfp8(m)
                    if not getattr(m, "block_fp8_mxfp8_ready", False):
                        raise RuntimeError(f"{mname}: block-fp8 -> MXFP8 derivation did not run")
                    n_lin += 1
            derived = B.check_derived(d, snap)
            # negative controls: both checks must fire
            neg = {}
            drop = "stages.1.self_attn.wq_b.weight"
            try:
                B.check_written(snap, set(seen) - {drop}, [])
                neg["unwritten_param"] = "NOT caught"
            except RuntimeError:
                neg["unwritten_param"] = "caught"
            swz = d.stages[1].self_attn.wq_b.weight_scale_inv_swizzled
            keep = swz.data.clone()
            for label, spoil in (("zeroed_swizzled", lambda t: t.zero_()),
                                 ("altered_swizzled", lambda t: t.view(torch.uint8).reshape(-1)[
                                     int(torch.nonzero(t.view(torch.uint8).reshape(-1))[0])].add_(1))):
                spoil(swz.data)
                try:
                    B.check_derived(d, snap)
                    neg[label] = "NOT caught"
                except RuntimeError:
                    neg[label] = "caught"
                swz.data.copy_(keep)
            rep["negative_controls"] = neg
            if any(v != "caught" for v in neg.values()):
                raise RuntimeError(f"a negative control was not caught: {neg}")
            rep["derived"] = {"fp8_linears": n_lin, "count": len(derived),
                              "names": sorted({n.split(".", 2)[-1] if n.startswith("stages.") else n for n in derived}),
                              "all_match_source_scales": all(v.get("matches_source_scales") for v in derived.values()
                                                             if "matches_source_scales" in v)}
            params = list(d.named_parameters())
            rep["params"] = len(params)
            rep["params_written"] = len(seen)
            by = {}
            for n, p in params:
                key = ("experts" if ".mlp.experts." in n else "markov" if n.startswith("markov_head") else
                       "shared" if "shared_experts" in n else "attn" if "self_attn" in n else
                       "main_proj" if "main_proj" in n else "other")
                by[key] = by.get(key, 0) + p.numel() * p.element_size()
            rep["bytes_GiB"] = {k: round(v / 2**30, 4) for k, v in by.items()}
            rep["bytes_GiB_total_as_loaded"] = round(sum(by.values()) / 2**30, 3)
            mk = d.markov_head.markov_w2.weight
            rep["bytes_GiB_total_production"] = round((sum(by.values()) - mk.numel() * mk.element_size()
                                                      + mk.numel() // 2 * 2) / 2**30, 3)
            st0 = d.stages[0]
            rep["quant_methods"] = {
                "experts": type(st0.mlp.experts.quant_method).__name__,
                "wq_b": type(st0.self_attn.wq_b.quant_method).__name__,
                "main_proj": type(st0.main_proj.quant_method).__name__,
                "wo_a_dtype": str(st0.self_attn.wo_a.weight.dtype)}
            ex = st0.mlp.experts
            rep["expert_shapes"] = {n: [list(p.shape), str(p.dtype)] for n, p in ex.named_parameters()}
            rep["attn_shapes"] = {n: list(p.shape) for n, p in st0.self_attn.named_parameters() if n.endswith("weight")}
            rep["local_heads"] = [st0.self_attn.n_local_heads, st0.self_attn.n_local_groups]
            rep["stage_moe"] = {"num_experts": int(st0.mlp.experts.num_experts), "top_k": int(st0.mlp.topk.topk_config.top_k)
                                if hasattr(st0.mlp.topk, "topk_config") else None,
                                "num_fused_shared_experts": int(st0.mlp.num_fused_shared_experts),
                                "shared_gate_up": list(st0.mlp.shared_experts.gate_up_proj.weight.shape)}

            # ---- 4. og-moe's layout asserts on the loaded experts (SM120 scale interleave emulated)
            from split_nv.og_moe import og_moe3
            og = []
            for s, st in enumerate(d.stages):
                e = st.mlp.experts
                for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                    prm = getattr(e, name)
                    prm.data = swizzle_like_sm120(prm.data.view(torch.uint8)).view(prm.data.dtype)
                lw = og_moe3.stage_weights(st.mlp, s, rank, B.CKPT)
                og.append({"stage": s, "scaled": lw.scaled, "args": len(lw.args)})
            rep["og_moe3_layout"] = og
            rep["ok"] = True
    except Exception as e:  # noqa: BLE001
        rep["ok"] = False
        rep["error"] = f"{type(e).__name__}: {e}"[:600]
        rep["traceback"] = traceback.format_exc()[-3000:]
    res_q.put(rep)


def main():
    os.environ.update(B.ENV)
    # the b12x GEMM hook patches fp8_utils.flashinfer_mm_mxfp8, which only exists on CUDA builds; it swaps GEMM
    # kernels at run time and plays no part in loading
    os.environ["SPLIT_NV_B12X"] = "0"
    sa, pa = server_args_for_view()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, sa, pa, q)) for r in range(2)]
    for p in procs:
        p.start()
    reps = sorted([q.get() for _ in procs], key=lambda r: r["rank"])
    for p in procs:
        p.join(60)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "load_dryrun.json"), "w") as f:
        json.dump(reps, f, indent=1)
    for r in reps:
        print(json.dumps({k: v for k, v in r.items() if k != "traceback"}), flush=True)
        if not r.get("ok"):
            print(r.get("traceback", ""), flush=True)
    ok = all(r.get("ok") for r in reps) and all(str(r.get("repro", "")).startswith("IndexError") for r in reps)
    print("DRYRUN " + ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
