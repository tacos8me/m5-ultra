"""DSpark-on-box drafter microbench (gate (b), and the build's perf-recovery A/B), on the PRODUCTION drafter code.

The whole per-cycle drafter job (H2D of the committed taps -> ring append -> 3 DSpark stages -> target head ->
Markov chain with max-prob -> D2H of drafts + probabilities -> host cost-policy width choice -> launch of the next
graph) is timed as one TP2 CUDA graph per width, exactly as split_nv.dspark_box.BoxDrafter runs it in the engine
(ring table indexed by the session slot on the device, one graph per width).

Runs in the production image on both GPUs with the engine STOPPED (tools/dspark_box/run_bench.sh). Per rank:
  1. a Mini engine (split_nv.engine.Engine's load path) on a one-layer view (tools/dspark_box/bench_view.py):
     parallel state, quant config, the b12x / consistent hooks, the full-vocab embed_tokens + lm_head as the engine;
  2. the 3 DSpark stages loaded by split_nv.dspark_box (SGLang's DSparkV4Stage + DSpark loader + post-processing,
     written / derived parameter checks);
  3. split_nv.dspark_box.BoxDrafter with every perf lever available, switched per variant:
       head   fp8 (block-FP8 GEMV copy of the lm_head shard) | bf16 (the shard, cuBLAS)
       markov fused (split_nv.dspark_markov) | ref (the gate-(b) chain)
       moe    og3 (og-moe decode, 128 experts top-3) | sgl (SGLang FusedMoE; --tune-moe: FlashInfer-autotuned for the
              drafter's (M, 5120) shapes into a private cache first)
  4. per variant: the full job at each width (graph == eager, ranks equal, 300 timed cycles); components (append,
     stages, embed, head fp8/bf16, Markov fused/ref, all-reduce, MoE og3/sgl[/tuned]); FP8-GEMV row M-invariance;
     fused vs ref Markov token equality on random logits; memory.
Verdict: the production variant (--head/--markov/--moe, default fp8:fused:og3) at W=5 <= --gate-ms (1.8); W=4 is the
production width and is reported with it.

Inputs: synthetic taps unless --taps FILE.npz (a Mac dump, gate (c): taps [N,15360], prime_rows, start_pos,
cycle_rows [C], anchors [C], mac_drafts [C,4]) -> per-depth and prefix agreement of the CUDA drafts with Metal's.

usage (inside run_bench.sh): python3 drafter_bench.py [--iters 300] [--widths 4,5] [--variants ...] [--tune-moe]
"""
import argparse
import json
import os
import statistics as S
import sys
import time

import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
VIEW = os.environ.get("VIEW", "/dsb/view")
CKPT = os.environ.get("OG_CKPT", "/ckpt")
LAYER = int(os.environ.get("LAYER", "3"))
FIRST = int(os.environ.get("FIRST", "2"))
MEM_GB = float(os.environ.get("MEM_GB", "0"))  # optional torch cap per GPU (0 = none; engine stopped)
ENV = dict(SGLANG_SM120_FLASHMLA_BACKEND="flashinfer", SGLANG_FLASHINFER_MOE_FUSED_FINALIZE="0",
           SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE="0", SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB="128",
           SGLANG_OPT_USE_TOPK_V2="1", SPLIT_NV_HOOKS="1", SPLIT_NV_CONFIG=f"{VIEW}/config.json",
           SPLIT_NV_DIR="/dsb/shm", SPLIT_NV_MAX_TOKENS="65536", SPLIT_NV_TRIM="1", SPLIT_NV_B12X="1",
           SPLIT_NV_OG_MOE="0", SPLIT_NV_PF_OVERLAP="0")
H, TAP = 5120, 3 * 5120
RING, HC = 128, 4
BLOCK_MAX = 5
P0 = 65536  # first drafted position (ring primed with the 128 before it)
# production cost tables (Mac pipe_session._PIPE_COSTS, context < 131072 tier / _FUSED_COSTS), ms for L = 2..5
COSTS = {"pipe": (24.1, 26.1, 28.2, 30.8), "fused": (13.1, 15.0, 16.9, 18.8)}
DEFAULT_VARIANTS = "bf16:ref:og3,fp8:ref:og3,bf16:fused:og3,fp8:fused:og3"


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def _db():
    from split_nv import dspark_box
    return dspark_box


# ---- the production loader, re-exported for load_dryrun.py (CKPT from OG_CKPT) ----------------------------------
def mtp_weights():
    return _db().mtp_weights(CKPT)


def make_draft(cfg, qc, dev):
    return _db().make_draft(cfg, qc, dev)


def load_draft(d, dev, postprocess=True):
    return _db().load_draft(d, dev, CKPT, postprocess)


def check_written(snap, seen, unexpected):
    return _db().check_written(snap, seen, unexpected)


def check_derived(model, snap):
    return _db().check_derived(model, snap)


def choose_depth(probs, costs, max_depth=4):
    from split_nv.dspark_wire import choose_cost_depth
    return choose_cost_depth(probs, costs, max_depth)


# ------------------------------------------------------------------------------------------------ model loading
def patch_target(ne):
    import sglang.srt.distributed as D
    from sglang.srt.models import deepseek_v4 as M

    D.get_pp_indices = lambda n, r, s: (FIRST, LAYER + 1)  # layer FIRST gives the pool sizer full-token KV
    orig = M.DeepseekV4ForCausalLM.load_weights

    def filtered(weights):
        for name, w in weights:
            if ".ffn.experts." in name:
                if int(name.split(".ffn.experts.")[1].split(".")[0]) >= ne:
                    continue
            elif name.endswith((".ffn.gate.weight", ".ffn.gate.bias")):
                w = w[:ne]
            yield name, w

    M.DeepseekV4ForCausalLM.load_weights = lambda self, weights, is_nextn=False: orig(self, filtered(weights), is_nextn)


def make_engine(rank, server_args, port_args):
    from split_nv import engine as EM

    class Mini(EM.Engine):
        """Engine.__init__'s load path only (no step runner, prefix store, og-moe on the target, prefill overlap)."""

        def __init__(self, server_args, port_args, gpu_id, tp_rank):
            from sglang.benchmark.one_batch import load_model
            from sglang.srt.layers.moe import initialize_moe_config
            from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
            from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
            from sglang.srt.runtime_context import get_schedule, publish

            self.tp_rank = tp_rank
            publish(server_args, role="scheduler")
            initialize_moe_config()
            initialize_fp8_gemm_config()
            initialize_fp4_gemm_config()
            runner, _ = load_model(server_args, port_args, gpu_id, tp_rank)
            self.mr = runner.torch_runner
            self.page_size = get_schedule().page_size
            from split_nv import hooks as Hk
            self.cap = Hk.CAP
            if self.cap is not None:
                self.cap.auto = False
            from split_nv.consistent import install
            install()
            self.sessions, self.steps, self.store, self.vision = {}, None, None, False

    return Mini(server_args, port_args, rank, rank)


def build_draft(E, rank):
    tgt = E.mr.model
    dev = torch.device("cuda", rank)
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    d = make_draft(tgt.config, tgt.quant_config, dev)
    info = load_draft(d, dev)
    torch.cuda.synchronize()
    return d, torch.cuda.memory_allocated() - a0, info


# ------------------------------------------------------------------------------------------------ helpers
def capture(fn, rank=None):
    return _db().graph_capture(fn)


def time_graph(g, n, pre=None):
    """GPU ms per replay (events), median / p90."""
    ts = []
    for _ in range(n):
        if pre is not None:
            pre()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return stats(ts)


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"median": round(S.median(xs), 4), "p10": round(xs[len(xs) // 10], 4), "p90": round(xs[(9 * len(xs)) // 10], 4),
            "p99": round(xs[min(len(xs) - 1, (99 * len(xs)) // 100)], 4), "n": len(xs)}


class Levers:
    """Switch the drafter's perf levers in place (the FP8 head copy and the fused Markov stay allocated)."""

    def __init__(self, D):
        self.D = D
        self.fp8 = (D.head_fp8, D.head_scale)
        self.fused = D.fused

    def set(self, head, markov, moe):
        D = self.D
        if head == "fp8" and self.fp8[0] is None:
            raise ValueError("no FP8 head copy")
        D.head_fp8, D.head_scale = self.fp8 if head == "fp8" else (None, None)
        if markov == "fused" and self.fused is None:
            raise ValueError("no fused Markov")
        D.fused = self.fused if markov == "fused" else None
        if moe == "og3" and D.lw is None:
            raise ValueError("og3 weights unavailable")
        D.moe_mode = moe
        D.head_mode, D.markov_mode = head, markov


def tune_sgl_moe(D, rank, widths, out_dir):
    """FlashInfer autotune of SGLang's FusedMoE at the drafter's shapes (M in widths, 5120), both ranks reducing
    their timings over the TP CPU group (SGLang's own rule), cache written to a private file."""
    from flashinfer.autotuner import autotune
    from sglang.srt.model_executor.runner.flashinfer_autotune import _autotune_process_group
    path = os.path.join(out_dir, "fi-autotune-dspark-moe.json")
    keep = D.moe_mode
    D.moe_mode = "sgl"
    t0 = time.perf_counter()
    try:
        with _autotune_process_group(D.tp.cpu_group), autotune(True, cache=path):
            for W in widths:
                for s in range(3):
                    x = torch.randn(W, H, device=D.dev).to(torch.bfloat16)
                    D.moe(s, D.stages[s], x)
        torch.cuda.synchronize()
    finally:
        D.moe_mode = keep
    return {"cache": path, "seconds": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------------------------------------ worker
def worker(rank, args, port_args, server_args):
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    if MEM_GB:
        torch.cuda.set_per_process_memory_fraction(MEM_GB * 2**30 / torch.cuda.get_device_properties(rank).total_memory, rank)
    sys.path.insert(0, HERE)
    from sglang.srt.distributed import get_tp_group
    DB = _db()

    out = {"rank": rank}
    patch_target(args.ne)
    E = make_engine(rank, server_args, port_args)
    torch.cuda.synchronize()
    out["target_alloc_GiB"] = round(torch.cuda.memory_allocated() / 2**30, 3)
    d, loaded, info = build_draft(E, rank)
    out["draft_weights_GiB_as_loaded"] = round(loaded / 2**30, 3)
    out["checkpoint_params"] = len(info["snap"]["params"])
    out["derived_after_postprocess"] = info["derived"]
    tp = get_tp_group()
    log(rank, f"[bench] target alloc {out['target_alloc_GiB']} GiB; drafter weights {out['draft_weights_GiB_as_loaded']} GiB/rank")
    a_head = torch.cuda.memory_allocated()
    try:
        D = DB.BoxDrafter(E.mr.model, d, rank, nslots=2, head="fp8", markov="fused", moe="og3", ckpt=CKPT)
    except Exception as e:  # noqa: BLE001 -- og3 layout differs from the target's: SGLang's MoE only
        out["og3_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        log(rank, f"[bench] og_moe3 unusable ({out['og3_error']}); MoE = SGLang FusedMoE")
        D = DB.BoxDrafter(E.mr.model, d, rank, nslots=2, head="fp8", markov="fused", moe="sgl", ckpt=CKPT)
    torch.cuda.synchronize()
    out["drafter_obj_GiB"] = round((torch.cuda.memory_allocated() - a_head) / 2**30, 3)
    out["head_fp8_GiB"] = round((D.head_fp8.numel() + D.head_scale.numel() * 4) / 2**30, 3)
    out["draft_resident_GiB"] = round((torch.cuda.memory_allocated() - out["target_alloc_GiB"] * 2**30) / 2**30, 3)
    lev = Levers(D)
    prod = (args.head, args.markov, args.moe.split(",")[0])
    variants = [tuple(v.split(":")) for v in args.variants.split(",") if v]
    if prod not in variants:
        variants.append(prod)
    if D.lw is None:
        variants = [v for v in variants if v[2] != "og3"] or [(prod[0], prod[1], "sgl")]
    pv = prod if prod in variants else variants[-1]  # the production variant as far as this load allows
    out["variants"] = [":".join(v) for v in variants]

    # ---- inputs
    gen = torch.Generator().manual_seed(1234)
    dump = None
    if args.taps:
        z = __import__("numpy").load(args.taps)
        t = z["taps"]
        taps_all = torch.from_numpy(t.astype("uint16")).view(torch.bfloat16) if t.dtype.kind == "u" else \
            torch.from_numpy(t.astype("float32")).to(torch.bfloat16)
        dump = {k: z[k] for k in ("prime_rows", "start_pos", "cycle_rows", "anchors", "mac_drafts") if k in z}
        anchors = dump["anchors"].tolist() if "anchors" in dump else None
        mac_drafts = dump.get("mac_drafts")
    else:
        taps_all = (torch.randn(RING + args.iters * BLOCK_MAX + 64, TAP, generator=gen) * 0.5).to(torch.bfloat16)
        anchors, mac_drafts = None, None
    need = RING + (args.warmup + args.iters) * BLOCK_MAX + 64
    if taps_all.shape[0] < need:
        taps_all = taps_all.repeat((need + taps_all.shape[0] - 1) // taps_all.shape[0], 1)[:need]
    ids = json.load(open(args.ids))
    ids = ids["tokens"] if isinstance(ids, dict) else ids

    def stage(pos, n_app, anchor, rows, slot=0):
        """Taps of committed positions [pos - n_app, pos), anchor at pos (production _stage)."""
        D._stage(slot, pos - n_app, n_app, rows, anchor, pos)

    def prime():
        D.ring.zero_()
        D.valid.zero_()
        for k in range(0, RING, BLOCK_MAX):
            n = min(BLOCK_MAX, RING - k)
            stage(P0 - RING + k + n, n, ids[0], taps_all[k:k + n])
            D.append()
            torch.cuda.synchronize()

    out["rope_selfcheck"] = D.rope_selfcheck()
    log(rank, f"[bench] rope self-check {out['rope_selfcheck']}")
    widths = [int(w) for w in args.widths.split(",")]
    if args.load_only:
        widths = [max(widths)]
        variants = [pv]

    # ---- eager correctness pass (production variant, W = 5)
    lev.set(*pv)
    prime()
    stage(P0, 1, ids[1], taps_all[RING:RING + 1])
    toks_e, probs_e = D.full(BLOCK_MAX)
    torch.cuda.synchronize()
    same = tp.all_gather(toks_e.view(1, -1), dim=0)
    out["eager"] = {"variant": f"{D.head_mode}:{D.markov_mode}:{D.moe_mode}", "tokens": toks_e.tolist(),
                    "probs": [round(x, 4) for x in probs_e.tolist()],
                    "tokens_equal_across_ranks": bool((same == same[0:1]).all()),
                    "finite": bool(torch.isfinite(probs_e).all())}
    log(rank, f"[bench] eager: {out['eager']}")

    # ---- lever checks (no timing): FP8 GEMV row M-invariance, FP8 vs BF16 top-1, fused vs ref Markov tokens
    if not args.load_only:
        out["checks"] = lever_checks(D, lev, rank, tp)
        log(rank, f"[bench] lever checks: {json.dumps(out['checks'])}")

    if args.tune_moe and not args.load_only:
        out["moe_tune"] = tune_sgl_moe(D, rank, widths, args.out)
        log(rank, f"[bench] MoE autotune: {out['moe_tune']}")

    dummy = torch.zeros(1024, device=D.dev)
    g_next, _ = capture(lambda: dummy.add_(1.0))
    ctab = COSTS[args.costs]
    for var in variants:
        lev.set(*var)
        vname = ":".join(var)
        graphs = {}
        for W in widths:
            prime()
            stage(P0, 1, ids[1], taps_all[RING:RING + 1])
            g, (tk, pb) = capture(lambda W=W: D.full(W))
            graphs[W] = g
            prime()
            stage(P0, 1, ids[1], taps_all[RING:RING + 1])
            ref, rp = D.full(W)
            ref, rp = ref.clone(), rp.clone()
            prime()
            stage(P0, 1, ids[1], taps_all[RING:RING + 1])
            g.replay()
            torch.cuda.synchronize()
            out.setdefault("graph_eq_eager", {})[f"{vname}/W{W}"] = bool(torch.equal(tk, ref) and torch.equal(pb, rp))
        if args.load_only:
            return load_only_verdict(rank, args, out, tp, graphs, widths[0], D, prime, stage, ids, taps_all)
        for W in widths:
            prime()
            g = graphs[W]
            host_ms, gpu_ms, bubble_ms, decide_ms, depths = [], [], [], [], []
            pos, t_i, anchor = P0, RING, ids[1]
            for it in range(args.warmup + args.iters):
                n_app = 1 + (it % 4)
                rows = taps_all[t_i:t_i + n_app]
                t_i += n_app
                tp.barrier()
                if rank == 1 and args.r1_skew_us > 0:
                    t_s = time.perf_counter() + args.r1_skew_us * 1e-6
                    while time.perf_counter() < t_s:
                        pass
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                t0 = time.perf_counter()
                e0.record()
                stage(pos, n_app, anchor, rows)
                g.replay()
                e1.record()
                e1.synchronize()
                t1 = time.perf_counter()
                probs = D.h_probs[:W].tolist()
                depth = choose_depth(probs, ctab, max_depth=min(4, W))
                toks = D.h_toks[:W].tolist()
                t2 = time.perf_counter()
                e2.record()
                g_next.replay()
                torch.cuda.synchronize()
                if it >= args.warmup:
                    host_ms.append((t1 - t0) * 1e3)
                    decide_ms.append((t2 - t1) * 1e3)
                    gpu_ms.append(e0.elapsed_time(e1))
                    bubble_ms.append(e1.elapsed_time(e2))
                    depths.append(depth)
                pos += n_app
                anchor = toks[min(depth, W - 1)] if anchors is None else anchors[it % len(anchors)]
            job = [h + b for h, b in zip(host_ms, bubble_ms)]
            out.setdefault("full", {}).setdefault(vname, {})[W] = {
                "host_ms": stats(host_ms), "gpu_ms": stats(gpu_ms), "decide_ms": stats(decide_ms),
                "bubble_ms": stats(bubble_ms), "job_ms": stats(job),
                "depth_hist": {k: depths.count(k) for k in sorted(set(depths))}}
            log(rank, f"[bench] {vname} W={W}: job {out['full'][vname][W]['job_ms']}  gpu {out['full'][vname][W]['gpu_ms']}")
        del graphs

    # ---- component breakdown (W = 5 inputs)
    comp = {}
    lev.set(*pv)
    prime()
    stage(P0, 4, ids[1], taps_all[RING:RING + 4])
    g, _ = capture(D.append)
    comp["ring_append_5rows"] = time_graph(g, args.iters)
    hs = torch.randn(BLOCK_MAX, HC, H, device=D.dev, generator=torch.Generator(D.dev).manual_seed(7)).to(torch.bfloat16)
    pres = torch.softmax(torch.randn(BLOCK_MAX, HC, device=D.dev), -1).float()

    def st(s):
        D.bind()
        return D.stage(s, hs, pres if s else None, BLOCK_MAX)

    for s in range(3):
        g, _ = capture(lambda s=s: st(s))
        comp[f"stage{s}"] = time_graph(g, args.iters)
    g, _ = capture(lambda: D.embed(D.blk_ids[:BLOCK_MAX]))
    comp["embed"] = time_graph(g, args.iters)
    for head in ("fp8", "bf16"):
        lev.set(head, D.markov_mode, D.moe_mode)
        for W in widths:
            g, _ = capture(lambda: D.head(hs[:W], pres[:W]))
            comp[f"head_{head}_W{W}"] = time_graph(g, args.iters)
    lgs = torch.randn(BLOCK_MAX, D.vloc, device=D.dev, generator=torch.Generator(D.dev).manual_seed(9)) * 3
    for mk in ("fused", "ref"):
        lev.set(D.head_mode, mk, D.moe_mode)
        for W in widths:
            g, _ = capture(lambda W=W: D.markov(lgs, W))
            comp[f"markov_{mk}_W{W}"] = time_graph(g, args.iters)
    g, _ = capture(lambda: D.all_reduce(torch.ones(BLOCK_MAX, H, dtype=torch.bfloat16, device=D.dev)))
    comp["all_reduce_5x5120"] = time_graph(g, args.iters)
    xs = torch.randn(BLOCK_MAX, H, device=D.dev).to(torch.bfloat16)
    ys = {}
    for v in (["og3"] if D.lw is not None else []) + ["sgl"]:
        D.moe_mode = v
        name = "sgl_tuned" if v == "sgl" and args.tune_moe else v
        try:
            ys[name] = D.moe(1, D.stages[1], xs).float().clone()
            g, _ = capture(lambda: D.moe(1, D.stages[1], xs))
            comp[f"moe_{name}"] = time_graph(g, args.iters)
        except Exception as e:  # noqa: BLE001
            comp[f"moe_{name}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            log(rank, f"[bench] MoE variant {name} failed: {comp[f'moe_{name}']}")
    if len(ys) == 2:
        a_, b_ = list(ys.values())
        comp["moe_og3_vs_sgl_cos"] = round(torch.nn.functional.cosine_similarity(a_.flatten(), b_.flatten(), 0).item(), 6)
    out["components"] = comp
    log(rank, "[bench] components: " + json.dumps(comp))
    lev.set(*pv)

    # ---- offline draft agreement vs the Mac (gate c): replay the dumped cycles exactly (production variant)
    if mac_drafts is not None and "cycle_rows" in dump:
        npr, pos0 = int(dump["prime_rows"]), int(dump["start_pos"])
        D.ring.zero_()
        D.valid.zero_()
        for k in range(0, npr, BLOCK_MAX):
            n = min(BLOCK_MAX, npr - k)
            stage(pos0 - npr + k + n, n, int(anchors[0]), taps_all[k:k + n])
            D.append()
            torch.cuda.synchronize()
        pos, t_i = pos0, npr
        per_depth, prefix, cycles = [0, 0, 0, 0], 0, 0
        for c, n in enumerate(dump["cycle_rows"].tolist()):
            n = int(n)
            if n > BLOCK_MAX:
                raise ValueError(f"cycle {c}: {n} rows > {BLOCK_MAX}")
            stage(pos + n, n, int(anchors[c]), taps_all[t_i:t_i + n])
            t_i += n
            pos += n
            toks, _ = D.full(4)
            torch.cuda.synchronize()
            box, mac = toks.tolist(), [int(x) for x in mac_drafts[c][:4]]
            run = True
            for i in range(4):
                per_depth[i] += box[i] == mac[i]
                run = run and box[i] == mac[i]
                prefix += run
            cycles += 1
        out["mac_agreement"] = {"cycles": cycles, "per_depth": [round(x / max(1, cycles), 4) for x in per_depth],
                                "prefix_mean": round(prefix / max(1, cycles), 4)}
        log(rank, f"[bench] Metal vs CUDA drafts: {out['mac_agreement']}")

    out["max_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    out["max_alloc_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    tp.barrier()
    if rank == 0 or args.all_ranks:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"bench-rank{rank}.json"), "w") as f:
            json.dump(out, f, indent=1)
    if rank == 0:
        pname = ":".join(pv)
        full = out["full"][pname]
        W = max(widths)
        job, p90 = full[W]["job_ms"]["median"], full[W]["job_ms"]["p90"]
        sane = {"graph_eq_eager": all(out["graph_eq_eager"].values()),
                "tokens_equal_across_ranks": out["eager"]["tokens_equal_across_ranks"], "finite": out["eager"]["finite"],
                **{k: v for k, v in out.get("checks", {}).items() if isinstance(v, bool)}}
        table = {v: {f"W{w}": r[w]["job_ms"]["median"] for w in r} for v, r in out["full"].items()}
        verdict = {"gate_ms": args.gate_ms, "variant": pname, "width": W, "job_median_ms": job, "job_p90_ms": p90,
                   "w4_job_median_ms": full.get(4, {}).get("job_ms", {}).get("median"), "variants_job_median_ms": table,
                   "sanity": sane, "draft_resident_GiB": out.get("draft_resident_GiB"),
                   "verdict": ("GO" if job <= args.gate_ms else "NO-GO") + ("" if all(sane.values()) else " (sanity check failed)")}
        print("VERDICT " + json.dumps(verdict), flush=True)
        with open(os.path.join(args.out, "verdict.json"), "w") as f:
            json.dump(verdict, f)


def lever_checks(D, lev, rank, tp):
    """Bitwise / agreement checks of the perf levers on this rank's real weights (no timing)."""
    res = {}
    g = torch.Generator(D.dev).manual_seed(21)
    # FP8 GEMV: a row's logits must not depend on how many rows share the call (M = 1, 2, 4, 5, 8)
    if lev.fp8[0] is not None:
        from sglang.kernels.ops.gemm.sm120_block_fp8_gemv import sm120_block_fp8_gemv
        x = torch.randn(8, H, device=D.dev, generator=g).to(torch.bfloat16)
        rows = {m: sm120_block_fp8_gemv(x[:m].contiguous(), lev.fp8[0], lev.fp8[1]) for m in (1, 2, 4, 5, 8)}
        res["fp8_gemv_rows_m_invariant"] = all(torch.equal(rows[m][:1], rows[1][:1]) for m in rows) and all(
            torch.equal(rows[m][:m], rows[8][:m]) for m in rows)
        xs = torch.randn(64, H, device=D.dev, generator=g).to(torch.bfloat16)
        f8 = torch.cat([sm120_block_fp8_gemv(xs[i:i + 8].contiguous(), lev.fp8[0], lev.fp8[1]) for i in range(0, 64, 8)]).float()
        bf = torch.matmul(xs, D.lm_head.weight.T).float()
        res["fp8_vs_bf16_local_top1_agree"] = round(float((f8.argmax(-1) == bf.argmax(-1)).float().mean()), 4)
        res["fp8_vs_bf16_logit_maxabs"] = round(float((f8 - bf).abs().max()), 4)
    # fused vs reference Markov chain: same tokens on random logits; both ranks identical
    if lev.fused is not None:
        eq, n = 0, 32
        rel = 0.0
        for i in range(n):
            lg = torch.randn(BLOCK_MAX, D.vloc, device=D.dev, generator=g) * (1 + i % 4)
            D.blk_ids[0] = int(torch.randint(0, 129280, (1,), generator=g, device=D.dev))
            tf, pf = (t.clone() for t in lev.fused(lg, D.blk_ids[:1], BLOCK_MAX))
            tr, pr = D.markov_ref(lg, BLOCK_MAX)
            eq += bool(torch.equal(tf, tr))
            rel = max(rel, float(((pf - pr).abs() / pr.clamp_min(1e-30)).max()))
        allt = tp.all_gather(tf.view(1, -1), dim=0)
        res["markov_fused_eq_ref"] = f"{eq}/{n}"
        res["markov_fused_all_eq_ref"] = eq == n
        res["markov_fused_prob_maxrel"] = round(rel, 7)
        res["markov_fused_ranks_equal"] = bool((allt == allt[0:1]).all())
    torch.cuda.synchronize()
    return res


def load_only_verdict(rank, args, out, tp, graphs, W, D, prime, stage, ids, taps_all):
    """--load-only: load, post-load checks, og-moe layout on the real swizzled scales, ONE eager and ONE
    graph-captured drafter cycle (already run by the caller), plus one timed replay as a smoke number."""
    prime()
    stage(P0, 1, ids[1], taps_all[RING:RING + 1])
    t0 = time.perf_counter()
    graphs[W].replay()
    torch.cuda.synchronize()
    out["one_replay_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    xs = torch.randn(BLOCK_MAX, H, device=D.dev, generator=torch.Generator(D.dev).manual_seed(11)).to(torch.bfloat16)
    ys, keep = {}, D.moe_mode
    for v in ("og3", "sgl"):
        if v == "og3" and D.lw is None:
            continue
        try:
            D.moe_mode = v
            ys[v] = D.moe(1, D.stages[1], xs).float()
        except Exception as e:  # noqa: BLE001
            out[f"moe_{v}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    D.moe_mode = keep
    if len(ys) == 2:
        out["og3_vs_sgl_moe_cos"] = round(torch.nn.functional.cosine_similarity(ys["og3"].flatten(), ys["sgl"].flatten(), 0).item(), 6)
    checks = {
        "checkpoint_params_written": True,  # load_draft raised otherwise
        "derived_populated": out.get("derived_after_postprocess") is not None,
        "og_moe3_layout_on_swizzled_scales": "og3_error" not in out,
        "eager_finite": out["eager"]["finite"],
        "tokens_equal_across_ranks": out["eager"]["tokens_equal_across_ranks"],
        "graph_eq_eager": all(out.get("graph_eq_eager", {}).values()),
        "og3_matches_sgl_moe": out.get("og3_vs_sgl_moe_cos", 1.0) >= 0.995,
    }
    out["load_only_checks"] = checks
    tp.barrier()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"load-only-rank{rank}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if rank == 0:
        ok = all(checks.values())
        print("LOAD-ONLY " + ("PASS " if ok else "FAIL ") + json.dumps(
            {"checks": checks, "width": W, "one_replay_ms": out["one_replay_ms"], "draft_resident_GiB": out.get("draft_resident_GiB"),
             "checkpoint_params": out.get("checkpoint_params"), "derived": len(out.get("derived_after_postprocess") or {}),
             "eager": out["eager"], "og3_vs_sgl_moe_cos": out.get("og3_vs_sgl_moe_cos"), "moe_sgl_error": out.get("moe_sgl_error"),
             "rope_selfcheck": out.get("rope_selfcheck")}), flush=True)
        with open(os.path.join(args.out, "load-only-verdict.json"), "w") as f:
            json.dump({"pass": ok, "checks": checks}, f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--widths", default="4,5")
    ap.add_argument("--head", default="fp8", choices=("fp8", "bf16"), help="production variant (verdict)")
    ap.add_argument("--markov", default="fused", choices=("fused", "ref"))
    ap.add_argument("--moe", default="og3", help="production variant's MoE (og3|sgl)")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS, help="head:markov:moe list timed as full jobs")
    ap.add_argument("--tune-moe", action="store_true", help="FlashInfer-autotune SGLang's MoE at the drafter shapes first")
    ap.add_argument("--costs", default="pipe", choices=sorted(COSTS))
    ap.add_argument("--ne", type=int, default=8, help="target layer experts kept (the view's n_routed_experts)")
    ap.add_argument("--taps", default="")
    ap.add_argument("--ids", default="/ref/ids-131072.json")
    ap.add_argument("--gate-ms", type=float, default=1.8)
    ap.add_argument("--r1-skew-us", type=float, default=60.0,
                    help="rank 1 starts each cycle this late (production: rank 0 pipes the STEPD + taps to rank 1)")
    ap.add_argument("--out", default="/dsb/out")
    ap.add_argument("--all-ranks", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="compile og_moe3 and exit (no GPU work)")
    ap.add_argument("--load-only", action="store_true",
                    help="stop after load + post-load checks + og-moe layout + one eager and one graph cycle (W=max)")
    args = ap.parse_args()
    os.environ.update(ENV)
    sys.path.insert(0, HERE)
    from split_nv.og_moe import og_moe3
    og_moe3.ext()  # compile once in the parent, before the ranks exist (no GPU needed)
    if args.build_only:
        print("og_moe3 built", flush=True)
        return
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(parser.parse_args([
        "--model-path", VIEW, "--trust-remote-code", "--tp", "2", "--mem-fraction-static", "0.94",
        "--context-length", "131072", "--max-total-tokens", "20480", "--max-running-requests", "1",
        "--chunked-prefill-size", "8192", "--enable-deepseek-v4-fp4-indexer", "--fp8-gemm-backend", "flashinfer_cutlass",
        "--disable-cuda-graph", "--disable-radix-cache"]
        + (["--disable-flashinfer-autotune"] if os.environ.get("AUTOTUNE", "1") == "0" else [])))
    server_args.enable_multimodal = False
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    mp.spawn(worker, args=(args, port_args, server_args), nprocs=2)
    if args.load_only:
        try:
            ok = json.load(open(os.path.join(args.out, "load-only-verdict.json")))["pass"]
        except (OSError, ValueError, KeyError):
            ok = False
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
