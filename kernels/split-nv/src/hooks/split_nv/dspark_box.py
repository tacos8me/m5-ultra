"""DSpark drafter on the box (SPLIT_NV_DSPARK=1, default off; protocol: STEPD-SPEC.md v1, codec split_nv.dspark_wire).

Loaded once per rank at engine start, after the target and before any graph: SGLang's own DSparkV4Stage modules and
DSpark loader (DeepseekV4ForCausalLMDSpark.load_weights: name remap, FP8 wo_a dequant, FusedMoE MXFP4 experts, then
process_weights_after_loading), mtp.* tensors read from the original checkpoint, with the written/derived-parameter
checks the gate-(b) harness proved on GPU (tools/dspark_box/drafter_bench.py, window 2026-09-30). The engine's own
TP2 embed_tokens and lm_head are shared (the head is resident but unused by the step path: hooks.TRIM skips it).

Per cycle (engine.cmd_stepd, both ranks, identical commands): the committed rows' taps -> the session's ring slot
(append), then [anchor, noise x 3] through the 3 stages over ring + block -> head -> Markov chain -> drafts and
max-probs D2H, all ONE CUDA graph per width. Rings: a table of SLOTS sessions x 3 stages x (128 + 1 trash) x 512 BF16
(slot chosen by the front end, in the command), indexed on the device so one graph serves every session.

Perf recovery vs the gate-(b) job (W=4 1.78 ms, W=5 1.89 ms), each switchable for the window A/B:
  SPLIT_NV_DSPARK_HEAD=fp8|bf16     block-FP8 [32x32] copy of the lm_head shard on the fork's streaming GEMV
                                    (sm120_block_fp8_gemv, one warp per vocab row: rows independent of M), -0.18 ms E
  SPLIT_NV_DSPARK_MARKOV=fused|ref  split_nv.dspark_markov: 3 kernels + the 12-byte all-gather per draft (~15 nodes
                                    before); lowest-vocab-index ties; identical on both ranks
  SPLIT_NV_DSPARK_MOE=og3|sgl       og-moe decode kernel rebuilt for 128 experts top-3 (0.10 ms/stage) or SGLang's
                                    FusedMoE (0.20 ms untuned; tools/dspark_box/microbench.py tunes and re-times it)
  SPLIT_NV_DSPARK_FREE_BF16_HEAD=1  (default) free the BF16 head shard once the FP8 copy exists (-0.31 GiB/GPU; only
                                    with hooks.TRIM, i.e. when nothing on the box computes target logits)
  SPLIT_NV_DSPARK_STREAM=side|main  (default side; live flag dspark_side_stream) the drafter graph replays on its own
                                    stream, after the queued work, so the step's host bookkeeping that STEPD runs
                                    meanwhile (StepRunner.prepare) is not serialized behind it: prepare's pageable H2D
                                    copies synchronize the stream they run on. Timing only (same graph, same inputs)
Drafts change only which rows the Mac verifies, never their values (SPEC s0); determinism holds per input (fixed
shapes, no atomics in the Markov reductions, rank-identical all-gathered winners).
"""
import contextlib
import copy
import json
import os
import time

import torch

from split_nv import dspark_wire as WIRE

H = 5120
TAP = WIRE.TAP_DIM
HC, ROPE = 4, 64
RING, BLOCK_MAX, WIDTH = WIRE.RING, WIRE.BLOCK, WIRE.WIDTH
CKPT = os.environ.get("SPLIT_NV_DSPARK_CKPT", "/home/ian/models/DeepSeek-V4.1-Flash-original")


def enabled():
    return os.environ.get("SPLIT_NV_DSPARK", "0") == "1"


def settings():
    s = {"slots": int(os.environ.get("SPLIT_NV_DSPARK_SLOTS", "8")),
         "head": os.environ.get("SPLIT_NV_DSPARK_HEAD", "fp8"),
         "markov": os.environ.get("SPLIT_NV_DSPARK_MARKOV", "fused"),
         "moe": os.environ.get("SPLIT_NV_DSPARK_MOE", "og3"),
         "widths": tuple(int(w) for w in os.environ.get("SPLIT_NV_DSPARK_WIDTHS", str(WIDTH)).split(",") if w),
         # with the FP8 copy the BF16 shard is dead weight on the box (hooks.TRIM skips the head GEMM): -0.31 GiB/GPU
         "free_bf16_head": os.environ.get("SPLIT_NV_DSPARK_FREE_BF16_HEAD", "1") == "1",
         "stream": os.environ.get("SPLIT_NV_DSPARK_STREAM", "side")}
    if (s["head"] not in ("fp8", "bf16") or s["markov"] not in ("fused", "ref") or s["moe"] not in ("og3", "sgl")
            or s["stream"] not in ("side", "main")):
        raise ValueError(f"SPLIT_NV_DSPARK_* settings {s}")
    if WIDTH not in s["widths"] or any(not 1 <= w <= BLOCK_MAX for w in s["widths"]):
        raise ValueError(f"SPLIT_NV_DSPARK_WIDTHS {s['widths']}: must include {WIDTH}, each 1..{BLOCK_MAX}")
    return s


def log(*a):
    print("[engine] dspark:", *a, flush=True)


# ------------------------------------------------------------------------------------------------ loading
def mtp_weights(ckpt=CKPT):
    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator
    wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    files = sorted({os.path.join(ckpt, f) for k, f in wm.items() if k.startswith("mtp.")})
    for name, w in safetensors_weights_iterator(files):
        if name.startswith("mtp."):
            yield name, w


def draft_class():
    """DeepseekV4ForCausalLMDSpark reduced to what the box drafter holds: 3 DSparkV4Stage modules + the Markov head
    (the target's embed_tokens / lm_head are shared, the confidence head is not used). Its inherited load_weights is
    SGLang's own DSpark loader (name remap, FP8 wo_a dequant, stacked + FusedMoE expert mappings)."""
    from torch import nn
    from sglang.srt.models import deepseek_v4_dspark as DS

    class BoxDraft(DS.DeepseekV4ForCausalLMDSpark):
        def __init__(self, cfg, qc):
            nn.Module.__init__(self)
            self.config = copy.copy(cfg)
            # load_weights builds its expert-name mapping from config.n_routed_experts: the stages' count
            self.config.n_routed_experts = int(cfg.dspark_n_routed_experts)
            self.quant_config = qc
            from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled
            self.num_fused_shared_experts = 0 if is_shared_experts_fusion_disabled() else int(cfg.n_shared_experts)
            self.num_stages = 3
            self.stages = nn.ModuleList([
                DS.DSparkV4Stage(config=cfg, layer_id=i, stage_id=i, num_stages=3, num_target_layers=3,
                                 quant_config=qc, prefix=f"stages.{i}", alt_streams=None) for i in range(3)])
            self.markov_head = DS.DSparkV4MarkovHead(vocab_size=int(cfg.vocab_size), markov_rank=int(cfg.dspark_markov_rank))
            self.confidence_head = None
            self.embed_tokens = self.lm_head = None
            self.hc_mult = int(cfg.hc_mult)

    return BoxDraft


def make_draft(cfg, qc, dev):
    """Build the stages as SGLang builds a draft model: inside draft_model_build_scope, with the draft's own
    shared-experts-fusion decision installed first (og-moe needs the shared expert unfused)."""
    from sglang.srt.layers.moe.utils import draft_model_build_scope, install_shared_experts_fusion_decision
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models import deepseek_v4_dspark as DS
    with draft_model_build_scope():
        install_shared_experts_fusion_decision(DS.DeepseekV4ForCausalLMDSpark, cfg, qc)
        with set_default_torch_dtype(torch.bfloat16), dev:
            d = draft_class()(cfg, qc)
    if any(st.mlp.num_fused_shared_experts for st in d.stages) or d.num_fused_shared_experts:
        raise RuntimeError("drafter stages fused the shared expert (--enforce-shared-experts-fusion?): og-moe needs it separate")
    return d


@contextlib.contextmanager
def draft_expert_location():
    """Load the stages with no global expert-location map (FusedMoE's local path = the identity for TP-only MoE),
    then restore the target's (window 2026-09-30: a target-sized map indexes stage experts out of range)."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata, set_global_expert_location_metadata
    saved = get_global_expert_location_metadata()
    set_global_expert_location_metadata(None, allow_overwrite=True)
    try:
        yield saved
    finally:
        set_global_expert_location_metadata(saved, allow_overwrite=True)


def track_loads(model):
    from sglang.srt.model_loader.weight_utils import default_weight_loader
    seen = set()
    for name, p in model.named_parameters():
        orig = getattr(p, "weight_loader", default_weight_loader)

        def wrapped(*a, _orig=orig, _name=name, **k):
            seen.add(_name)
            return _orig(*a, **k)

        if isinstance(getattr(type(p), "weight_loader", None), property):
            p._weight_loader = wrapped
        else:
            p.weight_loader = wrapped
    return seen


def _tensor_attrs(model):
    return {mn: {k for k, v in vars(m).items() if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter)}
            for mn, m in model.named_modules()}


def load_snapshot(model):
    return {"params": dict(model.named_parameters()), "buffers": {n for n, _ in model.named_buffers()},
            "attrs": _tensor_attrs(model)}


def check_written(snap, seen, unexpected):
    never = sorted(n for n, p in snap["params"].items() if n not in seen and not getattr(p, "_skip_weight_check", False))
    if never or unexpected:
        raise RuntimeError(f"drafter load incomplete: never written {never[:20]} ({len(never)}), "
                           f"unexpected checkpoint tensors {unexpected[:5]} ({len(unexpected)})")


def check_derived(model, snap):
    """Every parameter/buffer registered by post-processing is populated; swizzled scales hold their source bytes."""
    out = {}
    named = dict(model.named_modules())
    new = [(n, p, True) for n, p in model.named_parameters() if n not in snap["params"]]
    new += [(n, b, True) for n, b in model.named_buffers() if n not in snap["buffers"]]
    for mn, m in named.items():
        for k, v in vars(m).items():
            if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter) and k not in snap["attrs"].get(mn, set()):
                new.append((f"{mn}.{k}" if mn else k, v, False))
    bad, warn = [], []
    for n, t, fatal in new:
        sink = bad if fatal else warn
        rec = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if t.numel() == 0:
            sink.append(f"{n}: empty")
        elif t.is_floating_point() and t.element_size() > 1:
            if not bool(torch.isfinite(t).all()):
                sink.append(f"{n}: non-finite")
        elif int(torch.count_nonzero(t.detach().contiguous().view(torch.uint8))) == 0:
            sink.append(f"{n}: all zero")
        if n.endswith("weight_scale_inv_swizzled"):
            mod = named[n.rsplit(".", 1)[0]]
            try:
                from sglang.srt.layers.quantization.fp8_utils import block_fp8_scale_to_mxfp8_e8m0
                src = block_fp8_scale_to_mxfp8_e8m0(mod.weight_scale_inv.data, tuple(mod.weight.shape),
                                                    mod.quant_method.quant_config.weight_block_size)
                a = t.detach().contiguous().view(torch.uint8).reshape(-1)
                b = src.reshape(-1)
                if not torch.equal(torch.sort(a[a != 0])[0], torch.sort(b[b != 0])[0]):
                    bad.append(f"{n}: bytes are not a permutation of its source scales")
            except Exception as e:  # noqa: BLE001 -- the generic checks above still apply
                rec["source_check"] = f"skipped: {type(e).__name__}"
        out[n] = rec
    if warn:
        out["_warnings"] = warn
    if bad:
        raise RuntimeError(f"derived buffers not populated: {bad[:10]} ({len(bad)})")
    return out


def load_draft(d, dev, ckpt=CKPT, postprocess=True):
    import logging
    from sglang.srt.model_loader.loader import DefaultModelLoader
    snap = load_snapshot(d)
    seen = track_loads(d)
    unexpected = []

    class Catch(logging.Handler):
        def emit(self, rec):
            if "unexpected weight" in rec.getMessage():
                unexpected.append(rec.getMessage()[:200])

    orig_load = d.load_weights

    def load_then_check(weights):
        orig_load(weights)
        check_written(snap, seen, unexpected)

    h = Catch()
    lg = logging.getLogger("sglang.srt.models.deepseek_v4_dspark")
    lg.addHandler(h)
    d.load_weights = load_then_check
    try:
        with draft_expert_location():
            if postprocess:
                DefaultModelLoader.load_weights_and_postprocess(d, mtp_weights(ckpt), dev)
            else:
                d.load_weights(mtp_weights(ckpt))
    finally:
        del d.load_weights
        lg.removeHandler(h)
    derived = check_derived(d, snap) if postprocess else None
    return {"snap": snap, "seen": seen, "derived": derived}


def act_quant(x):
    """Mac quantize_activation(bits=8, group_size=32): FP8 E4M3 per 32 features, power-of-two (UE8M0) scale."""
    g = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(448.0 * 2.0 ** -126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return ((g / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape).to(x.dtype)


def graph_capture(fn):
    """CUDA-graph capture of fn in SGLang's graph context and the engine's global graph memory pool (shared with the
    step graphs), after two eager warm-ups with the ranks aligned."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.parallel_state import graph_capture as gc
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.model_executor.runner_utils import pool as P

    stream = P.get_or_create_global_graph_capture_stream()
    try:
        mempool = P.get_or_create_global_graph_memory_pool(torch.cuda)
    except Exception:  # noqa: BLE001
        mempool = P.get_global_graph_memory_pool()
    with model_capture_mode(), gc(stream=stream):
        for _ in range(2):
            torch.cuda.synchronize()
            get_tp_group().barrier()
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with P.graph_pool_capture_scope(), torch.cuda.graph(g, pool=mempool, stream=stream):
            out = fn()
    torch.cuda.synchronize()
    return g, out


# ------------------------------------------------------------------------------------------------ the drafter
class BoxDrafter:
    def __init__(self, model, d, rank, nslots=8, head="fp8", markov="fused", moe="og3", ckpt=CKPT, free_bf16_head=False,
                 stream="main"):
        from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        from sglang.kernels.ops.layernorm.mhc import hc_combine

        self.d, self.rank, self.nslots = d, rank, int(nslots)
        self.tp = get_tp_group()
        self.all_reduce = tensor_model_parallel_all_reduce
        self.TopKOutputChecker = TopKOutputChecker
        self.hc_combine = hc_combine
        self.stages = list(d.stages)
        self.embed = model.model.embed_tokens
        self.lm_head = model.lm_head
        self.dev = self.lm_head.weight.device
        a = self.stages[0].self_attn
        self.freqs = a.freqs_cis
        self.nh, self.hd, self.ng, self.olr = a.n_local_heads, a.head_dim, a.n_local_groups, a.o_lora_rank
        self.scale = a.softmax_scale
        self.sinks = [s.self_attn._local_attn_sink()[:self.nh].float().contiguous() for s in self.stages]
        self.wo_a = [s.self_attn.wo_a.weight.view(self.ng, self.olr, -1) for s in self.stages]
        V = int(model.config.vocab_size)
        self.vloc = int(self.lm_head.weight.shape[0])
        self.v0 = self.tp.rank_in_group * self.vloc
        self.v_real = max(0, min(self.vloc, V - self.v0))
        self.noise = int(model.config.dspark_noise_token_id)
        # Markov: rank-256 embedding (replicated) and this rank's vocab rows of the projection, BF16 (as the Mac)
        mk = d.markov_head
        self.mk_w1 = mk.markov_w1.weight
        w2 = mk.markov_w2.weight.data
        self.mk_w2 = torch.zeros(self.vloc, w2.shape[1], dtype=torch.bfloat16, device=self.dev)
        self.mk_w2[:self.v_real].copy_(w2[self.v0:self.v0 + self.v_real].to(torch.bfloat16))
        del mk.markov_w2
        # head
        self.head_mode = head
        self.head_fp8 = self.head_scale = None
        if head == "fp8":
            from sglang.srt.layers.dsv41_lm_head_fp8 import quantize_block_fp8
            from sglang.kernels.ops.gemm.sm120_block_fp8_gemv import prebuild_sm120_block_fp8_gemv
            self.head_fp8, self.head_scale = quantize_block_fp8(self.lm_head.weight.data)
            prebuild_sm120_block_fp8_gemv(self.vloc, H, 8)
            if free_bf16_head:
                self._free_bf16_head(model)
        # MoE
        self.moe_mode = moe
        self.lw = None
        if moe == "og3":
            from split_nv.og_moe import og_moe3
            self.og3 = og_moe3
            self.lw = [og_moe3.stage_weights(st.mlp, s, self.tp.rank_in_group, ckpt) for s, st in enumerate(self.stages)]
            og_moe3.workspace(self.dev)
        # static buffers (graph inputs / outputs)
        z = dict(device=self.dev)
        self.ring = torch.zeros(self.nslots, 3, RING + 1, self.hd, dtype=torch.bfloat16, **z)  # [.., RING] = trash
        self.valid = torch.zeros(self.nslots, RING + 1, dtype=torch.bool, **z)
        self.taps = torch.zeros(BLOCK_MAX, TAP, dtype=torch.bfloat16, **z)
        # per-launch metadata, one pinned H2D: append positions, ring slots (RING = trash), block ids, block positions,
        # session slot (every column)
        self.meta = torch.zeros(5, BLOCK_MAX, dtype=torch.int64, **z)
        self.h_meta = torch.zeros(5, BLOCK_MAX, dtype=torch.int64, pin_memory=True)
        self.h_taps = torch.zeros(BLOCK_MAX, TAP, dtype=torch.bfloat16, pin_memory=True)
        self.h_toks = torch.zeros(BLOCK_MAX, dtype=torch.int64, pin_memory=True)
        self.h_probs = torch.zeros(BLOCK_MAX, dtype=torch.float32, pin_memory=True)
        self.app_pos, self.app_slot, self.blk_ids, self.blk_pos, self.sess = (self.meta[i] for i in range(5))
        self.markov_mode = markov
        self.fused = None
        if markov == "fused":
            from split_nv.dspark_markov import FusedMarkov
            self.fused = FusedMarkov(self.mk_w1, self.mk_w2, self.v0, self.v_real, self.tp.world_size,
                                     lambda t: self.tp.all_gather(t, dim=0), BLOCK_MAX, self.dev)
        self.fast_rope = True
        self.graphs = {}
        self.staged = None  # event after the last launch that read the pinned staging buffers
        # the drafter's own stream (SPLIT_NV_DSPARK_STREAM=side): see the module docstring
        self.side = torch.cuda.Stream(device=self.dev) if stream == "side" and self.dev.type == "cuda" else None
        self.on_side = None  # the side stream of the launched, not yet collected draft
        self.stats = {"drafts": 0, "appends": 0, "rings": 0, "draft_ms_last": None}

    def _free_bf16_head(self, model):
        from split_nv import hooks
        w = self.lm_head.weight
        if not getattr(hooks, "TRIM", False):
            log("keeping the BF16 head: hooks.TRIM is off (the causal forward computes logits)")
            return
        if w.data_ptr() == self.embed.weight.data_ptr():
            log("keeping the BF16 head: tied to embed_tokens")
            return
        self.lm_head._parameters["weight"] = torch.nn.Parameter(torch.empty(0, H, dtype=w.dtype, device=w.device),
                                                              requires_grad=False)
        del w
        torch.cuda.empty_cache()
        log("freed the BF16 lm_head shard (FP8 copy serves the drafter)")

    # ---- pieces (graph bodies) -------------------------------------------------------------------------------
    def rope(self, x, pos, inverse=False):
        if self.fast_rope:
            try:
                from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
                fused_rope_inplace(x, None, self.freqs, positions=pos, inverse=inverse)
                return x
            except Exception as e:  # noqa: BLE001
                self.fast_rope = False
                log(f"fused_rope_inplace unavailable ({str(e)[:120]}); torch rope")
        from sglang.srt.models.deepseek_v4_dspark import apply_rotary_emb
        return apply_rotary_emb(x, self.freqs[pos], inverse=inverse)

    def rope_selfcheck(self):
        """fused_rope_inplace on the strided views used here vs the torch reference (a layout mismatch would not
        raise): falls back to torch rope on any difference > 0.05 or a touched non-rope feature."""
        from sglang.srt.models.deepseek_v4_dspark import apply_rotary_emb
        res = {}
        pos = torch.arange(65536, 65536 + BLOCK_MAX, device=self.dev)
        g = torch.Generator(self.dev).manual_seed(3)
        for heads in (self.nh, 1):
            for inv in (False, True):
                x = torch.randn(BLOCK_MAX, heads, self.hd, device=self.dev, generator=g).to(torch.bfloat16)
                a, b = x.clone(), x.clone()
                try:
                    from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
                    fused_rope_inplace(a[..., -ROPE:], None, self.freqs, positions=pos, inverse=inv)
                except Exception as e:  # noqa: BLE001
                    res[f"h{heads}_inv{int(inv)}"] = f"error {type(e).__name__}"
                    self.fast_rope = False
                    continue
                apply_rotary_emb(b[..., -ROPE:], self.freqs[pos], inverse=inv)
                dd = float((a.float() - b.float()).abs().max())
                res[f"h{heads}_inv{int(inv)}"] = round(dd, 5)
                if not dd <= 0.05 or not bool(torch.equal(a[..., :-ROPE], x[..., :-ROPE])):
                    self.fast_rope = False
        res["fast_rope"] = self.fast_rope
        return res

    def kv_rows(self, stage, x, pos):
        a = stage.self_attn
        kv, _ = a.wkv(x)
        kv = a.kv_norm(kv).view(-1, 1, self.hd)
        self.rope(kv[..., -ROPE:], pos)
        return act_quant(kv.view(-1, self.hd))

    def append(self):
        """The staged taps rows (BLOCK_MAX, padding rows -> trash slot RING) -> every stage's ring of the session."""
        s0 = self.stages[0]
        x, _ = s0.main_proj(self.taps)
        x = s0.main_norm(x)
        base = self.sess * (3 * (RING + 1))
        flat = self.ring.view(-1, self.hd)
        for s, stage in enumerate(self.stages):
            flat.index_copy_(0, base + s * (RING + 1) + self.app_slot, self.kv_rows(stage, x, self.app_pos))
        self.valid.view(-1).index_fill_(0, self.sess * (RING + 1) + self.app_slot, True)

    def attention(self, s, stage, x, W):
        a = stage.self_attn
        pos = self.blk_pos[:W]
        q, _ = a.wq_a(x)
        q = a.q_norm(q)
        q, _ = a.wq_b(q)
        q = q.view(W, self.nh, self.hd)
        self.rope(q[..., -ROPE:], pos)
        kv = self.kv_rows(stage, x, pos)  # the block's own keys (never committed)
        keys = torch.cat([self._ring[s, :RING], kv], 0).float()  # K = V (MQA)
        sc = torch.einsum("whd,kd->whk", q.float(), keys) * self.scale
        mask = torch.cat([self._valid[:RING], torch.ones(W, dtype=torch.bool, device=self.dev)])
        sc = sc.masked_fill(~mask, float("-inf"))
        sink = self.sinks[s].view(1, self.nh, 1).expand(W, self.nh, 1)
        p = torch.softmax(torch.cat([sc, sink], -1), -1)[..., :-1]  # the sink takes mass, no value
        o = torch.einsum("whk,kd->whd", p, keys).to(torch.bfloat16)
        self.rope(o[..., -ROPE:], pos, inverse=True)
        o = torch.einsum("bgd,grd->bgr", o.reshape(W, self.ng, -1), self.wo_a[s])
        out, _ = a.wo_b(o.reshape(W, -1))  # row-parallel: all-reduce inside
        return out

    def moe(self, s, stage, x):
        mlp = stage.mlp
        if self.moe_mode == "sgl":
            return mlp(x, None)
        router_logits = mlp.gate(x, None)
        topk = mlp.topk(x, router_logits, num_token_non_padded=None, expert_location_dispatch_info=None)
        if self.TopKOutputChecker.format_is_bypassed(topk):
            topk = topk.to_standard()
        ids = topk.topk_ids.to(torch.int32).contiguous()
        w = topk.topk_weights.float()
        lw = self.lw[s]
        if not lw.scaled:
            w = w * mlp.routed_scaling_factor
        return self.all_reduce(self.og3.moe(x.contiguous(), ids, w.contiguous(), lw))

    def stage(self, s, h, pre, W):
        st = self.stages[s]
        residual = h
        x, a_pre, a_post, a_comb = st._hc_mix_and_combine(h, st.hc_attn_fn, st.hc_attn_scale, st.hc_attn_base, apply_pre=pre)
        x = st.input_layernorm(x)
        h = st.hc_post(self.attention(s, st, x, W), residual, a_post, a_comb)
        residual = h
        x, f_pre, f_post, f_comb = st._hc_mix_and_combine(h, st.hc_ffn_fn, st.hc_ffn_scale, st.hc_ffn_base, apply_pre=a_pre)
        x = st.post_attention_layernorm(x)
        h = st.hc_post(self.moe(s, st, x), residual, f_post, f_comb)
        return h, f_pre

    def head(self, h, pre):
        x = self.hc_combine(h.flatten(1).float(), pre, HC, torch.bfloat16)
        x = self.stages[-1].norm(x)
        if self.head_fp8 is not None:
            from sglang.kernels.ops.gemm.sm120_block_fp8_gemv import sm120_block_fp8_gemv
            logits = sm120_block_fp8_gemv(x.to(torch.bfloat16).contiguous(), self.head_fp8, self.head_scale).float()
        else:
            logits = torch.matmul(x, self.lm_head.weight.T).float()  # [W, vloc]: this rank's vocab shard
        if self.v_real < self.vloc:
            logits[:, self.v_real:] = float("-inf")
        return logits

    def markov_ref(self, logits, W):
        """The gate-(b) reference chain (drafter_bench.BoxDrafter.markov): kept for the A/B and as a fallback."""
        prev = self.blk_ids[:1]
        toks, probs = [], []
        for i in range(W):
            bias = torch.nn.functional.linear(self.mk_w1[prev], self.mk_w2).float()
            step = logits[i:i + 1] + bias
            idx = torch.argmax(step, -1)
            m = step.gather(-1, idx.view(1, 1)).view(1)
            lse = torch.logsumexp(step, -1)
            pack = torch.stack([m, (idx + self.v0).float(), lse], -1)
            g = self.tp.all_gather(pack, dim=0)
            gmax = g[:, 0].max()
            win = torch.argmax((g[:, 0] == gmax).to(torch.int32))
            tok = g.index_select(0, win.view(1))[:, 1].long()
            probs.append(torch.exp(gmax - torch.logsumexp(g[:, 2], 0)).view(1))
            toks.append(tok)
            prev = tok
        return torch.cat(toks), torch.cat(probs)

    def markov(self, logits, W):
        if self.fused is not None:
            return self.fused(logits, self.blk_ids[:1], W)
        return self.markov_ref(logits, W)

    def bind(self):
        """The staged session's ring and validity ([3, RING + 1, hd], [RING + 1]), gathered on the device."""
        self._ring = self.ring.index_select(0, self.sess[:1])[0]
        self._valid = self.valid.index_select(0, self.sess[:1])[0]

    def block(self, W):
        self.bind()
        h = self.embed(self.blk_ids[:W])  # vocab-parallel: all-reduce inside
        h = h.unsqueeze(1).repeat(1, HC, 1).contiguous()
        pre = None
        for s in range(3):
            h, pre = self.stage(s, h, pre, W)
        return self.head(h, pre)

    def full(self, W):
        """One cycle: append the staged rows, draft W tokens, D2H drafts + max-probs (pinned)."""
        self.append()
        toks, probs = self.markov(self.block(W), W)
        self.h_toks[:W].copy_(toks, non_blocking=True)
        self.h_probs[:W].copy_(probs, non_blocking=True)
        return toks, probs

    # ---- graphs ----------------------------------------------------------------------------------------------
    def capture(self, widths):
        """Append graph (BLOCK_MAX rows) and one full-cycle graph per width; checks graph == eager (drafts bitwise)
        on a synthetic primed ring in slot 0, then leaves every slot invalid."""
        res = {"rope": self.rope_selfcheck()}
        with torch.no_grad():
            self._synthetic(0)
            self.graphs["append"], _ = graph_capture(self.append)
            for W in widths:
                self._synthetic(0)
                g, (tk, pb) = graph_capture(lambda W=W: self.full(W))
                self.graphs[W] = g
                self._synthetic(0)
                ref, rp = self.full(W)
                ref, rp = ref.clone(), rp.clone()
                self._synthetic(0)
                g.replay()
                torch.cuda.synchronize()
                res[f"graph_eq_eager_W{W}"] = bool(torch.equal(tk, ref) and torch.equal(pb, rp))
                res[f"eager_tokens_W{W}"] = ref.tolist()
                same = self.tp.all_gather(ref.view(1, -1), dim=0)
                res[f"ranks_equal_W{W}"] = bool((same == same[0:1]).all())
        self.valid.zero_()
        self.ring.zero_()
        torch.cuda.synchronize()
        res["failed"] = [k for k, v in res.items() if k.startswith(("graph_eq", "ranks_equal")) and not v]
        return res

    def _synthetic(self, slot):
        """Deterministic synthetic ring content for capture checks: 128 rows through the eager append."""
        g = torch.Generator().manual_seed(1234)
        self.valid[slot].zero_()
        base = 65536 - RING
        for k in range(0, RING, BLOCK_MAX):
            n = min(BLOCK_MAX, RING - k)
            rows = (torch.randn(n, TAP, generator=g) * 0.5).to(torch.bfloat16)
            self._stage(slot, base + k, n, rows, 1, 65536)
            self.append()
            torch.cuda.synchronize()
        self._stage(slot, 65536, 0, None, 1, 65536)
        torch.cuda.synchronize()

    # ---- host API (engine commands) ----------------------------------------------------------------------------
    def _wait_staging(self):
        if self.staged is not None:
            self.staged.synchronize()
            self.staged = None

    def _stage(self, slot, base, n, rows, anchor, keep):
        """Pinned staging + H2D for one launch: committed rows [base, base + n) (n <= BLOCK_MAX; rows: a BF16 tensor
        [n, TAP] or raw bytes), the block [anchor, noise...] at keep.., the session slot."""
        self._wait_staging()
        m = self.h_meta.numpy()
        for i in range(BLOCK_MAX):
            m[0, i] = base + i if i < n else 0
            m[1, i] = (base + i) % RING if i < n else RING
            m[2, i] = anchor if i == 0 else self.noise
            m[3, i] = keep + i
            m[4, i] = slot
        if n:
            if isinstance(rows, torch.Tensor):
                self.h_taps[:n].copy_(rows)
            else:
                import numpy as np
                self.h_taps.view(torch.int16).numpy()[:n] = np.frombuffer(rows, dtype="<i2", count=n * TAP).reshape(n, TAP)
        self.taps.copy_(self.h_taps, non_blocking=True)
        self.meta.copy_(self.h_meta, non_blocking=True)

    def _launched(self):
        ev = torch.cuda.Event()
        ev.record()
        self.staged = ev

    def run_append(self, slot, base, n, taps):
        """Append n committed rows [base, base + n) (raw BF16 bytes [n, TAP]) to a session's ring, 5 rows a launch."""
        mv = memoryview(taps).cast("B")
        g = self.graphs.get("append")
        for k in range(0, n, BLOCK_MAX):
            c = min(BLOCK_MAX, n - k)
            self._stage(slot, base + k, c, mv[k * WIRE.TAP_ROW_BYTES:(k + c) * WIRE.TAP_ROW_BYTES], 0, 0)
            if g is not None:
                g.replay()
            else:
                with torch.no_grad():
                    self.append()
            self._launched()
        self.stats["appends"] += n

    def _side_stream(self, graphed):
        """The stream this launch replays on: the side stream for a captured graph (eager ops would allocate on it)
        unless the live flag dspark_side_stream is off. Each rank reads the flag itself: a rank-local timing choice
        (same graph, same collectives in the same order on every rank; ordering is enforced per rank)."""
        if self.side is None or not graphed:
            return None
        from split_nv.perf_flags import flag
        return self.side if flag("dspark_side_stream", True) else None

    def launch_draft(self, slot, keep, anchor, base, n, taps, W):
        """Append the committed rows [base, base + n) and draft W tokens at keep (one graph); returns at once.
        On the side stream the graph waits for everything queued before it (the last step, a preempted prefill's
        segments): it then shares no time with any collective of the main stream, while the host's step
        bookkeeping (and its main-stream copies) proceeds. collect() orders the main stream after it."""
        mv = memoryview(taps).cast("B") if n else None
        if n > BLOCK_MAX:  # oldest rows first, append-only launches (main stream)
            extra = n - BLOCK_MAX
            self.run_append(slot, base, extra, mv[:extra * WIRE.TAP_ROW_BYTES])
            base, n, mv = base + extra, BLOCK_MAX, mv[extra * WIRE.TAP_ROW_BYTES:]
        g = self.graphs.get(W)
        side = self._side_stream(g is not None)
        ctx = contextlib.nullcontext()
        if side is not None:
            side.wait_stream(torch.cuda.current_stream())
            ctx = torch.cuda.stream(side)
        with ctx:
            self._stage(slot, base, n, mv, anchor, keep)
            if g is not None:
                g.replay()
            else:
                with torch.no_grad():
                    self.full(W)
            self._launched()
        self.on_side = side
        self.stats["drafts"] += 1
        self.stats["appends"] += n

    def drain(self):
        """The launched draft is complete and the main stream is ordered after it (the step graphs share its graph
        memory pool: they must never overlap it)."""
        self._wait_staging()
        if self.on_side is not None:
            torch.cuda.current_stream().wait_stream(self.on_side)
            self.on_side = None

    def collect(self, W):
        """Wait for the launched draft; -> (tokens, max-probs) as Python lists (identical on every rank)."""
        self.drain()
        return self.h_toks[:W].tolist(), self.h_probs[:W].tolist()

    def ring_set(self, slot, offset, keys, kr, taps, tr):
        """RING: the session's ring becomes exactly positions [offset - kr - tr, offset): kr stored keys (BF16 bytes
        [3, kr, KEY_DIM], oldest first) then tr taps rows (appended as per cycle)."""
        self._wait_staging()
        self.valid[slot].zero_()
        if kr:
            k = torch.frombuffer(bytearray(keys), dtype=torch.bfloat16).view(3, kr, self.hd).to(self.dev, non_blocking=False)
            pos = torch.arange(offset - kr - tr, offset - tr, dtype=torch.int64) % RING
            pos = pos.to(self.dev)
            self.ring[slot].index_copy_(1, pos, k)
            self.valid[slot].index_fill_(0, pos, True)
        if tr:
            self.run_append(slot, offset - tr, tr, taps)
        self._wait_staging()
        self.stats["rings"] += 1

    def summary(self):
        return {"head": self.head_mode, "markov": self.markov_mode, "moe": self.moe_mode, "slots": self.nslots,
                "side_stream": self.side is not None, "graphs": sorted(str(k) for k in self.graphs), **self.stats}


def load(engine):
    """Build + load the drafter on this rank (engine start, every rank at the same point). Returns BoxDrafter or
    None; every rank agrees (a failed rank disables the drafter on all ranks: no collective can then diverge)."""
    from sglang.srt.distributed import get_tp_group
    from split_nv.engine import gather_counts

    s = settings()
    model = engine.mr.model
    dev = model.lm_head.weight.device
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    D, err = None, None
    try:
        d = make_draft(model.config, model.quant_config, dev)
        info = load_draft(d, dev)
        torch.cuda.synchronize()
        loaded = torch.cuda.memory_allocated() - a0
        D = BoxDrafter(model, d, engine.tp_rank, s["slots"], s["head"], s["markov"], s["moe"], free_bf16_head=s["free_bf16_head"],
                       stream=s["stream"])
        torch.cuda.synchronize()
        D.load_info = {"checkpoint_params": len(info["snap"]["params"]), "derived": len(info["derived"] or {}),
                       "weights_GiB": round(loaded / 2**30, 3),
                       "resident_GiB": round((torch.cuda.memory_allocated() - a0) / 2**30, 3),
                       "load_s": round(time.perf_counter() - t0, 1), **s}
    except Exception as e:  # noqa: BLE001
        import traceback
        err = f"{type(e).__name__}: {e}"
        log(f"rank {engine.tp_rank}: drafter load FAILED: {err}\n{traceback.format_exc()}")
        D = None
    oks = gather_counts(get_tp_group(), [int(D is not None)])
    if not all(o[0] for o in oks):
        if D is not None:
            log(f"rank {engine.tp_rank}: another rank failed to load the drafter; disabled on every rank")
        D = None
        torch.cuda.empty_cache()
        return None
    m = engine.mem_summary() if hasattr(engine, "mem_summary") else {}
    g = 1 << 30
    log(f"rank {engine.tp_rank}: drafter loaded {json.dumps(D.load_info)}; allocated "
        f"{m.get('allocated_bytes.all.current', 0) / g:.2f} GiB, reserved {m.get('reserved_bytes.all.current', 0) / g:.2f} GiB, "
        f"device free {m.get('device_free', 0) / g:.2f} GiB")
    return D
