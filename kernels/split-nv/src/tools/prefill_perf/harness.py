"""TP2 prefill harness for split-chunk all-reduce overlap: the split-nv Engine's own chunk path (engine._extend /
_extend_split, hooks, consistent, og-moe) on a mini model built from the production encoder config with only the
layers [L0, L1) instantiated (real checkpoint weights; routed experts cut to the view's NE, vocab cut to 16384 rows),
both GPUs, <= 5 GB per GPU.

  harness.py check      byte identity: chunk as 1 x 8192 vs 2 x 4096 sequential chunks vs overlapped halves, over two
                        consecutive chunks (the second attends to the first); layer outputs, Engram hash ids, the
                        rank-0 captured state (SWA ring, compressed KV / index rows, compressor tail)
  harness.py time REP   chunk wall time (GPU seconds per 8K chunk) for full / seq2 / overlap, the instantiated layers
                        repeated REP times per forward (timing only)
Env: VIEW (default /pf/view-L2), LAYERS (default 2,3), NTOK (tokens of the prompt used, default 16384).
"""
import hashlib
import json
import os
import sys
import time

import torch
import torch.multiprocessing as mp

VIEW = os.environ.get("VIEW", "/pf/view-L2")
L0, L1 = (int(x) for x in os.environ.get("LAYERS", "2,3").split(","))
CFG = json.load(open(f"{VIEW}/config.json"))["text_config"]
NE, VOCAB = CFG["n_routed_experts"], CFG["vocab_size"]
ENV = dict(SGLANG_SM120_FLASHMLA_BACKEND="flashinfer", SGLANG_FLASHINFER_MOE_FUSED_FINALIZE="0",
           SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE="0", SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB=os.environ.get("IDX_BUDGET", "128"),
           SGLANG_OPT_USE_TOPK_V2="1", SPLIT_NV_HOOKS="1", SPLIT_NV_CONFIG=f"{VIEW}/config.json",
           SPLIT_NV_DIR="/pf/shm", SPLIT_NV_MAX_TOKENS="65536", SPLIT_NV_TRIM="1", SPLIT_NV_B12X="1",
           SPLIT_NV_OG_MOE="1", SPLIT_NV_PF_OVERLAP="0")
MEM_GB = float(os.environ.get("MEM_GB", "3.6"))  # torch allocator cap per GPU (context + NCCL come on top)


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def patch_model():
    import sglang.srt.distributed as D
    from sglang.srt.models import deepseek_v4 as M

    D.get_pp_indices = lambda n, r, s: (L0, L1)
    orig = M.DeepseekV4ForCausalLM.load_weights

    def filtered(weights):
        for name, w in weights:
            if name in ("embed.weight", "head.weight"):
                w = w[:VOCAB]
            elif ".ffn.experts." in name:
                if int(name.split(".ffn.experts.")[1].split(".")[0]) >= NE:
                    continue
            elif name.endswith((".ffn.gate.weight", ".ffn.gate.bias")):
                w = w[:NE]
            yield name, w

    M.DeepseekV4ForCausalLM.load_weights = lambda self, weights, is_nextn=False: orig(self, filtered(weights), is_nextn)


class Recorder:
    """Per 128-row block digests of every layer output (hidden [n, 4, H] + ffn pre-mix) and of the Engram hash ids."""

    def __init__(self):
        self.on = False
        self.d = {}
        self.host = None  # host timestamps of layer entries (time mode)
        self.vals = {}

    def add(self, key, pos, t):
        if not self.on:
            return
        t = t.contiguous()
        p0 = int(pos[0])
        if os.environ.get("NANCENSUS") == "1" and t.is_floating_point() and torch.cuda.current_device() == 0:
            bad = torch.cat([(~torch.isfinite(t[i:i + 256])).reshape(min(256, t.shape[0] - i), -1).any(1)
                             for i in range(0, t.shape[0], 256)])
            nb = int(bad.sum())
            if nb:
                idx = bad.nonzero().flatten()
                print(f"  NONFINITE {key} rows from {p0}: {nb} rows ({p0 + int(idx[0])}..{p0 + int(idx[-1])})", flush=True)
            else:
                print(f"  finite {key} rows {p0}..{p0 + t.shape[0] - 1}", flush=True)
        assert p0 % 128 == 0 and t.shape[0] % 128 == 0, (p0, t.shape)
        b = t.view(torch.uint8).reshape(t.shape[0], -1).cpu().numpy()
        for i in range(0, t.shape[0], 128):
            self.d[(key, p0 + i)] = hashlib.sha1(b[i:i + 128].tobytes()).hexdigest()
        if os.environ.get("RECHECK") == "1":  # read again after a whole-device sync: were bytes still changing?
            torch.cuda.synchronize()
            b2 = t.view(torch.uint8).reshape(t.shape[0], -1).cpu().numpy()
            if not (b == b2).all():
                rows = sorted({int(r) for r in (b != b2).any(axis=1).nonzero()[0]})
                print(f"  LATE WRITE {key} rows {p0 + rows[0]}..{p0 + rows[-1]} ({len(rows)} rows) "
                      f"thread {__import__('threading').current_thread().name}", flush=True)
                for i in range(0, t.shape[0], 128):
                    self.d[(key, p0 + i)] = hashlib.sha1(b2[i:i + 128].tobytes()).hexdigest()

    def install(self):
        from sglang.srt.layers.engram import EngramHasher
        from sglang.srt.models import deepseek_v4 as M

        rec = self
        orig_layer = M.DeepseekV4DecoderLayer.forward_hc_pre_from_prev

        def layer_forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global, prev_pre):
            if rec.on and os.environ.get("PREVAL") == "1":  # raw values of the pre-mix: as produced / as consumed
                if prev_pre is not None:
                    rec.vals[("in", self.layer_id, int(positions[0]))] = prev_pre.float().cpu().clone()
            if rec.host is not None:
                rec.host.append((time.perf_counter(), positions.shape[0]))
            out = orig_layer(self, positions, hidden_states, input_ids, forward_batch, input_ids_global, prev_pre)
            if os.environ.get("MEMTRACE") == "1" and torch.cuda.current_device() == 0:
                print(f"  layer pass rows {positions.shape[0]} allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
                if os.environ.get("MEMSNAP") == "1" and REPEATED[0]:
                    REPEATED[0] = False
                    memsnap()
            rec.add(f"L{self.layer_id}.h", positions, out[0])
            rec.add(f"L{self.layer_id}.pre", positions, out[1])
            if rec.on and os.environ.get("PREVAL") == "1":
                rec.vals[("out", self.layer_id, int(positions[0]))] = out[1].float().cpu().clone()
            return out

        M.DeepseekV4DecoderLayer.forward_hc_pre_from_prev = layer_forward
        orig_hash = EngramHasher.forward

        def hash_forward(self, input_ids, forward_batch, commit=True):
            ids = orig_hash(self, input_ids, forward_batch, commit)
            rec.add("hash", forward_batch.positions, ids)
            return ids

        EngramHasher.forward = hash_forward
        if os.environ.get("DBG") == "1":  # MoE input / partial / reduced output digests
            from sglang.srt.models import deepseek_v2 as D2
            from split_nv.og_moe import install as OI
            orig_moe_fwd = D2.DeepseekV2MoE.forward
            orig_moe = OI.moe
            cur = {}

            def moe_forward(self, hidden_states, forward_batch=None, *a, **kw):
                cur["pos"] = forward_batch.positions
                rec.add("moe.in", forward_batch.positions, hidden_states)
                out = orig_moe_fwd(self, hidden_states, forward_batch, *a, **kw)
                rec.add("moe.out", forward_batch.positions, out)
                return out

            def moe(x, ids, w, lw, valid=None):
                rec.add("moe.ids", cur["pos"], ids)
                out = orig_moe(x, ids, w, lw, valid)
                rec.add("moe.part", cur["pos"], out)
                return out

            D2.DeepseekV2MoE.forward = moe_forward
            OI.moe = moe


def persistent_moe_workspace():
    """Harness only: one og-moe prefill workspace (8192 rows) reused by every call, so the torch allocator never has to
    release and re-map 668 MB under the harness's 3.6 GB cap (production has the headroom; that stall does not exist
    there). MoE calls run one after another on one stream, halves included, so sharing it is safe."""
    import split_nv.og_moe as OG
    ws = {}

    def prefill(x, ids, w, lw):
        n = OG.ext().prefill_workspace_bytes(8192)
        buf = ws.get(x.device)
        if buf is None:
            buf = ws[x.device] = torch.empty(n, dtype=torch.uint8, device=x.device)
        assert OG.ext().prefill_workspace_bytes(x.shape[0]) <= n
        return OG.ext().prefill(x, ids, w, buf, *lw.args)

    OG.prefill = prefill


def install_diag(rec):
    if True:
        if os.environ.get("DIAG") == "1":  # layer-2 attention all-reduce of rows [12288, 16384): input and result
            from sglang.srt.distributed.parallel_state import GroupCoordinator
            from sglang.srt.models import deepseek_v4 as M
            orig_attn = M.MQALayer.forward
            rec.diag = {}
            rec.diag_on = [None]

            def attn_forward(self, x, positions, forward_batch, *a, **kw):
                p0 = int(positions[0]) if positions.numel() else -1
                n = positions.shape[0]
                want = rec.on and self.layer_id == 2 and p0 <= 12288 < p0 + n
                rec.diag_on[0] = (12288 - p0, n) if want else None
                try:
                    return orig_attn(self, x, positions, forward_batch, *a, **kw)
                finally:
                    rec.diag_on[0] = None

            M.MQALayer.forward = attn_forward
            inner = GroupCoordinator._all_reduce_in_place

            from split_nv import ce_allreduce as CE
            import sys as _sys
            mods = [CE] + ([_sys.modules["ce_push"]] if "ce_push" in _sys.modules else [])

            def ar(self, input_):
                d = rec.diag_on[0]
                if d is None or input_.dim() != 2:
                    return inner(self, input_)
                off, n = d
                rows = slice(off, off + 4096)
                snaps = {}
                for m in mods:
                    m.SNAP = lambda tag, t: snaps.__setitem__(tag, t)
                try:
                    r = inner(self, input_)
                finally:
                    for m in mods:
                        m.SNAP = None
                if snaps:  # copies taken on the comm stream: what the CE read / wrote
                    rec.diag["pre"], rec.diag["post"] = snaps["in"][rows], snaps["out"][rows]
                return r

            GroupCoordinator._all_reduce_in_place = ar


def make_engine(rank, server_args, port_args):
    from split_nv import engine as EM
    from split_nv.og_moe import install as OI
    if os.environ.get("MOE_WS", "1") == "1":
        persistent_moe_workspace()

    orig_lw = OI._layer_weights
    OI._layer_weights = lambda mlp, lid, r, ck, n_experts=NE: orig_lw(mlp, lid, r, ck, NE)

    class Mini(EM.Engine):
        """Engine.__init__ without the prefix-cache store and the step runner (not used here)."""

        def __init__(self, server_args, port_args, gpu_id, tp_rank):
            from sglang.benchmark.one_batch import load_model
            from sglang.srt.layers.moe import initialize_moe_config
            from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
            from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
            from sglang.srt.mem_cache import deepseek_v4_memory_pool as P
            from sglang.srt.mem_cache.cache_init_params import CacheInitParams
            from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
            from sglang.srt.runtime_context import get_schedule, publish

            self.tp_rank = tp_rank
            publish(server_args, role="scheduler")
            initialize_moe_config()
            initialize_fp8_gemm_config()
            initialize_fp4_gemm_config()
            orig_ring = P.get_compress_state_ring_size
            P.get_compress_state_ring_size = (
                lambda ratio, is_speculative=False, num_draft_tokens=0: 16 if ratio == 2 else orig_ring(ratio, is_speculative, num_draft_tokens))
            runner, _ = load_model(server_args, port_args, gpu_id, tp_rank)
            self.mr = runner.torch_runner
            self.page_size = get_schedule().page_size
            self.tree_cache = SWAChunkCache(CacheInitParams(
                disable=True, req_to_token_pool=self.mr.req_to_token_pool,
                token_to_kv_pool_allocator=self.mr.token_to_kv_pool_allocator, page_size=self.page_size,
                chunked_prefill_size=EM.CHUNK, sliding_window_size=int(getattr(self.mr.model_config.hf_text_config, "sliding_window", 128))))
            self.sessions = {}
            self.pending_capture = {}
            self.vision_spans = {}
            from split_nv import hooks as H
            self.cap = H.CAP
            self.cap.auto = False
            from split_nv.consistent import install
            install()
            from split_nv.og_moe.install import install as og_moe_install
            og_moe_install(self)
            from split_nv.engram_prefetch import install as engram_prefetch_install
            engram_prefetch_install(self)
            from split_nv import ce_allreduce, pf_overlap
            if os.environ.get("CE_IMPL") == "push":  # the first (push) protocol, for comparison
                sys.path.insert(0, "/work/tools/prefill_perf")
                import ce_push
                ce_allreduce.CEAllReduce = ce_push.CEAllReduce
            if ce_allreduce.enabled():
                from sglang.srt.distributed import get_tp_group
                ce_allreduce.setup(get_tp_group(), EM.CHUNK * self.mr.model_config.hf_text_config.hidden_size)
            pf_overlap.install()
            from split_nv import q_nocopy
            if q_nocopy.enabled():
                q_nocopy.install()
            self.window = self.tree_cache.sliding_window_size
            self.steps = None
            self.store = None
            self.vision = False

    return Mini(server_args, port_args, rank, rank)


def cap_state(cap):
    """Digest of the rank-0 captured prompt state (source rows are compared per chunk)."""
    out = {}
    for L, (buf, pos) in cap.swa.items():
        out[f"swa.{L}"] = hashlib.sha1(buf.cpu().numpy().tobytes() + pos.cpu().numpy().tobytes()).hexdigest()
    for L, t in cap.tail.items():
        out[f"tail.{L}"] = hashlib.sha1(b"".join(x.cpu().numpy().tobytes() for x in t)).hexdigest()
    return out


def set_ar(name):
    """plan name suffixes: _ce = copy-engine all-reduce (else NCCL), _q = q written in place (q_nocopy)"""
    from split_nv import ce_allreduce, q_nocopy
    ce_allreduce.ACTIVE = "_ce" in name
    q_nocopy.ACTIVE = name.endswith("_q") or os.environ.get("ALLQ") == "1"


def run_prompt(E, sid, ids, plan):
    """plan: [(rows, split)] consecutive chunks. Returns (chunk seconds, captured rows digests, capture state)."""
    E.cmd_prefill_begin(sid)
    a, dts, rows = 0, [], {}
    for n, split in plan:
        if split == "seq":  # the same rows as two consecutive plain chunks
            h = n // 2
            r1, d1 = E.cmd_prefill_chunk(sid, ids[a:a + h])
            r2, d2 = E.cmd_prefill_chunk(sid, ids[a + h:a + n])
            dt, parts = d1 + d2, [r1, r2]
        else:
            r, dt = E.cmd_prefill_chunk(sid, ids[a:a + n], None, split)
            parts = [r]
        for r in parts:
            for L, (ckv, idxk) in (r or {}).items():
                rows.setdefault(L, []).append((ckv, idxk))
        dts.append(dt)
        a += n
    dig = {L: hashlib.sha1(torch.cat([c for c, _ in v]).numpy().tobytes() + torch.cat([i for _, i in v]).numpy().tobytes()).hexdigest()
           for L, v in rows.items()}
    st = cap_state(E.cap) if E.cap.enabled else {}
    E.cmd_close(sid)
    return dts, dig, st


def worker(rank, mode, rep, port_args, server_args):
    torch.cuda.set_device(rank)
    if os.environ.get("HANGDUMP"):
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["HANGDUMP"]), repeat=True)
    if os.environ.get("MEMSNAP") == "1":
        torch.cuda.memory._record_memory_history(max_entries=200000)
    torch.cuda.set_per_process_memory_fraction(MEM_GB * 2**30 / torch.cuda.get_device_properties(rank).total_memory, rank)
    patch_model()
    rec = Recorder()
    rec.install()
    E = make_engine(rank, server_args, port_args)
    install_diag(rec)  # outermost wrapper of the all-reduce (after pf_overlap's)
    from sglang.srt.distributed import get_tp_group
    torch.cuda.synchronize()
    log(rank, f"model ready: layers [{L0},{L1}) NE {NE}; attn backend {type(E.mr.attn_backend).__name__}; "
              f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB, max_total_num_tokens {E.mr.max_total_num_tokens}")
    if os.environ.get("MEMSNAP") == "1" and rank == 0:
        memsnap()
    worker_rest(rank, mode, rep, E, rec)


REPEATED = [False]


def memsnap():
    if True:
        blocks = []
        for seg in torch.cuda.memory_snapshot():
            for b in seg["blocks"]:
                if b["state"] == "active_allocated":
                    fr = b.get("frames") or []
                    where = " <- ".join(f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in fr if f['filename'].endswith('.py') and 'torch/' not in f['filename'])
                    blocks.append((b["size"], where))
        blocks.sort(reverse=True)
        for sz, where in blocks[:25]:
            print(f"  {sz / 2**20:8.1f} MiB  {where[:300]}", flush=True)


def worker_rest(rank, mode, rep, E, rec):
    from sglang.srt.distributed import get_tp_group
    ntok = int(os.environ.get("NTOK", "16384"))
    ids = [i % VOCAB for i in json.load(open("/ref/ids-131072.json"))[:ntok]]
    sid = 1
    # warm-up (JIT for every shape used below)
    C = int(os.environ.get("CH", "8192"))
    for name, plan in (("w", [(C, 0), (C, 0)]), ("w", [(C, "seq")]), ("w", [(C, C // 2)]),
                       ("w_ce", [(C, 0)]), ("w_ce", [(C, C // 2)])):
        set_ar(name)
        run_prompt(E, sid, ids, plan)
        sid += 1
    torch.cuda.synchronize()
    log(rank, f"warm-up done; reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB max {torch.cuda.max_memory_reserved() / 2**30:.2f}")
    from split_nv import pf_overlap
    if mode == "stress":  # the reference once, then the overlapped CE path many times (race hunting)
        n = int(os.environ.get("STRESS_N", "20"))
        sp = os.environ.get("STRESS_PLAN", "ovl_ce_q")
        runs = [(os.environ.get("STRESS_REF", "full"), [(8192, 0), (8192, 0)])] + [(sp, [(8192, 0 if sp.startswith("full") else 4096)] * 2)] * n
        ref, bad = None, 0
        for i, (name, plan) in enumerate(runs):
            set_ar(name)
            rec.on, rec.d = True, {}
            run_prompt(E, sid, ids, plan)
            sid += 1
            rec.on = False
            if ref is None:
                ref_diag = dict(getattr(rec, "diag", {}))
                rec.diag = {}
                ref = dict(rec.d)
                refv = dict(rec.vals)
                rec.vals = {}
                continue
            if os.environ.get("DIAG") == "1" and ref_diag:
                if i == 1:
                    refd = {k: v.cpu() for k, v in ref_diag.items()}
                cur = {k: v.cpu() for k, v in rec.diag.items()}
                if i == 1 and rank == 0:
                    torch.save({"ref": refd, "cur": cur}, "/pf/diag-run1.pt")
                for k in ("pre", "post"):
                    if k in cur and not torch.equal(cur[k].view(torch.int16), refd[k].view(torch.int16)):
                        rows = (cur[k] != refd[k]).any(1).nonzero().flatten()
                        print(f"[rank {rank}] run {i} DIAG {k}-allreduce layer-2 attention rows 12288+[{int(rows[0])}..{int(rows[-1])}] "
                              f"({len(rows)} rows differ)", flush=True)
                rec.diag = {}
            vals, rec.vals = rec.vals, {}
            for key, v in vals.items():
                # stress rows come in halves; compare by absolute position against the reference chunk tensors
                kind, lid, p0 = key
                rk = [k for k in refv if k[0] == kind and k[1] == lid and k[2] <= p0 < k[2] + refv[k].shape[0]]
                if not rk:
                    continue
                r = refv[rk[0]][p0 - rk[0][2]:p0 - rk[0][2] + v.shape[0]]
                if not torch.equal(r, v):
                    bad_rows = (r != v).flatten(1).any(1).nonzero().flatten()
                    i0 = int(bad_rows[0])
                    print(f"[rank {rank}] run {i} {kind} L{lid} rows {p0 + i0}..{p0 + int(bad_rows[-1])} ({len(bad_rows)} rows): "
                          f"ref {r[i0].tolist()} got {v[i0].tolist()}", flush=True)
            from split_nv import pf_overlap as PO
            if PO._VERIFY is not None:
                for me, j, shape, nb, lo, hi in PO._VERIFY:
                    if int(nb):
                        print(f"[rank {rank}] run {i} VERIFY half {me} AR #{j} shape {shape}: {int(nb)} rows differ "
                              f"from NCCL, rows {int(lo)}..{int(hi)}", flush=True)
                PO._VERIFY.clear()
            mism = sorted(k for k in set(ref) | set(rec.d) if ref.get(k) != rec.d.get(k))
            bad += bool(mism)
            if mism:
                print(f"[rank {rank}] stress run {i}: {len(mism)} mismatching blocks {mism[:4]}", flush=True)
        print(f"[rank {rank}] stress {sp} {os.environ.get('CE_IMPL', 'pull')}: {n} prompts, {bad} with a mismatch "
              f"-> {'PASS' if bad == 0 else 'FAIL'}", flush=True)
    elif mode == "check":
        res = {}
        plans = {"full": [(8192, 0), (8192, 0)], "seq2": [(8192, "seq"), (8192, "seq")],
                 "ovl": [(8192, 4096), (8192, 4096)], "ovl_odd": [(8192, 2944), (8192, 4096)],
                 "full_ce": [(8192, 0), (8192, 0)], "ovl_ce": [(8192, 4096), (8192, 4096)],
                 "ovl_odd_ce": [(8192, 2944), (8192, 4096)], "full_q": [(8192, 0), (8192, 0)],
                 "ovl_ce_q": [(8192, 4096), (8192, 4096)]}
        plans = {k: v for k, v in plans.items() if k in os.environ.get("PLANS", ",".join(plans)).split(",")}
        for name, plan in plans.items():
            set_ar(name)
            rec.on, rec.d = True, {}
            dts, rows, st = run_prompt(E, sid, ids, plan)
            sid += 1
            rec.on = False
            res[name] = (dict(rec.d), rows, st, dts)
            log(rank, f"{name}: chunk seconds {[round(x, 4) for x in dts]}; {len(rec.d)} row-block digests")
        ref = res["full"]
        ok_all = True
        for name in plans:
            if name == "full":
                continue
            d, rows, st, _ = res[name]
            bad = sorted(k for k in set(ref[0]) | set(d) if ref[0].get(k) != d.get(k))
            ok = not bad and rows == ref[1] and st == ref[2]
            ok_all &= ok
            kinds = {}
            for k in bad:
                kinds[k[0]] = kinds.get(k[0], 0) + 1
            print(f"[rank {rank}] {name} vs full: mismatches by kind {kinds}; first {bad[:4]}", flush=True)
            print(f"[rank {rank}] {name} vs full: blocks {len(d)}/{len(ref[0])} mismatching {len(bad)} {bad[:6]}; "
                  f"captured rows equal {rows == ref[1]} ({sorted(rows)}); capture state equal {st == ref[2]} -> "
                  f"{'BYTE-IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
        print(f"[rank {rank}] check {'PASS' if ok_all else 'FAIL'}", flush=True)
    else:
        E.cap.enabled = False  # layers repeated: the capture's per-layer row bookkeeping does not apply
        model = E.mr.model.model
        base = [model.layers[i] for i in range(L0, L1)]
        seqr = base * rep
        if os.environ.get("PATTERN"):  # e.g. production-like source/non-source mix: 2,3,3,3,3,3,2,3,...
            seqr = [model.layers[int(i)] for i in os.environ["PATTERN"].split(",")]
        first = L0 if L0 + len(seqr) <= len(model.layers) else 0
        for j, lay in enumerate(seqr):
            model.layers[first + j] = lay
        model.start_layer, model.end_layer = first, first + len(seqr)
        REPEATED[0] = True
        log(rank, f"timing with {len(seqr)} layer passes per forward: {[l.layer_id for l in seqr]}")
        plans = {"full": [(C, 0), (C, 0)], "seq2": [(C, "seq"), (C, "seq")], "ovl": [(C, C // 2), (C, C // 2)],
                 "full_ce": [(C, 0), (C, 0)], "ovl_ce": [(C, C // 2), (C, C // 2)], "full_q": [(C, 0), (C, 0)],
                 "ovl_q": [(C, C // 2), (C, C // 2)], "ovl_ce_q": [(C, C // 2), (C, C // 2)]}
        plans = {k: v for k, v in plans.items() if k in os.environ.get("PLANS", ",".join(plans)).split(",")}
        for name, plan in plans.items():
            set_ar(name)
            run_prompt(E, sid, ids, plan)
            sid += 1
        ts = {k: [] for k in plans}
        for r in range(int(os.environ.get("REPS", "5"))):
            for name, plan in plans.items():
                set_ar(name)
                torch.cuda.nvtx.range_push(f"plan:{name}:{r}")
                dts, _, _ = run_prompt(E, sid, ids, plan)
                torch.cuda.nvtx.range_pop()
                sid += 1
                ts[name].append(dts)
        import statistics as S
        if os.environ.get("HOSTT") == "1":  # host enqueue time per layer pass (no GPU sync inside a forward)
            for name, plan in plans.items():
                set_ar(name)
                rec.host = []
                if os.environ.get("SYNCDBG") == "1" and name == "full":
                    torch.cuda.set_sync_debug_mode("warn")
                run_prompt(E, sid, ids, plan)
                torch.cuda.set_sync_debug_mode(0)
                sid += 1
                h = rec.host
                rec.host = None
                d = [(h[i + 1][0] - h[i][0]) * 1e3 for i in range(len(h) - 1)]
                log(rank, f"host {name:7s}: layer entries {len(h)}; median interval {S.median(d):6.2f} ms; "
                          f"intervals (first 12) {[round(x, 2) for x in d[:12]]}")
        for name in plans:
            c0 = S.median(x[0] for x in ts[name]) * 1e3
            c1 = S.median(x[1] for x in ts[name]) * 1e3
            log(rank, f"{name:5s}: chunk@0 {c0:7.1f} ms  chunk@{C} {c1:7.1f} ms  per layer pass {c1 / len(seqr):6.2f} ms")
        f0 = S.median(x[1] for x in ts["full"]) * 1e3
        for name in [p for p in plans if p != "full"]:
            c1 = S.median(x[1] for x in ts[name]) * 1e3
            log(rank, f"{name} - full @{C}: {c1 - f0:+7.1f} ms per chunk ({(c1 - f0) / len(seqr):+.3f} ms per layer pass)")
    log(rank, f"max reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    get_tp_group().barrier()


def main():
    os.environ.update(ENV)
    mode = sys.argv[1]
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    import argparse
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(parser.parse_args([
        "--model-path", VIEW, "--trust-remote-code", "--tp", "2", "--mem-fraction-static", "0.94",
        "--context-length", os.environ.get("CTX", "131072"), "--max-total-tokens", os.environ.get("MAX_TOTAL", "20480"), "--max-running-requests", "1",
        "--chunked-prefill-size", "8192", "--enable-deepseek-v4-fp4-indexer", "--fp8-gemm-backend", "flashinfer_cutlass",
        "--disable-cuda-graph", "--disable-radix-cache"]
        # with very few routed experts (NE=4) FlashInfer's MoE autotune warm-up hangs; that MoE is not used (og-moe)
        + (["--disable-flashinfer-autotune"] if os.environ.get("AUTOTUNE", "1") == "0" else [])))
    server_args.enable_multimodal = False
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    mp.spawn(worker, args=(mode, rep, port_args, server_args), nprocs=2)


if __name__ == "__main__":
    main()
