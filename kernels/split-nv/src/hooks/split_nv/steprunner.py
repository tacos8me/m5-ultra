"""Per-step forward of layers 0-20 for L new rows of one session, as a TARGET_VERIFY batch (eager or CUDA graph).

Built on the DSV4 backend's speculative-verify support: fixed bs=1, row width W, positions [N, N+W), KV slots
pre-written in req_to_token. Padded rows (>= L) get out_cache_loc 0 so their KV/ring writes are suppressed.
"""

import os
import time
from types import SimpleNamespace

import torch

WIDTHS = (1, 2, 3, 4, 5, 6, 8)
WINDOW = 128


class StepRunner:
    def __init__(self, engine, widths=WIDTHS):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        self.engine = engine
        self.mr = mr = engine.mr
        self.be = be = mr.attn_backend
        self.dev = mr.device
        self.widths = tuple(sorted(widths))
        self.max_w = self.widths[-1]
        R = mr.req_to_token_pool.req_to_token.shape[0]
        self.slots = {}
        for W in self.widths:
            # DSV4 derives verify geometry from the backend width, not the
            # DFlashVerifyInput alone. Graph tables otherwise collide at bs=1.
            be.speculative_num_draft_tokens = W
            be.init_cuda_graph_state(1, W)
            backend_state = dict(
                speculative_num_draft_tokens=W,
                extend_seq_lens_buffer=torch.full((R,), W, dtype=torch.int32, device=self.dev),
                extend_start_loc_buffer=torch.zeros(R, dtype=torch.int32, device=self.dev),
                cuda_graph_metadata_of_bucket_and_bs=be.cuda_graph_metadata_of_bucket_and_bs,
                draft_extend_num_tokens_per_req=W,
                _verify_mask=be._verify_mask,
                cuda_graph_swa_out_cache_loc=be.cuda_graph_swa_out_cache_loc,
            )
            ids = torch.ones(W, dtype=torch.int64, device=self.dev)
            pos = torch.arange(W, dtype=torch.int64, device=self.dev)
            ocl = torch.zeros(W, dtype=torch.int64, device=self.dev)
            rpi = torch.zeros(1, dtype=torch.int64, device=self.dev)
            sl = torch.ones(1, dtype=torch.int64, device=self.dev)
            spec = DFlashVerifyInput(draft_token=ids, positions=pos, draft_token_num=W, custom_mask=None,
                                     capture_hidden_mode=CaptureHiddenMode.NULL)
            fb = ForwardBatch(forward_mode=ForwardMode.TARGET_VERIFY, batch_size=1, input_ids=ids, req_pool_indices=rpi,
                              seq_lens=sl, seq_lens_cpu=None, orig_seq_lens=sl, out_cache_loc=ocl, seq_lens_sum=1,
                              return_logprob=False, positions=pos, spec_algorithm=mr.spec_algorithm, spec_info=spec,
                              capture_hidden_mode=CaptureHiddenMode.NULL, global_forward_mode=ForwardMode.TARGET_VERIFY)
            self.slots[W] = SimpleNamespace(W=W, ids=ids, pos=pos, ocl=ocl, rpi=rpi, sl=sl, spec=spec, fb=fb,
                                            graph=None, out=None, backend_state=backend_state)
        self.mode = ForwardMode.TARGET_VERIFY
        self.max_pos = int(engine.mr.model_config.context_len)
        self.host_out = None
        # Self-tests and numerics diagnostics read the output tensors; serving only needs the wire payload.
        self.full_outputs = bool(os.environ.get("SPLIT_NV_SELFTEST") or os.environ.get("SPLIT_NV_TRACE"))
        self.og_valid = None
        if os.environ.get("SPLIT_NV_OG_MOE") == "1":
            from split_nv import og_moe
            self.og_valid = og_moe.valid_rows(self.dev)

    def activate(self, slot):
        for name, value in slot.backend_state.items():
            setattr(self.be, name, value)

    # ---- session KV bookkeeping (identical on every rank) ---------------------------------------------------
    def ensure_alloc(self, sess, upto):
        mr, req = self.mr, sess.req
        if sess.alloc_len >= upto:
            return
        row = req.kv.req_pool_idx
        r2t = mr.req_to_token_pool.req_to_token
        a = sess.alloc_len
        prefix_cpu = torch.tensor([a], dtype=torch.int64)
        seq_cpu = torch.tensor([upto], dtype=torch.int64)
        locs = mr.token_to_kv_pool_allocator.alloc_extend(
            prefix_cpu.to(self.dev), prefix_cpu, seq_cpu.to(self.dev), seq_cpu, r2t[row, a - 1:a], upto - a)
        if locs is None:
            raise RuntimeError("KV pool exhausted")
        r2t[row, a:upto] = locs
        sess.alloc_len = upto
        req.kv.kv_allocated_len = upto

    def evict_swa(self, sess, keep):
        from sglang.srt.mem_cache.common import free_swa_out_of_window_slots

        free_swa_out_of_window_slots(sess.req, keep, sliding_window_size=WINDOW, page_size=self.engine.page_size,
                                     req_to_token_pool=self.mr.req_to_token_pool,
                                     token_to_kv_pool_allocator=self.mr.token_to_kv_pool_allocator, is_chunk_cache=True)

    def commit_pending(self, sess, keep):
        """Lazy Engram history commit: rows of the previous step accepted up to `keep`."""
        if sess.pending is None:
            if keep != sess.length:
                raise ValueError(f"cannot rewind committed prefix {sess.length} to {keep}")
            return
        base, ids = sess.pending
        if keep < base:
            raise ValueError(f"keep {keep} < previous step base {base}")
        sess.pending = None
        a = min(keep - base, len(ids))
        if a > 0:
            hasher = self.mr.model.model.engram_hasher
            ids_2d = torch.tensor([ids], dtype=torch.int64, device=self.dev)
            hasher.commit_after_verify(ids_2d, torch.tensor([sess.req.kv.req_pool_idx], dtype=torch.int64, device=self.dev),
                                       torch.tensor([a], dtype=torch.int64, device=self.dev))

    # ---- forward -----------------------------------------------------------------------------------------------
    def _load(self, slot, N, ids, locs, row):
        W, L = slot.W, len(ids)
        slot.ids[:L].copy_(torch.tensor(ids, dtype=torch.int64), non_blocking=True)
        if L < W:
            slot.ids[L:].fill_(1)
        # Padded rows (>= L) only need a valid position: clamp them inside the model's context (their KV and
        # ring writes are suppressed; rows are causal and row-independent, so real rows cannot see them).
        slot.pos.copy_(torch.clamp(torch.arange(N, N + W, dtype=torch.int64), max=self.max_pos - 1), non_blocking=True)
        slot.ocl[:L].copy_(locs[:L])
        if L < W:
            slot.ocl[L:].zero_()
        slot.rpi.fill_(row)
        slot.sl.fill_(N)
        slot.fb.seq_lens_sum = N
        if self.og_valid is not None:
            self.og_valid.fill_(L)  # og-moe: rows >= L are padding (their MoE output is zero; nothing reads them)

    def _dp(self, W):
        from sglang.srt.layers.dp_attention import set_dp_buffer_len, set_is_extend_in_batch

        set_dp_buffer_len(None, W, False, None)
        set_is_extend_in_batch(False)

    @torch.no_grad()
    def run_eager(self, slot):
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

        self.activate(slot)
        with forward_context(ForwardContext(attn_backend=self.be)):
            self.be.init_forward_metadata(slot.fb)
            self._dp(slot.W)
            self.mr.model.forward(slot.ids, slot.pos, slot.fb)

    @torch.no_grad()
    def capture(self, W):
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.parallel_state import graph_capture
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
        from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
        from sglang.srt.model_executor.runner_utils.pool import (
            get_global_graph_memory_pool, get_or_create_global_graph_capture_stream, graph_pool_capture_scope)

        slot = self.slots[W]
        self.activate(slot)
        cap = self.engine.cap
        slot.ids.fill_(1)
        slot.pos.copy_(torch.arange(W, dtype=torch.int64))
        slot.ocl.zero_()
        slot.rpi.zero_()
        slot.sl.fill_(1)
        slot.fb.seq_lens_sum = 1
        stream = get_or_create_global_graph_capture_stream()
        model = self.mr.model
        if hasattr(model, "engram_setup_decode_pregather"):
            model.engram_setup_decode_pregather(self.max_w, self.mode)
        with model_capture_mode(), graph_capture(stream=stream), forward_context(ForwardContext(attn_backend=self.be)):
            self.be.init_forward_metadata_out_graph(slot.fb, in_capture=True)

            def run_once():
                cap.step = {}
                self.be.init_forward_metadata_in_graph(slot.fb)
                self._dp(W)
                model.forward(slot.ids, slot.pos, slot.fb)
                return cap.step

            for _ in range(2):
                torch.cuda.synchronize()
                get_tp_group().barrier()
                run_once()
                self.be.on_after_cuda_graph_warmup()
            graph = torch.cuda.CUDAGraph()
            with graph_pool_capture_scope(), torch.cuda.graph(graph, pool=get_global_graph_memory_pool(), stream=stream):
                out = run_once()
        cap.step = None
        torch.cuda.synchronize()
        slot.graph, slot.out = graph, out

    @torch.no_grad()
    def replay(self, slot):
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

        self.activate(slot)
        view = SimpleNamespace(batch_size=1, forward_mode=self.mode, actual_forward_mode=self.mode, req_pool_indices=slot.rpi,
                               seq_lens=slot.sl, seq_lens_cpu=None, seq_lens_sum=slot.fb.seq_lens_sum, out_cache_loc=slot.ocl,
                               positions=slot.pos, spec_info=slot.spec, input_ids=slot.ids, forward_metadata_ready=False)
        model = self.mr.model
        with forward_context(ForwardContext(attn_backend=self.be)):
            self.be.init_forward_metadata_out_graph(view)
            if hasattr(model, "engram_fill_decode_pregather"):
                model.engram_fill_decode_pregather(view)
            slot.graph.replay()

    def pick(self, L):
        # A single row selects GEMV specializations with different rounding.
        # Keep the verified multi-row arithmetic; slot 0 masks the extra row's
        # KV/ring writes and only actual ids enter the Engram commit history.
        L = max(2, L)
        for W in self.widths:
            if W >= L:
                return self.slots[W]
        raise ValueError(f"L={L} exceeds max width {self.max_w}")

    def run(self, sess, keep, ids, use_graph=True):
        """Rows [keep, keep+L). Returns (capture dict for rank 0, seconds)."""
        L = len(ids)
        slot = self.pick(L)
        W = slot.W
        self.commit_pending(sess, keep)
        sess.length = keep
        self.evict_swa(sess, keep)
        if keep + L > self.max_pos:
            raise ValueError(f"step rows {keep}..{keep + L} exceed the context length {self.max_pos}")
        self.ensure_alloc(sess, min(keep + W, self.mr.req_to_token_pool.req_to_token.shape[1]))
        row = sess.req.kv.req_pool_idx
        locs = self.mr.req_to_token_pool.req_to_token[row, keep:keep + W]
        self._load(slot, keep, ids, locs, row)
        cap = self.engine.cap
        t0 = time.perf_counter()
        if use_graph and slot.graph is not None:
            self.replay(slot)
            out = slot.out
        else:
            cap.step = {}
            try:
                self.run_eager(slot)
                out = cap.step
            finally:
                cap.step = None
        if self.og_valid is not None:
            self.og_valid.fill_(1 << 30)  # prefill and other callers: every row is valid
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sess.length = keep + L
        sess.pending = (keep, list(ids))
        if not cap.enabled:
            return {}, dt
        if self.full_outputs:
            # Host copies: graph outputs alias graph-owned memory that the next replay overwrites.
            return {k: (v[:L].cpu() if torch.is_tensor(v) and v.shape[0] == W else v) for k, v in out.items()}, dt
        # The STPR payload (h19 BF16 | pre F32 | ckv20 U8 | idxk20 U8, row-major) in one pinned buffer, one sync.
        parts = [out["h"], out["pre"], out["ckv"], out["idxk"]]
        if any(t.shape[0] < L for t in parts):
            raise RuntimeError(f"step capture has {[tuple(t.shape) for t in parts]} for L={L}")
        if self.host_out is None:
            self.host_out = torch.empty(self.max_w * 41332, dtype=torch.uint8, pin_memory=True)
        off = 0
        for t in parts:
            flat = t[:L].contiguous().view(torch.uint8).reshape(-1)
            self.host_out[off:off + flat.numel()].copy_(flat, non_blocking=True)
            off += flat.numel()
        torch.cuda.current_stream().synchronize()
        return {"payload": self.host_out[:off].numpy().tobytes(), "L": L}, dt
