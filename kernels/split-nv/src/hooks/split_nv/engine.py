"""split-nv phase-3 engine: stateful DS-V4.1 encoder sessions driven directly on SGLang's ModelRunner (TP2, no Scheduler).

Rank 0 owns the sockets (split_nv.front): the TCP step API (default :10052, see ~/split-nv/STEP_API.md) and HTTP
(:10051 public, :10050 local: POST /v1/prefill, GET /health). Rank 1 replays every GPU command rank 0 sends over a
pipe, so both ranks build identical batches. SIGTERM drains (rank 0 refuses new OPENs, waits for open sessions up to
SPLIT_NV_DRAIN_S) and exits; a wedged GPU job trips the watchdog (exit 3) so the supervisor restarts the engine.

Launch inside the container: python3 -m split_nv.engine <ServerArgs CLI flags as for launch_server>
"""

import json
import multiprocessing
from multiprocessing.connection import wait
import os
import signal
import sys
import threading
import time
import traceback
from array import array

import torch

CHUNK = 8192


def pf_split(n):
    """Rows of the first half when a prefill chunk of n rows runs split with overlapped all-reduces (0 = unsplit):
    SPLIT_NV_PF_OVERLAP=1 (default off), overridable at runtime by the box-perf flag pf_overlap (read on rank 0 only;
    the decision travels in the command)."""
    from split_nv import pf_overlap
    from split_nv.perf_flags import flag

    return pf_overlap.split_rows(n, bool(flag("pf_overlap", pf_overlap.default_enabled())))

# Per-prompt fields of the capture state (hooks.Capture.reset); swapped per session when prefills interleave.
CAP_FIELDS = ("tokens", "ntok", "swa", "rows", "tail", "h_ring", "pending", "chunks", "t_start", "first_chunk_wall",
              "chunk_rows", "collect")
MEM_KEYS = ("allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current",
            "reserved_bytes.all.peak", "active_bytes.all.peak", "requested_bytes.all.peak",
            "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak", "reserved_bytes.large_pool.current",
            "reserved_bytes.small_pool.current", "segment.all.current", "num_alloc_retries", "num_ooms",
            "num_device_alloc", "num_device_free")

log_lock = threading.Lock()


def log(*a):
    with log_lock:
        print("[engine]", *a, flush=True)


# --------------------------------------------------------------------------------------------- GPU side (all ranks)
class Session:
    def __init__(self, sid):
        self.sid = sid
        self.req = None
        self.length = 0
        self.alloc_len = 0
        self.pending = None  # (base, ids) of the last step, committed to the Engram history at the next step
        self.batch = None


class Engine:
    def __init__(self, server_args, port_args, gpu_id, tp_rank):
        from sglang.benchmark.one_batch import load_model
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.srt.mem_cache import deepseek_v4_memory_pool as P
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
        from sglang.srt.runtime_context import get_schedule, publish
        from sglang.srt.utils import configure_logger

        self.tp_rank = tp_rank
        publish(server_args, role="scheduler")
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        configure_logger(server_args, prefix=f" TP{tp_rank}")
        # Ratio-2 pair ring: a rolled-back position's partner must survive up to MAX_STEP_ROWS rejected rows.
        orig_ring = P.get_compress_state_ring_size
        P.get_compress_state_ring_size = (
            lambda ratio, is_speculative=False, num_draft_tokens=0: 16 if ratio == 2 else orig_ring(ratio, is_speculative, num_draft_tokens)
        )
        runner, _ = load_model(server_args, port_args, gpu_id, tp_rank)
        self.mr = runner.torch_runner
        self.page_size = get_schedule().page_size
        self.tree_cache = SWAChunkCache(CacheInitParams(
            disable=True, req_to_token_pool=self.mr.req_to_token_pool,
            token_to_kv_pool_allocator=self.mr.token_to_kv_pool_allocator, page_size=self.page_size,
            chunked_prefill_size=CHUNK, sliding_window_size=int(getattr(self.mr.model_config.hf_text_config, "sliding_window", 128))))
        self.sessions = {}
        self.pending_capture = {}  # sid -> rank-0 capture state exported at prefill end, for a snapshot
        self.vision_spans = {}  # sid -> [(start, rows [len, hidden] bf16)] of images still to be prefilled
        from split_nv import hooks as H

        self.cap = H.CAP
        self.cap.auto = False
        from split_nv import topk_det
        topk_det.install()
        topk_det.install_audit()
        if os.environ.get("SPLIT_NV_SELFTEST") != "numerics":
            from split_nv.consistent import install
            install()
        if os.environ.get("SPLIT_NV_OG_MOE") == "1":
            from split_nv.og_moe.install import install as og_moe_install
            og_moe_install(self)
        from split_nv.engram_prefetch import install as engram_prefetch_install
        engram_prefetch_install(self)
        from split_nv import ce_allreduce, pf_overlap
        if ce_allreduce.enabled():
            # copy-engine all-reduce for the large prefill all-reduces (collective setup: every rank, same point)
            from sglang.srt.distributed import get_tp_group
            ce_allreduce.setup(get_tp_group(), CHUNK * self.mr.model_config.hf_text_config.hidden_size)
        pf_overlap.install()
        from split_nv import q_nocopy
        if q_nocopy.enabled():
            q_nocopy.install()
        self.window = self.tree_cache.sliding_window_size
        from split_nv.steprunner import StepRunner

        self.steps = StepRunner(self)
        from split_nv.prefix_cache import RankStore
        from split_nv.state_pack import NUMERICS

        self.store = RankStore(self, NUMERICS)
        # DSpark drafter (SPLIT_NV_DSPARK=1; default off: nothing loaded or captured, today's engine)
        self.dspark = None
        from split_nv import dspark_box
        if dspark_box.enabled():
            self.dspark = dspark_box.load(self)
        self.vision = getattr(self.mr.model, "vision", None) is not None
        self.image_token_id = int(getattr(self.mr.model_config.hf_config, "image_token_id", 129264))
        self.use_graph = False
        # fairness / memory prototypes (all default off)
        self.cap_swap = int(os.environ.get("SPLIT_NV_BYPASS_TOKENS", "0") or 0) > 0
        self.cap_owner = None
        self.cap_saved = {}
        self.preempt_peers = None  # rank 0: connections to the other ranks; others: the connection from rank 0
        self.preempt_take = None  # rank 0: the front end's parked-step source
        self.last_preempt = None  # (points, inline steps, step seconds) of the last preemptible chunk
        self.last_window = None  # its preempt.Window (timings for split_nv.front's parked-STEP log)
        # per-chunk maxima over preemptible chunks (ms): open -> first point, point -> point host time, wait for the GPU
        # LAG points back, last point -> chunk done (rows copied, grid blocks written)
        self.preempt_timing = {"head_max_ms": 0.0, "point_gap_max_ms": 0.0, "lag_wait_max_ms": 0.0, "tail_max_ms": 0.0}
        self.memlog = os.environ.get("SPLIT_NV_MEMLOG") == "1"
        self.mem_t = {}
        self.job_failures = 0  # commands this rank failed (rank 0: Front._execute; others: rank_main), see cmd_barrier
        if os.environ.get("SPLIT_NV_MEM_FRACTION"):
            # cap the caching allocator (cached blocks are released and the allocation retried before this is exceeded)
            torch.cuda.set_per_process_memory_fraction(float(os.environ["SPLIT_NV_MEM_FRACTION"]))
        if int(os.environ.get("SPLIT_NV_MEMHIST", "0") or 0) > 0:
            torch.cuda.memory._record_memory_history(max_entries=int(os.environ["SPLIT_NV_MEMHIST"]))
        # Order: idx_lowmem replaces sglang's dense_indexer_topk, idx_rowsplit wraps whatever is installed then
        # (rowsplit(lowmem)): each rank runs the low-memory tiling on its half of the rows.
        if os.environ.get("SPLIT_NV_IDX_LOWMEM") == "1":
            from split_nv import idx_lowmem
            idx_lowmem.install()
        elif os.environ.get("SPLIT_NV_IDX_FIXED_TILES") == "1":
            log(f"rank {tp_rank}: SPLIT_NV_IDX_FIXED_TILES=1 ignored: it is part of SPLIT_NV_IDX_LOWMEM=1 (off)")
        from split_nv import idx_rowsplit
        if idx_rowsplit.enabled():
            idx_rowsplit.install()
        if os.environ.get("SPLIT_NV_TRACE"):
            from split_nv.numerics import configure
            configure(self, 'trace')
        log(f"rank {tp_rank}: model ready, max_total_num_tokens={self.mr.max_total_num_tokens}, window={self.window}, page={self.page_size}")

    # ---- batch construction ---------------------------------------------------------------------------------------
    def _new_req(self, sid):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(rid=f"s{sid}", origin_input_text="", origin_input_ids=array("q", []),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        return req

    @torch.no_grad()
    def _extend(self, sess, new_ids, forward=True, replace=None):
        fb = self._prepare_extend(sess, new_ids, forward, replace)
        if fb is not None:
            self.mr.forward(fb)

    @torch.no_grad()
    def _extend_split(self, sess, new_ids, split, replace_fn):
        """One chunk as two halves [0, split) and [split, n) with overlapped all-reduces (split_nv.pf_overlap);
        the same bytes as _extend(sess, new_ids). replace_fn(start, n) -> image rows of [start, start + n)."""
        from split_nv import pf_overlap

        if not split:
            return self._extend(sess, new_ids, replace=replace_fn(sess.length, len(new_ids)))
        a, b = new_ids[:split], new_ids[split:]
        fb_a = self._prepare_extend(sess, a, True, replace_fn(sess.length, len(a)))
        # B is prepared before A has run: it must not evict A's out-of-window SWA slots (A has not written them yet).
        # Nothing is evicted mid-chunk, exactly like the unsplit chunk: the next chunk's prepare evicts them, so SWA
        # slot assignment -- and with it every page a prefix-cache entry gathers -- is the same as unsplit.
        pf_overlap.run_split(self.mr, fb_a, lambda: self._prepare_extend(
            sess, b, True, replace_fn(sess.length, len(b)), evict_swa=False), len(new_ids))

    def _prepare_extend(self, sess, new_ids, forward=True, replace=None, evict_swa=True):
        """Allocate the chunk's KV slots and build its ForwardBatch (None when forward is False). evict_swa=False
        skips the out-of-window SWA eviction prepare_for_extend does first (the caller runs it later)."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        req, mr = sess.req, self.mr
        start = sess.length
        req.origin_input_ids.extend(new_ids)
        if start:
            # write_cache_indices consumes int64 prefix pointers, as does
            # ChunkCache.cache_unfinished_req. The pool itself is int32.
            req.prefix_indices = mr.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :start].to(torch.int64, copy=True)
        else:
            req.prefix_indices = torch.empty((0,), dtype=torch.int64, device=mr.device)
        req.set_extend_range(start, start + len(new_ids))
        batch = ScheduleBatch.init_new(reqs=[req], req_to_token_pool=mr.req_to_token_pool,
                                       token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator, tree_cache=self.tree_cache,
                                       model_config=mr.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
        if not evict_swa:
            batch.maybe_evict_swa = lambda: None
        batch.prepare_for_extend()
        if not evict_swa:
            del batch.maybe_evict_swa
        fb = None
        if forward:
            mr.ngram_embedding_manager.prepare_for_forward(batch, chunked_req=None)
            if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
                batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
                batch.prefill_input_ids_cpu = None
            fb = ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)
            if replace is not None:
                # Image rows: the model embeds the chunk's ids, then these rows overwrite their positions
                # (model_runner's replace_embeds path). Chunks without image rows run exactly as before.
                fb.replace_positions, fb.replace_embeds = replace
        sess.length = start + len(new_ids)
        sess.batch = batch
        return fb

    # ---- per-session capture state (SPLIT_NV_BYPASS_TOKENS: prefills interleave at chunk boundaries) -----------
    def _cap_use(self, sid):
        """Make the global capture state the one of `sid`'s prefill: park the current owner's per-prompt fields (they
        are rebound, never mutated across prompts, by Capture.reset) and bring back sid's, if it has any."""
        if not self.cap_swap or self.cap_owner == sid:
            return
        if self.cap_owner is not None and self.cap_owner in self.sessions:
            self.cap_saved[self.cap_owner] = {f: getattr(self.cap, f) for f in CAP_FIELDS}
        st = self.cap_saved.pop(sid, None)
        if st is not None:
            for f, v in st.items():
                setattr(self.cap, f, v)
        self.cap_owner = sid

    def _cap_done(self, sid):
        self.cap_saved.pop(sid, None)
        if self.cap_owner == sid:
            self.cap_owner = None

    # ---- memory ---------------------------------------------------------------------------------------------------
    def mem_summary(self):
        st = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        out = {k: st.get(k, 0) for k in MEM_KEYS}
        out.update(rank=self.tp_rank, device_free=free, device_total=total,
                   non_torch=total - free - st.get("reserved_bytes.all.current", 0))
        return out

    def cmd_memstats(self, path=None, reset_peak=False, snapshot=None, segments=True):
        """Read-only allocator statistics of this rank (torch.cuda.memory_stats + a segment summary), written to
        <path>.rank<r>.json; rank 0 also returns them. snapshot: dump the allocation history (SPLIT_NV_MEMHIST)."""
        out = self.mem_summary()
        if segments:
            segs = torch.cuda.memory_snapshot()
            by_pool, free_blocks, big = {}, [], []
            for sg in segs:
                pool = str(tuple(sg.get("segment_pool_id", (0, 0))))
                p = by_pool.setdefault(pool, {"segments": 0, "total": 0, "allocated": 0})
                p["segments"] += 1
                p["total"] += sg["total_size"]
                p["allocated"] += sg["allocated_size"]
                for b in sg["blocks"]:
                    if b["state"] == "inactive":
                        free_blocks.append(b["size"])
                big.append((sg["total_size"], sg["allocated_size"], sg.get("stream", 0), pool))
            free_blocks.sort(reverse=True)
            big.sort(reverse=True)
            out["pools"] = by_pool
            out["free_in_segments"] = sum(free_blocks)
            out["largest_free_blocks"] = free_blocks[:16]
            out["largest_segments"] = big[:24]
            hist = {}
            for size, alloc, _, _ in big:
                key = "<20M" if size < 20 << 20 else "<256M" if size < 256 << 20 else "<1G" if size < 1 << 30 else ">=1G"
                h = hist.setdefault(key, [0, 0, 0])
                h[0] += 1
                h[1] += size
                h[2] += alloc
            out["segment_hist"] = hist  # bucket -> [count, reserved, allocated]
        if snapshot:
            try:
                torch.cuda.memory._dump_snapshot(f"{snapshot}.rank{self.tp_rank}.pickle")
                out["snapshot"] = f"{snapshot}.rank{self.tp_rank}.pickle"
            except Exception as e:  # noqa: BLE001
                out["snapshot_error"] = str(e)
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        if path:
            with open(f"{path}.rank{self.tp_rank}.json", "w") as f:
                json.dump(out, f)
        return out

    def cmd_trim(self):
        """Return the caching allocator's free cached segments to the device (SPLIT_NV_TRIM_MIN_TOKENS). Byte-neutral:
        only which addresses later allocations get changes (every block stays 512-byte aligned)."""
        before = torch.cuda.memory_reserved()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return before - torch.cuda.memory_reserved()

    def _mem_begin(self, sid):
        if self.memlog:
            torch.cuda.reset_peak_memory_stats()
            self.mem_t[sid] = time.perf_counter()

    def _mem_end(self, sid, sess):
        if self.memlog and sid in self.mem_t:
            self.mem_t.pop(sid)
            m = self.mem_summary()
            g = 1 << 30
            log(f"rank {self.tp_rank} mem after prefill {sid} ({sess.length} tokens): alloc {m['allocated_bytes.all.current'] / g:.2f} "
                f"peak {m['allocated_bytes.all.peak'] / g:.2f} GiB, reserved {m['reserved_bytes.all.current'] / g:.2f} "
                f"peak {m['reserved_bytes.all.peak'] / g:.2f} GiB, split-free {m['inactive_split_bytes.all.current'] / g:.2f} GiB, "
                f"non-torch {m['non_torch'] / g:.2f} GiB, device free {m['device_free'] / g:.2f} GiB, "
                f"retries {m['num_alloc_retries']} ooms {m['num_ooms']}")

    # ---- commands (executed identically on every rank) --------------------------------------------------------
    def cmd_prefill(self, sid, ids, dump=True):
        from split_nv import pf_overlap

        sess = Session(sid)
        sess.req = self._new_req(sid)
        self.sessions[sid] = sess
        self._cap_use(sid)
        self.cap.reset()
        self.cap.collect = not dump
        trace = os.environ.get("SPLIT_NV_TRACE") and len(ids) == 8213
        if trace:
            self.cap.trace_start = len(ids) - 8
            self.cap.trace = {}
        self.cap.t_start = time.perf_counter()
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            t0 = time.perf_counter()
            self.cap.tokens.append(torch.tensor(chunk, dtype=torch.int64))
            self.cap.ntok += len(chunk)
            # every rank runs this command on its own: only the static env default may decide the split here
            self._extend_split(sess, chunk, pf_overlap.split_rows(len(chunk), pf_overlap.default_enabled()),
                               lambda a, n: None)
            torch.cuda.synchronize()
            self.cap.chunks.append((len(chunk), time.perf_counter() - t0))
            if self.cap.collect:
                self.cap.chunk_rows = {L: [] for L in self.cap.chunk_rows}
        sess.alloc_len = sess.req.kv.kv_allocated_len
        if dump and self.cap.enabled:
            self.cap.pending = True
            self.cap.dump()
        if trace:
            if self.cap.enabled:
                from split_nv.numerics import snapshot
                torch.save(snapshot(self.cap), '/dev/shm/split-nv/cuda-trace-8213.pt')
            self.cap.trace = None
        return sess.length

    # Chunked prefill as separate GPU jobs so queued steps of other sessions run between chunks.
    # The front end serializes prefills, so the global capture state belongs to one prompt at a time.
    def cmd_prefill_begin(self, sid):
        sess = Session(sid)
        sess.req = self._new_req(sid)
        self.sessions[sid] = sess
        self._cap_use(sid)
        self._mem_begin(sid)
        self.cap.reset()
        self.cap.collect = True
        self.cap.t_start = time.perf_counter()

    @torch.no_grad()
    def cmd_vision(self, sid, images):
        """images: [(start, vit_h, vit_w, bf16 patch bytes [h*w, 3*14*14])] -> the span rows of each image, as the
        reference merges them: ViT + aligner rows at IMAGE slots, learned image_start/newline/end elsewhere.
        (Bytes, not tensors: rank 1 gets commands over a plain pipe.)"""
        from split_nv.imagekeys import DOWNSAMPLE
        from sglang.srt.multimodal.deepseek_v41_image_processing import image_token_types

        model = self.mr.model
        dev, dtype = model.image_start.device, model.image_start.dtype
        t0 = time.perf_counter()
        spans = []
        for start, h, w, data in images:
            patches = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
            x = patches.to(dev).view(h * w, 3, 14, 14).to(dtype)
            features = model.aligner(model.vision(x, h, w), h, w)
            types = image_token_types(-(-h // DOWNSAMPLE), -(-w // DOWNSAMPLE)).to(dev)
            rows = torch.empty((len(types), model.config.hidden_size), device=dev, dtype=dtype)
            rows[types == 0] = model.image_start.to(dtype)
            rows[types == 1] = features.to(dtype)
            rows[types == 2] = model.image_newline.to(dtype)
            rows[types == 3] = model.image_end.to(dtype)
            spans.append((start, rows))
        torch.cuda.synchronize()
        self.vision_spans[sid] = spans
        return time.perf_counter() - t0

    def _replace_rows(self, sid, a, n):
        """(chunk-relative positions, rows) of image rows inside positions [a, a + n), or None."""
        pos, rows = [], []
        for start, span in self.vision_spans.get(sid, ()):
            lo, hi = max(a, start), min(a + n, start + span.shape[0])
            if lo < hi:
                pos.append(torch.arange(lo - a, hi - a, dtype=torch.int64, device=span.device))
                rows.append(span[lo - start:hi - start])
        if not pos:
            return None
        return torch.cat(pos), torch.cat(rows)

    def cmd_prefill_chunk(self, sid, chunk, grid=None, split=0, preemptible=False, rowsplit=False):
        """Returns (rank 0) the packed source rows this chunk produced and its GPU seconds.
        grid = (blocks, entry key or None): save the 8K grid blocks [(bid, end)] this chunk completed, and a grid
        entry at the chunk end (which must then lie on the grid). split > 0: run the chunk as halves [0, split) and
        [split, n) with overlapped all-reduces (decided by rank 0, so every rank issues the same collectives).
        preemptible: parked STEPs run between its layers (split_nv.preempt; decided by rank 0).
        rowsplit: the prefill indexer's rows are split across the ranks (split_nv.idx_rowsplit; decided by rank 0)."""
        from split_nv import idx_rowsplit

        t0 = time.perf_counter()
        sess = self.sessions[sid]
        self._cap_use(sid)
        self.cap.tokens.append(torch.tensor(chunk, dtype=torch.int64))
        self.cap.ntok += len(chunk)
        with idx_rowsplit.chunk(rowsplit):
            if preemptible:
                from split_nv import preempt
                win = preempt.open_window(self, self.tp_rank == 0, self.preempt_peers, self.preempt_take)
                try:
                    self._extend_split(sess, chunk, split, lambda a, n: self._replace_rows(sid, a, n))
                finally:
                    preempt.close_window()
                self.last_preempt = (win.k, win.steps, win.step_s)
                self.last_window = win
            else:
                self._extend_split(sess, chunk, split, lambda a, n: self._replace_rows(sid, a, n))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.cap.chunks.append((len(chunk), dt))
        rows = self.cap.take_chunk_rows() if self.cap.enabled else None
        if grid is not None:
            blocks, key = grid
            for bid, end in blocks:
                if end % 8192 or end > sess.length:
                    raise RuntimeError(f"grid block ending at {end} (session at {sess.length})")
                self.store.write_block(sess, bid, end)
            if key is not None:
                if sess.length % 8192:
                    raise RuntimeError(f"grid entry at {sess.length}")
                self.store.write_entry(sess, key)
        if preemptible:
            pt = self.preempt_timing
            for name, v in (("head_max_ms", win.head_s), ("point_gap_max_ms", win.max_gap_s),
                            ("lag_wait_max_ms", win.max_lag_s),
                            ("tail_max_ms", time.perf_counter() - (win.t_point or win.t0))):
                if v is not None:
                    pt[name] = max(pt[name], round(v * 1e3, 1))
        return rows, dt

    def cmd_prefill_end(self, sid, keep_capture=False):
        """Returns (rank 0) the final state parts (window rings, ratio-2 tails, hidden tail)."""
        sess = self.sessions[sid]
        self._cap_use(sid)
        sess.alloc_len = sess.req.kv.kv_allocated_len
        self.vision_spans.pop(sid, None)
        # Free the SWA slots of the last chunk that fell out of the window now instead of at the first step
        # (identical range), so a prefilled session holds ~window SWA slots, not a whole chunk.
        self.steps.evict_swa(sess, sess.length)
        self._mem_end(sid, sess)
        try:
            if not self.cap.enabled:
                return None
            if keep_capture:
                self.pending_capture[sid] = self.cap.export_state()
            return self.cap.final_parts()
        finally:
            self._cap_done(sid)

    def cmd_restore(self, sid, key, blocks, P):
        sess = Session(sid)
        sess.req = self._new_req(sid)
        self.sessions[sid] = sess
        self._cap_use(sid)
        self._mem_begin(sid)
        self.cap.reset()
        t0 = time.perf_counter()
        self.store.restore(sess, key, blocks, P, CHUNK)
        self.steps.evict_swa(sess, sess.length)
        self.cap.collect = True
        self.cap.t_start = time.perf_counter()
        return sess.length, time.perf_counter() - t0

    def cmd_snapshot(self, sid, key):
        t0 = time.perf_counter()
        self.store.write_entry(self.sessions[sid], key, self.pending_capture.pop(sid, None) if self.cap.enabled else None)
        self.store.flush()
        return time.perf_counter() - t0

    def cmd_barrier(self):
        """Returns on rank 0 only once rank 1 has executed every earlier command (then its files are read/written):
        [[snapshot write errors, failed commands] of every rank] since start. The collective always runs: a rank that
        raised before it would leave the others blocked in it until the watchdog (300 s) restarts the engine."""
        from sglang.srt.distributed import get_tp_group

        try:
            self.store.flush()
        except Exception as e:  # noqa: BLE001
            self.store.n_errors += 1
            log(f"rank {self.tp_rank}: snapshot flush failed: {e!r}")
        return gather_counts(get_tp_group(), [self.store.n_errors, self.job_failures])

    def cmd_step(self, sid, keep, ids, use_graph=None):
        sess = self.sessions[sid]
        if keep > sess.length:
            raise ValueError(f"keep {keep} > session length {sess.length}")
        return self.steps.run(sess, keep, ids, use_graph=self.use_graph if use_graph is None else use_graph)

    # ---- DSpark on the box (STEPD-SPEC.md; decisions are made on rank 0 and travel in the commands) -------------
    def cmd_dspark_capture(self):
        from sglang.srt.distributed import get_tp_group
        from split_nv import dspark_box

        t0 = time.perf_counter()
        a0 = torch.cuda.memory_reserved()
        res = self.dspark.capture(dspark_box.settings()["widths"])
        log(f"rank {self.tp_rank}: dspark graphs captured in {time.perf_counter() - t0:.1f}s ({json.dumps(res)}), "
            f"reserved +{(torch.cuda.memory_reserved() - a0) / 2**30:.3f} GiB")
        if not all(ok for (ok,) in gather_counts(get_tp_group(), [int(not res["failed"])])):
            # graph != eager or ranks disagree on drafts on some rank: never draft (every rank drops it alike)
            log(f"rank {self.tp_rank}: dspark capture checks failed on a rank; drafter disabled")
            self.dspark = None
            res["disabled"] = True
        if self.memlog:
            m = self.mem_summary()
            g = 1 << 30
            log(f"rank {self.tp_rank} mem after dspark capture: alloc {m['allocated_bytes.all.current'] / g:.2f} GiB, "
                f"reserved {m['reserved_bytes.all.current'] / g:.2f} GiB, device free {m['device_free'] / g:.2f} GiB")
        return res

    def cmd_ring(self, sid, slot, offset, keys, kr, taps, tr):
        """RING (every rank): the session's ring slot becomes exactly positions [offset - kr - tr, offset)."""
        if sid not in self.sessions:
            raise ValueError(f"ring: no session {sid}")
        t0 = time.perf_counter()
        self.dspark.ring_set(slot, offset, keys, kr, taps, tr)
        return time.perf_counter() - t0

    def cmd_stepd(self, sid, slot, keep, anchor, app_base, n_app, taps, draft, W, dmax, costs, explicit, expect):
        """One STEPD job (every rank): [append the committed rows] -> [draft W at keep] -> step [anchor] + drafts.
        draft: the front end's decision (mode 0, drafting on, ring current, keep >= MIN_CTX, dmax >= 1); costs: the
        c2..c5 tier for (keep, regime). Rank-local width choice: every rank holds the same all-gathered drafts and
        max-probs, and choose_cost_depth is plain host arithmetic on them. expect = (base, nver) of the pending step
        the front end checked the accept against (None for a kickoff): a disagreement is a bug, never a draft."""
        from split_nv import dspark_wire as WIRE

        sess = self.sessions[sid]
        if expect is not None and (sess.pending is None or sess.pending[0] != expect[0] or len(sess.pending[1]) != expect[1]):
            raise RuntimeError(f"stepd: pending {None if sess.pending is None else (sess.pending[0], len(sess.pending[1]))} "
                               f"!= front end's {expect}")
        if keep > sess.length:
            raise ValueError(f"keep {keep} > session length {sess.length}")
        D = self.dspark
        t0 = time.perf_counter()
        info = {"drafted": False, "toks": [], "probs": [], "drafter_ms": 0.0}
        if draft:
            D.launch_draft(slot, keep, anchor, app_base, n_app, taps, W)
            # host bookkeeping of the step while the drafter graph runs (run() then finds it done)
            self.steps.prepare(sess, keep, keep + self.steps.pick(1 + dmax).W)
            toks, probs = D.collect(W)
            depth = min(WIRE.choose_cost_depth(probs, costs), dmax)
            ids = [anchor] + toks[:depth]
            info.update(drafted=True, toks=toks, probs=probs, drafter_ms=(time.perf_counter() - t0) * 1e3)
        else:
            if n_app:
                D.run_append(slot, app_base, n_app, taps)
            ids = [anchor] + list(explicit)
        out, _ = self.steps.run(sess, keep, ids, use_graph=self.use_graph)
        info["ids"] = ids
        return out, time.perf_counter() - t0, info

    def cmd_capture(self, widths):
        for W in widths:
            t0 = time.perf_counter()
            self.steps.capture(W)
            log(f"rank {self.tp_rank}: captured verify graph W={W} in {time.perf_counter() - t0:.1f}s")
        self.use_graph = True

    def cmd_close(self, sid):
        self._cap_done(sid)
        self.mem_t.pop(sid, None)
        self.pending_capture.pop(sid, None)
        self.vision_spans.pop(sid, None)
        sess = self.sessions.pop(sid, None)
        if sess is not None and sess.req is not None and sess.req.kv.req_pool_idx is not None:
            req = sess.req
            self.tree_cache.cache_finished_req(req, kv_len_to_handle=max(sess.alloc_len, req.kv.kv_allocated_len))
            self.mr.req_to_token_pool.free(req)
        if not self.sessions:
            self.audit_idle()

    def audit_idle(self):
        """With no session alive every slot must be free. An aborted allocation (e.g. a prefill that ran out of
        KV mid-way) can leave pages behind or push the padding page 0 into a free list; reset to pristine."""
        a, r2t = self.mr.token_to_kv_pool_allocator, self.mr.req_to_token_pool
        full, swa = a.full_attn_allocator.free_pages, a.swa_attn_allocator.free_pages
        state = (a.full_available_size(), a.swa_available_size(), len(r2t.free_slots))
        want = (a.size_full, a.size_swa, r2t._alloc_size - 1)
        pad = bool((full == 0).any()) or bool((swa == 0).any())
        if state != want or pad:
            log(f"rank {self.tp_rank}: idle pool audit: free (full, swa, rows) {state} != {want} or padding page free={pad}; resetting")
            a.clear()
            r2t.free_slots = list(range(1, r2t._alloc_size))

    def cmd_capacity(self):
        a, r2t = self.mr.token_to_kv_pool_allocator, self.mr.req_to_token_pool
        return {"rows": len(r2t.free_slots), "full": a.full_available_size(), "swa": a.swa_available_size(),
                "sessions": len(self.sessions)}

    def execute(self, cmd):
        kind = cmd[0]
        if kind == "prefill":
            return self.cmd_prefill(cmd[1], cmd[2], cmd[3] if len(cmd) > 3 else True)
        if kind == "prefill_begin":
            return self.cmd_prefill_begin(cmd[1])
        if kind == "prefill_chunk":
            return self.cmd_prefill_chunk(cmd[1], cmd[2], cmd[3] if len(cmd) > 3 else None, cmd[4] if len(cmd) > 4 else 0,
                                          bool(cmd[5]) if len(cmd) > 5 else False, bool(cmd[6]) if len(cmd) > 6 else False)
        if kind == "prefill_end":
            return self.cmd_prefill_end(cmd[1], cmd[2])
        if kind == "restore":
            return self.cmd_restore(cmd[1], cmd[2], cmd[3], cmd[4])
        if kind == "snapshot":
            return self.cmd_snapshot(cmd[1], cmd[2])
        if kind == "barrier":
            return self.cmd_barrier()
        if kind == "capacity":
            return self.cmd_capacity()
        if kind == "vision":
            return self.cmd_vision(cmd[1], cmd[2])
        if kind == "exec_vision_rows":  # start-up vision check only: take the rows of a pseudo session
            spans = self.vision_spans.pop(cmd[1], [])
            return spans[0][1].cpu() if spans and self.tp_rank == 0 else None
        if kind == "step":
            return self.cmd_step(cmd[1], cmd[2], cmd[3], cmd[4] if len(cmd) > 4 else None)
        if kind == "capture":
            return self.cmd_capture(cmd[1])
        if kind == "stepd":
            return self.cmd_stepd(*cmd[1:])
        if kind == "ring":
            return self.cmd_ring(*cmd[1:])
        if kind == "dspark_capture":
            return self.cmd_dspark_capture()
        if kind == "close":
            return self.cmd_close(cmd[1])
        if kind == "sync":
            torch.cuda.synchronize()
            return None
        if kind == "memstats":
            return self.cmd_memstats(*cmd[1:])
        if kind == "trim":
            return self.cmd_trim()
        if kind == "gc_freeze":
            from split_nv import stallwatch
            return stallwatch.freeze()
        if kind == "exec" and os.environ.get("SPLIT_NV_DEV") == "1":
            # Localhost-only numerics development hook: the same code runs on every rank.
            ns = self.__dict__.setdefault("_dev_ns", {"engine": self, "torch": torch, "os": os, "json": json})
            ns.pop("result", None)
            exec(cmd[1], ns)
            return ns.get("result")
        if kind == "diagnostic" and os.environ.get("SPLIT_NV_SELFTEST") == "numerics":
            from split_nv.numerics import configure
            return configure(self, *cmd[1:])
        if kind == "extend" and os.environ.get("SPLIT_NV_SELFTEST") == "numerics":
            return self._extend(self.sessions[cmd[1]], cmd[2])
        raise ValueError(kind)


def gather_counts(group, counts):
    """Every rank's int list `counts` (same length on all ranks), gathered over the TP CPU (gloo) group: the barrier."""
    t = torch.tensor(counts, dtype=torch.int64)
    out = [torch.zeros_like(t) for _ in range(group.world_size)]
    torch.distributed.all_gather(out, t, group=group.cpu_group)
    return [o.tolist() for o in out]


# --------------------------------------------------------------------------------------------- process entry
def rank_main(server_args, port_args, gpu_id, tp_rank, conns):
    cache_loader = None
    if tp_rank == 0 and os.environ.get("SPLIT_NV_CACHE_PRELOAD", "1") == "1" and not os.environ.get("SPLIT_NV_SELFTEST"):
        # rank 0's prefix index only reads files: build it while the model loads (joined by Front)
        from split_nv.prefix_cache import IndexLoader, budget_bytes
        from split_nv.state_pack import NUMERICS
        cache_loader = IndexLoader(NUMERICS, budget_bytes())
    engine = Engine(server_args, port_args, gpu_id, tp_rank)
    engine.preempt_peers = conns
    from split_nv import stallwatch
    if stallwatch.enabled():
        def ctx():
            from split_nv import preempt
            w = preempt._win
            cur = getattr(engine, "_front_current", lambda: None)()
            return (f"rank {tp_rank}, gpu job {cur[0] if cur else None}"
                    + (f", in a preemptible chunk at point {w.k}" if w is not None else ""))
        stallwatch.install(ctx)
    if tp_rank != 0:
        # Commands arrive over a local pipe from rank 0 (~20 us vs ~0.3 ms for a gloo broadcast). Within SPLIT_NV_SPIN_S
        # of a STEP the pipe is busy-polled: a blocking recv after the ~25 ms idle of a c1 cycle wakes a cold core
        # (~60 us) that then runs the step's host work ~3x slower, and rank 0 waits for this rank at the first reduction.
        from split_nv.perf_flags import flag
        spin_default = float(os.environ.get("SPLIT_NV_SPIN_S", "0.2"))
        spin_until = 0.0
        while True:
            if time.monotonic() < spin_until:
                while not conns.poll() and time.monotonic() < spin_until:
                    pass
            cmd = conns.recv()
            if cmd[0] in ("step", "stepd"):
                spin_s = float(flag("r1_spin_s", spin_default))
                spin_until = time.monotonic() + spin_s if spin_s > 0 else 0.0
            try:
                engine.execute(cmd)
            except Exception as e:  # noqa: BLE001
                engine.job_failures += 1
                log(f"rank {tp_rank} job failed: {e}\n{traceback.format_exc()}")
                from split_nv.front import fatal_cuda_error
                if fatal_cuda_error(e):
                    os._exit(4)
    from split_nv.front import Front, install_signals

    front = Front(engine, server_args, cache_loader)
    front.peers = conns
    engine._front_current = lambda: front.current
    threading.Thread(target=front.gpu_loop, daemon=True).start()
    selftest = os.environ.get("SPLIT_NV_SELFTEST")
    if selftest == "numerics":
        from split_nv.numerics import diagnose
        diagnose(front)
        os._exit(0)
    if selftest:
        # Each case: prefill P tokens, step L rows (keep=P), a rollback step; eager first, then graphs vs eager.
        cases = {"aligned256": (256, 3), "aligned256_L1": (256, 1), "unaligned8": (8, 3), "unaligned300": (300, 5), "long9000": (9000, 5)}
        widths = [int(w) for w in os.environ.get("SPLIT_NV_GRAPHS", "1,2,3,4,5,6,8").split(",") if w]
        graphs_done = False
        for name in selftest.split(","):
            P, L = cases[name]
            ids = [(i * 7919) % 100000 + 100 for i in range(P + 2 * L + 8)]
            log(f"selftest {name}: prefill {P}")
            front.submit(("prefill", 0, ids[:P], False))
            log(f"selftest {name}: eager step L={L}")
            e1, dt = front.submit(("step", 0, P, ids[P:P + L], False))
            log(f"selftest {name}: eager OK h{tuple(e1['h'].shape)} ckv{tuple(e1['ckv'].shape)} {dt * 1e3:.1f} ms")
            e2, dt = front.submit(("step", 0, P + 1, ids[P + 1:P + L], False))
            log(f"selftest {name}: eager rollback step OK {dt * 1e3:.1f} ms")
            if widths and not graphs_done:
                front.submit(("capture", widths))
                graphs_done = True
            if graphs_done:
                # A session can reject only rows of its most recent step.
                # Re-prefill so graph/eager see identical history and caches.
                front.submit(("close", 0))
                front.submit(("prefill", 0, ids[:P], False))
                g1, dt1 = front.submit(("step", 0, P, ids[P:P + L], True))
                g2, dt2 = front.submit(("step", 0, P + 1, ids[P + 1:P + L], True))
                for tag, e, g in (("step", e1, g1), ("rollback", e2, g2)):
                    for k in ("h", "pre", "ckv", "idxk"):
                        a, b = e[k].float(), g[k].float()
                        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
                        log(f"selftest {name}: graph vs eager {tag} {k}: cos {cos:.6f} max|d| {(a - b).abs().max().item():.4f}")
                log(f"selftest {name}: graph steps {dt1 * 1e3:.1f} / {dt2 * 1e3:.1f} ms")
            front.submit(("close", 0))
        log("selftest done")
        os._exit(0)
    # warm up the step path once (JIT), capture the verify graphs, then open the sockets
    warm = [1, 2, 3, 4, 5, 6, 7, 8]
    front.submit(("prefill", 0, warm, False))
    front.submit(("step", 0, 8, [9, 10, 11], False))
    front.submit(("step", 0, 9, [12], False))
    widths = [int(w) for w in os.environ.get("SPLIT_NV_GRAPHS", "1,2,3,4,5,6,8").split(",") if w]
    if widths:
        front.submit(("capture", widths))
        front.submit(("step", 0, 10, [13, 14]))
    front.submit(("close", 0))
    if engine.dspark is not None:
        # after the step graphs (same global graph pool); graph == eager and rank agreement checked inside
        front.submit(("dspark_capture",))
    # Prefill kernels JIT on their first use (~3 s): warm them before accepting traffic.
    t0 = time.perf_counter()
    for n in (8194, 300):
        front.submit(("prefill", 0, [100 + (i * 7919) % 100000 for i in range(n)], False))
        front.submit(("close", 0))
    log(f"warm-up done (prefill warm-up {time.perf_counter() - t0:.1f}s)")
    if os.environ.get("SPLIT_NV_GC_FREEZE", "0") == "1":
        front.submit(("gc_freeze",))  # every rank: warm-up objects out of the cyclic GC (split_nv.stallwatch)
    if engine.vision:
        vision_check(front)
    install_signals(front)
    threading.Thread(target=front.watchdog, daemon=True).start()
    threading.Thread(target=front.serve_http, args=("127.0.0.1", int(os.environ.get("SPLIT_NV_HTTP_PORT", "10050"))),
                     daemon=True).start()
    public = os.environ.get("SPLIT_NV_PUBLIC_HTTP", "0.0.0.0:10051")
    if public:
        host, port = public.rsplit(":", 1)
        threading.Thread(target=front.serve_http, args=(host, int(port)), daemon=True).start()
    if os.environ.get("SPLIT_NV_DEV") == "1":
        threading.Thread(target=front.serve_dev, args=(10059,), daemon=True).start()
    front.serve_tcp(os.environ.get("SPLIT_NV_STEP_HOST", "0.0.0.0"), int(os.environ.get("SPLIT_NV_STEP_PORT", "10052")))


def vision_check(front):
    """Warm the ViT path and write its span rows for ref/vision-check.safetensors (official preprocessing of the
    checkpoint's example images) to SPLIT_NV_DIR/vision-check-out.safetensors, for tools/vision_fidelity.py."""
    path = "/home/ian/split-nv/ref/vision-check.safetensors"
    if not os.path.exists(path):
        return
    from safetensors.torch import load_file, save_file

    ref = load_file(path)
    images = [(0, int(ref[f"grid.{i}"][0]), int(ref[f"grid.{i}"][1]),
               ref[f"patches.{i}"].contiguous().view(torch.int16).numpy().tobytes())
              for i in range(sum(k.startswith("patches.") for k in ref))]
    out = {}
    for i, image in enumerate(images):
        dt = front.submit(("vision", -1 - i, [image]))
        out[f"rows.{i}"] = front.submit(("exec_vision_rows", -1 - i))
        log(f"vision check image {i}: grid {image[1]}x{image[2]}, span {out[f'rows.{i}'].shape[0]} rows in {dt:.2f}s")
    try:
        os.makedirs(os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"), exist_ok=True)
        save_file(out, os.path.join(os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"), "vision-check-out.safetensors"))
    except OSError as e:
        log(f"vision check dump: {e}")


def main():
    import argparse

    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(parser.parse_args())
    # The engine merges image rows itself (cmd_vision + replace_embeds); SGLang's multimodal pipeline
    # (mm cache reservations, pad-id hashing) stays off even though the vision tower is loaded.
    server_args.enable_multimodal = False
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    procs = []
    pipes = [multiprocessing.Pipe(duplex=False) for _ in range(server_args.tp_size - 1)]
    for tp_rank in range(server_args.tp_size):
        conns = [w for _, w in pipes] if tp_rank == 0 else pipes[tp_rank - 1][0]
        with maybe_reindex_device_id(tp_rank) as gpu_id:
            p = multiprocessing.Process(target=rank_main, args=(server_args, port_args, gpu_id, tp_rank, conns))
            p.start()
            procs.append(p)
    # SIGTERM: rank 0 drains, then exits; everything is killed if that takes longer than the drain budget.
    deadline = []

    def on_term(*_):
        if not deadline:
            deadline.append(time.monotonic() + float(os.environ.get("SPLIT_NV_DRAIN_S", "30")) + 15)
            if procs[0].is_alive():
                os.kill(procs[0].pid, signal.SIGTERM)

    rank0_exiting = []

    def on_rank0_exit(*_):
        # rank 0 is about to os._exit (front.hard_exit): start the other ranks' teardown now, in parallel with its own
        rank0_exiting.append(1)
        kill_ranks(procs[1:], "rank 0 exiting")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGUSR2, on_rank0_exit)
    # A completed self-test or failed rank must not leave its peer resident.
    while not wait([p.sentinel for p in procs], timeout=1):
        if deadline and time.monotonic() > deadline[0]:
            break
    t0 = time.monotonic()
    # No rank holds state worth a graceful stop here: rank 0 has exited (drained, crashed or tripped the watchdog) or
    # overran the drain budget; the others only replay rank 0's commands, and prefix-cache entries become visible
    # only after both ranks flushed (a killed write leaves a .tmp that the next start deletes).
    # (rank 0 itself is left to finish its own os._exit when it announced it, so its exit code survives)
    kill_ranks(procs[1:] if rank0_exiting else procs,
               "a rank exited" if not deadline or time.monotonic() <= deadline[0] else "drain budget exceeded")
    for i, p in enumerate(procs):
        p.join()
        log(f"rank {i} reaped {time.monotonic() - t0:.1f}s after the first exit (exit code {p.exitcode})")
    sys.exit(exit_code([p.exitcode for p in procs]))


def kill_ranks(procs, why):
    alive = [p for p in procs if p.exitcode is None]
    if alive:
        log(f"{why}: SIGKILL to {len(alive)} rank process(es)")
    for p in alive:
        try:
            p.kill()
        except (OSError, AttributeError):
            pass


def exit_code(codes):
    """The unit's exit status: rank 0's own code (0 drain, 1 admin crash, 3 watchdog, 4 fatal CUDA error), else the
    first failing rank's; a rank this process killed (negative) counts as a failure only if nothing exited cleanly."""
    if codes[0] is not None and codes[0] >= 0:
        return codes[0]
    return next((c for c in codes[1:] if c is not None and c > 0), 1)


if __name__ == "__main__":
    main()
