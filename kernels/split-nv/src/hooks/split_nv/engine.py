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
        if os.environ.get("SPLIT_NV_SELFTEST") != "numerics":
            from split_nv.consistent import install
            install()
        if os.environ.get("SPLIT_NV_OG_MOE") == "1":
            from split_nv.og_moe.install import install as og_moe_install
            og_moe_install(self)
        self.window = self.tree_cache.sliding_window_size
        from split_nv.steprunner import StepRunner

        self.steps = StepRunner(self)
        from split_nv.prefix_cache import RankStore
        from split_nv.state_pack import NUMERICS

        self.store = RankStore(self, NUMERICS)
        self.vision = getattr(self.mr.model, "vision", None) is not None
        self.image_token_id = int(getattr(self.mr.model_config.hf_config, "image_token_id", 129264))
        self.use_graph = False
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
        batch.prepare_for_extend()
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
            mr.forward(fb)
        sess.length = start + len(new_ids)
        sess.batch = batch

    # ---- commands (executed identically on every rank) --------------------------------------------------------
    def cmd_prefill(self, sid, ids, dump=True):
        sess = Session(sid)
        sess.req = self._new_req(sid)
        self.sessions[sid] = sess
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
            self._extend(sess, chunk)
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

    def cmd_prefill_chunk(self, sid, chunk, grid=None):
        """Returns (rank 0) the packed source rows this chunk produced and its GPU seconds.
        grid = (blocks, entry key or None): save the 8K grid blocks [(bid, end)] this chunk completed, and a grid
        entry at the chunk end (which must then lie on the grid)."""
        t0 = time.perf_counter()
        sess = self.sessions[sid]
        self.cap.tokens.append(torch.tensor(chunk, dtype=torch.int64))
        self.cap.ntok += len(chunk)
        self._extend(sess, chunk, replace=self._replace_rows(sid, sess.length, len(chunk)))
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
        return rows, dt

    def cmd_prefill_end(self, sid, keep_capture=False):
        """Returns (rank 0) the final state parts (window rings, ratio-2 tails, hidden tail)."""
        sess = self.sessions[sid]
        sess.alloc_len = sess.req.kv.kv_allocated_len
        self.vision_spans.pop(sid, None)
        # Free the SWA slots of the last chunk that fell out of the window now instead of at the first step
        # (identical range), so a prefilled session holds ~window SWA slots, not a whole chunk.
        self.steps.evict_swa(sess, sess.length)
        if not self.cap.enabled:
            return None
        if keep_capture:
            self.pending_capture[sid] = self.cap.export_state()
        return self.cap.final_parts()

    def cmd_restore(self, sid, key, blocks, P):
        sess = Session(sid)
        sess.req = self._new_req(sid)
        self.sessions[sid] = sess
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
        """Returns on rank 0 only once rank 1 has executed every earlier command (then its files are read/written)."""
        from sglang.srt.distributed import get_tp_group

        self.store.flush()
        get_tp_group().barrier()

    def cmd_step(self, sid, keep, ids, use_graph=None):
        sess = self.sessions[sid]
        if keep > sess.length:
            raise ValueError(f"keep {keep} > session length {sess.length}")
        return self.steps.run(sess, keep, ids, use_graph=self.use_graph if use_graph is None else use_graph)

    def cmd_capture(self, widths):
        for W in widths:
            t0 = time.perf_counter()
            self.steps.capture(W)
            log(f"rank {self.tp_rank}: captured verify graph W={W} in {time.perf_counter() - t0:.1f}s")
        self.use_graph = True

    def cmd_close(self, sid):
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
            return self.cmd_prefill_chunk(cmd[1], cmd[2], cmd[3] if len(cmd) > 3 else None)
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
        if kind == "close":
            return self.cmd_close(cmd[1])
        if kind == "sync":
            torch.cuda.synchronize()
            return None
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


# --------------------------------------------------------------------------------------------- process entry
def rank_main(server_args, port_args, gpu_id, tp_rank, conns):
    engine = Engine(server_args, port_args, gpu_id, tp_rank)
    if tp_rank != 0:
        # Commands arrive over a local pipe from rank 0 (~20 us vs ~0.3 ms for a gloo broadcast).
        while True:
            cmd = conns.recv()
            try:
                engine.execute(cmd)
            except Exception as e:  # noqa: BLE001
                log(f"rank {tp_rank} job failed: {e}\n{traceback.format_exc()}")
                from split_nv.front import fatal_cuda_error
                if fatal_cuda_error(e):
                    os._exit(4)
    from split_nv.front import Front, install_signals

    front = Front(engine, server_args)
    front.peers = conns
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
    # Prefill kernels JIT on their first use (~3 s): warm them before accepting traffic.
    t0 = time.perf_counter()
    for n in (8194, 300):
        front.submit(("prefill", 0, [100 + (i * 7919) % 100000 for i in range(n)], False))
        front.submit(("close", 0))
    log(f"warm-up done (prefill warm-up {time.perf_counter() - t0:.1f}s)")
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

    signal.signal(signal.SIGTERM, on_term)
    # A completed self-test or failed rank must not leave its peer resident.
    while not wait([p.sentinel for p in procs], timeout=1):
        if deadline and time.monotonic() > deadline[0]:
            break
    exited = [p for p in procs if not p.is_alive()]
    code = next((p.exitcode for p in exited if p.exitcode), 0)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
    sys.exit(code)


if __name__ == "__main__":
    main()
