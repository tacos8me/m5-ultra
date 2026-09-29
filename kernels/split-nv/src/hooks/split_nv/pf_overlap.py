"""Split-chunk prefill overlap: a chunk of n rows runs as two halves, A = rows [0, h) and B = [h, n), that advance in
lockstep, so each half's TP all-reduces travel over PCIe while the other half computes.

Each half is an ordinary chunked-prefill forward (its own ScheduleBatch / ForwardBatch, prepared back to back exactly
as two consecutive chunks would be), run by the unmodified SGLang forward on the same CUDA stream. A runs on the
calling thread, B on a helper thread; exactly one of them executes Python at any time (a baton). At every in-place
TP all-reduce (the only collective of a prefill: attention output, MoE output, embedding) the running half launches
the all-reduce asynchronously on NCCL's stream and hands the baton over; when the baton comes back it makes the
compute stream wait for its all-reduce and continues. The resulting stream order is

    A.embed | B.embed | A.attn(0) | B.attn(0) | A.ffn(0) | B.ffn(0) | A.attn(1) | ...

with half X's all-reduce in flight during the other half's next compute segment. B.attn(L) is enqueued after
A.attn(L), so B reads A's layer-L KV exactly as the second of two sequential chunks does, and nothing else of A is
read by B. Both ranks run the same deterministic schedule, so the NCCL call order matches across ranks.

Numerics: every row's arithmetic in og-s4.4 is independent of how rows are chunked (the engine already prefills in
2048-row pieces under contention), and an all-reduce over two ranks is one bf16 add per element, so the bytes equal
those of the unsplit chunk. The halves' per-forward state is kept apart: B gets its own attention backend (forward
metadata, candidate masks), its own eager input registry, and the forward context / runner backend are swapped at
every baton hand-over.
"""
import contextvars
import dataclasses
import os
import threading

import torch

from split_nv import ce_allreduce

MIN_ROWS = int(os.environ.get("SPLIT_NV_PF_OVERLAP_MIN", "6144"))  # smallest chunk that is split
ALIGN = 128

_DEBUG_MODE = os.environ.get("SPLIT_NV_PF_OVERLAP_DEBUG", "")  # "", "noswitch", "syncswitch" (tests only)
# The copy-engine all-reduce issued asynchronously inside an overlapped chunk diverged in 1-2 of 40 prompts in the TP2
# harness stress (a 16-row tile of a half-B layer output; every all-reduce itself matched NCCL), while NCCL there and the
# CE all-reduce outside overlapped chunks never did. Until that is understood, overlapped chunks use NCCL.
_CE_IN_OVERLAP = os.environ.get("SPLIT_NV_CE_AR_IN_OVERLAP", "0") == "1"
_VERIFY = [] if os.environ.get("SPLIT_NV_CE_AR_VERIFY") == "1" else None  # diagnostics only
_state = None  # the running _Run (prefills are serialized by the front end, so at most one)
_installed = False
_backend_b = {}  # id(model runner) -> attention backend of half B
_registry_b = {}  # id(model runner) -> eager input registry of half B


def split_rows(n, enabled):
    """Rows of half A for a chunk of n rows (0 = run unsplit)."""
    if not enabled or n < MIN_ROWS:
        return 0
    return (n // 2) // ALIGN * ALIGN


def default_enabled():
    return os.environ.get("SPLIT_NV_PF_OVERLAP", "0") == "1"


class _Run:
    def __init__(self, mr):
        from sglang.srt.distributed import get_tp_group

        self.mr = mr
        self.group = get_tp_group()
        self.cv = threading.Condition()
        self.turn = "A"
        self.alive = {"A": True, "B": True}
        self.halves = {}  # thread ident -> "A" / "B"
        self.env = {}  # half -> (forward context, runner attn backend)
        self.globals = {}  # half -> its process-global per-forward state, saved at every hand-over (_save_globals)
        self.error = None
        self.n_async = 0
        self.inflight = {}  # half -> its all-reduce in flight while the other half runs (NCCL work or CE event)

    def other(self, me):
        return "B" if me == "A" else "A"

    def enter(self, me, ctx, backend):
        from sglang.srt.model_executor import forward_context as fc

        self.halves[threading.get_ident()] = me
        self.env[me] = (ctx, backend)
        fc.set_forward_context(ctx)
        self.mr.attn_backend = backend

    def wait_turn(self, me):
        with self.cv:
            self.cv.wait_for(lambda: self.turn == me)

    def _restore(self, me):
        from sglang.srt.model_executor import forward_context as fc

        ctx, backend = self.env[me]
        fc.set_forward_context(ctx)
        self.mr.attn_backend = backend
        if me in self.globals:
            _restore_globals(self.globals[me])

    def switch(self, me):
        """Hand the baton to the other half (if it is still running) and wait until it comes back."""
        other = self.other(me)
        self.globals[me] = _save_globals()
        with self.cv:
            if not self.alive[other]:
                return
            self.turn = other
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.turn == me)
        self._restore(me)

    def finish(self, me):
        with self.cv:
            self.alive[me] = False
            self.turn = self.other(me)
            self.cv.notify_all()


def _save_globals():
    """Process-global state a forward sets for itself and later reads: the capture hooks' mode/positions (the hooks'
    forward wrapper resets cur_mode when a forward returns -- A's return must not switch B's capture off: B's
    boundary layer runs after A has returned, and its capture rows would be lost) and SGLang's plain forward flags."""
    from split_nv import hooks as H
    from sglang.srt.runtime_context import get_forward

    cap = H.CAP
    return (None if cap is None else (cap.cur_mode, cap.cur_pos), dict(get_forward()._plain))


def _restore_globals(saved):
    from split_nv import hooks as H
    from sglang.srt.runtime_context import get_forward

    cap_state, plain = saved
    if cap_state is not None and H.CAP is not None:
        H.CAP.cur_mode, H.CAP.cur_pos = cap_state
    fw = get_forward()._plain
    fw.clear()
    fw.update(plain)


def fence():
    """Make the current stream wait for every all-reduce an overlapped chunk has in flight (split_nv.preempt: a step
    enqueued between two layers then starts after them; no two collectives run concurrently)."""
    run = _state
    if run is None:
        return
    for w in list(run.inflight.values()):
        if isinstance(w, torch.cuda.Event):
            torch.cuda.current_stream().wait_event(w)
        else:
            w.wait()


def _half(run):
    return None if run is None else run.halves.get(threading.get_ident())


def install():
    """Patch the in-place TP all-reduce (copy-engine path for large tensors when ce_allreduce is set up; async +
    baton hand-over inside an overlapped chunk) and the eager input registry (half B only)."""
    global _installed
    if _installed:
        return
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner

    orig_in_place = GroupCoordinator._all_reduce_in_place

    def all_reduce_in_place(self, input_):
        run = _state
        me = _half(run)
        pynccl, symm = self.pynccl_comm, self.torch_symm_mem_comm
        if (pynccl is not None and not pynccl.disabled) or (
                symm is not None and not symm.disabled and symm.should_torch_symm_mem_allreduce(input_)):
            return orig_in_place(self, input_)
        ce = ce_allreduce.for_tensor(self, input_)  # copy-engine all-reduce (SPLIT_NV_CE_AR), same bytes as NCCL
        if me is None or self is not run.group:
            if ce is not None:
                ce.all_reduce_(input_)
                return
            return orig_in_place(self, input_)
        if ce is not None and not _CE_IN_OVERLAP:
            ce = None  # inside an overlapped chunk: NCCL (asynchronous), see _CE_IN_OVERLAP
        if ce is not None and not _DEBUG_MODE:
            ref = input_.clone() if _VERIFY is not None else None
            done = ce.all_reduce_async(input_)
            run.n_async += 1
            run.inflight[me] = done
            run.switch(me)
            torch.cuda.current_stream().wait_event(done)
            run.inflight.pop(me, None)
            if ref is not None:  # diagnostics: the same sum by NCCL, compared on the GPU (no host sync)
                torch.distributed.all_reduce(ref, group=self.device_group)
                v = ref.view(ref.shape[0], -1) if ref.dim() > 1 else ref.view(1, -1)
                w = input_.view(v.shape)
                bad = (v.view(torch.int16) != w.view(torch.int16)).any(1)
                idx = torch.arange(bad.shape[0], device=bad.device)
                _VERIFY.append((me, run.n_async, tuple(input_.shape), bad.sum(),
                                torch.where(bad, idx, bad.shape[0]).min(), torch.where(bad, idx, -1).max()))
            return
        mode = _DEBUG_MODE
        if mode == "noswitch":  # debug: halves run one after the other (state separation only)
            return orig_in_place(self, input_)
        if mode == "syncswitch":  # debug: interleaved halves, synchronous all-reduces
            orig_in_place(self, input_)
            run.switch(me)
            return
        # The same collective the original issues (torch.distributed.all_reduce on this group), asynchronously.
        work = torch.distributed.all_reduce(input_, group=self.device_group, async_op=True)
        run.n_async += 1
        run.inflight[me] = work
        run.switch(me)
        work.wait()  # the compute stream waits for NCCL's stream
        run.inflight.pop(me, None)

    GroupCoordinator._all_reduce_in_place = all_reduce_in_place

    orig_load = EagerRunner.load_batch

    def load_batch(self, forward_batch, pp_proxy_tensors=None, **kwargs):
        run = _state
        if _half(run) != "B":
            return orig_load(self, forward_batch, pp_proxy_tensors, **kwargs)
        # Half B copies its inputs into its own fixed buffers: the shared registry still backs half A's batch.
        saved = self._eager_registry
        self._eager_registry = _registry_b[id(self.model_runner)]
        try:
            return orig_load(self, forward_batch, pp_proxy_tensors, **kwargs)
        finally:
            self._eager_registry = saved

    EagerRunner.load_batch = load_batch
    _installed = True


def _second_backend(mr):
    bk = _backend_b.get(id(mr))
    if bk is None:
        bk = _backend_b[id(mr)] = type(mr.attn_backend)(mr)
    return bk


def _second_registry(mr, rows):
    from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry

    reg = _registry_b.get(id(mr))
    if reg is None:
        er = mr.eager_runner
        reg = _registry_b[id(mr)] = build_decode_registry(
            device=mr.device, max_bs=er._eager_max_bs, max_num_token=max(rows, 8192, er._eager_max_bs),
            seq_len_fill_value=0, cache_loc_dtype=torch.int64, enable_mamba_track=False,
            is_encoder_decoder=False, encoder_len_fill_value=0, encoder_lens_dtype=torch.int32,
            enable_num_token_non_padded=False, register_global_num_tokens=False, require_gathered_buffer=False,
            require_mlp_tp_gather=False, dp_size=1, share_pool=False, source=None)
    return reg


def run_split(mr, fb_a, prepare_b, max_rows):
    """Forward fb_a (half A, already prepared) and the batch prepare_b() returns (half B) with overlapped
    all-reduces. prepare_b runs first, on this thread, so both batches exist before either forward starts."""
    global _state
    install()
    from sglang.srt.model_executor.forward_context import ForwardContext, get_forward_context, has_forward_context

    assert not has_forward_context() or get_forward_context() is None
    fb_b = prepare_b()
    bk_a, bk_b = mr.attn_backend, _second_backend(mr)
    _second_registry(mr, max_rows)
    run = _Run(mr)
    main_stream = torch.cuda.current_stream()
    device = torch.cuda.current_device()
    ctx_vars = contextvars.copy_context()
    ctx_a, ctx_b = ForwardContext(attn_backend=bk_a), ForwardContext(attn_backend=bk_b)

    def body_b():
        torch.cuda.set_device(device)
        with torch.cuda.stream(main_stream), torch.no_grad():
            run.wait_turn("B")
            run.enter("B", ctx_b, bk_b)
            try:
                mr.forward(fb_b)
            except BaseException as e:  # noqa: BLE001
                run.error = e
            finally:
                run.finish("B")

    helper = threading.Thread(target=lambda: ctx_vars.run(body_b), name="pf-overlap-B", daemon=True)
    from sglang.srt.model_executor import forward_context as fc

    prev_ctx = fc.set_forward_context(None)
    _state = run
    try:
        helper.start()
        run.enter("A", ctx_a, bk_a)
        try:
            with torch.no_grad():
                mr.forward(fb_a)
        finally:
            run.finish("A")
            helper.join()
    finally:
        _state = None
        fc.set_forward_context(prev_ctx)
        mr.attn_backend = bk_a
        # B's per-forward state (metadata incl. its q buffer, candidate masks) is not needed after the chunk
        bk_b.forward_metadata = None
        bk_b.tail_forward_metadata = None
        bk_b.candidate_masks = None
    if run.error is not None:
        raise run.error
    return run.n_async


__all__ = ["split_rows", "default_enabled", "run_split", "install", "fence", "MIN_ROWS"]
