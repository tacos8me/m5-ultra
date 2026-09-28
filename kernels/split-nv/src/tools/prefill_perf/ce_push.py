"""Copy-engine all-reduce for the two box GPUs (TP2 prefill): data moves over PCIe by DMA (the GPUs' copy engines,
no SMs), the ranks synchronize with GPU-side stream memory operations on flags in each other's memory, and one small
bf16 add kernel reduces. Result bytes equal NCCL's: with two ranks every element is reduced once, bf16(a + b) with
round-to-nearest (commutative), which is what NCCL's ring computes as well.

Per round k (both ranks issue rounds in the same order, like any collective), x = [n] contiguous bf16, rank r owns
elements [lo_r, hi_r) (rank 0 the first half):
    push x[peer part] -> peer.R ; write peer.flag_rs = k       (reduce-scatter, both directions at once)
    wait my.flag_rs >= k ; x[my part] += R                    (bf16 add, one kernel)
    push x[my part] -> peer.G ; write peer.flag_ag = k         (all-gather)
    wait my.flag_ag >= k ; x[peer part] <- G                  (local copy)
Single R/G slots are enough: a rank starts round k only after its round k-1 finished, which needed the peer's round
k-1 all-gather push, which the peer issued after it had consumed its R and before it could start round k.
Everything runs on one comm stream per rank (after an event from the producer stream); the caller makes its stream
wait for the returned event. Measured on the box (tools/prefill_perf/ce_bench.py): 53 GB/s one way, 98 GB/s both ways.
"""
import os

import torch

_INSTANCES = {}
SNAP = None  # diagnostics: callable(tag, tensor) given copies of x made on the comm stream (before / after)


def _check(res):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return res[1] if isinstance(res, tuple) and len(res) == 2 else res


class CEAllReduce:
    def __init__(self, cpu_group, rank, max_numel, device=None):
        import cupy
        from cuda.bindings import driver as cu

        self.cu = cu
        self.rank = rank
        self.peer = 1 - rank
        self.device = torch.device("cuda", torch.cuda.current_device() if device is None else device)
        self.max_numel = int(max_numel)
        half = (self.max_numel + 1) // 2
        with cupy.cuda.Device(self.device.index):
            # plain cudaMalloc allocations (IPC needs allocation base pointers), wrapped as torch tensors
            self._mem = [cupy.cuda.Memory(half * 2), cupy.cuda.Memory(half * 2), cupy.cuda.Memory(256)]
            handles = [cupy.cuda.runtime.ipcGetMemHandle(m.ptr) for m in self._mem]
        self.R = self._wrap(self._mem[0].ptr, half, torch.bfloat16)
        self.G = self._wrap(self._mem[1].ptr, half, torch.bfloat16)
        self.flags = self._wrap(self._mem[2].ptr, 32, torch.int64)
        self.flags.zero_()
        torch.cuda.synchronize(self.device)
        got = [None, None]
        torch.distributed.all_gather_object(got, handles, group=cpu_group)
        with cupy.cuda.Device(self.device.index):
            self.peer_ptrs = [cupy.cuda.runtime.ipcOpenMemHandle(h) for h in got[self.peer]]
        self.flag_rs, self.flag_ag = self._mem[2].ptr, self._mem[2].ptr + 8
        self.peer_R, self.peer_G = self.peer_ptrs[0], self.peer_ptrs[1]
        self.peer_flag_rs, self.peer_flag_ag = self.peer_ptrs[2], self.peer_ptrs[2] + 8
        self.stream = torch.cuda.Stream(self.device, priority=-1)
        self.k = 0
        torch.distributed.barrier(group=cpu_group)

    @staticmethod
    def _wrap(ptr, numel, dtype):
        import cupy
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        mem = cupy.cuda.UnownedMemory(ptr, nbytes, None)
        arr = cupy.ndarray((nbytes,), dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(mem, 0))
        return torch.as_tensor(arr, device="cuda").view(dtype)

    def eligible(self, x):
        return (x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous() and x.device == self.device
                and 2 <= x.numel() <= self.max_numel)

    def _memcpy(self, dst, src, nbytes):
        cu = self.cu
        _check(cu.cuMemcpyAsync(cu.CUdeviceptr(dst), cu.CUdeviceptr(src), nbytes, cu.CUstream(self.stream.cuda_stream)))

    def _write(self, addr, value):
        cu = self.cu
        _check(cu.cuStreamWriteValue64(cu.CUstream(self.stream.cuda_stream), cu.CUdeviceptr(addr), value,
                                       cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT))

    def _wait(self, addr, value):
        cu = self.cu
        _check(cu.cuStreamWaitValue64(cu.CUstream(self.stream.cuda_stream), cu.CUdeviceptr(addr), value,
                                      cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ))

    def all_reduce_async(self, x):
        """In-place sum of x over the two ranks, enqueued on the comm stream after the current stream's work.
        Returns the CUDA event the consumer stream must wait for; x must stay referenced until that wait is enqueued."""
        assert self.eligible(x)
        self.k += 1
        k = self.k
        n = x.numel()
        h = n // 2
        lo, hi = (0, h) if self.rank == 0 else (h, n)
        plo, phi = (h, n) if self.rank == 0 else (0, h)
        es = 2
        base = x.data_ptr()
        ready = torch.cuda.Event()
        ready.record()
        s = self.stream
        s.wait_event(ready)
        # no x.record_stream(s): the caller holds x until its stream has waited for `done`, so every later reuse of
        # x's memory is ordered after the comm stream is done with it (record_stream would only delay reuse)
        with torch.cuda.stream(s):
            if SNAP is not None:
                SNAP("in", x.clone())
            self._memcpy(self.peer_R, base + plo * es, (phi - plo) * es)
            self._write(self.peer_flag_rs, k)
            self._wait(self.flag_rs, k)
            x.view(-1)[lo:hi].add_(self.R[:hi - lo])
            self._memcpy(self.peer_G, base + lo * es, (hi - lo) * es)
            self._write(self.peer_flag_ag, k)
            self._wait(self.flag_ag, k)
            self._memcpy(base + plo * es, self.G.data_ptr(), (phi - plo) * es)
            if SNAP is not None:
                SNAP("out", x.clone())
            done = torch.cuda.Event()
            done.record(s)
        return done

    def all_reduce_(self, x):
        torch.cuda.current_stream().wait_event(self.all_reduce_async(x))
        return x


MIN_BYTES = int(os.environ.get("SPLIT_NV_CE_AR_MIN_BYTES", str(1 << 20)))
ACTIVE = True  # tests switch between CE and NCCL at runtime (both ranks alike); the engine leaves it on


def enabled():
    return os.environ.get("SPLIT_NV_CE_AR", "0") == "1"


def setup(group, max_numel):
    """Create the CE all-reduce of a GroupCoordinator (world size 2). Collective: every rank calls it at the same
    point (engine start)."""
    if group.world_size != 2:
        return None
    inst = _INSTANCES.get(id(group))
    if inst is None:
        inst = _INSTANCES[id(group)] = CEAllReduce(group.cpu_group, group.rank_in_group, max_numel)
    return inst


def for_tensor(group, x):
    """The CE all-reduce to use for an in-place all-reduce of x on group, or None (NCCL as before): only large bf16
    tensors outside CUDA-graph capture, and only after setup()."""
    inst = _INSTANCES.get(id(group))
    if (inst is None or not ACTIVE or x.numel() * x.element_size() < MIN_BYTES or not inst.eligible(x)
            or torch.cuda.is_current_stream_capturing()):
        return None
    return inst
