"""Copy-engine all-reduce for the two box GPUs (TP2 prefill): data moves over PCIe by DMA (the GPUs' copy engines,
no SMs), the ranks synchronize with GPU-side stream memory operations on flags, and one small bf16 add kernel
reduces. Result bytes equal NCCL's: with two ranks every element is reduced once, bf16(a + b) with round-to-nearest
(commutative), which is what NCCL's ring computes as well.

Pull protocol. Every remote access is a READ of data that its owner finished writing, in its own memory, before it
set a flag in its own memory; nothing is ever written into the other GPU. (A first push version -- DMA into the
peer's buffer, then cuStreamWriteValue into the peer's flag -- lost a 128-row block once in the TP2 harness: a remote
flag write is not ordered behind the copy engine's remote data writes.)

Per round k (both ranks issue rounds in the same order, like any collective), x = [n] contiguous bf16, rank r owns
elements [lo_r, hi_r) (rank 0 the first half). Registered (IPC) buffers per rank: S (partials for the peer),
S2 (my reduced part), R (receive), flags.
    x[peer part] -> S (local copy) ; my.flag_s = k
    wait peer.flag_s >= k ; peer.S -> R (DMA read) ; x[my part] += R (bf16 add) ; x[my part] -> S2 ; my.flag_g = k
    wait peer.flag_g >= k ; peer.S2 -> x[peer part] (DMA read)
Buffer reuse: I overwrite S in round k+1 only after my round k waited peer.flag_g >= k, which the peer set after it
had read my S(k); I overwrite S2 in round k+1 after waiting peer.flag_s >= k+1, set when the peer's round k (incl. its
read of my S2(k)) was done. Everything runs on one comm stream per rank after an event from the producer stream; the
caller makes its stream wait for the returned event. Measured on the box (tools/prefill_perf/ce_bench.py): 53 GB/s
one way, 98 GB/s both ways at once.
"""
import os

import torch

_INSTANCES = {}
SNAP = None  # diagnostics: callable(tag, tensor) given copies of x made on the comm stream (before / after)
_RECORD = os.environ.get("SPLIT_NV_CE_AR_RECORD", "0") == "1"  # diagnostics


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
            self._mem = [cupy.cuda.Memory(half * 2), cupy.cuda.Memory(half * 2), cupy.cuda.Memory(half * 2),
                         cupy.cuda.Memory(256)]
            handles = [cupy.cuda.runtime.ipcGetMemHandle(m.ptr) for m in (self._mem[0], self._mem[1], self._mem[3])]
        self.S = self._wrap(self._mem[0].ptr, half, torch.bfloat16)
        self.S2 = self._wrap(self._mem[1].ptr, half, torch.bfloat16)
        self.R = self._wrap(self._mem[2].ptr, half, torch.bfloat16)
        self.flags = self._wrap(self._mem[3].ptr, 32, torch.int64)
        self.flags.zero_()
        torch.cuda.synchronize(self.device)
        got = [None, None]
        torch.distributed.all_gather_object(got, handles, group=cpu_group)
        with cupy.cuda.Device(self.device.index):
            self.peer_S, self.peer_S2, peer_fl = [cupy.cuda.runtime.ipcOpenMemHandle(h) for h in got[self.peer]]
        fl = self._mem[3].ptr
        self.flag_s, self.flag_g = fl, fl + 8
        self.peer_flag_s, self.peer_flag_g = peer_fl, peer_fl + 8
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

    def _set(self, addr, value):  # local flag, ordered after all earlier work of the stream (with a memory barrier)
        cu = self.cu
        _check(cu.cuStreamWriteValue64(cu.CUstream(self.stream.cuda_stream), cu.CUdeviceptr(addr), value,
                                       cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT))

    def _wait(self, addr, value):  # remote flag
        cu = self.cu
        _check(cu.cuStreamWaitValue64(cu.CUstream(self.stream.cuda_stream), cu.CUdeviceptr(addr), value,
                                      cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ))

    def all_reduce_async(self, x):
        """In-place sum of x over the two ranks, enqueued on the comm stream after the current stream's work.
        Returns the CUDA event the consumer stream must wait for; x must stay referenced until that wait is enqueued
        (then every later reuse of its memory is ordered after the comm stream is done with it)."""
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
        if _RECORD:
            x.record_stream(s)
        with torch.cuda.stream(s):
            if SNAP is not None:
                SNAP("in", x.clone())
            self._memcpy(self.S.data_ptr(), base + plo * es, (phi - plo) * es)
            self._set(self.flag_s, k)
            self._wait(self.peer_flag_s, k)
            self._memcpy(self.R.data_ptr(), self.peer_S, (hi - lo) * es)
            x.view(-1)[lo:hi].add_(self.R[:hi - lo])
            self._memcpy(self.S2.data_ptr(), base + lo * es, (hi - lo) * es)
            self._set(self.flag_g, k)
            self._wait(self.peer_flag_g, k)
            self._memcpy(base + plo * es, self.peer_S2, (phi - plo) * es)
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
