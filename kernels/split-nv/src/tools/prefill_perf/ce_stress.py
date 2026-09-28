"""Stress test of a CE all-reduce implementation against NCCL: many rounds of two interleaved async all-reduces (as in
an overlapped chunk: AR(A) in flight while the producer stream computes B, then AR(B) while it computes A), random
sizes, GEMM load on the producer stream, bitwise comparison of every round with NCCL on a copy.
usage: ce_stress.py {pull|push} ROUNDS"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 5120


def worker(rank, impl, rounds):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29641")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    if impl == "push":
        sys.path.insert(0, "/work/tools/prefill_perf")
        from ce_push import CEAllReduce
    else:
        sys.path.insert(0, "/work/hooks")
        from split_nv.ce_allreduce import CEAllReduce
    ce = CEAllReduce(cpu, rank, 8192 * H)
    g = torch.Generator(device="cuda").manual_seed(77 + rank)
    gs = torch.Generator().manual_seed(5)  # same sizes on both ranks
    W = torch.randn(4096, 4096, device="cuda", generator=g).to(torch.bfloat16)
    C0 = W @ W  # reference product: every concurrent GEMM below must reproduce it bit for bit
    Y0 = (W.float() * 1.5 + 2.0).to(torch.bfloat16)
    bad = 0
    bad_c = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(rounds):
        na = int(torch.randint(200, 4097, (1,), generator=gs)) * H
        nb = int(torch.randint(200, 4097, (1,), generator=gs)) * H
        xa = (torch.randn(na, device="cuda", generator=g) * 20).to(torch.bfloat16)
        xb = (torch.randn(nb, device="cuda", generator=g) * 20).to(torch.bfloat16)
        ra, rb = xa.clone(), xb.clone()
        dist.all_reduce(ra)
        dist.all_reduce(rb)
        ea = ce.all_reduce_async(xa)
        for _ in range(2):  # "B computes" while A's all-reduce is in flight; outputs checked (stray DMA writes?)
            c = W @ W
            y = (W.float() * 1.5 + 2.0).to(torch.bfloat16)
            bad_c += (c.view(torch.int16) != C0.view(torch.int16)).sum() + (y.view(torch.int16) != Y0.view(torch.int16)).sum()
        eb = ce.all_reduce_async(xb)
        torch.cuda.current_stream().wait_event(ea)
        xa.mul_(1.0)  # consumer touches A right after its wait
        for _ in range(2):
            c = W @ W
            y = (W.float() * 1.5 + 2.0).to(torch.bfloat16)
            bad_c += (c.view(torch.int16) != C0.view(torch.int16)).sum() + (y.view(torch.int16) != Y0.view(torch.int16)).sum()
        torch.cuda.current_stream().wait_event(eb)
        ok = torch.equal(xa.view(torch.int16), ra.view(torch.int16)) and torch.equal(xb.view(torch.int16), rb.view(torch.int16))
        t = torch.tensor([0 if ok else 1], device="cuda")
        dist.all_reduce(t)
        bad += int(t.item() > 0)
    if rank == 0:
        print(f"{impl}: {rounds} rounds x 2 interleaved all-reduces, rounds with a mismatch on some rank: {bad}; "
              f"corrupted elements in concurrent compute (rank 0): {int(bad_c)}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(sys.argv[1], int(sys.argv[2])), nprocs=2)
