"""Two-process test of split_nv.ce_allreduce against NCCL (torch.distributed.all_reduce, what the engine's prefill
uses): bitwise equality over sizes, time per all-reduce alone, and both under a concurrent GEMM load (SM contention).
usage: ce_ar_test.py"""
import os
import statistics as st
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 5120


def sync():
    torch.cuda.synchronize()
    dist.barrier()


def timed(fn, n=20):
    for _ in range(3):
        fn()
    sync()
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29633")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    sys.path.insert(0, "/work/hooks")
    from split_nv.ce_allreduce import CEAllReduce
    ce = CEAllReduce(cpu, rank, 8192 * H)
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    ok_all = True
    for rows, extra in ((8192, 0), (4096, 0), (4097, 0), (100, 7), (2944, 0), (8191, 3)):
        n = rows * H + extra
        x0 = (torch.randn(n, device="cuda", generator=g) * torch.rand(1, device="cuda", generator=g) * 40).to(torch.bfloat16)
        x0[: n // 50] = (torch.randn(n // 50, device="cuda", generator=g) * 1e4).to(torch.bfloat16)  # wide exponents
        a = x0.clone()
        b = x0.clone()
        dist.all_reduce(a)
        ce.all_reduce_(b)
        torch.cuda.synchronize()
        ok = torch.equal(a.view(torch.int16), b.view(torch.int16))
        ok_all &= ok
        if rank == 0:
            print(f"numel {n:9d}: CE == NCCL bitwise {ok}", flush=True)
    # timing (84 MB and 42 MB)
    for rows in (8192, 4096):
        x = torch.randn(rows * H, device="cuda", generator=g).to(torch.bfloat16)
        tn = timed(lambda: dist.all_reduce(x))
        tc = timed(lambda: ce.all_reduce_(x))
        if rank == 0:
            mb = rows * H * 2 / 1e6
            print(f"{mb:5.1f} MB alone: NCCL {tn:6.3f} ms ({mb / tn:5.1f} GB/s)  CE {tc:6.3f} ms ({mb / tc:5.1f} GB/s)", flush=True)
    # under load: 8 x (8192^2 bf16 GEMM) on the default stream while 4 x 42 MB all-reduces run on their own stream
    A = torch.randn(8192, 8192, device="cuda", generator=g).to(torch.bfloat16)
    xs = [torch.randn(4096 * H, device="cuda", generator=g).to(torch.bfloat16) for _ in range(4)]
    side = torch.cuda.Stream(priority=-1)

    def gemms():
        for _ in range(8):
            A @ A

    def with_nccl():
        ev = torch.cuda.Event()
        ev.record()
        works = []
        with torch.cuda.stream(side):
            side.wait_event(ev)
            for x in xs:
                works.append(dist.all_reduce(x, async_op=True))
        gemms()
        for w in works:
            w.wait()

    def with_ce():
        evs = [ce.all_reduce_async(x) for x in xs]
        gemms()
        for e in evs:
            torch.cuda.current_stream().wait_event(e)

    tg = timed(gemms, 10)
    tn4 = timed(lambda: [dist.all_reduce(x) for x in xs], 10)
    tc4 = timed(lambda: [ce.all_reduce_(x) for x in xs], 10)
    tgn = timed(with_nccl, 10)
    tgc = timed(with_ce, 10)
    if rank == 0:
        print(f"8 GEMMs {tg:6.2f} ms; 4 x 42 MB AR: NCCL {tn4:6.2f} CE {tc4:6.2f}; overlapped: GEMMs+NCCL {tgn:6.2f} "
              f"(sum {tg + tn4:6.2f}, hidden {100 * (tg + tn4 - tgn) / tn4:5.1f}%), GEMMs+CE {tgc:6.2f} "
              f"(sum {tg + tc4:6.2f}, hidden {100 * (tg + tc4 - tgc) / tc4:5.1f}%)", flush=True)
        print(f"check {'PASS' if ok_all else 'FAIL'}; max allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
