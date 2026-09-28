"""Copy-engine (DMA) peer transfers between the two box GPUs vs what an all-reduce needs (one process, both GPUs,
<= 1 GB each). Measures: one-way and simultaneous two-way cudaMemcpyPeerAsync bandwidth at 42 / 84 MB; a CE all-reduce
(reduce-scatter by peer pulls + a bf16 add kernel + all-gather by peer pulls) against NCCL-free reference sums
(bitwise); and how much a concurrent compute kernel slows the copies and vice versa.
usage: ce_bench.py"""
import statistics as st

import torch

H = 5120


def ev():
    return torch.cuda.Event(enable_timing=True)


def time_it(fn, n=20, dev=0):
    import time
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)


def main():
    assert torch.cuda.device_count() == 2
    print("peer access 0->1", torch.cuda.can_device_access_peer(0, 1), "1->0", torch.cuda.can_device_access_peer(1, 0))
    s0 = torch.cuda.Stream(0)
    s1 = torch.cuda.Stream(1)
    for rows in (4096, 8192):
        mb = rows * H * 2 / 1e6
        x0 = torch.randn(rows, H, device="cuda:0").to(torch.bfloat16)
        x1 = torch.randn(rows, H, device="cuda:1").to(torch.bfloat16)
        r0 = torch.empty_like(x0)
        r1 = torch.empty_like(x1)

        def pull1():  # GPU1 pulls x0 (DMA issued on GPU1's stream)
            with torch.cuda.stream(s1):
                r1.copy_(x0, non_blocking=True)

        def push0():
            with torch.cuda.stream(s0):
                r1.copy_(x0, non_blocking=True)

        def both():
            with torch.cuda.stream(s1):
                r1.copy_(x0, non_blocking=True)
            with torch.cuda.stream(s0):
                r0.copy_(x1, non_blocking=True)

        t1 = time_it(pull1, dev=1)
        t2 = time_it(push0, dev=0)
        t3 = time_it(both, dev=0)
        print(f"{mb:5.1f} MB  pull 0->1 {t1:6.3f} ms ({mb / t1:5.1f} GB/s)  push 0->1 {t2:6.3f} ms ({mb / t2:5.1f} GB/s)  "
              f"both ways at once {t3:6.3f} ms ({2 * mb / t3:5.1f} GB/s total)", flush=True)

        # CE all-reduce: rank r owns rows [r*h, (r+1)*h). RS: each pulls the peer's partial of its own rows, adds.
        # AG: each pulls the peer's reduced rows. Two ways at once in each phase.
        h = rows // 2
        own0, own1 = slice(0, h), slice(h, rows)
        recv0 = torch.empty(h, H, device="cuda:0", dtype=torch.bfloat16)
        recv1 = torch.empty(h, H, device="cuda:1", dtype=torch.bfloat16)
        out0 = torch.empty_like(x0)
        out1 = torch.empty_like(x1)

        def ce_ar():
            e0, e1 = torch.cuda.Event(), torch.cuda.Event()
            with torch.cuda.stream(s0):
                recv0.copy_(x1[own0], non_blocking=True)
                torch.add(x0[own0], recv0, out=out0[own0])
                e0.record(s0)
            with torch.cuda.stream(s1):
                recv1.copy_(x0[own1], non_blocking=True)
                torch.add(x1[own1], recv1, out=out1[own1])
                e1.record(s1)
            s0.wait_event(e1)
            s1.wait_event(e0)
            with torch.cuda.stream(s0):
                out0[own1].copy_(out1[own1], non_blocking=True)
            with torch.cuda.stream(s1):
                out1[own0].copy_(out0[own0], non_blocking=True)
            s0.wait_stream(s0)

        t4 = time_it(ce_ar, dev=0)
        ref = (x0.float() + x1.to("cuda:0").float()).to(torch.bfloat16)
        ok = torch.equal(out0, ref) and torch.equal(out1.to("cuda:0"), ref)
        hadd = torch.equal(torch.add(x0, x1.to("cuda:0")), ref)
        print(f"{mb:5.1f} MB  CE all-reduce {t4:6.3f} ms (algbw {mb / t4:5.1f} GB/s); bitwise == bf16(fp32 a+b): {ok}; "
              f"torch bf16 add == bf16(fp32 a+b): {hadd}", flush=True)

        # interference: a long bf16 GEMM stream on each GPU while the copies run
        A0 = torch.randn(8192, 8192, device="cuda:0", dtype=torch.bfloat16)
        A1 = A0.to("cuda:1")
        g0 = torch.cuda.Stream(0)
        g1 = torch.cuda.Stream(1)

        def gemms():
            with torch.cuda.stream(g0):
                for _ in range(4):
                    A0 @ A0
            with torch.cuda.stream(g1):
                for _ in range(4):
                    A1 @ A1

        tg = time_it(gemms, n=10)
        tgc = time_it(lambda: (gemms(), ce_ar()), n=10)
        print(f"{mb:5.1f} MB  4 GEMMs alone {tg:6.3f} ms; with a CE all-reduce alongside {tgc:6.3f} ms "
              f"(+{tgc - tg:5.3f} ms = CE AR not hidden or GEMM slowed)", flush=True)
        del A0, A1
    print(f"max allocated dev0 {torch.cuda.max_memory_allocated(0) / 2**30:.2f} GiB dev1 {torch.cuda.max_memory_allocated(1) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
