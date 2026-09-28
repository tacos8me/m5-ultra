"""Host time of CEAllReduce.all_reduce_async per call (should be ~100 us: every step is an async enqueue)."""
import os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29634")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    sys.path.insert(0, "/work/hooks")
    from split_nv.ce_allreduce import CEAllReduce
    ce = CEAllReduce(cpu, rank, 8192 * 5120)
    x = torch.randn(8192 * 5120, device="cuda").to(torch.bfloat16)
    A = torch.randn(8192, 8192, device="cuda").to(torch.bfloat16)
    for _ in range(3):
        ce.all_reduce_(x)
    torch.cuda.synchronize(); dist.barrier()
    # keep the GPU busy with GEMMs so any host blocking shows
    for _ in range(20):
        A @ A
    t = []
    for i in range(10):
        t0 = time.perf_counter()
        ev = ce.all_reduce_async(x)
        t1 = time.perf_counter()
        torch.cuda.current_stream().wait_event(ev)
        t.append((t1 - t0) * 1e3)
    t2 = time.perf_counter()
    torch.cuda.synchronize()
    tw = []
    for i in range(10):
        t0 = time.perf_counter(); work = dist.all_reduce(x, async_op=True); tw.append((time.perf_counter() - t0) * 1e3); work.wait()
    torch.cuda.synchronize()
    if rank == 0:
        print("CE all_reduce_async host ms per call:", [round(v, 3) for v in t], flush=True)
        print("NCCL async all_reduce host ms per call:", [round(v, 3) for v in tw], flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
