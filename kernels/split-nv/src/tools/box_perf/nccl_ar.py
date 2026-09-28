"""NCCL all-reduce of prefill-sized bf16 tensors on the two box GPUs: time and result bytes per NCCL_PROTO."""
import hashlib, os, sys, torch, torch.distributed as dist
import torch.multiprocessing as mp

def main(rank, proto, out):
    for kv in proto.split("+"):
        if "=" in kv:
            k, v = kv.split("=")
            os.environ[k] = v
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29611")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    res = {}
    for rows in (8192, 2048, 128):
        x0 = (torch.randn(rows, 5120, device="cuda", generator=g) * 3).to(torch.bfloat16)
        x = x0.clone()
        for _ in range(3):
            x.copy_(x0); dist.all_reduce(x)
        torch.cuda.synchronize()
        h = hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        n = 20
        ev[0].record()
        for _ in range(n):
            dist.all_reduce(x)
        ev[1].record(); torch.cuda.synchronize()
        ms = ev[0].elapsed_time(ev[1]) / n
        res[rows] = (ms, h, rows * 5120 * 2 / ms / 1e6)
    if rank == 0:
        for rows, (ms, h, gbs) in res.items():
            print(f"proto={proto:8s} rows={rows:5d} {ms:7.3f} ms/AR  algbw {gbs:6.1f} GB/s  first-result sha {h}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    for proto in sys.argv[1:]:
        mp.spawn(main, args=(proto, None), nprocs=2)
