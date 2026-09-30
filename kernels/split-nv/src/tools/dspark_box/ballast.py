"""GPU memory ballast for the DSpark-on-box kill gate (a): hold N GiB on each GPU next to the running engine.

Stands in for the ~3.8-3.9 GiB/GPU of DSpark drafter weights (+ graph pool) the engine would hold if the drafter ran
on the box. N is the process's TOTAL device footprint per GPU (CUDA context included), measured with nvidia-smi
before and after, so the engine sees exactly N GiB less. No kernels run: the ballast only occupies memory.

Safety (it shares the GPUs with production):
  * refuses unless SPLIT_NV_WINDOW=1 (announced box window) or --dry-run;
  * refuses if a GPU has less than N + --min-free GiB free (a long prefill already grew the engine's cache: restart
    the engine first);
  * a watchdog checks every --poll s that the engine's GPU processes (the other compute apps on these GPUs at start)
    still exist (/proc, no engine work, so it cannot bias the timing); when one vanishes (crash, restart) it frees
    everything and exits 3 within ~--poll s, so the unit's crash restart never finds the memory taken;
  * SIGTERM/SIGINT free and exit 0; --seconds bounds the hold (default 3600).
Writes a status JSON (--status) once the memory is held: per GPU the measured footprint, context overhead, tensor bytes.

usage (host venv with CUDA torch, e.g. /home/ian/miniconda3/bin/python):
  SPLIT_NV_WINDOW=1 python3 tools/dspark_box/ballast.py --gib 3.9 --status /tmp/ballast.json &
  ... run the prefill suite ...; kill %1
  python3 tools/dspark_box/ballast.py --dry-run --gib 3.9       # prints the plan only (no CUDA)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

GIB = 1 << 30
MIB = 1 << 20


def smi_gpus():
    """[(index, total_MiB, used_MiB, free_MiB)] in PCI bus order (nvidia-smi's default order)."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    out = []
    for line in r.stdout.strip().splitlines():
        i, t, u, f = (int(x) for x in line.split(","))
        out.append((i, t, u, f))
    return out


def smi_uuid(index):
    r = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def smi_proc_mib(pid, uuid):
    """This process's used memory on one GPU (MiB), 0 if not listed yet."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    for line in r.stdout.strip().splitlines():
        p, u, m = (x.strip() for x in line.split(","))
        if int(p) == pid and u == uuid:
            return int(m)
    return 0


def smi_pids(uuids):
    """Host PIDs of the compute apps on these GPUs."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    out = set()
    for line in r.stdout.strip().splitlines():
        p, u = (x.strip() for x in line.split(","))
        if u in uuids:
            out.add(int(p))
    return out


def health(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:  # noqa: BLE001 -- refused / timeout: engine gone
        return None, {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=3.9, help="total footprint per GPU (context included)")
    ap.add_argument("--gpus", default="0,1", help="nvidia-smi indices (PCI bus order, as the engine's CUDA_DEVICE_ORDER)")
    ap.add_argument("--min-free", type=float, default=0.5, help="GiB that must stay free on each GPU after the ballast")
    ap.add_argument("--health", default="http://127.0.0.1:10051/health")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=3600.0)
    ap.add_argument("--status", default="")
    ap.add_argument("--no-watchdog", action="store_true", help="only for tests without an engine")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    before = {i: (t, u, f) for i, t, u, f in smi_gpus()}
    plan = {"gib": a.gib, "gpus": gpus, "free_before_GiB": {i: round(before[i][2] / 1024, 2) for i in gpus}}
    short = [i for i in gpus if before[i][2] / 1024 < a.gib + a.min_free]
    if a.dry_run:
        plan["would_refuse"] = short
        print(json.dumps(plan), flush=True)
        return 0
    if os.environ.get("SPLIT_NV_WINDOW") != "1":
        print("refusing: set SPLIT_NV_WINDOW=1 inside an announced box window", file=sys.stderr)
        return 2
    if short:
        print(f"refusing: GPUs {short} have < {a.gib + a.min_free:.1f} GiB free {plan['free_before_GiB']} "
              "(restart the engine first: its allocator cache grows during long prefills)", file=sys.stderr)
        return 2
    engine_pids = set()
    if not a.no_watchdog:
        code, body = health(a.health)
        if code != 200:
            print(f"refusing: engine /health {code}", file=sys.stderr)
            return 2
        engine_pids = smi_pids({smi_uuid(i) for i in gpus})
        if not engine_pids:
            print("refusing: no engine process on these GPUs", file=sys.stderr)
            return 2

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch

    pid = os.getpid()
    held, status = [], {"pid": pid, "gib_target": a.gib, "gpus": {}, "engine_pids": sorted(engine_pids)}
    for i in gpus:
        uuid = smi_uuid(i)
        torch.cuda.set_device(i)
        torch.zeros(1, device=f"cuda:{i}")  # create the context
        torch.cuda.synchronize(i)
        ctx_mib = 0
        for _ in range(20):  # nvidia-smi's per-process view lags a little
            ctx_mib = smi_proc_mib(pid, uuid)
            if ctx_mib:
                break
            time.sleep(0.1)
        want = int(a.gib * 1024) - ctx_mib  # MiB of tensor so that context + tensor = gib
        if want <= 0:
            raise SystemExit(f"GPU {i}: context alone is {ctx_mib} MiB >= target")
        target = int(a.gib * 1024)
        mine = []
        for _ in range(4):  # converge on the footprint: the allocator rounds, the context grows on first allocation
            if mine:
                mine.clear()
                torch.cuda.synchronize(i)
                torch.cuda.empty_cache()
            mine.append(torch.empty(want * MIB, dtype=torch.uint8, device=f"cuda:{i}"))
            torch.cuda.synchronize(i)
            time.sleep(0.3)
            total = smi_proc_mib(pid, uuid)
            err = target - total
            if abs(err) <= 32:
                break
            want += err
        held.extend(mine)
        status["gpus"][i] = {"uuid": uuid, "context_MiB": ctx_mib, "tensor_MiB": sum(x.numel() for x in held
                                                                                    if x.device.index == i) // MIB,
                             "footprint_MiB": total, "footprint_GiB": round(total / 1024, 3)}
    after = {i: (t, u, f) for i, t, u, f in smi_gpus()}
    status["free_after_GiB"] = {i: round(after[i][2] / 1024, 2) for i in gpus}
    status["held_at"] = time.time()
    line = json.dumps(status)
    print(line, flush=True)
    if a.status:
        with open(a.status + ".tmp", "w") as f:
            f.write(line + "\n")
        os.replace(a.status + ".tmp", a.status)

    stop = threading.Event()
    rc = [0]

    def on_signal(*_):
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    deadline = time.monotonic() + a.seconds
    while not stop.is_set() and time.monotonic() < deadline:
        gone = [p for p in engine_pids if not os.path.exists(f"/proc/{p}")]
        if gone:
            print(f"[ballast] engine process(es) {gone} gone: releasing the ballast", file=sys.stderr, flush=True)
            rc[0] = 3
            break
        stop.wait(a.poll)
    held.clear()
    torch.cuda.empty_cache()
    print(json.dumps({"released_at": time.time(), "rc": rc[0]}), flush=True)
    return rc[0]


if __name__ == "__main__":
    sys.exit(main())
