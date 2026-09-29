"""Maintenance-window memory measurement: OPEN (state none, no cache) prompts of increasing length and record, per
prompt, the nvidia-smi peak (100 ms sampling), the allocator's peaks and fragmentation from GET /debug/mem (fair branch),
and the OOM-retry count. usage: mem_window.py --label L [--sizes 8193,131073,524289,1040000] [--settle 6]"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/home/ian/split-nv-fair/tools")
from og_client import Session  # noqa: E402
from og_gate import load_ids  # noqa: E402

G = 1 << 30


def mem(reset=False, segments=True):
    q = "?" + "&".join(x for x in (("reset=1" if reset else ""), ("" if segments else "segments=0")) if x)
    return json.load(urllib.request.urlopen("http://127.0.0.1:10050/debug/mem" + q, timeout=60))


def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def row(label, m):
    out = {}
    for r, d in m.items():
        out[r] = {"alloc_peak": round(d["allocated_bytes.all.peak"] / G, 2), "alloc": round(d["allocated_bytes.all.current"] / G, 2),
                  "reserved_peak": round(d["reserved_bytes.all.peak"] / G, 2), "reserved": round(d["reserved_bytes.all.current"] / G, 2),
                  "split_free": round(d["inactive_split_bytes.all.current"] / G, 2), "non_torch": round(d["non_torch"] / G, 2),
                  "device_free": round(d["device_free"] / G, 2), "retries": d["num_alloc_retries"], "ooms": d["num_ooms"],
                  "segments": d["segment.all.current"], "seg_hist": d.get("segment_hist"),
                  "largest_free_MiB": [b >> 20 for b in d.get("largest_free_blocks", [])[:6]]}
    print(json.dumps({"at": label, **out}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="/home/ian/split-nv/ref/ids-1048576.json")
    ap.add_argument("--sizes", default="8193,131073,524289,1040000")
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="/mnt/nvme-1/split-nv-ops/fair/mem-window.jsonl")
    a = ap.parse_args()
    tokens = load_ids(a.ids)
    res = {"label": a.label, "start": row("start", mem()), "smi_start": smi(), "prompts": []}
    for n in [int(x) for x in a.sizes.split(",")]:
        mem(reset=True, segments=False)
        peak = [0, 0]
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                peak[:] = [max(p, v) for p, v in zip(peak, smi())]
                time.sleep(0.1)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t0 = time.perf_counter()
        s = Session()
        s.open(tokens[:n], state="none")
        s.close()
        dt = time.perf_counter() - t0
        stop.set()
        th.join()
        after = row(f"after {n}", mem())
        time.sleep(a.settle)
        settled = row(f"settled {n}", mem())
        rec = {"n": n, "open_s": round(dt, 2), "smi_peak_MiB": peak, "smi_settled_MiB": smi(), "after": after, "settled": settled}
        print(json.dumps({k: rec[k] for k in ("n", "open_s", "smi_peak_MiB", "smi_settled_MiB")}), flush=True)
        res["prompts"].append(rec)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
