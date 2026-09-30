"""Kill gate (a) for DSpark-on-box: does holding N GiB per GPU next to the engine regress long prefills?

Arms alternate control (A: nothing extra) and ballast (B: tools/dspark_box/ballast.py holding --gib per GPU, context
included). Every arm starts from a freshly restarted engine (the caching allocator's reserved pool only grows, so a
ballast can only be placed on a fresh engine, and fresh-vs-fresh keeps the arms comparable), then runs
  warm-up 8193 (not scored) -> 131073 -> 524289 -> 1040000
as fresh OPENs (cache 0, state/stream as production by default), recording per prompt the box's prefill_s, the
client OPEN time, nvidia-smi peaks (100 ms sampling, both GPUs), and from the journal (since the arm start) the
MEMLOG line of every prefill (alloc/reserved peaks, device free, cumulative allocator retries and OOMs), every
"memory allocation failed with OOM" warning (a caching-allocator flush + retry = a stall) and stallwatch lines.

Verdict (printed and written to <out>/summary.json):
  FAIL  1M prefill median(B) > median(A) * (1 + --max-reg, default 1%), or any OOM / failed OPEN / engine crash in B
  WARN  timing passes but B has more allocator retries per 1M prefill than A (fragmentation headroom is thin)
  PASS  otherwise
  INCONCLUSIVE when the control arms' own 1M spread exceeds the margin (run more pairs)

Safety: needs SPLIT_NV_WINDOW=1 (announced box window, Mac told ds41 is down), a quiet engine (0 sessions), the box
llama-swap stopped; restarts the engine with `systemctl --user restart split-nv-engine` (the pinned deploy clone,
unchanged config); the ballast frees itself if an engine process disappears. Ends with a restart unless
--no-final-restart, so production resumes on a fresh allocator.

usage: SPLIT_NV_WINDOW=1 /home/ian/.venv/bin/python tools/dspark_box/ballast_gate.py --gib 3.9 --pairs 2 \
         --out /mnt/nvme-1/split-nv-ops/dspark-box/ballast-<label>
       ... --dry-run   (checks preconditions, prints the plan, touches nothing)
"""
import argparse
import json
import os
import re
import signal
import statistics as S
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from og_client import Session  # noqa: E402

HEALTH = "http://127.0.0.1:10051/health"
TORCH_PY = "/home/ian/miniconda3/bin/python"
MEMLOG = re.compile(r"rank (\d) mem after prefill \d+ \((\d+) tokens\): alloc ([\d.]+) peak ([\d.]+) GiB, reserved ([\d.]+) "
                    r"peak ([\d.]+) GiB, split-free ([\d.]+) GiB, non-torch ([\d.]+) GiB, device free ([\d.]+) GiB, "
                    r"retries (\d+) ooms (\d+)")
OOMW = re.compile(r"\[rank(\d)\].*memory allocation failed with OOM on device \d while trying to allocate (\d+) bytes "
                  r"\(free: (\d+)")
STALL = re.compile(r"stallwatch: (host stall|gc) .*")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def health():
    try:
        with urllib.request.urlopen(HEALTH, timeout=5) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def smi_used():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def unit_active(name):
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", name]).returncode == 0


def restart_engine(timeout=900):
    t = time.time()
    log("restarting split-nv-engine")
    subprocess.run(["systemctl", "--user", "restart", "split-nv-engine"], check=True)
    time.sleep(5)
    while time.time() - t < timeout:
        h = health()
        if h and h.get("ok") and h.get("sessions") == 0:
            log(f"engine up after {time.time() - t:.0f}s: version {h.get('version')} numerics {h.get('numerics')}")
            return h
        time.sleep(3)
    raise RuntimeError("engine did not come up")


def journal_since(t0):
    r = subprocess.run(["journalctl", "--user", "-u", "split-nv-engine", "--since", f"@{int(t0)}", "--no-pager",
                        "-o", "cat"], capture_output=True, text=True)
    mem, ooms, stalls = [], [], []
    for line in r.stdout.splitlines():
        m = MEMLOG.search(line)
        if m:
            r_, n, a, ap, rs, rp, sf, nt, df, rt, oo = m.groups()
            mem.append({"rank": int(r_), "tokens": int(n), "alloc": float(a), "alloc_peak": float(ap), "reserved": float(rs),
                        "reserved_peak": float(rp), "split_free": float(sf), "non_torch": float(nt),
                        "device_free": float(df), "retries": int(rt), "ooms": int(oo)})
            continue
        m = OOMW.search(line)
        if m:
            ooms.append({"rank": int(m.group(1)), "bytes": int(m.group(2)), "free": int(m.group(3))})
            continue
        m = STALL.search(line)
        if m:
            stalls.append(m.group(0)[:200])
    return mem, ooms, stalls


class _ArmDone(Exception):
    pass


class Ballast:
    def __init__(self, gib, status):
        self.status = status
        if os.path.exists(status):
            os.unlink(status)
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.p = subprocess.Popen([TORCH_PY, os.path.join(HERE, "ballast.py"), "--gib", str(gib), "--status", status,
                                   "--seconds", "3600"], env=env)
        t = time.time()
        while not os.path.exists(status):
            if self.p.poll() is not None:
                raise RuntimeError(f"ballast exited {self.p.returncode} before holding memory")
            if time.time() - t > 120:
                self.stop()
                raise RuntimeError("ballast did not report within 120 s")
            time.sleep(0.2)
        self.info = json.load(open(status))
        log("ballast held:", json.dumps(self.info["gpus"]))

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        return self.p.returncode


def run_prompt(tokens, n, state, stream):
    peak = [0, 0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[:] = [max(p, v) for p, v in zip(peak, smi_used())]
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    rec = {"n": n}
    try:
        s = Session(timeout=1800)
        opts = {"stream": 1} if stream and state != "none" else {}
        ack, _tensors, _manifest, info = s.open(tokens[:n], state=state, **opts)
        rec["prefill_s"] = (info.get("end") or {}).get("prefill_s", ack.get("prefill_s"))
        rec["resumed"] = ack.get("resumed_tokens")
        s.close()
        rec["open_s"] = round(time.perf_counter() - t0, 3)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=str(e)[:300], open_s=round(time.perf_counter() - t0, 3))
    stop.set()
    th.join()
    rec["smi_peak_MiB"] = peak
    return rec


def run_arm(kind, idx, a, tokens, out):
    h = restart_engine() if a.restart else health()
    t0 = time.time()
    arm = {"arm": f"{kind}{idx}", "kind": kind, "t0": t0, "version": (h or {}).get("version"), "prompts": []}
    ballast = None
    try:
        if kind == "B":
            try:
                ballast = Ballast(a.gib, os.path.join(out, f"ballast-{idx}.json"))
            except Exception as e:  # noqa: BLE001 -- e.g. < gib + 0.5 GiB free on a fresh engine: a result in itself
                arm.update(aborted=True, ballast_error=str(e)[:300], prompts=[{"n": 0, "ok": False, "error": str(e)[:300]}])
                log(f"  ballast could not be placed: {e}")
                raise _ArmDone()
            arm["ballast"] = ballast.info
        arm["smi_start_MiB"] = smi_used()
        for n in [a.warm] + a.sizes:
            rec = run_prompt(tokens, n, a.state, a.stream)
            rec["scored"] = n != a.warm
            log(f"  {arm['arm']} n={n}: prefill {rec.get('prefill_s')} s, open {rec['open_s']} s, "
                f"smi peak {rec['smi_peak_MiB']} MiB" + ("" if rec["ok"] else f"  FAILED {rec['error']}"))
            arm["prompts"].append(rec)
            if not rec["ok"] or (ballast is not None and not ballast.alive()):
                arm["aborted"] = True
                break
            time.sleep(a.settle)
    except _ArmDone:
        pass
    finally:
        if ballast is not None:
            arm["ballast_rc"] = ballast.stop()
    time.sleep(2)
    mem, ooms, stalls = journal_since(t0)
    arm.update(memlog=mem, oom_retries=ooms, stalls=stalls, engine_ok=bool((health() or {}).get("ok")))
    with open(os.path.join(out, "arms.jsonl"), "a") as f:
        f.write(json.dumps(arm) + "\n")
    return arm


def summarize(arms, a):
    res = {"gib": a.gib, "sizes": a.sizes, "max_reg": a.max_reg, "per_size": {}}
    for n in a.sizes:
        row = {}
        for kind in "AB":
            vals = [p["prefill_s"] for arm in arms if arm["kind"] == kind for p in arm["prompts"]
                    if p["n"] == n and p.get("ok") and p.get("prefill_s") is not None]
            row[kind] = {"values": vals, "median": S.median(vals) if vals else None,
                         "spread": (max(vals) - min(vals)) / S.median(vals) if len(vals) > 1 else None}
        if row["A"]["median"] and row["B"]["median"]:
            row["delta"] = row["B"]["median"] / row["A"]["median"] - 1
        res["per_size"][n] = row
    for kind in "AB":
        ka = [arm for arm in arms if arm["kind"] == kind]
        res[f"oom_retries_{kind}"] = sum(len(arm["oom_retries"]) for arm in ka)
        res[f"ooms_{kind}"] = sum(max((m["ooms"] for m in arm["memlog"]), default=0) for arm in ka)
        res[f"failed_{kind}"] = sum(1 for arm in ka for p in arm["prompts"] if not p.get("ok"))
        res[f"engine_down_{kind}"] = sum(1 for arm in ka if not arm["engine_ok"] or arm.get("ballast_rc") == 3)
        res[f"reserved_peak_{kind}"] = max((m["reserved_peak"] for arm in ka for m in arm["memlog"]), default=None)
        res[f"alloc_peak_{kind}"] = max((m["alloc_peak"] for arm in ka for m in arm["memlog"]), default=None)
    big = res["per_size"][max(a.sizes)]
    verdict = "PASS"
    if res["failed_B"] or res["ooms_B"] or res["engine_down_B"]:
        verdict = "FAIL"
    elif big.get("delta") is None:
        verdict = "INCONCLUSIVE"
    elif big["delta"] > a.max_reg:
        verdict = "FAIL"
    elif (big["A"]["spread"] or 0) > a.max_reg:
        verdict = "INCONCLUSIVE"
    elif res["oom_retries_B"] > res["oom_retries_A"]:
        verdict = "WARN"
    res["verdict"] = verdict
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=3.9)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--order", default="AB", help="arm order inside a pair (AB or BA)")
    ap.add_argument("--ids", default="/home/ian/split-nv/ref/ids-1048576.json")
    ap.add_argument("--sizes", default="131073,524289,1040000")
    ap.add_argument("--warm", type=int, default=8193)
    ap.add_argument("--state", default="full", choices=["full", "lean", "none"])
    ap.add_argument("--no-stream", dest="stream", action="store_false")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--max-reg", type=float, default=0.01)
    ap.add_argument("--no-restart", dest="restart", action="store_false",
                    help="do not restart between arms (only valid for a single B arm on an already fresh engine)")
    ap.add_argument("--no-final-restart", dest="final_restart", action="store_false")
    ap.add_argument("--out", default="/mnt/nvme-1/split-nv-ops/dspark-box/ballast")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.sizes = [int(x) for x in a.sizes.split(",")]

    h = health()
    problems = []
    if os.environ.get("SPLIT_NV_WINDOW") != "1":
        problems.append("SPLIT_NV_WINDOW=1 not set (announced box window only)")
    if not h or not h.get("ok"):
        problems.append(f"engine not healthy: {h}")
    elif h.get("sessions"):
        problems.append(f"{h['sessions']} session(s) open: not a quiet box")
    if unit_active("llama-swap"):
        problems.append("box llama-swap is active: systemctl --user stop llama-swap first")
    if not os.path.exists(TORCH_PY):
        problems.append(f"{TORCH_PY} missing (the ballast needs CUDA torch)")
    arms = [k for _ in range(a.pairs) for k in a.order]
    plan = {"arms": arms, "gib": a.gib, "sizes": a.sizes, "state": a.state, "stream": a.stream, "restart": a.restart,
            "est_minutes": round(len(arms) * (2.0 + 1.9) + 2, 0), "engine": {k: (h or {}).get(k) for k in
                                                                           ("version", "numerics", "sessions")}}
    print(json.dumps({"plan": plan, "problems": problems}), flush=True)
    if a.dry_run:
        return 0
    if problems:
        return 2
    os.makedirs(a.out, exist_ok=True)
    tokens = json.load(open(a.ids))
    tokens = tokens["tokens"] if isinstance(tokens, dict) else tokens
    results, counts = [], {"A": 0, "B": 0}
    try:
        for kind in arms:
            counts[kind] += 1
            log(f"arm {kind}{counts[kind]}")
            arm = run_arm(kind, counts[kind], a, tokens, a.out)
            results.append(arm)
            if kind == "B" and (arm.get("aborted") or arm.get("ballast_rc") == 3):
                log("ballast arm aborted: stopping the gate (FAIL)")
                break
    finally:
        if a.final_restart:
            try:
                restart_engine()
            except Exception as e:  # noqa: BLE001 -- still write the summary; the operator sees this line
                log(f"FINAL RESTART FAILED: {e}")
    res = summarize(results, a)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    for n, row in res["per_size"].items():
        d = row.get("delta")
        log(f"n={n}: A {row['A']['values']}  B {row['B']['values']}  delta {'' if d is None else f'{100 * d:+.2f}%'}")
    log(f"OOM-retries A {res['oom_retries_A']} B {res['oom_retries_B']}; reserved peak A {res['reserved_peak_A']} "
        f"B {res['reserved_peak_B']} GiB; alloc peak A {res['alloc_peak_A']} B {res['alloc_peak_B']} GiB")
    log(f"VERDICT {res['verdict']}")
    return 0 if res["verdict"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
