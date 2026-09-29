"""Phase timeline of every engine stop/start in the journal (split-nv-engine user unit).

usage: python3 tools/restart_timeline.py [--since "2026-09-29 17:50"] [--until ...]
Per start: trigger (admin crash / drain done / watchdog / fatal / unit stop) -> unit exit (teardown), RestartSec gap,
Started -> Load weight begin (container + imports + spawn), weights (-> Finished streaming), post-load + Engram
(-> Load weight end), -> model ready, -> prefix cache line (index), -> warm-up done, -> http, and the outage from
the trigger to http. Engram detail (copy / register / kept-segment reuse) and the new teardown / prefetch / index
lines are printed underneath when present.
"""
import argparse
import datetime as D
import re
import subprocess

MARKS = [("admin crash", "trigger"), ("drain done", "trigger"), ("WATCHDOG", "trigger"), ("FATAL CUDA", "trigger"),
         ("Stopping split-nv-engine", "trigger"), ("Main process exited", "exit"), ("Control process exited", "exit"),
         ("Started split-nv-engine", "started"), ("Load weight begin", "lw"), ("Finished streaming dequant", "fin"),
         ("TP0] Load weight end", "lwend"), ("rank 0: model ready", "ready"), ("prefix cache:", "pcache"),
         ("warm-up done", "warm"), ("http on 0.0.0.0", "http")]
DETAIL = re.compile(r"(TP0\] engram host table layer \d+:.*|\[prefetch\].*|reaped .*|SIGKILL to .*|"
                    r"index built in [\d.]+s during the model load, waited [\d.]+s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="-1 day")
    ap.add_argument("--until", default="now")
    a = ap.parse_args()
    out = subprocess.run(["journalctl", "--user", "-u", "split-nv-engine", "--since", a.since, "--until", a.until,
                          "-o", "short-iso-precise", "--no-pager"], capture_output=True, text=True).stdout
    runs, cur, trig, last_exit = [], None, None, None
    for line in out.splitlines():
        try:
            ts = D.datetime.fromisoformat(line[:32])
        except ValueError:
            continue
        msg = line[33:]
        kind = next((k for pat, k in MARKS if pat in msg), None)
        if kind == "trigger":
            trig = (ts, next(p for p, k in MARKS if p in msg))
        elif kind == "exit":
            last_exit = ts
        elif kind == "started":
            cur = {"started": ts, "trigger": trig, "exit": last_exit if trig and last_exit and last_exit >= trig[0] else None,
                   "detail": []}
            runs.append(cur)
            trig = None
        elif cur is not None and kind and kind not in cur:
            cur[kind] = ts
        m = DETAIL.search(msg)
        if cur is not None and m:  # teardown lines land on the run that is ending
            cur["detail"].append(f"{ts:%H:%M:%S} {m.group(1)[:200]}")

    def d(r, a, b):
        return f"{(r[b] - r[a]).total_seconds():6.1f}" if a in r and b in r and r[a] and r[b] else "     -"

    print("started        trigger        teardown  gap start->LW weights post+engram ->ready ->index  ->warm  ->http "
          "start->http outage")
    for r in runs:
        t = r["trigger"]
        r["t0"] = t[0] if t else None
        outage = f"{(r['http'] - t[0]).total_seconds():6.0f}" if t and "http" in r else "     -"
        print(f"{r['started']:%m-%d %H:%M:%S} {(t[1] if t else '-')[:14]:14s} {d(r, 't0', 'exit')} {d(r, 'exit', 'started')[1:]}"
              f" {d(r, 'started', 'lw')}    {d(r, 'lw', 'fin')}  {d(r, 'fin', 'lwend')}     {d(r, 'lwend', 'ready')}"
              f" {d(r, 'ready', 'pcache')} {d(r, 'pcache', 'warm')} {d(r, 'warm', 'http')}  {d(r, 'started', 'http')}"
              f"      {outage}")
        for x in r["detail"]:
            print("      " + x)


if __name__ == "__main__":
    main()
