"""Per plan window (NVTX plan:<name>:<rep> from harness.py time): per device, wall, compute busy (union of non-NCCL
kernels), NCCL kernel time, NCCL time hidden under compute, exposed NCCL, idle; plus the top compute kernels.
usage: nsys_overlap.py TRACE.sqlite"""
import collections
import sqlite3
import sys


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def inter(a, b):
    """total length of (union a) intersect (union b)"""
    return union(a) + union(b) - union(a + b)


db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
ranges = db.execute("select start, end, text, globalTid from NVTX_EVENTS where text like 'plan:%'").fetchall()
kern = db.execute("select start, end, deviceId, shortName, streamId, globalPid from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
pid_dev = collections.Counter((k[5], k[2]) for k in kern)
dev_of_pid = {}
for (pid, dev), n in pid_dev.most_common():
    dev_of_pid.setdefault(pid, dev)
res = collections.defaultdict(list)
tops = collections.defaultdict(collections.Counter)
for s, e, text, gtid in ranges:
    pid = gtid >> 24 << 24 if False else None
    name = text.split(":")[1]
    for dev in (0, 1):
        ks = [(k[0], k[1], names[k[3]], k[4]) for k in kern if k[2] == dev and k[0] >= s and k[1] <= e]
        if not ks:
            continue
        nccl = [(a, b) for a, b, n, _ in ks if "nccl" in n.lower()]
        comp = [(a, b) for a, b, n, _ in ks if "nccl" not in n.lower()]
        wall = e - s
        res[(name, dev)].append((wall, union(comp), union(nccl), inter(comp, nccl), union(comp + nccl)))
        for a, b, n, _ in ks:
            tops[(name, dev)][n] += b - a
for key in sorted(res):
    v = res[key]
    n = len(v)
    w, c, nc, ov, busy = (sum(x[i] for x in v) / n / 1e6 for i in range(5))
    print(f"{key[0]:5s} dev{key[1]}: wall {w:7.1f} ms  compute busy {c:7.1f}  nccl {nc:6.1f}  nccl hidden {ov:6.1f} "
          f"({100 * ov / max(nc, 1e-9):4.1f}%)  exposed nccl {nc - ov:6.1f}  idle {w - busy:6.1f}")
for key in sorted(tops):
    if key[1] != 0:
        continue
    n = len(res[key])
    print(f"  top kernels {key[0]} dev0 (ms per window):")
    for kname, t in tops[key].most_common(14):
        print(f"    {t / n / 1e6:7.2f}  {kname[:90]}")
