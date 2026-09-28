import sys, re, collections, numpy as np
sys.path.insert(0, '/mnt/nvme-1/split-nv-ops/box-perf/prof')
import nsys_an as N
tr = N.Trace(sys.argv[1])
rng = N.cmd_ranges(tr, "step")
# pair rank0/rank1 ranges by text
by = collections.defaultdict(list)
for s, e, tid, text in rng:
    by[text].append((s, e, tid))
launch_ids = {i for i, v in tr.strings.items() if v.startswith("cudaGraphLaunch")}
rows = []
for text, rs in by.items():
    if len(rs) != 2:
        continue
    info = []
    for s, e, tid in rs:
        evs, ks, cps, sets, rts, dev = N.step_breakdown(tr, s, e, tid)
        gl = [r for r in rts if r[4] in launch_ids]
        gk = [k for k in ks if k[9] is not None]
        if not gl or not gk:
            break
        ar = [k for k in gk if 'cross_device' in tr.name(k)]
        info.append(dict(dev=dev, s=s, e=e, launch=gl[0][0], launch_end=gl[0][1], g0=gk[0][0], g1=gk[-1][1],
                         ar0=(ar[0][1] - ar[0][0]) if ar else 0, ar0_end=ar[0][1] if ar else 0))
    if len(info) != 2:
        continue
    a, b = sorted(info, key=lambda x: x['dev'])
    L = int(re.search(r"L=(\d+)", text).group(1)); keep = int(re.search(r"keep=(\d+)", text).group(1))
    rows.append((a['s'], L, keep, (b['s'] - a['s']) / 1e3, (a['launch'] - a['s']) / 1e3, (b['launch'] - b['s']) / 1e3,
                 (b['g0'] - a['g0']) / 1e3, a['ar0'] / 1e3, b['ar0'] / 1e3, (a['g1'] - a['g0']) / 1e3, (b['g1'] - b['g0']) / 1e3,
                 (a['e'] - a['s']) / 1e3, (a['launch_end'] - a['launch']) / 1e3))
rows.sort()
prev_end = None
print("cols: L keep | r1 cmd start lag | r0 prep->launch | r1 prep->launch | r1-r0 graph start | AR0 r0 | AR0 r1 | span r0 | span r1 | wall r0 | launchAPI r0 | idle-before")
groups = collections.defaultdict(list)
last_t = None
for r in rows:
    idle = (r[0] - last_t) / 1e6 if last_t else 0
    last_t = r[0]
    ctx = '8k' if r[2] < 100000 else '128k'
    tight = idle < 15
    groups[(ctx, r[1], 'tight' if tight else 'gap')].append(r[3:])
for k in sorted(groups):
    v = np.array(groups[k])
    med = np.median(v, axis=0)
    print(k, len(v), " ".join(f"{x:7.1f}" for x in med))
