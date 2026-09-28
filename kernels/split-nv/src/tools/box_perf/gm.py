import sys, re, collections, numpy as np, sqlite3
sys.path.insert(0, '/mnt/nvme-1/split-nv-ops/box-perf/prof')
import nsys_an as N
db = sys.argv[1]
tr = N.Trace(db)
con = sqlite3.connect(db)
types = sorted(t for (t,) in con.execute("select distinct typeId from GPU_METRICS"))
M = {}
for dev, t in enumerate(types):
    for mid in (0, 7, 19, 20):
        a = np.array(con.execute(f"select timestamp, value from GPU_METRICS where typeId={t} and metricId={mid} order by timestamp").fetchall(), dtype=np.float64)
        M[(dev, mid)] = a
def avg(dev, mid, t0, t1):
    a = M[(dev, mid)]
    i0, i1 = np.searchsorted(a[:, 0], [t0, t1])
    return a[i0:i1, 1].mean() if i1 > i0 else np.nan
# per-step clocks
rows = collections.defaultdict(list)
last = {}
for s, e, tid, text in N.cmd_ranges(tr, "step"):
    evs, ks, cps, sets, rts, dev = N.step_breakdown(tr, s, e, tid)
    gk = [k for k in ks if k[9] is not None]
    if not gk: continue
    g0, g1 = gk[0][0], gk[-1][1]
    idle = (s - last.get(dev, s)) / 1e6; last[dev] = e
    L = int(re.search(r"L=(\d+)", text).group(1))
    rows[(dev, 'tight' if idle < 12 else 'gap', L)].append((avg(dev, 0, s - 3e6, s), avg(dev, 0, g0, g0 + 5e5), avg(dev, 0, g0, g1), avg(dev, 19, g0, g1)))
print("GPC MHz: [3 ms before cmd] [first 0.5 ms of graph] [graph mean]  DRAM read % over graph")
for k in sorted(rows):
    v = np.nanmedian(np.array(rows[k]), axis=0)
    print(k, len(rows[k]), " ".join(f"{x:8.1f}" for x in v))
# DRAM % per kernel class, 8K L5 graph steps dev0/dev1
for dev in (0, 1):
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for s, e, text, ks, cps, sets in N.step_kernels(tr, "step", dev, "5", True, r"keep=8[0-9]{3} "):
        for k, st in ks:
            if k[9] is None: continue
            d = (k[1] - k[0])
            if d < 20000: 
                b = N.bucket(tr.name(k), st)
                acc[b + "(<20us)"][0] += d; acc[b + "(<20us)"][1] += d * np.nan_to_num(avg(dev, 19, k[0], k[1])); acc[b+"(<20us)"][2] += 1
                continue
            b = N.bucket(tr.name(k), st)
            r = avg(dev, 19, k[0], k[1]); c = avg(dev, 0, k[0], k[1])
            acc[b][0] += d; acc[b][1] += d * np.nan_to_num(r); acc[b][2] += d * np.nan_to_num(c)
    print(f"dev{dev}: DRAM read % (time-weighted) and GPC clock per bucket, kernels >= 20 us")
    for b, (d, r, c) in sorted(acc.items(), key=lambda x: -x[1][0]):
        if "(<20us)" in b: continue
        print(f"   {b:15s} {d/1e3/60:8.1f} us/step  DRAM rd {r/d:5.1f} %  clk {c/d:6.0f} MHz")
