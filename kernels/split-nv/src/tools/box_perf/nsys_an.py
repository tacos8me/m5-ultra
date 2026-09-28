"""Analysis of the split-nv engine nsys exports (sqlite).

  nsys_an.py FILE.sqlite steps  [--kind step|prefill_chunk] [--csv OUT]
  nsys_an.py FILE.sqlite detail --sid S [--L L] [--nth k]
Kernel -> module attribution: graph kernels via graphNodeId -> capture node -> NVTX stack at capture time
(nvtx-precapture); eager kernels via correlationId -> runtime API call -> NVTX stack at call time.
"""
import argparse
import bisect
import collections
import json
import re
import sqlite3
import sys

import numpy as np


class Trace:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = None
        q = self.q
        self.strings = dict(q("select id, value from StringIds"))
        # processes / devices
        self.kern = q("""select start, end, deviceId, streamId, correlationId, globalPid, shortName, demangledName,
                         graphNodeId, graphId, gridX*gridY*gridZ, blockX*blockY*blockZ from CUPTI_ACTIVITY_KIND_KERNEL order by start""")
        self.memcpy = q("select start, end, deviceId, streamId, correlationId, bytes, copyKind, graphNodeId from CUPTI_ACTIVITY_KIND_MEMCPY order by start")
        try:
            self.memset = q("select start, end, deviceId, streamId, correlationId, bytes, graphNodeId from CUPTI_ACTIVITY_KIND_MEMSET order by start")
        except sqlite3.OperationalError:
            self.memset = []
        self.rt = q("select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")
        self.nvtx = q("""select n.start, n.end, n.globalTid, coalesce(n.text, s.value) from NVTX_EVENTS n
                         left join StringIds s on n.textId = s.id where n.end is not null order by n.start""")
        self.node = q("select start, globalTid, graphNodeId, originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS")
        self.rt_by_corr = {r[3]: r for r in self.rt}
        self.nvtx_by_tid = collections.defaultdict(list)
        for s, e, tid, text in self.nvtx:
            self.nvtx_by_tid[tid].append((s, e, text))
        self.node_orig = {}
        self.node_birth = {}
        for s, tid, nid, orig in self.node:
            if orig is not None:
                self.node_orig[nid] = orig
            elif nid not in self.node_birth:
                self.node_birth[nid] = (s, tid)
        self._stack_cache = {}

    def q(self, sql):
        try:
            return self.db.execute(sql).fetchall()
        except sqlite3.OperationalError as e:
            print(f"warning: {e}", file=sys.stderr)
            return []

    def stack_at(self, tid, t):
        """Enclosing NVTX ranges (outermost first) on thread tid at time t."""
        return [text for s, e, text in self.nvtx_by_tid.get(tid, ()) if s <= t <= e]

    def kernel_stack(self, k):
        nid, corr = k[8], k[4]
        key = ("g", nid) if nid is not None else ("c", corr)
        if key in self._stack_cache:
            return self._stack_cache[key]
        st = []
        if nid is not None:
            orig = self.node_orig.get(nid, nid)
            b = self.node_birth.get(orig)
            if b is not None:
                st = self.stack_at(b[1], b[0])
        else:
            r = self.rt_by_corr.get(corr)
            if r is not None:
                st = self.stack_at(r[2], r[0])
        self._stack_cache[key] = st
        return st

    def name(self, k):
        return self.strings.get(k[6], "?")


CATS = [
    ("allreduce", r"allreduce|all_reduce|cross_device|AllReduce|nccl"),
    ("moe", r"og_moe|og::|moe_|Moe|fused_moe|expert"),
    ("attention", r"flash|mla|Prefill|prefill_|attention|BatchPrefill|sparse_attn"),
    ("indexer", r"mqa_logits|fp8_fp4|indexer|topk|radix|Topk|TopK"),
    ("gemm", r"gemm|Gemm|GEMM|gemv|cutlass|b12x|mxfp8|sm120_block|Kernel2|cublas|sgemm|nvjet|matmul"),
    ("mhc", r"hc_|sinkhorn|mhc"),
    ("norm_rope", r"rms|norm|rope|Rope"),
    ("elementwise", r"elementwise|vectorized|unrolled|copy|Copy|fill|index|cat|reduce_kernel|arange|where|gather|scatter"),
]


def category(name, stack):
    path = "/".join(stack)
    if re.search(r"moe_ffn|m:mlp", path) and not re.search(CATS[0][1], name):
        return "moe"
    for cat, rx in CATS:
        if re.search(rx, name):
            return cat
    return "other"


def module_of(stack):
    """Compact module path below the layer range, e.g. 'L3/attn.forward/be.forward'."""
    layer = next((s for s in stack if re.fullmatch(r"L\d+", s)), None)
    inner = [s for s in stack if not s.startswith(("cmd:", "sr.", "model")) and not re.fullmatch(r"L\d+", s)]
    return layer, "/".join(inner[-3:])


def cmd_ranges(tr, kind):
    out = []
    for tid, evs in tr.nvtx_by_tid.items():
        for s, e, text in evs:
            if text and text.startswith("cmd:" + kind):
                out.append((s, e, tid, text))
    return sorted(out)


def union_busy(iv):
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


def step_breakdown(tr, s, e, tid):
    """One engine command range on one rank: host phases, GPU kernels, busy/idle."""
    evs = [(a, b, t) for a, b, t in tr.nvtx_by_tid[tid] if a >= s and b <= e]
    # device of this thread: from any runtime call's kernel
    ks = []
    dev = None
    corrs = set()
    for r in tr.rt[bisect.bisect_left(tr.rt, (s,)):bisect.bisect_right(tr.rt, (e + 1,))]:
        if r[2] == tid:
            corrs.add(r[3])
    i0 = bisect.bisect_left(tr.kern, (s,))
    i1 = bisect.bisect_right(tr.kern, (e + 1,))
    for k in tr.kern[i0:i1]:
        if k[4] in corrs:
            ks.append(k)
            dev = k[2]
    if dev is not None:
        ks = [k for k in tr.kern[i0:i1] if k[2] == dev and k[1] <= e + 1000]
    cps = [m for m in tr.memcpy[bisect.bisect_left(tr.memcpy, (s,)):bisect.bisect_right(tr.memcpy, (e + 1,))] if m[2] == dev]
    sets = [m for m in tr.memset[bisect.bisect_left(tr.memset, (s,)):bisect.bisect_right(tr.memset, (e + 1,))] if m[2] == dev]
    rts = [r for r in tr.rt[bisect.bisect_left(tr.rt, (s,)):bisect.bisect_right(tr.rt, (e + 1,))] if r[2] == tid]
    return evs, ks, cps, sets, rts, dev


def phase_times(evs, s):
    ph = collections.defaultdict(float)
    for a, b, t in evs:
        if t.startswith(("sr.", "be.init_forward_metadata_out_graph", "engram_fill")):
            ph[t] += (b - a) / 1e3
    return ph


def cmd_steps(tr, args):
    rows = []
    for s, e, tid, text in cmd_ranges(tr, args.kind):
        evs, ks, cps, sets, rts, dev = step_breakdown(tr, s, e, tid)
        if dev is None:
            continue
        gk = [k for k in ks if k[9] is not None]
        ek = [k for k in ks if k[9] is None]
        g0 = min((k[0] for k in gk), default=None)
        g1 = max((k[1] for k in gk), default=None)
        ksum = sum(k[1] - k[0] for k in ks) / 1e3
        busy = union_busy([(k[0], k[1]) for k in ks] + [(m[0], m[1]) for m in cps + sets]) / 1e3
        ph = phase_times(evs, s)
        launch = [r for r in rts if tr.strings.get(r[4], "").startswith("cudaGraphLaunch")]
        syncs = [r for r in rts if "Synchronize" in tr.strings.get(r[4], "")]
        rows.append(dict(
            text=text, dev=dev, t0=s, wall=(e - s) / 1e3, kernels=len(ks), graph_kernels=len(gk), eager_kernels=len(ek),
            ksum=ksum, busy=busy, idle=(e - s) / 1e3 - busy,
            graph_span=((g1 - g0) / 1e3) if gk else 0.0,
            graph_ksum=sum(k[1] - k[0] for k in gk) / 1e3,
            pre_graph=((g0 - s) / 1e3) if gk else 0.0, post_graph=((e - g1) / 1e3) if gk else 0.0,
            launch_api=sum(r[1] - r[0] for r in launch) / 1e3, sync_api=sum(r[1] - r[0] for r in syncs) / 1e3,
            memcpy=len(cps), memcpy_us=sum(m[1] - m[0] for m in cps) / 1e3,
            **{f"ph_{k}": v for k, v in ph.items()}))
    if args.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
    by = collections.defaultdict(list)
    for r in rows:
        m = re.search(r"L=(\d+)", r["text"])
        key = (r["dev"], m.group(1) if m else r["text"][:30], r["graph_kernels"] > 0)
        by[key].append(r)
    for key in sorted(by):
        rs = by[key]
        med = {k: float(np.median([r.get(k, 0.0) for r in rs])) for k in rs[0] if isinstance(rs[0][k], (int, float))}
        print(f"dev{key[0]} {key[1]:>4} graph={key[2]} n={len(rs):3d} wall {med['wall']:7.3f} ms  kernels {med['kernels']:.0f} "
              f"ksum {med['ksum']:7.3f}  busy {med['busy']:7.3f}  idle {med['idle']:6.3f}  gspan {med['graph_span']:6.3f} "
              f"pre {med['pre_graph']:6.3f} post {med['post_graph']:6.3f}  launch {med['launch_api']:.3f} sync {med['sync_api']:.3f}")
        phs = sorted(k for k in med if k.startswith("ph_"))
        print("        " + "  ".join(f"{k[3:]} {med[k]:.3f}" for k in phs))
    return rows


def step_kernels(tr, kind, dev, Lsel, graph=True, text_rx=None):
    """[(range text, [(k, stack)])] for each matching command range on device dev."""
    out = []
    for s, e, tid, text in cmd_ranges(tr, kind):
        if Lsel and f"L={Lsel}" not in text.split():
            continue
        if text_rx and not re.search(text_rx, text):
            continue
        evs, ks, cps, sets, rts, d = step_breakdown(tr, s, e, tid)
        if d != dev:
            continue
        if graph is not None and (any(k[9] is not None for k in ks) != graph):
            continue
        out.append((s, e, text, [(k, tr.kernel_stack(k)) for k in ks], cps, sets))
    return out


def cmd_layers(tr, a):
    steps = step_kernels(tr, a.kind, a.dev, a.L, None if a.kind != "step" else (not a.eager), a.rx)
    if a.skip:
        steps = steps[a.skip:]
    n = len(steps)
    print(f"{n} ranges ({a.kind} dev{a.dev} L={a.L})")
    if not n:
        return
    per_layer = collections.defaultdict(float)
    per_cat = collections.defaultdict(float)
    per_mod = collections.defaultdict(float)
    per_name = collections.defaultdict(lambda: [0.0, 0])
    gaps = collections.defaultdict(float)
    walls = []
    for s, e, text, ks, cps, sets in steps:
        walls.append((e - s) / 1e3)
        prev_end = None
        for k, st in ks:
            d = (k[1] - k[0]) / 1e3
            layer, mod = module_of(st)
            cat = category(tr.name(k), st)
            per_layer[layer or "-"] += d
            per_cat[cat] += d
            per_mod[(cat, mod)] += d
            nm = tr.name(k)
            per_name[(cat, nm)][0] += d
            per_name[(cat, nm)][1] += 1
            if prev_end is not None and k[0] > prev_end:
                gaps[(layer or "-")] += (k[0] - prev_end) / 1e3
            prev_end = max(prev_end or 0, k[1])
    print(f"wall median {np.median(walls):.3f} ms")
    tot = sum(per_cat.values()) / n
    print(f"kernel sum/range {tot:.1f} us")
    print("per category (us/range):")
    for c, v in sorted(per_cat.items(), key=lambda x: -x[1]):
        print(f"   {c:12s} {v / n:9.1f}  {100 * v / n / tot:5.1f}%")
    print("per layer (kernel us/range, gap-before us/range):")
    def lk(x):
        m = re.fullmatch(r"L(\d+)", x)
        return (0, int(m.group(1))) if m else (1, x)
    for l in sorted(per_layer, key=lk):
        print(f"   {l:5s} {per_layer[l] / n:8.1f}  gaps {gaps[l] / n:7.1f}")
    print("top kernels (us/range, launches/range):")
    for (c, nm), (v, cnt) in sorted(per_name.items(), key=lambda x: -x[1][0])[:a.top]:
        print(f"   {v / n:8.1f} {cnt / n:6.1f}  {c:10s} {nm[:110]}")
    print("top modules (us/range):")
    for (c, m), v in sorted(per_mod.items(), key=lambda x: -x[1])[:a.top]:
        print(f"   {v / n:8.1f}  {c:10s} {m[:120]}")


def cmd_detail(tr, a):
    steps = step_kernels(tr, a.kind, a.dev, a.L, None if a.kind != "step" else (not a.eager), a.rx)
    if not steps:
        print("none")
        return
    s, e, text, ks, cps, sets = steps[min(a.nth, len(steps) - 1)]
    print(f"{text} wall {(e - s) / 1e3:.3f} ms, {len(ks)} kernels, {len(cps)} memcpy, {len(sets)} memset")
    ev = sorted([(k[0], k[1], "K", tr.name(k), st) for k, st in ks] + [(m[0], m[1], "C", f"memcpy kind{m[6]} {m[5]}B", []) for m in cps]
                + [(m[0], m[1], "S", f"memset {m[5]}B", []) for m in sets])
    prev = s
    for t0, t1, typ, nm, st in ev:
        layer, mod = module_of(st)
        print(f"{(t0 - s) / 1e3:9.1f} {(t1 - t0) / 1e3:8.1f} gap {(t0 - prev) / 1e3:7.1f} {typ} {layer or '-':4s} {mod[:60]:60s} {nm[:70]}")
        prev = max(prev, t1)
    print(f"end gap {(e - prev) / 1e3:.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("mode")
    ap.add_argument("--kind", default="step")
    ap.add_argument("--csv")
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--L", default="")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--rx")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--nth", type=int, default=5)
    a = ap.parse_args()
    tr = Trace(a.db)
    if a.mode == "steps":
        cmd_steps(tr, a)
    elif a.mode == "layers":
        cmd_layers(tr, a)
    elif a.mode == "buckets":
        cmd_buckets(tr, a)
    elif a.mode == "detail":
        cmd_detail(tr, a)



def bucket(name, stack):
    p = "/".join(stack)
    layer = next((s for s in stack if re.fullmatch(r"L\d+", s)), None)
    if "cross_device" in name or "AllReduce" in name or "nccl" in name:
        return "allreduce"
    if layer == "L20":
        return "L20_sources"
    if name.startswith("decode_") or "og_moe" in name or "moe" in name.lower() and "m:mlp" in p and "_linear" not in name and "router" not in name:
        return "moe.experts"
    if "m:mlp" in p:
        return "moe.router"
    if name.startswith("sparse_mla"):
        return "attn.kernel"
    if "be._forward_attention" in p:
        return "attn.prep"
    if "engram" in p or "engram" in name:
        return "engram.wkv" if "engram.wkv" in p else "engram.other"
    if "m:embed_tokens" in p:
        return "embed"
    if name == "Kernel2":
        return "gemm.wo_a"
    for m in ("wqkv_a", "wq_b", "wo_b"):
        if f"self_attn.{m}" in p and "indexer" not in p:
            return f"gemm.{m}"
    if "low_ratio" in p or "cmp." in p or "idx." in p or "indexer" in p or "compressor" in p:
        return "compress_index"
    if "mhc" in p or name.startswith(("_hc_", "_mhc")):
        return "mhc"
    if "layernorm" in p or "q_norm" in p or "m:norm" in p:
        return "norm"
    if "init_forward_metadata" in p:
        return "metadata"
    if "_compute_q_b" in p or "_compute_kv" in p or "rope" in name or "m:self_attn" in p:
        return "attn.misc"
    return "other"


def cmd_buckets(tr, a):
    steps = step_kernels(tr, a.kind, a.dev, a.L, True, a.rx)[a.skip:]
    n = len(steps)
    if not n:
        print("none")
        return
    kt = collections.defaultdict(float)
    kn = collections.defaultdict(float)
    gp = collections.defaultdict(float)
    spans = []
    for s, e, text, ks, cps, sets in steps:
        gks = [(k, st) for k, st in ks if k[9] is not None]
        spans.append((gks[-1][0][1] - gks[0][0][0]) / 1e3)
        prev = None
        for k, st in gks:
            b = bucket(tr.name(k), st)
            kt[b] += (k[1] - k[0]) / 1e3
            kn[b] += 1
            if prev is not None and k[0] > prev:
                gp[b] += (k[0] - prev) / 1e3
            prev = max(prev or 0, k[1])
    tot_k, tot_g = sum(kt.values()) / n, sum(gp.values()) / n
    print(f"{a.rx or ''} dev{a.dev} L={a.L}: {n} steps, graph span median {np.median(spans):.1f} us, kernel {tot_k:.1f} + gaps {tot_g:.1f} us")
    for b in sorted(kt, key=lambda x: -kt[x]):
        print(f"   {b:15s} {kt[b] / n:8.1f} us  {kn[b] / n:6.1f} launches  gaps-before {gp[b] / n:6.1f} us")
    return kt, gp, n


if __name__ == "__main__":
    main()
