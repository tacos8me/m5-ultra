"""GPU check of the topk-det module against the in-tree DS-V4 top-k v2 module (same image, same JIT flags).

  identity   inputs whose threshold bin holds <= 2048 candidates: topk-det == in-tree, byte for byte (sorted rows),
             on every path (ragged Register2/Register4/Streaming; paged Register/Streaming, small-batch Cluster,
             persistent Cluster + level-3 epilogue; INDICES and PAGE_TABLE modes), and both == the exact reference
  overflow   bins with > 2048 candidates: topk-det == exact reference (value desc, index asc) on every run;
             reports how often the in-tree module differs from itself / from the reference
  step==pf   the same rows through ragged (prefill) and paged INDICES (step, as consistent.sorted_topk calls it)
  timing     (TIMING=1) per-call time of both modules on non-overflow and overflow inputs

Small by default (every score tensor <= 64 MB, allocator capped at 1% of the GPU); FULL=1 adds production-size
shapes for a maintenance window. Data and references are built on the CPU before CUDA is initialised.
usage (inside sglang-dsv41-split:6152b54, PYTHONPATH=<tree>/hooks): python3 test_topk_det_gpu.py
"""
import json
import os
import sys
import time

import numpy as np

K = 512
MAX_TIE = 2048
FULL = os.environ.get("FULL") == "1"
RUNS = int(os.environ.get("RUNS", "10"))


def key64(v):
    b = v.view(np.uint32)
    k32 = np.where(b & 0x80000000, ~b, b | 0x80000000).astype(np.uint64)
    idx = np.arange(v.shape[-1], dtype=np.uint64)
    return (k32 << np.uint64(32)) | (~idx & np.uint64(0xFFFFFFFF))


def reference(v, lens, k):
    out = np.full((v.shape[0], k), -1, np.int64)
    keys = key64(v)
    for r, n in enumerate(lens):
        if n <= k:
            out[r, :n] = np.arange(n)
            continue
        top = np.argpartition(keys[r, :n], n - k)[n - k:]
        out[r] = np.sort(top)
    return out


def bin_count(v, lens, k, bits):
    """Candidates in the threshold coarse bin of each row (the kernel's count_eq)."""
    h = v.astype(np.float16).view(np.uint16).astype(np.uint32)
    coarse = np.where(h & 0x8000, ~h & 0xFFFF, h | 0x8000) >> (16 - bits)
    out = np.zeros(v.shape[0], np.int64)
    for r, n in enumerate(lens):
        if n > k:
            c = coarse[r, :n]
            out[r] = int((c == np.partition(c, n - k)[n - k]).sum())
    return out


def make(dist, rows, width, rng):
    if dist == "wide":
        v = rng.standard_normal((rows, width))
    elif dist == "narrow":
        v = 10 + 0.3 * rng.standard_normal((rows, width))
    elif dist == "tight":
        v = 10 + 0.01 * rng.standard_normal((rows, width))
    elif dist == "zeros97":
        v = np.where(rng.random((rows, width)) < 0.97, 0, rng.random((rows, width)))
    elif dist == "dup":
        v = rng.standard_normal(64)[rng.integers(0, 64, (rows, width))]
    elif dist == "period":  # repeated block of 97 scores + tiny jitter on half of them
        base = 5 + rng.standard_normal(97)
        v = np.tile(base, width // 97 + 1)[:width][None].repeat(rows, 0)
        v = v + 1e-5 * rng.standard_normal((rows, width)) * (rng.random((rows, width)) < 0.5)
    elif dist == "one":
        v = np.full((rows, width), 2.5)
    else:
        raise ValueError(dist)
    return np.ascontiguousarray(v.astype(np.float32))


def cases(rng):
    # (label, path, dist, rows, width, varlen)
    ragged = [(3000, 64), (8192, 64), (12000, 64), (16384, 64), (40000, 64), (200000, 16)]
    paged = [(1, 8000), (4, 16000), (4, 40000), (2, 200000), (8, 70000), (20, 70000), (20, 200000)]
    if FULL:
        ragged += [(131072, 256), (262144, 64)]
        paged += [(5, 524288), (40, 131072)]
    for dist in ("wide", "narrow", "tight", "zeros97", "dup", "period", "one"):
        for width, rows in ragged:
            yield ("ragged", dist, rows, width)
        for rows, width in paged:
            yield ("paged", dist, rows, width)
            yield ("paged_pt", dist, rows, width)


def main():
    rng = np.random.default_rng(0)
    plan = []
    for path, dist, rows, width in cases(rng):
        v = make(dist, rows, width, rng)
        lens = np.full(rows, width, np.int64)
        if rows > 1:
            lens[1::2] = rng.integers(width * 3 // 4, width + 1, rows // 2)
        # 12-bit bins everywhere except the paged Cluster rows (10-bit); which paged rows go to the cluster depends
        # on the plan, so a row counts as overflowing if either binning overflows
        bins = bin_count(v, lens, K, 12)
        if path != "ragged":
            bins = np.maximum(bins, bin_count(v, lens, K, 10))
        plan.append(dict(path=path, dist=dist, rows=rows, width=width, v=v, lens=lens,
                         ref=reference(v, lens, K), bins=bins))
    print(f"built {len(plan)} cases on CPU", flush=True)

    import torch

    torch.cuda.set_per_process_memory_fraction(0.01)
    from sglang.kernels.ops.attention.dsv4 import topk as T
    from split_nv import topk_det

    intree, det = T._jit_topk_v2_module, topk_det.module
    dev = torch.device("cuda")

    def run(mod, c, s, lens_t):
        T._jit_topk_v2_module = mod
        try:
            out = torch.empty(c["rows"], K, dtype=torch.int32, device=dev)
            if c["path"] == "ragged":
                T.topk_transform_ragged_v2(s, lens_t, out_offsets=torch.zeros_like(lens_t), out_indices=out)
            else:
                meta = T.plan_topk_v2(lens_t)
                pt = None
                if c["path"] == "paged_pt":  # identity page table: page-transformed index == raw index
                    pt = torch.arange((c["width"] + 63) // 64, dtype=torch.int32, device=dev)[None].repeat(c["rows"], 1)
                T.topk_transform_paged_v2(s, lens_t, pt, out, 64, meta)
            return out
        finally:
            T._jit_topk_v2_module = intree

    def canon(out):
        o = out.to(torch.int64).masked_fill(out < 0, 1 << 40).sort(-1).values
        return o.masked_fill(o == 1 << 40, -1).cpu().numpy()

    results, bad = [], 0
    for c in plan:
        s = torch.from_numpy(c["v"]).to(dev)
        lens_t = torch.from_numpy(c["lens"].astype(np.int32)).to(dev)
        d_runs = [canon(run(det, c, s, lens_t)) for _ in range(RUNS)]
        i_runs = [canon(run(intree, c, s, lens_t)) for _ in range(RUNS)]
        torch.cuda.synchronize()
        ovf = c["bins"] > MAX_TIE
        det_ok = all(np.array_equal(d, c["ref"]) for d in d_runs)
        det_stable = all(np.array_equal(d, d_runs[0]) for d in d_runs)
        # identity: rows without overflow must match the in-tree module on every run
        same_rows = ~ovf
        ident = all(np.array_equal(d[same_rows], i[same_rows]) for d in d_runs for i in i_runs)
        intree_wrong = sum(int((i[ovf] != c["ref"][ovf]).any(-1).sum()) for i in i_runs)
        intree_varies = sum(int(not np.array_equal(i, i_runs[0])) for i in i_runs)
        res = dict(path=c["path"], dist=c["dist"], rows=c["rows"], width=c["width"],
                   overflow_rows=int(ovf.sum()), max_bin=int(c["bins"].max()), det_exact=det_ok,
                   det_stable=det_stable, identity_non_overflow=ident, intree_wrong_rows=intree_wrong,
                   intree_runs_differing=intree_varies)
        bad += (not det_ok) + (not det_stable) + (not ident)
        results.append(res)
        print(json.dumps(res), flush=True)
        del s, lens_t

    # step == prefill: the same rows through ragged and paged INDICES (topk-det)
    for dist in ("narrow", "period", "one", "wide"):
        for width, rows in ((12000, 8), (50000, 4)):
            v = make(dist, rows, width, rng)
            s = torch.from_numpy(v).to(dev)
            lens_t = torch.full((rows,), width, dtype=torch.int32, device=dev)
            a = canon(run(det, dict(path="ragged", rows=rows, width=width), s, lens_t))
            b = canon(run(det, dict(path="paged", rows=rows, width=width), s, lens_t))
            ok = np.array_equal(a, b) and np.array_equal(a, reference(v, [width] * rows, K))
            bad += not ok
            print(json.dumps(dict(check="step==prefill", dist=dist, rows=rows, width=width, equal_and_exact=ok)))

    if os.environ.get("TIMING") == "1":
        for dist in ("wide", "period"):
            shapes = [("ragged", 64, 16384), ("ragged", 16, 200000), ("paged", 4, 200000)]
            if FULL:  # a prefill tile at 128K compressed positions and a 5-row step at 1M tokens
                shapes += [("ragged", 1024, 131072), ("paged", 5, 524288)]
            for path, rows, width in shapes:
                c = dict(path=path, rows=rows, width=width)
                s = torch.from_numpy(make(dist, rows, width, rng)).to(dev)
                lens_t = torch.full((rows,), width, dtype=torch.int32, device=dev)
                t = {}
                for name, mod in (("intree", intree), ("det", det)):
                    run(mod, c, s, lens_t)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(20):
                        run(mod, c, s, lens_t)
                    torch.cuda.synchronize()
                    t[name] = round((time.perf_counter() - t0) / 20 * 1e3, 3)
                print(json.dumps(dict(timing_ms=t, dist=dist, path=path, rows=rows, width=width)))

    print(json.dumps(dict(summary=True, cases=len(results), failures=bad,
                          overflow_cases=sum(r["overflow_rows"] > 0 for r in results),
                          intree_wrong_rows=sum(r["intree_wrong_rows"] for r in results),
                          peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1))))
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
