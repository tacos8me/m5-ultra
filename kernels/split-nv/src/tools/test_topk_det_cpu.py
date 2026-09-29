"""CPU model of the DS-V4 top-k v2 selection (topk_impl.cuh) and of the topk-det overflow path.

Checks, on distributions that do and do not overflow the threshold coarse bin (> kMaxNumTie = 2048 candidates):
  1. overflow_select, emulated round by round exactly as hooks/split_nv/kernels/topk_impl_det.cuh writes it, returns
     the exact top-k by (value key desc, index asc) -- the order radix_tie_select already uses;
  2. without overflow the in-tree kernel (and so the unchanged topk-det code path) already returns that exact set,
     so topk-det cannot change any row whose selection is deterministic today;
  3. with overflow the in-tree kernel's result depends on arrival order (two orders -> two sets), and topk-det's does
     not;
  4. the 12-bit (prefill ragged / Register / Streaming) and 10-bit (step Cluster) paths agree after the fix.
usage: python3 test_topk_det_cpu.py   (numpy only, no GPU)
"""
import sys

import numpy as np

K_MAX_TIE = 2048
SMALL_TIE = 128  # handle_tie's warp paths (float compare) up to 4 * 32 ties


def key32(v):
    b = v.astype(np.float32).view(np.uint32)
    return np.where(b & 0x80000000, ~b, b | 0x80000000).astype(np.uint32)


def coarse_bin(v, bits):
    h = v.astype(np.float32).astype(np.float16).view(np.uint16)
    k = np.where(h & 0x8000, ~h, h | 0x8000).astype(np.uint32) & 0xFFFF
    return k >> (16 - bits)


def reference(v, k):
    """exact top-k: value key descending, index ascending (NaN excluded like the kernel's comparisons)."""
    idx = np.arange(v.size)
    ok = ~np.isnan(v)
    order = np.lexsort((idx[ok], -key32(v[ok]).astype(np.int64)))
    return np.sort(idx[ok][order][:k])


def overflow_select(v, cand, remain):
    """TopKConfig::overflow_select, round by round (cand: indices of the threshold-bin candidates)."""
    keys = (key32(v[cand]).astype(np.uint64) << np.uint64(32)) | (~cand.astype(np.uint32)).astype(np.uint64)
    prefix = mask = np.uint64(0)
    out, want = [], remain
    for rnd in range(8):
        shift = np.uint64(56 - 8 * rnd)
        m = (keys & mask) == prefix
        d = ((keys[m] >> shift) & np.uint64(0xFF)).astype(np.int64)
        hist = np.bincount(d, minlength=256)
        above = hist.sum() - np.cumsum(hist)
        hits = np.nonzero((above < remain) & (above + hist >= remain))[0]
        assert hits.size == 1, "threshold bin not unique"
        thr = int(hits[0])
        remain -= int(above[thr])
        take = remain == hist[thr]
        if above[thr] > 0 or take:
            sel = (d > thr) | (take & (d == thr))
            out.append(cand[m][sel])
        if take:
            break
        prefix |= np.uint64(thr) << shift
        mask |= np.uint64(0xFF) << shift
    else:
        raise AssertionError("no termination")
    out = np.concatenate(out)
    assert out.size == want and np.unique(out).size == want
    return out


def handle_tie(v, ties, remain):
    if ties.size <= remain:
        return ties
    if ties.size <= SMALL_TIE:  # is_greater: float compare, then lower index
        order = sorted(ties.tolist(), key=lambda i: (-float(v[i]), i))
        return np.asarray(order[:remain])
    keys = (key32(v[ties]).astype(np.uint64) << np.uint64(32)) | (~ties.astype(np.uint32)).astype(np.uint64)
    return ties[np.argsort(keys)[::-1][:remain]]  # radix_tie_select: (value key, ~idx) keys are unique


def kernel(v, k, bits, det, arrival_seed=None):
    """One row through find_threshold -> collect -> handle_tie (or overflow_select when det)."""
    n = v.size
    if n <= k:
        return np.arange(n), 0
    bins = coarse_bin(v, bits)
    valid = ~np.isnan(v)  # NaN: every classification comparison is false in the kernel's collect
    counts = np.bincount(bins, minlength=1 << bits)  # the histogram counts NaN bins too
    above = n - np.cumsum(counts)
    thr = int(np.nonzero((above < k) & (above + counts >= k))[0][0])
    idx = np.arange(n)
    gt = idx[(bins > thr) & valid]
    ties = idx[(bins == thr) & valid]
    remain = k - gt.size
    if ties.size > K_MAX_TIE and remain > 0:
        if det:
            sel = overflow_select(v, ties, remain)
        else:  # the in-tree kernel: first K_MAX_TIE arrivals (arrival order = a thread-timing permutation)
            rng = np.random.default_rng(arrival_seed)
            sel = handle_tie(v, np.sort(rng.permutation(ties)[:K_MAX_TIE]), remain)
    else:
        sel = handle_tie(v, ties, remain)
    return np.sort(np.concatenate([gt, sel])), ties.size


def dists(rng):
    n = 20000
    yield "wide", rng.standard_normal(n).astype(np.float32)
    yield "narrow", (10 + 0.3 * rng.standard_normal(n)).astype(np.float32)
    yield "tight", (10 + 0.01 * rng.standard_normal(n)).astype(np.float32)
    yield "zeros97", np.where(rng.random(n) < 0.97, 0, rng.random(n)).astype(np.float32)
    yield "signed_zeros", np.where(rng.random(n) < 0.5, np.float32(0), np.float32(-0.0)) + np.where(
        rng.random(n) < 0.01, rng.random(n), 0).astype(np.float32)
    grp = rng.standard_normal(40).astype(np.float32)
    yield "dup_groups", grp[rng.integers(0, 40, n)]
    yield "one_value", np.full(n, 3.25, np.float32)
    yield "neg_inf_mix", np.where(rng.random(n) < 0.9, -np.inf, rng.standard_normal(n)).astype(np.float32)
    yield "fp16_edge", (65500 + 30 * rng.random(n)).astype(np.float32)
    yield "tiny", (1e-30 * rng.random(n)).astype(np.float32)
    yield "repeat_period", np.tile((5 + rng.standard_normal(37)).astype(np.float32), n // 37 + 1)[:n] + (
        1e-4 * rng.standard_normal(n)).astype(np.float32) * (rng.random(n) < 0.5)
    yield "nan_some", np.where(rng.random(n) < 0.01, np.nan, 10 + 0.01 * rng.standard_normal(n)).astype(np.float32)
    big = 262144
    yield "long_narrow", (7 + 0.2 * rng.standard_normal(big)).astype(np.float32)
    yield "long_repeat", np.tile((5 + rng.standard_normal(61)).astype(np.float32), big // 61 + 1)[:big]


def main():
    rng = np.random.default_rng(0)
    fails, rows = 0, []
    for name, v in dists(rng):
        for k in (512, 2048, 1, 100):
            ref = reference(v, k)
            for bits in (12, 10):
                det, nties = kernel(v, k, bits, det=True)
                a, _ = kernel(v, k, bits, det=False, arrival_seed=1)
                b, _ = kernel(v, k, bits, det=False, arrival_seed=2)
                overflow = nties > K_MAX_TIE
                ok_det = np.array_equal(det, ref)
                if name == "nan_some" and not overflow:
                    # NaN scores: the histogram counts them, collect drops them, so a NaN-holding threshold bin
                    # underfills (in-tree quirk, unchanged by topk-det: same code, same result as the in-tree model)
                    ok_det = np.array_equal(det, a) and np.array_equal(det, b)
                    ref = det
                # without overflow the in-tree kernel is exact (so topk-det, which runs the same code, is unchanged)
                ok_old = overflow or (np.array_equal(a, ref) and np.array_equal(b, ref))
                # signed zeros: the <=128-tie warp paths compare floats (-0 == +0) where the radix paths order keys
                if name == "signed_zeros" and not ok_old and nties <= SMALL_TIE:
                    ok_old = True
                old_varies = overflow and not np.array_equal(a, b)
                rows.append((name, k, bits, nties, overflow, ok_det, ok_old, old_varies))
                fails += (not ok_det) + (not ok_old)
    print(f"{'dist':14s} {'k':>5s} {'bits':>4s} {'bin':>7s} overflow det==ref old==ref(no ovf) old_varies")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:5d} {r[2]:4d} {r[3]:7d} {str(r[4]):8s} {str(r[5]):7s} {str(r[6]):15s} {r[7]}")
    n_ovf = sum(r[4] for r in rows)
    n_var = sum(r[7] for r in rows)
    print(f"\ncases {len(rows)}, overflow {n_ovf} (in-tree result arrival-dependent in {n_var}), failures {fails}")
    # prefill (12-bit) == step (10-bit) after the fix: both equal the reference above; spell it out once more
    v = np.tile(np.float32([1.0, 1.01, 0.99, 1.005]), 50000)
    s12, n12 = kernel(v, 512, 12, True)
    s10, n10 = kernel(v, 512, 10, True)
    same = np.array_equal(s12, s10) and np.array_equal(s12, reference(v, 512))
    print(f"prefill(12-bit, bin {n12}) == step(10-bit, bin {n10}) == reference: {same}")
    fails += not same
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
