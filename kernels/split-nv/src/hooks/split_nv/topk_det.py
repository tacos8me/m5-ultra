"""Exact top-k when the indexer's threshold bin overflows (DS-V4 top-k v2, prefill and steps).

The fused top-k (SGLang kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh) buckets scores by
a coarse fp16 key (12 bits in the Register/Streaming paths, 10 in the Cluster path), finds the bin
holding the k-th score, and keeps at most kMaxNumTie = 2048 of that bin's candidates, in
shared-memory atomic arrival order. With more than 2048 candidates in the bin the kept set is
inexact and changes run to run, and the prefill (ragged, 12-bit) and step (paged, 10-bit Cluster
above 32K positions) paths can pick different sets for the same row.

kernels/topk_{v2,impl}_det.cuh are the in-tree sources (a618602) with one change: when the bin
holds more than kMaxNumTie candidates, an exact radix select over every candidate's
(value, lowest index) key replaces the truncated tie buffer (TopKConfig::overflow_select). Any
input with at most kMaxNumTie candidates takes the unchanged code, so its selection is the same
bytes as before. The module is swapped in behind sglang's own entry points (plan, paged, ragged),
so every caller -- including consistent.sorted_topk -- uses it.
"""
import hashlib
import os
import pathlib

KERNELS = pathlib.Path(__file__).resolve().with_name("kernels")
SOURCES = ("topk_v2_det.cuh", "topk_impl_det.cuh")


def enabled():
    return os.environ.get("SPLIT_NV_TOPK_DET", "1") == "1"


def digest():
    # sglang's JIT cache key hashes the .cuh it compiles but not the headers that file includes
    h = hashlib.sha256()
    for name in SOURCES:
        h.update((KERNELS / name).read_bytes())
    return h.hexdigest()[:16]


_module = None


def module():
    global _module
    if _module is None:
        from sglang.kernels.jit.utils import load_jit

        _module = load_jit(
            "dpsk_v4_topk_v2_det",
            digest(),
            cuda_files=[str(KERNELS / "topk_v2_det.cuh")],
            cuda_wrappers=[
                ("topk_transform_paged", "topk_det::TopKKernel::transform_paged"),
                ("topk_transform_ragged", "topk_det::TopKKernel::transform_ragged"),
                ("topk_plan", "topk_det::TopKKernel::plan"),
            ],
        )
    return _module


_installed = False


def installed():
    return _installed


def install():
    global _installed
    from sglang.kernels.ops.attention.dsv4 import topk as T

    if _installed or not enabled():
        return
    T._jit_topk_v2_module = module
    _installed = True
    print(f"[split-nv] topk-det installed ({digest()})", flush=True)


# ---- audit (diagnostic, SPLIT_NV_TOPK_AUDIT=1; maintenance windows only) -----------------------------------------
# Counts prefill rows whose threshold bin overflows, from the logits themselves, for the 12-bit bins the ragged
# prefill kernel uses and the 10-bit bins a step's Cluster path would use on the same scores. Prefill only: it
# syncs the host, so it cannot sit inside a captured step graph. Prints one line per ragged call that overflows.

AUDIT = {"calls": 0, "rows": 0, "ovf12": 0, "ovf10": 0, "max_bin12": 0, "max_bin10": 0}


def overflow_counts(scores, lens, k, bits, max_tie=2048):
    """(rows whose threshold coarse bin holds > max_tie candidates, largest bin) for scores[r, :lens[r]]."""
    import torch

    rows, width = scores.shape
    chunk = max(1, (8 << 20) // max(1, width))  # ~32 MB per int32 temporary
    cols = torch.arange(width, device=scores.device)
    n_over, biggest = 0, 0
    for r0 in range(0, rows, chunk):
        s, n = scores[r0:r0 + chunk], lens[r0:r0 + chunk].to(torch.int64)
        h = s.half().view(torch.int16).to(torch.int32) & 0xFFFF
        key = torch.where(h >= 0x8000, 0xFFFF - h, h | 0x8000) >> (16 - bits)
        key = key.masked_fill(cols[None, :] >= n[:, None], -1)
        live = n > k
        if not bool(live.any()):
            continue
        kth = key.topk(k, dim=-1).values[:, -1]
        cnt = ((key == kth[:, None]) & live[:, None]).sum(-1)
        n_over += int((cnt > max_tie).sum())
        biggest = max(biggest, int(cnt.max()))
    return n_over, biggest


def install_audit():
    if os.environ.get("SPLIT_NV_TOPK_AUDIT") != "1":
        return
    from sglang.srt.layers.attention import deepseek_v4_backend as B

    original = B.topk_transform_ragged_v2

    def audited(scores, seq_lens, *, out_offsets, out_indices, row_starts=None):
        k = out_indices.shape[1]
        if row_starts is None:
            o12, b12 = overflow_counts(scores, seq_lens, k, 12)
            o10, b10 = overflow_counts(scores, seq_lens, k, 10)
            a = AUDIT
            a["calls"] += 1
            a["rows"] += scores.shape[0]
            a["ovf12"] += o12
            a["ovf10"] += o10
            a["max_bin12"] = max(a["max_bin12"], b12)
            a["max_bin10"] = max(a["max_bin10"], b10)
            if o12 or o10:
                print(f"[split-nv] topk-audit rows={scores.shape[0]} width={scores.shape[1]} "
                      f"max_len={int(seq_lens.max())} overflow12={o12} (max bin {b12}) "
                      f"overflow10={o10} (max bin {b10}) total={a}", flush=True)
        return original(scores, seq_lens, out_offsets=out_offsets, out_indices=out_indices, row_starts=row_starts)

    B.topk_transform_ragged_v2 = audited
    print("[split-nv] topk-audit installed (prefill ragged top-k)", flush=True)
