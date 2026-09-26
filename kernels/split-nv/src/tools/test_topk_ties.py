"""Fused top-k v2 with many exact ties at the threshold: the kept set must be (value desc, index asc) and identical
on every run (the radix tie path used to hand the last slots to equal values in atomic arrival order)."""
import torch
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

torch.manual_seed(0)
rows, n, k = 64, 20000, 512
bad = 0
for n_ties, n_above in ((300, 400), (1500, 100), (129, 450), (60, 480)):
    scores = torch.rand(rows, n, device="cuda") * 0.5
    expect = []
    for r in range(rows):
        perm = torch.randperm(n, device="cuda")
        scores[r, perm[:n_above]] = 5.0 + torch.rand(n_above, device="cuda")
        ties = perm[n_above:n_above + n_ties]
        scores[r, ties] = 1.0
        keep = torch.sort(ties).values[:k - n_above]
        expect.append(set(perm[:n_above].tolist()) | set(keep.tolist()))
    lens = torch.full((rows,), n, dtype=torch.int32, device="cuda")
    offs = torch.zeros(rows, dtype=torch.int32, device="cuda")
    outs = []
    for _ in range(30):
        out = torch.empty(rows, k, dtype=torch.int32, device="cuda")
        topk_transform_ragged_v2(scores, lens, out_offsets=offs, out_indices=out)
        outs.append(torch.sort(out, dim=-1).values)
    same = all(torch.equal(outs[0], o) for o in outs)
    right = sum(set(outs[0][r].tolist()) == expect[r] for r in range(rows))
    print({"ties": n_ties, "above": n_above, "deterministic_30_runs": same, "rows_correct": f"{right}/{rows}"})
    bad += (not same) + (right != rows)
raise SystemExit(bad)
