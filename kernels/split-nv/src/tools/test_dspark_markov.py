"""CPU test of split_nv.dspark_markov (the fused Markov chain) through Triton's interpreter, two emulated TP ranks.

Checks: tokens == the full-vocab reference chain (greedy, BF16 bias, lowest index on ties); max-probs within 2e-6
relative; both ranks bitwise identical (tokens and probs); ties resolved to the lowest vocab index inside a block,
across blocks and across the rank boundary; a padded shard tail (v_real < vloc) never wins; replays deterministic.
Bias weights are small dyadic values so every fp32 bias sum is exact (order-independent), making the reference exact.
usage: CUDA_VISIBLE_DEVICES= python tools/test_dspark_markov.py
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))
import torch  # noqa: E402

from split_nv import dspark_markov as DM  # noqa: E402

TP = 2


def dyadic(shape, g, scale=16):
    return (torch.randint(-2, 3, shape, generator=g).float() / scale).to(torch.bfloat16)


def run_ranks(logits_full, anchor, w1, w2_full, W, vloc, v_real=None):
    """Both ranks' FusedMarkov on their shards, stepped in lockstep (the interpreter is not thread-safe), the
    all-gather emulated by concatenating the packs in rank order. -> [(toks, probs)] per rank."""
    V = logits_full.shape[1]
    fms, lgs = [], []
    for r in range(TP):
        v0 = r * vloc
        real = max(0, min(vloc, (v_real if v_real is not None else V) - v0))
        lg = torch.full((W, vloc), -1e30)
        lg[:, :real] = logits_full[:, v0:v0 + real]
        w2 = torch.zeros(vloc, w2_full.shape[1], dtype=torch.bfloat16)
        w2[:real] = w2_full[v0:v0 + real]
        fms.append(DM.FusedMarkov(w1, w2, v0, real, TP, None, W, torch.device("cpu")))
        lgs.append(lg)
    a = torch.tensor([anchor], dtype=torch.int64)
    for i in range(W):
        g = torch.cat([fm.local(i, lg, a).clone() for fm, lg in zip(fms, lgs)], 0)
        for fm in fms:
            fm.put(i, g)
    return [tuple(x.clone() for x in fm.finish(W)) for fm in fms]


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {detail}"), flush=True)
    return bool(cond)


def main():
    res = []
    g = torch.Generator().manual_seed(5)
    R, W = 64, 4
    for trial in range(6):
        V = 2 * (DM.BV * 7 + 5 + trial)  # partial last block per rank
        vloc = V // TP
        logits = torch.randn(W, V, generator=g) * 3
        if trial % 2:
            logits = (logits * 2).round() / 2  # many exact ties
        w1 = dyadic((V, R), g)
        w2 = dyadic((V, R), g)
        anchor = int(torch.randint(0, V, (1,), generator=g))
        ref_t, ref_p = DM.reference_chain(logits, anchor, w1, w2, W)
        (t0, p0), (t1, p1) = run_ranks(logits, anchor, w1, w2, W, vloc)
        res.append(check(f"trial {trial}: tokens == reference", t0.tolist() == ref_t, (t0.tolist(), ref_t)))
        rel = max(abs(a - b) / max(b, 1e-30) for a, b in zip(p0.tolist(), ref_p))
        res.append(check(f"trial {trial}: max-probs within 2e-6", rel < 2e-6, (p0.tolist(), ref_p)))
        res.append(check(f"trial {trial}: ranks bitwise identical", torch.equal(t0, t1) and torch.equal(
            p0.view(torch.int32), p1.view(torch.int32))))
        (t2, p2), _ = run_ranks(logits, anchor, w1, w2, W, vloc)
        res.append(check(f"trial {trial}: replay deterministic", torch.equal(t0, t2) and torch.equal(
            p0.view(torch.int32), p2.view(torch.int32))))

    # ties with zero bias (w2 = 0): the lowest vocab index wins
    V = 2 * DM.BV * 4
    vloc = V // TP
    w1 = dyadic((V, R), g)
    w2 = torch.zeros(V, R, dtype=torch.bfloat16)
    cases = {"same block": (3, 7), "across blocks": (DM.BV + 1, 2 * DM.BV + 3), "across ranks": (vloc - 1, vloc),
             "rank-1 only pair": (vloc + 5, vloc + DM.BV + 2)}
    for name, (i, j) in cases.items():
        logits = torch.zeros(W, V)
        logits[:, i] = logits[:, j] = 9.0
        (t0, p0), (t1, _) = run_ranks(logits, 0, w1, w2, W, vloc)
        res.append(check(f"tie {name}: lowest index {i} wins every draft", t0.tolist() == [i] * W and torch.equal(t0, t1),
                         t0.tolist()))
        expect = 1.0 / (2 + (V - 2) * float(torch.exp(torch.tensor(-9.0))))
        res.append(check(f"tie {name}: max-prob = 1/(2 + (V-2)e^-9)", abs(p0[0].item() - expect) < 1e-6 * expect,
                         (p0[0].item(), expect)))

    # padded shard tail: vocab 2*vloc - 3 real ids; the padding's huge logits must never win
    V = 2 * DM.BV * 3
    vloc = V // TP
    logits = torch.randn(W, V, generator=g)
    logits[:, V - 3:] = 1e4
    w1 = dyadic((V, R), g)
    w2 = dyadic((V, R), g)
    (t0, _), _ = run_ranks(logits, 1, w1, w2, W, vloc, v_real=V - 3)
    ref_t, _ = DM.reference_chain(logits[:, :V - 3], 1, w1[:V - 3], w2[:V - 3], W)
    res.append(check("padded tail never wins; tokens == reference over real ids", t0.tolist() == ref_t, (t0.tolist(), ref_t)))

    ok = all(res)
    print("ALL PASS" if ok else "SOME FAILED", f"({sum(res)}/{len(res)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
