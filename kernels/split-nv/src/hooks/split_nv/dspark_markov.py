"""Fused Markov chain of the box DSpark drafter (SPLIT_NV_DSPARK_MARKOV=fused, the default).

DSpark's rank-256 Markov head adds W2 . W1[prev] to each block position's base logits, left to right, with prev = the
previous greedy draft (the anchor first). On the box the logits and W2 are vocab-sharded (TP2): per draft each rank
biases its shard, reduces it to (max, argmax, logsumexp), and one 12-byte all-gather gives every rank the global
winner. The reference path (drafter_bench.BoxDrafter.markov) spends ~15 graph nodes per draft on that; this module
spends 3 plus the all-gather:

  partial[grid]  : winner of the previous draft (from the gathered packs) -> e = W1[prev] -> per block of BV vocab rows
                   bias = W2 . e (fp32 accumulate, rounded to BF16 as the reference's BF16 GEMV output) -> s = logits
                   + bias -> block max, lowest index attaining it, sum exp(s - max)
  reduce[1]      : blocks in vocab order -> (M, lowest index attaining M + v0, logsumexp) of this rank's shard
  all_gather     : [tp, 3] (rank order = vocab order)
  decode[1]      : after the last draft, every draft's token and max-prob from its gathered pack

Tie-break everywhere: lowest vocab index (lowest block, then lowest row in the block, then lowest rank), as MLX argmax.
Every reduction is over a fixed shape in a fixed order (no atomics), so a draft is a deterministic function of its
inputs, and both ranks derive tokens / max-probs from the same gathered bytes with the same kernel: identical.
The winner rule of `partial` (next draft's prev) and of `decode` (reported token) is the same code (_winner).

CPU: TRITON_INTERPRET=1 runs these kernels on CPU tensors (tools/test_dspark_markov.py).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover -- the reference path still works
    triton = tl = None

BV = 32  # vocab rows per partial program (a 32 x 256 fp32 tile)


if triton is not None:
    @triton.jit
    def _winner(src_ptr, TP: tl.constexpr):
        """(token as fp32, max) of a gathered pack [TP, 3]: the lowest rank attaining the max."""
        r = tl.arange(0, TP)
        m = tl.load(src_ptr + r * 3)
        idx = tl.load(src_ptr + r * 3 + 1)
        gmax = tl.max(m, 0)
        w = tl.min(tl.where(m == gmax, r, TP), 0)
        tok = tl.sum(tl.where(r == w, idx, 0.0), 0)
        return tok, gmax

    @triton.jit
    def _partial(logits_ptr, w1_ptr, w2_ptr, src_ptr, part_ptr, v_real, R: tl.constexpr, BLOCK_V: tl.constexpr,
                 FIRST: tl.constexpr, TP: tl.constexpr):
        pid = tl.program_id(0)
        if FIRST:
            prev = tl.load(src_ptr).to(tl.int64)
        else:
            tok, _ = _winner(src_ptr, TP)
            prev = tok.to(tl.int64)
        k = tl.arange(0, R)
        e = tl.load(w1_ptr + prev * R + k).to(tl.float32)
        rows = pid * BLOCK_V + tl.arange(0, BLOCK_V)
        valid = rows < v_real
        w = tl.load(w2_ptr + rows.to(tl.int64)[:, None] * R + k[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        bias = tl.sum(w * e[None, :], 1).to(tl.bfloat16).to(tl.float32)
        lg = tl.load(logits_ptr + rows, mask=valid, other=0.0)
        s = tl.where(valid, lg + bias, float("-inf"))
        m = tl.max(s, 0)
        i = tl.argmax(s, 0, tie_break_left=True)
        se = tl.sum(tl.where(valid, tl.exp(s - m), 0.0), 0)
        se = tl.where(m == float("-inf"), 0.0, se)
        tl.store(part_ptr + pid * 3, m)
        tl.store(part_ptr + pid * 3 + 1, (pid * BLOCK_V + i).to(tl.float32))
        tl.store(part_ptr + pid * 3 + 2, se)

    @triton.jit
    def _reduce(part_ptr, pack_ptr, nb, v0, NB: tl.constexpr):
        b = tl.arange(0, NB)
        ok = b < nb
        m = tl.load(part_ptr + b * 3, mask=ok, other=float("-inf"))
        idx = tl.load(part_ptr + b * 3 + 1, mask=ok, other=0.0)
        se = tl.load(part_ptr + b * 3 + 2, mask=ok, other=0.0)
        M = tl.max(m, 0)
        sel = tl.min(tl.where(ok & (m == M), idx, 3.0e38), 0)
        S = tl.sum(tl.where(ok & (m > float("-inf")), se * tl.exp(m - M), 0.0), 0)
        tl.store(pack_ptr, M)
        tl.store(pack_ptr + 1, sel + v0)
        tl.store(pack_ptr + 2, M + tl.log(S))

    @triton.jit
    def _decode(g_ptr, tok_ptr, prob_ptr, TP: tl.constexpr):
        i = tl.program_id(0)
        src = g_ptr + i * TP * 3
        tok, gmax = _winner(src, TP)
        r = tl.arange(0, TP)
        lse = tl.load(src + r * 3 + 2)
        lm = tl.max(lse, 0)
        total = lm + tl.log(tl.sum(tl.exp(lse - lm), 0))
        tl.store(tok_ptr + i, tok.to(tl.int64))
        tl.store(prob_ptr + i, tl.exp(gmax - total))


def _p2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


class FusedMarkov:
    """Static-buffer Markov chain for CUDA-graph capture. all_gather(pack [1, 3]) -> [tp, 3]."""

    def __init__(self, w1, w2, v0, v_real, tp, all_gather, max_w, device):
        if triton is None:
            raise RuntimeError("triton unavailable")
        self.w1, self.w2 = w1.contiguous(), w2.contiguous()
        self.R = int(w1.shape[1])
        if self.R & (self.R - 1) or int(w2.shape[1]) != self.R:
            raise ValueError(f"Markov rank {self.R} (W2 {tuple(w2.shape)}): needs a power of two shared by W1 and W2")
        self.vloc = int(w2.shape[0])
        self.v0, self.v_real, self.tp = int(v0), int(v_real), int(tp)
        if self.tp & (self.tp - 1):
            raise ValueError("tp must be a power of two")
        self.all_gather = all_gather
        self.nb = -(-self.vloc // BV)
        z = dict(device=device, dtype=torch.float32)
        self.parts = torch.zeros(self.nb, 3, **z)
        self.pack = torch.zeros(1, 3, **z)
        self.g = torch.zeros(max_w, self.tp, 3, **z)
        self.toks = torch.zeros(max_w, dtype=torch.int64, device=device)
        self.probs = torch.zeros(max_w, **z)

    def __call__(self, logits, anchor, W):
        """logits [>= W, vloc] fp32 (this rank's shard), anchor int64 [>= 1] (device) -> (toks [W], probs [W])."""
        for i in range(W):
            self.put(i, self.all_gather(self.local(i, logits, anchor)))
        return self.finish(W)

    def local(self, i, logits, anchor):
        """Draft i on this rank's shard -> its pack [1, 3] (max, global index, logsumexp)."""
        src = anchor if i == 0 else self.g[i - 1]
        _partial[(self.nb,)](logits[i], self.w1, self.w2, src, self.parts, self.v_real, R=self.R, BLOCK_V=BV,
                             FIRST=i == 0, TP=self.tp)
        _reduce[(1,)](self.parts, self.pack, self.nb, float(self.v0), NB=_p2(self.nb))
        return self.pack

    def put(self, i, gathered):
        self.g[i].copy_(gathered)

    def finish(self, W):
        _decode[(W,)](self.g, self.toks, self.probs, TP=self.tp)
        return self.toks[:W], self.probs[:W]


def reference_chain(logits_full, anchor, w1, w2_full, W):
    """Full-vocab reference (one rank holding everything; tests): greedy chain with BF16 bias and lowest-index ties.
    Returns (tokens list, max-probs list) in float64 math on the fp32 biased logits."""
    prev = int(anchor)
    toks, probs = [], []
    for i in range(W):
        bias = (w2_full.float() @ w1[prev].float()).to(torch.bfloat16).float()
        s = logits_full[i].float() + bias
        m = s.max()
        tok = int(torch.nonzero(s == m)[0, 0])
        probs.append(float(torch.exp((m - torch.logsumexp(s.double(), 0)).double())))
        toks.append(tok)
        prev = tok
    return toks, probs
