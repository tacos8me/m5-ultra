# ds41-attn: attention sublayer kernels (MERGE-READY)

Branch `ds41-attn` on `f56f7ffa` (the served ds41-og tree). Worktree `~/src/wt/ds41-attn`. Not deployed.

## What changed

All changes apply to short decode/verify blocks (1-8 rows, outside `DS41_VERIFY_TILE` scopes). Every change is
**bitwise identical** to the code it replaces. Edits are limited to `Attention`/`Indexer` in `language.py`, the
attention and index-selection paths in `kernels.py`, the new `attn_fusions.py`, and one constant in `fast_qmv.py`.

- `attn_fusions.rms_quant`: `q_norm`, and the FP8 round trip of its output, in 1 kernel instead of 5 (MLX
  `row_reduce_looped` order for the Sum). `wq_b` and the indexer's `wq_b` share the rounded `qr`, so `qr` is
  quantized once per layer, not twice.
- `attn_fusions.kv_rows`: `kv_norm`, partial RoPE, `pack_fp8` and the window concatenate in 1 kernel instead of 7.
- `kernels.attention_shared`: the sparse-attention scan decodes each packed KV row once for 8 heads (all 64
  heads read the same single KV head). Before, it decoded each row once per head. The partials are unchanged.
- `kernels.merge_wide` / `merge_rope`: the merge uses 4 simdgroups per head, which is 4x the parallelism. It also
  applies the inverse output RoPE, which removes the separate `rope_range` dispatch.
- `attn_fusions.index_q`: indexer query RoPE plus the FP4 round trip in 1 kernel instead of 7. It replicates the
  compiled MLX graph: precise `log2`, the 7-digit minimum literal, NaN-propagating max/min.
- `attn_fusions.select_rows`: exact top-k in one threadgroup per row (radix select, then ordered compaction).
  - Output: the ids ascending, with -1 in front for -inf selections.
  - It replaces the candidate layers' argsort → gather → mask → gather → sort chain.
  - It also replaces layer 20's tile sort + rank merges (keys and blocks, for widths up to 32768).
  - Order: score desc, position asc. -0 and +0 are equal. The forced latest block counts as +inf.
- Candidate-id expansion is memoized per forward. Before, each of the 4 candidate layers rebuilt it.
- `fast_qmv.MIN_ROWS` 4 → 3. At M=3 the attention projections use the fast wide kernel instead of MLX
  `qmv_wide`: same arithmetic (bitwise checked), 51-56 vs 61-85 us per call.

Kill switches, all defaulting to 1 (on); 0 restores the old path:

| Env flag | Controls |
|---|---|
| `DS41_ATTN_FUSE` | norm, kv, merge-RoPE and index_q fusions |
| `DS41_ATTN_SHARED` | shared-decode attention scan |
| `DS41_MERGE_WIDE` | 4-simdgroup merge |
| `DS41_INDEX_SELECT` | one-kernel top-k |
| `DS41_FAST_QMV_MIN_ROWS` | set to 4 for the old M=3 path |

## Speed

Attention sublayer, 20 layers, all 20 distinct real weight sets (no aliasing, so no extrapolation). Synthetic caches.
GPU time: the graph queues behind a spin kernel, so host encoding is excluded. Old and new run interleaved in one
process; the table shows the minimum of 8-10 rounds.

| rows k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| old, 8K (ms) | 5.77 | 6.30 | 6.82 | 6.72 | 7.15 |
| new, 8K (ms) | 4.80 | 5.28 | 5.42 | 5.74 | 6.27 |
| saved, 8K (ms) | **0.97** | 1.02 | 1.40 | 0.98 | **0.88** |
| saved, 128K (ms) | 0.95 | | | | 1.26 |

- Medians saved (8K): 0.65 / 1.8 / 1.5 / 1.5 / 1.1 ms. The medians are noisier: another agent's partial-load
  worker shared the GPU during every run.
- **% of the bandwidth floor** (attention sublayer floor 2.82 / 2.85 ms from REPORT.md): 49% → 59% at k=1 and
  40% → 45% at k=5.
- The chain includes about 0.08 ms of benchmark glue.
- Host side: the full forward (partial load, `benchmarks/og/fwd_ab.py`) builds about 0.9-1.1 ms faster per forward.
  About 150 fewer primitives per forward means less Python graph build and less MLX encode (build_ms k=1 9.96 → 9.06,
  k=5 18.3 → 17.1). This counts wherever the forward is host-bound.
- Per-cycle estimate: about -1.0 ms GPU at k=1-5, plus the host-build saving. That is below the -2.5 to -3.5 ms
  target, for three reasons:
  - The dense projections were already at 85-100% of achievable bandwidth once measured serialized: wq_b 1016 GB/s,
    wo_b 935 GB/s, wo_a 913-1118 GB/s. Vectorized-load and prefetch GEMV variants did not beat them.
  - The sparse-attention scan is ALU-bound on its per-key online softmax. The remaining cost at k=5 is about
    1.5 ms per 20 layers, and it cannot shrink further without changing the reduction order.
  - Not done, each worth about 0.08 ms: fusing q RoPE into wq_b, and fusing the wo_b input quantize into wo_a.

## Numerics: bitwise identical, no tolerance used

| Check | Scope | Result |
|---|---|---|
| `benchmarks/og/attn_bitwise.py` (seconds, ~1 GB) | 235 old-vs-new comparisons of every kernel at 1-8 rows. Includes -0/+0, -inf tails, causal masks, ties and forced blocks. | all identical |
| `benchmarks/og/attn_bench.py` with `ATTN_DUMP` + `cmp_npz.py` | All 20 attention layer outputs, k=1..5, at 8K and 128K. | identical to f56f7ffa |
| `benchmarks/og/fwd_real.py` | Real box session (8K prompt, real layer-20 KV and index rows, real boundaries), partial load, k=1..5 at 15 positions. Logits and DSpark hidden compared old vs new. | identical |

- Top-k selections are identical, so the tie rate is 0.
- Unit tests: 252 pass in the attention, index, kernel, rope, batched and CED suites.
- Under the production env (`DS41_SPARSE=1`) the same 3 `attention_rounding` tests fail on f56f7ffa and on this
  branch alike. They assume `DS41_SPARSE=0`.

## Reproduce (Mac)

```bash
cd ~/src/wt/ds41-attn; PY=~/llm/.venv-ds41-omlx-tiles/bin/python
$PY benchmarks/og/attn_bitwise.py                   # bitwise kernel gate
ATTN_TESTS=ab $PY benchmarks/og/attn_bench.py ab    # old/new GPU A/B (3 GB, no lock; production idle)
cd benchmarks/og && $PY fwd_real.py                 # real-session bitwise gate (31 GB partial; production idle)
```
