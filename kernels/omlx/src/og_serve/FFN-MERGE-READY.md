# ds41-ffn: FFN-side fusion + decode glue + depth-controller recalibration (MERGE-READY, not deployed)

Branch `ds41-ffn` on production `2a70d1b0` (ds41-probe), worktree `~/src/wt/ds41-ffn`. The 16 ignored native
kernels were copied from ds41-probe with `cp -p` and checked with `cmp`. Nothing native (`csrc`) changed.
Opus ds41-ffn, 2026-09-29.

## What changed

All kernel changes are **bitwise identical** to the path they replace. Every output element keeps its original
arithmetic: the same lane-to-K mapping, decode, accumulation order, reductions and intermediate roundings. Only
the grouping of rows into threadgroups and launches changes. Every change has a kill switch; all are on by default except attn_out, which is bitwise but measured neutral to slightly slower.

| Piece | Launches/layer before -> after | Kill switch |
|---|---|---|
| **FFN sublayer** (`ffn_fuse.py`, new): pre-norm + FP32 rows + FP8 round trip; shared expert gate/up + SwiGLU/FP8 **with** a bitwise replica of MLX's router GEMM in the same launch; top-6 (unchanged kernel); routed gate/up + weighted SwiGLU/FP8; routed + shared down; hc post with the routed+shared combine in place | ~15-16 -> 6 | `DS41_FFN_FUSE=0` |
| **Attention input** (`attn_in.py`, new): hc pre-norm also writes the FP8 round trip; wq_a + wkv in one launch; wq_b + query partial RoPE in one launch | 6 -> 3 | `DS41_ATTN_IN_FUSE=0` |
| **Attention output** (`attn_out.py`, new, **opt-in, off by default**): wo_a grouped GEMV (BF16 or woa_compact codes) + wo_b's FP8 input round trip, 32 rows per threadgroup | 2 -> 1 | `DS41_ATTN_OUT_FUSE=1` to enable |
| **Depth controller** (`pipe_session.py`, `og_model.py`): current C(L) table, plus a fused-pair table used with >= DS41_OG_FUSE_MIN (3) sessions | - | `DS41_PIPE_COSTS`, `DS41_PIPE_COSTS_FUSED` |

Scope:
- **Covered:** singleton verify rows 1-5 and og_fused pairs (2-5 + 2-5, up to 16 rows). attn_out also covers
  single-stream and batched DSpark drafts.
- **Not covered (wq_b+RoPE):** og_fused pairs keep their wq_b and per-stream RoPE.
- **Unchanged:** prefill/replay (rows > 5 outside og_fused), `DS41_VERIFY_TILE`, vision rows (`image_mask`), q3,
  and the DSpark blocks other than their wo_a/wo_b step.
- **Rows 6-8 singleton stay unfused.** MLX's quantized_matmul serves the shared expert and wq_a/wkv there.

Key details:

- **Router GEMM replica** (`ffn_fuse._ROUTER`). MLX computes `x.astype(f32) @ w.astype(f32).T` (384 x 5120) with
  two kernels:
  - 1 row: `gemv` (bm4 sn32 tm4 tn4). The replica keeps the lane order and the shuffle-down tree.
  - 2-32 rows: `steel_gemm_splitk` (bm16 bn32 bk16, 32 partitions of 160) + `splitk_accum`. The replica runs
    the same `simdgroup_multiply_accumulate` 8x8x8 chain per partition, starting from zero with rows past M
    zero, then sums the partitions in order.
  - It reads the BF16 gate weight, which gives the same FP32 values at half the bytes.
  - Bitwise on all 20 real gate weights: 1120/1120 gemv, and 140/140 split-K for each M=2..8 (real and random
    inputs). og_fused requests share MMA fragments, because an output depends only on its row and column;
    checked at 10-16 rows.
- **Pre-norm tail.** Its serial per-element FP8 tail runs in J threadgroups per row, each recomputing the row
  norm with the same 256-thread reduction: J=10 for 1 row, J=5 otherwise (`DS41_FFN_NORM_SPLIT_1`,
  `DS41_FFN_NORM_SPLIT`). A single threadgroup was ~20 us/layer slower than the unfused chain.
- **Shared expert.** It rides in the router's launch: no dependency, so no barrier, and the latency-bound
  router hides under the bandwidth-bound shared expert. Launch order is fixed by MLX's BFS tape, so merging
  the two was the only way to overlap them.
- **wq_b + RoPE.** A RoPE pair is two adjacent wq_b output rows. The projection kernels already keep those in
  one simdgroup (rows kernel halves) or one threadgroup (one-row kernel). Each row is rounded to BF16, then
  rotated with fast_rope's expressions and the memoized tables.

## Timing

Partial-load harness: layers 20/24/25 aliased over 20-39, plus the head and DSpark. woa_compact is installed as
in production, and the forwards run against 2 real 8K box sessions. Times are wall ms including graph build,
flags alternating, 3 warm + 12 kept, medians. **"new" = shipped defaults** (FFN + attention input; attn_out off).
Commit 8afa2e14 + the attn_out default.

| Mac forward | prod | FFN only | **new** | saved |
|---|---:|---:|---:|---:|
| 1 row | 10.875 | 10.264 | **9.916** | **0.96** |
| 2 rows | 12.536 | 11.986 | **11.663** | **0.87** |
| 3 rows | 14.264 | 13.663 | **13.272** | **0.99** |
| 4 rows | 15.726 | 15.104 | **14.719** | **1.01** |
| 5 rows | 17.569 | 17.045 | **16.702** | **0.87** |
| og_fused 3+3 | 20.702 | 19.886 | **19.623** | **1.08** |
| og_fused 5+5 | 28.034 | 27.428 | **27.108** | **0.93** |
| DSpark draft, width 4 | 3.446 | 3.449 | 3.450 | 0 (path unchanged) |
| batched draft 4+4 | 4.731 | 4.732 | 4.744 | 0 |

The rerun of the same A/B an hour earlier matched within 0.03-0.1 ms.

attn_out alone, on top of new:

| | Change |
|---|---|
| singleton widths | +0.00 / +0.06 / +0.02 / +0.15 / +0.05 ms (slower) |
| 3+3 | +0.09 ms |
| 5+5 | -0.08 ms |
| drafts | +0.04 ms |

attn_out therefore stays off by default.

- **FFN sublayer alone.** 60-layer chain over 3 real layers, serialized (`ffn_bench.py`), per layer:

  | Rows | Before (us) | After (us) |
  |---|---:|---:|
  | 1 | 257.7 | 204.5 |
  | 2 | 346.0 | 297.2 |
  | 3 | 441.9 | 388.6 |
  | 4 | 525.1 | 471.1 |
  | 5 | 610.4 | 556.8 |
  | og_fused 2+2 | 534.7 | 471.3 |
  | og_fused 3+3 | 677.2 | 611.8 |
  | og_fused 5+5 | 939.7 | 882.4 |

  The real forward saves less than 20 x this, because part of the old glue overlapped other work there.
- **Router alone.** 20 distinct real gate weights, serialized, including ~4 us of glue: 38 -> 27 us/layer at
  1 row, 42 -> 26 us/layer at 5 rows.
- **New FFN per-epoch profile, 5 rows:**

  | Epoch | us | Floor / note |
  |---|---:|---|
  | hc pre/post | ~21 | |
  | shared + router | ~40 | 24 |
  | routed up + down | ~387 | union ~20 experts; ~84% of 1.17 TB/s, the unchanged kernels' efficiency |
  | top-6 | ~4 | |

  Running routed up and down with no barrier between them gained nothing, because they are bandwidth-bound.
  A persistent MoE megakernel would therefore not pay; the remaining FFN gap is the MoE kernels' own DRAM
  efficiency.
- **Launches per 5-row forward** (custom kernels): 542 -> 342. MLX ops also dropped: Matmul 25 -> 5, Full
  24 -> 4, AsType 30 -> 8. At 2 rows: 397 -> 317.

## Identity (all bitwise, no tolerance)

Final code (8afa2e14 with every flag on, attn_out included), partial forward with woa_compact installed as in production. All flags off vs all on,
2 real 8K box sessions:

| Check | Result |
|---|---|
| Singleton verify, both streams, widths 1-5 | **162/162 arrays** each: logits, DSpark hidden, every layer 20-39 cache slot + verify window. 20 fused FFN calls per forward. |
| og_fused pairs (2,2) (3,3) (5,5) (2,5) (4,3) | **324/324** each |
| DSpark `dspark_forward` widths 2/3/4 | **5/5** (logits, hidden, context caches) |
| Batched drafts (4,4) and (2,2) | **8/8** |

Other checks:
- **FFN chain** (`ffn_bench.py check`): 1-5 rows plus og_fused 2+2, 3+3, 5+5, 2+5, 4+3, 5+5+5, 4+5+3+2. Each ran
  with real, shifted-real, x8 and random inputs over 6 chained layers. All bitwise.
- **Synthetic kernel smokes**, checkpoint dims, all bitwise:
  - `ffn_smoke.py`: rows 1-16 x 3 scales, router included.
  - `attn_smoke.py`: wq_a/wkv at M 1-16, including M=2 against MLX qmm.
  - `attn_out_smoke.py`: BF16 M 2-16 and compact 2-5 with escapes.
  - `qrope_smoke.py`: 1-5 rows x ratio 0/2 x 3 positions.
- **Unit tests** `tests/test_deepseek_v41_ffn_fuse.py`: 33 passed (FFN forward rows 1/2/3/5 and og_fused 4/10/14 rows at 2 scales, 3 chained layers each; attention input at 1-16 rows; wq_b+RoPE at 1/2/3/5 rows x 3 position/ratio cases). With the related suites (attention projection/rounding, og_fused, mtp, woa_compact, fast_rope, pipe_costs): 201 passed. `tests/test_deepseek_v41_pipe_costs.py`: 4 passed.
- **All 45 `tests/test_deepseek_v41_*.py` files**, one pytest per file, on f54d9810: the only failures are
  affine (5), offload (5) and prefill_backpressure (1), and they fail identically on 2a70d1b0.

## Depth controller (target 2)

- **Outputs cannot change: widths 2-5 are row-prefix invariant.** Real boundaries, bitwise, **18/18 with
  production kernels and 18/18 with the new ones**. For each stream and each width n = 2..5 and keep <= n:
  - the width-n STEP's box boundary rows equal the first n rows of the width-5 STEP;
  - the Mac logits rows are equal;
  - after `rollback_boundary(keep)`, the committed Mac state (140 arrays) and the next step's logits are equal.

  The cost policy only chooses widths 2-5 (at context >= 1024). Width 1 differs from 2+ in MLX's kernels, and the
  policy never picks it.
- **New tables** (ms, L=2..5, 8K tier; the 128K/512K tiers are in the code):

  | Regime | Old (Sep 25) | New |
  |---|---|---|
  | c1/c2 | 29.6/32.6/34.4/36.7 | 24.1/26.1/28.2/30.8 |
  | fused-pair (>= 3 sessions) | - | 13.1/15.0/16.9/18.8 |

  - New c1/c2 costs: box RTT after box-perf + Mac forward + drafter 3.5 + host 0.4.
  - Fused-pair costs: the box is hidden, so this is the per-request Mac share of a fused pass + half the batched
    draft + scheduler overhead fitted to the served c4/c1 ratio.
  - c2 needs no table of its own: after scaling it has the c1 shape, and EVICT uses only cost ratios.
- **Offline simulation** of the served EVICT rule on per-cycle calibration data (459 DSpark cycles, og-speed
  Sep 25, code prompts at 8K/128K/512K), old table vs new:

  | Regime | Gain |
  |---|---|
  | c1 | +0.7% (code) to +1.0% (prose-like reweighting) |
  | c2 | +0.65% to +0.94% |
  | c4 | +2.8% (code) to +4.6% (prose-like), almost all from the fused table |

  Paired block bootstrap, 90% intervals: c1 +0.4 to +1.1%; c4 fused vs c1 table +1.1 to +2.2%.
- **Caveats:**
  - The calibration set is small and code-only; the Sep 25 prose capture failed at box OPEN.
  - The prose strata are reweightings of those same code cycles, not measured prose.
  - Copy-draft cycles (20%) and contexts under 1024 are unaffected.
  - Scripts and outputs: `benchmarks/og/depth_sim/`. They read `~/llm/ds41/og-speed/calib.jsonl`.
- **Optional served A/B, no code change.** A steeper table did better in simulation, because the drafter's
  max-prob is overconfident: 2.50 accepted drafts/cycle predicted vs 2.32 observed.
  - Env: `DS41_PIPE_COSTS='0:24.1,28.1,32.1,36.2'` and `DS41_PIPE_COSTS_FUSED='0:13.1,15.7,18.3,21.0'`.
  - Simulated: +0.6-1.3% at c1/c2 and +1.3% at c4 over the measured table.
  - Also worth a `DS41_PIPE_CALIB` prose capture.

## Projected served throughput (from measured Mac savings; not measured served)

Baselines are the served fixed-prompt numbers: c1 84.6, c2 117.1, c4 140.6 tok/s. Accepted tokens per cycle are
unchanged by the kernels. The kernel savings are measured Mac time; the depth gains are simulated.

| Level | Cycle model | Kernel saving | Kernels alone | + depth table |
|---|---|---|---|---|
| c1 | ~30.7 ms cycle | ~0.9 ms (widths 2-5) | **~+3.0% -> ~87.1** | +0.6-1.0% -> **~87.6-88.0** |
| c2 | Mac-bound, ~2 x (17.6 + 3.5 + ~2.4) ~ 47 ms per pair | 2 x 0.9 | **~+3.9% -> ~121.7** | +0.65-0.94% -> **~122.5-122.8** |
| c4 | fused pairs, ~28 + 4.7 + ~5 ~ 38 ms per pair | ~0.93-1.08 ms per pair | **~+2.6% -> ~144.3** | fused table +2.8-4.6% -> **~148-151** |

Caveats:
- The full model has 20 distinct layers. The saving is mostly launch/latency, which carries over. The router
  byte saving (BF16 read instead of the FP32 copy) is larger with distinct weights than in the harness.
- These are projections, not a served measurement. The coordinator's served A/B is the gate.

## Memory

- No new weights or caches.
- The router replica reads the BF16 gate weight. Prefill/replay still creates the unfused FP32 copy, as before.
- Threadgroup memory per launch is at most about 17 KB.

## Deploy / rollback

- **Deploy:**
  1. Set `DS41_TREE=/Users/ian/src/wt/ds41-ffn` in the llama-swap ds41 env (keep `DS41_OG_CACHE_GIB=16`) and reload.
  2. After the switch, check three things (see the Sep 27 stale-worker incident):
     - the new omlx-server pid and start time;
     - the gpu.lock holder;
     - `woa_compact encoded 23` in og-child.log.
  3. Run the usual fixed-prompt identity check: `og_serve/fe_bench.py ... --identity` + `fe_cmp.py`. c1/c2/c4
     fixed-prompt outputs must be identical to ds41-probe.
- **Rollback, same tree:**
  - Kernels: `DS41_FFN_FUSE=0 DS41_ATTN_IN_FUSE=0`. attn_out is already off.
  - Depth table:
    `DS41_PIPE_COSTS='0:29.6,32.6,34.4,36.7;131072:29.9,32.9,34.7,37.1;524288:30.2,33.4,35.4,37.8'`. This restores
    the old table, and while `DS41_PIPE_COSTS_FUSED` is unset it also replaces the fused table.
- **Rollback, other tree:** `DS41_TREE=/Users/ian/src/wt/ds41-probe`.

## Reproduce (Mac; production idle only)

`idle_guard.py` stops its own child on traffic, and `~/llm/ds41/ffn/wait_idle.py` also waits for other agents'
partial loads.

```bash
cd ~/src/wt/ds41-ffn; PY=~/llm/.venv-ds41-omlx-tiles/bin/python; G="$PY benchmarks/og/idle_guard.py"
for s in ffn attn attn_out qrope; do $G $PY -u benchmarks/og/${s}_smoke.py; done       # synthetic, < 1 GB
DECODE_MAX_SECONDS=400 $G $PY -u benchmarks/og/ffn_bench.py                           # ~22 GB: FFN of 3 layers
DECODE_MAX_SECONDS=200 $G $PY -u benchmarks/og/router_proto.py                        # 20 real gate weights
DECODE_MAX_SECONDS=400 FW_TESTS=identity,draft $G $PY -u benchmarks/og/ffn_fwd.py     # ~34 GB, 2 box sessions
DECODE_MAX_SECONDS=400 FW_TESTS=depth $G $PY -u benchmarks/og/ffn_fwd.py              # FW_DEPTH_FLAGS=off: prod kernels
DECODE_MAX_SECONDS=400 FW_TESTS=draft,timing $G $PY -u benchmarks/og/ffn_fwd.py
$PY -m pytest -q tests/test_deepseek_v41_ffn_fuse.py tests/test_deepseek_v41_pipe_costs.py
```

- **Logs:** `~/llm/ds41/ffn/` (`forward.jsonl`, `bench.jsonl`, `fwd_runs*.log`).
- **Limits:**
  - The model is partial: 3 real layers aliased over 20, plus the full head and DSpark.
  - At most 2 box sessions were used; there was no full-model or served run.
  - Savings are measured Mac forward time.
- **Not done (small):**
  - Folding top-6 into the router launch. It needs a cross-threadgroup atomic handoff; ~0.08 ms; judged not
    worth the race risk.
  - The drafter's own mHC/router glue: ~250 dispatches per draft. It needs bitwise replicas of the unfused
    hc_pre + RMSNorm reductions. Estimated 0.2-0.4 ms/cycle.
