# ds41 c4: why the served number moved only 139.4 -> 141.1, and what fixes it

Branch `ds41-c4` (worktree `~/src/wt/ds41-c4`, based on production b16205ac; the 16 ignored native files were
copied with `cp -p` and checked with `cmp`). Nothing is deployed and nothing is pushed. Opus ds41-c4, 2026-09-29.

## Short answer

1. **The steady-state c4 rate did improve, by about 6% at short context.** The number reported as 141.1 is the
   median of two samples, 132.5 and 150.7. Earlier processes gave 134-141, with the two samples of each run within
   about 1 tok/s of each other.
   - The first sample was the first c4 traffic after the restart. Every fused-pair kernel shape built its Metal
     pipelines then. Those shapes have 6-10 rows, and nothing before them in that process (warmup, c1, c2) had run
     them.
   - The second, warm sample (150.7) is +7.5% over history and matches the ffn projection (144 from kernels alone,
     148-151 with the depth table).
2. **The depth tables cannot act on that benchmark.** fixedconc's prompts are 14-22 tokens and the outputs at most
   256 tokens. Every request therefore stays below 1024 tokens of context. Below 1024 the cost policy is off
   (`choose_cost_depth` and `_dspark_prepare` both gate on `>= 1024`), so the depth is set by acceptance only and
   `DS41_PIPE_COSTS*` is never read. The projected +2.8-4.6% "fused-table" gain has nothing to act on in fixedconc.
   It also cannot show up in an A/B that uses these prompts.
3. **The 8K nonce suite's "c4 136.7" is a span metric, and admission dominates it.** The scheduler admits a
   burst one request at a time: only `waiting[0]` is offered to `should_defer`, so the next box OPEN starts only
   after the previous request has been opened, imported and stepped.
   - First-token times of the 4 streams: 0.62 / 1.56 / 2.41 / 3.15 s.
   - All 4 streams decode together for only 3.5-4.1 s of the ~7.5 s span.
   - The steady c4 rate inside that window is about 187 tok/s (estimated from the logs, ±10%), not 137.
4. **The fused-pair table is applied at c4.** `make_mtp_depth_controller` returns `PipelineDepthController(depth,
   fused=_fused_regime)` for every DSpark request. The regime is `og_fused.ENABLED and len(SESSIONS) >= 3`.
   - The harness trace saw the fused-regime branch taken on every draft (`depth: {"fused4": 199}` with
     FUSE_MIN=2).
   - At 8K, draft widths are always 4 (`max_depth`), so pairs batch their drafts unless one of them took a
     copy/extra-source proposal.

## c4 pair-pass breakdown (measured)

**Partial-load served flow** (`benchmarks/og/c4_host.py`): the unchanged og_server/omlx scheduler with 2 real 8K
box sessions, `DS41_OG_FUSE_MIN=2` and 5+5 rows (`HS_FIXED_DEPTH=1`). Mean ms per scheduler step over 100-step
windows, with one pair pass per step:

| Part of one pair pass (5+5 rows, 8K) | ms | Notes |
|---|---:|---|
| Fused Mac forward (og_fused) | **27.5** | 21.9 in the Python graph build, with the GPU running alongside it, plus a 5.6 GPU tail at the `mx.eval` in `_advance_groups` |
| Batched DSpark drafts + presend (`mtp_draft_jobs`) | **6.0** | ~4.8 GPU; `_dspark_finish` waits for it (4.3) |
| Accept/commit, rollback, row batches, `_replace_cache_rows`, presend | **~1.6** | chain 0.65, replace_rows 0.4-0.6, row_batch 0.18, rollback 0.10, presend 0.05 |
| Emit + engine-loop gap between steps | 0.1 | per step |
| Box, 2 x 5-row STEP | 15.8 | Exposed at c2 because the pair's steps are needed right away. Hidden at c4 because the other pair's ~30-35 ms pass covers it (box utilisation ~50%). |
| Admission slices | 0 | Only while a request is being imported (90 ms slices) |

**c4 round** (two pair passes plus the step overhead):

- **At 5+5 rows:** 2 x 35 = ~70 ms.
- **At production 8K widths (L ~3.8):** fused ~22.7 ms, pass ~30.3 ms, round ~61 ms. With 12.9 tokens per
  round that is ~210 tok/s.
- The production log estimate is 69 ms/round. The difference (≈4 ms/pass) is either box waits at c4 or
  partial-model optimism. The A/B's `--stats` column `box_wait_ms_per_step` will tell which.

**Short context (fixedconc):**

- Model: L ~2.8, so pairs of ~5.7 rows, fused ~19.3 ms.
- Drafts mostly run unbatched: widths are `cur` and differ, and width 1 never batches. Two ~3.2 ms drafts plus
  ~1.2 ms host come to ~7.6 ms.
- Host ~1.6 ms.
- That gives a pass of ~28.5 ms and a **round of ~57 ms**.
- Production matches: in the warm group the first finisher did ~72 rounds in 4.14 s = **57.5 ms/round**
  (history: ~61 ms; cold group: 68 ms).
- Steady state is 9.52 tokens per round / 57.5 ms = **~166 tok/s**. fixedconc reports 150.7 because it also
  counts the ~0.4 s staggered start and a ~1.1 s tail at c3/c2/c1 (the 4 outputs are 188-256 tokens).

**Pairing and widths.** Unequal pairs cost what their total rows cost: (2,5) 21.3 ms vs (3,4) 21.2 ms, and
(2,2) 16.2 -> (5,5) 27.1 ms is 1.82 ms per row. Pairing policy therefore does not matter. No request ever runs
at 1 row: DSpark depth is >= 1 and copy drafts are 4. Width mismatch breaks batched drafting only below 1024
tokens. Ragged batching (both widths >= 2, but different) would help ~16% of short-context passes by ~1.5 ms:
under 1%, so not done.

## Root cause of the "shortfall"

| Factor | Effect on the reported c4 |
|---|---|
| n=2, and sample 1 was the first c4 traffic after deploy (cold pair pipelines, see below) | 132.5 vs 150.7; median 141.1 |
| Projection included the depth table, which cannot act below 1024 tokens (fixedconc) | the +2.8-4.6% part was never reachable |
| Span/wall metrics include staggered admission (serial OPEN+import) and the unequal-length tail | 8K: 137 reported vs ~187 steady; short: 151 vs ~166 steady |

**Cold pipelines** (`benchmarks/og/c4_cold.py`, fresh process, partial load; cold time minus warm time per first
use):

- Fused pairs: +208 ms in all. This includes one 191 ms stall at (4,5) rows; the others were 0-6.5 ms each.
- Single-draft width 1: +38 ms.
- The probe counted 148 distinct kernel builds (name + template), 42 of them while priming.
- In a served c2-fused run after the normal warmup, 39 new builds landed in the first windows. The first
  traced window's step time was 75 ms against 38 ms steady.
- The cold group's rounds were +10 ms slower throughout (68 vs 57.5 ms). The builds explain only part of that.
  The rest is not reproducible from the logs (box or GPU state at 10:37Z, right before the box window).

## Fixes (branch ds41-c4)

### 9fe97a71 — chained box OPENs for queued requests (`og_model.OgPrefill.preopen`)

While the head is deferred, the OPEN of up to `DS41_OG_PREOPEN` (3) text requests queued behind it starts within
the free scheduler slots, so at most `max_num_seqs` sessions exist.

- **Chain mode (default):** the next OPEN starts once every earlier OPEN is done on the box. The box never
  prefills two of our prompts at once, so the head's TTFT is unchanged. The Mac import and scheduler steps of
  request k overlap the box prefill of k+1.
- `DS41_OG_PREOPEN_CHAIN=0` starts them all at once, and the box then interleaves them. The whole burst starts
  together, but the head waits: 0.70 -> 1.2-1.44 s at c2 8K. That is not the default.
- Admission order, imports and arithmetic are unchanged. `DS41_OG_PREOPEN=0` restores the old path.

**Evidence** (partial load, 2 real sessions, `DS41_OG_BOX_CACHE=0`):

| c2 8K fresh prompts, 4 reps (3 warm) | TTFT head / 2nd (s) | span tok/s |
|---|---|---:|
| off (production) | 0.70 / 1.65 | 43.2 |
| chain (new default) | **0.70 / 1.39** | **47.2** |
| all at once | 1.20-1.44 / 1.21-1.43 | 55.4 |

- **Identity:** `batch_identity.py` (4 mixed prompts, 3K-12K tokens, 128 tokens each, alone and in pairs) gives
  identical text in every leg and across legs (off vs chain vs all: 4/4 each). The partial model's text is
  meaningless, but it is deterministic.
- **Unit tests:** `tests/test_deepseek_v41_og_preopen.py`, 5/5 (order, chain, slot bound, images/invalid,
  abort). `test_deepseek_v41_og_ttft2.py` 15/15 and `pipe_costs` 4/4 still pass.

**Expected in production:**

- The window metric is unchanged.
- In a c4 8K cold burst, the later streams' first tokens come ~0.15-0.35 s earlier each: 4th about 3.1 -> ~2.7 s.
  The box prefill (0.45-0.7 s per 8K prompt, serial) stays the floor.
- Short fixed prompts: the 4th stream is admitted ~0.2 s earlier, about +3% on fixedconc c4.

### c5d530c7 — warm the fused-pair shapes at worker start (`og_server.warm_pairs`)

After the usual warmup, two concurrent requests run on the ~4K-token warmup prompt, with pairs allowed at 2
sessions and the cost policy cycling the verify widths through 2..5 x 2..5. Both are restored afterwards.

- Widths 2-5 are row-prefix invariant (ffn: 18/18 bitwise), and no client sees these outputs.
- It is best effort: an error prints `warmup_pairs: skipped`, and the worker still becomes healthy.
- `DS41_OG_WARM_PAIRS=0` turns it off.

**Evidence** (served partial flow, DS41_WARMUP=1):

- `warmup_pairs done, 4.6 s, 90 fused calls, errors []`.
- New kernel builds during the client's c2-fused run: **39 -> 0**.
- It adds ~5 s to worker start.

## Served A/B plan (the coordinator runs it in a Mac window)

**Benchmark:** `/home/ian/mac/bench/c4ab.py`, run from Linux against `http://192.168.1.203:8080`. A copy is in
`benchmarks/og/c4ab.py`.

- **Prompts:** 8 fixed code documents of 2.2K-4.7K tokens (`bench/c4ab-prompts.json`, sha 3dd704dd2cb4fc47). All
  run at >= 1024 context, so the tables are live.
- **Sampling:** temperature 0, streamed, 384 tokens.
- **Groups:** one discarded warm-up group, then `--groups` groups of c (8 by default; 10 recommended).
- **Metrics:**
  - **window_tok_s:** the rate while all c streams decode. This is the A/B metric.
  - span_tok_s (the runthru2 metric) and wall_tok_s (the fixedconc metric).
  - `--stats` og/stats deltas: tok/step, rows/step, fused share, draft_batches, and **box_wait_ms_per_step**
    (tells whether the box is hidden at c4).
- **Identity gate:**
  - Within a run, a repeated prompt must give the same text.
  - `compare` requires every prompt's text to be identical across all files, including c1/c2/c4 runs of the same
    prompts. It then prints the paired per-group ratio with a bootstrap 95% CI.
- **Resolution:** historical within-process fixed c4 spread was ~0.5%, and the runthru2 c4 SD was 1.3%. So 8-10
  groups give a CI of about ±1%, enough to resolve 3%.

**Variants.** All are env-only on the deployed tree except C.

| Label | Env / tree |
|---|---|
| A | current production (b16205ac defaults) |
| B | `DS41_PIPE_COSTS='0:24.1,28.1,32.1,36.2'` `DS41_PIPE_COSTS_FUSED='0:13.1,15.7,18.3,21.0'`. A single tier-0 table applies at every context, including 128K/512K. |
| C (optional, code) | `DS41_TREE=/Users/ian/src/wt/ds41-c4`, default tables. Its kill switches `DS41_OG_PREOPEN=0 DS41_OG_WARM_PAIRS=0` give back A's behaviour. |

```bash
S='ssh m5 curl -s localhost:12147/og/stats'
cd /home/ian/mac/bench
# per variant, after the reload + /health ok (order A, B, A2, [C]):
python3 c4ab.py run A  --groups 10 --stats "$S"
python3 c4ab.py run A-c1 --c 1 --groups 8 --stats "$S"; python3 c4ab.py run A-c2 --c 2 --groups 8 --stats "$S"
# ... same for B (B, B-c1, B-c2), then A2 (drift check), then C
python3 c4ab.py compare c4ab-A.json c4ab-B.json c4ab-A2.json [c4ab-C.json]
python3 c4ab.py compare c4ab-A-c1.json c4ab-B-c1.json
python3 c4ab.py compare c4ab-A.json c4ab-A-c1.json c4ab-A-c2.json   # outputs independent of concurrency
```

Each c4 run takes ~2 min. Worker restarts now take ~5 s more (warm pairs) on the C tree.

- **Ship B only if** `identity_ok` holds, the c4 window ratio CI excludes 1.0 on the positive side, and c1/c2
  do not regress.
- **C should show** window ≈ A, better span/wall, and no first-group penalty.

**Expected gains:**

- **B:** the simulation says +1.3% at c4 and +0.6-1.3% at c1/c2 (ffn depth_sim; the gain comes from drafter
  over-confidence, 2.50 predicted vs 2.32 accepted per cycle).
  - Measured per-row pair cost is 1.82 ms. That matches the current fused slope (1.9) better than the steeper one
    (2.6), so a null result is plausible.
  - B has no effect on fixedconc.
- **C:** window 0.
  - Span/wall at c4 8K cold burst: +5-8%.
  - fixedconc c4: ~+3%, and the first group after a restart stops being ~10% low.
- Recommendation for fixedconc itself: discard one c4 warm-up group and use >= 6 groups. Its c4 has only one
  group of 4 prompts (n=2).

## Not done / next levers

- **GPU idle inside a pass.** About 3 ms per pass is host-only: accept/commit, row batches and the draft
  prepare/Markov steps. Overlapping pair B's graph build with pair A's draft could recover up to ~5-10% of c4
  steady state. It needs a restructure of `_advance_groups`, with presend ordering kept so the box stays hidden.
- **Ragged batched drafts below 1024 tokens:** < 1%.
- **A c3 triple fuse** would expose the box (3 x 8 ms); pair + single is right.
- **Box prefill** is the floor for 8K burst admission: 4 x 0.45-0.7 s serial.

## Files

| File | What it is |
|---|---|
| `og_serve/C4-ANALYSIS.md` | this note |
| `benchmarks/og/c4_cold.py` | cold vs warm per c4 shape |
| `benchmarks/og/c4_host.py` | traced served flow |
| `benchmarks/og/c4_leg.sh` | one idle-guarded leg, <= 2 sessions |
| `benchmarks/og/c4ab.py`, `c4ab-prompts.json` | the A/B client |

Logs are in `~/llm/ds41/c4/`: `cold.jsonl`, `cold-run1.log`, `server-*.log` (c4trace lines), `client-*.log`. The
identity files are `~/llm/ds41/batch/identity-preopen{0,3,-chain}.json`.

**Limits:**

- The model is partial: 3 real layers aliased over 20-39, with at most 2 box sessions, so c4 itself was not run
  on the harness.
- The production c4 numbers come from existing logs: the 10:37Z fixedconc groups and the runthru2 leg3 records.
