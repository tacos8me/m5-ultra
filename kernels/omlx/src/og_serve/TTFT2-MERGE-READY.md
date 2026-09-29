# ds41-ttft2: Mac tail replay, admission stall, first-token path — MERGE-READY

Branch `ds41-ttft2` on `2a70d1b0` (the served ds41-probe tree). Worktree `~/src/wt/ds41-ttft2`, with the 16 gitignored
compiled kernels copied from ds41-probe (`cp -p`, `cmp` clean, 16/16). Not deployed. No kernel sources (csrc) changed.
All numbers are partial-load (layers 20/24/25 aliased, ~31 GB, production idle, never concurrent with ds41-ffn),
real box sessions (<= 2 at once), box b4f5ea4 / og-s4.4.

## Changes (all flags default on; each is arithmetic-neutral, see Identity)

1. **8-row MoE blocks for the replay** (`language.py`, `DS41_MOE_BM8=1`). The sorted MXFP4 expert path (32-170 rows =
   the replay) used 16-row steel blocks; real routing at 128 rows gives ~5 rows/expert (histogram over 40 layers: 35%
   of active experts get 1 row), so half the MMA tiles were padding. Why it is exact: the steel BlockMMA kernels
   compute every output as one sequential fp32 chain over k (tested against 7 summation orders on adversarial inputs:
   only the sequential chain matches, 0 of 1.77M mismatches; `benchmarks/og/ttft2_fma.py`), so every block variant is
   bitwise equal (variants 0-4 checked, real + adversarial inputs). Per layer at 128 rows: gate/up 4.58 -> 3.65 ms,
   down 2.31 -> 1.84 ms.
2. **Two-segment overlap** (`encoder_replay.py`, `DS41_OG_REPLAY_OVERLAP=1`). Prompts just past an 8K boundary replay
   two segments (the finished chunk's 128-row tail, then the short final chunk). Merging them into one pass is NOT
   bitwise (dense projections, the fp32 router GEMM and the routed MoE change kernel with the row count: checked,
   refuted), so segment 1's layer i now runs on a side GPU stream right after segment 0's layer i (the window it
   reads): same ops, same inputs, different stream and order.
3. **First-segment store** (`encoder_replay.py`, `DS41_OG_SEG_CACHE=8` entries, 0 = off). Segment 0 of a two-segment
   replay is a function of tokens[:b], its layer-20 tail rows and layer 20's global rows. A follow-up turn whose final
   chunk is still short (every bench resume, and real short follow-ups after a document) reuses the stored layer 20-39
   windows/slots and DSpark prime context. Checked on use: numerics key, token ids [0,b) bitwise, tail rows bitwise
   (a mismatch recomputes and counts `seg0_rejects`); prompts with images are never stored. Stats: `/og/stats` -> `replay`.
4. **Streaming detokenizer reuse** (`output_parser.py`, `DS41_DETOK_REUSE=1`). Every request's first token paid
   `convert_ids_to_tokens(range(129,280))` (25-45 ms on the engine thread) to build mlx-lm's BPE detokenizer. Now one
   template per tokenizer and a shallow copy + `reset()` per session (mlx-lm's documented reset contract; the token map
   is shared read-only). "first step -> first output" drops from 26-33 ms to 3.3 ms on every request.
5. **Admission slices** (`og_model.py`, `DS41_OG_ADMIT_SLICE_MS=90`, 0 = one piece). While other requests decode, a
   new request's import+replay (now a resumable generator: `import_state_steps` / `replay_steps`, with no stream context
   held across a yield) runs in ~90 ms GPU slices between their scheduler steps. With nobody decoding, it is unchanged.
6. Tools: `benchmarks/og/ttft2_*`: replay profiler + digest A/B (`ttft2_replay_prof.py`, `ttft2_ab.sh`, `ttft2_cmp.py`),
   served partial leg (`ttft2_leg.sh`, `ttft2_host.py` stamps), c2 stall client (`ttft2_stall.py`), guard
   (`ttft2_guard.py`: production idle, no other heavy process, kills its whole process tree by PID), runner
   (`ttft2_run.sh`), kernel studies. `fe_report.py` honours `FE_LOGS`. New trace stamp `og.defer_done`.

## Measured

**Replay only** (profiler, real box states, `ttft2_ab.sh`; ms, median of 5; A = ds41-probe, B = this tree):

| case (tokens) | segments | A | B, store off | B, store hit |
|---|---|---|---|---|
| 8217 (8K fresh bench shape) | 128+24 | 279 | 229 | 89 |
| 8290 (resume shape) | 128+97 | 378 | 295 | 165 |
| 8600 | 128 | 201 | 179 | n/a |
| 300 / 180 / 40 | 128 / 128 / 39 | 175 / 181 / 111 | 159 / 161 / 94 | n/a |
| 131162 / 131500 | 2 / 1 | 428 / 233 | 329 / 213 | 188 / n/a |
| 524400 | 128+111 | 495 | 411 | 239 |

**Served TTFT** (fe_bench through this tree's supervisor + partial worker; windows w4/w5; medians):

| | A (prod) | B | delta |
|---|---|---|---|
| 8K fresh | 0.820 s | 0.727 s | -93 ms (replay -57, first output -29) |
| resume 8K | 0.536 | 0.271 | -265 (store hit) |
| resume 128K | 0.638 | 0.410 | -228 |
| resume 512K | 0.960 | 0.728 | -232 |
| identity short prompts (8) | 0.177 | 0.128 | -49 |

8K fresh per stage (ms, A -> B): replay seg0+prime+seg1 287 -> 230; first step -> first output 32.8 -> 3.5; box
prefill+state 449 -> 454 (box side, unchanged).

**Admission stall at c2** (A decoding while B arrives; engine step durations from `TT2_STEPS`, window s5): the step that
admits B took 202 ms (8600) / 251 ms (8217) in one piece. With 90 ms slices those steps take 104-111 ms (with 45 ms
slices, ~65 ms). B pays about +33 ms TTFT at 90 ms and about +50-70 ms at 45 ms. While B's prompt prefills on the box,
A's steps take 140-155 ms each (the box interleaves steps between prefill sub-chunks), so with 90 ms slices the Mac
no longer causes the largest gap. Client-side chunk gaps are dominated by omlx's 100 ms single-request decode burst,
so use the step logs rather than those. c4 was not run (<= 2 box sessions rule); the slicing logic does not depend
on the concurrency level.

## Identity

- Replay state digests (all layer 20-39 slots + DSpark prime keys), A vs B: 9/9 cases identical with the store off,
  with overlap, and on the store-hit path (`ttft2_cmp.py`, labels bm8/ovl1/seg1 in ~/llm/ds41/ttft2/).
- Served partial-load fe_bench `--identity`: 12/12 identical full streamed outputs A vs B; prompt-id and imported-state
  digests equal 12/12 (doc8k_turn2 took the store-hit path in B); 30 distinct prompts, one state digest each.
- c2 concurrent == alone (fixed prompts, stall client): A and B texts identical alone vs concurrent in every run
  (legs s2, s3, s4, s5; slices 0/45/90 ms).
- fe_cmp vs prod-id2 needs the full model. That is the coordinator's gate at deploy (partial-load outputs are
  aliased-layer text).
- Tests: `tests/test_deepseek_v41_og_ttft2.py` 15/15. They cover: sequential == overlap == sliced steps for 9 prompt
  geometries; store hit == recompute; rejects on other tail rows, prefix or numerics; LRU; admission slices including
  the failure path; detokenizer reuse == fresh on 45 streams including multibyte and DSML. Related suites: og_cache
  (fixture updated for `replay_steps`), og_ttft, og_fused, og_wake, tool_output, spec_probe and ced give 89 pass;
  og_serve test_import_fallback 8/8 and test_wire_failover all pass.

## Tried, not kept
- One merged 152-row pass for two segments: not bitwise (M-dependent kernels), refuted numerically.
- Custom MoE kernels using the proven sequential-chain arithmetic (all bitwise, 0 mismatches):
  - scalar row kernel: 19 ms, too slow;
  - direct simdgroup-MMA without weight staging: 3.9 ms vs 3.65 for bm=8;
  - small-block hybrid (scalar for experts with <= S rows, steel for the rest): no gain at S=1, slower at S>=2.

  The 128-row MoE is MMA-compute bound (~9.5 T FMA/s including padding), so bm=8 is kept.
- Side-stream replay concurrent with decode: no gain (MLX async backpressure serializes submission).

## Remaining levers (not done)
- Box side: stream the finished chunk's tail rows with the first TENS part, so a two-segment replay starts at
  t_first_part. For 8K fresh that is the 72 ms between first part and end; needs a box change.
- Packed attention is latency bound at replay row counts (grid = rows; 1.3 ms/layer, flat from 24 to 128 rows). A
  head-split grid is per-row bitwise but needs a native kernel rebuild (~-10..-20 ms per segment).
- 512K and up: ~20-35 ms of token checks and array import on the engine thread could move to the Job thread.

## Deploy (coordinator)
1. `~/llm/llama-swap.yaml`, model ds41, env `DS41_TREE=/Users/ian/src/wt/ds41-ttft2` (keep `DS41_OG_CACHE_GIB=16`); reload.
2. Gates: `fe_bench.py --base http://127.0.0.1:8080 --model ds41 --label post-ttft2 --identity` and
   `fe_cmp.py prod-id2 post-ttft2` -> 12/12; fixed-prompt c1/c2/c4 identical vs ds41-probe; `/og/stats` replay.seg0_rejects 0.
3. Rollback: `DS41_TREE=/Users/ian/src/wt/ds41-probe`. Per feature: `DS41_MOE_BM8=0`, `DS41_OG_REPLAY_OVERLAP=0`,
   `DS41_OG_SEG_CACHE=0`, `DS41_DETOK_REUSE=0`, `DS41_OG_ADMIT_SLICE_MS=0`.

## Resume steps (work was interrupted by a shutdown)
- The final replay A/B re-run (`ttft2_ab.sh final` and `ttft2_ab.sh final-noseg DS41_OG_SEG_CACHE=0`) was killed by
  the shutdown. The last complete evidence is w4/w5/s5, which ran the committed code except for one removed unused
  import in pipe_decoder. Re-run both A/Bs (expect digests 9/9); nothing else is outstanding.
- Raw logs: `~/llm/ds41/ttft2/` (jsonl per label, `fe/<window>/` supervisor and worker logs, drive*.out).
