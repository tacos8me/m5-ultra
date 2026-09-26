# ds41-draft: MERGE-READY (batched DSpark drafter per fused pair)

Branch `ds41-draft` off `8fe0a2df` (ds41-batch, live). Not deployed.

## What changed
- **Drafting a pair together.** After a fused pair verifies (og_fused, >= 3 requests decoding), each request's chain queues its draft instead of running it (`draft_jobs`). Then `og_model.mtp_draft_jobs` drafts both, one DSpark decoder pass for the pair, and pre-sends both next STEPs.
- **What the shared pass shares** (`dspark.proposal_forward_batch`):
  - Shared: the 3 stages' MXFP8 attention and shared-expert projections, BF16 wo_a, the MXFP4 expert launch, and the BF16 draft head.
  - Per request: the context caches, attention, router, mHC, embedding, and the Markov sampling loop.
- **Refactor, same ops and order:** `_dspark_next_drafts` is now `_dspark_prepare` + forward + `_dspark_finish`. `dspark_draft_jobs` runs the prepares, one shared forward (`host.dspark_forward_batch`, which returns None and does nothing when not eligible), then the finishes. Two details keep each request's draft its own:
  - It uses each request's controller depth from before this cycle's `observe`, because the chain queues the draft after observe.
  - It applies the head-cache trim the chain skips when it queues a draft.
- **Eligibility:** the requests share one width of 2-4, at most 8 rows in total, the BF16 draft head is on, and the og_fused kernels are on. Otherwise drafts run per request.
- **c1 and c2 are unchanged:** the jobs path exists only inside fused groups.
- **Flag:** `DS41_OG_DRAFT_BATCH=1` by default; 0 drafts per request with the ds41-batch behaviour. `/og/stats` gains `draft_batches`.

## Identity
- **Decoder pass** (`batch_bench.py BB_TESTS=draft`, partial load, real boundaries): widths 4+4, 2+2, 3+3 and 2+2+2 were checked. Draft logits and all stage context caches were **bitwise equal** to each request's own `dspark_forward`.
- **Served** (host_server partial load, 4 fixed prompts, temperature 0, `HS_DRAFTS=1` fingerprints every verified draft block per request):

  | run | batched draft passes | fingerprints | texts |
  |---|---|---|---|
  | alone, then c4 and c3 (flag on) | 460 at c4, 231 at c3 | the same 4 on every run | identical |
  | flag off (alone, c4) | 0 | the same 4 | identical to flag on |
  | fixed 5-row depth, alone vs c4 | 465 | c4 equal to alone | identical |

  - The 4 fingerprints are the ones ds41-next and ds41-batch gave earlier for these prompts.

## Speed (partial load, c4 8K, fixed 5-row cycles, copy drafts off, 3 reps each, same tree, flag A/B)
- **Cycles/s:** 51.8 -> **54.7** while all 4 decode (**+5.6%**); 48.3 -> 50.7 over the whole run (+5.0%).
- **Mac DSpark decoder pass** (excluding Markov sampling): 2x4 rows 6.07 -> 4.76 ms; 2x2 5.71 -> 3.97; 3x2 7.93 -> 4.58.
- **Expected on the real model:** about +5% at c3/c4 on top of ds41-batch. c1/c2: same code path.

## Tests
- og tests (og_fused, og_cache, og_wake) pass; `og_serve/test_import_fallback.py` passes 8/8.
- `tests/test_deepseek_v4_dspark.py` has the same 12 failures as ds41-batch; they are pre-existing.

## Deploy / rollback
- **Deploy:**
  - Set `DS41_TREE=/Users/ian/src/wt/ds41-draft` in llama-swap ds41 (keep `DS41_OG_CACHE_GIB=16`), then reload.
  - The 16 kernel artifacts are already copied into `~/src/wt/ds41-draft` and checked with `cmp`.
  - Gate: fixed-prompt identity at c1/c4. `/og/stats` should show `draft_batches` > 0 at c4.
- **Rollback:** `DS41_OG_DRAFT_BATCH=0` (per-request drafts, as ds41-batch), or `DS41_TREE=/Users/ian/src/wt/ds41-batch`.
