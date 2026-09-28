# ds41-probe: stats-only copy-lock pre-send probe (MERGE-READY, not deployed)

This is step 1 of `/home/ian/mac/PRESEND-FEASIBILITY.md` (Linux), variant (b), copy-lock pre-send, approved by the owner. The probe predicts, at every c1 cycle, the STEP that a copy-lock pre-send would send next, and later checks it against the STEP that is really sent. It **sends nothing extra, changes no outputs and does no GPU work**.

Branch `ds41-probe`: code 8476793c plus this note, on `ds41-woa` 1bc1271a (lossless wo_a, merge-ready). Worktree `~/src/wt/ds41-probe`. The 16 gitignored native kernels were copied from `~/src/wt/ds41-woa` with `cp -p` and checked with `cmp` (16/16).

## Change

- **`mlx_lm_mtp/copy_draft.py`: `CopyIndex.speculate(budget)`.**
  - Takes the last `propose()` (source `s`, `k` drafts) and assumes all `k` drafts are accepted and the bonus is the token after the copied span.
  - Runs the real `observe(k)`, `append(drafts + [bonus])` and `propose(budget - k - 1)`, then restores every field.
  - Restored fields: `n`, the policy (`cur`, `min_match`, `expected`, `pending`), the appended `_extra` n-gram entries and the buffer bytes.
  - Returns `(bonus, drafts)` or None. p50 cost is 18 µs at 8K, 128K and 512K prompts.
- **`deepseek_v41/spec_probe.py` (new): per-request `Record` plus global counters in `og_model.STATS`.**
- **`og_model.presend()`: before the send, `observe_step` resolves the pending prediction against the real (keep, ids).**
  - This is a tuple compare.
  - The resolution counts only if exactly one box session is open (`len(SESSIONS) == 1`); a prediction resolved with more sessions open counts as a concurrency miss.
- **`og_model.presend()`: after the send, `predict` runs `speculate` on copy cycles.**
  - The box is computing the step meanwhile, so this is off the c1 critical path.
- **`_remote`:**
  - It observes STEPs that `presend` never saw (first step, fallback paths). Steps it did see are deduplicated.
  - `note_recv` records the box wait of hit steps.
- **`close_request`: one `ds41-og spec-probe <request_id>: ...` line per request.**
- **Flag `DS41_OG_SPEC_PROBE`:** default `1` on this branch. `0` disables every call and leaves the ds41-woa behaviour.
- **Test fix:** `og_serve/test_import_fallback.py` now loads `spec_probe` and stubs `woa_compact`. It already failed on ds41-woa because `woa_compact` was missing from its stub package.

## What it counts

Only c1 cycles are counted: exactly one box session open.

| `/og/stats` key | Meaning |
|---|---|
| `spec_probe_enabled` | 1 if the probe runs |
| `spec_probe_c1_cycles` | Real STEPs sent while exactly one session is open. **This is the denominator.** |
| `spec_probe_copy_cycles` | c1 STEPs whose drafts came from the copy index |
| `spec_probe_would` | Copy cycles after which a copy-lock pre-send would have been sent. The prediction exists: the next block would also be a copy block |
| `spec_probe_no_pred` | Copy cycles whose hypothetical next block is not copy, or does not fit in 5 rows |
| **`spec_probe_hits`** | The next real STEP equals the prediction byte for byte (keep + ids). A pre-send would have hidden that STEP's box round trip |
| `spec_probe_miss_partial` | Not all copy drafts were accepted: the real keep is smaller than predicted |
| `spec_probe_miss_ended` | All accepted, but the bonus left the copied span |
| `spec_probe_miss_dspark` | Same bonus, but the next block came from DSpark |
| `spec_probe_miss_copy_diff` | Copy again, but from another source or with another length |
| `spec_probe_miss_concurrency` | A second session opened before the next STEP |
| `spec_probe_miss_other` | Anything else, e.g. a STEP seen only in `_remote`, where the draft source is unknown |
| `spec_probe_unresolved` | The request ended with a prediction outstanding |
| `spec_probe_errors` | Internal inconsistencies, e.g. `pending` k differs from the number of drafts. Statistics only |
| `spec_probe_c1_cycle_ms` / `_c1_cycles_timed` | Sum and count of c1 cycle times (interval between consecutive c1 STEPs) |
| `spec_probe_hit_cycle_ms` / `_hit_cycles_timed` | The same, only for cycles that verify a hit STEP |
| `spec_probe_hit_wait_ms` / `_hit_waits` | Sum and count of the Mac box wait (`recv_step` `wait_s`) on hit STEPs. This is the time a pre-send would have hidden |

**Identity:** `would = hits + sum(miss_*) + unresolved`, over requests whose predictions all resolved.

**Reading it** (the counters are cumulative since worker start; take differences between two reads for a window):

```text
hit rate (go/no-go)   = spec_probe_hits / spec_probe_c1_cycles
mean c1 cycle         = spec_probe_c1_cycle_ms / spec_probe_c1_cycles_timed
mean saved per hit    = spec_probe_hit_wait_ms / spec_probe_hit_waits - 0.25      (0.25 ms = spec send + buffered read)
projected c1 gain     = 1 / (1 - (spec_probe_hit_wait_ms - 0.25*would) / spec_probe_c1_cycle_ms) - 1
```

`0.25*would` charges every would-be pre-send the Mac overhead, hits included. That is conservative. The projection counts only the c1 wall time the probe saw.

Example from the log:

```text
ds41-og spec-probe <request_id>: c1_cycles=253 copy_cycles=5 would=5 hits=3 hit_pct=1.2 miss[partial=2,ended=0,dspark=0,copy_diff=0,concurrency=0,other=0] no_pred=0 unresolved=0 errors=0 cycle_ms[c1=25.31,hit=21.25] hit_wait_ms=6.79
```

`hit_pct` = hits / c1_cycles for that request. Per-request lines let you split agent/edit requests from chat and prose ones.

Quick read:

```bash
curl -s 127.0.0.1:10001/og/stats | python3 -c 'import json,sys; s=json.load(sys.stdin); print({k[11:]:round(v,1) for k,v in s.items() if k.startswith("spec_probe_")})'
```

`:10001` is the supervisor; the worker is `:12147`.

**Go/no-go:**
- **GO** for step 2, the box 2-level rollback plus the real pre-send, if `hits / c1_cycles >= 15%` over a representative window of real c1 traffic. The feasibility estimate for the production mix is 15-21%, which is about +5-7.5% c1.
- Check at the same time that `hit_wait_ms / hit_waits` is at least ~6 ms.
- Below 15%, stop. Copy-lock would then pay only on the agent/edit requests, which the per-request lines identify.

## Tests

- **`tests/test_deepseek_v41_spec_probe.py`: 29 passed.**
  - **Exactness, 12 seeds.** A synthetic served loop runs against target streams that copy spans of the prompt and of themselves. Whenever the hypothesis held (all k accepted and bonus == predicted bonus), the prediction equalled the real next `propose()`, None included. It was exercised > 5 times per seed.
  - **Identity, 12 seeds.** The loop with `speculate()` every cycle and the loop without it give identical proposal sequences and a byte-identical final index state: buffer, sorted tables, `_extra` including key order, and policy.
  - **Restore.** Repeated `speculate()` calls at 3 budgets leave every field unchanged.
  - **Edge cases.** Periodic self-overlap (the copied span ends at `n`, so the bonus comes from the drafts), no pending proposal, and too small a budget.
  - **Bookkeeping.** hit, partial, ended, dspark, no_pred, not-c1, concurrency miss, a pending/draft mismatch, unresolved, the `_remote` duplicate, the hit wait and cycle timers, and the log line.
- **Related suites:** og_ttft (copy index), og_cache, og_fused, og_wake and woa_compact give 23 passed, 11 skipped. The skips are the known device-order skips of the woa tests in a combined run. `og_serve/test_import_fallback.py` passes 8/8.
- **Served identity, partial-load harness.** `host_server.py` ran the real og_server, scheduler, DSpark/copy loop, presend and box sessions over the aliased partial model at ~30.7 GB, with production idle under `idle_guard`, 1 box session at a time and no windows. `batch_identity.py` sent 4 fixed prompts at c1 (3-12K tokens, 256 tokens each), and `HS_DRAFTS=1` fingerprinted every draft block:

  | Leg | Draft fingerprints (4 requests, blocks) | Texts | Probe counters (c1) |
  |---|---|---|---|
  | probe on | `48cc075a` 251, `df698f20` 252, `f745b3cc` 253, `3b57dcbb` 251 | reference | c1_cycles 1015, copy_cycles 18, would 18, **hits 12**, miss_partial 6, 0 errors, 0 unresolved |
  | probe off (`DS41_OG_SPEC_PROBE=0`) | **identical** | **identical** (4/4, 256 tokens each) | all 0, `enabled` 0, no spec-probe lines |

  - **Cross-check against the MTP lines.** The copy cycles here are 1-token copies (k = 1), and `copy[... accept=a/c]` gives 3/5, 3/4, 2/3 and 4/6. So the full-accept copy cycles are exactly 3+3+2+4 = **12 = hits**, and the 6 partial misses are the rejected copy cycles.
  - **Timing on this harness.** Mean c1 cycle was 25.5 ms, hit cycles 21.5 ms, and the box wait on hit steps was 7.1 ms. The box wait is what a pre-send would hide.
  - **Timing neutrality.** Summed backbone time was 21,313 ms with the probe on and 21,228 ms off, i.e. +0.4%. That is one leg each and within run-to-run noise.
  - **Isolated cost.** The probe's host work on a non-copy cycle is **1.2 µs**: observe before the send, the `_remote` dedupe, `note_recv` and the predict early-out. Measured with a 20K-cycle CPU loop.

The text is meaningless because the layers are aliased, so DSpark acceptance is ~1%. The model still falls into repetition, so the copy index fires and the probe resolves real predictions end to end.

## Cost

- **Host time per c1 cycle:**
  - One tuple compare and two `perf_counter` calls before the send (µs).
  - `speculate` after the send: 18 µs p50, 33 µs p99, measured at 8K/128K/512K prompt indexes. It runs while the box computes the step.
- **What it avoids:** MLX evals, extra syncs and box traffic.
- **Memory:** one small record per open request.

## Deploy / rollback

- **Deploy:** set `DS41_TREE=/Users/ian/src/wt/ds41-probe` in the llama-swap ds41 env (keep `DS41_OG_CACHE_GIB=16`), then reload.
- **Gates:**
  - `/og/stats` shows `spec_probe_enabled: 1`, and `spec_probe_c1_cycles` grows during a c1 request.
  - og-child.log has one `spec-probe` line per finished request.
  - The usual fixed-prompt identity check matches (`og_serve/fe_bench.py --base http://127.0.0.1:8080 --model ds41 --label post-probe --identity`, then `og_serve/fe_cmp.py <prod-label> post-probe`), plus c1/c2/c4 fixed-prompt outputs. Outputs must be identical.
  - Decode tok/s at c1/c2/c4 unchanged within noise versus ds41-woa. This tree includes ds41-woa, so compare against ds41-woa if that is live, or expect the woa gain on top of ds41-ttft.
- **Rollback:**
  - `DS41_OG_SPEC_PROBE=0` (same tree, probe off; behaviour = ds41-woa).
  - `DS41_TREE=/Users/ian/src/wt/ds41-woa`.
  - `DS41_TREE=/Users/ian/src/wt/ds41-ttft` (current production).

## Reproduce

```bash
cd ~/src/wt/ds41-probe; PY=~/llm/.venv-ds41-omlx-tiles/bin/python
$PY -m pytest -q tests/test_deepseek_v41_spec_probe.py
$PY og_serve/test_import_fallback.py
zsh ~/llm/ds41/probe/run_ab.sh on off    # waits for 60 s of production idle; partial load under idle_guard, c1 only
```

Logs are in `~/llm/ds41/probe/`: `server-probe-{on,off}.log`, `stats-probe-{on,off}.json` and `identity.jsonl`.
