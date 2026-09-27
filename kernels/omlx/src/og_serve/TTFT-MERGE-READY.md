# ds41-ttft: time to first token quick wins (ENHANCEMENTS #2) — MERGE-READY

Branch `ds41-ttft` on `51233501` (the served ds41-draft tree). Worktree `~/src/wt/ds41-ttft`, with the 16
gitignored compiled kernels copied from ds41-draft (`cp -p`, `cmp` clean). Not deployed.
Box during all runs: `4dd01ac` (og-s4.4).

## Changes

1. **First token before the MTP post-init** (`omlx/patches/mlx_lm_mtp/batch_generator.py`, `og_model.py`).
   Before, the first sampled token (main_tok) went out only after `_post_init_mtp`: a second 1-row forward
   (box STEP + wire + Mac k=1), the first DSpark draft and the CopyIndex build. Now a fresh singleton on a
   host that opts in (`_omlx_mtp_early_first`, ds41-og only, request-preserving hosts only) emits main_tok at
   once through the normal `_emit_response` (same token, logprobs, length/stop checks). The og host pre-sends
   the post-init forward's STEP right away (`mtp_first_presend`; `ensure_step` finds it in flight). The next
   call runs the unchanged post-init with `main_emitted`: the queue holds only next_main, the draft budget
   (`max_tokens - num_tokens - len(queue) - 1`) and the copy index come out the same. Early markers travel with
   `extend` and are dropped by `filter`. If a post-init ever failed after an early emission, the standard step
   raises instead of emitting main_tok twice. A join between the early emission and the post-init (c2 start)
   goes through the batch post-init, which honours the marker.
2. **Copy-draft index built while the box prefills** (`copy_draft.PromptIndex`, `CopyIndex.from_prompt`,
   `og_model.CopyPrompt`). A side thread started with the box OPEN hashes and sorts the prompt n-grams; the
   post-init inserts the one new n-gram (the stable argsort puts it after equal keys), which gives exactly the
   `CopyIndex(prompt + [main])` arrays. Critical path at 1M: 96 ms -> 2 ms (512K 44 -> 1 ms, 128K 9 -> 0.3 ms).
   Checked against the request's tokens (length, first/last 256); any mismatch builds it locally as before.
3. **Replay host syncs**: `DS41_OG_REPLAY_EVAL_EVERY` default 1 -> 20 (one wait per replay segment instead of
   one per layer; same kernels on the same inputs).

Flags (default on): `DS41_OG_EARLY_FIRST=1`, `DS41_OG_COPY_PREBUILD=1`, `DS41_OG_REPLAY_EVAL_EVERY=20`.
In trace mode `/og/fe` also switches `early_first`, `copy_prebuild` at runtime (fe_window phases).

## Measured per stage (ms, medians; fe_report.py joins client, supervisor and worker stamps)

**Full model, window `ttft-w1`** (one worker, phase A = all three off = the served behaviour, B = on;
`--ttft 8192:3 --resume 8192:3,131072:3,524288:2 --identity`):

| stage | 8K fresh A | B | resume 8K A | B | resume 128K A | B | resume 512K A | B |
|---|---|---|---|---|---|---|---|---|
| box ack + prefill/state (box, noise ±30) | 490 | 518 | 93 | 138 | 171 | 187 | 369 | 374 |
| replay seg0 | 236 | 218 | 224 | 212 | 235 | 220 | 256 | 247 |
| replay seg1 | 85 | 79 | 185 | 178 | 198 | 188 | 216 | 206 |
| first step -> first output | **51** | **26** | **50** | **26** | **61** | **26** | **100** | **26** |
| TTFT (client) | 908 | 895 | 581 | 584 | 722 | 675 | 1070 | 1010 |
| TTFT minus box stages | 418 | 377 | 488 | 446 | 551 | 488 | 701 | 636 |

Short prompts (8 identity cases < 2K tokens): TTFT 215 -> 181. conv60k: 373 -> 329.
The Mac-side saving is -41 ms (8K), -42 (resume 8K), -63 (128K), -65 (512K); client medians move less
where the box stages happened to be slower in B (B ran after A; same box, same code on the box).
1M (not run: window time / partial memory): first step -> output grows ~+0.1 s per 1M before (CopyIndex), flat
26 ms after, so about -0.15 s there.

**Partial load (31-33 GB, production idle, one box session at a time), legs ttft-p1 A (51233501) vs B:**
first step -> output 51/51/63/100 -> 26/26/26/27 ms (8K fresh / resume 8K / 128K / 512K); replay segments
-10..-45 ms; TTFT 956/653/789/1033 -> 837/539/651/990 ms.

## Identity

- Full model, window ttft-w1: `fe_cmp.py prod-id2 ttft-w1-A ttft-w1-B` -> **12/12 identical** (all API shapes,
  image, 60K tools conversation, 8K doc + turn 2); prompt-id and imported-state digests equal A/B for all 12
  cases (EVAL_EVERY 20 bitwise), 40 prompts each with one state digest.
- Partial load, base tree vs new tree: identity suite 12/12 identical, imported-state digests 12/12 equal;
  DSpark draft fingerprints (sha over every draft block) equal for all 20 deterministic requests
  (the copy index from the prebuilt path proposes the same blocks); `batch_identity.py` groups 1 and 2:
  each prompt alone == 2 concurrent (4/4) in both trees, and texts equal across trees.
- c3/c4 (fused pairs) not run here (<= 2 box sessions rule); fused verify itself is untouched (rows with a
  queued token skip verify exactly as before), but it is part of the coordinator's final check.
- Tests: `tests/test_deepseek_v41_og_ttft.py` (12: PromptIndex == constructor state and behaviour on 24
  random prompts incl. 2-token vocab and NGRAM-edge lengths; tiny qwen3_5 MTP model through a real
  GenerationBatch: same token stream, first token one forward earlier, max_tokens 1/2/3, stop on the first and
  second token; extend carries markers; standard step refused after an early emission). Related suites
  (mlx_lm_mtp_patch, mtp_prompt_priming, og_cache, og_fused, og_wake, dspark, thinking_budget, og_images):
  same failure set as 51233501 (49 pre-existing, 439 pass). og_serve `test_import_fallback` 8/8,
  `test_wire_failover` all pass.

## The ~1 s/1M client-vs-server TTFT gap (item 4): found, not changed

Reproduced with bench-leg2-shaped requests (one user message = the same doc + a different question;
`benchmarks/og/ttft_gap.py`): 512K client 1.75-1.90 s vs server `time_to_first_token` 1.32-1.46 s.
Stage trace: **`count tokens` 404 ms at 512K (92 ms at 128K), ~0.77 µs/token**. The FE piece cache cuts at
template tokens, so a changed message is one new piece and `count_chat_tokens` re-encodes it single-threaded
before the handler starts the server clock. HTTP is not it: a private llama-swap -> supervisor -> fake worker
chain adds 10 ms at a 4.3 MB body (direct 1.8 ms). Multi-message conversations (turn N+1 = old pieces + new
ones) do not pay it (fe_bench resume: pre-admission 37 ms at 512K).
Fix (not quick, identity-sensitive): split large pieces at provable pre-tokenizer boundaries (newline followed
by non-whitespace, no added token spanning or lstrip-eating the cut) and encode the parts in parallel
(encode_batch), with the FE delimiter-safety checks extended to those cuts; est. 1M -0.6..-0.7 s.
Same requests also show the box re-prefilling up to one snapshot chunk (box prefill+state 639 ms at 512K vs
96 ms for an appended turn): box-side, for the box agent.

## Deploy (coordinator)

1. `~/llm/llama-swap.yaml`, model `ds41`, env `DS41_TREE=/Users/ian/src/wt/ds41-ttft` (keep
   `DS41_OG_CACHE_GIB=16`); reload ds41. Kernels are already in the tree.
2. Gates: `cd ~/src/wt/ds41-ttft && ~/llm/.venv-ds41-omlx-tiles/bin/python og_serve/fe_bench.py --base
   http://127.0.0.1:8080 --model ds41 --label post-ttft --identity && og_serve/fe_cmp.py prod-id2 post-ttft`
   -> 12/12; c2/c4 fixed-prompt identity as for ds41-draft. `/og/stats`: `first_presend` ~ requests,
   `copy_prebuilt` ~ greedy requests >= 64 tokens, `presend_errors` 0.
3. Rollback: `DS41_TREE=/Users/ian/src/wt/ds41-draft`. Per feature: `DS41_OG_EARLY_FIRST=0`,
   `DS41_OG_COPY_PREBUILD=0`, `DS41_OG_REPLAY_EVAL_EVERY=1`.

## Runs and windows

- Partial loads (`benchmarks/og/ttft_ab.sh`: this tree's supervisor on :12698 with a `host_server.py` partial
  worker on :12697, no gpu.lock, 33 GB RSS): ttft-p1 A/B 23:43-23:47 UTC, ttft-p2 (gap) 23:48, switch smoke
  test. Logs `~/llm/ds41/fe/ttft-p{1,2}/`, `~/llm/ds41/fe/ttft-p*-*.jsonl`.
- One Mac window `ttft-w1`, 23:53:11-23:54:49 UTC (announced in PROGRESS.md, ds41 unloaded via llama-swap, restored by a chat
  request and verified). Logs `~/llm/ds41/fe/ttft-w1/`, `~/llm/ds41/ttft/window-w1.log`.
