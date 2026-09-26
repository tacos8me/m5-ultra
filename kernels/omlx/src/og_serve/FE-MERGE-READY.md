# ds41-fe: request front-end overhead — MERGE-READY

Branch `ds41-fe` on `f56f7ffa` (the served ds41-og tree). Worktree `~/src/wt/ds41-fe`.
Not deployed. Box during all measurements: `1954bf6`, numerics `og-s4.4`.

## Result

| streamed TTFT, s | production f56f7ffa (measured) | ds41-fe (projected, see method) | target |
|---|---|---|---|
| 8K fresh | 0.984 | ~0.92 | 0.75 (not reached: box + replay bound, below) |
| resume 8K (+~60 tokens) | 0.709 | ~0.66 | - |
| resume 128K | 1.119 | ~0.73 | 0.9 met |
| resume 512K | 2.804 | ~1.20 | 1.5 met |
| short prompts (12-344 tokens) | 0.31-0.48 | ~0.03 lower | - |

Production numbers: `fe_bench.py` through llama-swap (`prod-base1`, 18:22 UTC, medians of 3/3/3/2).
The projection subtracts, from those measured production requests, only the stage savings measured A/B in one
worker process (window `fe-w2`, below) plus the post-window `prepare_inputs` saving measured on the real engine
code path on CPU: pre-admission 512K −1.576 s, 128K −0.357 s, 8K resume −0.021 s, 8K fresh −0.029 s;
engine wake −~20 ms (the removed poll wait is uniform 0-50 ms); kickoff presend −10 ms. Allow ±25 ms.

## Where the time went (production baseline)

Four full re-tokenizations of the whole prompt before the scheduler saw the request: `count_chat_tokens` (event
loop), `preflight_chat` (event loop), the streaming path's thinking-state probe (render + encode), and mlx_vlm's text-only
`prepare_inputs`, which ignores `tokenizer.encode` and calls the tokenizer's `__call__` (Rust encode + a Python
flatten into an int32 [1, N] array) on the MLX thread. At 512K each was ~0.38-0.43 s: 1.6 s of the 2.8 s. JSON,
Pydantic, template rendering and the supervisor were each < 3 ms at 512K. Then the engine loop noticed a finished
box OPEN only on its next 50 ms poll, and the first step waited a full box round trip after the import.

## What changed

- `omlx/patches/deepseek_v41/fast_encode.py` (new): incremental tokenization, exactly equal to a full encode.
  A HF fast tokenizer cuts its input at added-token matches and encodes each remaining piece on its own; cutting
  the prompt at the template tokens (`<｜User｜>`, `<｜Assistant｜>`, `</think>`, EOS, ... 8 delimiters, each checked
  at startup against all 1,283 added tokens for overlap/containment/extension, plus lstrip/rstrip/single_word flags,
  an identity-normalizer probe, `add_special_tokens` no-op check and a self-check) gives
  `encode(prompt) == concat(encode(pieces))`. Pieces are cached (LRU, `DS41_FE_ENCODE_CACHE_MB`=256), so turn N+1
  encodes only its new messages; misses go through one `encode_batch` (parallel), so even a fresh 512K
  conversation encodes 7x faster (51 vs 358 ms). Installed as a per-instance `tokenizer.encode`; each tokenizer copy
  (omlx deep-copies it for the event loop and the scheduler) encodes on its own private Rust backend clone with
  padding/truncation off (HF `__call__` leaves padding enabled on the shared backend, which would pad batched
  misses) and shares the piece cache. `install_prepare_inputs()` routes mlx_vlm's text-only `prepare_inputs` for
  the DS41 `Processor` through the cache and returns the same int32 [1, N] ids / ones mask (0.43 s -> 7 ms at 512K).
  Images/audio/video, several prompts or any other argument go to the original.
- `og_model.py`: installs the above for the og model; a Job that finishes its box OPEN marks the scheduler and wakes
  the engine loop (`opened_wake`/`opened_step`: an OPEN finishing during a step makes that step report work, so the
  wake cannot be lost to the loop's clear); the kickoff STEP (the last prompt token at N-1) is sent right after the
  OPEN (`DS41_OG_KICKOFF`), so the box computes it during the Mac import and the first forward finds it in flight
  (`ensure_step`), bit-identical bytes either way.
- `ds41_og.py` (supervisor): forwards the client's JSON bytes when the body would not change (model id already the
  child's, no resume) instead of `json.loads`+`json.dumps` of every request; env-gated stamps.
- Tracing (all off by default): `fe_trace.py` (`DS41_FE_TRACE=1`: one JSON line per request with ~30 wall-clock
  stamps across supervisor, HTTP, handler, count, preflight, render+tokenize, scheduler, box ACK/state, import
  arrays, replay per segment, prime, first step, first output, first SSE byte; `DS41_FE_PROFILE` cProfile of the
  scheduler work of >=100K prompts; runtime A/B switches at `POST /og/fe` in trace mode only), `DS41_FE_DIGEST=1`
  (sha256 of the prompt ids and of every replayed Mac-side state array after each import), `pipe_wire` ACK/first
  part/end times, `encoder_replay` per-segment marks and `DS41_OG_REPLAY_EVAL_EVERY` (default 1 = unchanged).
- Tools: `fe_bench.py` (TTFT fresh/resume + a 12-case identity suite over chat, system multi-turn, thinking, tools,
  tool results, Anthropic `/v1/messages`, `/v1/responses`, an image, a 60K multi-message tool conversation and its
  turn 2, an 8K document and its turn 2), `fe_cmp.py`, `fe_report.py` (joins client/supervisor/worker stamps
  per stage), `fe_window.py` (guarded Mac window: idle wait, PROGRESS announce, llama-swap unload, private
  supervisor on :12648, runtime A/B phases, TERM by PID, restore + answer check), `fe_conv.py` (randomized
  conversations).

## Measured per stage (window fe-w2, one worker, phases switched at runtime; medians, ms)

A = baseline paths (encode cache, kickoff, wake off), B = all on. C/D = B + replay eval every 4/20 layers.

| stage | 512K resume A | B | 128K resume A | B | 8K resume A | B | 8K fresh A | B |
|---|---|---|---|---|---|---|---|---|
| count tokens (event loop) | 392 | 5.6 | 88 | 1.5 | 5.4 | 0.2 | 14.1 | 15.1 (cold) |
| preflight (event loop) | 383 | 5.2 | 86 | 1.3 | 4.8 | 0.1 | 12.7 | 0.3 |
| thinking-state probe (stream path) | 392 | 7.1 | 87 | 2.2 | 5.3 | 0.5 | 10.7 | 1.5 |
| render+tokenize (MLX thread) | 435 | 432* | 101 | 102* | 7.0 | 7.2* | 11 | 16* |
| = pre-admission | 1615 | 464* | 371 | 113* | 26 | 12* | 58 | 44* |
| engine poll gap (->prepare) | 17 | 0.3/50** | 33 | 0.3 | 43 | 0.3 | 30 | 0.3 |
| prepare -> first step | 33 | 23 | 25 | 15 | 21 | 11 | 21 | 11 |
| box ACK + state | 499 | 385 | 234 | 186 | 114 | 108 | 489 | 486 |
| replay seg0 + seg1 (+prime) | 601 | 602 | 561 | 555 | 534 | 536 | 421 | 427 |
| first step -> first output | 100 | 100 | 62 | 64 | 51 | 52 | 51 | 52 |
| = TTFT (client, direct) | 2937 | 1673 | 1325 | 964 | 800 | 731 | 1093 | 1025 |

\* before the post-window `prepare_inputs` path: the same engine code path on CPU (`VLMBatchedEngine.
_process_chat_messages`, warm pieces) takes 7.3 ms at 512K, 1.9 ms at 128K, 0.5 ms at 8K (0.438 s at 512K
without it, matching the window's 432-435 ms). ** a lost wake (fixed afterwards by `opened_step`). Box stages vary
by ±0.1 s run to run at 512K (A/D slow, B/C fast, same code): not attributed. Other sizes: resume 100K 977 -> 689.
Replay eval every 4/20 layers: −5..−25 ms per segment, outputs and state digests identical; left at 1 (default):
small gain, and validated only on the generic-kernel path. The window's absolute model-path times are ~0.1 s slower than production's because that worktree
lacked the gitignored native kernels (see identity).

## Bit identity

- Tokenization: `tests/test_deepseek_v41_fast_encode.py` (459 tests): 120 random conversations x 3 turns through one
  shared cache (chat/thinking, reasoning effort, tools, tool calls/results, CJK/emoji/digits/whitespace, literal
  special tokens in content), 300 fuzz strings of delimiters and delimiter fragments, a 400K-char conversation's
  turn 2 (only the new text missed), tool call/result rendering, images through `Processor.__call__` (ids and
  spans), the real `VLMBatchedEngine` text path (`_process_chat_messages` on the MLX-thread tokenizer and
  `count_chat_tokens` on a deep copy, 24 conversations x 2 turns), `prepare_inputs` fast path vs the original on 40
  prompts (values, int32 dtypes, shapes, mask), deep copies binding their own backends, concurrent encodes from two
  copies, HF-`__call__`-left padding not leaking, unsafe delimiters rejected, non-identity normalizer refused, LRU
  bound, env off switch. All pass.
- Served path, window fe-w2: the 12 identity cases gave identical full streamed output (content, reasoning, tool
  calls) in A, B, C and D, and identical prompt-id and imported-state digests per case in all four phases
  (12/12); 122 imports, 73 distinct prompts, each with exactly one state digest.
- Versus production: in that window 5/12 cases differed from production for every phase, the baseline-equivalent A
  included. Cause: a new worktree has none of the gitignored native kernels (`omlx/custom_kernels/*/_ext*.so`,
  `*.dylib`, `*.metallib`), so glm_moe_dsa's DS41 attention/expert kernels fell back to generic MLX paths.
  Production is self-consistent (`prod-base1` 18:22 == `prod-id2` 19:15 via llama-swap == `prod-direct` to its
  supervisor, 12/12). The 16 artifacts are now copied into `~/src/wt/ds41-fe` (sha256-identical to ds41-og's;
  kernel sources unchanged vs f56f7ffa) and `glm_fast.is_native_available()` is True there.
- Everything in the deployed configuration is arithmetic-neutral by construction: the same prompt ids (proven above),
  the same box bytes (kickoff only sends the identical STEP earlier; box step == prefill), engine wake and body
  forwarding are scheduling only, tracing and digests are off, replay evaluation unchanged (every layer).
- Validated offline only (after the last GPU window): the `prepare_inputs` text path, private backends, the
  `opened_step` wake. Hence the post-deploy identity gate below.

## Tests

- New: `tests/test_deepseek_v41_fast_encode.py` (459), `tests/test_deepseek_v41_og_wake.py` (1): pass.
- og_serve: `test_import_fallback.py` 8/8 (harness now also loads fe_trace/fast_encode), `test_wire_failover.py` all
  pass, `test_supervisor_failover.py` 45/45 (`SUP_TEST_PORT=12610 FAKEBOX_PORT=12630`).
- Related pytest (og_images, og_cache, deepseek_v41, prefix_cache, ced, tool_output, template_append_only,
  thinking_toggle, prefill_glue, anthropic_adapter, chat_tool_call): 194 pass, 17 fail identically on the base tree
  f56f7ffa (16 `test_rope_range_matches_rope` order-dependent in a combined run, pass alone; 1 flaky sorted-prefill).

## Bounds (why 8K fresh stays ~0.92 s)

For the 8,217-token bench prompt production spends: box OPEN 0.52 s (box prefill 0.45 s), Mac tail replay 0.32 s
(two segments: the 8K grid chunk's last 128 rows + 24 rows; MoE-weight-bandwidth bound at ~0.15-0.2 s per 128-row
sweep of layers 20-39), first step 0.01 s, first step -> first token 0.05 s (omlx emits the first token only after
the MTP post-init: a second forward at the first token + drafter + copy index, 0.05-0.1 s), front end ~0.03 s.
Next levers, none front-end: emit the first token before the post-init forward (~0.03-0.05 s on every request;
batch_generator change), let the box stream the tail hidden of finished chunks so segment 1 replays during the last
chunk (~0.1 s for prompts just past an 8K boundary; box change), faster 128-row replay kernels.

## Deploy (coordinator)

1. Native kernels must be in the tree (done for `~/src/wt/ds41-fe`; for a fresh checkout:
   `cd ~/src/wt/ds41-og && git status --ignored --short | grep '^!! omlx/custom_kernels/.*\.\(so\|dylib\|metallib\)$' | cut -c4- | while read f; do cp -p "$f" ~/src/wt/ds41-fe/"$f"; done`).
2. `~/llm/llama-swap.yaml`, model `ds41`, env: `DS41_TREE=/Users/ian/src/wt/ds41-fe` (branch ds41-fe HEAD), keep
   `DS41_OG_CACHE_GIB=16`; reload ds41.
3. Gates: `cd ~/src/wt/ds41-fe && ~/llm/.venv-ds41-omlx-tiles/bin/python og_serve/fe_bench.py --base http://127.0.0.1:8080 --model ds41 --label post-deploy --identity`
   then `og_serve/fe_cmp.py prod-id2 post-deploy` must print `12/12 identical` (reference: production f56f7ffa,
   box 1954bf6/og-s4.4; after a box numerics change re-take it on ds41-og first). Timing:
   `fe_bench.py ... --label post-deploy-t --ttft 8192:3 --resume 8192:3,131072:3,524288:2`.
   `/og/stats`: `kickoff_sent` == `opened`, `kickoff_errors` 0, `encode.hits` growing, `encode.fallbacks` ~0.
4. Rollback: `DS41_TREE=/Users/ian/src/wt/ds41-og` + reload. Per feature, without a tree change:
   `DS41_FE_ENCODE_CACHE=0` (piece cache and the prepare_inputs path), `DS41_OG_KICKOFF=0`, `DS41_OG_WAKE=0`.

## Windows used

`fe-w1` (18:35-18:44 UTC): no data; the worker did not start (the piece cache was not deep-copyable; fixed, tests
added), a ds41 request reloaded production meanwhile. `fe-w2` (19:06-19:12 UTC): phases A/B/C/D + a profile phase
on one private worker; a tracing closure bug killed the first warm-up, the worker was restarted inside the same
announced window. Logs: `~/llm/ds41/fe/fe-w2/` (supervisor + worker traces), `~/llm/ds41/fe/fe-w2-*.jsonl`,
production references `~/llm/ds41/fe/prod-{base1,id2,direct}.jsonl`.
