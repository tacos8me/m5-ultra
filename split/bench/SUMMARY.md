# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-27, box engine 4dd01ac, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,238 | 0.93 (0.90-0.94) | 8,883 | 18,307 | 3 | 3.29 | 2,498 | 3.6x |
| 16K | 16,430 | 1.34 (1.33-1.35) | 12,261 | 19,179 | 3 | 6.38 | 2,573 | 4.8x |
| 32K | 32,812 | 2.29 (2.23-2.40) | 14,320 | 19,495 | 3 | 12.56 | 2,610 | 5.5x |
| 64K | 65,581 | 3.94 (3.94-3.95) | 16,639 | 19,754 | 3 | 25.30 | 2,591 | 6.4x |
| 128K | 131,115 | 7.51 (7.48-7.53) | 17,449 | 19,667 | 3 | 51.09 | 2,545 | 6.9x |
| 256K | 262,192 | 14.78 | 17,744 | 19,166 | 1 | 111.08 | 2,360 | 7.5x |
| 512K | 524,332 | 30.37 | 17,264 | 18,238 | 1 | 234.65 | 2,234 | 7.7x |
| 768K | 786,477 | 48.33 | 16,275 | 17,079 | 1 | 376.24 | 2,090 | 7.8x |
| 1M | 1,040,048 | 66.45 | 15,651 | 16,330 | 1 | 520.53 | 1,998 | 7.8x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.89-0.94 s (n=5), 128K 7.46-7.53 s (n=5), 256K 14.71-14.78 s (n=2), 512K 30.37-30.47 s (n=3), 1M 66.45-66.64 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,238 | **96.4** | 87.0-111.1 | 75% | 3.01 | 7.41 | 13.90 | 71.7 (5) |
| 8K free-prose | 8,250 | **79.7** | 69.5-89.0 | 65% | 2.47 | 7.28 | 13.35 | 66.7 (2) short essay |
| 128K code-summary | 131,117 | **92.0** | 85.5-97.3 | 74% | 2.90 | 7.38 | 14.61 | 66.2 (1) |
| 256K code-summary | 262,191 | **94.7** | 84.0-109.9 | 75% | 3.01 | 7.39 | 14.67 | 63.5 (1) |
| 512K code-summary | 524,336 | **97.7** | 80.2-112.3 | 78% | 3.18 | 7.49 | 15.89 | 65.6 (4) |
| 1M code-summary | 1,040,046 | **91.2** | 75.3-102.9 | 76% | 3.04 | 7.54 | 15.90 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 64.7 (55.7-73.8) | 112.5 (111.9-113.7) | 0.90 / 1.90 | 0.97 / 2.07 |
| c2 128K | 64.4 (54.2-76.6) | 107.2 (103.1-112.5) | 1.32 / 2.52 | 7.56 / 17.58 |
| c4 8K | 40.1 (30.8-50.1) | 117.3 (114.5-119.8) | 0.92 / 2.11 / 3.17 / 4.19 | 0.94 / 2.08 / 3.16 / 4.17 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 decodes all four streams at once as two fused pairs; its later arrivals wait only for the earlier prefills (see TTFT by arrival order). This leg uses fresh nonce prompts, so its aggregate varies with draft acceptance; a fixed-prompt identical-output A/B gave c4 113 -> 139 tok/s for fused pairs + batched drafting vs the previous build.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.89 | 0.65 (0.60-0.70) | 71-74 | - | - |
| 128K | 7.49 | 0.80 (0.79-0.83) | 69-72 | 0.63 (0.63-0.63), identical output 3/3 | 0.85 |
| 512K | 30.39 | 1.15 (1.13-1.18) | 66-69 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.37 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.38 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.41 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.42 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.42 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.44 | Based on the images provided:  *   The first image contains **5** blue | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.38 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.39 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.41 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.63 | red, green | red, green | yes |
| 2img_digits rep1 | 2 | 0.43 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.44 | 5, 3 | 5, 3 | yes |

1 image: TTFT 0.39 s, 6/6 correct. 2 images: TTFT 0.46 s, 6/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~1.0 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 1.3 s at 128K and 3.3 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
