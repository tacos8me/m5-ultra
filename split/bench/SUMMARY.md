# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-29, box engine 1074e0b, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,237 | 0.85 (0.76-0.97) | 9,801 | 19,336 | 3 | 3.29 | 2,498 | 3.9x |
| 16K | 16,432 | 1.25 (1.19-1.36) | 13,161 | 20,723 | 3 | 6.38 | 2,573 | 5.1x |
| 32K | 32,813 | 1.94 (1.92-1.96) | 16,939 | 21,780 | 3 | 12.56 | 2,610 | 6.5x |
| 64K | 65,580 | 3.54 (3.52-3.56) | 18,538 | 21,764 | 3 | 25.30 | 2,591 | 7.2x |
| 128K | 131,119 | 6.80 (6.78-6.81) | 19,295 | 21,589 | 3 | 51.09 | 2,545 | 7.6x |
| 256K | 262,189 | 13.50 | 19,424 | 21,026 | 1 | 111.08 | 2,360 | 8.2x |
| 512K | 524,333 | 28.02 | 18,715 | 19,861 | 1 | 234.65 | 2,234 | 8.4x |
| 768K | 786,476 | 44.53 | 17,663 | 18,518 | 1 | 376.24 | 2,090 | 8.5x |
| 1M | 1,040,044 | 61.87 | 16,809 | 17,580 | 1 | 520.53 | 1,998 | 8.4x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.74-0.97 s (n=5), 128K 6.72-6.81 s (n=5), 256K 13.41-13.50 s (n=2), 512K 27.95-28.02 s (n=3), 1M 61.79-61.87 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,241 | **109.6** | 105.1-120.8 | 78% | 3.10 | 6.99 | 10.28 | 71.7 (5) |
| 8K free-prose | 8,248 | **85.9** | 73.2-103.8 | 63% | 2.39 | 6.82 | 9.98 | 66.7 (2) short essay |
| 128K code-summary | 131,117 | **100.8** | 89.3-106.7 | 74% | 2.86 | 6.96 | 10.38 | 66.2 (1) |
| 256K code-summary | 262,193 | **102.9** | 81.2-112.2 | 76% | 2.98 | 6.97 | 10.59 | 63.5 (1) |
| 512K code-summary | 524,336 | **112.1** | 98.9-127.9 | 81% | 3.36 | 7.13 | 11.36 | 65.6 (4) |
| 1M code-summary | 1,040,046 | **98.5** | 73.7-112.5 | 75% | 2.98 | 7.11 | 12.77 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 76.1 (66.3-81.9) | 136.2 (133.1-139.6) | 0.64 / 1.17 | 0.78 / 1.33 |
| c2 128K | 73.9 (65.0-82.8) | 132.3 (130.6-134.8) | 1.00 / 1.66 | 6.79 / 13.94 |
| c4 8K | 49.2 (40.9-54.0) | 146.6 (143.8-148.4) | 0.64 / 1.19 / 2.01 / 2.76 | 0.77 / 1.36 / 2.10 / 2.75 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 decodes all four streams at once as two fused pairs; its later arrivals wait only for the earlier prefills (see TTFT by arrival order). This leg uses fresh nonce prompts, so its aggregate varies with draft acceptance; a fixed-prompt identical-output A/B gave c4 113 -> 139 tok/s for fused pairs + batched drafting vs the previous build.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.74 | 0.40 (0.40-0.41) | 77-80 | - | - |
| 128K | 6.74 | 0.49 (0.46-0.50) | 62-65 | 0.35 (0.35-0.35), identical output 3/3 | 0.85 |
| 512K | 28.02 | 0.90 (0.83-1.04) | 61-64 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.31 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.34 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.32 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.36 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.36 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.37 | Based on the images provided:  *   **First image:** There are **5** bl | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.31 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.34 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.33 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.35 | red, green | red, green | yes |
| 2img_digits rep1 | 2 | 0.37 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.37 | 5, 3 | 5, 3 | yes |

1 image: TTFT 0.32 s, 6/6 correct. 2 images: TTFT 0.36 s, 6/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~0.7 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 0.9 s at 128K and 2.5 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
