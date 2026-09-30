# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-30, box engine bf17aa4, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,236 | 0.76 (0.76-0.77) | 10,794 | 19,768 | 3 | 3.29 | 2,498 | 4.3x |
| 16K | 16,431 | 1.17 (1.16-1.19) | 14,017 | 20,981 | 3 | 6.38 | 2,573 | 5.4x |
| 32K | 32,813 | 1.95 (1.94-1.98) | 16,811 | 21,636 | 3 | 12.56 | 2,610 | 6.4x |
| 64K | 65,582 | 3.55 (3.54-3.57) | 18,450 | 21,716 | 3 | 25.30 | 2,591 | 7.1x |
| 128K | 131,117 | 6.79 (6.78-6.79) | 19,320 | 21,804 | 3 | 51.09 | 2,545 | 7.6x |
| 256K | 262,188 | 13.21 | 19,846 | 21,544 | 1 | 111.08 | 2,360 | 8.4x |
| 512K | 524,333 | 26.73 | 19,618 | 20,856 | 1 | 234.65 | 2,234 | 8.8x |
| 768K | 786,479 | 41.19 | 19,096 | 20,094 | 1 | 376.24 | 2,090 | 9.1x |
| 1M | 1,040,048 | 56.19 | 18,511 | 19,466 | 1 | 520.53 | 1,998 | 9.3x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.76-0.78 s (n=5), 128K 6.71-6.79 s (n=5), 256K 13.21-13.21 s (n=2), 512K 26.72-26.73 s (n=3), 1M 56.18-56.19 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,238 | **122.2** | 115.3-136.0 | 79% | 3.21 | 8.36 | 11.63 | 71.7 (5) |
| 8K free-prose | 8,248 | **96.8** | 87.0-108.0 | 65% | 2.43 | 8.19 | 11.05 | 66.7 (2) short essay |
| 128K code-summary | 131,122 | **114.1** | 101.2-133.3 | 76% | 3.00 | 8.33 | 11.86 | 66.2 (1) |
| 256K code-summary | 262,193 | **109.7** | 98.5-122.0 | 75% | 2.89 | 8.36 | 11.93 | 63.5 (1) |
| 512K code-summary | 524,331 | **113.9** | 100.8-123.5 | 77% | 3.10 | 8.41 | 12.34 | 65.6 (4) |
| 1M code-summary | 1,040,048 | **106.9** | 82.3-120.9 | 76% | 2.98 | 8.50 | 14.16 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 90.1 (76.7-102.5) | 155.8 (147.1-165.6) | 0.70 / 1.26 | 0.80 / 1.33 |
| c2 128K | 88.8 (74.7-99.3) | 151.8 (149.7-156.1) | 1.02 / 1.67 | 7.02 / 14.22 |
| c4 8K | 57.5 (44.6-73.0) | 170.8 (166.2-179.7) | 0.69 / 1.25 / 2.05 / 2.76 | 0.80 / 1.37 / 2.12 / 2.77 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 decodes all four streams at once as two fused pairs; its later arrivals wait only for the earlier prefills (see TTFT by arrival order). This leg uses fresh nonce prompts, so its aggregate varies with draft acceptance; a fixed-prompt identical-output A/B gave c4 113 -> 139 tok/s for fused pairs + batched drafting vs the previous build.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.78 | 0.46 (0.33-0.63) | 77-80 | - | - |
| 128K | 6.72 | 0.50 (0.47-0.55) | 65-68 | 0.36 (0.35-0.37), identical output 3/3 | 0.85 |
| 512K | 26.73 | 0.83 (0.81-0.85) | 57-60 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.33 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.34 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.33 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.35 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.36 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.38 | Based on the images provided:  *   **First image:** There are **5** bl | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.52 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.35 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.33 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.39 | red, green | red, green | yes |
| 2img_digits rep1 | 2 | 0.38 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.38 | Based on the images provided:  *   The first image contains **5** blue | 5, 3 | yes |

1 image: TTFT 0.37 s, 6/6 correct. 2 images: TTFT 0.37 s, 6/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~0.7 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 0.9 s at 128K and 2.5 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
