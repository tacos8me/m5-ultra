# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-30, box engine 6e85d7e, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,236 | 0.77 (0.75-0.78) | 10,733 | 19,618 | 3 | 3.29 | 2,498 | 4.3x |
| 16K | 16,429 | 1.23 (1.14-1.38) | 13,438 | 21,248 | 3 | 6.38 | 2,573 | 5.2x |
| 32K | 32,812 | 1.95 (1.93-1.96) | 16,856 | 21,683 | 3 | 12.56 | 2,610 | 6.5x |
| 64K | 65,581 | 3.56 (3.55-3.58) | 18,413 | 21,692 | 3 | 25.30 | 2,591 | 7.1x |
| 128K | 131,116 | 6.82 (6.74-6.97) | 19,222 | 21,828 | 3 | 51.09 | 2,545 | 7.6x |
| 256K | 262,185 | 13.33 | 19,673 | 21,544 | 1 | 111.08 | 2,360 | 8.3x |
| 512K | 524,332 | 26.66 | 19,669 | 20,881 | 1 | 234.65 | 2,234 | 8.8x |
| 768K | 786,478 | 41.10 | 19,134 | 20,094 | 1 | 376.24 | 2,090 | 9.2x |
| 1M | 1,040,045 | 56.05 | 18,556 | 19,451 | 1 | 520.53 | 1,998 | 9.3x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.75-0.78 s (n=5), 128K 6.74-6.97 s (n=5), 256K 13.20-13.33 s (n=2), 512K 26.57-26.69 s (n=3), 1M 55.85-56.05 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,240 | **109.8** | 104.5-118.3 | 76% | 3.04 | 6.98 | 10.29 | 71.7 (5) |
| 8K free-prose | 8,250 | **90.6** | 80.7-97.0 | 66% | 2.45 | 6.80 | 9.99 | 66.7 (2) short essay |
| 128K code-summary | 131,121 | **106.8** | 90.5-114.8 | 76% | 2.99 | 6.96 | 10.57 | 66.2 (1) |
| 256K code-summary | 262,191 | **107.5** | 88.5-116.6 | 77% | 3.06 | 7.01 | 10.68 | 63.5 (1) |
| 512K code-summary | 524,331 | **103.3** | 92.9-115.4 | 75% | 3.02 | 7.07 | 11.05 | 65.6 (4) |
| 1M code-summary | 1,040,046 | **101.4** | 78.5-117.2 | 76% | 3.04 | 7.15 | 13.00 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 78.6 (68.8-85.5) | 140.6 (138.1-144.2) | 0.64 / 1.17 | 0.80 / 1.33 |
| c2 128K | 75.5 (67.4-88.5) | 132.3 (117.1-143.3) | 1.01 / 1.67 | 6.79 / 13.92 |
| c4 8K | 50.8 (45.9-57.8) | 151.1 (147.1-155.1) | 0.63 / 1.16 / 2.01 / 2.74 | 0.79 / 1.36 / 2.09 / 2.73 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 decodes all four streams at once as two fused pairs; its later arrivals wait only for the earlier prefills (see TTFT by arrival order). This leg uses fresh nonce prompts, so its aggregate varies with draft acceptance; a fixed-prompt identical-output A/B gave c4 113 -> 139 tok/s for fused pairs + batched drafting vs the previous build.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.75 | 0.41 (0.40-0.41) | 76-79 | - | - |
| 128K | 6.80 | 0.49 (0.46-0.51) | 66-69 | 0.35 (0.34-0.35), identical output 3/3 | 0.85 |
| 512K | 26.69 | 0.87 (0.80-1.02) | 43-46 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.32 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.33 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.33 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.36 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.36 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.37 | 5, 3 | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.32 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.33 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.35 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.37 | red, green | red, green | yes |
| 2img_digits rep1 | 2 | 0.38 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.36 | Based on the images provided:  *   The first image contains **5** blue | 5, 3 | yes |

1 image: TTFT 0.33 s, 6/6 correct. 2 images: TTFT 0.37 s, 6/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~0.7 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 0.9 s at 128K and 2.5 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
