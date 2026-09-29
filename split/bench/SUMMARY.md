# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-27, box engine 0802427, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,238 | 0.85 (0.83-0.86) | 9,722 | 19,014 | 3 | 3.29 | 2,498 | 3.9x |
| 16K | 16,428 | 1.16 (1.14-1.18) | 14,157 | 20,888 | 3 | 6.38 | 2,573 | 5.5x |
| 32K | 32,812 | 1.96 (1.94-1.97) | 16,745 | 21,541 | 3 | 12.56 | 2,610 | 6.4x |
| 64K | 65,584 | 3.57 (3.54-3.61) | 18,391 | 21,645 | 3 | 25.30 | 2,591 | 7.1x |
| 128K | 131,119 | 6.76 (6.75-6.76) | 19,405 | 21,554 | 3 | 51.09 | 2,545 | 7.6x |
| 256K | 262,186 | 13.51 | 19,401 | 20,992 | 1 | 111.08 | 2,360 | 8.2x |
| 512K | 524,333 | 28.05 | 18,696 | 19,734 | 1 | 234.65 | 2,234 | 8.4x |
| 768K | 786,478 | 44.52 | 17,664 | 18,471 | 1 | 376.24 | 2,090 | 8.5x |
| 1M | 1,040,049 | 61.76 | 16,841 | 17,556 | 1 | 520.53 | 1,998 | 8.4x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.74-0.86 s (n=5), 128K 6.71-6.95 s (n=5), 256K 13.51-13.54 s (n=2), 512K 27.90-28.05 s (n=3), 1M 61.75-61.76 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,243 | **104.6** | 95.9-117.6 | 75% | 2.95 | 6.98 | 10.58 | 71.7 (5) |
| 8K free-prose | 8,249 | **89.1** | 80.3-101.9 | 65% | 2.47 | 6.82 | 10.17 | 66.7 (2) short essay |
| 128K code-summary | 131,118 | **105.3** | 99.1-110.9 | 77% | 3.02 | 7.00 | 10.74 | 66.2 (1) |
| 256K code-summary | 262,190 | **102.0** | 92.6-111.7 | 75% | 2.94 | 6.99 | 10.80 | 63.5 (1) |
| 512K code-summary | 524,337 | **109.2** | 90.3-119.7 | 79% | 3.27 | 7.11 | 11.62 | 65.6 (4) |
| 1M code-summary | 1,040,045 | **98.0** | 77.4-113.9 | 75% | 2.98 | 7.14 | 12.91 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 71.5 (60.7-81.3) | 125.2 (121.8-129.8) | 0.64 / 1.42 | 0.78 / 1.73 |
| c2 128K | 72.1 (61.9-85.8) | 122.1 (111.9-131.0) | 1.01 / 1.92 | 6.79 / 16.33 |
| c4 8K | 45.7 (38.1-54.8) | 136.7 (134.8-138.8) | 0.65 / 1.50 / 2.43 / 3.17 | 0.82 / 1.89 / 2.93 / 3.93 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 decodes all four streams at once as two fused pairs; its later arrivals wait only for the earlier prefills (see TTFT by arrival order). This leg uses fresh nonce prompts, so its aggregate varies with draft acceptance; a fixed-prompt identical-output A/B gave c4 113 -> 139 tok/s for fused pairs + batched drafting vs the previous build.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.74 | 0.38 (0.35-0.39) | 68-71 | - | - |
| 128K | 6.95 | 0.50 (0.46-0.52) | 72-75 | 0.35 (0.35-0.36), identical output 3/3 | 0.85 |
| 512K | 27.92 | 0.82 (0.80-0.85) | 56-59 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.33 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.33 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.34 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.56 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.38 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.41 | 5, 3 | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.34 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.36 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.34 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.38 | red, green | red, green | yes |
| 2img_digits rep1 | 2 | 0.37 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.38 | 5, 3 | 5, 3 | yes |

1 image: TTFT 0.34 s, 6/6 correct. 2 images: TTFT 0.41 s, 6/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~0.7 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 0.9 s at 128K and 2.5 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
