# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-27, box engine b4f5ea4, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,237 | 0.86 (0.83-0.90) | 9,548 | 20,098 | 3 | 3.29 | 2,498 | 3.8x |
| 16K | 16,427 | 1.27 (1.25-1.31) | 12,903 | 20,892 | 3 | 6.38 | 2,573 | 5.0x |
| 32K | 32,811 | 2.04 (2.03-2.04) | 16,095 | 21,682 | 3 | 12.56 | 2,610 | 6.2x |
| 64K | 65,581 | 3.63 (3.60-3.67) | 18,062 | 21,788 | 3 | 25.30 | 2,591 | 7.0x |
| 128K | 131,114 | 6.88 (6.85-6.90) | 19,058 | 21,565 | 3 | 51.09 | 2,545 | 7.5x |
| 256K | 262,189 | 13.62 | 19,253 | 21,076 | 1 | 111.08 | 2,360 | 8.2x |
| 512K | 524,333 | 27.84 | 18,832 | 19,899 | 1 | 234.65 | 2,234 | 8.4x |
| 768K | 786,483 | 44.41 | 17,710 | 18,540 | 1 | 376.24 | 2,090 | 8.5x |
| 1M | 1,040,045 | 61.16 | 17,005 | 17,670 | 1 | 520.53 | 1,998 | 8.5x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.83-0.90 s (n=5), 128K 6.85-6.97 s (n=5), 256K 13.61-13.62 s (n=2), 512K 27.84-27.92 s (n=3), 1M 61.16-61.39 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,240 | **106.2** | 101.3-109.2 | 77% | 3.13 | 7.05 | 12.83 | 71.7 (5) |
| 8K free-prose | 8,250 | **83.8** | 70.7-89.4 | 64% | 2.44 | 6.91 | 11.94 | 66.7 (2) short essay |
| 128K code-summary | 131,115 | **106.1** | 96.9-124.3 | 78% | 3.17 | 7.08 | 13.60 | 66.2 (1) |
| 256K code-summary | 262,190 | **94.8** | 80.6-104.6 | 73% | 2.85 | 7.02 | 13.15 | 63.5 (1) |
| 512K code-summary | 524,339 | **106.9** | 90.4-120.3 | 79% | 3.33 | 7.20 | 14.89 | 65.6 (4) |
| 1M code-summary | 1,040,045 | **92.3** | 81.4-104.6 | 73% | 2.90 | 7.17 | 13.60 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 64.1 (53.5-74.3) | 110.4 (107.3-115.8) | 0.89 / 1.88 | 0.85 / 1.86 |
| c2 128K | 66.3 (56.2-79.2) | 111.2 (108.8-113.0) | 1.25 / 2.40 | 6.88 / 16.57 |
| c4 8K | 40.7 (30.8-50.4) | 119.0 (116.2-122.4) | 0.90 / 2.06 / 3.14 / 4.11 | 0.91 / 2.07 / 3.10 / 4.05 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 decodes all four streams at once as two fused pairs; its later arrivals wait only for the earlier prefills (see TTFT by arrival order). This leg uses fresh nonce prompts, so its aggregate varies with draft acceptance; a fixed-prompt identical-output A/B gave c4 113 -> 139 tok/s for fused pairs + batched drafting vs the previous build.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.84 | 0.58 (0.55-0.62) | 70-73 | - | - |
| 128K | 6.97 | 0.75 (0.75-0.76) | 60-63 | 0.63 (0.61-0.64), identical output 3/3 | 0.85 |
| 512K | 27.84 | 1.18 (1.06-1.37) | 63-66 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.35 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.36 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.40 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.42 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.43 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.41 | Based on the images provided:  *   **First image:** There are **5** bl | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.37 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.39 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.38 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.41 | first, second | red, green | NO |
| 2img_digits rep1 | 2 | 0.40 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.41 | 5, 3 | 5, 3 | yes |

1 image: TTFT 0.37 s, 6/6 correct. 2 images: TTFT 0.41 s, 5/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~0.9 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 1.1 s at 128K and 2.6 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
