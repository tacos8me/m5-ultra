# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-26, box engine 1954bf6, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 97 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,237 | 0.94 (0.93-0.95) | 8,757 | 18,304 | 3 | 3.29 | 2,498 | 3.5x |
| 16K | 16,429 | 1.45 (1.38-1.58) | 11,393 | 19,329 | 3 | 6.38 | 2,573 | 4.4x |
| 32K | 32,813 | 2.28 (2.26-2.32) | 14,400 | 19,610 | 3 | 12.56 | 2,610 | 5.5x |
| 64K | 65,582 | 4.08 (3.99-4.18) | 16,071 | 19,734 | 3 | 25.30 | 2,591 | 6.2x |
| 128K | 131,117 | 7.56 (7.53-7.58) | 17,350 | 19,599 | 3 | 51.09 | 2,545 | 6.8x |
| 256K | 262,189 | 14.82 | 17,696 | 19,152 | 1 | 111.08 | 2,360 | 7.5x |
| 512K | 524,333 | 31.34 | 16,730 | 17,714 | 1 | 234.65 | 2,234 | 7.5x |
| 768K | 786,480 | 52.23 | 15,057 | 15,686 | 1 | 376.24 | 2,090 | 7.2x |
| 1M | 1,040,043 | 78.80 | 13,199 | 13,703 | 1 | 520.53 | 1,998 | 6.6x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 0.93-0.97 s (n=5), 128K 7.53-7.60 s (n=5), 256K 14.82-14.85 s (n=2), 512K 31.21-31.34 s (n=3), 1M 78.43-78.80 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,240 | **98.9** | 92.4-117.0 | 76% | 3.06 | 7.39 | 14.13 | 71.7 (5) |
| 8K free-prose | 8,251 | **81.5** | 71.3-91.6 | 65% | 2.48 | 7.20 | 13.29 | 66.7 (2) short essay |
| 128K code-summary | 131,117 | **99.7** | 87.5-110.9 | 77% | 3.13 | 7.42 | 14.89 | 66.2 (1) |
| 256K code-summary | 262,192 | **97.6** | 84.8-115.0 | 77% | 3.08 | 7.36 | 14.84 | 63.5 (1) |
| 512K code-summary | 524,337 | **103.5** | 89.3-114.6 | 80% | 3.36 | 7.50 | 16.05 | 65.6 (4) |
| 1M code-summary | 1,040,046 | **93.2** | 77.5-107.0 | 76% | 3.11 | 7.59 | 15.33 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 64.5 (54.0-73.6) | 110.8 (107.6-113.2) | 1.00 / 1.98 | 0.96 / 1.98 |
| c2 128K | 63.8 (51.9-75.6) | 107.9 (104.3-111.8) | 1.37 / 2.53 | 7.58 / 17.84 |
| c4 8K | 40.3 (27.6-50.9) | 118.0 (103.5-125.3) | 1.15 / 2.14 / 3.48 / 4.47 | 1.37 / 2.38 / 3.44 / 4.46 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 is accepted but runs 2 at a time: streams 3-4 queue until the first pair finishes (~5 s at 8K), so the aggregate stays around 97 tok/s.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 0.97 | 0.64 (0.63-0.65) | 56-59 | - | - |
| 128K | 7.54 | 0.85 (0.84-0.85) | 72-75 | 0.68 (0.67-0.68), identical output 3/3 | 0.85 |
| 512K | 31.21 | 1.24 (1.20-1.29) | 50-53 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.39 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.43 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.43 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.43 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.46 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.47 | 5, 3 | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.41 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.42 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.65 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.45 | red, green | red, green | yes |
| 2img_digits rep1 | 2 | 0.46 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.47 | 5, 3 | 5, 3 | yes |

1 image: TTFT 0.45 s, 6/6 correct. 2 images: TTFT 0.45 s, 6/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

Box/Mac system sampling (VRAM, power, link throughput) was not collected in this run.

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~1.0 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 1.3 s at 128K and 3.3 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Round-trip ms/step counts from the moment a step is pre-sent to the box, which now happens before the Mac finishes the previous step, so it overlaps Mac work. It is not comparable with the link-only figure of the earlier run; box ms/step is.
- Four requests now decode together as two fused pairs (earlier builds ran two at a time and queued the rest).
