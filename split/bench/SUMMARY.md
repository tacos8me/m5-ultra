# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)

Run 2026-09-26, box engine c75a0a2, numerics og-s4.4, served by llama-swap as `ds41`. All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server (warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. Contention: 0 foreign requests and 0 non-idle starts across 98 measured requests.

Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.

## 1. Prefill (fresh prompts, TTFT includes the first token)

| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,238 | 1.03 (1.02-1.05) | 7,975 | 18,306 | 3 | 3.29 | 2,498 | 3.2x |
| 16K | 16,428 | 1.50 (1.48-1.52) | 10,966 | 19,252 | 3 | 6.38 | 2,573 | 4.3x |
| 32K | 32,814 | 2.38 (2.35-2.41) | 13,812 | 19,571 | 3 | 12.56 | 2,610 | 5.3x |
| 64K | 65,581 | 4.19 (4.16-4.21) | 15,663 | 19,636 | 3 | 25.30 | 2,591 | 6.0x |
| 128K | 131,116 | 7.87 (7.84-7.89) | 16,655 | 19,589 | 3 | 51.09 | 2,545 | 6.5x |
| 256K | 262,188 | 15.50 | 16,911 | 19,040 | 1 | 111.08 | 2,360 | 7.2x |
| 512K | 524,335 | 32.60 | 16,081 | 17,672 | 1 | 234.65 | 2,234 | 7.2x |
| 768K | 786,479 | 54.08 | 14,543 | 15,645 | 1 | 376.24 | 2,090 | 7.0x |
| 1M | 1,040,046 | 81.53 | 12,756 | 13,681 | 1 | 520.53 | 1,998 | 6.4x |

Repeat fresh samples from legs 2/4 agree within 1%: 8K 1.02-1.05 s (n=5), 128K 7.78-7.89 s (n=5), 256K 15.50-15.51 s (n=2), 512K 32.51-32.61 s (n=3), 1M 81.28-81.53 s (n=2).

## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)

| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K code-summary | 8,243 | **87.2** | 76.5-97.4 | 73% | 2.85 | 7.34 | 9.71 | 71.7 (5) |
| 8K free-prose | 8,251 | **75.8** | 66.9-87.6 | 64% | 2.48 | 7.25 | 9.72 | 66.7 (2) short essay |
| 128K code-summary | 131,116 | **94.2** | 86.7-101.4 | 77% | 3.14 | 7.42 | 9.81 | 66.2 (1) |
| 256K code-summary | 262,188 | **86.4** | 74.3-99.0 | 73% | 2.88 | 7.35 | 9.76 | 63.5 (1) |
| 512K code-summary | 524,337 | **94.4** | 75.7-110.7 | 78% | 3.23 | 7.48 | 9.87 | 65.6 (4) |
| 1M code-summary | 1,040,046 | **88.9** | 68.2-104.1 | 77% | 3.14 | 7.61 | 10.65 | 61.4 (4) |

Decode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: 63.8 tok/s (n=4).

## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)

| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |
|---|---:|---:|---|---|
| c2 8K | 61.8 (53.3-69.7) | 107.5 (101.8-112.9) | 1.17 / 2.01 | 1.18 / 2.04 |
| c2 128K | 59.7 (50.4-71.8) | 102.7 (100.4-106.3) | 1.74 / 2.87 | 7.85 / 17.64 |
| c4 8K | 61.4 (49.3-80.7) | 97.2 (95.8-99.0) | 1.18 / 2.05 / 6.80 / 7.76 | 1.20 / 2.08 / 6.66 / 7.53 |

q3 baseline c2 (short prompts): aggregate 77.6 tok/s, per-stream 45.0. c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill (at 8K the cached follow-ups also re-prefill; see caveats). c4 is accepted but runs 2 at a time: streams 3-4 queue until the first pair finishes (~5 s at 8K), so the aggregate stays around 97 tok/s.

## 4. Resume (prefix cache on box + Mac)

| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |
|---|---:|---:|---:|---:|---:|
| 8K | 1.04 | 0.79 (0.77-0.80) | 56-59 | - | - |
| 128K | 7.78 | 1.25 (1.24-1.25) | 77-80 | 1.06 (1.06-1.07), identical output 3/3 | 0.85 |
| 512K | 32.51 | 2.93 (2.89-2.99) | 60-63 | - | - |

## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)

| Task | Images | TTFT s | Answer | Expected | OK |
|---|---:|---:|---|---|---|
| 1img_color rep0 | 1 | 0.43 | Red | red | yes |
| 1img_digits rep0 | 1 | 0.45 | 4827 | 4827 | yes |
| 1img_count rep0 | 1 | 0.47 | 5 | 5, five | yes |
| 2img_colors rep0 | 2 | 0.46 | red, green | red, green | yes |
| 2img_digits rep0 | 2 | 0.52 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep0 | 2 | 0.52 | 5, 3 | 5, 3 | yes |
| 1img_color rep1 | 1 | 0.44 | Red | red | yes |
| 1img_digits rep1 | 1 | 0.45 | 4827 | 4827 | yes |
| 1img_count rep1 | 1 | 0.53 | 5 | 5, five | yes |
| 2img_colors rep1 | 2 | 0.52 | first, second | red, green | NO |
| 2img_digits rep1 | 2 | 0.53 | 4827, 391 | 4827, 391 | yes |
| 2img_count rep1 | 2 | 0.54 | Based on the images provided:  *   **First image:** There are **5** bl | 5, 3 | yes |
| 2img_colors_rephrased (diagnostic re-ask) | 2 | 0.49 | The first image is red, and the second image is green. | red, green | yes |

1 image: TTFT 0.46 s, 6/6 correct. 2 images: TTFT 0.52 s, 5/6 correct. The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.

## 6. System

| Item | Value |
|---|---|
| Mac omlx-server footprint (after bench) | 154 GB (16384 bytes per page) |
| Mac omlx-server RSS (after bench) | 151.0 GiB |
| Mac og worker peak RSS / footprint (watcher) | 154.0 / 157.1 GiB of 256 GB |
| Box gpu0 VRAM used (start / peak / total) | 88,947 / 89,615 / 97,887 MiB |
| Box gpu0 during 1M prefill | util 91.5% mean, 468.5 W mean (peak 536.4 W) |
| Box gpu1 VRAM used (start / peak / total) | 88,895 / 89,563 / 97,887 MiB |
| Box gpu1 during 1M prefill | util 92.7% mean, 474.8 W mean (peak 539.05 W) |
| Box compute per decode step (layers 0-19, 8K-1M) | 7.41 ms mean (7.17-7.77) |
| Mac-observed box round trip per step (incl. 10GbE) | 9.92 ms mean (9.53-13.42) |
| Link box->Mac during 512K prefill | 7.2 MB/s mean, 35.2 MB/s peak, 0.23 GB total |
| Link box->Mac during 1M prefill | 5.5 MB/s mean, 33.7 MB/s peak, 0.444 GB total |
| Link box->Mac during 1M follow-up window (5 cached questions incl. tail re-prefills) | 3.9 MB/s mean, 36.2 MB/s peak, 0.147 GB total |

The link is nowhere near saturated: a prefill ships about 0.43 KB per token of layer-20 state. Step times are Mac-side og/stats deltas per request; the box engine log records only per-session prefill times (box/engine_sessions.log).

## Caveats

- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.
- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.
- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).
- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~1.0 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 1.5 s at 128K and 5.8 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.
- Turn-2 resume at 128K takes 1.25 s, slower than q3 (0.85 s). Box prefill is only 0.05 s; the Mac import of box state takes about 0.55 s, and the Mac layers 20-39 take the rest.
- The llama-swap description still says "text only", but vision is live (box /health vision:true) and was tested here.
