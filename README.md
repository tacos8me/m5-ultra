# M5 Ultra

Local LLM inference on a Mac Studio M5 Ultra (256 GB): DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, Qwen3.8-Flash-Next and Qwen3.8-27B.

**Results page: https://tacos8me.github.io/m5-ultra/**

## Prefill at depth

Prompt tokens ÷ time to first token. Measured in 8K chunks: DeepSeek up to 1M tokens, MiMo up to 512K; a needle planted at the start of a 1M-token prompt was recalled correctly. Qwen3.8 was measured through the server from 8K to 1M on stock oMLX. Past its native 262K window it runs without YaRN scaling (speed only); it still recalled a needle at 512K, 768K (2,061 tok/s) and 1M. The 768K and 1M points need a small memory patch ([`kernels/omlx/qwen38/`](kernels/omlx/qwen38/)).

| Prompt | DeepSeek-V4.1-Flash | MiMo-V2.6-Flash | Qwen3.8-Flash-Next | Qwen3.8-27B | RTX PRO 6000 ×2 · MiMo |
|---|---:|---:|---:|---:|---:|
| ~130K | 2,473 tok/s · 53.0 s | 2,112 tok/s · 62.1 s | 2,387 tok/s · 54.9 s | 1,116 tok/s · 118 s | 9,724 tok/s · 13.4 s |
| ~520K | 2,166 tok/s · 242 s | 873 tok/s · 600 s | 2,173 tok/s · 241 s | 509 tok/s · 1,030 s | ~4,880 tok/s · 107 s |
| 1M | 1,898 tok/s · 553 s | — | 1,944 tok/s · 535 s | ~294 tok/s (est.) | — |

## Decode

Output tokens per second, greedy, speculative decoding on.

| | DeepSeek-V4.1-Flash | MiMo-V2.6-Flash | Qwen3.8-Flash-Next | Qwen3.8-27B | RTX PRO 6000 ×2 · MiMo |
|---|---:|---:|---:|---:|---:|
| 1 request | 79 | 117 | 109 | 77 | 160 |
| 2 requests, total | 79 | — | — | — | — |
| 8 requests, total | — | 194 | — | — | 627 |
| At 130K context | 61 | 82 | 73 | 39 | — |
| At 523K context | 51 | 71 | 62 | 22 | 101 |
| At 1M context | 25 (no DSpark) | — | 44 | ~14 (est.) | — |

## What made it faster

DeepSeek-V4.1-Flash, same day. Every change keeps greedy output bit-identical.

- Engram table reads moved off the critical path: parallel native SSD reads, one chunk ahead of the GPU.
- Sparse attention and indexer kernels rewritten: accelerator scoring, exact radix top-k, KV decoded once per query tile.
- Buffer pool kept across layers; the next layer is built while the GPU runs the current one.
- Decode: faster attention, Sinkhorn and KV-packing kernels; verify-step GEMVs share activations across rows.
- No per-token kernel recompiles and no SSD page faults on fresh text.
- Two requests batched exactly under speculative decoding.

## Kernels

All modified kernels and code are in [`kernels/`](kernels/), as patches against upstream MLX, mlx-lm and oMLX plus the changed source files.

## Setup

- **DeepSeek-V4.1-Flash:** 3-bit LSQ experts (group 128), 8-bit elsewhere, 223.5 GiB resident. oMLX with custom Metal kernels, encoder-only prefill in 8K chunks, DSpark speculative decoding (k=4).
- **MiMo-V2.6-Flash:** MXFP4 experts, 8-bit elsewhere. mlx-lm with custom Metal kernels, MTP speculative decoding.
- **Qwen3.8-Flash-Next:** oQ8e (8-bit) with MTP, stock oMLX 0.7.0.dev4, no custom kernels; 1M adds the memory patch in `kernels/omlx/qwen38/`.
- **Qwen3.8-27B:** dense, oQ8e (8-bit) with MTP, stock oMLX. Dense attention makes prefill slow with length; 768K and 1M are extrapolated from the measured curve.
- Served through an OpenAI-compatible endpoint (llama-swap). Measured 2026-09-24.
- RTX PRO 6000 pair figures are MiMo-V2.6-Flash on original weights.
