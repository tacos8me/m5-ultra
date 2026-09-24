# M5 Ultra

Local LLM inference on a Mac Studio M5 Ultra (256 GB): DeepSeek-V4.1-Flash, MiMo-V2.6-Flash and Qwen3.8-Flash-Next.

**Results page: https://tacos8me.github.io/m5-ultra/**

## Prefill at depth

Prompt tokens ÷ time to first token. Measured in 8K chunks: DeepSeek up to 1M tokens, MiMo up to 512K; a needle planted at the start of a 1M-token prompt was recalled correctly. Qwen3.8 was measured through the server from 8K to 512K on stock oMLX. Past its native 262K window it runs without YaRN scaling (speed only), and it recalled a needle at 512K. At 1M it runs out of memory.

| Prompt | DeepSeek-V4.1-Flash | MiMo-V2.6-Flash | Qwen3.8-Flash-Next | RTX PRO 6000 ×2 · MiMo |
|---|---:|---:|---:|---:|
| ~130K | 2,473 tok/s · 53.0 s | 2,112 tok/s · 62.1 s | 2,387 tok/s · 54.9 s | 9,724 tok/s · 13.4 s |
| ~520K | 2,166 tok/s · 242 s | 873 tok/s · 600 s | 2,173 tok/s · 241 s | ~4,880 tok/s · 107 s |
| 1M | 1,898 tok/s · 553 s | — | — | — |

## Decode

Output tokens per second, greedy, speculative decoding on.

| | DeepSeek-V4.1-Flash | MiMo-V2.6-Flash | Qwen3.8-Flash-Next | RTX PRO 6000 ×2 · MiMo |
|---|---:|---:|---:|---:|
| 1 request | 79 | 117 | 109 | 160 |
| 2 requests, total | 79 | — | — | — |
| 8 requests, total | — | 194 | — | 627 |
| At 130K context | 61 | 82 | 73 | — |
| At 523K context | 51 | 71 | 62 | 101 |
| At 1M context, plain decode | 25 | — | — | — |

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
- **Qwen3.8-Flash-Next:** oQ8e (8-bit) with MTP, stock oMLX 0.7.0.dev4, no custom kernels.
- Served through an OpenAI-compatible endpoint (llama-swap). Measured 2026-09-24.
- RTX PRO 6000 pair figures are MiMo-V2.6-Flash on original weights.
