# M5 Ultra

Local LLM inference on a Mac Studio M5 Ultra (256 GB): DeepSeek-V4.1-Flash and MiMo-V2.6-Flash.

**Results page: https://tacos8me.github.io/m5-ultra/**

## Prefill at depth

Prompt tokens ÷ time to first token.

| Prompt | DeepSeek-V4.1-Flash | MiMo-V2.6-Flash | RTX PRO 6000 ×2 · MiMo |
|---|---:|---:|---:|
| ~130K | 2,498 tok/s · 52.5 s | 2,015 tok/s · 64.5 s | 9,724 tok/s · 13.4 s |
| ~520K | 2,184 tok/s · 240 s | 790 tok/s · 661 s | ~4,880 tok/s · 107 s |

## Decode

Output tokens per second, greedy, speculative decoding on.

| | DeepSeek-V4.1-Flash | MiMo-V2.6-Flash | RTX PRO 6000 ×2 · MiMo |
|---|---:|---:|---:|
| 1 request | 79 | 117 | 160 |
| 2 requests, total | 79 | — | — |
| 8 requests, total | — | 194 | 627 |
| At 130K context | 61 | 82 | — |
| At 523K context | 51 | 71 | 101 |

## Setup

- **DeepSeek-V4.1-Flash:** 3-bit LSQ experts (group 128), 8-bit elsewhere, 223.5 GiB resident. oMLX with custom Metal kernels, encoder-only prefill in 8K chunks, DSpark speculative decoding (k=4).
- **MiMo-V2.6-Flash:** MXFP4 experts, 8-bit elsewhere. mlx-lm with custom Metal kernels, MTP speculative decoding.
- Served through an OpenAI-compatible endpoint (llama-swap). Measured 2026-09-24.
- RTX PRO 6000 pair figures are MiMo-V2.6-Flash on original weights.
