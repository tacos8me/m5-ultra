# Kernels

Every code change behind the results on the [results page](https://tacos8me.github.io/m5-ultra/), as patches against
the upstream commit each was built on, plus full copies of the changed files under `src/`. Every DeepSeek change
was merged only with bit-identical greedy output, caches and logits.

| Directory | Upstream | What it serves |
|---|---|---|
| [`mlx/`](mlx/) | [ml-explore/mlx](https://github.com/ml-explore/mlx) @ `59d600b` | Metal kernels used by both models |
| [`mlx-lm/`](mlx-lm/) | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) + PR #1219 (MiMo) | MiMo-V2.6-Flash |
| [`omlx/`](omlx/) | [jundot/omlx](https://github.com/jundot/omlx) @ `d4298ad` | DeepSeek-V4.1-Flash |
| [`split-nv/`](split-nv/) | SGLang DeepSeek-V4.1 port | DeepSeek-V4.1-Flash, RTX half of the [Mac + RTX split](https://tacos8me.github.io/m5-ultra/split/) |

Each directory has `BASE.txt` (exact commits), `COMMITS.txt` (commit list), the patch file(s), `src/` and the
upstream `LICENSE`.

## `mlx/`

- `01-mimo-fast.patch`
  - Fused scaled-dot-product attention on the GPU neural accelerators for 192/128 head dims, with sliding window and
    attention sinks (`steel_attention_nax`).
  - Split-KV multi-query decode attention.
  - Expert-aligned MXFP4 `gather_qmm` tiles (`fp_quantized_nax`).
- `02-deepseek-3bit-tiles.patch`: expert-aligned tiles extended to 3-bit group-128 affine weights (`quantized_nax`).
  Applies on top of 01.

## `mlx-lm/`

`mimo-fast.patch`, on top of PR #1219's MiMo model:
- **Kernels:** router top-k, fused/grouped MXFP4 MoE decode, sliding-window and bf16 projection prefill.
- **Speculative decoding:** 3-layer MTP draft with speculative sampling, windowed MTP catch-up, a fresh draft cache
  per request.
- **Serving:** geometric KV-cache growth, and server fixes (listen backlog, logprob sync, concurrency handoff).

## `omlx/`

`deepseek-v41.patch`, on the DeepSeek-V4.1 layer:
- **Sparse attention (native):** queries read into registers, KV rows decoded once and reused across neighbouring
  queries (`custom_kernels/glm_moe_dsa/csrc/`).
- **Indexer:** accelerator scoring, exact two-read radix top-k, and runtime-width top-k so decode never
  recompiles (`index_nax.py`, `prefill_index.py`, `kernels.py`).
- **Prefill:**
  - capped buffer pool instead of clearing the allocator every layer;
  - next layer built while the GPU runs the current one;
  - single-pass RoPE (`language.py`, `fast_rope.py`).
- **Engram:** parallel native SSD reads with next-chunk lookahead and shrink-safe reuse, for prefill and decode
  (`engram_io.c`, `storage.py`).
- **Decode:**
  - faster attention, Sinkhorn and KV-pack kernels (`decode_fusions.py`);
  - bit-exact MXFP8 GEMV replicas that share activations across verify rows (`fast_qmv.py`).
- **Concurrency:** exact cross-request batching with speculative decoding (`batched_verify.py`, `ragged_projection.py`,
  `routed_batch.py`, `scalar_batch.py`, `mlx_lm_mtp/`).
- **Quantization:** LSQ 3-bit quantizer used to build the served checkpoint (`lsq_quant.py`).

### `omlx/qwen38/`

`qwen38_serve.py` runs `omlx serve` for Qwen3.8-Flash-Next at 1M tokens (prefill and decode stay under 225 GiB; stock oMLX passed 242 GiB). No
kernel changes; output is unchanged:
- The sparse-attention KV cache is allocated once for the whole prompt instead of doubling. Stock doubling holds the
  old and new buffers of all 12 layers at once, about +24 GiB at the 512K to 1M step.
- The MTP head's indexer keys and pooled block bank are evaluated after each prompt-priming chunk. Stock priming
  never evaluates them, so every chunk pins a copy of the index buffer until the first draft token.

Usage: `python qwen38_serve.py serve --model-dir … --base-path …` (same arguments as `omlx serve`).

## Applying

```bash
git clone https://github.com/ml-explore/mlx && cd mlx
git checkout 59d600b5e64c238427d0f8d897ab7c682ef4d3d2
git apply /path/to/kernels/mlx/01-mimo-fast.patch /path/to/kernels/mlx/02-deepseek-3bit-tiles.patch
```

The same applies to `mlx-lm/` (check out `kernelpool/add-mimo-v2` at the commit in `BASE.txt`) and `omlx/`.

## Licenses

`mlx/` and `mlx-lm/` are modified MIT-licensed code (© Apple Inc.). `omlx/` is modified Apache-2.0 code from
jundot/omlx. The files under `omlx/src/` were changed from the originals, and `deepseek-v41.patch` shows every change.
Each directory keeps its upstream license.

## DeepSeek-V4.1-Flash, Mac + RTX split

Results: [tacos8me.github.io/m5-ultra/split](https://tacos8me.github.io/m5-ultra/split/) · design notes: [`../split/README.md`](../split/README.md).

- `omlx/deepseek-v41-split.patch` (applies on top of `deepseek-v41.patch`, `ef88391e..f56f7ffa`, commit list in
  `omlx/COMMITS-split.txt`): the Mac half. Original-precision layers 20-39 + head + DSpark, the pipeline decode loop,
  prefix reuse, image input, verify-cost-aware draft depth, extra draft sources, and the `og_serve/` supervisor
  (OpenAI-compatible server, failover that never swaps models, parity/soak/bench harnesses). `src/` holds the changed files.
- `split-nv/`: the RTX half. `src/` is the engine (SGLang hooks, step API server, prefix cache, streamed state,
  fused MoE decode kernel, deploy and gate tools); `sglang-split.patch` is the three-commit change to the SGLang
  DeepSeek-V4.1 port it runs on (prompt-state capture hooks, deterministic top-k tie order). `BASE.txt`, `COMMITS.txt`.
