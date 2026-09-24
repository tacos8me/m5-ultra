# Kernels

Every code change behind the results on the [results page](https://tacos8me.github.io/m5-ultra/), as patches against
the upstream commit each was built on, plus full copies of the changed files under `src/`. Every DeepSeek change
was merged only with bit-identical greedy output, caches and logits.

| Directory | Upstream | What it serves |
|---|---|---|
| [`mlx/`](mlx/) | [ml-explore/mlx](https://github.com/ml-explore/mlx) @ `59d600b` | Metal kernels used by both models |
| [`mlx-lm/`](mlx-lm/) | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) + PR #1219 (MiMo) | MiMo-V2.6-Flash |
| [`omlx/`](omlx/) | [jundot/omlx](https://github.com/jundot/omlx) @ `d4298ad` | DeepSeek-V4.1-Flash |

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
