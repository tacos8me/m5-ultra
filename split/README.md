# DeepSeek-V4.1-Flash on a Mac and two RTX PRO 6000s

DeepSeek-V4.1-Flash on its **original FP4/FP8 weights**, served as one model across two very different machines:
a Mac Studio M5 Ultra (Metal/MLX) and a pair of NVIDIA RTX PRO 6000 Blackwell GPUs (CUDA/SGLang), joined by a
plain 10GbE cable. It is the default `ds41` model behind llama-swap on the Mac.

Results page: https://tacos8me.github.io/m5-ultra/split/ · raw data in [`bench/`](bench/) · code in [`../kernels/`](../kernels/) ([`omlx/deepseek-v41-split.patch`](../kernels/omlx/deepseek-v41-split.patch), [`split-nv/`](../kernels/split-nv/)).

---

## Why it works

DeepSeek-V4.1-Flash is a 40-layer MoE (384 routed experts, top-6) with an unusual attention stack:

- Only **four layers produce KV** (2, 8, 14, 20). Layers 21-39 reuse layer 20's KV; three of the source layers are
  2x-compressed. The whole prompt state is **~0.9 KB per token**: 120 MB at 128K, ~0.95 GB at 1M.
- Prefill therefore only needs layers 0-20 over the whole prompt; the upper layers run over a short tail.

So the model splits along a seam that already exists:

```
 ┌─────────────── RTX box (2x RTX PRO 6000, TP2) ───────────────┐   10GbE    ┌──────────── Mac Studio M5 Ultra ────────────┐
 │ embeddings · Engram (layers 1, 14) · layers 0-19             │ ─────────▶ │ layers 20-39 · head · DSpark drafter        │
 │ + layer-20 KV/index rows · vision tower                      │ ◀───────── │ verify · accept · draft · stream to client  │
 │ PREFILL: whole prompt → state streamed to the Mac            │  state     │ tail replay after prefill                    │
 │ DECODE: 1-5 verify rows per step → h19 + L20 rows (41 KB/row)│  per step  │                                              │
 └──────────────────────────────────────────────────────────────┘            └──────────────────────────────────────────────┘
```

Neither machine can hold the original model alone (≈285 GB without the Engram tables). Together they hold it with
room for 3M tokens of KV on the box and 1M-context requests on the Mac.

## A request, end to end

1. **Admission (Mac).** The supervisor checks the prefix cache (box snapshots + Mac layer-20 rows keyed by token digest
   and image hash). A resumed conversation skips straight to the new suffix.
2. **Prefill (box).** Layers 0-20 run over the prompt on both GPUs (FP4 experts, FP8 elsewhere); images go through the
   DS-V4.1 vision tower on the box and are injected at the image-token positions. The state streams to the Mac in 8K
   chunks while prefill continues.
3. **Tail replay (Mac).** Layers 20-39 run over the last ≤256 prompt rows to fill the decoder windows and prime DSpark.
4. **Decode (both).** Each step: the Mac drafts up to 4 tokens (DSpark, verify-cost-aware depth), the box runs layers
   0-19 for the verify rows (6-7 ms), the Mac runs layers 20-39 + head, accepts, and drafts the next rows.
   Two requests pipeline: the box works on one while the Mac works on the other.

## Correctness

- **Within the pipeline, bit-exact where it matters.** A decode step equals prefilling the same rows (relRMS exactly 0,
  including rollback of rejected drafts and page boundaries). Resumed state equals a fresh prefill byte for byte.
  A box engine restart mid-stream rebuilds the session and the output continues identically.
- **Across backends, judged on quality, not cosine.** CUDA and Metal amplify a single-ulp difference to ~0.97 cosine by
  layer 20 (a 1-ulp flip on the *same* backend does the same), so layer-local fidelity against the checkpoint's own
  `inference/model.py` arithmetic plus end-to-end quality are the gates.
- **Quality vs the previous Mac-only 3-bit build:** lower continuation NLL, needles recalled at 128K and 1M, same
  speculative acceptance. See the [results page](https://tacos8me.github.io/m5-ultra/split/).

## Operations

| What | Where / how |
|---|---|
| Serve | llama-swap `ds41` (aliases `deepseek-v4.1-flash`, `ds41-og`), `:think` variant; preloaded |
| Mac supervisor | `~/llm/ds41/og/ds41-og` → `~/src/wt/ds41-og/og_serve/ds41_og.py` (CPU; GPU child takes `gpu.lock`) |
| Mac weights | `~/models/DeepSeek-V4.1-Flash-pipe1-mlx` (layers 20-39 + head + DSpark, 148 GiB, original precision) |
| Box engine | systemd user unit `split-nv-engine` (restart on crash, starts at boot), pinned deploy `/home/ian/split-nv-deploy` |
| Box health | `curl 10.10.10.1:10051/health` (version, numerics, sessions, cache, restart_pending) |
| Deploy / rollback | `tools/s4_deploy.sh <commit> <label>` (then the gate script); rollback = previous commit |
| Box down | requests ride out an engine restart (≤240 s, bit-identical continuation), then a retryable 503 — never another model |
| Monitor | `all-smi view --hosts http://10.10.10.2:9090 http://10.10.10.1:9090 --icculis` ([fork](https://github.com/tacos8me/all-smi/tree/consolidated)) |

## Components

| Piece | Location | Notes |
|---|---|---|
| Box engine | `split-nv` (SGLang SM120 fork) | encoder half, step API (`:10052`), prefix cache in `/dev/shm`, streamed state, vision tower, fused MoE decode kernel (96% of DRAM floor) |
| Wire protocol | `pipe_wire.py`, `wire.py` | framed TCP; state stream; 41 KB/row steps |
| Mac decoder half | `ds41-og` (oMLX fork) | original-precision layers 20-39, MXFP4/FP8 kernels, one-pass vocab head, async drafter |
| Supervisor | `og_serve/ds41_og.py` | OpenAI/Anthropic/Responses APIs, failover, keepalives, cache routing |

## How it was built (Sep 25, 2026)

One long day, one coordinator and a rotating team of agents (Codex, Claude Opus, Claude Fable):

1. **Remote prefill.** RTX box prefills, Mac decodes on its 3-bit build (4-5x faster TTFT, marginal quality).
2. **Original weights on both sides.** The owner's call ("use og weights on mac"): pipelined decode across the link.
3. **Sprint 4, four lanes.** Prefix cache, Mac-half speed (vocab head read once, fused MXFP4 experts), box engine
   (streaming, 3M-token pool, supervisor, determinism fixes), serving (failover, API parity, soak, cutover).
4. **Cutover.** `ds41` became the split pipeline; native vision added; the 3-bit build retired; fused MoE kernel shipped.

Lessons worth keeping: gate on quality, not cross-backend cosine; measure the real step budget before setting latency
gates; never let a workaround swap the default model; keep engine cores off the root disk (`--ulimit core=0`).
