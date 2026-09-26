# ds41-batch: MERGE-READY (fused multi-request Mac verify, og path)

Branch `ds41-batch` off `005842b9` (ds41-next, the served tree). Not deployed. Go/no-go: **GO**. Fused c4 is **+17-19%** aggregate at 8K and **+15%** at 128K against today's code. The bar was 12%.

## What changed
| change | files | env (default) |
|---|---|---|
| `og_fused.forward`: one Mac pass over layers 20-39 + head for several requests' verify rows. mHC, the MXFP8 projections (wq_a, wkv, wq_b, wo_b, shared expert), BF16 wo_a, the MoE pair kernels + combine, and the BF16 head run once over all rows. Attention, indexer, router, caches and verify stash stay per request. | `og_fused.py` (new); `pipe_decoder.py` (validation split into `_open_boundary`, new `forward_boundaries`); `language.Attention(return_heads=True)` (returns heads before wo_a) | `DS41_OG_FUSE=1` (0 = off) |
| Scheduler: `fused_batch.advance` asks the host for verify groups (`mtp_verify_groups`, og only). With >= 3 requests verifying, consecutive 2-5-row requests pair up; a 1-row request stays alone. Groups run one after another: fused verify, then per-request accept/draft/presend. The box computes one pair's steps while the Mac serves the other pair. | `mlx_lm_mtp/fused_batch.py` (`_advance_groups`), `og_model.py` (`mtp_verify_groups`, fused branch in `_remote`, `fused_calls`/`fused_steps` in /og/stats) | `DS41_OG_FUSE_MIN=3` (2 = also fuse c2; slower, see below) |
| og worker default concurrency: 4 when fusion is on, 2 when it is off. Today streams 3-4 queue. | `og_serve/og_server.py` | `DS41_OG_CONCURRENCY` (explicit value wins) |

Other models are unaffected: only a host with `mtp_verify_groups` (the og decoder) takes the new branch. c1 never reaches `fused_batch.advance`, and `_remote` checks `len(caches) > 1` first, so the c1 path is unchanged.

## Numerics / identity
- **The fused forward is bitwise identical per request** to that request's own `forward_boundary` (`benchmarks/og/batch_bench.py numerics`, partial load, real 8K box boundaries). Row mixes checked: 5+5, 2+5, 3+4, 2+2, 5+5+5, 4+5+3+2. For each request, compared bitwise: logits, DSpark hidden, and all 160 cache and verify-window arrays of layers 20-39.
- **Why it is bitwise:**
  - The kernels used (fast_qmv rows, grouped_gemv, head rows, MoE pair/combine, mHC) compute each row with the same lanes, order and reductions for any M.
  - 1-row requests are never fused, because MLX uses different M=1 kernels.
  - The router's fp32 GEMM stays per request.
- **Served temperature-0 identity** (`batch_identity.py`, host_server partial load, `DS41_OG_BOX_CACHE=0`):
  - Setup: 4 fixed prompts (8K/3K/8K/12K), 256 tokens each. Each was run alone and then in concurrent waves, and every text and token count matched.

  | run | fused steps | identical |
  |---|---|---|
  | c4, production depth policy | 948/1011 | 4/4 |
  | c3, production depth policy | 482/1011 | 4/4 |
  | 5-row depth: c4 | 984/1011 | 4/4 |
  | 5-row depth: c2 with `FUSE_MIN=2` | 982/1011 | 4/4 |
  | production tree ds41-next alone | - | c1 texts == ds41-batch c1 (4/4) |
  | production tree ds41-next c2 | - | 4/4 |

  - The aliased partial model generates chaotic text, so any numeric difference would show up quickly.

## Measured (partial load: layers 20/24/25 aliased over 20 layers + head, real box sessions, <= 4)
**Mac pass alone (`batch_bench.py timing`, 8K, median ms):**

| rows | separate | fused | saving |
|---|---|---|---|
| 1x5 | 17.9 | - | - |
| 2x5 | 35.5 | 27.7 | -22% |
| 2x4 | 32.1 | 24.1 | -25% |
| 2x3 | 29.2 | 20.6 | -29% |
| 3x5 | 53.2 | 39.3 | -26% |

**Served flow** (`batch_ab.sh` = host_server.py with the real omlx scheduler, DSpark, presend and box sessions):
- The metric is verify cycles/s summed over all streams.
- Aliased layers produce garbage text, so acceptance is about 0. `HS_FIXED_DEPTH=1` keeps every cycle at 5 rows, which is production-like work (production averages 4.4 rows).
- tok/s = cycles/s x tokens/cycle (production 8K: ~3.3).
- "agg" is measured while all c streams decode. "span" is all tokens over first-to-last token; it includes ramp and queueing, and is the only metric for the current code at c3/c4, where streams 3-4 wait.

| 8K, cycles/s | c1 | c2 | c3 | c4 |
|---|---|---|---|---|
| **current** (`DS41_OG_FUSE=0`, concurrency 2 = ds41-next behaviour) | 30.8 | agg 45.0-45.6 / span 42.1-42.3 | span 37.1 | span 41.0 |
| **fused** (default: `DS41_OG_FUSE=1`, concurrency 4) | 33.3 (same code path; run noise) | agg 44.9 / span 42.0 (same path, not fused) | agg 50.5 / span 46.2 (**+25%** span) | agg 51.9-54.8 / span 48.0-48.6 (**+17-19%** span; agg +15-20% vs current c2 agg) |
| unfused, concurrency 4 (4-way interleave) | | | | agg 43.7-47.2 / span 41.2-43.2 |
| fused c2 (`DS41_OG_FUSE_MIN=2`), copy drafts off | | agg 39.8 vs 43.7 per-request (**-9%**) | | |

**128K, c4 (copy drafts off, 768 tokens, 2 reps each):**
- Fused: 50.8 cycles/s.
- Unfused 4-way: 43.1 (fused is +18%).
- Current code (concurrency 2, measured as c2 at the same settings): 44.0 (fused is **+15%**).
- At 128K each box wait is 5-9 ms per step in every configuration. It is the same with fusion off, so the cause is box-side and was not investigated.

**Projection to production:**
- The full model reads 20 distinct layers, so the fused 2x5 pass is about 31.5 ms against 2 x 19.85 (profile).
- Per pair: separate 2 x (19.85 + 3.9 draft + ~1.5 host) = ~50.5 ms, against fused 31.5 + 7.8 + ~3 = ~42 ms, which is **+19%**. That agrees with the served-flow measurement.
- Estimate for the owner-quoted current c4 aggregate of ~97-107 tok/s: about **115-125 tok/s**. This needs validation on the real model.
- Latency trade-off at c4: requests 3-4 now start at once (8K TTFT 1-4 s instead of ~20 s queued). Each of the 4 streams decodes at about 1/4 of the aggregate, where today 2 streams get 1/2 each.

## Why not more (and why c2 stays per-request)
- **c2:** a single fused pair must wait for both box steps (~15 ms plus wire) before it starts. The per-request interleave already hides the box. Measured -9%, so `FUSE_MIN=3`. Different context lengths do not change this, because the box step is nearly flat with context.
- **c3:** one pair plus one single. A fused triple would expose the box, like c2.
- **The profile's +27%** assumed a batched DSpark drafter for the pair (~5 ms instead of 2 x 3.9). Drafts still run per request here. The drafter is its own 3-stage forward with per-request caches; batching it is a follow-up worth about +7%.
- **Harness fix:** `benchmarks/og/host_server.py` imported og_model before og_server.py's env defaults. So DS41_MHC, DS41_GATHER, DS41_NATIVE_DECODE and the rest were **off** in the served-flow harness, including ds41-host's served A/B. It now sets the same defaults first. Production (og_server.py) was never affected.

## Tests
- `tests/test_deepseek_v41_og_fused.py` (new: grouping policy, eligibility bounds): 4 pass.
- `tests/test_deepseek_v41_og_cache.py`, `tests/test_deepseek_v41_og_wake.py`: pass.
- `og_serve/test_import_fallback.py` 8/8, with an og_fused stub added. `og_serve/test_wire_failover.py`: ALL PASS.

## Deploy (coordinator)
- **Deploy:** in the llama-swap ds41 env, set `DS41_TREE=/Users/ian/src/wt/ds41-batch` (keep `DS41_OG_CACHE_GIB=16`), then reload.
  - The worker then admits 4 concurrent requests and uses up to 4 of the box's 8 sessions.
  - Before reloading, copy the 16 gitignored kernel artifacts (`omlx/custom_kernels/*/{*.so,*.dylib,*.metallib}`) from ds41-next into the deployed tree and check them with `cmp`. They are already copied into `~/src/wt/ds41-batch`.
- **Gates:**
  - Identity: `og_serve/fe_bench.py ... --identity` against the production reference at c1.
  - A c4 bench_api run: 4 concurrent 8K requests; `/og/stats` `fused_steps` > 0.
  - Optionally, `benchmarks/og/batch_identity.py` against the full worker.
- **Rollback:** any one of:
  - `DS41_OG_FUSE=0`: old path and concurrency 2, no reload of the tree needed.
  - `DS41_OG_CONCURRENCY=2`: fusion never triggers, since it needs 3 requests.
  - `DS41_TREE=/Users/ian/src/wt/ds41-next`.

## Reproduce (Mac, production idle, no gpu.lock)
```
cd ~/llm/ds41/batch
DS41_TREE=~/src/wt/ds41-batch ~/llm/.venv-ds41-omlx-tiles/bin/python -u ~/src/wt/ds41-batch/benchmarks/og/batch_bench.py   # numerics + Mac timing
BC_TOKENS=384 ~/src/wt/ds41-batch/benchmarks/og/batch_ab.sh ds41-batch fuse "1 2 3 4" HS_FIXED_DEPTH=1 DS41_OG_CONCURRENCY=4
BC_TOKENS=384 ~/src/wt/ds41-batch/benchmarks/og/batch_ab.sh ds41-batch base "1 2 3 4" HS_FIXED_DEPTH=1 DS41_OG_FUSE=0 DS41_OG_CONCURRENCY=2
BC_IDENTITY=1 ~/src/wt/ds41-batch/benchmarks/og/batch_ab.sh ds41-batch id "1,4,3" DS41_OG_BOX_CACHE=0 DS41_OG_CONCURRENCY=4
```
Raw results: `~/llm/ds41/batch/{bench.jsonl,runs.jsonl,identity.jsonl}`.
