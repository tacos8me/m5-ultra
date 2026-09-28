# ds41-woa: lossless byte storage for the Mac wo_a (MERGE-READY, not deployed)

Branch `ds41-woa` on production `3339e76d` (ds41-ttft), worktree `~/src/wt/ds41-woa`; the 16 ignored native
kernels were copied from ds41-ttft with `cp -p` and checked with `cmp`. Productionizes Codex's prototype
(`ds41-codex` 185883ec; `/home/ian/mac/codex-logs/decode-findings.md` section 1 on Linux).

## Change

- `omlx/patches/deepseek_v41/woa_compact.py` (new). The pipe1 `wo_a` is original FP8 E4M3 times block scales,
  stored as BF16, so the low 4 mantissa bits are always clear. Each weight becomes 1 byte: sign, the 3 live
  mantissa bits, and a 4-bit code for the 15 consecutive exponents that hold the most values. Code 15 escapes to
  the resident BF16 weight. The kernel is `decode_fusions` grouped GEMV with one output row per simdgroup. It keeps
  the same loads, dot products, unroll-8 blocks and shuffle reduction, so the output is bitwise equal to the
  BF16 kernel and to `mx.einsum`. Exponents are decoded arithmetically.
- `language.Attention` (singleton verify, 2-5 rows, i.e. where `grouped_gemv_supported` already chose the BF16
  kernel) and `DSparkAttention` (single-stream drafts, widths 2-4) call `woa_compact.grouped_gemv`.
  Unchanged paths: 1 row (MLX gemv), og_fused pairs, batched drafts, `DS41_VERIFY_TILE`, and prefill/replay.
- `og_model.load` encodes every BF16 `wo_a` after `load_decoder`: layers 20-39 and the 3 DSpark stages. The codes
  are a plain attribute, not a parameter. A replaced weight never reads stale codes (identity check).
- Flag: `DS41_WOA_COMPACT=0` switches this off. It then encodes nothing and uses no extra memory. The default is
  on. `DS41_DECODE_KERNELS_V2=0` still disables every grouped-GEMV path, this one included.

## Saving (partial-load harness, real 8K box boundaries, alternating A/B, 3 warm + 9 kept pairs)

| Verify rows | Forward off -> on (ms) | Saved, harness (3 real layers x20) | 20-call kernel chain: 3 rotated / 20 distinct | **Extrapolated, 20 distinct layers** |
|---:|---:|---:|---:|---:|
| 1 | 10.862 -> 10.860 | 0.002 (path unchanged) | - | 0 |
| 2 | 13.059 -> 12.540 | **0.519** (pairs 0.48-0.60) | 0.530 / 0.517 | **~0.51** |
| 3 | 14.676 -> 14.290 | **0.386** (0.36-0.48) | 0.440 / 0.373 | **~0.33** |
| 4 | 16.100 -> 15.792 | **0.308** (0.21-0.40) | 0.364 / 0.326 | **~0.28** |
| 5 | 17.865 -> 17.584 | **0.281** (0.23-0.33) | 0.310 / 0.283 | **~0.26** |
| DSpark draft, width 4 (3 real distinct stages) | 3.506 -> 3.469 | **0.037** | - | 0.037 |
| Fused 3+3 / 5+5 (path unchanged, 0 compact calls) | 20.670 -> 20.693 / 28.254 -> 28.176 | noise | - | 0 |

Extrapolation: harness forward saving x (20-distinct chain / 3-rotated chain). The 3 rotated compact weights
(100 MB) partly fit the SLC, which the 20 distinct weights (670 MB) do not. The 20-distinct chain therefore
saves 2-15% less, and that is the full-model estimate. A c1 5-row cycle saves ~0.26 + 0.04 = **~0.30 ms**
(~0.9% of a ~33 ms cycle). Shorter widths, which the cost-based depth controller also picks, save up to ~0.55 ms.

Wider shapes (kernel chain over 20 distinct weights, random input, bitwise equal): 6 rows 0.23 ms/20 calls, 7
rows 0.20, 8 rows 0.135, 10 rows -0.02. og_fused pairs stay on the BF16 kernel: the common c4 pair is 5+5 = 10
rows, where there is no gain, and the fused forward was not measured with the compact kernel. A follow-up could
enable pairs of 8 rows or fewer after a real-boundary fused A/B.

## Identity

- **Partial forward, flag off vs on, bitwise**, both real sessions, widths 1-5: logits + DSpark hidden + 160
  cache/verify arrays = **162/162** each. There were 20 compact calls per forward at widths 2-5 and 0 at width 1.
- **Fused pairs** (2,2) (3,3) (5,5) (2,5) (4,3): **324/324** with 0 compact calls.
- **DSpark** `dspark_forward` widths 2/3/4: logits, hidden and context caches **5/5** with 3 compact calls.
  Batched drafts (4,4) and (2,2): **8/8** with 0 calls.
- **All 23 wo_a weights in the checkpoint** (layers 20-39 + mtp 0-2), each loaded separately. The GPU codes
  equal the numpy reference encoder. Compact == BF16 grouped GEMV == einsum on **95 inputs per weight**, with
  **0 mismatches**. The inputs were 83 real wo_a inputs captured from the partial forward (20 layer positions x
  widths 2-5, plus 3 DSpark stages at width 4) and 12 random inputs (widths 2-5 x scales 1, 64, 1/64).
- **Eligibility, CPU scan of every wo_a** (`woa_scan.py`): low 4 mantissa bits clear in **100%** of values
  for all 23. Decoded bits equal the original for every non-escaped value.

| Weights | Exponent window | Escape rate | Escapes / 33.5M | Rows with >= 1 escape |
|---|---|---|---|---|
| layers 20-39 | 110-124 (all 20) | 0.0185% (L27) - 0.0290% (L38) | 6,209 - 9,725 | 53-66% |
| mtp 0-2 (drafter) | 111-125 | 0.0149% - 0.0215% | 4,995 - 7,224 | 45-52% |

Escapes are zeros (259-382 per weight) and values outside the window: mostly tiny magnitudes (exponent < 110)
and, in some layers, a few values of magnitude >= 0.25 (exponent 125-126). A weight escaping more than 1% would keep the
BF16 kernel; none does.

- Unit tests `tests/test_deepseek_v41_woa_compact.py`: 11 passed. They cover synthetic FP8-like weights with
  forced escapes, K 512/4096, widths 2-5 at 3 scales, install/alias/fallback/stale-weight cases, and flag off.
  The related suites (attention projection/rounding, mtp, og_cache/fused/ttft/wake) give the same result as
  base 3339e76d: 160 passed. Run together, 4 `test_packed_attention_preserves_online_bf16_probability_rounding`
  cases fail on **both** trees. That failure is an existing test-order interaction: the file passes alone on
  both trees. In that combined run the new tests skip, because an earlier module leaves the CPU as the default
  device.

## Memory

**+0.72 GiB** of codes (23 x 32 MiB: 0.625 GiB target + 0.094 GiB drafter). The production worker's
footprint peak goes from 159.3 GiB to ~160.1 GiB, well inside the 235 GiB MLX limit.

No sparse escape store: it would add memory rather than save it. The BF16 `wo_a` must stay resident for width 1,
og_fused pairs, batched drafts and prefill/replay, and escapes read it directly at no extra cost. Freeing the
BF16 copy would need compact kernels on all of those paths, and several of them have no speed win (10 rows is
slower) or would need a bitwise replica of MLX's 1-row gemv. That is out of scope.

## Startup cost

The encode runs on the GPU: a histogram kernel plus elementwise MLX ops over each resident weight, at
**~10.4 ms per weight**. That is **~0.25 s** for 23 weights plus a one-time kernel compile, against a ~17 s
weight load. No disk cache is needed. Numpy on the CPU would take ~0.11 s per weight (2.5 s total). The worker
logs one line, `{"event": "woa_compact", "encoded": 23, "skipped": 0, "gib": 0.72, ...}`.

## Deploy / rollback

- Deploy: set `DS41_TREE=/Users/ian/src/wt/ds41-woa` in the llama-swap ds41 env (keep `DS41_OG_CACHE_GIB=16`),
  then reload. Gate: og-child.log shows `woa_compact encoded 23`. Also rerun the usual fixed-prompt identity check
  (`og_serve/fe_bench.py --base http://127.0.0.1:8080 --model ds41 --label post-woa --identity` then
  `og_serve/fe_cmp.py <prod-label> post-woa`, plus c1/c2/c4 fixed-prompt outputs). Outputs must be identical.
- Rollback: `DS41_WOA_COMPACT=0` (same tree, BF16 kernel, no codes), or `DS41_TREE=/Users/ian/src/wt/ds41-ttft`.

## Reproduce (production idle only; `idle_guard.py` stops its own child PID on traffic)

```bash
cd ~/src/wt/ds41-woa; PY=~/llm/.venv-ds41-omlx-tiles/bin/python
$PY benchmarks/og/woa_scan.py ~/llm/ds41/woa/scan.jsonl                     # CPU only, all 23 weights
DECODE_MAX_SECONDS=240 WB_MODE=forward $PY benchmarks/og/idle_guard.py $PY -u benchmarks/og/woa_bench.py   # ~34 GB, 2 box sessions
DECODE_MAX_SECONDS=240 WB_MODE=layers  $PY benchmarks/og/idle_guard.py $PY -u benchmarks/og/woa_bench.py   # ~3 GB, no box
WB_CHAIN_ROWS=6,7,8,10 WB_MODE=layers  $PY benchmarks/og/idle_guard.py $PY -u benchmarks/og/woa_bench.py   # fused widths
$PY -m pytest -q tests/test_deepseek_v41_woa_compact.py
```

Logs: `~/llm/ds41/woa/{scan,forward,layers}.jsonl`, real inputs `~/llm/ds41/woa/real-inputs.safetensors`.
Limits: this was a partial model (3 real layers aliased, full head + DSpark), with at most 2 box sessions and no
full-model or served run. The saving is measured Mac forward time. It is not measured served tok/s.
