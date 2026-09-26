# ds41-moe: MERGE-READY (routed MoE decode kernels, Mac half)

Base f56f7ffa. Not deployed. Opus ds41-moe, 2026-09-26.

## What changed

- `omlx/patches/deepseek_v41/moe_decode.py`: the fused MXFP4 pair kernels (gate/up + down) now take 1-16 rows. Before, they took 1-5 rows, and 6-16 fell back to three `gather_qmm` launches plus `glm_fast.deepseek_v41_grouped_expert`.
  - Per-pair arithmetic does not depend on the row count, so the outputs are bitwise those of the path they replace.
  - Env `DS41_MOE_FUSED_ROWS` (default 16). Setting it to 5 gives the f56f7ffa behaviour. `DS41_MOE_FUSED=0` still turns the fused path off.
- No call-site change.
  - `Expert.__call__` keeps `max_grouped_tokens=8`, so the verify path gets 6-8 rows.
  - `batched_verify` (32) gets 9-16 rows.
  - `forward_boundary` still guards at 5 rows. The k>5 gains land when wider verify or 2-stream batching does.
- `tests/test_deepseek_v41_moe_decode.py`: fused vs gather at 1/5/6/8/10/16 rows, bitwise.
- `benchmarks/og/moe_fused_bench.py`:
  - Loads the layer 20-22 ffn only (21 GiB, no gpu.lock).
  - Bitwise check on real hidden states: snap-8k boundary rows, ffn_norm, real gate.
  - Per-layer microbench.

## Numbers (per layer, µs; 60 serialized expert calls rotating 3 real layers; forced routing at the measured real unions; includes SwiGLU quant and serializing glue)

| k | union | before (f56f7ffa) | after | % of floor after | ms/cycle saved (x20 layers) |
|---|---|---|---|---|---|
| 1 | 6 | 151.0 | 151.6 | 64% | 0 |
| 2 | 10 | 231.4 | 230.7 | 70% | 0 |
| 4 | 17 | 366.0 | 365.1 | 75% | 0 |
| 5 | 20 | 437.8 | 437.8 | 73% | 0 |
| 6 | 25 | 666.6 | 527.8 | 76% | **-2.8** |
| 8 | 29 | 863.6 | 622.4 | 75% | **-4.8** |
| 10 (2x5) | 37 | 1068.9 | 797.9 | 75% | **-5.4** |

Floor = union x 18.8 MB / 1.17 TB/s. The table's % includes about 50 µs/layer of bench glue. The marginal cost per distinct expert is 16.9 µs, which is **95% of bandwidth** (slope over union 6 to 20).

## Numerics

Bitwise identical at k = 1-10 on layers 20/21/22, with max abs deviation 0. Two outputs were compared:
- the routed expert outputs;
- the full MoE output (routed + shared + combine).

Routing was real gate routing plus forced unions. See `~/llm/ds41/moe/final.jsonl`.

## Expert-major (union) kernel: built, measured, not shipped

The "-1.2 ms at k=5" estimate assumed that duplicate (row, expert) pairs re-read DRAM. They do not: the pair kernel dispatches pairs fastest, so duplicates hit cache.

Upper bound: the pair kernel with every duplicate threadgroup dropped. The outputs are wrong, so this only bounds the gain. It saves 4 / 8 / 19 / **22** / 28 / **65** / 88 µs/layer at k = 2 / 3 / 4 / 5 / 6 / 8 / 10. At k=5 that is at most **-0.44 ms/cycle**.

Correct union kernels were all slower than the pair kernel. Every row below is a bitwise-verified kernel except the NAX one.

| variant | result |
|---|---|
| Scalar leader (all pairs of an expert in one threadgroup) | k=5: 440 vs 418; k=8: 725 vs 593 |
| Chunked leaders (2/3/4/8 pairs per weight pass) + next-block prefetch | k=5: +1% / +11% / +15% / +28%; k=8: +8% .. +62%; k=10: +3% .. +53% |
| M5 matrix units (NAX, bf16 x bf16 -> fp32, 16x32x16 simdgroup MMA; deviation of 1 bf16 ulp) | 2.4-2.6x slower. The M dimension pads to 16 pair slots, and an expert averages 1-2 pairs. |

Control experiments:
- Faking the E2M1 decode, vectorizing the weight loads, and using 1-8 simdgroups per threadgroup all left the time unchanged.
- The kernel is bandwidth-bound per distinct expert. What remains is per-pair work that the union forms cannot overlap better.

Raw data: `~/llm/ds41/moe/dev*.jsonl`. The union and NAX sources are in the session scratchpad and can be provided on request. They are not committed (YAGNI).
