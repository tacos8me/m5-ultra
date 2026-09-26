# ds41-host: MERGE-READY (host critical path + small-op fusion, Mac half)

Branch `ds41-host` off `f56f7ffa` (served ds41-og). Not deployed. All changes are bitwise-neutral or scheduling-only.

## What changed
| change | files | env (default) |
|---|---|---|
| Fused mHC: 4 hc launches per decode Block instead of 8 (`project+pre_norm`, `post+Sinkhorn mix`); same ops, order and fp-contract/reassociate modes as the DS41_MHC kernels | `hc_fuse.py` (new), 3-line hook in `language.Block.__call__` | `DS41_HC_FUSE=1` |
| Early first submit: the first Block's attention is `async_eval`ed as soon as it is built | `hc_fuse.py` | `DS41_HC_EARLY_SUBMIT=1` |
| GPU keep-warm during the box wait: 4 MB add on a side stream every 0.5 ms from the recv spin loop (never waited on) | `og_model.py` (`GpuWarm`), `pipe_wire.recv_step(idle=)` | `DS41_OG_GPU_WARM_US=500` (0 = off) |
| 4 MB SO_RCVBUF per STEP socket (per-socket, no sysctl) | `pipe_wire.py` | `DS41_OG_RCVBUF_KB=4096` (0 = OS default) |
| No host eval in og `mtp_partial_rollback` (27a9b621's removal, lost in the 45ef51f8 rebase) | `og_model.py` | - |
| DSpark cost policy: one sync for probabilities + draft ids; host-built `drafts` and `next_main` (presend needs no extra GPU round trip) | `mlx_lm_mtp/batch_generator.py` (+ test) | - |

## Measured (partial load: layers 20/24/25 aliased over all 20 layers + head, real 8K box boundaries; so these are 20-layer numbers)
- Numerics: every Block output (h, pre), logits and DSpark hidden **bitwise identical** fused vs unfused, rows 1..5 (`host_bench.py numerics`); 20 chained hc stages bitwise too. Max abs/rel deviation 0.
- Fused hc + early submit, forward (median of 15, 3 interleaved pairs): k=1 13.04 -> 12.81 (-0.23 ms), k=2 -0.10, k=3 -0.14, k=4 -0.32, k=5 21.68 -> 21.48 (-0.14 to -0.2). Isolated hc chain 70 -> 55 us/layer.
- Keep-warm: forward right after a 10 ms GPU idle with the host spinning (production's box wait): k=1 13.3 -> 12.9, k=5 22.1 -> 21.6 ms (medians; no-idle baseline 12.86 / 21.56). The profile's full-model idle penalty was larger (+1.0 / +3.0 ms with a sleeping host), so the production gain may exceed this.
- rcvbuf (tight STEP loop vs the box): 5-row payload 0.76-0.90 -> 0.05-0.26 ms, roundtrip - box_s 1.69-1.75 -> 1.16 ms; 1-3 rows neutral (reply < 128 KB).
- Handoff syncs (served flow, per step): rollback 0.12 -> 0.06 ms, presend 0.07 -> 0.03 ms.
- **Estimated per c1 cycle: k=1 about -0.7 ms, k=5 about -1.3 ms.** The served-flow A/B (`host_server.py` + `host_ab.sh`, real omlx scheduler + box on the partial model) is functional (no errors, presend hits 754/763), but its run-to-run noise is +-1 ms, so it cannot resolve this. The coordinator's full validation run is the e2e number.

## Why this is short of the -3..-5 ms target
- The Mac forward is GPU-bound. Python graph build is only 2.2-2.9 ms per forward. The rest of the 9-16 ms "build" is time blocked inside `async_eval`, because MLX caps in-flight command buffers and the host stays about 10 buffers ahead. The first GPU work now starts about 0.4 ms after the reply lands.
- Small kernels cost about 1-4 us each on the GPU; it is not 3 us of pure dispatch per kernel. Removing 80 hc kernels per forward saved 0.1-0.3 ms.
- Hot sync round trips cost 25-60 us each; the profile's 190-250 us figure was measured on a cold GPU.
- Wire:
  - Mac client (recv copies + `mlx_step`): about 0.1 ms.
  - roundtrip - box_s: 0.95-1.35 ms in a tight loop, +0.7 ms when steps are 25 ms apart (box_s itself +0.15), and 2.4-3.0 ms in the served flow.
  - The idle-dependent part is box/NIC side, not Mac host. The GIL switch interval had no effect.

## Tests
`tests/test_deepseek_v41_*.py` + `tests/test_deepseek_v4_dspark.py`: same 65 pre-existing failures as f56f7ffa, +1 new passing test (cost-policy drafts). `og_serve/test_wire_failover.py`, `tests/test_deepseek_v41_og_cache.py` pass.

## Benches
- `benchmarks/og/host_bench.py`: partial-load forward benches. Tests: numerics, timing, hcmicro, host, gap.
- `benchmarks/og/host_wire.py`: STEP wire loop, with rcvbuf / advance / idle-gap variants.
- `benchmarks/og/host_server.py`, `host_cycle.py`, `host_ab.sh`: served-flow A/B on the partial model.
