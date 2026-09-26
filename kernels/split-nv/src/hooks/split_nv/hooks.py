"""Capture the DS41 encoder prompt state from a running SGLang prefill, packed the Mac way.

Installed into the SGLang model-runner process (rank 0 only) when SPLIT_NV_HOOKS=1.
Per prefill (extend) chunk it captures, before SGLang's own FP8 requantization:
  * SWA window rows for layers 0-20 (pre-norm kv -> rms_norm -> rope -> Mac FP8 pack)
  * compressed KV + index K rows for source layers 2/8/14/20 (bf16 latent -> rope -> Mac FP4 pack)
  * the ratio-2 compressor tails (fp32 kv/gate of a dangling odd token)
  * the last 256 rows of the hidden state entering layer 20 (4 hc streams) and its pre-mix
The first non-extend forward after a prefill writes one safetensors file to SPLIT_NV_DIR.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

from . import macpack

SOURCE_LAYERS = (2, 8, 14, 20)
ENCODER_LAYERS = 21
WINDOW = 128
TAIL_ROWS = 256

CAP = None


class Capture:
    def __init__(self, text_config, out_dir, max_tokens):
        self.text = text_config
        self.ratios = list(text_config["compress_ratios"])
        self.rope = macpack.RopeParams(text_config)
        self.out_dir = out_dir
        self.max_tokens = max_tokens
        self.enabled = None  # decided at first forward (rank 0 only)
        self.auto = True  # server mode: bookkeeping + dump driven by the forward wrapper; the engine sets False
        self.step = None  # dict while capturing a decode step (engine mode)
        self.cur_mode = None
        self.cur_pos = None
        self.trace = None  # opt-in eager numerical diagnostic; never serving
        self.host = {}  # (kind, layer) -> pinned host buffer
        self.collect = False  # engine mode: source rows are handed out per chunk instead of written to host_buf
        self.reset()

    def reset(self):
        self.tokens = []
        self.ntok = 0
        self.swa = {}  # layer -> (buf[128,528] u8, pos[128] i64)
        self.rows = {L: 0 for L in SOURCE_LAYERS}
        self.tail = {}  # layer -> (pos i64[1], kv f32[512], gate f32[512])
        self.h_ring = None
        self.pending = False
        self.chunks = []
        self.t_start = None
        self.first_chunk_wall = None
        self.chunk_rows = {L: [] for L in SOURCE_LAYERS}  # collect mode: GPU (ckv, idxk) packed rows of this chunk

    # ---- buffers -------------------------------------------------------
    def host_buf(self, kind, layer, width, device):
        key = (kind, layer)
        if key not in self.host:
            ratio = self.ratios[layer]
            rows = self.max_tokens // ratio + 2
            self.host[key] = torch.empty((rows, width), dtype=torch.uint8, pin_memory=True)
        return self.host[key]

    def swa_buf(self, layer, device):
        if layer not in self.swa:
            self.swa[layer] = (
                torch.zeros((WINDOW, 528), dtype=torch.uint8, device=device),
                torch.full((WINDOW,), -1, dtype=torch.int64, device=device),
            )
        return self.swa[layer]

    # ---- capture points ------------------------------------------------
    def on_swa(self, layer_id, kv, kv_weight, eps, positions):
        if layer_id >= ENCODER_LAYERS:
            return
        n = kv.shape[0]
        take = min(n, WINDOW)
        rows = kv[-take:]
        pos = positions[-take:].to(torch.int64)
        normed = macpack.rms_norm(rows, kv_weight, eps)
        freq = self.rope.freq(self.ratios[layer_id] > 0, kv.device)
        packed = macpack.pack_swa_row(macpack.apply_rope(normed, pos, freq))
        buf, bpos = self.swa_buf(layer_id, kv.device)
        slot = pos % WINDOW
        buf[slot] = packed
        bpos[slot] = pos

    def on_write_group(self, layer, pooled, group_pos):
        L = layer.layer_id
        if L not in SOURCE_LAYERS or self.cur_pos is None:
            return
        ratio = layer.compress_ratio
        pos = self.cur_pos
        if ratio == 2:
            valid = pos % 2 == 1
            pooled = pooled[valid]
            group_pos = group_pos[valid]
        n = pooled.shape[0]
        if n == 0:
            return
        latent = layer.compressor.finish(pooled)
        gpos = group_pos.to(torch.int64)
        freq = self.rope.freq(True, latent.device)
        ckv = macpack.pack_ckv_row(macpack.apply_rope(latent, gpos, freq))
        k = layer.indexer.k_norm(layer.indexer.forward_wk(latent))
        idxk = macpack.pack_idxk_row(macpack.apply_rope(k, gpos, freq))
        if self.step is not None:
            if L == ENCODER_LAYERS - 1:
                self.step["ckv"], self.step["idxk"], self.step["kv_pos"] = ckv, idxk, gpos
            return
        rowidx = gpos // ratio
        r0 = int(rowidx[0])
        span = int(rowidx[-1]) - r0 + 1
        if span != n or r0 > self.rows[L]:
            raise RuntimeError(f"split-nv: non-contiguous rows layer {L}: r0={r0} span={span} n={n} have={self.rows[L]}")
        if self.collect:
            if r0 != self.rows[L]:
                raise RuntimeError(f"split-nv: rows layer {L} restart at {r0}, have {self.rows[L]}")
            self.chunk_rows[L].append((ckv, idxk))
        else:
            self.host_buf("ckv", L, 288, latent.device)[r0:r0 + n].copy_(ckv, non_blocking=True)
            self.host_buf("idxk", L, 68, latent.device)[r0:r0 + n].copy_(idxk, non_blocking=True)
        self.rows[L] = r0 + n

    def on_compress_inputs(self, layer, pos, kv, score):
        if layer.compress_ratio == 2 and score is not None and pos.shape[0] > 0:
            self.tail[layer.layer_id] = (pos[-1:].to(torch.int64).clone(), kv[-1].float().clone(), score[-1].float().clone())

    def on_layer20_input(self, hidden_states, prev_pre, positions):
        if self.step is not None:
            self.step["h"] = hidden_states.clone()
            self.step["pre"] = prev_pre.float().clone()
            self.step["pos"] = positions.clone()
            return
        n = hidden_states.shape[0]
        take = min(n, TAIL_ROWS)
        if self.h_ring is None:
            self.h_ring = (
                torch.zeros((TAIL_ROWS, *hidden_states.shape[1:]), dtype=hidden_states.dtype, device=hidden_states.device),
                torch.zeros((TAIL_ROWS, hidden_states.shape[1]), dtype=torch.float32, device=hidden_states.device),
                torch.full((TAIL_ROWS,), -1, dtype=torch.int64, device=hidden_states.device),
            )
        h, pre, hpos = self.h_ring
        pos = positions[-take:].to(torch.int64)
        slot = pos % TAIL_ROWS
        h[slot] = hidden_states[-take:]
        pre[slot] = prev_pre[-take:].float()
        hpos[slot] = pos

    # ---- collect mode (engine) -------------------------------------------
    def take_chunk_rows(self):
        """Host copies of the source rows produced since the last call: {L: (ckv u8[n,288], idxk u8[n,68])}.
        Call after the chunk's GPU work finished (the engine synchronizes)."""
        out = {}
        for L in SOURCE_LAYERS:
            parts = self.chunk_rows[L]
            if parts:
                ckv = torch.cat([c for c, _ in parts]) if len(parts) > 1 else parts[0][0]
                idxk = torch.cat([i for _, i in parts]) if len(parts) > 1 else parts[0][1]
                out[L] = (ckv.cpu(), idxk.cpu())
            self.chunk_rows[L] = []
        return out

    def final_parts(self):
        """Host copies of everything that is only final after the last chunk (split-nv-raw-v1 names)."""
        n1 = self.ntok
        out = {}
        for L in range(ENCODER_LAYERS):
            buf, bpos = self.swa[L]
            keep = bpos >= 0
            order = torch.argsort(bpos[keep])
            rows = buf[keep][order].cpu()
            if rows.shape[0] != min(n1, WINDOW):
                raise RuntimeError(f"split-nv: swa rows layer {L}: {rows.shape[0]} != {min(n1, WINDOW)}")
            out[f"swa.{L}"] = rows
        for L in SOURCE_LAYERS:
            if self.rows[L] != n1 // self.ratios[L]:
                raise RuntimeError(f"split-nv: layer {L} rows {self.rows[L]} != {n1 // self.ratios[L]}")
            if self.ratios[L] == 2:
                if n1 % 2 == 1:
                    tpos, kv, gate = self.tail[L]
                    if int(tpos[0]) != n1 - 1:
                        raise RuntimeError(f"split-nv: layer {L} tail pos {int(tpos[0])} != {n1 - 1}")
                    out[f"tail_kv.{L}"] = kv[None].cpu()
                    out[f"tail_gate.{L}"] = gate[None].cpu()
                else:
                    out[f"tail_kv.{L}"] = torch.zeros((0, 512), dtype=torch.float32)
                    out[f"tail_gate.{L}"] = torch.zeros((0, 512), dtype=torch.float32)
        h, pre, hpos = self.h_ring
        keep = hpos >= 0
        order = torch.argsort(hpos[keep])
        out["tail_hidden"] = h[keep][order].cpu()
        out["tail_pre"] = pre[keep][order].cpu()
        tail_pos = hpos[keep][order]
        if int(tail_pos[-1]) != n1 - 1:
            raise RuntimeError("split-nv: hidden tail does not end at the last prompt token")
        out["tail_first_position"] = int(tail_pos[0])
        return out

    def export_state(self):
        """Everything a later prefill continuing from this prefix needs (host copies)."""
        return {
            "ntok": self.ntok,
            "rows": dict(self.rows),
            "swa": {L: (b.cpu(), p.cpu()) for L, (b, p) in self.swa.items()},
            "tail": {L: tuple(t.cpu() for t in v) for L, v in self.tail.items()},
            "h_ring": tuple(t.cpu() for t in self.h_ring) if self.h_ring is not None else None,
        }

    def import_state(self, st, tokens, device):
        self.reset()
        self.ntok = st["ntok"]
        self.tokens = [torch.as_tensor(tokens, dtype=torch.int64)]
        self.rows = dict(st["rows"])
        self.swa = {L: (b.to(device), p.to(device)) for L, (b, p) in st["swa"].items()}
        self.tail = {L: tuple(t.to(device) for t in v) for L, v in st["tail"].items()}
        self.h_ring = tuple(t.to(device) for t in st["h_ring"]) if st["h_ring"] is not None else None

    # ---- dump ------------------------------------------------------------
    def dump(self):
        from safetensors.torch import save_file

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n1 = self.ntok
        ids = torch.cat(self.tokens).to(torch.int64).numpy()
        assert ids.shape[0] == n1
        digest = hashlib.sha256(ids.astype("<u4").tobytes()).hexdigest()
        tensors = {"tokens_prefix": torch.from_numpy(ids.astype(np.int32))}
        for L in range(ENCODER_LAYERS):
            buf, bpos = self.swa[L]
            keep = bpos >= 0
            order = torch.argsort(bpos[keep])
            rows = buf[keep][order].cpu()
            w = min(n1, WINDOW)
            if rows.shape[0] != w:
                raise RuntimeError(f"split-nv: swa rows layer {L}: {rows.shape[0]} != {w}")
            tensors[f"swa.{L}"] = rows
        for L in SOURCE_LAYERS:
            ratio = self.ratios[L]
            rows = self.rows[L]
            if rows != n1 // ratio:
                raise RuntimeError(f"split-nv: layer {L} rows {rows} != {n1 // ratio}")
            tensors[f"ckv.{L}"] = self.host_buf("ckv", L, 288, None)[:rows].clone()
            tensors[f"idxk.{L}"] = self.host_buf("idxk", L, 68, None)[:rows].clone()
            if ratio == 2:
                if n1 % 2 == 1:
                    tpos, kv, gate = self.tail[L]
                    if int(tpos[0]) != n1 - 1:
                        raise RuntimeError(f"split-nv: layer {L} tail pos {int(tpos[0])} != {n1 - 1}")
                    tensors[f"tail_kv.{L}"] = kv[None].cpu()
                    tensors[f"tail_gate.{L}"] = gate[None].cpu()
                else:
                    tensors[f"tail_kv.{L}"] = torch.zeros((0, 512), dtype=torch.float32)
                    tensors[f"tail_gate.{L}"] = torch.zeros((0, 512), dtype=torch.float32)
        h, pre, hpos = self.h_ring
        keep = hpos >= 0
        order = torch.argsort(hpos[keep])
        tensors["tail_hidden"] = h[keep][order].cpu()
        tensors["tail_pre"] = pre[keep][order].cpu()
        tail_pos = hpos[keep][order]
        if int(tail_pos[-1]) != n1 - 1:
            raise RuntimeError("split-nv: hidden tail does not end at the last prompt token")
        meta = {
            "format": "split-nv-raw-v1",
            "prefix_tokens": n1,
            "prefix_sha256": digest,
            "tail_first_position": int(tail_pos[0]),
            "chunks": self.chunks,
            "prefill_seconds": sum(c[1] for c in self.chunks),
            "wall_seconds": (time.perf_counter() - self.t_start) if self.t_start else None,
            "layers": ENCODER_LAYERS,
        }
        os.makedirs(self.out_dir, exist_ok=True)
        final = os.path.join(self.out_dir, f"raw-{digest}.safetensors")
        tmp = final + ".tmp"
        save_file(tensors, tmp, metadata={"meta": json.dumps(meta)})
        os.chmod(tmp, 0o644)
        os.replace(tmp, final)
        self.pending = False
        print(f"[split-nv] dumped {final} tokens={n1} prefill={meta['prefill_seconds']:.2f}s "
              f"({n1 / max(meta['prefill_seconds'], 1e-9):.0f} tok/s) dump={time.perf_counter() - t0:.2f}s", flush=True)


TRIM = os.environ.get("SPLIT_NV_TRIM") == "1"


def _layer20_sources_only(layer, positions, hidden_states, forward_batch, prev_pre):
    """Layer 20 on the box: only what the boundary needs (its window KV, ratio-1 compressed KV and index keys).

    The encoder's outputs are the hidden state entering layer 20 (captured before this call) and those
    rows, so layer 20's attention, MoE and indexer queries are never used; skipping them does not
    change any exported byte. Same kernels as the full layer up to the cache writes.
    """
    attn = layer.self_attn
    x, attn_pre, _, _ = layer._hc_mix_and_combine(
        hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, apply_pre=prev_pre, stats_stream=None)
    x = layer.input_layernorm(x)
    backend = _attn_backend()
    qkv_a, _ = attn.wqkv_a(x)
    attn._compute_kv_to_cache(x, positions, forward_batch, backend, qkv_a=qkv_a)
    q_lora = attn.q_norm(qkv_a[..., : attn.q_lora_rank])
    backend.forward_low_ratio_sources(layer=attn, x=x, q_lora=q_lora, positions=positions,
                                      forward_batch=forward_batch, run_indexer=False)
    return hidden_states, attn_pre


def _install_b12x():
    """Route FlashInfer MXFP8 dense GEMMs from the CUTLASS backend to b12x.

    Bit-identical outputs for every model shape at every M (checked standalone and by byte-equal
    8K prefill dumps), batch-invariant, and 3-12x faster at step sizes where the CUTLASS tactic is
    pathological (e.g. K=1280 N=16384 M=5: 72 -> 6 us). Shapes outside b12x's requirements fall back.
    """
    from sglang.srt.layers.quantization import fp8_utils as F

    orig = F.flashinfer_mm_mxfp8
    ok = {}

    def mm(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False, backend="auto"):
        key = (q_input.shape[1], weight_t.shape[1])
        if backend == "cutlass" and ok.get(key, True):
            try:
                out = orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                           use_8x4_sf_layout=use_8x4_sf_layout, backend="b12x")
                ok[key] = True
                return out
            except Exception:  # noqa: BLE001
                ok[key] = False
        return orig(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                    use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)

    F.flashinfer_mm_mxfp8 = mm


def _attn_backend():
    from sglang.srt.models import deepseek_v4 as M

    return M.get_attn_backend()


def _rank0():
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank() == 0


def install():
    global CAP
    if CAP is not None:
        return
    with open(os.environ["SPLIT_NV_CONFIG"]) as f:
        text = json.load(f)["text_config"]
    CAP = Capture(text, os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"), int(os.environ.get("SPLIT_NV_MAX_TOKENS", "1056768")))
    cap = CAP

    from sglang.srt.layers.attention import deepseek_v4_backend as B
    from sglang.srt.mem_cache import deepseek_v4_memory_pool as P
    from sglang.srt.models import deepseek_v4 as M

    orig_forward = M.DeepseekV4Model.forward

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        if cap.enabled is None:
            cap.enabled = _rank0()
            print(f"[split-nv] hooks active on this rank: {cap.enabled}", flush=True)
        if not cap.enabled:
            return orig_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        if cap.step is not None:
            cap.cur_mode = "step"
            try:
                return orig_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
            finally:
                cap.cur_mode = None
        if forward_batch.forward_mode.is_extend() and not cap.auto:
            cap.cur_mode = "extend"
            try:
                return orig_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
            finally:
                cap.cur_mode = None
        if forward_batch.forward_mode.is_extend():
            if int(positions[0]) == 0:
                cap.reset()
                cap.t_start = time.perf_counter()
            cap.cur_mode = "extend"
            cap.tokens.append(input_ids.detach().to("cpu"))
            cap.ntok += int(input_ids.shape[0])
            cap.pending = True
            t0 = time.perf_counter()
            try:
                out = orig_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
            finally:
                cap.cur_mode = None
            torch.cuda.synchronize()
            cap.chunks.append((int(input_ids.shape[0]), time.perf_counter() - t0))
            return out
        cap.cur_mode = None
        if cap.pending and cap.auto:
            cap.dump()
        return orig_forward(self, input_ids, positions, forward_batch, *args, **kwargs)

    M.DeepseekV4Model.forward = forward

    orig_layer = M.DeepseekV4DecoderLayer.forward_hc_pre_from_prev

    def layer_forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global, prev_pre):
        if cap.trace is not None:
            mask = positions >= cap.trace_start
            cap.trace_mask = mask
            cap.trace_positions = positions
            from split_nv.numerics import record
            record(cap, str(self.layer_id), hidden_states)
        if cap.cur_mode in ("extend", "step") and self.layer_id == ENCODER_LAYERS - 1:
            cap.on_layer20_input(hidden_states, prev_pre, positions)
        if TRIM and self.layer_id == ENCODER_LAYERS - 1:  # every rank, so collectives stay matched
            return _layer20_sources_only(self, positions, hidden_states, forward_batch, prev_pre)
        return orig_layer(self, positions, hidden_states, input_ids, forward_batch, input_ids_global, prev_pre)

    M.DeepseekV4DecoderLayer.forward_hc_pre_from_prev = layer_forward

    if os.environ.get("SPLIT_NV_B12X") == "1":
        _install_b12x()

    if TRIM:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        orig_causal = M.DeepseekV4ForCausalLM.forward

        def causal_forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
            # Nothing on the box consumes logits (every rank runs this): skip the LM head GEMM.
            with M.get_attn_tp_context().maybe_input_scattered(forward_batch):
                self.model.forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
            return LogitsProcessorOutput(next_token_logits=None)

        M.DeepseekV4ForCausalLM.forward = causal_forward

    orig_swa = P.DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope

    def set_swa(self, layer_id, swa_loc, kv, kv_weight, eps, freqs_cis, positions):
        orig_swa(self, layer_id, swa_loc, kv, kv_weight, eps, freqs_cis, positions)
        if cap.cur_mode == "extend":
            cap.on_swa(layer_id, kv, kv_weight, eps, positions)

    P.DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope = set_swa

    orig_compress = B.DeepseekV4AttnBackend._low_ratio_compress_torch

    def compress(self, layer, x, req, pos, projected=None):
        if cap.cur_mode not in ("extend", "step") or (cap.cur_mode == "step" and layer.layer_id != ENCODER_LAYERS - 1):
            return orig_compress(self, layer, x, req, pos, projected)
        if projected is None and pos.shape[0]:
            projected = layer.compressor.project(x)
        if projected is not None and cap.cur_mode == "extend":
            cap.on_compress_inputs(layer, pos, projected[0], projected[1])
        cap.cur_pos = pos.to(torch.int64)
        try:
            return orig_compress(self, layer, x, req, pos, projected)
        finally:
            cap.cur_pos = None

    B.DeepseekV4AttnBackend._low_ratio_compress_torch = compress

    orig_write = B.DeepseekV4AttnBackend._low_ratio_write_group

    def write_group(self, layer, pooled, slots, group_pos, *, fuse_index_store=False):
        orig_write(self, layer, pooled, slots, group_pos, fuse_index_store=fuse_index_store)
        if cap.cur_mode == "extend" or (cap.cur_mode == "step" and layer.layer_id == ENCODER_LAYERS - 1):
            cap.on_write_group(layer, pooled, group_pos)

    B.DeepseekV4AttnBackend._low_ratio_write_group = write_group
    print("[split-nv] hooks installed", flush=True)
