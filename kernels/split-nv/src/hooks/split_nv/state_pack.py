"""Assemble a `ds41-encoder-state-v1` safetensors blob from a raw hook dump (shared by server.py and engine.py)."""

import os
import hashlib
import json
import struct

import numpy as np
import torch
from safetensors.torch import load_file

FORMAT = "ds41-encoder-state-v1"
SOURCE = {2: 2, 8: 2, 14: 2, 20: 1}
ENCODER_LAYERS = 21
_DTYPES = {torch.uint8: "U8", torch.int32: "I32", torch.int64: "I64", torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
PRODUCER = {"engine": "sglang dsv41-attn-longctx 757e8f35 + split-nv hooks", "weights": "DeepSeek-V4.1-Flash FP8/FP4 originals",
            "layers": "0-20 all 384 experts", "gpus": "2x RTX PRO 6000 Blackwell (TP2)"}


class TokenMap:
    def __init__(self, model_dir, expected_size):
        from transformers import AutoTokenizer

        from .engram_map import build_compressed_token_map

        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
        lookup, size = build_compressed_token_map(tok)
        if size != expected_size:
            raise RuntimeError(f"compressed vocab {size} != {expected_size}")
        self.map = np.asarray(lookup, dtype=np.int64)

    def history(self, ids):
        mapped = self.map[np.asarray(ids[-3:], dtype=np.int64)]
        pad = 3 - mapped.shape[0]
        return torch.from_numpy(np.concatenate([np.full(pad, -1, np.int64), mapped]))[None]


def serialize(arrays, metadata):
    """Minimal safetensors writer (uint32 tokens, bf16 payloads) independent of the safetensors package version."""
    header, chunks, pos = {"__metadata__": metadata}, [], 0
    for name, t in arrays.items():
        if isinstance(t, np.ndarray):
            dtype, data = {np.dtype("uint32"): "U32"}[t.dtype], t.tobytes()
        else:
            dtype = _DTYPES[t.dtype]
            data = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).contiguous().numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
        chunks.append(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    return b"".join([struct.pack("<Q", len(raw)), raw, *chunks])


def token_digest(ids):
    return hashlib.sha256(struct.pack("<%dI" % len(ids), *ids)).hexdigest()


def prefix_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def assemble(raw_path, tokens, identity, token_map: TokenMap, timing):
    raw = load_file(str(raw_path))
    with open(raw_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(json.loads(f.read(n))["__metadata__"]["meta"])
    n1 = meta["prefix_tokens"]
    if n1 != len(tokens) - 1 or raw["tokens_prefix"].tolist() != list(tokens[:-1]):
        raise RuntimeError("raw state does not match the request tokens")
    arrays = {"tokens": np.asarray(tokens, dtype=np.uint32)}
    layers = []
    for i in range(ENCODER_LAYERS):
        ratio = SOURCE.get(i, 0)
        slots = []

        def put(j, value):
            name = f"layer.{i}.slot.{j}"
            arrays[name] = value
            slots.append(name)

        put(0, torch.tensor([n1], dtype=torch.int32))
        put(1, raw[f"swa.{i}"][None])
        if ratio:
            put(2, raw[f"ckv.{i}"][None])
            put(3, raw[f"idxk.{i}"][None])
        else:
            put(2, torch.zeros((1, 0, 288), dtype=torch.uint8))
            put(3, torch.zeros((1, 0, 68), dtype=torch.uint8))
        if ratio > 1:
            put(4, raw[f"tail_kv.{i}"][None])
            put(5, raw[f"tail_gate.{i}"][None])
        else:
            put(4, torch.zeros((1, 0, 512), dtype=torch.bfloat16))
            put(5, torch.zeros((1, 0, 512), dtype=torch.bfloat16))
        put(6, token_map.history(tokens[:-1]) if i == 0 else torch.zeros((1, 0), dtype=torch.int64))
        layers.append({"compress_ratio": ratio, "slots": slots, "left_padding": None, "lengths": None})
    arrays["tail.hidden"] = raw["tail_hidden"][None]
    arrays["tail.pre"] = raw["tail_pre"][None]
    size = sum(x.nbytes if isinstance(x, np.ndarray) else x.numel() * x.element_size() for x in arrays.values())
    manifest = dict(
        format=FORMAT, identity=identity, token_sha256=token_digest(tokens), prompt_tokens=len(tokens), bytes=size,
        layers=layers, encoder_layers=ENCODER_LAYERS,
        tail={"first_position": meta["tail_first_position"], "rows": int(raw["tail_hidden"].shape[0]),
              "hidden": "tail.hidden", "pre": "tail.pre", "layer": ENCODER_LAYERS - 1},
        timing=dict(timing, prefill_seconds=meta["prefill_seconds"], chunks=meta["chunks"],
                    prefill_tok_s=n1 / max(meta["prefill_seconds"], 1e-9)),
        producer=PRODUCER,
    )
    return serialize(arrays, {"manifest": json.dumps(manifest)}), manifest


# ---- engine path: the same arrays from in-memory parts (no raw file), whole or streamed -----------------------


def dtype_name(t):
    return "U32" if isinstance(t, np.ndarray) else _DTYPES[t.dtype]


def tensor_bytes(t):
    if isinstance(t, np.ndarray):
        return t.tobytes()
    return (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).contiguous().numpy().tobytes()


# og-s4.3: deterministic top-k ties (lowest index); og-s4.4: og-moe routed + shared experts (official weight order,
# promoted MXFP4/MXFP8 GEMMs, same arithmetic for prefill and steps). Bump on any change of exported bytes.
NUMERICS = "og-s4.4" if os.environ.get("SPLIT_NV_OG_MOE") == "1" else "og-s4.3"
LEAN_LAYER = ENCODER_LAYERS - 1


def row_names(n1, lean=False, delta_from=0):
    """Streamed (growing) tensors: name -> (source layer, slot, ratio, first row, rows, width)."""
    out = {}
    for L, ratio in SOURCE.items():
        if lean and L != LEAN_LAYER:
            continue
        first = delta_from // ratio
        rows = n1 // ratio - first
        out[f"layer.{L}.slot.2"] = (L, 2, ratio, first, rows, 288)
        out[f"layer.{L}.slot.3"] = (L, 3, ratio, first, rows, 68)
    return out


def _skip_rows(chunks, first):
    out, pos = [], 0
    for c in chunks:
        n = int(c.shape[0])
        if pos + n > first:
            out.append(c[max(0, first - pos):])
        pos += n
    return out


def assemble_parts(tokens, rows, final, identity, token_map: TokenMap, timing, lean=False, delta_from=0):
    """Same arrays/manifest as assemble(), from engine parts.

    rows: {L: (list of ckv u8[n,288] chunks, list of idxk u8[n,68] chunks)} in position order;
    final: Capture.final_parts() dict (split-nv-raw-v1 names + tail_first_position).
    Row tensors are left as chunk lists (keyed '<name>' -> list) so a streamer can send them without concatenation.
    """
    n1 = len(tokens) - 1
    arrays = {"tokens": np.asarray(tokens, dtype=np.uint32)}
    layers = []
    for i in range(ENCODER_LAYERS):
        ratio = SOURCE.get(i, 0)
        slots = []

        def put(j, value):
            name = f"layer.{i}.slot.{j}"
            arrays[name] = value
            slots.append(name)

        put(0, torch.tensor([n1], dtype=torch.int32))
        if lean and i != LEAN_LAYER:
            # lean: layers 0-19 stay on the box; only the offset travels
            put(1, torch.zeros((1, 0, 528), dtype=torch.uint8))
            put(2, torch.zeros((1, 0, 288), dtype=torch.uint8))
            put(3, torch.zeros((1, 0, 68), dtype=torch.uint8))
            put(4, torch.zeros((1, 0, 512), dtype=torch.bfloat16))
            put(5, torch.zeros((1, 0, 512), dtype=torch.bfloat16))
            put(6, torch.zeros((1, 0), dtype=torch.int64))
            layers.append({"compress_ratio": 0, "slots": slots, "left_padding": None, "lengths": None})
            continue
        put(1, final[f"swa.{i}"][None])
        if ratio:
            ck, ik = rows.get(i, ([], []))
            first = delta_from // ratio
            put(2, RowChunks(_skip_rows(ck, first), n1 // ratio - first, 288))
            put(3, RowChunks(_skip_rows(ik, first), n1 // ratio - first, 68))
        else:
            put(2, torch.zeros((1, 0, 288), dtype=torch.uint8))
            put(3, torch.zeros((1, 0, 68), dtype=torch.uint8))
        if ratio > 1:
            put(4, final[f"tail_kv.{i}"][None])
            put(5, final[f"tail_gate.{i}"][None])
        else:
            put(4, torch.zeros((1, 0, 512), dtype=torch.bfloat16))
            put(5, torch.zeros((1, 0, 512), dtype=torch.bfloat16))
        put(6, token_map.history(tokens[:-1]) if i == 0 else torch.zeros((1, 0), dtype=torch.int64))
        layers.append({"compress_ratio": ratio, "slots": slots, "left_padding": None, "lengths": None})
    arrays["tail.hidden"] = final["tail_hidden"][None]
    arrays["tail.pre"] = final["tail_pre"][None]
    size = sum(a.nbytes if isinstance(a, (np.ndarray, RowChunks)) else a.numel() * a.element_size() for a in arrays.values())
    chunks = timing.pop("chunks", [])
    prefill_s = sum(c[1] for c in chunks)
    manifest = dict(
        format=FORMAT, identity=identity, token_sha256=token_digest(tokens), prompt_tokens=len(tokens), bytes=size,
        layers=layers, encoder_layers=ENCODER_LAYERS,
        tail={"first_position": final["tail_first_position"], "rows": int(final["tail_hidden"].shape[0]),
              "hidden": "tail.hidden", "pre": "tail.pre", "layer": ENCODER_LAYERS - 1},
        timing=dict(timing, prefill_seconds=prefill_s, chunks=chunks,
                    prefill_tok_s=(n1 - timing.get("resumed_tokens", 0)) / max(prefill_s, 1e-9)),
        producer=PRODUCER, numerics=NUMERICS, state="lean" if lean else "full",
    )
    if delta_from:
        manifest["delta_from"] = delta_from
    return arrays, manifest


class RowChunks:
    """A [1, rows, width] U8 tensor held as position-ordered row chunks."""

    def __init__(self, chunks, rows, width):
        got = sum(int(c.shape[0]) for c in chunks)
        if got != rows:
            raise RuntimeError(f"row chunks {got} != {rows}")
        self.chunks, self.rows, self.width = chunks, rows, width
        self.shape = (1, rows, width)
        self.nbytes = rows * width

    def tobytes(self):
        return b"".join(c.contiguous().numpy().tobytes() for c in self.chunks)


def serialize_parts(arrays, manifest):
    """safetensors bytes identical to serialize(assemble(...)) for the same arrays."""
    header, chunks, pos = {"__metadata__": {"manifest": json.dumps(manifest)}}, [], 0
    for name, t in arrays.items():
        if isinstance(t, RowChunks):
            dtype, data = "U8", t.tobytes()
        else:
            dtype, data = dtype_name(t), tensor_bytes(t)
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
        chunks.append(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    return b"".join([struct.pack("<Q", len(raw)), raw, *chunks])
