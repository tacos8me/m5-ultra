"""Original-precision DS41 layers 20..39; encoder sessions live on the RTX box."""
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from .cache import DeepseekV41Cache
from .config import ModelConfig
from .dspark import make_stages
from .encoder_replay import FORMAT, build_cache, empty_slot, replay
from .handoff import token_digest
from .head import project_logits
from .language import Block, LanguageModel, RMSNorm, hc_pre
from .loading import _load_shard, set_module
from .pipe_wire import mlx_state
from .quantization import QuantizedProjection
from . import growth


_NP_KINDS = {"<u4": ("U32",), "<i4": ("I32",)}


class DecoderHalf(LanguageModel):
    """Keep absolute layer IDs and the served DSpark/rollback arithmetic."""
    def __init__(self, config):
        nn.Module.__init__(self)
        self._config = config
        if config.n_layers != 40 or config.dim != 5120 or not config.preserve_mtp:
            raise ValueError("Expected original 40-layer DS41 with DSpark")
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [nn.Module() for _ in range(20)] + [Block(config, i) for i in range(20, 40)]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.mtp = make_stages(config)
        self._hasher = None
        self._async_every = int(os.environ.get("DS41_DECODE_ASYNC", "4"))
        # Also submit after these first layers, so the GPU starts while
        # Python still builds the rest (scheduling only).
        self._async_front = tuple(int(x) for x in os.environ.get("DS41_DECODE_ASYNC_FRONT", "1").split(",") if x)
        self.configure_mtp(True, 4)

    def import_prefill(self, arrays, manifest, tokens, *, identity):
        cache, _ = build_cache(self, arrays, manifest, tokens, identity=identity)
        # Encoder caches are remote after the handoff. Keep offset placeholders
        # for the served DSpark prompt-ring seam, without retaining global data.
        for i in range(20):
            for slot in range(1, 7):
                cache[i][slot] = empty_slot(self._config, slot, mx.bfloat16)
        return cache

    def import_state(self, tensors, manifest, tokens, *, identity, base_rows=None):
        """Import a full, lean or delta ``ds41-encoder-state-v1`` (raw wire tensors).

        Returns (cache, rows): rows = layer 20's packed global KV / index K for
        positions [0, N-1), which a prefix store may keep for a later delta
        import. A delta state (manifest ``delta_from`` = P) carries only rows
        [P, N-1); ``base_rows`` supplies [0, P) for the same token prefix.

        Only layer 20, the tail and the offsets become MLX arrays: layers 0-19
        run on the box and this half never reads their Mac-packed caches, so the
        result equals import_prefill() for the same bytes (same replay).
        """
        c = self._config
        mid = c.n_layers // 2
        prefilled = len(tokens) - 1
        if (manifest["format"] != FORMAT or manifest["identity"] != identity
                or manifest["prompt_tokens"] != len(tokens) or manifest["token_sha256"] != token_digest(tokens)):
            raise ValueError("Encoder state format, identity or prompt mismatch")
        layers = manifest["layers"]
        if manifest.get("encoder_layers", len(layers)) != mid + 1 or len(layers) != mid + 1:
            raise ValueError("Encoder state must carry layers 0..%d" % mid)

        def raw(name, dtype, count=None):
            kind, shape, data, _ = tensors[name]
            value = np.frombuffer(data, dtype)
            if kind not in _NP_KINDS[dtype] or (count is not None and value.size != count):
                raise ValueError("Unexpected encoder tensor " + name)
            return value

        if raw("tokens", "<u4").tolist() != list(tokens):
            raise ValueError("Encoder state tokens mismatch")
        ratios = [c.compress_ratios[i] if i in c.kv_source_layers else 0 for i in range(mid + 1)]
        for i, layer in enumerate(layers):
            # A lean state keeps only the offset of layers 0-19 (manifest ratio 0).
            if i == mid:
                valid = layer["compress_ratio"] == ratios[i] and len(layer["slots"]) == 7
            else:
                valid = layer["compress_ratio"] in (ratios[i], 0) and 1 <= len(layer["slots"]) <= 7
            if not valid:
                raise ValueError("Encoder layer %d layout mismatch" % i)
            if int(raw(layer["slots"][0], "<i4", 1)[0]) != prefilled:
                raise ValueError("Encoder layer %d offset mismatch" % i)
        tail = manifest["tail"]
        rows, first = int(tail["rows"]), int(tail["first_position"])
        if tail.get("layer", mid) != mid or first + rows != prefilled or rows < min(prefilled, 2 * c.window_size):
            raise ValueError("Encoder tail does not cover the replay rows")
        layer = layers[mid]
        names = [n for n in layer["slots"] if n is not None and n in tensors]
        names += [n for n in (layer.get("left_padding"), layer.get("lengths")) if n is not None]
        arrays = mlx_state(tensors, names + [tail["hidden"], tail["pre"]])
        hidden, pre = arrays[tail["hidden"]], arrays[tail["pre"]]
        if hidden.shape != (1, rows, c.hc_mult, c.dim) or pre.shape != (1, rows, c.hc_mult):
            raise ValueError("Encoder tail shapes mismatch")
        delta = int(manifest.get("delta_from") or 0)
        slots = []
        for j, name in enumerate(layer["slots"]):
            if name is None or name in arrays:
                slots.append(arrays.get(name))
            elif j in (2, 3) and delta == prefilled:
                # A streamed zero-row delta (exact repeat of a stored prompt) carries no row tensor.
                slots.append(mx.zeros((1, 0, 288 if j == 2 else 68), mx.uint8))
            else:
                raise ValueError("Encoder state lacks " + name)
        if delta:
            if base_rows is None or not 0 < delta <= prefilled:
                raise ValueError("Delta encoder state without base rows")
            for slot, base in zip((2, 3), base_rows):
                if base.shape[1] < delta or slots[slot].shape[1] != prefilled - delta:
                    raise ValueError("Delta encoder rows do not continue the base rows")
                slots[slot] = mx.concatenate([base[:, :delta], slots[slot]], 1)
        if slots[2].shape != (1, prefilled, 288) or slots[3].shape != (1, prefilled, 68):
            raise ValueError("Layer %d global rows do not cover the prompt" % mid)
        cache = []
        for i in range(mid):
            item = DeepseekV41Cache(ratios[i])
            item.cache = [mx.array([prefilled], mx.int32)] + [empty_slot(c, s, mx.bfloat16) for s in range(1, 7)]
            item.left_padding = item.lengths = None
            cache.append(item)
        item = DeepseekV41Cache(ratios[mid])
        item.cache = slots
        item.left_padding = arrays.get(layer.get("left_padding"))
        item.lengths = arrays.get(layer.get("lengths"))
        cache.append(item)
        for i in range(mid + 1, c.n_layers):
            item = DeepseekV41Cache(0)
            item.cache = [mx.array([prefilled], mx.int32)] + [None] * 6
            item.left_padding = item.lengths = None
            cache.append(item)
        replay(self, cache, hidden, pre, first, tokens)
        return cache, (slots[2], slots[3])

    def forward_boundary(self, h, pre, cache, *, start, kv=None, index=None,
                         capture=True, verify=False, verify_states=None):
        """Forward a validated entering-layer20 boundary. Caller owns rollback.

        If KV/index increments are absent, compute layer20's global rows locally
        from the same original weights. Otherwise append each received row once,
        and use split-wire's prebuilt attention path.
        """
        c = self._config
        if h.ndim != 4 or h.shape[0] != 1 or tuple(h.shape[2:]) != (c.hc_mult, c.dim):
            raise ValueError("Invalid encoder hidden shape")
        length = h.shape[1]
        if not 1 <= length <= 5 or pre.shape != h.shape[:-1]:
            raise ValueError("Decode boundary must contain 1..5 rows")
        if h.dtype != mx.bfloat16 or pre.dtype != mx.float32:
            raise ValueError("Unexpected encoder hidden/pre dtype")
        if len(cache) != 40 or any(item.size() != start for item in cache):
            raise ValueError("Decoder/session offsets diverged")
        if (kv is None) != (index is None):
            raise ValueError("Supply both global KV and index increments")
        snapshots = [(list(x.cache), x.left_padding, x.lengths) for x in cache] if verify else None
        states = (verify_states if verify_states is not None else [{} for _ in cache]) if verify else None
        if kv is not None:
            if kv.shape != (1, length, 288) or index.shape != (1, length, 68):
                raise ValueError("Invalid layer20 increments")
            if kv.dtype != mx.uint8 or index.dtype != mx.uint8:
                raise ValueError("Expected packed layer20 bytes")
            for slot, value in ((2, kv), (3, index)):
                cache[20][slot] = growth.append(cache[20], slot, cache[20][slot], value, start)
        shared, captured = {}, {}
        for i in range(20, 40):
            if verify:
                cache[i]._mtp_verify_state = states[i]
            if capture and i in c.dspark_target_layer_ids:
                captured[i] = mx.mean(h, axis=2)
            h, pre = self.layers[i](h, pre, cache[i], shared, start, None,
                                   prebuilt_end=start+length if i == 20 and kv is not None else None)
            cache[i][0] = mx.array([start + length], mx.int32)
            for slot in range(1, 7):
                if cache[i][slot] is None:
                    cache[i][slot] = empty_slot(c, slot, h.dtype)
            if verify:
                del cache[i]._mtp_verify_state
            if self._async_every and ((i - 19) % self._async_every == 0 or (i - 19) in self._async_front):
                # Start the GPU on finished layers while Python builds the rest.
                mx.async_eval(h, pre)
        for i in range(20):
            cache[i][0] = mx.array([start + length], mx.int32)
        logits = project_logits(self.norm(hc_pre(h, pre)), self.head)
        hidden = mx.concatenate([captured[i] for i in c.dspark_target_layer_ids], -1) if capture else None
        if verify:
            cache[0]._pipe1_verify = (start, length, snapshots, states)
        return logits, hidden

    def rollback_boundary(self, cache, keep):
        """Retain the causal prefix of the most recent verify; remote mirrors it."""
        stash = getattr(cache[0], "_pipe1_verify", None)
        if stash is None:
            raise ValueError("No verify transaction")
        before, length, snapshots, states = stash
        if not 1 <= keep <= length or any(x.size() != before + length for x in cache):
            raise ValueError("Invalid verify commit")
        end = before + keep
        if keep != length:
            for i in range(20, 40):
                item = cache[i]
                window_end = min(before, self._config.window_size) + keep
                item[1] = states[i]["window"][:, max(0, window_end-self._config.window_size):window_end]
                if item.compress_ratio:
                    if item.compress_ratio != 1:
                        raise ValueError("Decoder only has ratio1 global cache")
                    growth.truncate(item, 2, end)
                    growth.truncate(item, 3, end)
        for item, snapshot in zip(cache, snapshots):
            item[0] = mx.array([end], mx.int32)
            _, item.left_padding, item.lengths = snapshot
        cache[0]._pipe1_verify = None


class DecoderContainer(nn.Module):
    def __init__(self, config, cls=None):
        super().__init__()
        self.language_model = (cls or DecoderHalf)(config)


def load_decoder(path, cls=None):
    """Load only the extracted tensors, rejecting holes and wrong shapes."""
    path = Path(path)
    if (path / "conversion.inprogress.json").exists():
        raise ValueError("Checkpoint extraction is incomplete")
    raw = json.loads((path / "config.json").read_text())
    if raw.get("pipe1", {}).get("version") != 1:
        raise ValueError("Expected a pipe1 checkpoint")
    config = ModelConfig.from_dict(raw)
    config.ced_prefill = True
    model = DecoderContainer(config, cls)
    expected = {k: v.shape for k, v in tree_flatten(model.parameters())}
    mapping = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    specs = raw["omlx_deepseek_v41"]["quantized_modules"]
    seen = set()
    begin = time.monotonic()
    for filename in dict.fromkeys(mapping.values()):
        values = _load_shard(path / filename)
        if set(values) != {k for k,v in mapping.items() if v == filename} or seen.intersection(values):
            raise ValueError("Misplaced/duplicate checkpoint tensor")
        quantized = {name: spec for name, spec in specs.items() if name+".weight" in values}
        for name, spec in quantized.items():
            logical = expected[name+".weight"]
            packed = (*logical[:-1], logical[-1]*spec["bits"]//32)
            scales = (*logical[:-1], logical[-1]//32)
            if values[name+".weight"].shape != packed or values[name+".scales"].shape != scales:
                raise ValueError("Invalid packed projection: " + name)
            set_module(model, name, QuantizedProjection(values[name+".weight"], values[name+".scales"], **spec))
        for key, value in values.items():
            if key.rsplit(".",1)[0] not in quantized and expected.get(key) != value.shape:
                raise ValueError("Unexpected tensor: " + key)
        model.load_weights(list(values.items()), strict=False)
        mx.eval(values)
        seen.update(values)
        print(json.dumps(dict(event="load_shard", file=filename, seconds=time.monotonic()-begin,
                              active_gib=mx.get_active_memory()/2**30)), flush=True)
        del values
        mx.clear_cache()
    if seen != {k for k,v in tree_flatten(model.parameters())}:
        raise ValueError("Incomplete decoder checkpoint")
    model.eval()
    return model.language_model
