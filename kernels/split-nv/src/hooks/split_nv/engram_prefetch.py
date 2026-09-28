"""Prefill: start every engram layer's host-table row gather on a side stream as soon as the chunk's hash ids exist.

In an 8K chunk each engram lookup (layers 1 and 14) is a ~7.4 ms zero-copy gather from the pinned host table over
PCIe that otherwise stalls the main stream. Launched early, it overlaps layers 0..13. Same kernel, same inputs, same
output bytes; only when it runs changes. A lookup whose indices do not match the prefetched shape (or any
non-EXTEND / captured forward) takes the normal path. Runtime switch: perf flag engram_prefetch (default off).
"""
import torch

from split_nv import pf_overlap
from split_nv.perf_flags import flag


def install(engine):
    from sglang.srt.layers import engram as E
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    model = engine.mr.model.model
    hasher = getattr(model, "engram_hasher", None)
    layers = [l.engram for l in model.layers if getattr(l, "engram", None) is not None]
    if hasher is None or not layers:
        return 0
    stream = torch.cuda.Stream()
    pending = {}
    orig_hash = type(hasher).forward
    orig_embed = E.EngramEmbedding.forward

    def hash_forward(self, input_ids, forward_batch, commit=True):
        ids = orig_hash(self, input_ids, forward_batch, commit)
        pending.clear()
        if (self is hasher and forward_batch is not None and forward_batch.forward_mode == ForwardMode.EXTEND
                and ids.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing()
                and flag("engram_prefetch", False) and pf_overlap._state is None):
            # (not while a chunk runs as overlapped halves: both halves would share the one pending slot per layer)
            main = torch.cuda.current_stream()
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                for eng in layers:
                    idx = ids[:, eng.layer_hash_index]
                    out = orig_embed(eng.embed, idx, forward_batch)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    pending[id(eng.embed)] = (tuple(idx.shape), out, ev)
            ids.record_stream(stream)
        return ids

    def embed_forward(self, indices, forward_batch=None, *, cp_all_tokens=False):
        hit = pending.pop(id(self), None)
        if hit is not None and not cp_all_tokens and hit[0] == tuple(indices.shape):
            _, out, ev = hit
            main = torch.cuda.current_stream()
            main.wait_event(ev)
            out.record_stream(main)
            return out
        return orig_embed(self, indices, forward_batch, cp_all_tokens=cp_all_tokens)

    type(hasher).forward = hash_forward
    E.EngramEmbedding.forward = embed_forward
    return len(layers)
