# SPDX-License-Identifier: MIT
"""Request-local geometric storage for packed compressed KV and index keys.

Only the live prefix is serialized. External replacement, extraction and DSpark
rollback invalidate the saved view; the next append reseeds from that prefix.
"""
import os
import mlx.core as mx

ENABLED = os.environ.get('DS41_GROWTH', '0') == '1'

def append(cache, slot, previous, values, start):
    if not ENABLED or not values.shape[1]:
        return mx.concatenate([previous, values], 1)
    end = start + values.shape[1]
    buffers = getattr(cache, '_ds41_buffers', None)
    if buffers is None:
        buffers = cache._ds41_buffers = {}
    old = buffers.get(slot)
    # Slice operations made by the caller may produce a new object. Compare
    # against the actual stored cache view, before the caller replaces it.
    valid = old is not None and cache[slot] is old[1] and start <= old[2]
    capacity = old[0].shape[1] if valid else 0
    if capacity < end:
        capacity = ((max(end, capacity + capacity//2, 256) + 255)//256)*256
        buffer = mx.zeros((values.shape[0], capacity, *values.shape[2:]), values.dtype)
        if start:
            buffer[:, :start] = previous[:, :start]
    elif valid:
        buffer = old[0]
    buffer[:, start:end] = values
    view = buffer[:, :end]
    buffers[slot] = (buffer, view, end)
    return view
