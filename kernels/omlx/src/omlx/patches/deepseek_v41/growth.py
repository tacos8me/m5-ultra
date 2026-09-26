# SPDX-License-Identifier: MIT
"""Request-local geometric storage for packed compressed KV and index keys.

Only the live prefix is serialized. External replacement, extraction and DSpark
rollback invalidate the saved view; the next append reseeds from that prefix.

Ping-pong storage (DS41_GROWTH_PINGPONG, default on): each slot keeps the
buffer behind the current view plus an idle buffer from the previous append.
A short append writes the rows the idle buffer is missing (the last append's
rows and the new ones) into it with one slice update. Nothing else refers to
the idle buffer by then, so MLX donates it and updates it in place: decode no
longer copies or allocates the whole compressed cache on every token. The
values of every view are unchanged; when a stray reference blocks donation,
MLX falls back to a copy with the same result. extract() keeps a batch-1
cache's arrays and registry, and truncate() records DSpark rollback, so
neither breaks the chain.
"""
import os
import mlx.core as mx

ENABLED = os.environ.get('DS41_GROWTH', '0') == '1'
PINGPONG = os.environ.get('DS41_GROWTH_PINGPONG', '1') == '1'
# Appends up to this many rows (decode, DSpark verify) use the ping-pong
# buffers; prefill chunks keep a single buffer so prefill memory is unchanged.
PINGPONG_ROWS = 64


def _capacity(end, rows):
    if rows > PINGPONG_ROWS:
        return ((max(end, 256) + 255) // 256) * 256
    return ((end + max(4096, end // 16) + 255) // 256) * 256


def _fresh(previous, values, start, end, capacity):
    parts = [previous[:, :start], values] if start else [values]
    if capacity > end:
        parts.append(mx.zeros((values.shape[0], capacity - end, *values.shape[2:]), values.dtype))
    return mx.concatenate(parts, 1)


def append(cache, slot, previous, values, start):
    if not ENABLED or not values.shape[1]:
        return mx.concatenate([previous, values], 1)
    if not PINGPONG:
        return _append_single(cache, slot, previous, values, start)
    end = start + values.shape[1]
    buffers = getattr(cache, '_ds41_buffers', None)
    if buffers is None:
        buffers = cache._ds41_buffers = {}
    entry = buffers.get(slot)
    chain = (
        isinstance(entry, dict)
        and cache[slot] is entry['view']
        and start <= entry['end']
        and previous.shape[1] >= start
    )
    buffer = None
    if chain and values.shape[1] <= PINGPONG_ROWS:
        idle, valid = entry['idle'], min(entry['valid'], start)
        entry['idle'] = None
        if (
            idle is not None
            and idle.shape[1] >= end
            and idle.shape[2:] == values.shape[2:]
            and idle.shape[0] == values.shape[0]
            and idle.dtype == values.dtype
        ):
            update = values if valid == start else mx.concatenate([previous[:, valid:start], values], 1)
            # The registry no longer holds `idle`; after this rebinding the
            # slice update is the buffer's only user and can donate it.
            idle[:, valid:end] = update
            buffer = idle
        del idle
    if buffer is None:
        buffer = _fresh(previous, values, start, end, _capacity(end, values.shape[1]))
    keep = chain and values.shape[1] <= PINGPONG_ROWS
    view = buffer[:, :end]
    buffers[slot] = {
        'buffer': buffer,
        'view': view,
        'end': end,
        # The buffer behind `previous` holds valid rows [0, start).
        'idle': entry['buffer'] if keep else None,
        'valid': start if keep else 0,
    }
    return view


def truncate(cache, slot, length):
    """cache[slot][:, :length], keeping the ping-pong registry valid."""
    value = cache[slot]
    view = value[:, :length]
    entry = getattr(cache, '_ds41_buffers', {}).get(slot)
    if isinstance(entry, dict) and value is entry['view'] and length <= entry['end']:
        entry['view'], entry['end'] = view, length
    cache[slot] = view
    return view


def _append_single(cache, slot, previous, values, start):
    end = start + values.shape[1]
    buffers = getattr(cache, '_ds41_buffers', None)
    if buffers is None:
        buffers = cache._ds41_buffers = {}
    old = buffers.get(slot)
    # Slice operations made by the caller may produce a new object. Compare
    # against the actual stored cache view, before the caller replaces it.
    valid = isinstance(old, tuple) and cache[slot] is old[1] and start <= old[2]
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
