"""Engram native row reads and scheduler lookahead: bytes equal the mmap gather."""

import json
import struct
from concurrent.futures import Future

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import storage
from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding, EngramPrefetch


def _raw_safetensors(path, tensors):
    header, payload, offset = {}, [], 0
    for name, (value, dtype) in tensors.items():
        data = value.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.append(data)
    encoded = json.dumps(header).encode()
    # Deliberately unaligned data start: rows straddle 4 KiB blocks.
    encoded += b" " * (-len(encoded) % 8 + 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


@pytest.fixture
def table(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 120, (5000, 64), dtype=np.uint8)
    scales = rng.integers(120, 130, (5000, 2), dtype=np.uint8)
    path = tmp_path / "engram.safetensors"
    _raw_safetensors(path, {"weight": (raw, "F8_E4M3"), "scale": (scales, "F8_E8M0")})
    return path


def _mmap_reference(path, ids, monkeypatch):
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1 << 62)
    embed = DiskEngramEmbedding(path, "weight", "scale")
    try:
        return np.array(embed(mx.array(ids)).astype(mx.float32))
    finally:
        embed.close()
        monkeypatch.undo()


def test_native_gather_matches_mmap_rows(table, monkeypatch):
    if storage._native_gather() is None:
        pytest.skip("native Engram reader unavailable")
    ids = np.random.default_rng(1).integers(0, 5000, (1, 700, 24))
    ids[0, :50] = ids[0, 50:100]  # duplicates are deduplicated then expanded
    expected = _mmap_reference(table, ids, monkeypatch)
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    embed = DiskEngramEmbedding(table, "weight", "scale")
    raw = embed.gather(ids)
    reference = embed._read_rows(ids.reshape(-1))
    for (a, da), (b, db) in zip(raw, reference):
        assert da == db
        np.testing.assert_array_equal(a, b)
    actual = np.array(embed(mx.array(ids)).astype(mx.float32))
    embed.close()
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(RuntimeError, match="closed"):
        embed(mx.array(ids))


def test_lookahead_claim_patches_wrong_rows_and_keeps_later_chunks(table, monkeypatch):
    rng = np.random.default_rng(2)
    first = rng.integers(0, 5000, (1, 300, 24))
    second = rng.integers(0, 5000, (1, 300, 24))
    expected = _mmap_reference(table, first, monkeypatch)
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    embed = DiskEngramEmbedding(table, "weight", "scale")
    prefetch = EngramPrefetch()
    guessed = first.copy()
    guessed[0, :3] = 7  # e.g. a wrong n-gram history guess for the chunk head
    prefetch.submit(embed, guessed, lookahead=True)
    prefetch.submit(embed, second, lookahead=True)
    prefetch.submit(embed, first)  # close enough to the queued guess: not re-read
    assert len(embed._queue.items) == 2
    actual = np.array(embed(mx.array(first)).astype(mx.float32))
    np.testing.assert_array_equal(actual, expected)
    assert len(embed._queue.items) == 1
    np.testing.assert_array_equal(embed._queue.items[0].rows, second[0])
    assert embed._queue.stats["read_at_claim"] == 3 * 24
    prefetch.close()
    embed.close()
    assert not embed._queue.items


def test_claims_slice_ranges_across_windows(table, monkeypatch):
    """A chunk inside one window, or spanning two, is sliced out, not re-read."""
    ids = np.random.default_rng(6).integers(0, 5000, (1, 40, 24))
    expected = _mmap_reference(table, ids, monkeypatch)
    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    embed = DiskEngramEmbedding(table, "weight", "scale")
    prefetch = EngramPrefetch()
    prefetch.submit(embed, ids[:, :17], lookahead=True)
    prefetch.submit(embed, ids[:, 17:], lookahead=True)
    for a, b in [(5, 12), (12, 29), (29, 40), (0, 5)]:
        actual = np.array(embed(mx.array(ids[:, a:b])).astype(mx.float32))
        np.testing.assert_array_equal(actual, expected[:, a:b])
    stats = embed._queue.stats
    assert stats["read_ahead"] == ids.size and stats["read_at_claim"] == 0
    prefetch.close()
    embed.close()


def _window(rows):
    done = Future()
    done.set_result(None)
    return storage._Window(np.asarray(rows).reshape(-1, 24), done)


def test_pending_queue_is_bounded(table):
    embed = DiskEngramEmbedding(table, "weight", "scale")
    for i in range(8):
        embed.add_pending(_window(np.full((1, 1, 24), i)))
    assert [int(w.rows[0, 0]) for w in embed._queue.items] == [2, 3, 4, 5, 6, 7]
    embed.close()


def test_pending_state_is_not_a_module_child(table):
    embed = DiskEngramEmbedding(table, "weight", "scale")
    embed.add_pending(_window(np.zeros((1, 1, 24), np.int64)))
    assert "_queue" not in embed and not embed.children()
    embed.close()


def test_lookahead_evicts_oldest_beyond_byte_cap(table, monkeypatch):
    embed = DiskEngramEmbedding(table, "weight", "scale")
    prefetch = EngramPrefetch()
    row = embed.selected_bytes(1)
    monkeypatch.setattr(storage, "LOOKAHEAD_MAX_BYTES", row * 400)
    rng = np.random.default_rng(4)
    prefetch.submit(embed, rng.integers(0, 5000, (1, 10, 24)), lookahead=True)
    newest = rng.integers(0, 5000, (1, 10, 24))
    prefetch.submit(embed, newest, lookahead=True)
    assert len(embed._queue.items) == 1 and embed.pending_bytes() == row * 240
    np.testing.assert_array_equal(embed._queue.items[0].rows, newest[0])
    prefetch.submit(embed, rng.integers(0, 5000, (1, 20, 24)))  # the forward's own read
    assert len(embed._queue.items) == 2
    prefetch.close()
    embed.close()


def _tiny_engram_pair(tmp_path, monkeypatch):
    from test_deepseek_v41 import load_reference_weights, tiny_ced

    from omlx.patches.deepseek_v41.language import LanguageModel

    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    c = tiny_ced(
        engram_layer_ids=(1,),
        engram_num_embeddings=(72,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    ram, ssd = LanguageModel(c), LanguageModel(c)
    for model in (ram, ssd):
        load_reference_weights(model)
        model.set_token_map(np.arange(64))
    table = ram.layers[1].engram.embed.weight.astype(mx.bfloat16)
    path = tmp_path / "engram.safetensors"
    mx.save_safetensors(str(path), {"embed": table})
    ssd.layers[1].engram.embed = DiskEngramEmbedding(path, "embed", None)
    ram.layers[1].engram.embed.weight = table
    ssd._engram_prefetch = EngramPrefetch()
    return ram, ssd


@pytest.mark.parametrize(
    "announced,actual",
    [
        (9, [6, 7, 5, 9, 4]),  # every chunk shrinks after its announcement
        (8, [8, 3, 8, 1, 2, 9]),  # odd sizes, a 1-token chunk, a longer-than-announced one
        (16, [16, 7, 8]),  # the last chunk is a partial remainder
    ],
)
def test_shrinking_chunk_schedule_reuses_lookahead_bitwise(
    tmp_path, monkeypatch, announced, actual
):
    """The scheduler announces `announced` tokens but runs `actual`: each row
    is read exactly once and the output equals the in-RAM table bit for bit."""
    ram, ssd = _tiny_engram_pair(tmp_path, monkeypatch)
    ids = mx.array(np.random.default_rng(len(actual)).integers(0, 64, (1, sum(actual) + 1)))
    n = sum(actual)
    caches = ram.make_cache(), ssd.make_cache()
    ssd.prefetch_ple(ids[:, : min(announced, n)], ids[:, :0])
    pos = 0
    for m in actual:
        if pos + m < n:
            ahead = min(announced, n - pos - m)
            ssd.prefetch_ple(ids[:, pos + m : pos + m + ahead], ids[:, pos : pos + m])
        expected = ram._omlx_prefill(ids[:, pos : pos + m], cache=caches[0])
        got = ssd._omlx_prefill(ids[:, pos : pos + m], cache=caches[1])
        np.testing.assert_array_equal(np.array(got), np.array(expected))
        pos += m
    token = ids[:, n:]
    for _ in range(2):
        expected = ram(token, cache=caches[0])
        np.testing.assert_array_equal(np.array(ssd(token, cache=caches[1])), np.array(expected))
        token = mx.argmax(expected[:, -1:], -1)
    embed = ssd.layers[1].engram.embed
    heads = 6
    stats = embed._queue.stats
    # Every position (prompt + two decode tokens) is read once, ahead of use;
    # after a chunk shorter than the n-gram history the forward plans the next.
    assert stats["read_ahead"] == stats["claimed"] == (n + 2) * heads
    assert stats["read_at_claim"] == 0
    assert not embed._queue.items
    ssd._engram_prefetch.close()
    embed.close()


def test_scheduler_lookahead_prefill_and_decode_are_bitwise(tmp_path, monkeypatch):
    """CED chunked prefill + decode with SSD rows, native reads and lookahead
    equals the in-memory Engram table bit for bit."""
    from test_deepseek_v41 import load_reference_weights, tiny_ced

    from omlx.patches.deepseek_v41.language import LanguageModel

    monkeypatch.setattr(storage, "NATIVE_MIN_ROWS", 1)
    c = tiny_ced(
        engram_layer_ids=(1,),
        engram_num_embeddings=(72,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    ram, ssd = LanguageModel(c), LanguageModel(c)
    for model in (ram, ssd):
        load_reference_weights(model)
        model.set_token_map(np.arange(64))
    table = ram.layers[1].engram.embed.weight.astype(mx.bfloat16)
    path = tmp_path / "engram.safetensors"
    mx.save_safetensors(str(path), {"embed": table})
    ssd.layers[1].engram.embed = DiskEngramEmbedding(path, "embed", None)
    ram.layers[1].engram.embed.weight = table
    ssd._engram_prefetch = EngramPrefetch()
    ids = mx.array(np.random.default_rng(5).integers(0, 64, (1, 23)))
    caches = ram.make_cache(), ssd.make_cache()
    chunks = [(0, 9), (9, 18), (18, 22)]
    ssd.prefetch_ple(ids[:, :9], ids[:, :0])
    for i, (a, b) in enumerate(chunks):
        if i + 1 < len(chunks):
            ssd.prefetch_ple(ids[:, b : chunks[i + 1][1]], ids[:, a:b])
        expected = ram._omlx_prefill(ids[:, a:b], cache=caches[0])
        actual = ssd._omlx_prefill(ids[:, a:b], cache=caches[1])
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
    token = ids[:, 22:]
    for _ in range(3):
        expected = ram(token, cache=caches[0])
        actual = ssd(token, cache=caches[1])
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
        token = mx.argmax(expected[:, -1:], -1)
    from mlx.utils import tree_flatten

    for x, y in zip(*caches):
        left, right = tree_flatten(x.state), tree_flatten(y.state)
        assert len(left) == len(right)
        for (_, u), (_, v) in zip(left, right):
            np.testing.assert_array_equal(np.array(u), np.array(v))
    assert not ssd.layers[1].engram.embed._queue.items
    ssd._engram_prefetch.close()
    ssd.layers[1].engram.embed.close()


