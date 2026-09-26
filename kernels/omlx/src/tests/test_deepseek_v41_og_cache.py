"""ds41-og prefix reuse, Mac side: row store, OPEN flags, full/lean/delta import (CPU only)."""

import hashlib
import json
import socket
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from omlx.patches.deepseek_v41 import og_cache, pipe_decoder, pipe_wire  # noqa: E402
from omlx.patches.deepseek_v41.pipe_decoder import DecoderHalf  # noqa: E402

IDENTITY = pipe_wire.IDENTITY
RATIOS = [0, 0] + [2] * 18 + [1] + [0] * 19
SOURCES = (2, 8, 14, 20)


def config():
    return SimpleNamespace(n_layers=40, dim=5120, hc_mult=4, window_size=128, head_dim=512, index_head_dim=128,
                           compress_ratios=RATIOS, kv_source_layers=SOURCES)


def serialize(arrays, manifest):
    header, chunks, pos = {"__metadata__": {"manifest": json.dumps(manifest)}}, [], 0
    kinds = {np.dtype("uint32"): "U32", np.dtype("int32"): "I32", np.dtype("int64"): "I64", np.dtype("uint8"): "U8",
             np.dtype("float32"): "F32", np.dtype("uint16"): "BF16"}
    for name, value in arrays.items():
        data = value.tobytes()
        header[name] = {"dtype": kinds[value.dtype], "shape": list(value.shape), "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
        chunks.append(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    manifest["bytes"] = pos
    header["__metadata__"]["manifest"] = json.dumps(manifest)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    return b"".join([struct.pack("<Q", len(raw)), raw, *chunks])


def state_blob(tokens, *, delta_from=0, lean=False, seed=0, numerics=None, omit_empty_rows=False):
    """A synthetic ds41-encoder-state-v1 with the box's layout (state_pack.assemble)."""
    rng = np.random.default_rng(seed)
    n1 = len(tokens) - 1
    kv20 = np.random.default_rng(99).integers(0, 256, (1, n1, 288), dtype=np.uint8)
    idx20 = np.random.default_rng(98).integers(0, 256, (1, n1, 68), dtype=np.uint8)
    arrays = {"tokens": np.asarray(tokens, np.uint32)}
    layers = []
    for i in range(21):
        ratio = {2: 2, 8: 2, 14: 2, 20: 1}.get(i, 0)
        slots = []

        def put(j, value):
            arrays[f"layer.{i}.slot.{j}"] = value
            slots.append(f"layer.{i}.slot.{j}")

        put(0, np.array([n1], np.int32))
        if lean and i < 20:
            layers.append({"compress_ratio": 0, "slots": slots, "left_padding": None, "lengths": None})
            continue
        window = np.random.default_rng(95) if i == 20 else rng
        put(1, window.integers(0, 256, (1, min(n1, 128), 528), dtype=np.uint8))
        if i == 20 and omit_empty_rows and delta_from == n1:
            # A streamed OPEN sends no TENS part for a zero-row tensor.
            slots += [f"layer.20.slot.2", f"layer.20.slot.3"]
        elif i == 20:
            put(2, kv20[:, delta_from:])
            put(3, idx20[:, delta_from:])
        elif ratio:
            put(2, rng.integers(0, 256, (1, n1 // ratio, 288), dtype=np.uint8))
            put(3, rng.integers(0, 256, (1, n1 // ratio, 68), dtype=np.uint8))
        else:
            put(2, np.zeros((1, 0, 288), np.uint8))
            put(3, np.zeros((1, 0, 68), np.uint8))
        put(4, np.zeros((1, 0, 512), np.uint16))
        put(5, np.zeros((1, 0, 512), np.uint16))
        put(6, np.array([[5, 6, 7]], np.int64) if i == 0 else np.zeros((1, 0), np.int64))
        layers.append({"compress_ratio": ratio, "slots": slots, "left_padding": None, "lengths": None})
    rows = min(n1, 256)
    arrays["tail.hidden"] = np.random.default_rng(97).integers(0, 2**15, (1, rows, 4, 5120), dtype=np.uint16)
    arrays["tail.pre"] = np.random.default_rng(96).random((1, rows, 4), dtype=np.float32)
    digest = hashlib.sha256(struct.pack("<%dI" % len(tokens), *tokens)).hexdigest()
    manifest = dict(format="ds41-encoder-state-v1", identity=IDENTITY, token_sha256=digest, prompt_tokens=len(tokens),
                    layers=layers, encoder_layers=21,
                    tail={"first_position": n1 - rows, "rows": rows, "hidden": "tail.hidden", "pre": "tail.pre",
                          "layer": 20})
    if delta_from:
        manifest["delta_from"] = delta_from
    if numerics:
        manifest["numerics"] = numerics
    return serialize(arrays, manifest)


@pytest.fixture
def replays(monkeypatch):
    calls = []

    def fake_replay(lm, cache, hidden, pre, first, tokens):
        calls.append((np.array(hidden.view(mx.uint16)), np.array(pre), first, len(tokens)))
        return cache

    monkeypatch.setattr(pipe_decoder, "replay", fake_replay)
    return calls


def host(value):
    return np.array(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)


def imported(blob, tokens, base_rows=None):
    tensors, manifest = pipe_wire.parse_state_blob(blob)
    fake = SimpleNamespace(_config=config())
    return DecoderHalf.import_state(fake, tensors, manifest, tokens, identity=IDENTITY, base_rows=base_rows)


def test_full_lean_and_delta_imports_are_identical(replays):
    tokens = list(range(1000, 1600))
    full, rows = imported(state_blob(tokens), tokens)
    lean, lean_rows = imported(state_blob(tokens, lean=True), tokens)
    delta, delta_rows = imported(state_blob(tokens, delta_from=321, lean=True), tokens, base_rows=rows)
    whole, whole_rows = imported(state_blob(tokens, delta_from=len(tokens) - 1, lean=True), tokens, base_rows=rows)
    repeat, repeat_rows = imported(state_blob(tokens, delta_from=len(tokens) - 1, lean=True, omit_empty_rows=True),
                                   tokens, base_rows=rows)
    for other, other_rows in ((lean, lean_rows), (delta, delta_rows), (whole, whole_rows), (repeat, repeat_rows)):
        for slot in range(7):
            assert np.array_equal(host(full[20][slot]), host(other[20][slot]))
        for a, b in zip(rows, other_rows):
            assert np.array_equal(host(a), host(b))
        for i in range(20):
            assert other[i].size() == len(tokens) - 1 and other[i][2].shape[1] == 0
            assert other[i].compress_ratio == full[i].compress_ratio == (2 if i in (2, 8, 14) else 0)
        assert all(other[i].size() == len(tokens) - 1 for i in range(21, 40))
    assert len(replays) == 5
    for call in replays[1:]:
        assert np.array_equal(call[0], replays[0][0]) and np.array_equal(call[1], replays[0][1])
        assert call[2:] == replays[0][2:]


def test_delta_needs_matching_base_rows(replays):
    tokens = list(range(1000, 1600))
    _, rows = imported(state_blob(tokens), tokens)
    blob = state_blob(tokens, delta_from=321)
    with pytest.raises(ValueError):
        imported(blob, tokens)
    with pytest.raises(ValueError):
        imported(blob, tokens, base_rows=(rows[0][:, :300], rows[1][:, :300]))
    with pytest.raises(ValueError):
        imported(state_blob(tokens), tokens[:-1] + [7])
    tensors, manifest = pipe_wire.parse_state_blob(state_blob(tokens, delta_from=321))
    del tensors["layer.20.slot.2"]
    with pytest.raises(ValueError):  # rows may be absent only when none are due
        DecoderHalf.import_state(SimpleNamespace(_config=config()), tensors, manifest, tokens,
                                 identity=IDENTITY, base_rows=rows)


def test_row_store_longest_prefix_budget_and_numerics():
    key = ("id", None)
    store = og_cache.RowStore(2 * 1000 * (og_cache.ROW_BYTES + 4))

    def rows(n):
        return mx.zeros((1, n, 288), mx.uint8), mx.zeros((1, n, 68), mx.uint8)

    turn1 = list(range(900))
    assert store.record(turn1, *rows(900), key)
    turn2 = turn1 + [5] * 50
    assert store.record(turn2, *rows(950), key)
    assert store.summary()["tokens"] == [950] and store.stats["dropped_prefixes"] == 1
    assert not store.record(turn1[:500], *rows(500), key)      # covered by a longer entry
    entry, n = store.lookup(turn2[:-3] + [1, 2, 3, 4], key)
    assert entry is not None and n == 947
    assert store.lookup([7] + turn1, key) == (None, 0)
    other = [9] * 900
    assert store.record(other, *rows(900), key)
    assert store.record([8] * 900, *rows(900), key)            # evicts the least recently used entry
    assert len(store.entries) == 2 and store.stats["evictions"] == 1
    assert store.record([6] * 10, *rows(10), ("id", "v2"))       # new numerics drop every older entry
    assert store.summary()["tokens"] == [10]
    assert store.lookup([6] * 10, key) == (None, 0)


def test_open_sends_cache_lean_and_delta_flags():
    tokens = list(range(1000, 1600))
    blob = state_blob(tokens, delta_from=321, lean=True)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    seen = {}

    def serve():
        conn, _ = server.accept()
        tag, hlen, plen = pipe_wire.FRAME.unpack(pipe_wire.receive(conn, 16))
        seen["open"] = (tag, json.loads(bytes(pipe_wire.receive(conn, hlen))), bytes(pipe_wire.receive(conn, plen)))
        pipe_wire.send(conn, b"ACK ", {"ok": True, "session": 7, "resumed_tokens": 512})
        pipe_wire.send(conn, b"STAT", {"format": "ds41-encoder-state-v1", "bytes": len(blob)}, blob)
        tag, hlen, plen = pipe_wire.FRAME.unpack(pipe_wire.receive(conn, 16))
        seen["close"] = (tag, json.loads(bytes(pipe_wire.receive(conn, hlen))))
        conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    session = pipe_wire.EncoderSession("127.0.0.1", server.getsockname()[1])
    tensors, manifest = session.open(tokens, "r1", cache=True, state="lean", delta_from=321)
    session.close()
    thread.join(5)
    tag, header, payload = seen["open"]
    assert tag == b"OPEN" and header["cache"] == 1 and header["state"] == "lean" and header["delta_from"] == 321
    assert header["prefix_sha256"] == hashlib.sha256(payload[:4 * 321]).hexdigest()
    assert session.open_info["resumed_tokens"] == 512 and manifest["delta_from"] == 321
    assert tensors["layer.20.slot.2"][1] == [1, len(tokens) - 1 - 321, 288]
    assert seen["close"] == (b"CLOS", {"session": 7})
    plain = pipe_wire.EncoderSession.__new__(pipe_wire.EncoderSession)
    with pytest.raises(ValueError):
        plain.session = None
        plain.open(tokens, delta_from=len(tokens))


def fake_box(blobs):
    """Serve one OPEN per connection: answers with the next blob; records every OPEN header."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    headers = []

    def serve():
        for blob in blobs:
            conn, _ = server.accept()
            tag, hlen, plen = pipe_wire.FRAME.unpack(pipe_wire.receive(conn, 16))
            headers.append(json.loads(bytes(pipe_wire.receive(conn, hlen))))
            pipe_wire.receive(conn, plen)
            pipe_wire.send(conn, b"ACK ", {"ok": True, "session": len(headers), "resumed_tokens": 0})
            pipe_wire.send(conn, b"STAT", {"format": "ds41-encoder-state-v1", "bytes": len(blob)}, blob)
            try:
                pipe_wire.receive(conn, 16)
            except (ConnectionError, OSError):
                pass
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], headers


def test_job_drops_rows_of_other_numerics_and_reopens_without_delta(monkeypatch):
    from omlx.patches.deepseek_v41 import og_model

    tokens = list(range(1000, 1600))
    key = (IDENTITY, "og-s4.1")
    store = og_cache.RowStore(1 << 30)
    store.record(tokens[:500], mx.zeros((1, 500, 288), mx.uint8), mx.zeros((1, 500, 68), mx.uint8), key)
    monkeypatch.setattr(og_model, "STORE", store)
    monkeypatch.setattr(og_model, "NUMERICS", [key])
    monkeypatch.setattr(og_model, "DELTA_MIN", 100)
    port, headers = fake_box([state_blob(tokens, delta_from=500, lean=True, numerics="og-s4.2"),
                              state_blob(tokens, lean=True, numerics="og-s4.2")])
    before = og_model.STATS["delta_retries"]
    job = og_model.Job("127.0.0.1", port, "r1", tokens)
    job.run()
    assert job.error is None and job.base_rows is None
    assert og_model.STATS["delta_retries"] == before + 1 and not store.entries
    assert headers[0]["delta_from"] == 500 and "delta_from" not in headers[1]
    assert all(h["cache"] == 1 and h["state"] == og_model.STATE for h in headers)
    job.encoder.close()


def test_job_uses_stored_rows_for_a_matching_delta(monkeypatch):
    from omlx.patches.deepseek_v41 import og_model

    tokens = list(range(1000, 1600))
    key = (IDENTITY, "og-s4.2")
    store = og_cache.RowStore(1 << 30)
    kv, idx = mx.ones((1, 500, 288), mx.uint8), mx.ones((1, 500, 68), mx.uint8)
    store.record(tokens[:500], kv, idx, key)
    monkeypatch.setattr(og_model, "STORE", store)
    monkeypatch.setattr(og_model, "NUMERICS", [key])
    monkeypatch.setattr(og_model, "DELTA_MIN", 100)
    port, headers = fake_box([state_blob(tokens, delta_from=500, lean=True, numerics="og-s4.2")])
    job = og_model.Job("127.0.0.1", port, "r2", tokens)
    job.run()
    assert job.error is None and job.base_rows[0] is kv and job.base_rows[1] is idx
    assert len(headers) == 1 and headers[0]["delta_from"] == 500
    job.encoder.close()
