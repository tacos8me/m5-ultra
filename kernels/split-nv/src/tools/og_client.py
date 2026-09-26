"""Client for the split-nv step API (STEP_API.md) covering OPEN options (stream, cache) and the step path.

Used by og_gate.py; plain sockets + numpy only, so it runs on the host venv or inside the container.
"""
import hashlib
import json
import math
import socket
import struct
import time

FRAME = struct.Struct("<4sIQ")
STEP_HDR = struct.Struct("<IIH")
STPR_HDR = struct.Struct("<IIHIf")
SIZES = {"U8": 1, "I8": 1, "U16": 2, "I16": 2, "BF16": 2, "F16": 2, "U32": 4, "I32": 4, "F32": 4, "I64": 8, "U64": 8}


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, 4 << 20))
        if k == 0:
            raise ConnectionError("EOF")
        got += k
    return buf


def recv_frame(sock):
    tag, hlen, plen = FRAME.unpack(recv_exact(sock, FRAME.size))
    header = bytes(recv_exact(sock, hlen)) if hlen else b""
    return tag, header, plen


def send_frame(sock, tag, header, payload=b""):
    h = json.dumps(header, separators=(",", ":")).encode() if isinstance(header, dict) else header
    sock.sendall(FRAME.pack(tag, len(h), len(payload)) + h + payload)


def parse_safetensors(blob):
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + n])
    meta = header.pop("__metadata__")
    base = 8 + n
    out = {}
    for name, e in header.items():
        a, b = e["data_offsets"]
        out[name] = (e["dtype"], list(e["shape"]), bytes(blob[base + a:base + b]))
    return out, json.loads(meta["manifest"])


class Session:
    def __init__(self, host="127.0.0.1", port=10052, timeout=900):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self.sid = None
        self.length = None

    def open(self, tokens, state="full", images=None, **options):
        """Returns (ack, tensors {name: (dtype, shape, bytes)} or None, manifest or None, info).
        images: [(start, vit_h, vit_w, bf16 patch bytes)] -> OPEN header images + payload tokens || patches."""
        payload = struct.pack("<%dI" % len(tokens), *tokens)
        digest = hashlib.sha256(payload).hexdigest()
        header = dict(proto=1, identity="", prompt_tokens=len(tokens), token_sha256=digest, state=state, **options)
        if images:
            specs, tail = [], []
            for start, vh, vw, data in images:
                h = hashlib.sha256(struct.pack("<II", vh, vw)); h.update(data)
                specs.append({"start": start, "grid": [vh, vw], "sha256": h.hexdigest()})
                tail.append(data)
            header["images"] = specs
            payload = payload + b"".join(tail)
        t0 = time.perf_counter()
        send_frame(self.sock, b"OPEN", header, payload)
        tag, h, n = recv_frame(self.sock)
        ack = json.loads(h)
        if tag != b"ACK " or not ack.get("ok") or n:
            raise RuntimeError(f"OPEN refused: {tag} {ack}")
        self.sid = int(ack["session"])
        self.length = len(tokens) - 1
        info = {"ack_s": time.perf_counter() - t0}
        if state == "none":
            info["open_s"] = time.perf_counter() - t0
            return ack, None, None, info
        tensors, manifest, total, frames = {}, None, 0, 0
        while True:
            tag, h, n = recv_frame(self.sock)
            frames += 1
            if tag == b"STAT":
                blob = recv_exact(self.sock, n)
                tensors, manifest = parse_safetensors(blob)
                info.update(open_s=time.perf_counter() - t0, bytes=n, frames=frames, mode="stat")
                return ack, tensors, manifest, info
            hd = json.loads(h) if h else {}
            if tag == b"TENS":
                name, dtype, shape, off = hd["name"], hd["dtype"], list(hd["shape"]), hd.get("offset", 0)
                size = math.prod(shape) * SIZES[dtype]
                if name not in tensors:
                    tensors[name] = [dtype, shape, bytearray(size), 0]
                    total += size
                e = tensors[name]
                if [dtype, shape] != e[:2] or off != e[3] or off + n > size:
                    raise ValueError(f"bad TENS part {hd} have {e[3]}")
                e[2][off:off + n] = recv_exact(self.sock, n)
                e[3] += n
                info.setdefault("first_part_s", time.perf_counter() - t0)
            elif tag == b"PROG":
                info["last_prog"] = hd
            elif tag == b"MANI":
                manifest = hd
            elif tag == b"END ":
                if manifest is None or any(len(e[2]) != e[3] for e in tensors.values()):
                    raise ValueError("incomplete streamed state")
                if hd.get("bytes") != total or manifest.get("bytes") != total:
                    raise ValueError(f"byte count mismatch end={hd.get('bytes')} manifest={manifest.get('bytes')} got={total}")
                info.update(open_s=time.perf_counter() - t0, bytes=total, frames=frames, mode="stream", end=hd)
                return ack, {k: (e[0], e[1], bytes(e[2])) for k, e in tensors.items()}, manifest, info
            elif tag == b"ERR ":
                raise RuntimeError(f"ERR {hd}")
            else:
                raise RuntimeError(f"unexpected frame {tag} {hd}")

    def step(self, keep, ids):
        t0 = time.perf_counter()
        self.sock.sendall(FRAME.pack(b"STEP", STEP_HDR.size, 4 * len(ids)) + STEP_HDR.pack(self.sid, keep, len(ids))
                          + struct.pack("<%dI" % len(ids), *ids))
        tag, h, n = recv_frame(self.sock)
        if tag != b"STPR":
            raise RuntimeError(f"STEP failed: {tag} {h}")
        sid, length, L, nbytes, box_s = STPR_HDR.unpack(h)
        payload = bytes(recv_exact(self.sock, n))
        assert sid == self.sid and L == len(ids) and nbytes == n == len(payload), (sid, L, nbytes, n)
        self.length = length
        return payload, {"box_s": box_s, "rtt_s": time.perf_counter() - t0}

    def close(self, **fields):
        try:
            send_frame(self.sock, b"CLOS", dict(session=self.sid, **fields))
        finally:
            self.sock.close()


def step_script(tokens, n):
    """Deterministic steps after OPEN(tokens[:n]): accepts, rejections (rollback) and a final keep-all."""
    t = tokens
    b = n - 1
    return [
        (b, t[b:b + 5]),
        (b + 2, t[b + 2:b + 5]),
        (b + 5, [t[b + 5], 11, 22]),
        (b + 6, t[b + 6:b + 10]),
        (b + 10, t[b + 10:b + 11]),
        (b + 11, t[b + 11:b + 13]),
    ]
