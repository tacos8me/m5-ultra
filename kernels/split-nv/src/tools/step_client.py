"""Client for the split-nv step API (STEP_API.md): validation and latency modes.

  step_client.py validate ref/ids-8192.json [--host H] [--port 10052]
      opens a session on tokens[:N+1] (prefix N), STEPs rows N..N+L-1 and compares h19/pre/ckv20/idxk20
      against a session opened on tokens[:N+L+1] (the prefill path), with a rollback case and a page-boundary case.
  step_client.py latency ref/ids-131072.json [--reps 20]
      opens a session on the prompt, then times STEP L=1..5 (each step rolls back to the prompt length).
"""
import argparse
import json
import socket
import struct
import sys
import time

import numpy as np

try:
    import torch
except ImportError:  # latency mode with state="none" needs no torch (Mac side)
    torch = None

FRAME = struct.Struct("<4sIQ")
STEP_HDR = struct.Struct("<IIH")
STPR_HDR = struct.Struct("<IIHIf")
H_ROW, PRE_ROW, CKV_ROW, IDXK_ROW = 4 * 5120 * 2, 4 * 4, 288, 68


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("EOF")
        got += k
    return bytes(buf)


def recv_frame(sock):
    tag, hlen, plen = FRAME.unpack(recv_exact(sock, FRAME.size))
    header = recv_exact(sock, hlen) if hlen else b""
    payload = recv_exact(sock, plen) if plen else b""
    return tag, header, payload


def send_frame(sock, tag, header, payload=b""):
    h = json.dumps(header, separators=(",", ":")).encode() if isinstance(header, dict) else header
    sock.sendall(FRAME.pack(tag, len(h), len(payload)) + h + payload)


def parse_safetensors(blob):
    n = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + n])
    base = 8 + n
    out = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        a, b = item["data_offsets"]
        raw = blob[base + a:base + b]
        dt = {"U8": torch.uint8, "I32": torch.int32, "I64": torch.int64, "F32": torch.float32, "BF16": torch.bfloat16, "U32": torch.int32}[item["dtype"]]
        wire_dtype = torch.int16 if dt == torch.bfloat16 else dt
        t = (torch.frombuffer(bytearray(raw), dtype=wire_dtype) if raw
             else torch.empty(0, dtype=wire_dtype))
        if dt == torch.bfloat16:
            t = t.view(torch.bfloat16)
        out[name] = t.reshape(item["shape"])
    return out, json.loads(header["__metadata__"]["manifest"])


class Session:
    def __init__(self, host, port, tokens, state="full"):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t0 = time.perf_counter()
        send_frame(self.sock, b"OPEN", {"proto": 1, "identity": "", "prompt_tokens": len(tokens), "state": state},
                   struct.pack("<%dI" % len(tokens), *tokens))
        tag, header, _ = recv_frame(self.sock)
        ack = json.loads(header)
        if tag != b"ACK " or not ack.get("ok"):
            raise RuntimeError(f"OPEN failed: {tag} {ack}")
        self.sid = ack["session"]
        self.state = self.manifest = None
        if state != "none":
            tag, header, payload = recv_frame(self.sock)
            if tag != b"STAT":
                raise RuntimeError(f"expected STAT, got {tag} {header}")
            self.state, self.manifest = parse_safetensors(payload)
        self.open_seconds = time.perf_counter() - t0
        self.length = len(tokens) - 1

    def step(self, keep, ids):
        t0 = time.perf_counter()
        self.sock.sendall(FRAME.pack(b"STEP", STEP_HDR.size, 4 * len(ids)) + STEP_HDR.pack(self.sid, keep, len(ids)) + struct.pack("<%dI" % len(ids), *ids))
        tag, header, payload = recv_frame(self.sock)
        rtt = time.perf_counter() - t0
        if tag != b"STPR":
            raise RuntimeError(f"STEP failed: {tag} {header}")
        sid, length, L, nbytes, box_s = STPR_HDR.unpack(header)
        assert sid == self.sid and L == len(ids) and nbytes == len(payload)
        if torch is None:
            self.length = length
            return {"box_s": box_s, "rtt_s": rtt, "L": L}
        o = 0
        h = torch.frombuffer(bytearray(payload[o:o + L * H_ROW]), dtype=torch.int16).view(torch.bfloat16).reshape(L, 4, 5120); o += L * H_ROW
        pre = torch.frombuffer(bytearray(payload[o:o + L * PRE_ROW]), dtype=torch.float32).reshape(L, 4); o += L * PRE_ROW
        ckv = torch.frombuffer(bytearray(payload[o:o + L * CKV_ROW]), dtype=torch.uint8).reshape(L, CKV_ROW); o += L * CKV_ROW
        idxk = torch.frombuffer(bytearray(payload[o:o + L * IDXK_ROW]), dtype=torch.uint8).reshape(L, IDXK_ROW)
        self.length = length
        return {"h": h, "pre": pre, "ckv": ckv, "idxk": idxk, "box_s": box_s, "rtt_s": rtt}

    def close(self):
        try:
            send_frame(self.sock, b"CLOS", {"session": self.sid})
        finally:
            self.sock.close()


def stats(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (torch.nn.functional.cosine_similarity(a, b, dim=0).item(), (a - b).abs().max().item(),
            ((a - b).norm() / b.norm().clamp_min(1e-12)).item())


def compare(name, out, ref_rows, ref_state, first_pos):
    """out: step result for rows [first_pos, first_pos+L); ref_state: prefill state of a longer prompt."""
    sys.path.insert(0, "/home/ian/split-nv/hooks")
    from split_nv.macpack import unpack_activation
    L = out["h"].shape[0]
    tail0 = ref_state_manifest_first(ref_state)
    hs = ref_state["tail.hidden"][0, first_pos - tail0:first_pos - tail0 + L]
    ps = ref_state["tail.pre"][0, first_pos - tail0:first_pos - tail0 + L]
    ck = ref_state["layer.20.slot.2"][0, first_pos:first_pos + L]
    ik = ref_state["layer.20.slot.3"][0, first_pos:first_pos + L]
    rows = []
    metrics = {}
    for label, a, b, fmt in (("h19", out["h"], hs, None), ("pre", out["pre"], ps, None), ("ckv20", out["ckv"], ck, (4, 16, True)), ("idxk20", out["idxk"], ik, (4, 32, False))):
        if fmt:
            same = (a == b).float().mean().item()
            a, b = unpack_activation(a, *fmt), unpack_activation(b, *fmt)
        else:
            same = None
        cos, mx, rel = stats(a, b)
        metrics[label] = dict(cosine=cos, max_abs=mx, rel_rms=rel, bytes_equal=same)
        rows.append(f"{label}: cos {cos:.6f} max|d| {mx:.4f} relRMS {rel:.5f}" + (f" bytes_equal {same:.4f}" if same is not None else ""))
    print(f"[{name}] rows {first_pos}..{first_pos + L - 1} box {out['box_s'] * 1e3:.1f} ms rtt {out['rtt_s'] * 1e3:.1f} ms\n    " + "\n    ".join(rows))
    print(json.dumps({'validation': name, 'rows': L, 'first_position': first_pos, 'metrics': metrics}))
    return metrics


_MANIFEST = {}


def ref_state_manifest_first(state):
    return _MANIFEST[id(state)]


def validate(args):
    tokens = json.load(open(args.ids))
    N = args.n if args.n else 8190  # crosses a 256-page boundary at 8192
    L = 5
    ref = Session(args.host, args.port, tokens[:N + L + 6 + 1])  # prefix N+L+6 rows: enough for the rollback case
    _MANIFEST[id(ref.state)] = ref.manifest["tail"]["first_position"]
    print(f"reference session: prefix {ref.length}, open {ref.open_seconds:.2f}s, tail first pos {ref.manifest['tail']['first_position']}")
    s = Session(args.host, args.port, tokens[:N + 1], state="none")
    print(f"test session: prefix {s.length}, open {s.open_seconds:.2f}s")
    results = []
    for k in (1, 2, 3, 4, 5):
        out = s.step(N, tokens[N:N + k])
        results.append(compare(f"L={k} from {N}", out, None, ref.state, N))
    # rollback: 5 rows of wrong tokens, then keep 2 of them? no: keep=N+2 means rows N,N+1 accepted (they were correct), 3 rejected
    s.step(N, tokens[N:N + 2] + [11, 22, 33])
    out = s.step(N + 2, tokens[N + 2:N + 2 + 4])
    results.append(compare("rollback 5->2 then L=4", out, None, ref.state, N + 2))
    # continue from the accepted rows (odd/even boundary for the ratio-2 pair tail)
    out = s.step(N + 5, tokens[N + 5:N + 5 + 1])
    results.append(compare("keep-all then L=1", out, None, ref.state, N + 5))
    s.close()
    ref.close()
    failed = [(i, k, v['rel_rms']) for i, r in enumerate(results) for k, v in r.items()
              if not np.isfinite(v['rel_rms']) or v['rel_rms'] > 1e-3]
    if failed:
        raise AssertionError(f'step/prefill relative RMS exceeds 1e-3: {failed}')
    print(json.dumps({'numerical_gate': 'PASS', 'prefix': N, 'cases': len(results)}))


def latency(args):
    tokens = json.load(open(args.ids))
    s = Session(args.host, args.port, tokens, state=args.state)
    N = s.length
    print(f"session prefix {N}, open {s.open_seconds:.2f}s")
    res = {}
    samples = {}
    for L in (1, 2, 3, 4, 5):
        box, rtt = [], []
        for _ in range(args.reps):
            out = s.step(N, [t % 100000 + 1000 for t in tokens[-L:]])
            box.append(out["box_s"])
            rtt.append(out["rtt_s"])
        box, rtt = np.array(box[2:]) * 1e3, np.array(rtt[2:]) * 1e3
        res[L] = (float(np.median(box)), float(np.median(rtt)))
        samples[L] = {'box_ms': box.tolist(), 'rtt_ms': rtt.tolist()}
        print(f"context {N:>8} L={L}: box median {np.median(box):6.1f} ms (min {box.min():5.1f})  client rtt median {np.median(rtt):6.1f} ms")
    s.close()
    print(json.dumps({"context": N, "host": args.host, "warmup_steps_per_width": 2,
                     "steps": {L: {"box_ms": b, "rtt_ms": r} for L, (b, r) in res.items()}, "samples": samples}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "latency"])
    ap.add_argument("ids")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10052)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--state", default="none")
    a = ap.parse_args()
    (validate if a.mode == "validate" else latency)(a)
