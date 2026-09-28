"""CPU-only (no Metal): every wo_a in the pipe1 checkpoint -> byte-code eligibility and escape rate.

Reference encoder in numpy, the same rule as omlx/patches/deepseek_v41/woa_compact.encode; the
decoded BF16 bits of every non-escaped value must equal the original bits. Writes JSON lines.
  python benchmarks/og/woa_scan.py [out.jsonl]
"""
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

MODEL = Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx'


def read_bf16_bits(path, key):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[key]
        assert meta['dtype'] == 'BF16', meta
        begin, end = meta['data_offsets']
        f.seek(8 + n + begin)
        return np.frombuffer(f.read(end - begin), np.uint16).reshape(meta['shape'])


def reference(bits):
    exponent = ((bits >> 7) & 255).astype(np.int64)
    eligible = (bits & 15) == 0
    hist = np.bincount(np.where(eligible, exponent, 256).ravel(), minlength=257)
    window = np.convolve(hist[:256], np.ones(15, np.int64), mode='valid')
    base = 1 + int(np.argmax(window[1:241]))
    code = exponent - base
    ok = eligible & (code >= 0) & (code < 15)
    code = np.where(ok, code, 15).astype(np.uint8)
    codes = (code << 3) | ((bits >> 8) & 128).astype(np.uint8) | ((bits >> 4) & 7).astype(np.uint8)
    return codes, base, hist, ok


def decode(codes, base):
    c = codes.astype(np.uint32)
    return (((c & 128) << 8) | (((c & 127) + base * 8) << 4)).astype(np.uint16)


def main():
    out = open(sys.argv[1], 'a') if len(sys.argv) > 1 else None
    mapping = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    keys = sorted((k for k in mapping if k.endswith('attn.wo_a.weight')),
                  key=lambda k: (('mtp' in k), int(k.split('.')[2])))
    for key in keys:
        bits = read_bf16_bits(MODEL/mapping[key], key)
        t = time.perf_counter()
        codes, base, hist, ok = reference(bits)
        seconds = time.perf_counter() - t
        lossless = bool(np.array_equal(decode(codes, base)[ok], bits[ok]))
        exps = np.nonzero(hist[:256])[0]
        rec = dict(key=key, shape=list(bits.shape), low4_clear=float(1 - hist[256] / bits.size),
                   base=base, exponent_min=int(exps.min()), exponent_max=int(exps.max()),
                   escape_rate=float(1 - ok.mean()), escapes=int((~ok).sum()), zeros=int(((bits & 0x7fff) == 0).sum()),
                   rows_with_escape=float(np.mean((~ok).any(axis=1))), lossless=lossless,
                   numpy_encode_s=round(seconds, 3))
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + '\n')


if __name__ == '__main__':
    main()
