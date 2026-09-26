"""Unpack a Mac v1 state's packed slots with the torch port and repack them: bytes must match."""
import json
import sys

sys.path.insert(0, "/home/ian/split-nv/hooks")
import torch
from safetensors import safe_open
from split_nv.macpack import pack_activation, unpack_activation

path = sys.argv[1]
with safe_open(path, "pt") as f:
    manifest = json.loads(f.metadata()["manifest"])
    print("format", manifest["format"], "prompt_tokens", manifest["prompt_tokens"], "bytes", manifest["bytes"])
    layers = manifest["layers"]
    for i in [0, 1, 2, 8, 14, 19, 20, 21, 39]:
        d = layers[i]
        names = d["slots"]
        row = [f"L{i} r{d['compress_ratio']}"]
        for j, n in enumerate(names):
            if n is None:
                row.append(f"s{j}=None")
                continue
            t = f.get_tensor(n)
            row.append(f"s{j}={tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}")
        print("  ".join(row))
    bad = 0
    for i in range(40):
        for j, (bits, g, e4) in {1: (8, 32, False), 2: (4, 16, True), 3: (4, 32, False)}.items():
            n = layers[i]["slots"][j]
            if n is None:
                continue
            packed = f.get_tensor(n)
            if packed.numel() == 0:
                continue
            values = unpack_activation(packed, bits, g, e4)
            repacked = pack_activation(values, bits, g, e4)
            if not torch.equal(repacked, packed):
                nbad = (repacked != packed).sum().item()
                print(f"MISMATCH layer {i} slot {j}: {nbad}/{packed.numel()} bytes differ")
                bad += 1
    print("roundtrip", "OK" if bad == 0 else f"FAILED ({bad} tensors)")
    hist = f.get_tensor(layers[0]["slots"][6])
    print("layer0 slot6 history", hist.tolist(), "slot0", f.get_tensor(layers[0]["slots"][0]).tolist())
    print("prime", manifest["prime"]["expected_target_offset"], [s["offset"] for s in manifest["prime"]["stages"]])
