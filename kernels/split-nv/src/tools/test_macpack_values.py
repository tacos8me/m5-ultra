import json, sys
sys.path.insert(0, "/home/ian/split-nv/hooks")
import torch
from safetensors import safe_open
from split_nv.macpack import pack_activation, unpack_activation
path = sys.argv[1]
with safe_open(path, "pt") as f:
    layers = json.loads(f.metadata()["manifest"])["layers"]
    worst = 0.0; n = 0
    for i in range(40):
        for j, (bits, g, e4) in {1: (8, 32, False), 2: (4, 16, True), 3: (4, 32, False)}.items():
            name = layers[i]["slots"][j]
            packed = f.get_tensor(name)
            if packed.numel() == 0: continue
            v1 = unpack_activation(packed, bits, g, e4)
            v2 = unpack_activation(pack_activation(v1, bits, g, e4), bits, g, e4)
            d = (v1 - v2).abs().max().item(); worst = max(worst, d); n += 1
            if d != 0: print("VALUE DIFF layer", i, "slot", j, d, "of max", v1.abs().max().item())
    print("tensors", n, "worst abs value diff after repack", worst)
