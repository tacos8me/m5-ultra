"""Byte-compare two raw encoder dumps (split-nv-raw-v1): every tensor must be identical."""
import sys
import torch
from safetensors.torch import load_file

a, b = (load_file(p) for p in sys.argv[1:3])
assert a.keys() == b.keys(), set(a) ^ set(b)
bad = [k for k in sorted(a) if not torch.equal(a[k].view(torch.uint8) if a[k].dtype != torch.uint8 else a[k],
                                               b[k].view(torch.uint8) if b[k].dtype != torch.uint8 else b[k])]
print({'tensors': len(a), 'different': bad})
sys.exit(1 if bad else 0)
