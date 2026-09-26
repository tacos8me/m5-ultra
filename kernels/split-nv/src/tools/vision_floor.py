"""Noise floor of the official ViT + aligner: the same code with SDPA math vs flash/efficient kernels."""
import json, sys
from types import SimpleNamespace
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors import safe_open
from safetensors.torch import load_file
sys.path.insert(0, "/home/ian/models/DeepSeek-V4.1-Flash-original/inference")
import vision as V
from vision import ViT, Aligner
_sdpa = torch.nn.functional.scaled_dot_product_attention
V.F.scaled_dot_product_attention = lambda q, k, v, **kw: _sdpa(q[None], k[None], v[None], **kw)[0]  # 4-D for flash
ORIG = "/home/ian/models/DeepSeek-V4.1-Flash-original"
args = SimpleNamespace(vision_dim=1024, vision_n_heads=16, vision_inter_dim=2816, vision_patch_size=14, vision_rope_theta=10000.0,
                       vision_n_layers=32, vision_downsample_ratio=3, dim=5120)
torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda")
vit, aligner = ViT(args), Aligner(args)
sd = {}
with safe_open(f"{ORIG}/model-00001-of-00048.safetensors", "pt", device="cuda") as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)
vit.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("vision.")}); aligner.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("aligner.")})
ref = load_file("/home/ian/split-nv/ref/vision-check.safetensors")
out = []
with torch.no_grad():
    for i in range(2):
        h, w = (int(x) for x in ref[f"grid.{i}"]); p = ref[f"patches.{i}"].cuda()
        res = {}
        for name, be in (("math", SDPBackend.MATH), ("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(be):
                    res[name] = aligner(vit(p, h, w), h, w).float()
            except RuntimeError as e:
                res[name] = None
        a = res["math"]
        for name in ("flash", "efficient"):
            b = res[name]
            if b is not None:
                out.append({"image": i, "vs": f"math-{name}", "cos": round(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(), 6),
                            "relrms": round(((a - b).norm() / a.norm()).item(), 5)})
print(json.dumps(out))
