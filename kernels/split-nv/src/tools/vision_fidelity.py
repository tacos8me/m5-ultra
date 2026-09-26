"""Official ViT + aligner (checkpoint inference/vision.py, weights from shards 1-2) on ref/vision-check patches,
compared with the engine's span rows (SPLIT_NV_DIR/vision-check-out.safetensors). Run on a free GPU."""
import json, math, sys
from types import SimpleNamespace
import torch
from safetensors import safe_open
from safetensors.torch import load_file
sys.path.insert(0, "/home/ian/models/DeepSeek-V4.1-Flash-original/inference")
from vision import ViT, Aligner

ORIG = "/home/ian/models/DeepSeek-V4.1-Flash-original"
args = SimpleNamespace(vision_dim=1024, vision_n_heads=16, vision_inter_dim=2816, vision_patch_size=14, vision_rope_theta=10000.0,
                       vision_n_layers=32, vision_downsample_ratio=3, dim=5120)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")  # as the official generate.py: its rotary tables are default tensors
vit, aligner = ViT(args).cuda(), Aligner(args).cuda()
sd = {}
for shard in (1, 2):
    with safe_open(f"{ORIG}/model-{shard:05}-of-00048.safetensors", "pt", device="cuda") as f:
        for k in f.keys():
            if k.startswith(("vision.", "aligner.", "image_")):
                sd[k] = f.get_tensor(k)
vit.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("vision.")}, strict=True)
aligner.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("aligner.")}, strict=True)
ref = load_file("/home/ian/split-nv/ref/vision-check.safetensors")
got = load_file(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/split-nv/vision-check-out.safetensors")
res = []
with torch.no_grad():
    for i in range(sum(k.startswith("patches.") for k in ref)):
        h, w = (int(x) for x in ref[f"grid.{i}"])
        feats = aligner(vit(ref[f"patches.{i}"].cuda(), h, w), h, w)
        nh, nw = math.ceil(h / 3), math.ceil(w / 3)
        types = [0] + ([1] * nw + [2]) * nh + [3]
        rows = torch.empty(len(types), 5120, dtype=torch.bfloat16, device="cuda")
        t = torch.tensor(types, device="cuda")
        rows[t == 0], rows[t == 2], rows[t == 3] = sd["image_start"], sd["image_newline"], sd["image_end"]
        rows[t == 1] = feats.to(torch.bfloat16)
        g = got[f"rows.{i}"].cuda()
        a, b = g.float(), rows.float()
        m = t == 1
        res.append({"image": i, "rows": len(types), "delimiters_equal": bool(torch.equal(g[~m], rows[~m])),
                    "features_cos": round(torch.nn.functional.cosine_similarity(a[m].flatten(), b[m].flatten(), dim=0).item(), 6),
                    "features_relrms": round(((a[m] - b[m]).norm() / b[m].norm()).item(), 5),
                    "bitwise_equal": bool(torch.equal(g, rows))})
print(json.dumps(res))
