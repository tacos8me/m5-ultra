"""ref/vision-check.safetensors: the official preprocessing (inference/image_processor.load_image) of the
checkpoint's example image; the engine computes its span rows at start (SPLIT_NV_VISION_CHECK) so they can be
compared offline with the official ViT + aligner (tools/vision_fidelity.py)."""
import sys
from types import SimpleNamespace
import torch
from safetensors.torch import save_file
sys.path.insert(0, "/home/ian/models/DeepSeek-V4.1-Flash-original/inference")
from image_processor import load_image

args = SimpleNamespace(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=1024,
                       vision_min_pixels=544 * 544, vision_max_wh_ratio=None)
out = {}
for i, name in enumerate(sys.argv[1:]):
    patches, vh, vw, nh, nw = load_image({"url": name}, args)
    out[f"patches.{i}"] = patches.contiguous()
    out[f"grid.{i}"] = torch.tensor([vh, vw], dtype=torch.int32)
    print(name, tuple(patches.shape), (vh, vw), (nh, nw), "tokens", nh * (nw + 1) + 2)
save_file(out, "/home/ian/split-nv/ref/vision-check.safetensors")
