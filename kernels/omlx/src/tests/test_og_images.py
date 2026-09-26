"""CPU: og image payloads/keys match the box (test vector), the processor's layout builds, wire specs."""
import json, sys
import numpy as np
import mlx.core as mx
from PIL import Image
sys.path.insert(0, ".")
from omlx.patches.deepseek_v41 import og_images as G
from omlx.patches.deepseek_v41.processing import Processor
from omlx.patches.deepseek_v41.config import ModelConfig

d = G.image_digest(6, 9, bytes(range(256)) * 4)
assert d == "ca566a533a9e8cea032edbb18b02c205173b3fa69dd450129e9babf366abad35", d
assert [int(x) for x in G.span_keys(d, 3)] == [16016058611342194163, 15360604927821645464, 15651555166149315402]
raw = json.load(open("/Users/ian/models/DeepSeek-V4.1-Flash-pipe1-mlx/config.json"))
cfg = ModelConfig.from_dict(raw)
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("/Users/ian/models/DeepSeek-V4.1-Flash-pipe1-mlx")
proc = Processor(tok, cfg)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many carrots are there?"}]}]
from omlx.patches.deepseek_v41.processing import format_messages
out, _ = format_messages(msgs, 1)
prompt = proc.apply_chat_template(out, tokenize=False)
res = proc(text=prompt, images=[Image.open("/tmp/carrots.jpeg")])
ids = res["input_ids"]
imgs = G.build(ids, res["pixel_values"], cfg.image_token_id, res["image_grids"], res["image_spans"], res["image_types"])
im = imgs[0]
print(json.dumps({"tokens": int(ids.shape[1]), "image_start": im.start, "length": im.length, "grid": [im.vit_h, im.vit_w],
                  "bytes": len(im.data), "digest": im.digest}))
keys = G.prompt_keys(np.asarray(ids).reshape(-1)[:-1].tolist(), imgs)
assert all(int(k) >> 63 for k in keys[im.start:im.start + im.length]) and int(keys[0]) == int(np.asarray(ids)[0, 0])
specs, tail = G.wire(imgs)
assert len(tail) == im.vit_h * im.vit_w * 588 * 2 and specs[0]["sha256"] == im.digest
ph = mx.zeros((1, 3, 1)); G.REGISTRY.put(ph, imgs); assert G.REGISTRY.take(ph) == imgs and G.REGISTRY.take(ph) is None
print("og_images ok")
