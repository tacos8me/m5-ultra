"""Mini model view for the prefill-perf harness: the production encoder config (21 layers, text-only), but only the
shards of the listed layers + embeddings + head, and vocab cut to VOCAB rows (the harness slices embed/head and cuts
routed experts to NE at load time; see harness.py).  usage: mini_view.py DEST LAYERS NE (e.g. 2 128)"""
import json
import sys
from pathlib import Path

VOCAB = 16384
orig = Path("/home/ian/models/DeepSeek-V4.1-Flash-original")
enc = Path("/home/ian/split-nv-deploy/encoder-model")
dest = Path(sys.argv[1])
layers = [int(x) for x in sys.argv[2].split(",")]
NE = int(sys.argv[3])
dest.mkdir(parents=True, exist_ok=True)
cfg = json.loads((enc / "config.json").read_text())
cfg["text_config"]["vocab_size"] = VOCAB
cfg["text_config"]["n_routed_experts"] = NE
cfg["text_config"]["max_position_embeddings"] = 131072  # rope tables sized for the harness (values per position unchanged)
(dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
wm = json.loads((orig / "model.safetensors.index.json").read_text())["weight_map"]
keep = {}
for k, v in wm.items():
    head = k.split(".")[0]
    if head in ("embed", "head", "norm") or (head == "layers" and int(k.split(".")[1]) in layers):
        if ".engram." in k:
            continue
        keep[k] = v
for f in sorted(set(keep.values())) + ["tokenizer.json", "tokenizer_config.json"]:
    t = dest / f
    if not t.exists():
        t.symlink_to(orig / f)
(dest / "model.safetensors.index.json").write_text(json.dumps({"weight_map": keep}))
print("view", dest, "shards", sorted(set(keep.values())), "tensors", len(keep))
