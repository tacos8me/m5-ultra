"""21-layer encoder view of the FP8 originals for SGLang: layers 0-20, all 384 experts,
embeddings/head, Engram projections extracted to a small shard, Engram tables left in
shards 47/48 so the SGLang host-table loader can build sglang_engram_1.bin from them."""
import argparse
import json
import struct
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dest", default="/home/ian/split-nv/encoder-model")
ap.add_argument("--vision", action="store_true", help="keep the vision tower, aligner, image embeddings and bias_vl")
args = ap.parse_args()

original = Path("/home/ian/models/DeepSeek-V4.1-Flash-original")
dest = Path(args.dest)
dest.mkdir(exist_ok=True)

config = json.loads((original / "config.json").read_text())
tc = config["text_config"]
tc["num_hidden_layers"] = 21
tc["num_nextn_predict_layers"] = 0
tc["dspark_target_layer_ids"] = []
tc["index_source_layer_ids"] = [2, 8, 14, 20]
tc["compress_ratios"] = tc["compress_ratios"][:21]
if not args.vision:
    config["vision_config"]["num_hidden_layers"] = 0
(dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")

small = dest / "engram-small.safetensors"
if not small.exists():
    entries, pieces, pos = {}, [], 0
    for shard in (47, 48):
        with (original / f"model-{shard:05}-of-00048.safetensors").open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            for name, item in header.items():
                if name == "__metadata__" or ".embed." in name:
                    continue
                begin, end = item["data_offsets"]
                size = end - begin
                assert size < 160 * 2**20, (name, size)
                f.seek(8 + n + begin)
                data = f.read(size)
                assert len(data) == size
                entries[name] = {**item, "data_offsets": [pos, pos + size]}
                pos += size
                pieces.append(data)
    raw = json.dumps(entries).encode()
    raw += b" " * ((-len(raw)) % 8)
    tmp = small.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for data in pieces:
            f.write(data)
    tmp.rename(small)
    print("engram-small", pos, "bytes", len(entries), "tensors")

# shards 3..23 = layers 0..20, 2 = embeddings, 43 = head/final norm, 47/48 = Engram tables; 1 (+2) = vision
files = [f"model-{i:05}-of-00048.safetensors" for i in [*([1] if args.vision else []), 2, *range(3, 24), 43, 47, 48]]
for name in files + ["tokenizer.json", "tokenizer_config.json"]:
    target = dest / name
    if not target.exists():
        target.symlink_to(original / name)

index = json.loads((original / "model.safetensors.index.json").read_text())
mapped = {}
for k, v in index["weight_map"].items():
    if v in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"):
        mapped[k] = v if ".embed." in k else "engram-small.safetensors"
    elif v in files:
        if not args.vision and k.startswith(("vision.", "aligner.", "image_")):
            continue
        mapped[k] = v
(dest / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapped}))
layers = sorted({int(k.split(".")[1]) for k in mapped if k.startswith("layers.")})
print("encoder view", dest, "tensors", len(mapped), "layers", layers[0], "..", layers[-1], "count", len(layers))
print("engram keys", sorted(k for k in mapped if "engram" in k and ".embed." in k))
print("vision tensors", sum(k.startswith(("vision.", "aligner.", "image_")) for k in mapped),
      "bias_vl", sum(k.endswith("bias_vl") for k in mapped), "vision layers", config["vision_config"]["num_hidden_layers"])
