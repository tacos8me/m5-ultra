"""Model view for the DSpark drafter microbench: the production text-only encoder config with ONE target layer
(so SGLang's loader builds the parallel state, quant config, attention backend and -- what the drafter needs -- the
full-vocab embed_tokens and lm_head exactly as the engine holds them), its routed experts cut to NE. The drafter's
own mtp.* tensors are read by the bench straight from the original checkpoint. No GPU, no copies (symlinks).

usage: bench_view.py DEST [--layer 3] [--ne 8] [--max-pos 131072]
  layer 3: first non-hash, non-source layer (no tid2eid, no capture); its id does not alias drafter stages 0..2.
"""
import argparse
import json
from pathlib import Path

ORIG = Path("/home/ian/models/DeepSeek-V4.1-Flash-original")
ENC = Path("/home/ian/split-nv-deploy/encoder-model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--first", type=int, default=2, help="lowest layer kept (the pool sizer needs a layer with full-token KV)")
    ap.add_argument("--ne", type=int, default=8)
    ap.add_argument("--max-pos", type=int, default=131072)
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ENC / "config.json").read_text())
    tc = cfg["text_config"]
    tc["n_routed_experts"] = a.ne
    tc["max_position_embeddings"] = a.max_pos  # rope tables sized for the bench (values per position unchanged)
    orig_tc = json.loads((ORIG / "config.json").read_text())["text_config"]
    for k in ("dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank", "dspark_n_routed_experts",
              "dspark_num_experts_per_tok"):
        if tc.get(k) != orig_tc.get(k):
            raise SystemExit(f"{k}: encoder view {tc.get(k)} != original {orig_tc.get(k)}")
    (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    wm = json.loads((ORIG / "model.safetensors.index.json").read_text())["weight_map"]
    keep = {}
    for k, v in wm.items():
        head = k.split(".")[0]
        if head in ("embed", "head", "norm") or (head == "layers" and a.first <= int(k.split(".")[1]) <= a.layer):
            if ".engram." in k:
                continue
            keep[k] = v
    for f in sorted(set(keep.values())) + ["tokenizer.json", "tokenizer_config.json"]:
        t = dest / f
        if not t.exists():
            t.symlink_to(ORIG / f)
    (dest / "model.safetensors.index.json").write_text(json.dumps({"weight_map": keep}))
    mtp = sorted({v for k, v in wm.items() if k.startswith("mtp.")})
    print(json.dumps({"view": str(dest), "layer": a.layer, "ne": a.ne, "tensors": len(keep),
                      "shards": sorted(set(keep.values())), "mtp_shards": mtp}))


if __name__ == "__main__":
    main()
