"""Compare a split-nv encoder state against a Mac v1 state for the same prompt, layer by layer.

usage: compare_state.py mine.safetensors mac.safetensors
Reports per layer/slot: cosine similarity, max |diff|, relative RMS error of the unpacked values.
"""
import json
import sys

sys.path.insert(0, "/home/ian/split-nv/hooks")
import torch
from safetensors import safe_open
from split_nv.macpack import unpack_activation

SLOT_FMT = {1: (8, 32, False), 2: (4, 16, True), 3: (4, 32, False)}


def stats(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    diff = (a - b).abs()
    rel = (diff.norm() / b.norm().clamp_min(1e-12)).item()
    return cos, diff.max().item(), rel


def main(mine, mac):
    fm, fc = safe_open(mine, "pt"), safe_open(mac, "pt")
    mm, mc = json.loads(fm.metadata()["manifest"]), json.loads(fc.metadata()["manifest"])
    print(f"mine: {mm['format']} tokens={mm['prompt_tokens']} identity={mm['identity']}")
    print(f"mac : {mc['format']} tokens={mc['prompt_tokens']} identity={mc['identity']}")
    assert mm["token_sha256"] == mc["token_sha256"], "different prompts"
    if "timing" in mm:
        t = mm["timing"]
        print(f"mine timing: prefill {t['prefill_seconds']:.2f}s = {t['prefill_tok_s']:.0f} tok/s, request {t.get('encoder_request_seconds', 0):.2f}s")
    worst = {}
    print(f"{'layer':>5} {'slot':>4} {'rows':>7} {'cosine':>9} {'max|d|':>9} {'relRMS':>8} {'|mac|max':>9}")
    for i, (dm, dc) in enumerate(zip(mm["layers"], mc["layers"])):
        assert dm["compress_ratio"] == dc["compress_ratio"], (i, dm["compress_ratio"], dc["compress_ratio"])
        for j in range(7):
            a, b = fm.get_tensor(dm["slots"][j]), fc.get_tensor(dc["slots"][j])
            if j == 0 or j == 6:
                same = torch.equal(a, b)
                if not same:
                    print(f"{i:>5} {j:>4} MISMATCH {a.tolist()} vs {b.tolist()}")
                continue
            if a.shape != b.shape:
                print(f"{i:>5} {j:>4} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            if a.numel() == 0:
                continue
            if j in SLOT_FMT:
                bits, g, e4 = SLOT_FMT[j]
                a, b = unpack_activation(a, bits, g, e4), unpack_activation(b, bits, g, e4)
            cos, mx, rel = stats(a, b)
            worst[j] = min(worst.get(j, 1.0), cos)
            print(f"{i:>5} {j:>4} {a.shape[1]:>7} {cos:>9.5f} {mx:>9.4f} {rel:>8.4f} {b.abs().max().item():>9.3f}")
    print("worst cosine per slot:", {k: round(v, 5) for k, v in sorted(worst.items())})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
