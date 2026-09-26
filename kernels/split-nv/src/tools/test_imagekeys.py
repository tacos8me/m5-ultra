"""CPU: image spans/keys (box side). The Mac's copy must reproduce KEY_VECTOR exactly."""
import hashlib, json, struct, sys
import numpy as np
sys.path.insert(0, "/home/ian/split-nv/hooks")
from split_nv import imagekeys as K

IMG = 129264
# test vector shared with the Mac implementation
d = K.image_digest(6, 9, bytes(range(256)) * 4)
KEY_VECTOR = [int(x) for x in K.span_keys(d, 3)]
print(json.dumps({"digest": d, "keys": KEY_VECTOR}))

# a prompt with one 6x9-patch image: nh=2, nw=3 -> 2*(3+1)+2 = 10 tokens
L = K.num_image_tokens(6, 9)
assert L == 10
tokens = [0, 5, 6] + [IMG] * L + [7, 8, 9]
patches = np.random.default_rng(0).standard_normal((54, 588)).astype(np.float32)
raw = (patches.view(np.uint32) >> 16).astype(np.uint16).tobytes()  # bf16 bytes
dig = K.image_digest(6, 9, raw)
hdr = {"images": [{"start": 3, "grid": [6, 9], "sha256": dig}]}
imgs = K.parse_images(hdr, tokens, raw, IMG)
assert imgs[0][:5] == (3, 10, 6, 9, dig)
keys = K.prompt_keys(tokens[:-1], [(3, 10, dig)])
assert list(keys[:3]) == [0, 5, 6] and all(int(k) >> 63 for k in keys[3:13]) and list(keys[13:]) == [7, 8]
assert list(K.key_tokens(keys, IMG)) == tokens[:-1]
# a different image with the same layout gives different keys from the first image position on
raw2 = bytearray(raw); raw2[0] ^= 1
k2 = K.prompt_keys(tokens[:-1], [(3, 10, K.image_digest(6, 9, bytes(raw2)))])
assert int(np.flatnonzero(keys != k2)[0]) == 3
# errors
for bad, why in (({"images": [{"start": 3, "grid": [6, 9], "sha256": "0" * 64}]}, "sha"),
                 ({"images": [{"start": 4, "grid": [6, 9]}]}, "span"),
                 ({"images": [{"start": 3, "grid": [9, 9]}]}, "grid")):
    try:
        K.parse_images(bad, tokens, raw, IMG); raise SystemExit(f"accepted bad {why}")
    except ValueError:
        pass
try:
    K.parse_images({"images": []}, tokens, b"", IMG); raise SystemExit("accepted stray image tokens")
except ValueError:
    pass
# text-only prompts: keys are the tokens; delta digest unchanged (uint32 tokens)
t = [1, 2, 3, 4]
assert K.prefix_digest(t, K.prompt_keys(t, []), 3, False) == hashlib.sha256(struct.pack("<3I", 1, 2, 3)).hexdigest()
print("imagekeys ok")
