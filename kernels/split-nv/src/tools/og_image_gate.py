"""Box-side image gates (live engine; byte-exact state tensors + STEP payloads):
  determinism   the same image prompt twice (fresh) -> identical
  resume        cache OPEN of p1 (text + image + text), then p2 = p1 + text: resumes at len(p1)-1, == fresh p2
  other-image   same tokens, different image: never resumes past the image start, == its own fresh state, != p2
  mid-span      image across the 8K grid (grid entry inside the span); a longer prompt resumes there, == fresh
  errors        image tokens without images; wrong sha256
Images: ref/vision-check.safetensors (official preprocessing of the checkpoint's example images)."""
import hashlib, json, os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from og_client import Session, step_script  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

IMG = 129264
ref = load_file("/home/ian/split-nv/ref/vision-check.safetensors")
PICS = []
for i in range(2):
    vh, vw = (int(x) for x in ref[f"grid.{i}"])
    PICS.append((vh, vw, ref[f"patches.{i}"].contiguous().view(torch.int16).numpy().tobytes()))
text = json.load(open("/home/ian/split-nv/ref/ids-131072.json"))


def span(vh, vw):
    nh, nw = -(-vh // 3), -(-vw // 3)
    return nh * (nw + 1) + 2


def prompt(pre, pic, post, off=0):
    vh, vw, data = PICS[pic]
    L = span(vh, vw)
    toks = text[:pre] + [IMG] * L + text[50000 + off:50000 + off + post]
    return toks, [(pre, vh, vw, data)]


def run(toks, imgs, **opts):
    s = Session()
    ack, t, m, info = s.open(toks, images=imgs, **opts)
    filler = text[90000:90040]
    steps = [hashlib.sha256(s.step(k, ids)[0]).hexdigest() for k, ids in step_script(toks + filler, len(toks))]
    s.close()
    return ack, {k: hashlib.sha256(v[2]).hexdigest() for k, v in t.items()}, steps, info, m


def same(a, b):
    return a[1] == b[1] and a[2] == b[2]


results = []


def gate(name, ok, **kw):
    results.append(dict(gate=name, ok=bool(ok), **kw))
    print(json.dumps(results[-1]), flush=True)


urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:10050/v1/cache/clear", b"{}"), timeout=60).read() \
    if "--clear" in sys.argv else None
nonce = [int(x) for x in np.random.default_rng(int.from_bytes(os.urandom(4), "little")).integers(1000, 100000, 8)]
text = text[:1] + nonce + text[1:]   # fresh keys every run: no hits from earlier runs

p1, i1 = prompt(3000, 0, 300)
a = run(p1, i1)
b = run(p1, i1)
gate("determinism", same(a, b), open_s=round(a[3]["open_s"], 3), images=a[4].get("images"))
p2, _ = prompt(3000, 0, 1300)
s = Session(); s.open(p1, state="none", images=i1, cache=1); s.close()
r = run(p2, i1, cache=1)
f = run(p2, i1)
gate("resume", same(r, f) and r[0].get("resumed_tokens") == len(p1) - 1, resumed=r[0].get("resumed_tokens"),
     cached_open_s=round(r[3]["open_s"], 3), fresh_open_s=round(f[3]["open_s"], 3))
# same layout, other image content: patches of image 0 with one element perturbed
vh, vw, data = PICS[0]
d2 = bytearray(data); d2[1000] ^= 0x40
o = run(p2, [(3000, vh, vw, bytes(d2))], cache=1)
of = run(p2, [(3000, vh, vw, bytes(d2))])
gate("other-image", same(o, of) and (o[0].get("resumed_tokens") or 0) <= 3000 and o[1] != f[1],
     resumed=o[0].get("resumed_tokens"))
# image across the 8K grid point: a grid entry at 8192 lands inside the span
p3, i3 = prompt(8000, 0, 700, off=5000)
s = Session(); s.open(p3, state="none", images=i3, cache=1); s.close()
p4, i4 = prompt(8000, 0, 9000, off=5000)
p4 = p4[:8000 + span(vh, vw) + 700] + text[70000:70000 + 400]
m4 = run(p4, i4, cache=1)
m4f = run(p4, i4)
gate("resume-after-image", same(m4, m4f) and (m4[0].get("resumed_tokens") or 0) >= 8192, resumed=m4[0].get("resumed_tokens"))
p5 = text[:8100] + [IMG] * span(vh, vw) + text[60000:60400]
i5 = [(8100, vh, vw, data)]
s = Session(); s.open(p5, state="none", images=i5, cache=1); s.close()
p6 = text[:8100] + [IMG] * span(vh, vw) + text[61000:61400]   # same text + image, other continuation
g6 = run(p6, i5, cache=1)
g6f = run(p6, i5)
gate("mid-span-grid", same(g6, g6f) and g6[0].get("resumed_tokens") == 8192, resumed=g6[0].get("resumed_tokens"),
     note="the only matching entry is the 8192 grid point inside the span 8100..8544: the delta starts mid-image")
for label, toks, imgs in (("image tokens without images", p1, None),
                          ("wrong sha256", p1, [(3000, vh, vw, data[:-2] + b"\x00\x00")])):
    try:
        s = Session()
        if imgs and label == "wrong sha256":
            import struct
            payload = struct.pack("<%dI" % len(toks), *toks)
            hdr = dict(proto=1, identity="", prompt_tokens=len(toks), token_sha256=hashlib.sha256(payload).hexdigest(),
                       state="none", images=[{"start": 3000, "grid": [vh, vw], "sha256": "0" * 64}])
            from og_client import send_frame, recv_frame
            send_frame(s.sock, b"OPEN", hdr, payload + imgs[0][3])
            tag, h, n = recv_frame(s.sock)
            gate(label, tag == b"ERR ", reply=h.decode()[:120])
        else:
            s.open(toks, state="none", images=imgs)
            gate(label, False, reply="accepted")
    except RuntimeError as e:
        gate(label, "ERR" in str(e), reply=str(e)[:120])
print(json.dumps({"passed": sum(r["ok"] for r in results), "of": len(results)}))
sys.exit(0 if all(r["ok"] for r in results) else 1)
