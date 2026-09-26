"""cache + stream + lean + delta_from together: rows [D, N-1) equal the fresh full state; wrong prefix_sha256 refused."""
import hashlib, json, struct, sys, urllib.request
sys.path.insert(0, "/home/ian/split-nv/tools")
from og_client import Session
t = json.load(open("/home/ian/split-nv/ref/ids-131072.json"))
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:10050/v1/cache/clear", b"{}")).read()
s = Session(); s.open(t[:20001], state="none", cache=1); s.close()
D = 25000
pre = hashlib.sha256(struct.pack("<%dI" % D, *t[:D])).hexdigest()
s = Session(); ack, d, man, info = s.open(t[:30001], stream=1, state="lean", cache=1, delta_from=D, prefix_sha256=pre); s.close()
s = Session(); _, f, _, _ = s.open(t[:30001]); s.close()
ok = []
for k, w in (("layer.20.slot.2", 288), ("layer.20.slot.3", 68)):
    ok.append(d[k][1] == [1, 30000 - D, w] and d[k][2] == f[k][2][D * w:])
for k in ("layer.20.slot.1", "tail.hidden", "tail.pre", "tokens"):
    ok.append(d[k][2] == f[k][2])
ok.append(d["layer.8.slot.2"][1] == [1, 0, 288])
print(json.dumps({"resumed": ack.get("resumed_tokens"), "numerics": ack.get("numerics"), "delta_from": man.get("delta_from"),
                  "state": man.get("state"), "bytes": info["bytes"], "checks": ok, "pass": all(ok)}))
try:
    s = Session(); s.open(t[:30001], stream=1, state="lean", delta_from=D, prefix_sha256="0" * 64); print("UNEXPECTED accept")
except RuntimeError as e:
    print(json.dumps({"bad_prefix_sha": str(e)[:120]}))
