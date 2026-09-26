"""POST /v1/prefill (10051) returns the same tensors as a step-API OPEN of the same prompt."""
import json, sys, urllib.request
sys.path.insert(0, "/home/ian/split-nv/tools")
from og_client import Session, parse_safetensors
tokens = json.load(open(sys.argv[1]))[:int(sys.argv[2])]
req = urllib.request.Request("http://127.0.0.1:10051/v1/prefill", json.dumps({"tokens": tokens}).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    blob, meta = r.read(), json.loads(r.headers["X-Split-NV-Meta"])
a, _ = parse_safetensors(blob)
s = Session()
_, b, _, _ = s.open(tokens)
s.close()
bad = [k for k in a if a[k] != b.get(k)]
print(json.dumps({"gate": "http /v1/prefill == OPEN STAT", "tensors": len(a), "mismatch": bad, "numerics": meta.get("numerics"), "pass": not bad and set(a) == set(b)}))
sys.exit(1 if bad or set(a) != set(b) else 0)
