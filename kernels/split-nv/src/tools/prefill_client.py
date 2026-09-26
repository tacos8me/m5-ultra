"""POST token ids to the split-nv service, save the returned state, print timing."""
import json, sys, time, urllib.request
ids_path, out_path = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:10051"
tokens = json.load(open(ids_path))
if isinstance(tokens, dict): tokens = tokens["tokens"]
t0 = time.perf_counter()
req = urllib.request.Request(f"{base}/v1/prefill", json.dumps({"tokens": tokens}).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    blob = r.read(); meta = json.loads(r.headers["X-Split-NV-Meta"])
dt = time.perf_counter() - t0
open(out_path, "wb").write(blob)
t = meta["timing"]
print(json.dumps({"tokens": len(tokens), "state_bytes": len(blob), "bytes_per_token": round(len(blob)/len(tokens),1),
    "prefill_s": round(t["prefill_seconds"],3), "prefill_tok_s": round(t["prefill_tok_s"]), "encoder_request_s": round(t["encoder_request_seconds"],3),
    "assemble_s": round(t["assemble_seconds"],3), "client_total_s": round(dt,3), "chunks": len(t["chunks"])}))
