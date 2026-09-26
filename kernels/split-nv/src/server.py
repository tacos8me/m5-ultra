"""split-nv prefill service: token ids in, DS41 encoder prompt state (Mac packed format) out.

POST /v1/prefill  {"tokens": [...], "identity": "...", "save": false}
  -> application/octet-stream safetensors, format ds41-encoder-state-v1 (see PROGRESS.md), header X-Split-NV-Meta.
GET  /health, GET /v1/info
Runs on the vllm box; talks to the SGLang encoder on 127.0.0.1:10050 (run_encoder.sh).
"""

import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent / "hooks"))
from split_nv.state_pack import FORMAT, ENCODER_LAYERS, SOURCE, TokenMap, assemble, prefix_digest  # noqa: E402

ENCODER = os.environ.get("SPLIT_NV_ENCODER", "http://127.0.0.1:10050")
RAW_DIR = Path(os.environ.get("SPLIT_NV_DIR", "/dev/shm/split-nv"))
SAVE_DIR = Path(os.environ.get("SPLIT_NV_SAVE_DIR", "/home/ian/split-nv/states"))
MODEL_DIR = Path("/home/ian/split-nv/encoder-model")
IDENTITY = os.environ.get("SPLIT_NV_IDENTITY", "split-nv:sglang-757e8f35+hooks:fp8-original:enc0-20")

app = FastAPI()
CFG = json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
TOKEN_MAP = None


@app.on_event("startup")
def _startup():
    global TOKEN_MAP
    t0 = time.time()
    TOKEN_MAP = TokenMap(MODEL_DIR, CFG["engram_compressed_vocab_size"])
    print(f"[server] token map ready ({time.time() - t0:.1f}s)", flush=True)


@app.get("/health")
def health():
    try:
        with urllib.request.urlopen(f"{ENCODER}/health", timeout=3) as r:
            ok = r.status == 200
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "encoder": str(e), "token_map": TOKEN_MAP is not None}, status_code=503)
    return {"ok": ok and TOKEN_MAP is not None, "encoder": "up", "token_map": TOKEN_MAP is not None}


@app.get("/v1/info")
def info():
    return {"format": FORMAT, "identity": IDENTITY, "encoder_layers": ENCODER_LAYERS, "source_layers": SOURCE,
            "boundary": "state after prompt tokens[:-1] through layers 0-20 (no decoder replay, no DSpark prime); "
                        "Mac must run layers 21-39 over tail.hidden/tail.pre rows and prime DSpark",
            "max_tokens": 1048576, "chunk": 8192}


@app.post("/v1/prefill")
async def prefill(request: Request):
    body = await request.json()
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < 2:
        raise HTTPException(400, "tokens: list of >= 2 ints")
    identity = body.get("identity") or IDENTITY
    ids = [int(t) for t in tokens[:-1]]
    digest = prefix_digest(ids)
    raw_path = RAW_DIR / f"raw-{digest}.safetensors"
    if raw_path.exists():
        raw_path.unlink()
    t0 = time.perf_counter()
    payload = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 2, "temperature": 0, "ignore_eos": True}}).encode()
    req = urllib.request.Request(f"{ENCODER}/generate", payload, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            gen = json.load(r)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"encoder failed: {e}")
    t_gen = time.perf_counter() - t0
    deadline = time.time() + 120
    while not raw_path.exists():
        if time.time() > deadline:
            raise HTTPException(500, "encoder produced no state dump")
        time.sleep(0.02)
    t1 = time.perf_counter()
    timing = {"encoder_request_seconds": t_gen, "sglang_meta": gen.get("meta_info")}
    blob, manifest = assemble(raw_path, [int(t) for t in tokens], identity, TOKEN_MAP, timing)
    raw_path.unlink()
    manifest["timing"]["assemble_seconds"] = time.perf_counter() - t1
    manifest["timing"]["total_seconds"] = time.perf_counter() - t0
    if body.get("save"):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / f"{manifest['token_sha256']}.safetensors").write_bytes(blob)
    print(f"[server] {len(tokens)} tokens: prefill {manifest['timing']['prefill_seconds']:.2f}s "
          f"({manifest['timing']['prefill_tok_s']:.0f} tok/s), request {t_gen:.2f}s, state {len(blob) / 1e6:.1f} MB", flush=True)
    headers = {"X-Split-NV-Meta": json.dumps({k: v for k, v in manifest.items() if k != "layers"})}
    return Response(content=blob, media_type="application/octet-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("SPLIT_NV_HOST", "0.0.0.0"), port=int(os.environ.get("SPLIT_NV_PORT", "10051")), log_level="warning")
