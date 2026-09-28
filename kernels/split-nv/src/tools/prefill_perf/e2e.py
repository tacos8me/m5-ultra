"""End-to-end byte identity of everything the Mac receives, flags off vs on: runs the real split-nv engine
(`python3 -m split_nv.engine`: front, sessions, capture hooks, state packing, streaming, prefix cache, steps) on the
5-layer mini model (tools/prefill_perf/mini_site, PF_MINI=1) inside one container, drives it through the step API
with a fixed request sequence, records per request: ACK (minus timing), every state tensor (sha256), the manifest
(minus timing), for streamed OPENs every frame in order (tag, header minus timing, payload sha256), and the 6 step
payloads; then stops the engine and hashes the prefix-cache files it wrote.

  e2e.py run TAG [ENV=VALUE ...]     -> /pf/e2e/TAG/{result.json, engine.log, cache.json}
  e2e.py compare TAG_A TAG_B
"""
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/work/tools")
from og_client import FRAME, Session, recv_exact, recv_frame, send_frame, step_script  # noqa: E402

OUT = "/pf/e2e"
VIEW = "/pf/view-e2e"
IDS = "/pf/e2e/ids-mini.json"
TIMING = ("_s", "second", "timing", "tok_s", "chunks", "est_s")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not any(t in k for t in TIMING)}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def nonfinite(b):
    """non-finite bf16 values in a byte string (exponent bits all ones)"""
    import numpy as np
    u = np.frombuffer(b, dtype=np.uint16)
    return int(((u & 0x7F80) == 0x7F80).sum())


def open_raw(s, tokens, **options):
    """OPEN recording every frame; returns (ack, frames [(tag, header, payload sha)], tensors {name: sha})."""
    payload = struct.pack("<%dI" % len(tokens), *tokens)
    header = dict(proto=1, identity="", prompt_tokens=len(tokens), token_sha256=hashlib.sha256(payload).hexdigest(),
                  state="full")
    header.update(options)
    send_frame(s.sock, b"OPEN", header, payload)
    tag, h, n = recv_frame(s.sock)
    ack = json.loads(h)
    if tag != b"ACK " or not ack.get("ok"):
        raise RuntimeError(f"OPEN refused {tag} {ack}")
    s.sid, s.length = int(ack["session"]), len(tokens) - 1
    frames, tensors = [], {}
    while True:
        tag, h, n = recv_frame(s.sock)
        hd = json.loads(h) if h and tag != b"STAT" else {}
        body = bytes(recv_exact(s.sock, n)) if n else b""
        frames.append([tag.decode(), strip(hd), hashlib.sha256(body).hexdigest() if tag != b"STAT" else None])
        if tag == b"STAT":
            from og_client import parse_safetensors
            t, manifest = parse_safetensors(body)
            tensors = {k: [v[0], v[1], hashlib.sha256(v[2]).hexdigest()] for k, v in t.items()}
            tensors["_nonfinite_tail_hidden"] = nonfinite(t["tail.hidden"][2])
            frames[-1][1] = strip(manifest)
            return strip(ack), frames, tensors
        if tag == b"TENS":
            name = hd["name"]
            tensors.setdefault(name, hashlib.sha256())
            tensors[name].update(body)
            if name == "tail.hidden":
                frames[-1].append(nonfinite(body))
        if tag == b"END ":
            return strip(ack), frames, {k: v.hexdigest() for k, v in tensors.items()}
        if tag == b"ERR ":
            raise RuntimeError(f"ERR {hd}")


def request(tokens, n, **opts):
    s = Session("127.0.0.1", 10052)
    ack, frames, tensors = open_raw(s, tokens[:n], **opts)
    steps = []
    for keep, ids in step_script(tokens, n):
        p, _ = s.step(keep, ids)
        steps.append(hashlib.sha256(p).hexdigest())
    s.close()
    return {"n": n, "opts": opts, "ack": ack, "frames": frames, "tensors": tensors, "steps": steps}


SEQUENCE = [  # (n, options): odd lengths, >= 3 chunk boundaries, streamed and whole, cache writes + resumes
    (8193, {}), (8195, {}), (16385, {}), (16386, {"stream": 1}), (20001, {}), (20001, {"stream": 1}),
    (131073, {"stream": 1}),
    (20001, {"cache": 1}), (30001, {"cache": 1, "stream": 1}), (47111, {"cache": 1}), (131073, {"cache": 1}),
    (131073, {"cache": 1, "stream": 1}), (16385, {"cache": 1, "state": "lean"}),
]


def run(tag, extra_env):
    d = f"{OUT}/{tag}"
    os.makedirs(d, exist_ok=True)
    tokens = json.load(open(IDS))
    env = dict(os.environ, PF_MINI="1", PYTHONPATH="/work/tools/prefill_perf/mini_site:/work/hooks",
               SGLANG_SM120_FLASHMLA_BACKEND="flashinfer", SGLANG_FLASHINFER_MOE_FUSED_FINALIZE="0",
               SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE="0", SGLANG_DSV41_INDEXER_LOGITS_BUDGET_MB="128",
               SGLANG_OPT_USE_TOPK_V2="1", SPLIT_NV_HOOKS="1", SPLIT_NV_CONFIG=f"{VIEW}/config.json",
               SPLIT_NV_DIR="/dev/shm/pfnv", SPLIT_NV_MAX_TOKENS="139264", SPLIT_NV_CACHE_GB="6",
               SPLIT_NV_PUBLIC_HTTP="", SPLIT_NV_TRIM="1", SPLIT_NV_B12X="1", SPLIT_NV_OG_MOE="1",
               SPLIT_NV_DEV="0", SPLIT_NV_SPIN_S="0.2", SPLIT_NV_PF_OVERLAP="0",
               SPLIT_NV_Q_NOCOPY="0", SPLIT_NV_CE_AR="0", NCCL_BUFFSIZE="1048576",
               # no CUDA graphs for the steps (eager steps in both runs): graph capture's custom-all-reduce IPC does not
               # work with expandable segments, which the 5 GB budget needs (no fragmentation reserve)
               SPLIT_NV_GRAPHS="", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.update(extra_env)
    log = open(f"{d}/engine.log", "w")
    args = ["python3", "-m", "split_nv.engine", "--model-path", VIEW, "--trust-remote-code", "--served-model-name", "mini",
            "--tp", "2", "--host", "127.0.0.1", "--port", "10050", "--mem-fraction-static", "0.94",
            "--context-length", "139264", "--max-total-tokens", "139264", "--max-running-requests", "2",
            "--chunked-prefill-size", "8192", "--enable-deepseek-v4-fp4-indexer", "--fp8-gemm-backend",
            "flashinfer_cutlass", "--disable-cuda-graph", "--disable-radix-cache"]
    eng = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    res = {"tag": tag, "env": extra_env, "requests": [], "error": None}
    try:
        t0 = time.time()
        while True:
            if eng.poll() is not None:
                raise RuntimeError(f"engine exited with {eng.returncode} during start-up")
            try:
                h = json.loads(urllib.request.urlopen("http://127.0.0.1:10050/health", timeout=2).read())
                if h.get("ok"):
                    res["health"] = strip(h)
                    break
            except OSError:
                pass
            if time.time() - t0 > 1200:
                raise RuntimeError("engine start-up timeout")
            time.sleep(3)
        print(f"[e2e {tag}] engine up in {time.time() - t0:.0f}s: {h.get('prefill_perf')}", flush=True)
        for n, opts in SEQUENCE:
            t1 = time.time()
            r = request(tokens, n, **opts)
            res["requests"].append(r)
            nf = r["tensors"].get("_nonfinite_tail_hidden", sum(f[3] for f in r["frames"] if len(f) > 3))
        print(f"[e2e {tag}] n={n} {opts}: non-finite tail.hidden values {nf}; {len(r['tensors'])} tensors, {len(r['frames'])} frames, "
                  f"resumed {r['ack'].get('resumed_tokens')}, {time.time() - t1:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        print(f"[e2e {tag}] ERROR {res['error']}", flush=True)
    finally:
        if eng.poll() is None:
            os.killpg(eng.pid, signal.SIGTERM)
            try:
                eng.wait(90)
            except subprocess.TimeoutExpired:
                os.killpg(eng.pid, signal.SIGKILL)
                eng.wait()
    files = {}
    root = "/dev/shm/split-nv/cache"  # the container's private /dev/shm (run.sh has no --ipc=host)
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            files[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if f.startswith(("ent", "idx")):  # keep the small ones for a field-level comparison
                import shutil
                os.makedirs(f"{d}/cache", exist_ok=True)
                shutil.copy(p, f"{d}/cache/{f}")
    res["cache_files"] = files
    # entries / indexes carry time-based ids: canonical form = (P, rank file, tensor bytes) and (P, capture, digest, #blocks)
    import glob
    canon = []
    for p in glob.glob(f"{root}/**/ent-*", recursive=True):
        b = open(p, "rb").read()
        n = struct.unpack("<Q", b[:8])[0]
        hd = json.loads(b[8:8 + n])
        meta = hd.pop("__metadata__", {})
        tens = {k: hashlib.sha256(b[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]]).hexdigest() for k, v in hd.items()}
        canon.append(["ent", meta.get("P"), p.rsplit(".", 1)[-1], meta, tens])
    for p in glob.glob(f"{root}/**/idx-*", recursive=True):
        j = json.load(open(p))
        canon.append(["idx", j.get("P"), j.get("capture"), j.get("digest"), len(j.get("blocks", []))])
    res["cache_canon"] = sorted(canon, key=lambda x: json.dumps(x, sort_keys=True))
    json.dump(res, open(f"{d}/result.json", "w"), indent=1)
    print(f"[e2e {tag}] done: {len(res['requests'])} requests, {len(files)} cache files, error {res['error']}", flush=True)


def compare(a_tag, b_tag):
    a = json.load(open(f"{OUT}/{a_tag}/result.json"))
    b = json.load(open(f"{OUT}/{b_tag}/result.json"))
    ok = a["error"] is None and b["error"] is None and len(a["requests"]) == len(b["requests"]) == len(SEQUENCE)
    print(f"errors: {a_tag}={a['error']} {b_tag}={b['error']}; requests {len(a['requests'])} / {len(b['requests'])}")
    for ra, rb in zip(a["requests"], b["requests"]):
        diff = []
        for k in ("ack", "frames", "tensors", "steps"):
            if ra[k] != rb[k]:
                if k == "tensors":
                    diff.append(f"tensors {sorted(x for x in set(ra[k]) | set(rb[k]) if ra[k].get(x) != rb[k].get(x))[:6]}")
                elif k == "frames":
                    idx = [i for i, (x, y) in enumerate(zip(ra[k], rb[k])) if x != y]
                    diff.append(f"frames {len(ra[k])}/{len(rb[k])} differing at {idx[:6]}")
                else:
                    diff.append(k)
        ok &= not diff
        nb = sum(1 for f in ra["frames"] if f[0] == "TENS")
        print(f"  n={ra['n']:6d} {json.dumps(ra['opts']):36s} tensors {len(ra['tensors']):3d} TENS frames {nb:3d} "
              f"steps {len(ra['steps'])} resumed {ra['ack'].get('resumed_tokens')}: {'IDENTICAL' if not diff else 'DIFF ' + '; '.join(diff)}")
    # prefix cache: file names carry random ids; compare the multiset of (kind, content hash)
    def kinds(files):
        out = {}
        for name, h in files.items():
            base = os.path.basename(name)
            kind = base.split("-")[0]
            out.setdefault(kind, []).append(h)
        return {k: sorted(v) for k, v in out.items()}
    ka, kb = kinds(a["cache_files"]), kinds(b["cache_files"])
    for k in sorted(set(ka) | set(kb)):
        same = ka.get(k) == kb.get(k)
        print(f"  cache {k:12s}: {len(ka.get(k, []))} / {len(kb.get(k, []))} files, contents {'IDENTICAL' if same else 'DIFFERENT'}")
    ca, cb = a.get("cache_canon"), b.get("cache_canon")
    if ca is not None and cb is not None:
        same = ca == cb
        ok &= same and ka.get("blk") == kb.get("blk") and ka.get("rows") == kb.get("rows") and ka.get("tok") == kb.get("tok")
        print(f"  cache entries+indexes (canonical: P, rank, tensor bytes / P, capture, digest, #blocks): "
              f"{sum(x[0] == 'ent' for x in ca)} entry files, {sum(x[0] == 'idx' for x in ca)} indexes: {'IDENTICAL' if same else 'DIFFERENT'}")
    print("COMPARE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], dict(kv.split("=", 1) for kv in sys.argv[3:]))
    else:
        compare(sys.argv[2], sys.argv[3])
