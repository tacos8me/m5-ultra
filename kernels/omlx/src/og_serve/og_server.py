"""ds41-og worker: omlx OpenAI server over the original-weight split pipeline.

Run under ~/llm/bin/gpu-exec (this PID holds gpu.lock and owns ~150 GiB). The
RTX box (10.10.10.1:10052) runs layers 0-19 for prefill and every decode step;
this process runs layers 20-39, the head and DSpark (og_model.py). /health is
503 until a real pipeline warmup through the box has completed. An external
245 GiB RSS/footprint SIGKILL watchdog (watch.py, bound to this PID) and an
in-process 245 GiB footprint SIGTERM guard protect the machine.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HOME = Path.home()
HERE = Path(__file__).resolve().parent
TREE = os.environ.get('DS41_TREE', str(HERE.parent))
sys.path[:0] = [TREE]
for key, value in dict(MLX_ENABLE_TF32='0', OMLX_BONJOUR='0', OMLX_DISCOVERY='0', DS41_NATIVE_VERIFY='0',
                       DS41_MHC='1', DS41_GROWTH='1', DS41_GATHER='1', DS41_INDEX_NAX='1', DS41_SPARSE='1',
                       DS41_NATIVE_DECODE='1', DS41_MTP_COST_POLICY='1', DS41_COPY_DRAFT='1',
                       DS41_EXTRA_DRAFT='1', DS41_PREFIX_CACHE_GIB='0',
                       # Per-request verify + pre-sent next rows overlaps one request's box
                       # step with the other's Mac verify and draft (ragged verify is lock-step).
                       DS41_BATCH_VERIFY='0').items():
    os.environ.setdefault(key, value)
OG = Path(os.environ.get('DS41_OG_HOME', str(HOME/'llm/ds41/og')))
GIB = 1024**3

watch = subprocess.Popen(['/opt/homebrew/bin/python3', str(HERE/'watch.py'), str(os.getpid()),
                          str(OG/'logs'/'og-worker.memory.json'), os.environ.get('DS41_OG_LEASE_S', '0')])

import mlx.core as mx  # noqa: E402

_set_wired = mx.set_wired_limit
mx.set_wired_limit = lambda value: _set_wired(min(value, 245 * GIB))
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
_set_cache = mx.set_cache_limit
mx.set_cache_limit = lambda value: _set_cache(min(value, 512 * 1024**2))
mx.set_cache_limit(512 * 1024**2)
_set_memory = mx.set_memory_limit
mx.set_memory_limit = lambda value: _set_memory(min(value, 235 * GIB))
mx.set_memory_limit(235 * GIB)

from omlx.cluster import worker_shim  # noqa: E402
worker_shim.ensure_cluster_python_shim = lambda **kwargs: None
from omlx.patches.deepseek_v41 import og_model  # noqa: E402

box_host, _, box_port = os.environ.get('DS41_OG_BOX', '10.10.10.1:10052').rpartition(':')
og_model.install(box_host, int(box_port))

from omlx.utils.proc_memory import get_phys_footprint  # noqa: E402


def guard():
    while True:
        if get_phys_footprint() > 245 * GIB:
            print(json.dumps({'safety_stop': 'physical footprint exceeded 245 GiB', 'pid': os.getpid()}), flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
            return
        time.sleep(1)


threading.Thread(target=guard, daemon=True).start()

WARM = threading.Event()
from fastapi.responses import JSONResponse  # noqa: E402
from omlx import server  # noqa: E402


@server.app.middleware('http')
async def gate_health(request, call_next):
    if request.url.path == '/health' and not WARM.is_set():
        return JSONResponse({'status': 'warming'}, status_code=503)
    return await call_next(request)


@server.app.get('/og/stats')
async def og_stats():
    from omlx.patches.deepseek_v41 import pipe_wire
    store = og_model.STORE.summary() if og_model.STORE is not None else None
    return dict(og_model.STATS, sessions=len(og_model.SESSIONS), recoveries=pipe_wire.RECOVERIES[-20:],
                resume=og_resume.STATS, prefix_rows=store)


sys.path.insert(0, str(HERE))
import og_resume  # noqa: E402
og_resume.install(server.app)


def warmup():
    import urllib.request
    port = sys.argv[sys.argv.index('--port') + 1] if '--port' in sys.argv else '8000'
    base = f'http://127.0.0.1:{port}'
    model = os.environ.get('DS41_OG_MODEL_ID', 'ds41-og')
    source = Path(TREE)/'omlx/patches/deepseek_v41/language.py'
    text = source.read_text()[:16000].replace('｜', '|').replace('<think>', '[think]').replace('</think>', '[/think]')
    prompts = [('Say hi.', 16),
               ('Explain in detail how TCP congestion control works, including slow start and BBR.', 128),
               (text + '\n\nSummarize this code.', 64)]
    t0 = time.time()
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(base + '/v1/models', timeout=2)
                break
            except Exception:
                time.sleep(1)
        for content, max_tokens in prompts:
            body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': content}],
                               'max_tokens': max_tokens, 'temperature': 0}).encode()
            req = urllib.request.Request(base + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
            reply = json.loads(urllib.request.urlopen(req, timeout=900).read())
            if not reply.get('choices'):
                raise RuntimeError(f'warmup reply without choices: {str(reply)[:200]}')
        print(json.dumps({'warmup': 'done', 's': round(time.time() - t0, 1)}), flush=True)
        WARM.set()
    except Exception as e:
        # Stay unhealthy: the supervisor treats a missing /health as not ready.
        print(json.dumps({'warmup': 'failed', 'error': repr(e)[:300]}), flush=True)
        os.kill(os.getpid(), signal.SIGTERM)


if os.environ.get('DS41_WARMUP', '1') == '1':
    threading.Thread(target=warmup, daemon=True).start()
else:
    WARM.set()


def main():
    args = [a for a in sys.argv[1:]]
    sys.argv = [sys.argv[0], 'serve', '--base-path', str(OG/'profile'), '--model-dir', str(OG/'models'),
                '--no-hf-cache', '--no-cache', '--max-concurrent-requests',
                os.environ.get('DS41_OG_CONCURRENCY', '2'), *args]
    from omlx.cli import main as cli
    cli()


if __name__ == '__main__':
    main()
