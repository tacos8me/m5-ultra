"""c1 ms per og step through a running og worker (host_server.py): same prompts, streamed, og/stats deltas."""
import json, os, sys, time, statistics, urllib.request
from pathlib import Path
HOME = Path.home()
sys.path.insert(0, str(Path(os.environ['DS41_TREE'])/'og_serve'))
from api_bench import chat_stream, document  # noqa: E402

base = sys.argv[1]
label = sys.argv[2]
reps = int(os.environ.get('HC_REPS', '4'))
n = int(os.environ.get('HC_CTX', '8192'))
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))


def stats():
    return json.loads(urllib.request.urlopen(base + '/og/stats', timeout=5).read())


doc = 'host-cycle fixed prompt\n' + document(tok, n).split('\n', 1)[1]
out = []
for r in range(reps + 1):
    s0 = stats()
    msgs = [{'role': 'user', 'content': doc + f'\n\nQuestion {r % 2}: summarize the code above in detail.'}]
    t0 = time.time()
    res = chat_stream(base, 'ds41-og', msgs, max_tokens=int(os.environ.get('HC_TOKENS', '256')))
    s1 = stats()
    steps = s1['steps'] - s0['steps']
    rec = dict(label=label, rep=r, steps=steps, rows=s1['rows'] - s0['rows'],
               box_ms=round(1000 * (s1['box_s'] - s0['box_s']) / max(1, steps), 3),
               wait_ms=round(1000 * (s1['wait_s'] - s0['wait_s']) / max(1, steps), 3),
               rt_minus_box_ms=round(1000 * ((s1['roundtrip_s'] - s0['roundtrip_s']) - (s1['box_s'] - s0['box_s'])) / max(1, steps), 3),
               decode_ms_per_step=round(1000 * (res['last_at'] - res['first_at']) / max(1, steps - 1), 3),
               tokens=res['completion_tokens'], text_head=res['text'][:40])
    print(json.dumps(rec), flush=True)
    if r:
        out.append(rec)
summ = {k: round(statistics.median(r[k] for r in out), 3) for k in ('decode_ms_per_step', 'box_ms', 'wait_ms', 'rt_minus_box_ms')}
print(json.dumps(dict(label=label, summary=summ, rows_per_step=round(sum(r['rows'] for r in out) / sum(r['steps'] for r in out), 3))), flush=True)
