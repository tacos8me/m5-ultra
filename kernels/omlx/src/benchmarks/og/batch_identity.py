"""Temperature-0 token identity through a running og worker (host_server.py): each prompt alone vs together.

  DS41_TREE=... python batch_identity.py http://127.0.0.1:12699 <label> [groups=1,4,3,2] [max_tokens=256]
Fixed prompts (api_bench documents without the nonce) of mixed lengths. Every group size g runs the prompts
in waves of g concurrent requests; texts and token counts must equal the alone (g=1) run. Prints og/stats
fused deltas per wave so a fused run is visible. Start the worker with DS41_OG_BOX_CACHE=0 so every
request is a fresh box prefill.
"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HOME = Path.home()
sys.path.insert(0, str(Path(os.environ['DS41_TREE'])/'og_serve'))
from api_bench import QUESTIONS, chat_stream, document  # noqa: E402

base, label = sys.argv[1], sys.argv[2]
sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,4,3,2').split(',')]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
LENGTHS = [8192, 3000, 8192, 12000]
docs = ['identity prompt %d\n' % i + document(tok, n).split('\n', 1)[1] + '\n\n' + QUESTIONS[i % 4]
        for i, n in enumerate(LENGTHS)]


def stats():
    return json.loads(urllib.request.urlopen(base + '/og/stats', timeout=5).read())


def run(d):
    r = chat_stream(base, 'ds41-og', [{'role': 'user', 'content': d}], max_tokens=max_tokens)
    return dict(text=r['text'], tokens=r['completion_tokens'])


ref, ok = None, True
for g in sizes:
    s0 = stats()
    outs = []
    for w in range(0, len(docs), g):
        with ThreadPoolExecutor(g) as pool:
            outs += list(pool.map(run, docs[w:w + g]))
    s1 = stats()
    rec = dict(label=label, group=g, tokens=[o['tokens'] for o in outs],
               fused_steps=s1.get("fused_steps", 0) - s0.get("fused_steps", 0), draft_batches=s1.get("draft_batches", 0) - s0.get("draft_batches", 0), steps=s1["steps"] - s0["steps"])
    if ref is None:
        ref = outs
    else:
        rec['identical'] = [a == b for a, b in zip(ref, outs)]
        ok &= all(rec['identical'])
    print(json.dumps(rec), flush=True)
Path(HOME/'llm/ds41/batch'/f'identity-{label}.json').write_text(json.dumps(dict(ref=ref)))
print(json.dumps(dict(label=label, identity_ok=ok)), flush=True)
