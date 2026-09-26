"""cN aggregate/per-stream decode tok/s through a running og worker (host_server.py), og/stats deltas.

  DS41_TREE=... python batch_cycle.py http://127.0.0.1:12699 <label> [c=4] [ctx=8192] [reps=3] [max_tokens=256]
Each rep starts c concurrent fresh N-token prompts (api_bench method: document + question, unique nonce,
temperature 0, streamed). agg_tok_s = tokens emitted by all requests inside the window where all c decode
(chunk timestamps) / window; per_stream = each request's own (tokens-1)/(last-first).
"""
import json, os, sys, time, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HOME = Path.home()
sys.path.insert(0, str(Path(os.environ['DS41_TREE'])/'og_serve'))
from api_bench import QUESTIONS, chat_stream, document  # noqa: E402

base, label = sys.argv[1], sys.argv[2]
c, n, reps, max_tokens = (int(x) for x in (sys.argv[3:] + ['4', '8192', '3', '256'][len(sys.argv[3:]):]))
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))


def stats():
    return json.loads(urllib.request.urlopen(base + '/og/stats', timeout=5).read())


out = []
for r in range(reps):
    docs = [document(tok, n) + '\n\n' + QUESTIONS[(r + i) % 4] for i in range(c)]
    s0 = stats()
    with ThreadPoolExecutor(c) as pool:
        res = list(pool.map(lambda d: chat_stream(base, 'ds41-og', [{'role': 'user', 'content': d}], max_tokens=max_tokens), docs))
    s1 = stats()
    lo, hi = max(x['first_at'] for x in res), min(x['last_at'] for x in res)
    both = sum(sum(1 for t in x['chunks'] if lo < t <= hi) * x['completion_tokens'] / max(1, len(x['chunks'])) for x in res)
    d = {k: s1[k] - s0[k] for k in ('steps', 'rows', 'fused_calls', 'fused_steps', 'box_s', 'wait_s') if k in s1}
    first, last = min(x['first_at'] for x in res), max(x['last_at'] for x in res)
    total = sum(x['completion_tokens'] for x in res)
    tps = total / max(1, d['steps'])  # tokens per og step (verify cycle) over the run
    rec = dict(label=label, c=c, ctx=n, rep=r, agg_tok_s=round(both / (hi - lo), 1) if hi > lo else None,
               span_tok_s=round(total / (last - first), 1), tok_per_step=round(tps, 2),
               agg_steps_s=round(both / (hi - lo) / tps, 1) if hi > lo else None,
               span_steps_s=round(total / (last - first) / tps, 1),
               window_s=round(hi - lo, 2), per_stream=[round(x['decode_tok_s'] or 0, 1) for x in res],
               tokens=[x['completion_tokens'] for x in res], ttft=[round(x['ttft_s'], 2) for x in res],
               rows_per_step=round(d['rows'] / max(1, d['steps']), 2), fused_share=round(d.get('fused_steps', 0) / max(1, d['steps']), 2),
               wait_ms=round(1000 * d['wait_s'] / max(1, d['steps']), 2))
    print(json.dumps(rec), flush=True)
    out.append(rec)
ok = [x for x in out if x['agg_tok_s']]
print(json.dumps(dict(label=label, c=c, ctx=n, summary=dict(
    agg_tok_s=round(statistics.median(x['agg_tok_s'] for x in ok), 1) if ok else None,
    agg_steps_s=round(statistics.median(x['agg_steps_s'] for x in ok), 1) if ok else None,
    span_tok_s=round(statistics.median(x['span_tok_s'] for x in out), 1),
    span_steps_s=round(statistics.median(x['span_steps_s'] for x in out), 1),
    per_stream=round(statistics.median(v for x in out for v in x['per_stream']), 1),
    rows_per_step=round(statistics.median(x['rows_per_step'] for x in out), 2)))), flush=True)
