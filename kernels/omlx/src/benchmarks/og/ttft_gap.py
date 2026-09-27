"""Bench-leg2-shaped follow-ups: one user message = the same doc + a different question (the doc piece is new text
each time for the piece cache). Writes fe_bench-format records (kind 'ttft') for fe_report.py."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"og_serve"))
import fe_bench
base, label, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
doc = fe_bench.document(tok, n, '[gap doc]\n')
out = (fe_bench.LOGS/f'{label}.jsonl').open('a')
for i, q in enumerate(fe_bench.QUESTIONS):
    r = fe_bench.stream(base, '/v1/chat/completions', dict(model='ds41-og', messages=[{'role': 'user', 'content': doc + '\n\n' + q}],
                                                        max_tokens=8, temperature=0))
    rec = dict(t=time.time(), label=label, kind='ttft', n=n, rep=i, ttft_s=r['ttft_s'], t0=r['t0'], usage=r['usage'])
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(json.dumps(dict(rep=i, ttft_s=round(r['ttft_s'], 3), srv=(r['usage'] or {}).get('time_to_first_token'))), flush=True)
