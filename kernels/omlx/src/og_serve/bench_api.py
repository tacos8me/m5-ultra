"""Through-API benchmark of a ds41 endpoint, same method for og and q3.

Method (as og_serve/api_bench.py / coord decode_samples): a fresh N-token document of DS41
source (unique nonce, no prefix reuse) + a question, temperature 0, streamed.
  ttft      fresh prompt of N tokens, 16 output tokens: time to the first content chunk
  resume    turn 1 = N-token doc + question (128 tokens out); turn 2 = turn 1 + answer + a new
            question: turn-2 TTFT (prefix reuse), and whether a repeat of turn 2 gives the same text
            (exactness against a cache-free prefill is og-cache's gate, not measurable here)
  c1        decode tok/s = (completion_tokens - 1) / (last - first content chunk), 256 tokens
  c2        two concurrent N-token requests; total tok/s inside the window where both decode

  python og_serve/bench_api.py --label sv-bench-og --base http://127.0.0.1:12161 --model ds41-og \
      --ttft 8192,131072,524288,1048576 --resume 131072 --c1 8192:3,131072:3,524288:2 --c2 8192:2
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from api_bench import QUESTIONS, chat_stream, document  # noqa: E402

HOME = Path.home()
LOGS = Path(os.environ.get('OG_LOGS', str(HOME/'llm/ds41/og-serve')))


def pairs(spec):
    out = []
    for part in [p for p in spec.split(',') if p]:
        n, _, reps = part.partition(':')
        out.append((int(n), int(reps or 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--ttft', default='')
    ap.add_argument('--resume', default='')
    ap.add_argument('--c1', default='')
    ap.add_argument('--c2', default='')
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--budget', type=float, default=840)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
    LOGS.mkdir(parents=True, exist_ok=True)
    out = (LOGS/f'{args.label}.bench.jsonl').open('a')
    deadline = time.monotonic() + args.budget
    summary = dict(label=args.label, model=args.model, ttft={}, resume={}, c1={}, c2={})

    def emit(rec):
        out.write(json.dumps(rec) + '\n'); out.flush()
        print(json.dumps(rec)[:600], flush=True)

    def left(n):
        if time.monotonic() > deadline:
            emit(dict(kind='skip', n=n, reason='budget'))
            return False
        return True

    for n, reps in pairs(args.ttft) or []:
        vals = []
        for rep in range(reps):
            if not left(n):
                break
            r = chat_stream(args.base, args.model, [{'role': 'user', 'content': document(tok, n) + '\n\n' + QUESTIONS[rep % 4]}],
                            max_tokens=16)
            vals.append(r['ttft_s'])
            emit(dict(kind='ttft', n=n, rep=rep, ttft_s=r['ttft_s'], prompt_tokens=r['prompt_tokens'], backend=r['backend']))
        if vals:
            summary['ttft'][n] = round(statistics.median(vals), 2)

    for n, reps in pairs(args.resume) or []:
        for rep in range(reps):
            if not left(n):
                break
            doc = document(tok, n)
            msgs = [{'role': 'user', 'content': doc + '\n\n' + QUESTIONS[0]}]
            r1 = chat_stream(args.base, args.model, msgs, max_tokens=128)
            msgs += [{'role': 'assistant', 'content': r1['text']}, {'role': 'user', 'content': QUESTIONS[1]}]
            r2 = chat_stream(args.base, args.model, msgs, max_tokens=64)
            r3 = chat_stream(args.base, args.model, msgs, max_tokens=64)
            emit(dict(kind='resume', n=n, rep=rep, turn1_ttft_s=r1['ttft_s'], turn2_ttft_s=r2['ttft_s'],
                      turn2_repeat_ttft_s=r3['ttft_s'], turn2_prompt_tokens=r2['prompt_tokens'],
                      turn2_repeat_identical=r2['text'] == r3['text'], backend=r2['backend']))
            summary['resume'].setdefault(n, []).append(dict(turn1=round(r1['ttft_s'], 2), turn2=round(r2['ttft_s'], 2),
                                                            repeat_identical=r2['text'] == r3['text']))

    for n, reps in pairs(args.c1) or []:
        vals, ttfts = [], []
        for rep in range(reps):
            if not left(n):
                break
            r = chat_stream(args.base, args.model, [{'role': 'user', 'content': document(tok, n) + '\n\n' + QUESTIONS[rep % 4]}],
                            max_tokens=args.max_tokens)
            vals.append(r['decode_tok_s'])
            ttfts.append(r['ttft_s'])
            emit(dict(kind='c1', n=n, rep=rep, decode_tok_s=r['decode_tok_s'], ttft_s=r['ttft_s'],
                      completion_tokens=r['completion_tokens'], prompt_tokens=r['prompt_tokens'], backend=r['backend']))
        if vals:
            summary['c1'][n] = dict(median=round(statistics.median(vals), 1), all=[round(v, 1) for v in vals],
                                    ttft=round(statistics.median(ttfts), 2))

    for n, reps in pairs(args.c2) or []:
        totals = []
        for rep in range(reps):
            if not left(n):
                break
            docs = [document(tok, n) + '\n\n' + QUESTIONS[i] for i in range(2)]
            with ThreadPoolExecutor(2) as pool:
                pair = list(pool.map(lambda d: chat_stream(args.base, args.model, [{'role': 'user', 'content': d}],
                                                           max_tokens=args.max_tokens), docs))
            lo = max(x['first_at'] for x in pair)
            hi = min(x['last_at'] for x in pair)
            both = sum(sum(1 for t in x['chunks'] if lo < t <= hi) * x['completion_tokens'] / max(1, len(x['chunks']))
                       for x in pair)
            total = both / (hi - lo) if hi > lo else None
            totals.append(total)
            emit(dict(kind='c2', n=n, rep=rep, overlap_total_tok_s=total, overlap_s=hi - lo,
                      per_request=[x['decode_tok_s'] for x in pair], ttft_s=[x['ttft_s'] for x in pair]))
        vals = [t for t in totals if t]
        if vals:
            summary['c2'][n] = round(statistics.median(vals), 1)
    (LOGS/f'{args.label}.bench.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
