"""Multi-turn resume TTFT through the OpenAI API (ds41-og prefix reuse).

Each conversation: turn 1 = a fresh N-token document (unique nonce) + a
question; turns 2..T append the assistant reply and a follow-up question.
Streams every turn at temperature 0 and records TTFT, prompt tokens and the
reply. `--replay FILE` re-sends the recorded turns of an earlier run as
independent requests (e.g. to a worker with the cache disabled) and reports
whether every reply is identical. `--conversations 2` runs two conversations
side by side (turn k of both in flight together).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import urllib.request

from api_bench import LOGS, chat_stream, document

QUESTIONS = ['Summarize the code above in one sentence.', 'Now name the single most complex function.',
             'Which invariant does it rely on? One sentence.', 'Suggest one simplification.',
             'Summarize our discussion in one sentence.', 'Any risks left? One sentence.']


def og_stats(base):
    try:
        with urllib.request.urlopen(base + '/og/stats', timeout=5) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001 -- not an og worker
        return None


def conversation(base, model, tok, size, turns, max_tokens, think, emit, tag):
    messages = [{'role': 'user', 'content': document(tok, size) + '\n\n' + QUESTIONS[0]}]
    extra = {'chat_template_kwargs': {'enable_thinking': True}} if think else {}
    records = []
    for turn in range(1, turns + 1):
        before = og_stats(base)
        sent = time.time()
        r = chat_stream(base, model, messages, max_tokens=max_tokens, **extra)
        after = og_stats(base)
        rec = dict(kind='turn', tag=tag, size=size, turn=turn, think=think, prompt_tokens=r['prompt_tokens'],
                   completion_tokens=r['completion_tokens'], ttft_s=round(r['ttft_s'], 3),
                   decode_tok_s=r['decode_tok_s'] and round(r['decode_tok_s'], 1), text=r['text'],
                   sent_wall=sent, first_wall=sent + r['ttft_s'], messages=messages)
        if before and after:
            rec['box_resumed_tokens'] = after['box_resumed_tokens'] - before['box_resumed_tokens']
            rec['delta_rows'] = after['delta_rows'] - before['delta_rows']
        emit({k: v for k, v in rec.items() if k != 'messages'})
        records.append(rec)
        messages = messages + [{'role': 'assistant', 'content': r['text']},
                               {'role': 'user', 'content': QUESTIONS[turn % len(QUESTIONS)]}]
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--model', default='ds41-og')
    ap.add_argument('--label', required=True)
    ap.add_argument('--sizes', default='8192,131072')
    ap.add_argument('--turns', type=int, default=5)
    ap.add_argument('--max-tokens', type=int, default=64)
    ap.add_argument('--conversations', type=int, default=1)
    ap.add_argument('--think', action='store_true')
    ap.add_argument('--replay', help='jsonl of a previous run: resend its turns and compare replies')
    args = ap.parse_args()
    out = (LOGS/(args.label + '.cache.jsonl')).open('a')

    def emit(record):
        record = dict(t=time.time(), **record)
        out.write(json.dumps(record) + '\n')
        out.flush()
        print(json.dumps({k: (v[:80] if isinstance(v, str) else v) for k, v in record.items()}), flush=True)

    if args.replay:
        same = total = 0
        for line in Path(args.replay).read_text().splitlines():
            rec = json.loads(line)
            if rec.get('kind') != 'turn_messages':
                continue
            extra = {'chat_template_kwargs': {'enable_thinking': True}} if rec.get('think') else {}
            r = chat_stream(args.base, args.model, rec['messages'], max_tokens=args.max_tokens, **extra)
            total += 1
            same += r['text'] == rec['text']
            emit(dict(kind='replay', tag=rec['tag'], size=rec['size'], turn=rec['turn'], ttft_s=round(r['ttft_s'], 3),
                      identical=r['text'] == rec['text'], prompt_tokens=r['prompt_tokens']))
        emit(dict(kind='replay_summary', identical=same, total=total))
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get('DS41_OG_TOKENIZER',
                                                       str(Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx')))
    for size in [int(x) for x in args.sizes.split(',')]:
        with ThreadPoolExecutor(args.conversations) as pool:
            jobs = [pool.submit(conversation, args.base, args.model, tok, size, args.turns, args.max_tokens,
                                args.think, emit, f'{size}-{i}') for i in range(args.conversations)]
            for job in jobs:
                for rec in job.result():
                    out.write(json.dumps(dict(rec, kind='turn_messages')) + '\n')
        emit(dict(kind='stats', size=size, og=og_stats(args.base)))


if __name__ == '__main__':
    main()
