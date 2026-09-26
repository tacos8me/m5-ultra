"""OpenAI-API checks and decode/TTFT samples against a ds41 endpoint.

Same method as coord/decode_samples.py: a fresh N-token document of DS41 source
(unique nonce, no prefix reuse) + a question, temperature 0, 256 tokens,
streamed; decode tok/s = (completion_tokens - 1) / (last - first content chunk).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import urllib.request

HOME = Path.home()
LOGS = Path(os.environ.get('OG_LOGS', str(HOME/'llm/ds41/og-mac')))
QUESTIONS = ['Summarize the code above in five bullet points.', 'Explain how the Engram tables are used, in detail.',
             'List the main classes and what each one does.', 'Describe the prefill path step by step.']


def post(base, path, body, stream=False):
    req = urllib.request.Request(base+path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800)


def chat_stream(base, model, messages, max_tokens=256, **extra):
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0, stream=True,
                stream_options={'include_usage': True}, **extra)
    t0 = time.perf_counter(); first = last = None; usage = None; text = []; tools = []; backend = None; chunks = []
    with post(base, '/v1/chat/completions', body) as r:
        backend = r.headers.get('x-ds41-og-backend')
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            d = json.loads(line[6:])
            if d.get('error'):
                raise RuntimeError(d['error'])
            usage = d.get('usage') or usage
            if d.get('choices'):
                delta = d['choices'][0].get('delta', {})
                if delta.get('content') or delta.get('tool_calls') or delta.get('reasoning_content'):
                    now = time.perf_counter(); first = first or now; last = now; chunks.append(now)
                if delta.get('content'):
                    text.append(delta['content'])
                if delta.get('tool_calls'):
                    tools.extend(delta['tool_calls'])
    c = usage['completion_tokens'] if usage else None
    return dict(prompt_tokens=usage and usage['prompt_tokens'], completion_tokens=c,
                ttft_s=None if first is None else first-t0, first_at=first, last_at=last,
                decode_tok_s=(c-1)/(last-first) if c and last and last > first else None,
                text=''.join(text), tool_calls=tools, backend=backend, chunks=chunks)


def document(tok, n):
    src = Path(os.environ.get('DS41_TREE', str(HOME/'src/wt/ds41-og')))/'omlx/patches/deepseek_v41'
    corpus = '\n\n'.join(f'File: {f.name}\n{f.read_text()}' for f in sorted(src.glob('*.py')))
    for s in tok.all_special_tokens:
        corpus = corpus.replace(s, '[special token literal]')
    corpus = corpus.replace('｜', '|').replace('<think>', '[think tag]').replace('</think>', '[/think tag]')
    base = tok.encode(corpus, add_special_tokens=False)
    return f'[run {time.time_ns()}]\n' + tok.decode((base * (n // len(base) + 2))[:n])


def checks(base, model, emit):
    with urllib.request.urlopen(base+'/v1/models', timeout=30) as r:
        emit(dict(kind='models', ids=[m['id'] for m in json.load(r)['data']]))
    with post(base, '/v1/chat/completions', dict(model=model, messages=[{'role': 'user', 'content': 'Say ready.'}],
                                                  max_tokens=8, temperature=0)) as r:
        d = json.load(r)
        emit(dict(kind='nonstream', text=d['choices'][0]['message']['content'], usage=d.get('usage'),
                  backend=r.headers.get('x-ds41-og-backend')))
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
              'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    r = chat_stream(base, model, [{'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'}],
                    max_tokens=128, tools=tools)
    emit(dict(kind='tool_call', tool_calls=r['tool_calls'], text=r['text'][:200], ttft_s=r['ttft_s']))
    r = chat_stream(base, model, [{'role': 'user', 'content': 'Write a Python function that merges overlapping intervals, with a docstring.'}])
    emit(dict(kind='stream', decode_tok_s=r['decode_tok_s'], ttft_s=r['ttft_s'], completion_tokens=r['completion_tokens'],
              text=r['text'][:300], backend=r['backend']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:12147')
    ap.add_argument('--model', default='ds41-og')
    ap.add_argument('--label', required=True)
    ap.add_argument('--contexts', default='8192,131072')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--pairs', type=int, default=1)
    ap.add_argument('--checks', action='store_true')
    ap.add_argument('--budget', type=float, default=700)
    ap.add_argument('--c2-max-tokens', type=int, default=256)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
    out = (LOGS/(args.label+'.api.jsonl')).open('a')
    deadline = time.monotonic() + args.budget

    def emit(record):
        out.write(json.dumps(record)+'\n'); out.flush()
        print(json.dumps({k: v for k, v in record.items() if k != 'pair'})[:1500], flush=True)
    if args.checks:
        checks(args.base, args.model, emit)
    for n in [int(x) for x in args.contexts.split(',') if x]:
        for rep in range(args.reps):
            if time.monotonic() > deadline:
                emit(dict(kind='skip', n=n)); return
            r = chat_stream(args.base, args.model, [{'role': 'user', 'content': document(tok, n) + '\n\n' + QUESTIONS[rep % 4]}])
            emit(dict(kind='c1', n=n, rep=rep, **{k: r[k] for k in ('prompt_tokens', 'completion_tokens', 'ttft_s', 'decode_tok_s', 'backend')}))
        for p in range(args.pairs):
            if time.monotonic() > deadline:
                emit(dict(kind='skip', n=n)); return
            docs = [document(tok, n) + '\n\n' + QUESTIONS[i] for i in range(2)]
            with ThreadPoolExecutor(2) as pool:
                pair = list(pool.map(lambda d: chat_stream(args.base, args.model, [{'role': 'user', 'content': d}],
                                                           max_tokens=args.c2_max_tokens), docs))
            lo = max(x['first_at'] for x in pair); hi = min(x['last_at'] for x in pair)
            # Tokens per content chunk, per request, applied to chunks inside the both-decoding window.
            both = sum(sum(1 for t in x['chunks'] if lo < t <= hi) * x['completion_tokens'] / max(1, len(x['chunks']))
                       for x in pair)
            emit(dict(kind='c2', n=n, per_request_tok_s=[x['decode_tok_s'] for x in pair], ttft_s=[x['ttft_s'] for x in pair],
                      overlap_s=hi-lo, overlap_total_tok_s=both/(hi-lo) if hi > lo else None))
    try:
        with urllib.request.urlopen(args.base+'/og/stats', timeout=5) as r:
            emit(dict(kind='og_stats', **json.load(r)))
    except Exception:
        pass


if __name__ == '__main__':
    main()
