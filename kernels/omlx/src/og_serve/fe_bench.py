"""Front-end latency bench + output-identity suite for a ds41 endpoint (streaming, temperature 0).

  ttft     fresh N-token prompt (unique nonce), 16 tokens out: time to the first content chunk
  resume   one conversation per N (fixed doc + question, turn 1 with 32 tokens out, answered once
           per process), then per rep a new turn 2 = turn 1 + answer + a question with a unique nonce
           (~60-80 new tokens): turn-2 TTFT. The box and Mac prefix caches hold turn 1, as in a
           real conversation.
  identity fixed prompts over every API shape (chat, thinking, tools, multi-turn, Anthropic
           /v1/messages, /v1/responses, an image, a 20K multi-message conversation + its turn 2, an
           8K resume): the full streamed output of each, for a byte comparison between two servers.

  python og_serve/fe_bench.py --base http://127.0.0.1:8080 --model ds41 --label prod \
      --ttft 8192:3 --resume 8192:3,131072:3,524288:2 --identity
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
import urllib.request
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HOME = Path.home()
LOGS = Path(os.environ.get('FE_LOGS', str(HOME/'llm/ds41/fe')))
QUESTIONS = ['Summarize the code above in five bullet points.', 'Explain how the Engram tables are used, in detail.',
             'List the main classes and what each one does.', 'Describe the prefill path step by step.']


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800)


def stream(base, path, body):
    """Stream one request; returns ttft (first generated text/tool/reasoning event) and the full output."""
    body = dict(body, stream=True)
    if path == '/v1/chat/completions':
        body['stream_options'] = {'include_usage': True}
    t0 = time.perf_counter()
    first = None
    out = dict(content='', reasoning='', tools='', events=0, usage=None, t0=time.time())
    with post(base, path, body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                continue
            d = json.loads(payload)
            if isinstance(d, dict) and d.get('error'):
                raise RuntimeError(d['error'])
            got = ''
            if path == '/v1/chat/completions':
                out['usage'] = d.get('usage') or out['usage']
                for choice in d.get('choices') or []:
                    delta = choice.get('delta') or {}
                    out['content'] += delta.get('content') or ''
                    out['reasoning'] += delta.get('reasoning_content') or ''
                    for call in delta.get('tool_calls') or []:
                        out['tools'] += json.dumps(call.get('function') or {}, sort_keys=True)
                    got = (delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'))
            elif path == '/v1/messages':
                kind = d.get('type')
                if kind == 'content_block_delta':
                    delta = d.get('delta') or {}
                    text = delta.get('text') or delta.get('thinking') or delta.get('partial_json') or ''
                    out['content' if delta.get('type') == 'text_delta' else 'reasoning'] += text
                    got = text
                elif kind == 'message_delta':
                    out['usage'] = d.get('usage')
            elif path == '/v1/responses':
                kind = d.get('type', '')
                if kind.endswith('.delta'):
                    text = d.get('delta') or ''
                    out['reasoning' if 'reasoning' in kind else 'content'] += text
                    got = text
                elif kind == 'response.completed':
                    out['usage'] = (d.get('response') or {}).get('usage')
            if got and first is None:
                first = time.perf_counter()
            out['events'] += 1
    out['ttft_s'] = None if first is None else first - t0
    out['total_s'] = time.perf_counter() - t0
    return out


def document(tok, n, nonce=''):
    # A fixed corpus (the production tree at f56f7ffa) so every server sees the same prompts.
    src = Path(os.environ.get('FE_DOC_TREE', str(HOME/'src/wt/ds41-og')))/'omlx/patches/deepseek_v41'
    corpus = '\n\n'.join(f'File: {f.name}\n{f.read_text()}' for f in sorted(src.glob('*.py')))
    for s in tok.all_special_tokens:
        corpus = corpus.replace(s, '[special token literal]')
    corpus = corpus.replace('｜', '|').replace('<think>', '[think tag]').replace('</think>', '[/think tag]')
    base = tok.encode(corpus, add_special_tokens=False)
    return nonce + tok.decode((base * (n // len(base) + 2))[:n])


def png(width=64, height=48, rgb=(220, 40, 40)):
    raw = b''.join(b'\x00' + bytes(rgb) * width for _ in range(height))

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    data = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def identity_cases(model, tok):
    import fe_conv
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
              'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    chat = '/v1/chat/completions'
    conv, conv_tools = fe_conv.conversation(4242, 60_000, tools=True, reasoning=True)
    doc = document(tok, 8192, '[identity doc]\n')
    cases = [
        ('chat', chat, dict(messages=[{'role': 'user', 'content': 'Explain TCP slow start in two sentences.'}],
                            max_tokens=48)),
        ('system_multiturn', chat, dict(messages=[
            {'role': 'system', 'content': 'You are terse. 中文也可以。'},
            {'role': 'user', 'content': 'Name three prime numbers.'},
            {'role': 'assistant', 'content': '2, 3, 5.'},
            {'role': 'user', 'content': 'Now three more, larger than 100.'}], max_tokens=48)),
        ('thinking', chat, dict(messages=[{'role': 'user', 'content': 'What is 17*23? Think briefly.'}],
                                max_tokens=64, chat_template_kwargs={'enable_thinking': True})),
        ('tools', chat, dict(messages=[{'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'}],
                             tools=tools, max_tokens=64)),
        ('tool_result', chat, dict(messages=[
            {'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"temp_c": 18, "sky": "clear"}'}],
            tools=tools, max_tokens=48)),
        ('anthropic', '/v1/messages', dict(system='Answer in one sentence.', max_tokens=48, messages=[
            {'role': 'user', 'content': 'What is a hash map?'},
            {'role': 'assistant', 'content': 'A key-value table with O(1) average lookups.'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'And a B-tree?'}]}])),
        ('responses', '/v1/responses', dict(max_output_tokens=48, input=[
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Give two uses of a Bloom filter.'}]}])),
        ('image', chat, dict(max_tokens=32, messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What color is this image? One word.'},
            {'type': 'image_url', 'image_url': {'url': png()}}]}])),
        ('conv60k', chat, dict(messages=conv, tools=conv_tools, max_tokens=48)),
        ('conv60k_turn2', chat, dict(messages=fe_conv.next_turn(conv, 5), tools=conv_tools, max_tokens=48)),
        ('doc8k', chat, dict(messages=[{'role': 'user', 'content': doc + '\n\n' + QUESTIONS[0]}], max_tokens=48)),
        ('doc8k_turn2', chat, dict(messages=[{'role': 'user', 'content': doc + '\n\n' + QUESTIONS[0]},
                                             {'role': 'assistant', 'content': 'It is the DS41 model code.'},
                                             {'role': 'user', 'content': QUESTIONS[1]}], max_tokens=48)),
    ]
    return [(name, path, dict(body, model=model, temperature=0)) for name, path, body in cases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--ttft', default='')
    ap.add_argument('--resume', default='')
    ap.add_argument('--identity', action='store_true')
    ap.add_argument('--budget', type=float, default=780)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
    LOGS.mkdir(parents=True, exist_ok=True)
    out = (LOGS/f'{args.label}.jsonl').open('a')
    deadline = time.monotonic() + args.budget
    summary = dict(label=args.label, ttft={}, resume={})

    def emit(rec):
        rec = dict(t=time.time(), label=args.label, **rec)
        out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()
        print(json.dumps({k: (v[:80] if isinstance(v, str) else v) for k, v in rec.items()}, ensure_ascii=False)[:400],
              flush=True)

    def pairs(spec):
        return [(int(n), int(r or 1)) for n, _, r in (p.partition(':') for p in spec.split(',') if p)]

    chat = '/v1/chat/completions'
    for n, reps in pairs(args.ttft):
        vals = []
        for rep in range(reps):
            if time.monotonic() > deadline:
                break
            msgs = [{'role': 'user', 'content': document(tok, n, f'[run {time.time_ns()}]\n') + '\n\n' + QUESTIONS[rep % 4]}]
            r = stream(args.base, chat, dict(model=args.model, messages=msgs, max_tokens=16, temperature=0))
            vals.append(r['ttft_s'])
            emit(dict(kind='ttft', n=n, rep=rep, ttft_s=r['ttft_s'], t0=r['t0'], usage=r['usage']))
        if vals:
            summary['ttft'][n] = dict(median=round(statistics.median(vals), 3), all=[round(v, 3) for v in vals])

    for n, reps in pairs(args.resume):
        if time.monotonic() > deadline:
            break
        msgs = [{'role': 'user', 'content': document(tok, n, f'[conversation {n}]\n') + '\n\n' + QUESTIONS[0]}]
        r1 = stream(args.base, chat, dict(model=args.model, messages=msgs, max_tokens=32, temperature=0))
        emit(dict(kind='turn1', n=n, ttft_s=r1['ttft_s'], total_s=r1['total_s'], usage=r1['usage']))
        vals = []
        for rep in range(reps):
            if time.monotonic() > deadline:
                break
            turn2 = msgs + [{'role': 'assistant', 'content': r1['content']},
                            {'role': 'user', 'content': f'(question {time.time_ns()}) ' + QUESTIONS[1 + rep % 3]}]
            r2 = stream(args.base, chat, dict(model=args.model, messages=turn2, max_tokens=16, temperature=0))
            vals.append(r2['ttft_s'])
            emit(dict(kind='resume', n=n, rep=rep, ttft_s=r2['ttft_s'], t0=r2['t0'], usage=r2['usage']))
        if vals:
            summary['resume'][n] = dict(median=round(statistics.median(vals), 3), all=[round(v, 3) for v in vals])

    if args.identity:
        for name, path, body in identity_cases(args.model, tok):
            try:
                r = stream(args.base, path, body)
                emit(dict(kind='identity', case=name, ttft_s=r['ttft_s'], t0=r['t0'], content=r['content'], reasoning=r['reasoning'],
                          tools=r['tools'], usage=r['usage']))
            except Exception as exc:  # noqa: BLE001 -- recorded; the comparison flags it
                emit(dict(kind='identity', case=name, error=repr(exc)[:300]))
    (LOGS/f'{args.label}.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
