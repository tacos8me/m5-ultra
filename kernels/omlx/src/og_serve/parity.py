"""OpenAI API parity suite for ds41 backends (reference q3 `ds41` vs candidate `ds41-og`).

Each case checks one behaviour a client relies on, with the same request on each
endpoint; the models differ numerically, so cases assert structure and task success,
not identical text. A case that passes on the reference and fails on the candidate is
a parity failure.

  python og_serve/parity.py --label sv-p0 --target q3=http://127.0.0.1:8080/v1@ds41 \
      [--target og=http://127.0.0.1:12161/v1@ds41-og] [--tier long] [--think-model ds41:think]

Tiers: base (default, ~3-5 min per backend), long (+128K needle, c2 at 8K), xl (+512K needle).
Results: <logs>/<label>.parity.jsonl, summary printed and in <label>.parity.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import socket
import time
import urllib.error
import urllib.request

HOME = Path.home()
LOGS = Path(os.environ.get('OG_LOGS', str(HOME/'llm/ds41/og-serve')))
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Current weather for a city',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'City name'}},
                       'required': ['city']}}},
    {'type': 'function', 'function': {
        'name': 'calculator', 'description': 'Evaluate an arithmetic expression exactly',
        'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}},
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': 'Read a text file from the workspace',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
]
FILES = {'notes/todo.txt': 'buy milk\nfix the build\ncall Ana at 5pm\n', 'data/numbers.txt': '17\n25\n58\n'}


class Target:
    def __init__(self, spec):
        name, rest = spec.split('=', 1)
        self.base, self.model = rest.rsplit('@', 1)
        self.name = name

    def post(self, path, body, timeout=1800):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=timeout)

    def chat(self, messages, **kw):
        body = dict(model=kw.pop('model', self.model), messages=messages, **kw)
        t0 = time.perf_counter()
        with self.post('/chat/completions', body) as r:
            d = json.load(r)
        d['_s'] = time.perf_counter() - t0
        return d

    def stream(self, messages, stop_after=None, path='/chat/completions', **kw):
        body = dict(model=kw.pop('model', self.model), stream=True, stream_options={'include_usage': True}, **kw)
        if messages is not None:
            body['messages'] = messages
        out = dict(content='', reasoning='', text='', tool_calls={}, ids=set(), roles=0, finish=None, usage=None,
                   done=False, errors=[], chunks=0, keepalives=0, first=None, last=None, backend=None, lines_ok=True)
        t0 = time.perf_counter()
        with self.post(path, body) as r:
            out['backend'] = r.headers.get('x-ds41-og-backend')
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith(b'data: '):
                    out['lines_ok'] = out['lines_ok'] and line.startswith(b':')
                    continue
                if line == b'data: [DONE]':
                    out['done'] = True
                    continue
                d = json.loads(line[6:])
                if 'error' in d:
                    out['errors'].append(d['error'])
                    continue
                if d.get('model') == 'keepalive' or (d.get('created') == 0 and [c.get('delta') for c in d.get('choices') or []]
                                                 == [{'role': 'assistant', 'content': ''}]):
                    out['keepalives'] += 1
                    continue
                out['chunks'] += 1
                out['ids'].add(d.get('id'))
                out['usage'] = d.get('usage') or out['usage']
                for c in d.get('choices') or []:
                    delta = c.get('delta') or {}
                    out['roles'] += 'role' in delta and bool(delta['role'])
                    piece = (delta.get('content') or '') + (delta.get('reasoning_content') or '') + (c.get('text') or '')
                    if piece or delta.get('tool_calls'):
                        now = time.perf_counter()
                        out['first'] = out['first'] or now - t0
                        out['last'] = now - t0
                    out['content'] += delta.get('content') or ''
                    out['reasoning'] += delta.get('reasoning_content') or ''
                    out['text'] += c.get('text') or ''
                    for tc in delta.get('tool_calls') or []:
                        entry = out['tool_calls'].setdefault(tc.get('index', 0), dict(id=None, name='', arguments=''))
                        entry['id'] = entry['id'] or tc.get('id')
                        fn = tc.get('function') or {}
                        entry['name'] += fn.get('name') or ''
                        entry['arguments'] += fn.get('arguments') or ''
                    out['finish'] = c.get('finish_reason') or out['finish']
                if stop_after is not None and out['chunks'] >= stop_after:
                    break
        out['ids'] = sorted(i for i in out['ids'] if i)
        out['tool_calls'] = [out['tool_calls'][k] for k in sorted(out['tool_calls'])]
        return out


def expect(cond, message):
    if not cond:
        raise AssertionError(message)


def usage_ok(u, max_tokens=None):
    expect(u and u.get('prompt_tokens', 0) > 0 and u.get('completion_tokens', -1) >= 0, f'usage missing: {u}')
    expect(u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'], f'usage total inconsistent: {u}')
    if max_tokens is not None:
        expect(u['completion_tokens'] <= max_tokens, f'completion_tokens {u["completion_tokens"]} > max_tokens {max_tokens}')


def args_json(call):
    try:
        return json.loads(call['arguments'] if isinstance(call, dict) and 'arguments' in call else call['function']['arguments'])
    except (ValueError, KeyError, TypeError):
        raise AssertionError(f'tool arguments are not JSON: {call}')


# ----------------------------------------------------------------------------------------------- cases
def c_models(t):
    with urllib.request.urlopen(t.base + '/models', timeout=30) as r:
        ids = [m['id'] for m in json.load(r)['data']]
    expect(ids, 'no models listed')
    return dict(ids=ids[:12])


def c_nonstream(t):
    d = t.chat([{'role': 'user', 'content': 'Reply with exactly the word: ready'}], max_tokens=16, temperature=0)
    msg = d['choices'][0]['message']
    expect(msg['role'] == 'assistant' and 'ready' in (msg.get('content') or '').lower(), f'content: {msg}')
    expect(d['choices'][0]['finish_reason'] in ('stop', 'length'), d['choices'][0]['finish_reason'])
    usage_ok(d.get('usage'), 16)
    return dict(content=msg['content'], usage=d['usage'], s=round(d['_s'], 2))


def c_stream(t):
    r = t.stream([{'role': 'user', 'content': 'Write a Python function that merges overlapping intervals.'}],
                 max_tokens=200, temperature=0)
    expect(r['done'] and not r['errors'], f'errors {r["errors"]} done {r["done"]}')
    expect(r['roles'] == 1, f'role chunks: {r["roles"]}')
    expect(len(r['ids']) == 1, f'chunk ids: {r["ids"]}')
    expect(r['content'].strip() and r['finish'] in ('stop', 'length'), f'finish {r["finish"]}')
    usage_ok(r['usage'], 200)
    expect(r['lines_ok'], 'non-SSE line in stream')
    return dict(chars=len(r['content']), finish=r['finish'], usage=r['usage'], ttft=r['first'], backend=r['backend'])


def c_deterministic(t):
    msgs = [{'role': 'user', 'content': 'List the first ten prime numbers, comma separated, then one sentence about them.'}]
    a = t.stream(msgs, max_tokens=80, temperature=0)
    b = t.stream(msgs, max_tokens=80, temperature=0)
    expect(a['content'] == b['content'], f'temperature 0 differs:\n{a["content"]!r}\n{b["content"]!r}')
    return dict(chars=len(a['content']))


def c_max_tokens(t):
    r = t.stream([{'role': 'user', 'content': 'Write a long essay about the history of the printing press.'}],
                 max_tokens=7, temperature=0)
    expect(r['finish'] == 'length', f'finish {r["finish"]}')
    expect(r['usage']['completion_tokens'] == 7, f'usage {r["usage"]}')
    d = t.chat([{'role': 'user', 'content': 'Write a long essay about the history of the printing press.'}],
               max_tokens=7, temperature=0)
    expect(d['choices'][0]['finish_reason'] == 'length' and d['usage']['completion_tokens'] == 7, f'{d["usage"]}')
    return dict(stream=r['usage'], nonstream=d['usage'])


def c_stop(t):
    msgs = [{'role': 'user', 'content': 'Count from 1 to 20 separated by commas and spaces. Output only the numbers.'}]
    r = t.stream(msgs, max_tokens=100, temperature=0, stop=['9'])
    expect(r['finish'] == 'stop', f'finish {r["finish"]}')
    expect('9' not in r['content'] and '8' in r['content'], f'content {r["content"]!r}')
    d = t.chat(msgs, max_tokens=100, temperature=0, stop=['9', 'zzz'])
    content = d['choices'][0]['message']['content'] or ''
    expect(d['choices'][0]['finish_reason'] == 'stop' and '9' not in content, f'{content!r}')
    return dict(stream=r['content'][-24:], nonstream=content[-24:])


def c_tool_call(t):
    msgs = [{'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'}]
    r = t.stream(msgs, tools=TOOLS, max_tokens=300, temperature=0)
    expect(r['finish'] == 'tool_calls' and r['tool_calls'], f'finish {r["finish"]} content {r["content"][:200]!r}')
    call = r['tool_calls'][0]
    expect(call['name'] == 'get_weather' and call['id'], f'call {call}')
    expect('paris' in args_json(call).get('city', '').lower(), f'args {call["arguments"]}')
    d = t.chat(msgs, tools=TOOLS, max_tokens=300, temperature=0)
    calls = d['choices'][0]['message'].get('tool_calls') or []
    expect(d['choices'][0]['finish_reason'] == 'tool_calls' and calls
           and calls[0]['function']['name'] == 'get_weather', f'{d["choices"][0]}')
    args_json(calls[0])
    return dict(stream=call, nonstream=calls[0]['function'])


def c_tool_choice_none(t):
    d = t.chat([{'role': 'user', 'content': 'Say hello in French. Do not use tools.'}], tools=TOOLS, tool_choice='none',
               max_tokens=40, temperature=0)
    msg = d['choices'][0]['message']
    expect(not msg.get('tool_calls') and (msg.get('content') or '').strip(), f'{msg}')
    return dict(content=msg['content'])


def run_tool(name, args):
    if name == 'get_weather':
        return json.dumps({'city': args.get('city'), 'temp_c': 18, 'sky': 'clear'})
    if name == 'calculator':
        expr = str(args.get('expression', ''))
        if not set(expr) <= set('0123456789+-*/(). '):
            return 'error: unsupported expression'
        return str(eval(expr))  # noqa: S307 -- digits and operators only
    if name == 'read_file':
        return FILES.get(args.get('path', ''), 'error: no such file')
    return 'error: unknown tool'


def agent_loop(t, task, stream=True, steps=6, think=False):
    """Hermes-style loop: call, execute tools, append results, repeat until a plain answer."""
    msgs = [{'role': 'system', 'content': 'You are a careful agent. Use the tools when they help; answer concisely.'},
            {'role': 'user', 'content': task}]
    calls_made = []
    extra = dict(chat_template_kwargs={'enable_thinking': True}) if think else {}
    for step in range(steps):
        if stream:
            r = t.stream(msgs, tools=TOOLS, max_tokens=1200, temperature=0, **extra)
            expect(r['done'] and not r['errors'], f'step {step}: {r["errors"]}')
            calls = [dict(id=c['id'], type='function', function=dict(name=c['name'], arguments=c['arguments']))
                     for c in r['tool_calls']]
            content, finish = r['content'], r['finish']
        else:
            d = t.chat(msgs, tools=TOOLS, max_tokens=1200, temperature=0, **extra)
            msg = d['choices'][0]['message']
            calls, content, finish = msg.get('tool_calls') or [], msg.get('content') or '', d['choices'][0]['finish_reason']
        if not calls:
            expect(finish == 'stop', f'step {step}: finish {finish}')
            return dict(answer=content, calls=calls_made, steps=step + 1)
        expect(finish == 'tool_calls', f'step {step}: finish {finish} with {len(calls)} calls')
        msgs.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
        for call in calls:
            name, args = call['function']['name'], args_json(call['function'])
            calls_made.append(name)
            msgs.append({'role': 'tool', 'tool_call_id': call['id'], 'content': run_tool(name, args)})
    raise AssertionError(f'no final answer after {steps} steps: {calls_made}')


def c_agent_loop(t):
    out = agent_loop(t, 'Read data/numbers.txt, add up the numbers with the calculator, and tell me the total.')
    expect('100' in out['answer'].replace(',', ''), f'answer {out["answer"]!r} calls {out["calls"]}')
    expect('read_file' in out['calls'], f'calls {out["calls"]}')
    return out


def c_agent_loop_nonstream(t):
    out = agent_loop(t, 'What is the weather in Tokyo and in Oslo? Then tell me which is warmer.', stream=False)
    expect(out['calls'].count('get_weather') >= 2, f'calls {out["calls"]}')
    return out


def c_think(t, think_model=None):
    msgs = [{'role': 'user', 'content': 'A train leaves at 14:40 and arrives at 17:05. How long is the trip?'}]
    r = t.stream(msgs, max_tokens=1500, temperature=0, chat_template_kwargs={'enable_thinking': True})
    expect(r['reasoning'].strip() and r['content'].strip(), f'reasoning {len(r["reasoning"])} content {r["content"][:80]!r}')
    expect('2' in r['content'] and '25' in r['content'], f'content {r["content"]!r}')
    out = dict(reasoning_chars=len(r['reasoning']), content=r['content'][-80:])
    if think_model:
        r2 = t.stream(msgs, max_tokens=1500, temperature=0, model=think_model)
        expect(r2['reasoning'].strip() and r2['content'].strip(), f':think model gave no reasoning ({think_model})')
        out['think_model_reasoning_chars'] = len(r2['reasoning'])
    r3 = t.stream(msgs, max_tokens=200, temperature=0)
    expect(not r3['reasoning'].strip(), 'reasoning without enable_thinking')
    return out


def c_agent_think(t):
    out = agent_loop(t, 'Read notes/todo.txt and tell me what time I must call Ana.', think=True)
    expect('5' in out['answer'], f'answer {out["answer"]!r}')
    return out


def c_multiturn(t):
    msgs = [{'role': 'user', 'content': 'My name is Priya and my favourite number is 42. Acknowledge briefly.'}]
    d = t.chat(msgs, max_tokens=60, temperature=0)
    msgs += [{'role': 'assistant', 'content': d['choices'][0]['message']['content']},
             {'role': 'user', 'content': 'What is my name and my favourite number?'}]
    d2 = t.chat(msgs, max_tokens=60, temperature=0)
    content = d2['choices'][0]['message']['content']
    expect('Priya' in content and '42' in content, f'{content!r}')
    return dict(content=content)


def c_unicode(t):
    r = t.stream([{'role': 'user', 'content': 'Translate to Chinese and Japanese: "good morning, friend". '
                   'Then add one emoji.'}], max_tokens=80, temperature=0)
    expect('�' not in r['content'], f'replacement character in {r["content"]!r}')
    expect(any(ord(ch) > 0x3000 for ch in r['content']), f'no CJK in {r["content"]!r}')
    return dict(content=r['content'])


def c_json_mode(t):
    d = t.chat([{'role': 'user', 'content': 'Give a JSON object with keys "city" and "country" for the Eiffel Tower.'}],
               max_tokens=80, temperature=0, response_format={'type': 'json_object'})
    obj = json.loads(d['choices'][0]['message']['content'])
    expect(isinstance(obj, dict) and 'city' in obj, f'{obj}')
    return dict(obj=obj)


def c_completions(t):
    r = t.stream(None, path='/completions', prompt='The capital of France is', max_tokens=8, temperature=0)
    expect(r['done'] and 'Paris' in r['text'], f'text {r["text"]!r}')
    return dict(text=r['text'])


def c_bad_request(t):
    try:
        t.post('/chat/completions', dict(model=t.model, messages='not a list'), timeout=60).read()
    except urllib.error.HTTPError as e:
        expect(400 <= e.code < 500, f'status {e.code}')
        return dict(status=e.code)
    raise AssertionError('invalid request accepted')


def c_cancel(t):
    """Disconnect mid-stream; the server must abort that generation and stay fast."""
    body = dict(model=t.model, stream=True, max_tokens=3000, temperature=0,
                messages=[{'role': 'user', 'content': 'Write a very long story about a lighthouse keeper.'}])
    host, port = t.base.split('//', 1)[1].split('/', 1)[0].split(':')
    sock = socket.create_connection((host, int(port)), timeout=60)
    payload = json.dumps(body).encode()
    path = '/' + t.base.split('//', 1)[1].split('/', 1)[1] + '/chat/completions'
    sock.sendall(f'POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n'
                 f'Content-Length: {len(payload)}\r\n\r\n'.encode() + payload)
    got, t0 = b'', time.time()
    while got.count(b'data: ') < 25 and time.time() - t0 < 300:
        chunk = sock.recv(65536)
        if not chunk:
            break
        got += chunk
    sock.close()
    expect(got.count(b'data: ') >= 25, 'stream did not start')
    time.sleep(2)
    r = t.stream([{'role': 'user', 'content': 'Reply with the word: pong'}], max_tokens=8, temperature=0)
    expect(r['done'] and r['first'] is not None and r['first'] < 20, f'next request after cancel: ttft {r["first"]}')
    return dict(next_ttft=r['first'])


def c_c2(t):
    msgs = [[{'role': 'user', 'content': q}] for q in (
        'Explain how TCP congestion control works, in detail.',
        'Write a detailed guide to sourdough bread baking.')]
    with ThreadPoolExecutor(2) as pool:
        pair = list(pool.map(lambda m: t.stream(m, max_tokens=300, temperature=0), msgs))
    for r in pair:
        expect(r['done'] and not r['errors'] and r['usage']['completion_tokens'] > 50, f'{r["errors"]} {r["usage"]}')
    lo = max(r['first'] for r in pair)
    hi = min(r['last'] for r in pair)
    return dict(ttft=[round(r['first'], 2) for r in pair], tokens=[r['usage']['completion_tokens'] for r in pair],
                overlap_s=round(hi - lo, 2))


def needle_doc(n_tokens, seed):
    rng = random.Random(seed)
    words = ('river stone lantern orchard copper meadow signal harbor quiet ember violet summit thread canyon '
             'pepper marble falcon willow cobalt ribbon anchor maple glacier').split()
    code = rng.randint(100000, 999999)
    # ~1.09 tokens per word for this vocabulary (measured: 92160 words -> 100356 prompt tokens)
    total = int(n_tokens / 1.09)
    pos = rng.randint(total // 4, 3 * total // 4)
    parts = []
    for i in range(total):
        if i == pos:
            parts.append(f'\nThe secret vault code is {code}.\n')
        parts.append(rng.choice(words))
    return ' '.join(parts), code


def c_needle(t, n):
    doc, code = needle_doc(n, n)
    msgs = [{'role': 'user', 'content': f'[doc {time.time_ns()}]\n{doc}\n\nWhat is the secret vault code? Reply with the number only.'}]
    r = t.stream(msgs, max_tokens=24, temperature=0)
    expect(str(code) in r['content'], f'expected {code}, got {r["content"]!r}')
    return dict(ttft=round(r['first'], 1), prompt_tokens=r['usage']['prompt_tokens'])


BASE = ['models', 'nonstream', 'stream', 'deterministic', 'max_tokens', 'stop', 'tool_call', 'tool_choice_none',
        'agent_loop', 'agent_loop_nonstream', 'think', 'agent_think', 'multiturn', 'unicode', 'json_mode',
        'completions', 'bad_request', 'cancel', 'c2']
LONG = ['needle_128k']
XL = ['needle_512k']


def run_case(t, name, think_model):
    fn = {'needle_128k': lambda x: c_needle(x, 128000), 'needle_512k': lambda x: c_needle(x, 500000),
          'think': lambda x: c_think(x, think_model)}.get(name) or globals()['c_' + name]
    t0 = time.perf_counter()
    try:
        detail = fn(t)
        ok, err = True, None
    except Exception as exc:  # noqa: BLE001 -- recorded per case
        detail, ok, err = None, False, f'{type(exc).__name__}: {exc}'[:600]
    return dict(target=t.name, case=name, ok=ok, error=err, detail=detail, s=round(time.perf_counter() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--target', action='append', required=True, help='name=http://host:port/v1@model')
    ap.add_argument('--tier', default='base', choices=['base', 'long', 'xl'])
    ap.add_argument('--cases', default='')
    ap.add_argument('--think-model', action='append', default=[], help='name=model for the :think alias check')
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    cases = [c for c in args.cases.split(',') if c] or BASE + (LONG if args.tier != 'base' else []) + (XL if args.tier == 'xl' else [])
    think = dict(x.split('=', 1) for x in args.think_model)
    out = (LOGS/f'{args.label}.parity.jsonl').open('a')
    results = []
    for spec in args.target:
        t = Target(spec)
        for name in cases:
            rec = run_case(t, name, think.get(t.name))
            results.append(rec)
            out.write(json.dumps(rec, default=str) + '\n'); out.flush()
            print(f'{t.name:4} {name:22} {"PASS" if rec["ok"] else "FAIL"} {rec["s"]:7.1f}s  '
                  f'{rec["error"] or json.dumps(rec["detail"], default=str)[:150]}', flush=True)
    names = [Target(s).name for s in args.target]
    table = {c: {r['target']: r['ok'] for r in results if r['case'] == c} for c in cases}
    summary = dict(label=args.label, targets=names, cases=table,
                   failures={n: [c for c in cases if table[c].get(n) is False] for n in names})
    if len(names) >= 2:
        ref = names[0]
        summary['parity_failures'] = {n: [c for c in cases if table[c].get(ref) and not table[c].get(n)] for n in names[1:]}
    (LOGS/f'{args.label}.parity.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
