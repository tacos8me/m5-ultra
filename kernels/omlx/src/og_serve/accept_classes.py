"""Per-class DSpark acceptance through the served ds41 (temperature 0, c1, prompts >= 1024 tokens).

Gate for the MXFP8 draft head (THREADS-2026-09-30 4.2): run once with DS41_DRAFT_HEAD unset and once with
DS41_DRAFT_HEAD=mxfp8 on the same tree, then compare; stop if tok/cycle drops by more than 1% in any class.

  python og_serve/accept_classes.py run LABEL [--base http://127.0.0.1:8080] [--model ds41] [--max-tokens 384]
  python og_serve/accept_classes.py compare BASE_LABEL VARIANT_LABEL [--max-drop 0.01]
  python og_serve/accept_classes.py prompts      # freeze the prompt set, print token counts (no requests)

Classes, 4 prompts each (built once from this machine's stdlib and this tree's docs, frozen into
~/llm/ds41/next4/accept-prompts.json; its sha256 is recorded and compared):
  code   a stdlib module + an explanation question
  prose  an English doc of this tree (code, tables, links stripped) + a flowing-prose essay/story request
  sum8k  ~8K tokens of English docs + a five-bullet summary request
  tool   a stdlib module + write_file/run_tests tools; the answer is a tool call with a long argument
  cjk    README.zh/ja/ko of this tree + a request to answer in Chinese / Japanese / Korean
Requests are non-streaming and sequential. Each one's og worker summary line (MTP[..] tokens, cycles, accept,
copy) is read from og-child.log: the lines appended while it ran must hold exactly one MTP line, else the
request is retried once (other traffic), then marked contaminated and left out of the sums.
Per class: tok/cycle = sum(tokens) / sum(cycles); dspark accept = sum(accepted) / sum(proposed).
compare: every prompt's output (content + tool calls) must be byte-identical and every prompt >= 1024
tokens; exit 1 if a class's tok/cycle ratio VARIANT/BASE < 1 - max_drop or identity fails.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
TREE = HERE.parent
OUT = Path(os.environ.get('ACCEPT_OUT', str(Path.home()/'llm/ds41/next4')))
PROMPTS = OUT/'accept-prompts.json'
CHILD_LOG = Path(os.environ.get('DS41_OG_LOGS', str(Path.home()/'llm/ds41/og/logs')))/'og-child.log'
MTP = re.compile(r'MTP\[\d+\] finish=(\S+) tokens=(\d+) cycles=(\d+) tok/cycle=[\d.]+ accept=(\d+)/(\d+)')
COPY = re.compile(r'copy\[cycles=(\d+) accept=(\d+)/(\d+)\]')
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'write_file', 'description': 'Create or overwrite a file in the repository.',
        'parameters': {'type': 'object', 'required': ['path', 'content'], 'properties': {
            'path': {'type': 'string', 'description': 'Repository-relative path.'},
            'content': {'type': 'string', 'description': 'The complete file content.'}}}}},
    {'type': 'function', 'function': {
        'name': 'run_tests', 'description': 'Run pytest on one test file and return the report.',
        'parameters': {'type': 'object', 'required': ['path'], 'properties': {'path': {'type': 'string'}}}}},
]
CODE_Q = ['Explain in detail how this module works: its purpose, the main functions and how they interact.',
          'Walk through the most important function in this module step by step and explain its edge cases.',
          'Describe the data structures this module relies on and why they were chosen.',
          'Review this module: what could go wrong in production use, and how would you harden it?']
PROSE_Q = ['Write a reflective essay in flowing prose (no lists, no code, no headings) about the ideas in the text above.',
           'Retell the text above as a short story told by an engineer on a night shift. Prose only, no lists or code.',
           'Write a letter to a colleague explaining, in plain flowing prose, why the design above matters. No lists.',
           'Write a thoughtful magazine-style column about the text above, in continuous prose without lists.']
CJK_Q = [('README.zh.md', '请用中文详细介绍上面文档描述的项目：它解决什么问题、主要功能有哪些、以及如何开始使用。'),
         ('README.ja.md', '上の文書が説明しているプロジェクトについて、目的、主な機能、使い方を日本語で詳しく説明してください。'),
         ('README.ko.md', '위 문서가 설명하는 프로젝트의 목적, 주요 기능, 사용 방법을 한국어로 자세히 설명해 주세요.'),
         ('README.zh.md', '请用中文写一篇关于上述项目设计理念的评论文章，使用连贯的段落，不要使用列表。')]
CODE_FILES = ['heapq.py', 'json/decoder.py', 'contextlib.py', 'fnmatch.py']
TOOL_FILES = ['calendar.py', 'base64.py', 'glob.py', 'textwrap.py']
DOCS = ['docs/MoE_Expert_Offload.md', 'docs/heterogeneous-cluster.md', 'docs/TESTING.md', 'docs/distributed-cluster.md',
        'README.md', 'docs/oQ_Quantization.md', 'docs/usage-analytics.md', 'docs/rdma-links.md']


def prose_of(markdown):
    """Markdown without code fences, tables, images, links' targets and HTML."""
    text = re.sub(r'```.*?```', '', markdown, flags=re.S)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith('|')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def build():
    import textwrap  # the stdlib directory of this interpreter
    lib = Path(textwrap.__file__).parent
    items = []
    for i, name in enumerate(CODE_FILES):
        src = (lib/name).read_text()[:16000]
        items.append(dict(cls='code', name=name, messages=[{'role': 'user', 'content':
                     f'File: {name}\n```python\n{src}\n```\n\n{CODE_Q[i]}'}]))
    for i, doc in enumerate(DOCS[:4]):
        items.append(dict(cls='prose', name=doc, messages=[{'role': 'user', 'content':
                     prose_of((TREE/doc).read_text())[:14000] + '\n\n' + PROSE_Q[i]}]))
    docs = [prose_of((TREE/d).read_text()) for d in DOCS]
    for i in range(4):
        order = docs[i:] + docs[:i]
        items.append(dict(cls='sum8k', name=f'docs@{i}', messages=[{'role': 'user', 'content':
                     '\n\n---\n\n'.join(order)[:32000] + '\n\nSummarize the documents above in five bullet points.'}]))
    for name in TOOL_FILES:
        src = (lib/name).read_text()[:14000]
        items.append(dict(cls='tool', name=name, tools=TOOLS, messages=[{'role': 'user', 'content':
                     f'File: {name}\n```python\n{src}\n```\n\nWrite a thorough pytest module for the code above and '
                     f'save it with the write_file tool as tests/test_{Path(name).stem}.py.'}]))
    for i, (doc, question) in enumerate(CJK_Q):
        items.append(dict(cls='cjk', name=f'{doc}#{i}', messages=[{'role': 'user', 'content':
                     prose_of((TREE/doc).read_text())[:9000] + '\n\n' + question}]))
    return items


def prompts():
    if not PROMPTS.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        PROMPTS.write_text(json.dumps(build(), ensure_ascii=False))
    return json.loads(PROMPTS.read_text()), hashlib.sha256(PROMPTS.read_bytes()).hexdigest()[:16]


def chat(base, model, item, max_tokens):
    body = dict(model=model, messages=item['messages'], max_tokens=max_tokens, temperature=0)
    if item.get('tools'):
        body['tools'] = item['tools']
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    msg = d['choices'][0]['message']
    text = (msg.get('reasoning_content') or '') + '\x00' + (msg.get('content') or '') + '\x00' + json.dumps(
        msg.get('tool_calls') or [], sort_keys=True, ensure_ascii=False)
    return text, d.get('usage') or {}


def mtp_lines(offset):
    """MTP summary lines appended to og-child.log after byte offset."""
    with CHILD_LOG.open('rb') as f:
        f.seek(offset)
        lines = [line.decode(errors='replace') for line in f]
    return [line for line in lines if MTP.search(line)]


def measured(base, model, item, max_tokens):
    for attempt in range(2):
        offset = CHILD_LOG.stat().st_size
        t = time.perf_counter()
        text, usage = chat(base, model, item, max_tokens)
        wall = time.perf_counter() - t
        deadline = time.monotonic() + 5
        lines = mtp_lines(offset)
        while len(lines) < 1 and time.monotonic() < deadline:
            time.sleep(0.2)
            lines = mtp_lines(offset)
        time.sleep(0.3)  # a concurrent request's line would land now
        lines = mtp_lines(offset)
        if len(lines) == 1:
            m = MTP.search(lines[0])
            c = COPY.search(lines[0])
            return dict(cls=item['cls'], name=item['name'], sha=hashlib.sha256(text.encode()).hexdigest()[:16],
                        prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                        finish=m.group(1), tokens=int(m.group(2)), cycles=int(m.group(3)),
                        accepted=int(m.group(4)), proposed=int(m.group(5)),
                        copy_cycles=int(c.group(1)) if c else 0, wall_s=round(wall, 3), text=text)
    return dict(cls=item['cls'], name=item['name'], contaminated=len(lines),
                sha=hashlib.sha256(text.encode()).hexdigest()[:16], prompt_tokens=usage.get('prompt_tokens'), text=text)


def summarize(records):
    out = {}
    for cls in dict.fromkeys(r['cls'] for r in records):
        rs = [r for r in records if r['cls'] == cls and 'cycles' in r]
        tokens, cycles = sum(r['tokens'] for r in rs), sum(r['cycles'] for r in rs)
        acc, prop = sum(r['accepted'] for r in rs), sum(r['proposed'] for r in rs)
        out[cls] = dict(n=len(rs), tokens=tokens, cycles=cycles, tok_per_cycle=round(tokens / cycles, 4) if cycles else None,
                        accept=round(acc / prop, 4) if prop else None, copy_cycles=sum(r['copy_cycles'] for r in rs),
                        min_prompt_tokens=min((r['prompt_tokens'] or 0) for r in records if r['cls'] == cls))
    return out


def show(args):
    items, sha = prompts()
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    except Exception as exc:  # noqa: BLE001 -- counts are optional
        tok = None
        print(json.dumps(dict(tokenizer_unavailable=repr(exc)[:200])))
    for item in items:
        text = '\n'.join(m['content'] for m in item['messages'])
        n = len(tok.encode(text).ids) if tok else None
        print(json.dumps(dict(cls=item['cls'], name=item['name'], chars=len(text), tokens=n,
                              ok=None if n is None else n >= 1100), ensure_ascii=False))
    names = [(i['cls'], i['name']) for i in items]
    assert len(set(names)) == len(names), 'prompt names must be unique per class'
    print(json.dumps(dict(file=str(PROMPTS), prompts_sha=sha, n=len(items))))
    return 0


def run(args):
    items, sha = prompts()
    base = args.base.rstrip('/')
    print(json.dumps(dict(event='start', label=args.label, prompts_sha=sha, n=len(items), log=str(CHILD_LOG))), flush=True)
    chat(base, args.model, dict(messages=[{'role': 'user', 'content': 'Say ready.'}]), 8)  # warm-up
    records = []
    for item in items:
        r = measured(base, args.model, item, args.max_tokens)
        records.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'text'}, ensure_ascii=False), flush=True)
    res = dict(label=args.label, prompts_sha=sha, max_tokens=args.max_tokens, time=time.time(), records=records,
               summary=summarize(records))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/f'accept-{args.label}.json'
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(dict(event='done', file=str(path), summary=res['summary'])), flush=True)
    return 0


def compare(args):
    base, var = (json.loads((OUT/f'accept-{label}.json').read_text()) for label in (args.base, args.variant))
    ok = base['prompts_sha'] == var['prompts_sha'] and base['max_tokens'] == var['max_tokens']
    if not ok:
        print(json.dumps(dict(error='different prompts or max_tokens')))
    by = {(r['cls'], r['name']): r for r in base['records']}
    diff = [f"{r['cls']}/{r['name']}" for r in var['records'] if by.get((r['cls'], r['name']), {}).get('sha') != r['sha']]
    short = [f"{r['cls']}/{r['name']}" for r in base['records'] + var['records'] if (r.get('prompt_tokens') or 0) < 1024]
    dirty = [f"{r['cls']}/{r['name']}" for r in base['records'] + var['records'] if 'cycles' not in r]
    print(json.dumps(dict(identical=not diff, differing=diff, under_1024=short, contaminated=dirty)))
    ok &= not diff and not short
    for cls, b in base['summary'].items():
        v = var['summary'].get(cls) or {}
        ratio = (v['tok_per_cycle'] / b['tok_per_cycle']) if b.get('tok_per_cycle') and v.get('tok_per_cycle') else None
        passed = ratio is not None and ratio >= 1 - args.max_drop
        ok &= passed
        print(json.dumps(dict(cls=cls, base=b['tok_per_cycle'], variant=v.get('tok_per_cycle'),
                              ratio=round(ratio, 4) if ratio else None, accept_base=b['accept'],
                              accept_variant=v.get('accept'), cycles=[b['cycles'], v.get('cycles')], pass_=passed)))
    print(json.dumps(dict(accept_gate_ok=ok, max_drop=args.max_drop)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('label')
    r.add_argument('--base', default='http://127.0.0.1:8080')
    r.add_argument('--model', default='ds41')
    r.add_argument('--max-tokens', type=int, default=384)
    c = sub.add_parser('compare')
    c.add_argument('base')
    c.add_argument('variant')
    c.add_argument('--max-drop', type=float, default=0.01)
    p = sub.add_parser('prompts', help='freeze the prompt set (no requests) and print its token counts')
    p.add_argument('--tokenizer', default=str(Path.home()/'llm/ds41/og/models/ds41-og/tokenizer.json'))
    a = ap.parse_args()
    sys.exit(dict(run=run, compare=compare, prompts=show)[a.cmd](a))


if __name__ == '__main__':
    main()
