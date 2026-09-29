"""Fixed-prompt c4 A/B for ds41 through llama-swap (temperature 0, streamed, identical-output gate).

  c4ab.py run LABEL [--c 4] [--groups 8] [--max-tokens 384] [--url http://192.168.1.203:8080] [--stats CMD] [--out DIR]
  c4ab.py compare BASE.json VARIANT.json [MORE.json ...]

Prompts: bench/c4ab-prompts.json, 8 fixed code documents (2.2K-4.7K tokens, frozen on first use; its sha256 is
recorded), so every request runs at context >= 1024 where the EVICT cost policy and the fused-pair depth
table are live (the old 6 short fixedconc prompts stay below 1024 tokens: acceptance-only depth, DS41_PIPE_COSTS*
never consulted). The 8 prompts form 8/c sets of c, cycled (set = group % (8/c)); one discarded warm-up group first
(first-use Metal pipeline builds of the fused-pair / batched-draft shapes cost ~0.2 s after a restart).

Per group (c concurrent requests; --c 1/2 for the c1/c2 regression legs), from streamed chunk times:
  window_tok_s  tokens emitted while all c decode / that window   <- cN decode rate, the A/B metric
  span_tok_s    all tokens / (last token - first token)           (runthru2 c4 metric)
  wall_tok_s    all tokens / (last token - send)                  (fixedconc c4 metric)
  --stats CMD   og/stats before/after (e.g. --stats "ssh m5 curl -s localhost:12147/og/stats"):
                og steps, rows/step, tokens/step, fused_calls, draft_batches, box wait ms/step
Identity: a prompt repeated inside a run must give the same text; `compare` requires every prompt's text to be
identical across all files (outputs must not depend on the variant) and then reports the paired per-group
ratio VARIANT/BASE (same prompt set, same group index) with a bootstrap 95% CI. 8 groups resolve ~2-3%.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PROMPTS = HERE/'c4ab-prompts.json'
SOURCES = ['graphlib.py', 'queue.py', 'shlex.py', 'textwrap.py', 'cmd.py', 'fileinput.py', 'timeit.py', 'selectors.py']
QUESTIONS = ['Walk through this module in detail: its purpose, every public class and function, and how they interact.',
             'Explain how this code works step by step, then list three subtle edge cases it handles and why.',
             'Write a thorough technical review of this module: design, correctness risks, and concrete improvements.',
             'Describe the control flow of the main entry points of this module in detail, with short examples.']


def prompts():
    if not PROMPTS.exists():
        import textwrap  # stdlib directory of this interpreter: frozen once into the prompts file
        base = Path(textwrap.__file__).parent
        items = [f'File: {name}\n```python\n{(base/name).read_text()}\n```\n\n{QUESTIONS[i % 4]}'
                 for i, name in enumerate(SOURCES)]
        PROMPTS.write_text(json.dumps(items))
    items = json.loads(PROMPTS.read_text())
    return items, hashlib.sha256(PROMPTS.read_bytes()).hexdigest()[:16]


def stream(url, prompt, max_tokens, out, i, t0, model='ds41'):
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens,
                           temperature=0, stream=True, stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(url + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    chunks, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=1800) as r:
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
                piece = delta.get('content') or delta.get('reasoning_content')
                if piece:
                    chunks.append(time.perf_counter() - t0)
                    text.append(piece)
    out[i] = dict(text=''.join(text), chunks=chunks, tokens=usage['completion_tokens'],
                  prompt_tokens=usage['prompt_tokens'])


def group(url, items, max_tokens, model='ds41'):
    out, t0 = [None] * len(items), time.perf_counter()
    th = [threading.Thread(target=stream, args=(url, p, max_tokens, out, i, t0, model)) for i, p in enumerate(items)]
    [t.start() for t in th]
    [t.join() for t in th]
    if any(o is None for o in out):
        raise RuntimeError('a request failed')
    lo, hi = max(o['chunks'][0] for o in out), min(o['chunks'][-1] for o in out)
    # chunks may carry several tokens: scale each stream's chunk count to its token count (batch_cycle.py)
    inside = sum(sum(1 for t in o['chunks'] if lo < t <= hi) * o['tokens'] / len(o['chunks']) for o in out)
    total = sum(o['tokens'] for o in out)
    first, last = min(o['chunks'][0] for o in out), max(o['chunks'][-1] for o in out)
    return out, dict(window_tok_s=round(inside / (hi - lo), 2) if hi > lo else None, window_s=round(hi - lo, 3),
                     span_tok_s=round(total / (last - first), 2), wall_tok_s=round(total / last, 2),
                     tokens=[o['tokens'] for o in out], ttft=[round(o['chunks'][0], 3) for o in out],
                     per_stream=[round((o['tokens'] - 1) / (o['chunks'][-1] - o['chunks'][0]), 1) for o in out])


def og_stats(cmd):
    if not cmd:
        return None
    try:
        return json.loads(subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout)
    except Exception:  # noqa: BLE001 -- stats are optional
        return None


def stats_delta(s0, s1, tokens):
    if not s0 or not s1:
        return {}
    d = {k: s1[k] - s0[k] for k in ('steps', 'rows', 'fused_calls', 'fused_steps', 'draft_batches', 'wait_s', 'opened')
         if k in s0 and k in s1}
    steps = max(1, d.get('steps', 0))
    return dict(og_steps=d.get('steps'), rows_per_step=round(d.get('rows', 0) / steps, 3),
                tok_per_step=round(tokens / steps, 3), fused_calls=d.get('fused_calls'),
                fused_share=round(d.get('fused_steps', 0) / steps, 3), draft_batches=d.get('draft_batches'),
                box_wait_ms_per_step=round(1000 * d.get('wait_s', 0) / steps, 3), opened=d.get('opened'))


def summary(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return dict(n=len(v), median=round(statistics.median(v), 2), mean=round(statistics.mean(v), 2),
                sd=round(sd, 2), ci95_abs=round(1.96 * sd / len(v) ** 0.5, 2))


def run(args):
    items, sha = prompts()
    c = args.c
    sets = [items[i:i + c] for i in range(0, len(items) - c + 1, c)]
    url = args.url.rstrip('/')
    print(json.dumps(dict(event='start', label=args.label, prompts_sha=sha, c=c, groups=args.groups,
                          max_tokens=args.max_tokens)), flush=True)
    group(url, sets[-1], args.max_tokens, args.model)  # warm-up: discarded (first-use pipelines, prompt caches)
    texts, groups, mismatch = {}, [], 0
    for g in range(args.groups):
        k = g % len(sets)
        s0 = og_stats(args.stats)
        out, rec = group(url, sets[k], args.max_tokens, args.model)
        s1 = og_stats(args.stats)
        rec.update(group=g, set=k, **stats_delta(s0, s1, sum(rec['tokens'])))
        for j, o in enumerate(out):
            key = str(c * k + j)
            if key in texts and texts[key] != o['text']:
                mismatch += 1
            texts.setdefault(key, o['text'])
        rec['prompt_tokens'] = [o['prompt_tokens'] for o in out]
        groups.append(rec)
        print(json.dumps(rec), flush=True)
    res = dict(label=args.label, prompts_sha=sha, url=url, c=c, max_tokens=args.max_tokens, time=time.time(),
               within_run_identical=mismatch == 0, groups=groups, texts=texts,
               summary={m: summary([r[m] for r in groups]) for m in ('window_tok_s', 'span_tok_s', 'wall_tok_s')})
    path = Path(args.out)/f'c4ab-{args.label}.json'
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(event='done', label=args.label, file=str(path), within_run_identical=mismatch == 0,
                          **res['summary'])), flush=True)
    return 0 if mismatch == 0 else 1


def compare(files):
    runs = [json.loads(Path(f).read_text()) for f in files]
    base = runs[0]
    ok = True
    for r in runs[1:]:
        if (r['prompts_sha'], r['max_tokens']) != (base['prompts_sha'], base['max_tokens']):
            print(json.dumps(dict(error='different prompts or max_tokens', file=r['label'])))
            ok = False
        # texts are keyed by prompt index, so c1/c2/c4 runs of the same prompts must agree too
        diff = [k for k in base['texts'] if k in r['texts'] and r['texts'][k] != base['texts'][k]]
        print(json.dumps(dict(identity=r['label'], vs=base['label'], identical=not diff, differing_prompts=diff)))
        ok &= not diff and r['within_run_identical']
    rng = random.Random(0)
    for metric in ('window_tok_s', 'span_tok_s', 'wall_tok_s'):
        b = {g['group']: g[metric] for g in base['groups'] if g[metric]}
        print(json.dumps(dict(metric=metric, label=base['label'], **(base['summary'][metric] or {}))))
        for r in runs[1:]:
            if r.get('c', 4) != base.get('c', 4):
                continue  # different concurrency: identity only
            v = {g['group']: g[metric] for g in r['groups'] if g[metric]}
            pairs = [v[k] / b[k] for k in sorted(set(b) & set(v))]
            if not pairs:
                continue
            boots = sorted(statistics.mean(rng.choices(pairs, k=len(pairs))) for _ in range(4000))
            print(json.dumps(dict(metric=metric, label=r['label'], **(r['summary'][metric] or {}),
                                  paired_ratio=round(statistics.mean(pairs), 4), n_pairs=len(pairs),
                                  ratio_ci95=[round(boots[100], 4), round(boots[3899], 4)])))
    print(json.dumps(dict(identity_ok=ok)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('label')
    r.add_argument('--c', type=int, default=4, choices=(1, 2, 4), help='concurrent requests per group')
    r.add_argument('--groups', type=int, default=8)
    r.add_argument('--max-tokens', type=int, default=384)
    r.add_argument('--url', default='http://192.168.1.203:8080')
    r.add_argument('--model', default='ds41')
    r.add_argument('--stats', default='', help='shell command printing og/stats JSON')
    r.add_argument('--out', default=str(HERE))
    c = sub.add_parser('compare')
    c.add_argument('files', nargs='+')
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == 'run' else compare(a.files))


if __name__ == '__main__':
    main()
