"""Mixed-traffic soak for a ds41 endpoint (default: 2 h) with memory sampling.

Two client lanes: lane A always busy, lane B busy half of the time, so c2 overlaps are
frequent. Each request is drawn from a weighted mix and checked:

  short     1-2 turn chat, stream or not, 32-400 tokens
  tools     Hermes-style agent loop (parity.agent_loop), streamed or not
  think     enable_thinking, reasoning + answer present
  doc128k   ~128K-token needle document, answer must contain the code
  resume    one conversation over ~8-48K tokens grown turn by turn (prefix reuse)
  cancel    stream disconnected after ~20 chunks, next request must be fast

Errors of any kind (HTTP, SSE error event, missing [DONE], failed check) count as errors.
Every 30 s: child footprint/RSS (supervisor /health pid), og /og/stats, supervisor
counters, macOS swap. Summary: <logs>/<label>.soak.json; events <label>.soak.jsonl.

  python og_serve/soak.py --label sv-soak1 --base http://127.0.0.1:12161/v1 --model ds41-og --hours 2
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity  # noqa: E402

HOME = Path.home()
LOGS = Path(os.environ.get('OG_LOGS', str(HOME/'llm/ds41/og-serve')))
MIX = dict(short=34, tools=20, think=10, doc128k=6, resume=24, cancel=6)
TOPICS = ['the water cycle', 'binary search trees', 'the French revolution', 'how vaccines work', 'TCP handshakes',
          'photosynthesis', 'the Rust borrow checker', 'Fourier transforms', 'sourdough starters', 'black holes',
          'git rebase vs merge', 'the Krebs cycle', 'SQL indexes', 'plate tectonics', 'Python asyncio']


def footprint(pid):
    buf = ctypes.create_string_buffer(512)
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    if lib.proc_pid_rusage(int(pid), 2, buf) != 0:
        return None, None
    raw = buf.raw
    return int.from_bytes(raw[72:80], 'little') / 2**30, int.from_bytes(raw[64:72], 'little') / 2**30


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except ValueError:
            return None
    except OSError:
        return None


class Soak:
    def __init__(self, args):
        self.args = args
        self.t = parity.Target(f'x={args.base}@{args.model}')
        self.out = (LOGS/f'{args.label}.soak.jsonl').open('a')
        self.lock = threading.Lock()
        self.stop_at = time.monotonic() + args.hours * 3600
        self.records = []
        self.samples = []
        self.conversations = {}
        self.docs = {}

    def emit(self, **rec):
        rec = dict(t=round(time.time(), 1), **rec)
        with self.lock:
            self.out.write(json.dumps(rec, default=str) + '\n'); self.out.flush()
            if rec.get('kind') not in (None, 'sample'):
                self.records.append(rec)
        if rec.get('kind') != 'sample' and (not rec.get('ok', True) or self.args.verbose):
            print(json.dumps(rec, default=str)[:600], flush=True)

    # --- request kinds ---------------------------------------------------------------------------
    def short(self, rng):
        topic = rng.choice(TOPICS)
        n = rng.choice([32, 64, 128, 256, 400])
        msgs = [{'role': 'user', 'content': f'Explain {topic} to a curious teenager.'}]
        if rng.random() < 0.5:
            r = self.t.stream(msgs, max_tokens=n, temperature=rng.choice([0, 0.7]))
            parity.expect(r['done'] and not r['errors'] and r['content'].strip(), f'{r["errors"]} done={r["done"]}')
            parity.usage_ok(r['usage'], n)
            return dict(tokens=r['usage']['completion_tokens'], ttft=r['first'], backend=r['backend'],
                        decode=self.rate(r))
        d = self.t.chat(msgs, max_tokens=n, temperature=0)
        parity.expect((d['choices'][0]['message'].get('content') or '').strip(), 'empty content')
        parity.usage_ok(d.get('usage'), n)
        return dict(tokens=d['usage']['completion_tokens'], request_s=round(d['_s'], 2))

    @staticmethod
    def rate(r):
        c = (r.get('usage') or {}).get('completion_tokens') or 0
        if c > 16 and r['last'] and r['first'] and r['last'] > r['first']:
            return round((c - 1) / (r['last'] - r['first']), 1)
        return None

    def tools(self, rng):
        task = rng.choice([
            ('Read data/numbers.txt, add up the numbers with the calculator, and tell me the total.', '100'),
            ('What is the weather in Tokyo and in Oslo? Then tell me which is warmer.', None),
            ('Read notes/todo.txt and tell me what time I must call Ana.', '5'),
            ('Use the calculator to compute (1234 * 5678) - 91011 and report the result.', '6915641')])
        out = parity.agent_loop(self.t, task[0], stream=rng.random() < 0.7)
        if task[1]:
            parity.expect(task[1] in out['answer'].replace(',', ''), f'answer {out["answer"][:200]!r}')
        return dict(steps=out['steps'], calls=out['calls'])

    def think(self, rng):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        r = self.t.stream([{'role': 'user', 'content': f'What is {a} * {b}? Think it through, then give the number.'}],
                          max_tokens=2000, temperature=0, chat_template_kwargs={'enable_thinking': True})
        parity.expect(r['done'] and not r['errors'], f'{r["errors"]}')
        parity.expect(r['reasoning'].strip(), 'no reasoning')
        parity.expect(str(a * b) in r['content'].replace(',', ''), f'expected {a*b}: {r["content"][-120:]!r}')
        return dict(tokens=r['usage']['completion_tokens'], ttft=r['first'], decode=self.rate(r))

    def doc128k(self, rng):
        seed = rng.randint(0, 10**9)
        doc, code = parity.needle_doc(self.args.doc_tokens, seed)
        r = self.t.stream([{'role': 'user', 'content': f'{doc}\n\nWhat is the secret vault code? Reply with the number only.'}],
                          max_tokens=24, temperature=0)
        parity.expect(r['done'] and not r['errors'], f'{r["errors"]}')
        parity.expect(str(code) in r['content'], f'expected {code}: {r["content"]!r}')
        return dict(prompt_tokens=r['usage']['prompt_tokens'], ttft=r['first'])

    def resume(self, rng, lane):
        """One conversation per lane, only ever extended (prefix reuse): an ~8K document, then follow-up
        turns, some adding another 8-24K of document; a new conversation past ~48K tokens."""
        conv = self.conversations.get(lane)
        if conv is None or conv['tokens'] > 48000:
            doc, code = parity.needle_doc(8000, rng.randint(0, 10**9))
            conv = dict(messages=[{'role': 'system', 'content': 'You answer questions about the document.'},
                                  {'role': 'user', 'content': doc + '\n\nSummarize this document in two sentences.'}],
                        code=code, turns=0, tokens=0)
            self.conversations[lane] = conv
        elif rng.random() < 0.3:
            more, other = parity.needle_doc(rng.choice([8000, 24000]), rng.randint(0, 10**9))
            more = more.replace(f'\nThe secret vault code is {other}.\n', '\n')  # one code per conversation
            conv['messages'].append({'role': 'user', 'content': 'More of the document:\n' + more +
                                     '\n\nIn one sentence, what does this new part add?'})
        else:
            conv['messages'].append({'role': 'user', 'content': rng.choice([
                'What is the secret vault code in the document?', 'Name three words that appear often.',
                'Write a haiku about the document.', 'Continue: add one more paragraph in the same style.'])})
        r = self.t.stream(conv['messages'], max_tokens=rng.choice([64, 200, 600]), temperature=0)
        parity.expect(r['done'] and not r['errors'] and r['content'].strip(), f'{r["errors"]}')
        if 'secret vault code' in conv['messages'][-1]['content']:
            parity.expect(str(conv['code']) in r['content'], f'expected {conv["code"]}: {r["content"][:120]!r}')
        conv['messages'].append({'role': 'assistant', 'content': r['content']})
        conv['turns'] += 1
        conv['tokens'] = r['usage']['prompt_tokens'] + r['usage']['completion_tokens']
        return dict(turn=conv['turns'], prompt_tokens=r['usage']['prompt_tokens'], ttft=r['first'], decode=self.rate(r))

    def cancel(self, rng):
        return parity.c_cancel(self.t)

    # --- lanes --------------------------------------------------------------------------------------
    def lane(self, name, duty, seed):
        rng = random.Random(seed)
        kinds, weights = zip(*MIX.items())
        while time.monotonic() < self.stop_at:
            if duty < 1 and rng.random() > duty:
                time.sleep(rng.uniform(5, 30))
                continue
            kind = rng.choices(kinds, weights)[0]
            t0 = time.time()
            try:
                fn = getattr(self, kind)
                detail = fn(rng, name) if kind == 'resume' else fn(rng)
                detail = {k: v for k, v in (detail or {}).items() if k not in ('kind', 'lane', 'ok', 's')}
                self.emit(kind=kind, lane=name, ok=True, s=round(time.time() - t0, 2), **detail)
            except Exception as exc:  # noqa: BLE001 -- every failure is an error in the soak
                self.emit(kind=kind, lane=name, ok=False, s=round(time.time() - t0, 2),
                          error=f'{type(exc).__name__}: {exc}'[:500])
                if kind == 'resume':
                    self.conversations.pop(name, None)
                time.sleep(2)

    def sampler(self):
        sup = self.args.health
        while time.monotonic() < self.stop_at + 30:
            h = get(sup + '/health') if sup else None
            pid = (h or {}).get('pid')
            fp, rss = footprint(pid) if pid else (None, None)
            stats = get(self.args.og_stats) if self.args.og_stats else None
            swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.strip()
            boxh = get(self.args.box_health, timeout=3) if self.args.box_health else None
            rec = dict(kind='sample', pid=pid, backend=(h or {}).get('backend'), footprint_gib=fp and round(fp, 2),
                       rss_gib=rss and round(rss, 2), swap=swap,
                       sup={k: (h or {}).get(k) for k in ('failovers', 'box_lost', 'resumed', 'resume_failed', 'broken', 'inflight')},
                       og={k: (stats or {}).get(k) for k in ('opened', 'open_failed', 'open_retries', 'closed', 'sessions', 'box_lost', 'steps')},
                       recoveries=len((stats or {}).get('recoveries') or []),
                       box={k: (boxh or {}).get(k) for k in ('ok', 'uptime_s', 'sessions', 'version')} if boxh else None)
            self.samples.append(rec)
            self.emit(**rec)
            time.sleep(30)

    def run(self):
        threads = [threading.Thread(target=self.lane, args=('A', 1.0, 1), daemon=True),
                   threading.Thread(target=self.lane, args=('B', 0.5, 2), daemon=True),
                   threading.Thread(target=self.sampler, daemon=True)]
        for th in threads:
            th.start()
        for th in threads[:2]:
            th.join()
        time.sleep(1)
        return self.summary()

    def summary(self):
        recs = self.records
        by = {}
        for r in recs:
            b = by.setdefault(r['kind'], dict(n=0, errors=0, ttft=[], decode=[]))
            b['n'] += 1
            b['errors'] += not r['ok']
            if r.get('ttft'):
                b['ttft'].append(r['ttft'])
            if r.get('decode'):
                b['decode'].append(r['decode'])

        def pct(xs, p):
            xs = sorted(xs)
            return round(xs[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else None
        kinds = {k: dict(n=v['n'], errors=v['errors'], ttft_p50=pct(v['ttft'], .5), ttft_p95=pct(v['ttft'], .95),
                         decode_p50=pct(v['decode'], .5)) for k, v in by.items()}
        fps = [s['footprint_gib'] for s in self.samples if s.get('footprint_gib')]
        third = max(1, len(fps) // 3)
        summary = dict(label=self.args.label, hours=self.args.hours, requests=len(recs), errors=sum(not r['ok'] for r in recs),
                       kinds=kinds, footprint=dict(min=min(fps, default=None), max=max(fps, default=None),
                                                   first_third_mean=round(sum(fps[:third]) / third, 2) if fps else None,
                                                   last_third_mean=round(sum(fps[-third:]) / third, 2) if fps else None),
                       last_sample=self.samples[-1] if self.samples else None,
                       error_samples=[r for r in recs if not r['ok']][:20])
        (LOGS/f'{self.args.label}.soak.json').write_text(json.dumps(summary, indent=1, default=str))
        print(json.dumps({k: v for k, v in summary.items() if k != 'error_samples'}, default=str))
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--base', required=True, help='http://host:port/v1')
    ap.add_argument('--model', default='ds41-og')
    ap.add_argument('--hours', type=float, default=2.0)
    ap.add_argument('--health', default='', help='supervisor base URL for /health (child pid, counters)')
    ap.add_argument('--og-stats', default='', help='og worker /og/stats URL')
    ap.add_argument('--doc-tokens', type=int, default=128000)
    ap.add_argument('--box-health', default='http://10.10.10.1:10051/health')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    summary = Soak(args).run()
    sys.exit(0 if summary['errors'] == 0 else 1)


if __name__ == '__main__':
    main()
