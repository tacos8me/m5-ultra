"""Mac + RTX split (ds41 split box+Mac) benchmark harness. Runs on the Mac against llama-swap. Usage: bench.py <leg> [args]"""
import base64, hashlib, io, json, os, random, subprocess, sys, threading, time, urllib.request, uuid
from pathlib import Path

OUT = Path.home() / 'llm/ds41/mac-rtx-split-bench'
URL = 'http://127.0.0.1:8080/v1/chat/completions'
BOX_HEALTH = 'http://10.10.10.1:10051/health'
MAC_HEALTH = 'http://127.0.0.1:10001/health'
OG_STATS = 'http://127.0.0.1:10001/og/stats'
OG_LOG = Path.home() / 'llm/ds41/og/logs/og-child.log'
SRC = Path.home() / 'src/wt/ds41-og'
TOK_DIR = Path.home() / 'models/DeepSeek-V4.1-Flash-original'
CACHE = OUT / '.corpus'


def getj(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        return {'error': repr(e)}


def idle_snapshot():
    b, m = getj(BOX_HEALTH), getj(MAC_HEALTH)
    return {'box_sessions': b.get('sessions'), 'box_conn': b.get('connections'), 'box_gpu_job': b.get('gpu_job'),
            'box_queued': b.get('queued_jobs'), 'mac_inflight': m.get('inflight'), 'box_version': b.get('version'),
            'numerics': b.get('numerics')}


def wait_idle(max_wait=300):
    """Return (snapshot, waited_s). Waits until box sessions==0 and Mac inflight==0."""
    t0 = time.time()
    while True:
        s = idle_snapshot()
        if s['box_sessions'] == 0 and s['mac_inflight'] == 0 and not s['box_gpu_job']:
            return s, round(time.time() - t0, 1)
        if time.time() - t0 > max_wait:
            s['contention'] = 'not idle after wait'
            return s, round(time.time() - t0, 1)
        time.sleep(2)


# ---------------- corpus ----------------
_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    return _tok


def sanitize(text):
    t = tok()
    for s in t.all_special_tokens:
        text = text.replace(s, '[special token literal]')
    return text.replace('｜', '|').replace('<think>', '[think tag]').replace('</think>', '[/think tag]')


def corpus_ids(kind):
    import numpy as np
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{kind}.npy'
    if f.exists():
        return np.load(f)
    seen, parts = set(), []
    if kind == 'code':
        files = sorted(p for p in SRC.rglob('*') if p.is_file() and p.suffix in ('.py', '.md') and '.git' not in p.parts
                       and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
                       and not any(p.name.startswith(f'README.{x}') for x in ('fr', 'ja', 'ko', 'zh')))
    else:  # prose: English markdown docs with fenced code blocks and tables stripped
        files = sorted(p for p in SRC.rglob('*.md') if '.git' not in p.parts
                       and not any(p.name.startswith(f'README.{x}') for x in ('fr', 'ja', 'ko', 'zh')))
    for p in files:
        try:
            txt = p.read_text()
        except Exception:
            continue
        if kind == 'prose':
            out, fence = [], False
            for line in txt.splitlines():
                if line.strip().startswith('```'):
                    fence = not fence; continue
                if fence or line.lstrip().startswith('|') or not line.strip():
                    continue
                out.append(line)
            txt = '\n'.join(out)
            if len(txt) < 400:
                continue
        h = hashlib.sha1(txt.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        parts.append(f'File: {p.relative_to(SRC)}\n{txt[:300_000]}')
    ids = np.array(tok().encode(sanitize('\n\n'.join(parts)), add_special_tokens=False), dtype=np.int32)
    np.save(f, ids)
    return ids


_off = {'code': 0, 'prose': 0}


def make_doc(n, kind='code'):
    """Fresh document of ~n tokens: unique nonce first (defeats prefix cache), then corpus slice at a rotating offset."""
    ids = corpus_ids(kind)
    L = len(ids)
    o = _off[kind] % L
    _off[kind] = o + n + 7919
    import numpy as np
    sl = ids[(np.arange(n) + o) % L].tolist()
    nonce = f'[doc {uuid.uuid4().hex} {time.time_ns()}]\n'
    return nonce + tok().decode(sl)


# ---------------- request ----------------
def og_stats():
    s = getj(OG_STATS)
    return {k: s.get(k) for k in ('opened', 'steps', 'box_s', 'wait_s', 'roundtrip_s', 'payload_s', 'rows', 'present_hits')}


def stream_chat(messages, max_tokens=256, keep_text=True):
    body = json.dumps({'model': 'ds41', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
                       'stream_options': {'include_usage': True}}).encode()
    t0 = time.perf_counter(); wall0 = time.time()
    first = last = None; usage = None; out = []; finish = None; reasoning = 0; chunks = 0
    with urllib.request.urlopen(urllib.request.Request(URL, body, {'Content-Type': 'application/json'}), timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            d = json.loads(line[6:])
            usage = d.get('usage') or usage
            if d.get('choices'):
                ch = d['choices'][0]
                finish = ch.get('finish_reason') or finish
                dl = ch.get('delta', {})
                if dl.get('reasoning_content'):
                    reasoning += 1
                if dl.get('content'):
                    now = time.perf_counter(); first = first or now; last = now; chunks += 1
                    out.append(dl['content'])
    c = usage['completion_tokens']
    res = {'wall_start': round(wall0, 3), 'prompt_tokens': usage['prompt_tokens'],
           'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
           'completion_tokens': c, 'finish': finish, 'ttft_s': round(first - t0, 3),
           'decode_tok_s': round((c - 1) / (last - first), 1) if c > 1 and last > first else None,
           'total_s': round(last - t0, 3), 'chunks': chunks, 'reasoning_chunks': reasoning,
           'srv_ttft_s': usage.get('time_to_first_token'), 'srv_gen_tok_s': usage.get('generation_tokens_per_second'),
           '_t0': t0, '_first': first, '_last': last}
    res['text'] = ''.join(out) if keep_text else ''.join(out)[:200]
    return res


import re
RX_OPEN = re.compile(r'ds41-og \S+: (\d+) tokens, (\d+) image\(s\), box open ([\d.]+)s \(resumed (\d+), prefill ([\d.]+)s, (\d+) bytes.*?import\+replay ([\d.]+)s')
RX_MTP = re.compile(r'MTP\[\d+\] finish=(\w+) tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+)')


def parse_log(off):
    time.sleep(0.3)
    try:
        with open(OG_LOG, 'rb') as f:
            f.seek(off); txt = f.read().decode(errors='replace')
    except Exception:
        return {}
    opens = [dict(zip(('tokens', 'images', 'box_open_s', 'box_resumed', 'box_prefill_s', 'bytes', 'import_s'),
                      (int(m[0]), int(m[1]), float(m[2]), int(m[3]), float(m[4]), int(m[5]), float(m[6])))) for m in RX_OPEN.findall(txt)]
    mtps = [dict(zip(('finish', 'tokens', 'cycles', 'tok_per_cycle', 'accepted', 'drafted'),
                     (m[0], int(m[1]), int(m[2]), float(m[3]), int(m[4]), int(m[5])))) for m in RX_MTP.findall(txt)]
    return {'opens': opens, 'mtp': mtps}


def measured(messages, max_tokens=256, keep_text=True, check_idle=True):
    pre = None; waited = 0
    if check_idle:
        pre, waited = wait_idle()
    s0 = og_stats(); lo = log_offset()
    r = stream_chat(messages, max_tokens, keep_text)
    s1 = og_stats()
    pl = parse_log(lo)
    if len(pl.get('opens', [])) == 1:
        r['og_open'] = pl['opens'][0]
    if len(pl.get('mtp', [])) == 1:
        m = pl['mtp'][0]; r['mtp'] = m; r['accept_rate'] = round(m['accepted'] / m['drafted'], 3) if m['drafted'] else None
    if len(pl.get('opens', [])) > 1:
        r['log_opens'] = pl['opens']
    d = {k: (round(s1[k] - s0[k], 4) if isinstance(s1.get(k), (int, float)) and isinstance(s0.get(k), (int, float)) else None) for k in s0}
    r['og_delta'] = d
    if d.get('steps'):
        r['box_ms_per_step'] = round(1000 * d['box_s'] / d['steps'], 2)
        r['roundtrip_ms_per_step'] = round(1000 * d['roundtrip_s'] / d['steps'], 2)
        r['wait_ms_per_step'] = round(1000 * d['wait_s'] / d['steps'], 2)
    r['foreign_requests'] = (d['opened'] - 1) if d.get('opened') is not None else None
    r['pre_idle'] = pre; r['idle_wait_s'] = waited
    return r


def emit(fh, rec):
    rec = {k: v for k, v in rec.items() if not k.startswith('_')}
    fh.write(json.dumps(rec) + '\n'); fh.flush()
    print(json.dumps({k: v for k, v in rec.items() if k not in ('text', 'pre_idle', 'og_delta', 'log', 'mtp', 'log_opens')})[:400], flush=True)


def warmup(fh=None):
    r = stream_chat([{'role': 'user', 'content': f'[warmup {uuid.uuid4().hex}] Reply with the word ready.'}], 8)
    d = make_doc(8192)
    r2 = stream_chat([{'role': 'user', 'content': d + '\n\nSummarize the code above in five bullet points.'}], 64, False)
    if fh:
        emit(fh, {'event': 'warmup_discarded', 'short_ttft_s': r['ttft_s'], 'w8k_ttft_s': r2['ttft_s'], 'w8k_prompt': r2['prompt_tokens']})


def log_offset():
    try:
        return OG_LOG.stat().st_size
    except Exception:
        return 0


def save_log_slice(off, name):
    try:
        with open(OG_LOG, 'rb') as f:
            f.seek(off); data = f.read()
        (OUT / 'logs').mkdir(exist_ok=True)
        (OUT / 'logs' / f'{name}.og-child.log').write_bytes(data)
    except Exception as e:
        print('log slice failed', e)


CODE_Q = ['Summarize the code above in five bullet points.', 'List the main classes and what each one does.',
          'Explain how caching is handled in the code above, in detail.', 'Describe the request handling path step by step.',
          'What are the most important configuration options, and what do they control?',
          'Point out three potential bugs or risky spots in the code above and explain why.']
PROSE_Q = ['Write a reflective essay in flowing prose (no lists, no code, no headings) about the ideas in the text above.',
           'Retell the text above as a short story told by an engineer on a night shift. Prose only, no lists or code.',
           'Write a persuasive letter to a skeptical manager arguing for the approach described above. Prose paragraphs only.',
           'Describe, in plain narrative prose for a general reader, what problem the text above is trying to solve and why it matters.',
           'Write an op-ed style piece about the trade-offs discussed above. Continuous prose, no bullet points.',
           'Imagine you are explaining the text above to a curious teenager over dinner; write that conversation as narrative prose.']


# ---------------- legs ----------------
def leg_prefill(sizes):
    fh = open(OUT / 'leg1_prefill.jsonl', 'a'); off = log_offset()
    warmup(fh)
    for n, reps in sizes:
        for i in range(reps):
            d = make_doc(n)
            r = measured([{'role': 'user', 'content': d + '\n\n' + CODE_Q[0]}], 256, keep_text=False)
            r.update(event='prefill', target=n, rep=i, prefill_tok_s=round(r['prompt_tokens'] / r['ttft_s'], 1))
            emit(fh, r)
    save_log_slice(off, 'leg1')


def leg_decode(points):
    fh = open(OUT / 'leg2_decode.jsonl', 'a'); off = log_offset()
    warmup(fh)
    for n, kind in points:
        qs = CODE_Q if kind == 'code' else PROSE_Q
        d = make_doc(n, kind)
        for i, q in enumerate(qs):
            r = measured([{'role': 'user', 'content': d + '\n\n' + q}], 256)
            r.update(event='decode', target=n, workload=kind, q=i, fresh=(i == 0))
            if i == 0:
                r['prefill_tok_s'] = round(r['prompt_tokens'] / r['ttft_s'], 1)
            emit(fh, r)
    save_log_slice(off, 'leg2')


def run_concurrent(msg_lists, max_tokens=256):
    res = [None] * len(msg_lists); errs = []

    def go(j):
        try:
            res[j] = stream_chat(msg_lists[j], max_tokens, keep_text=False)
        except Exception as e:
            errs.append(repr(e))
    s0 = og_stats(); lo = log_offset()
    th = [threading.Thread(target=go, args=(j,)) for j in range(len(msg_lists))]
    for t in th:
        t.start(); time.sleep(0.02)
    for t in th:
        t.join()
    s1 = og_stats()
    if errs:
        return {'errors': errs}
    start = min(r['_first'] for r in res); end = max(r['_last'] for r in res)
    ov_start = max(r['_first'] for r in res); ov_end = min(r['_last'] for r in res)
    t0 = min(r['_t0'] for r in res)
    return {'streams': len(res), 'per_stream_tok_s': [r['decode_tok_s'] for r in res],
            'ttft_s': [r['ttft_s'] for r in res], 'completion_tokens': [r['completion_tokens'] for r in res],
            'prompt_tokens': [r['prompt_tokens'] for r in res], 'cached_tokens': [r['cached_tokens'] for r in res],
            'first_rel_s': [round(r['_first'] - t0, 3) for r in res], 'last_rel_s': [round(r['_last'] - t0, 3) for r in res],
            'aggregate_tok_s': round(sum(r['completion_tokens'] for r in res) / (end - start), 1),
            'overlap_s': round(max(0, ov_end - ov_start), 3), 'serialized': ov_end <= ov_start, 'log': parse_log(lo),
            'foreign_requests': (s1['opened'] - s0['opened'] - len(res)) if s0.get('opened') is not None else None}


def leg_conc(configs):
    fh = open(OUT / 'leg3_concurrency.jsonl', 'a'); off = log_offset()
    warmup(fh)
    for n, c, rounds in configs:
        # cold arrival: c fresh prompts at once
        pre, waited = wait_idle()
        r = run_concurrent([[{'role': 'user', 'content': make_doc(n) + '\n\n' + CODE_Q[0]}] for _ in range(c)])
        r.update(event='cold_arrival', target=n, c=c, pre_idle=pre); emit(fh, r)
        # warm decode: prime c docs sequentially, then concurrent cached follow-ups
        docs = [make_doc(n) for _ in range(c)]
        for d in docs:
            wait_idle(); stream_chat([{'role': 'user', 'content': d + '\n\n' + CODE_Q[0]}], 1, False)
        for k in range(rounds):
            pre, waited = wait_idle()
            q = CODE_Q[1 + k % (len(CODE_Q) - 1)]
            r = run_concurrent([[{'role': 'user', 'content': d + '\n\n' + q}] for d in docs])
            r.update(event='warm_decode', target=n, c=c, round=k, pre_idle=pre); emit(fh, r)
    save_log_slice(off, 'leg3')


def leg_resume(sizes, regen_sizes):
    fh = open(OUT / 'leg4_resume.jsonl', 'a'); off = log_offset()
    warmup(fh)
    follow = ['Now name the single most complex function and say why.', 'Which file would you read first, and why?',
              'Give one concrete improvement you would make.']
    for n in sizes:
        d = make_doc(n)
        m1 = [{'role': 'user', 'content': d + '\n\nSummarize the code above in one sentence.'}]
        r1 = measured(m1, 64)
        r1.update(event='turn1', target=n, prefill_tok_s=round(r1['prompt_tokens'] / r1['ttft_s'], 1)); emit(fh, r1)
        for k, f in enumerate(follow):
            m2 = m1 + [{'role': 'assistant', 'content': r1['text']}, {'role': 'user', 'content': f}]
            r2 = measured(m2, 64)
            r2.update(event='turn2', target=n, k=k, new_tokens=r2['prompt_tokens'] - (r2['cached_tokens'] or 0)); emit(fh, r2)
        if n in regen_sizes:
            for k in range(3):
                r3 = measured(m1, 64)
                r3.update(event='regenerate', target=n, k=k, same_answer=(r3['text'] == r1['text'])); emit(fh, r3)
    save_log_slice(off, 'leg4')


def png_b64(img):
    b = io.BytesIO(); img.save(b, 'PNG'); return base64.b64encode(b.getvalue()).decode()


def make_images(seed):
    from PIL import Image, ImageDraw, ImageFont
    rnd = random.Random(seed)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 120)
    j = lambda: rnd.randint(-8, 8)
    red = Image.new('RGB', (256, 256), (220 + j(), 20 + abs(j()), 30 + j()))
    green = Image.new('RGB', (256, 256), (30 + j(), 170 + j(), 40 + j()))
    def digits(s):
        im = Image.new('RGB', (560, 220), 'white'); ImageDraw.Draw(im).text((40 + j(), 40 + j()), s, fill='black', font=font); return im
    def shapes(n, kind):
        im = Image.new('RGB', (512, 512), 'white'); dr = ImageDraw.Draw(im); placed = []
        while len(placed) < n:
            x, y = rnd.randint(60, 452), rnd.randint(60, 452)
            if all((x - a) ** 2 + (y - b) ** 2 > 110 ** 2 for a, b in placed):
                placed.append((x, y))
                if kind == 'circle':
                    dr.ellipse((x - 40, y - 40, x + 40, y + 40), fill=(30, 80, 220))
                else:
                    dr.polygon([(x, y - 45), (x - 42, y + 35), (x + 42, y + 35)], fill=(230, 140, 20))
        return im
    return red, green, digits('4827'), digits('391'), shapes(5, 'circle'), shapes(3, 'triangle')


def leg_vision(reps=2):
    fh = open(OUT / 'leg5_vision.jsonl', 'a'); off = log_offset()
    OUT.joinpath('images').mkdir(exist_ok=True)

    def msg(imgs, q):
        parts = [{'type': 'text', 'text': f'[req {uuid.uuid4().hex}]'}]
        parts += [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + png_b64(im)}} for im in imgs]
        parts.append({'type': 'text', 'text': q})
        return [{'role': 'user', 'content': parts}]
    red, green, d1, d2, circ, tri = make_images(0)
    w = stream_chat(msg([circ], 'Describe this image briefly.'), 16)
    emit(fh, {'event': 'warmup_discarded', 'ttft_s': w['ttft_s'], 'text': w['text']})
    stream_chat([{'role': 'user', 'content': f'[warmup {uuid.uuid4().hex}] Reply with the word ready.'}], 8)
    for rep in range(reps):
        red, green, d1, d2, circ, tri = make_images(rep + 1)
        for name, im in zip(('red', 'green', 'digits4827', 'digits391', 'circles5', 'triangles3'), (red, green, d1, d2, circ, tri)):
            im.save(OUT / 'images' / f'{name}.rep{rep}.png')
        tasks = [
            ('1img_color', [red], 'What color is this image? Answer with one word.', ['red']),
            ('1img_digits', [d1], 'What number is written in the image? Answer with the digits only.', ['4827']),
            ('1img_count', [circ], 'How many circles are in the image? Answer with a single number.', ['5', 'five']),
            ('2img_colors', [red, green], 'What is the color of the first image and of the second image? Answer as: first, second.', ['red', 'green']),
            ('2img_digits', [d1, d2], 'What number is written in each image? Answer as: first, second.', ['4827', '391']),
            ('2img_count', [circ, tri], 'How many circles are in the first image and how many triangles are in the second? Answer as: circles, triangles.', ['5', '3']),
        ]
        for name, imgs, q, expect in tasks:
            r = measured(msg(imgs, q), 32)
            low = r['text'].lower()
            ok = all(e in low for e in expect) if name != '1img_count' else any(e in low for e in expect)
            r.update(event='vision', task=name, images=len(imgs), rep=rep, expected=expect, correct=ok)
            emit(fh, r)
    save_log_slice(off, 'leg5')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    leg = sys.argv[1]
    if leg == 'corpus':
        for k in ('code', 'prose'):
            print(k, len(corpus_ids(k)))
    elif leg == 'prefill':
        spec = sys.argv[2] if len(sys.argv) > 2 else '8192:3,16384:3,32768:3,65536:3,131072:3,262144:1,524288:1,786432:1,1040000:1'
        leg_prefill([tuple(int(x) for x in p.split(':')) for p in spec.split(',')])
    elif leg == 'decode':
        spec = sys.argv[2] if len(sys.argv) > 2 else '8192:code,8192:prose,131072:code,262144:code,524288:code,1040000:code'
        leg_decode([(int(p.split(':')[0]), p.split(':')[1]) for p in spec.split(',')])
    elif leg == 'conc':
        spec = sys.argv[2] if len(sys.argv) > 2 else '8192:2:4,131072:2:4,8192:4:4'
        leg_conc([tuple(int(x) for x in p.split(':')) for p in spec.split(',')])
    elif leg == 'resume':
        leg_resume([8192, 131072, 524288], {131072})
    elif leg == 'vision':
        leg_vision()
