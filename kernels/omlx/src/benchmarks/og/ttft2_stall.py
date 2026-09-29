"""ds41-ttft2 admission-stall client (partial-load leg; <= 2 box sessions at once).

  stall.py <base> <label> [n_b_sizes]

For each B size: A alone (fixed prompt, temp 0, A_TOKENS out), B alone (fresh 8K-class prompt, fixed text, 16 out),
then A with B started after A's 40th token. Reports A's inter-chunk gaps around B's admission (max gap, gaps > 60 ms),
B's TTFT alone/concurrent, and whether A's and B's full outputs match their alone runs.
"""
import http.client, json, os, sys, threading, time
from pathlib import Path
from urllib.parse import urlparse

base, label = sys.argv[1], sys.argv[2]
OUT = Path.home()/'llm/ds41/ttft2/fe'/(label + '-stall.jsonl')
A_TOKENS = int(os.environ.get('A_TOKENS', '300'))
B_SIZES = [int(x) for x in os.environ.get('B_SIZES', '8217,8600').split(',')]
REPS = int(os.environ.get('STALL_REPS', '2'))
sys.path.insert(0, os.environ.get('FE_DOC_TREE', str(Path.home()/'src/wt/ds41-og')))
from transformers import PreTrainedTokenizerFast  # noqa: E402
tok = PreTrainedTokenizerFast.from_pretrained(str(Path.home()/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
corpus = ''
for p in sorted(Path(os.environ.get('FE_DOC_TREE', str(Path.home()/'src/wt/ds41-og')), 'omlx').rglob('*.py')):
    corpus += p.read_text(errors='ignore') + '\n'
    if len(corpus) > 3_000_000:
        break
ids = tok.encode(corpus, add_special_tokens=False)


def doc(n, salt):
    # a user message of about n tokens (chat template adds ~10)
    start = (salt * 7919) % (len(ids) - n - 1)
    return tok.decode(ids[start:start + n - 12])


def stream(body, marks=None):
    u = urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    t0 = time.time()
    conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f'{resp.status} {resp.read()[:200]}')
    text, times, buf = [], [], b''
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if data == b'[DONE]':
            break
        chunk = json.loads(data)
        for choice in chunk.get('choices', []):
            piece = (choice.get('delta') or {}).get('content') or (choice.get('delta') or {}).get('reasoning_content') or ''
            if piece:
                now = time.time()
                text.append(piece)
                times.append(now)
                if marks is not None:
                    marks.append(now)
    conn.close()
    return dict(t0=t0, text=''.join(text), times=times, ttft=(times[0] - t0) if times else None)


def body(content, n):
    return dict(model='ds41-og', stream=True, temperature=0, max_tokens=n,
                messages=[dict(role='user', content=content)])


A_BODY = body('Write a long, detailed technical essay about memory hierarchies in GPUs. ' + doc(600, 1), A_TOKENS)
out = OUT.open('a')
alone_a = stream(A_BODY)
for size in B_SIZES:
    for rep in range(REPS):
        salt = 100 + rep + 10 * size + sum(map(ord, label))
        b_body = body(f'[{label} B {size} rep {rep}]\n' + doc(size, salt), 16)
        # concurrent first (B's prompt is fresh for the box and the Mac), then B alone (identity reference)
        # concurrent: A streams; B starts after A's 40th chunk
        a_marks, res = [], {}
        ta = threading.Thread(target=lambda: res.__setitem__('a', stream(A_BODY, a_marks)))
        ta.start()
        while len(a_marks) < 40 and ta.is_alive():
            time.sleep(0.002)
        b_start = time.time()
        res['b'] = stream(b_body)
        ta.join()
        alone_b = stream(b_body)
        a = res['a']
        gaps = [(t1 - t0) * 1000 for t0, t1 in zip(a['times'], a['times'][1:])]
        # gaps of A while B was being admitted: from B's start to B's first token + 0.2 s
        b_first = res['b']['times'][0] if res['b']['times'] else b_start
        win = [(t1 - t0) * 1000 for t0, t1 in zip(a['times'], a['times'][1:]) if t1 >= b_start and t0 <= b_first + 0.2]
        base_gaps = sorted(gaps[:35])
        rec = dict(label=label, b_size=size, rep=rep, same_a=a['text'] == alone_a['text'], same_b=res['b']['text'] == alone_b['text'],
                   b_ttft_alone=round(alone_b['ttft'], 3), b_ttft_conc=round(res['b']['ttft'], 3),
                   a_gap_median_before=round(base_gaps[len(base_gaps) // 2], 1), a_max_gap_admission=round(max(win), 1) if win else None,
                   a_gaps_admission=[round(g, 1) for g in win], a_tokens=len(a['times']))
        out.write(json.dumps(rec) + '\n'); out.flush()
        print(json.dumps(rec), flush=True)
