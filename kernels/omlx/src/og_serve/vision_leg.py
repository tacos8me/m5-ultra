"""og-box vision leg (one guarded Mac GPU lease; no q3 anywhere): the og worker of this tree (images on the box) on
an image QA set with objective answers, cache exactness (repeats, turn 2), then the production-tree og worker (text
must be identical). Stops the served llama-swap ds41 by its own PID, holds gpu.lock for the whole leg (children run
without gpu-exec; og_server keeps its 245 GiB guard), TERMs each worker and waits, then releases the lock."""
import base64, io, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE/'benchmarks/og'))
sys.path.insert(0, str(HERE))
from lease import stop_served  # noqa: E402
from sv_leg import hold_gpu_lock  # noqa: E402

OUT = HOME/'llm/ds41/og-vision'
LOG = OUT/f'vision-leg-{time.strftime("%H%M%S")}.jsonl'
PY = str(HOME/'llm/.venv-ds41-omlx-tiles/bin/python')
PORT = 12190
T0 = time.monotonic()
LIMIT = float(os.environ.get('LEG_S', '820'))


def log(obj):
    obj = dict(obj, t=round(time.monotonic() - T0, 1))
    print(json.dumps(obj), flush=True)
    with LOG.open('a') as f:
        f.write(json.dumps(obj) + '\n')


def png(draw):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (900, 500), 'white')
    draw(ImageDraw.Draw(img), ImageFont)
    buf = io.BytesIO(); img.save(buf, 'PNG')
    return buf.getvalue()


def data_url(raw, kind='png'):
    return f'data:image/{kind};base64,' + base64.b64encode(raw).decode()


EX = Path('/tmp')
carrots = (EX/'carrots.jpeg').read_bytes()
corn = (EX/'corn.jpeg').read_bytes()
ocr = png(lambda d, F: d.text((60, 180), 'SPLIT NV 4217', fill='black', font=F.load_default(size=110)))
circles = png(lambda d, F: [d.ellipse((60 + i * 160, 180, 180 + i * 160, 300), fill='red') for i in range(5)])
red = png(lambda d, F: d.rectangle((0, 0, 900, 500), fill='red'))
OFFICIAL = [{'type': 'text', 'text': '请按“第一张、第二张”的顺序回答：第一张图'},
            {'type': 'image_url', 'image_url': {'url': data_url(carrots, 'jpeg')}},
            {'type': 'text', 'text': '和第二张图'},
            {'type': 'image_url', 'image_url': {'url': data_url(corn, 'jpeg')}},
            {'type': 'text', 'text': '中分别是什么食材？它们通常食用的部位分别是什么？'}]
# (name, content, check(answer) -> bool): objective answers; q3 answered "Red" to og-serve's red-image check
IMAGE_QA = [
    ('official-two-images', OFFICIAL, lambda a: '胡萝卜' in a and '玉米' in a),
    ('carrots-what', ('carrots', 'What vegetable is shown in this image? Answer with one word.'), lambda a: 'carrot' in a.lower()),
    ('corn-what', ('corn', 'What vegetable is shown in this image? Answer with one word.'), lambda a: 'corn' in a.lower() or 'maize' in a.lower()),
    ('ocr', ('ocr', 'What text is written in this image? Reply with the text exactly.'), lambda a: 'SPLITNV4217' in a.upper().replace(' ', '')),
    ('circles-count', ('circles', 'How many red circles are in this image? Answer with a number only.'), lambda a: '5' in a or 'five' in a.lower()),
    ('red-color', ('red', 'What color is this image? Answer with one word.'), lambda a: 'red' in a.lower()),
]
PICS = dict(carrots=(carrots, 'jpeg'), corn=(corn, 'jpeg'), ocr=(ocr, 'png'), circles=(circles, 'png'), red=(red, 'png'))


def qa_content(spec):
    if isinstance(spec, list):
        return spec
    raw, kind = PICS[spec[0]]
    return image_msg(raw, kind, spec[1])


TEXT = ['Explain in two sentences what a hash table is.', 'Write a haiku about the sea.']


def chat(model, content, max_tokens=160, messages=None):
    body = dict(model=model, messages=messages or [{'role': 'user', 'content': content}], max_tokens=max_tokens,
                temperature=0, chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            reply = json.load(r)
        return reply['choices'][0]['message'].get('content') or '', round(time.perf_counter() - t, 2), None
    except urllib.error.HTTPError as e:
        return None, round(time.perf_counter() - t, 2), e.read().decode()[:300]


def image_msg(raw, kind, text):
    return [{'type': 'image_url', 'image_url': {'url': data_url(raw, kind)}}, {'type': 'text', 'text': text}]


def start(label, cmd, env):
    logf = (OUT/'logs'/f'{label}.log').open('w')
    p = subprocess.Popen(cmd, env=dict(os.environ, **env), stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 420
    model = None
    while time.monotonic() < deadline and p.poll() is None:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=2) as r:
                model = json.load(r)['data'][0]['id']
            if label.startswith('og'):
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                    if r.status != 200:
                        raise OSError
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    if model is None:
        stop(p, label)
        raise RuntimeError(f'{label} did not come up (rc {p.poll()})')
    log(dict(event='up', worker=label, model=model, pid=p.pid))
    return p, model


def stop(p, label):
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
    log(dict(event='stopped', worker=label, rc=p.returncode))


def og_worker(label, tree):
    env = dict(DS41_TREE=str(tree), DS41_OG_HOME=str(OUT), DS41_OG_LEASE_S=str(int(LIMIT)), DS41_WARMUP='1')
    return start(label, [PY, '-u', str(tree/'og_serve/og_server.py'), '--host', '127.0.0.1', '--port', str(PORT)], env)


def main():
    stop_served(log)
    fd = hold_gpu_lock(180)
    log(dict(event='lock'))
    results = {}
    try:
        # 1. og with images (this tree)
        p, model = og_worker('og-vision', TREE)
        try:
            for name, spec, ok in IMAGE_QA:
                ans, sec, err = chat(model, qa_content(spec), max_tokens=200)
                results[name] = ans
                log(dict(worker='og', case=name, correct=bool(ans) and ok(ans), answer=ans, s=sec, err=err))
            again = chat(model, OFFICIAL, max_tokens=200)
            log(dict(worker='og', case='official-repeat', identical=again[0] == results['official-two-images'], s=again[1]))
            turn2 = [{'role': 'user', 'content': OFFICIAL},
                     {'role': 'assistant', 'content': results['official-two-images'] or ''},
                     {'role': 'user', 'content': '第二张图里的食材是什么颜色？一个词回答。'}]
            t2 = chat(model, None, messages=turn2, max_tokens=64)
            t2b = chat(model, None, messages=turn2, max_tokens=64)
            log(dict(worker='og', case='turn2', answer=t2[0], s=t2[1], correct=bool(t2[0]) and ('黄' in t2[0]),
                     repeat_identical=t2[0] == t2b[0], repeat_s=t2b[1], err=t2[2]))
            for i, q in enumerate(TEXT):
                results[f'text{i}'] = chat(model, q, max_tokens=96)[0]
                log(dict(worker='og', case=f'text{i}', answer=results[f'text{i}']))
        finally:
            stop(p, 'og-vision')
        # 2. the production tree's og worker: text must be identical; images were rejected there
        if time.monotonic() - T0 < LIMIT - 240:
            p, model = og_worker('og-base', HOME/'src/wt/ds41-og')
            try:
                for i, q in enumerate(TEXT):
                    base = chat(model, q, max_tokens=96)
                    log(dict(worker='og-base', case=f'text{i}', identical=base[0] == results[f'text{i}'], answer=base[0]))
            finally:
                stop(p, 'og-base')
    finally:
        os.close(fd)
        log(dict(event='released'))


if __name__ == '__main__':
    main()
