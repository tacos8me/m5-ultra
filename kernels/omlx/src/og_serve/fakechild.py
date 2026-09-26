"""A fake omlx child for CPU-only supervisor tests (mimics the SSE shapes of omlx chat/text completions).

The "model" emits a fixed token sequence 0..N-1 (N = max_tokens, default 40); token t reads
f'w{t} '. With chat_template_kwargs.enable_thinking the first 10 tokens go to reasoning_content.
Body field `fake_fail_after` (honoured only by --mode og) makes it stop after that many tokens with
the og worker's resume-marker error; `fake_die_after` exits the process mid-stream. A request with
header x-ds41-resume replays body.ds41_resume.output first, like og_resume.
"""
import argparse
import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

MARK = 'ds41-og-resume:'
ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True)
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--delay', type=float, default=0.01)
args = ap.parse_args()
if args.mode == 'og' and os.environ.get('FAKE_OG_NOSTART') and os.path.exists(os.environ['FAKE_OG_NOSTART']):
    raise SystemExit(3)  # like an og worker whose warm-up through the box fails
if args.mode == 'q3' and os.environ.get('FAKE_Q3_NOSTART') and os.path.exists(os.environ['FAKE_Q3_NOSTART']):
    raise SystemExit(3)
if args.mode == 'q3' and os.environ.get('FAKE_Q3_START_DELAY'):
    time.sleep(float(os.environ['FAKE_Q3_START_DELAY']))  # a slow model load: the supervisor sends keepalives
app = FastAPI()
STATS = dict(requests=0, resumed=0)


@app.get('/health')
async def health():
    return {'status': 'ok', 'mode': args.mode}


@app.get('/v1/models')
async def models():
    return {'data': [{'id': 'ds41-og' if args.mode == 'og' else 'ds41-q3g128'}]}


@app.get('/fake/stats')
async def stats():
    return STATS


def text_of(t):
    return f'w{t} '


def marker(output, prompt_tokens):
    info = dict(kind='box', reason='fake box lost', prompt_tokens=prompt_tokens, output=output)
    return f'503: ds41-og: box lost [{MARK}{json.dumps(info, separators=(",", ":"))}]'


def has_image(body):
    # FAKE_OG_VISION=1: an og worker with native vision (the box runs the tower) serves images like text.
    if os.environ.get('FAKE_OG_VISION') == '1':
        return False
    return 'image_url' in json.dumps(body.get('messages') or []) or '"image"' in json.dumps(body.get('messages') or [])


@app.post('/v1/messages')
async def messages(request: Request):
    body = await request.json()
    if args.mode == 'og' and has_image(body):
        return JSONResponse({'type': 'error', 'error': {'message': 'ds41-og serves text only'}}, status_code=500)
    return {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': body.get('model'),
            'content': [{'type': 'text', 'text': 'saw it'}], 'stop_reason': 'end_turn',
            'usage': {'input_tokens': 9, 'output_tokens': 2}}


@app.post('/v1/chat/completions')
@app.post('/v1/completions')
async def complete(request: Request):
    body = await request.json()
    STATS['requests'] += 1
    if args.mode == 'og' and has_image(body):
        # What the real og worker does with an image: a stream error (the live bug).
        if body.get('stream'):
            async def broken():
                yield 'data: {"error": {"message": "ds41-og serves text only", "type": "server_error"}}\n\n'
            return StreamingResponse(broken(), media_type='text/event-stream')
        return JSONResponse({'error': {'message': 'ds41-og serves text only'}}, status_code=500)
    chat = request.url.path.endswith('/chat/completions')
    total = int(body.get('max_tokens') or 40)
    replay = []
    if request.headers.get('x-ds41-resume'):
        replay = list(body['ds41_resume']['output'])
        STATS['resumed'] += 1
    think = bool((body.get('chat_template_kwargs') or {}).get('enable_thinking'))
    fail_after = body.get('fake_fail_after') if args.mode == 'og' else None
    die_after = body.get('fake_die_after') if args.mode == 'og' else None
    prompt_tokens = 17 + len(replay)
    rid = f'chatcmpl-{args.mode}{int(time.time()*1e6) % 10**8}'
    model = body.get('model')

    def chunk(delta=None, finish=None, text=None):
        if chat:
            choice = dict(index=0, delta=delta or {}, finish_reason=finish)
            return dict(id=rid, object='chat.completion.chunk', created=1, model=model, choices=[choice])
        return dict(id=rid, object='text_completion', created=1, model=model,
                    choices=[dict(index=0, text=text or '', logprobs=None, finish_reason=finish)])

    def piece(t):
        if not chat:
            return chunk(text=text_of(t))
        key = 'reasoning_content' if think and t < 10 else 'content'
        return chunk({key: text_of(t)})

    if not body.get('stream'):
        if fail_after is not None:
            return JSONResponse({'error': {'message': marker(list(range(fail_after)), prompt_tokens)}}, status_code=503)
        reasoning = ''.join(text_of(t) for t in range(total) if think and t < 10)
        content = ''.join(text_of(t) for t in range(total) if not (think and t < 10))
        usage = dict(prompt_tokens=prompt_tokens, completion_tokens=total, total_tokens=prompt_tokens + total)
        if chat:
            message = dict(role='assistant', content=content)
            if reasoning:
                message['reasoning_content'] = reasoning
            return dict(id=rid, object='chat.completion', model=model, usage=usage,
                        choices=[dict(index=0, message=message, finish_reason='length')])
        return dict(id=rid, object='text_completion', model=model, usage=usage,
                    choices=[dict(index=0, text=reasoning + content, finish_reason='length')])

    async def events():
        if chat:
            yield f'data: {json.dumps(chunk(dict(role="assistant")))}\n\n'
        for _ in range(int(body.get('fake_keepalive') or 0)):  # omlx's keepalive frame, byte for byte
            yield ('data: {"id":"' + rid + '","object":"chat.completion.chunk","created":0,"model":"keepalive",'
                   '"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n')
        for t in range(total):
            if fail_after is not None and t == fail_after:
                yield f'data: {json.dumps({"error": {"message": marker(list(range(t)), prompt_tokens), "type": "server_error"}})}\n\n'
                yield 'data: [DONE]\n\n'
                return
            if die_after is not None and t == die_after:
                os._exit(3)
            if t >= len(replay):
                await asyncio.sleep(args.delay)
            yield f'data: {json.dumps(piece(t))}\n\n'
        yield f'data: {json.dumps(chunk({}, "length"))}\n\n'
        if (body.get('stream_options') or {}).get('include_usage'):
            usage = dict(prompt_tokens=prompt_tokens, completion_tokens=total, total_tokens=prompt_tokens + total)
            yield f'data: {json.dumps(dict(id=rid, object="chat.completion.chunk", model=model, choices=[], usage=usage))}\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(events(), media_type='text/event-stream')


uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
