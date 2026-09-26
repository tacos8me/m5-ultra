"""Synthetic multi-turn conversations for the ds41 front-end tests and benchmarks.

Deterministic per seed. Text comes from this tree's own source files (code, prose,
unicode, whitespace runs), with extra edge cases mixed in: CJK and emoji, digit runs,
CRLF, trailing spaces, special-token strings inside content, empty contents.
"""
import json
import os
from pathlib import Path
import random

# A fixed corpus tree (default: the production worktree at f56f7ffa) so prompts do not change with edits here.
_FIXED = Path.home()/'src/wt/ds41-og'
ROOT = Path(os.environ.get('FE_DOC_TREE', str(_FIXED if _FIXED.is_dir() else Path(__file__).resolve().parent.parent)))
_CORPUS = None
EDGE = ['', ' ', '\n', '\r\n\r\n', '   \t  ', '1234567890123', '中文字符测试 日本語のテキスト カタカナ', 'emoji 🙂🚀 done',
        '<｜User｜>', '<｜Assistant｜>', '<｜end▁of▁sentence｜>', '</think>', '<think>', '<｜place▁holder▁no▁7｜>',
        '｜DSML｜', 'trailing   ', '\n\n\n', 'a' * 300, ' '.join(['x'] * 50), ' nbsp em', 'é combining']


def corpus():
    global _CORPUS
    if _CORPUS is None:
        parts = []
        for path in sorted((ROOT/'omlx').rglob('*.py')):
            try:
                parts.append(path.read_text())
            except (OSError, UnicodeDecodeError):
                continue
        # The template rejects the image placeholder inside text (images are content blocks).
        _CORPUS = ''.join(parts).replace('<｜deepseek_image｜>', '[image placeholder]')
    return _CORPUS


def chunk(rng, chars):
    text = corpus()
    at = rng.randrange(0, max(1, len(text) - chars))
    out = text[at:at + chars]
    if rng.random() < 0.3:
        cut = rng.randrange(0, len(out) + 1)
        out = out[:cut] + rng.choice(EDGE) + out[cut:]
    return out


TOOLS = [
    {'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file from disk.',
                                      'parameters': {'type': 'object', 'properties': {
                                          'path': {'type': 'string', 'description': 'absolute path'},
                                          'limit': {'type': 'integer'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'bash', 'description': 'Run a shell command. 中文 ok.',
                                      'parameters': {'type': 'object', 'properties': {
                                          'command': {'type': 'string'}, 'timeout': {'type': 'number'}},
                                          'required': ['command']}}},
]


def conversation(seed, target_chars, *, tools=None, reasoning=None, system=None, final_user=True):
    """(messages, tools): an OpenAI chat conversation of about target_chars characters."""
    rng = random.Random(seed)
    tools = rng.random() < 0.5 if tools is None else tools
    reasoning = rng.random() < 0.5 if reasoning is None else reasoning
    system = rng.random() < 0.7 if system is None else system
    messages = []
    if system:
        messages.append({'role': 'system', 'content': chunk(rng, rng.randrange(0, 4000))})
    size = sum(len(m['content']) for m in messages)
    call = 0
    while size < target_chars:
        room = max(64, target_chars - size)
        text = chunk(rng, min(room, rng.choice([50, 400, 3000, 20000, 80000])))
        messages.append({'role': 'user', 'content': text})
        size += len(text)
        if size >= target_chars and final_user:
            break
        reply = {'role': 'assistant', 'content': chunk(rng, rng.randrange(0, 2000))}
        if reasoning and rng.random() < 0.8:
            reply['reasoning_content'] = chunk(rng, rng.randrange(0, 1500))
        if tools and rng.random() < 0.6:
            calls = []
            for _ in range(rng.choice([1, 1, 2, 3])):
                call += 1
                name = rng.choice(['read_file', 'bash'])
                args = {'path': '/tmp/' + str(call)} if name == 'read_file' else {'command': chunk(rng, 80)}
                calls.append({'id': f'call_{call}', 'type': 'function',
                              'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=rng.random() < .5)}})
            reply['tool_calls'] = calls
            messages.append(reply)
            for c in calls:
                result = chunk(rng, min(room, rng.choice([100, 2000, 30000])))
                messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': result})
                size += len(result)
            continue
        messages.append(reply)
        size += len(reply['content'])
    if final_user and messages[-1]['role'] != 'user':
        messages.append({'role': 'user', 'content': chunk(rng, 200)})
    return messages, (TOOLS if tools else None)


def next_turn(messages, seed, *, reply_chars=300, user_chars=200):
    """The conversation after one more assistant reply and a new user message (turn N+1)."""
    rng = random.Random(seed)
    reply = {'role': 'assistant', 'content': chunk(rng, reply_chars)}
    if rng.random() < 0.5:
        reply['reasoning_content'] = chunk(rng, reply_chars)
    return messages + [reply, {'role': 'user', 'content': chunk(rng, user_chars)}]
