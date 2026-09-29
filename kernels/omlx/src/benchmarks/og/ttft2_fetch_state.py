"""Fetch real box encoder states (lean, streamed, box cache) for replay profiling; one session at a time.
usage: fetch_state.py N [N...]   -> ~/llm/ds41/ttft2/states/state-<N>.pkl"""
import os, pickle, sys, time, subprocess
from pathlib import Path
HOME = Path.home()
TREE = os.environ.get('DS41_TREE', str(HOME/'src/wt/ds41-ttft2'))
sys.path.insert(0, TREE)
from omlx.patches.deepseek_v41.pipe_wire import EncoderSession
from omlx.patches.deepseek_v41.pipe_session import open_remote
from transformers import PreTrainedTokenizerFast
OUT = HOME/'llm/ds41/ttft2/states'; OUT.mkdir(parents=True, exist_ok=True)
tok = PreTrainedTokenizerFast.from_pretrained(str(HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'))
text = ''
for p in sorted(Path(TREE, 'omlx').rglob('*.py')):
    text += p.read_text(errors='ignore') + '\n'
    if len(text) > 12_000_000:
        break
ids = tok.encode(text, add_special_tokens=False)
print('corpus tokens', len(ids), flush=True)
for n in map(int, sys.argv[1:]):
    if subprocess.run(['python3', str(HOME/'src/wt/ds41-prof/benchmarks/og/prof_idle.py')]).returncode:
        print('NOT IDLE, stop'); break
    reps = -(-n // len(ids))
    tokens = (ids * reps)[:n]
    tokens[0] = tok.convert_tokens_to_ids('<｜begin▁of▁sentence｜>') if n > 0 else tokens[0]
    enc = EncoderSession()
    try:
        t0 = time.time()
        tensors, manifest, open_s = open_remote(enc, tokens, f'ttft2-fetch-{n}', cache=True, state='lean', stream=True)
        info = dict(enc.open_info)
    finally:
        enc.close()
    saved = {k: (v[0], v[1], bytes(v[2])) for k, v in tensors.items()}
    with open(OUT/f'state-{n}.pkl', 'wb') as f:
        pickle.dump(dict(tokens=tokens, tensors=saved, manifest=manifest, info=info, identity=enc.identity), f)
    print(n, 'open_s %.3f' % open_s, {k: (v[0], v[1]) for k, v in tensors.items()}, {k: info[k] for k in info}, flush=True)
    print('manifest keys', {k: (v if k != 'layers' else len(v)) for k, v in manifest.items()}, flush=True)
    print('layer20', manifest['layers'][20], flush=True)
