"""Bitwise gate on a real box session (partial load: real layer-20 KV/index rows and real boundaries):
the Mac forward with the attention fusions on vs off must give identical logits and DSpark hidden,
k = 1..5 rows, several positions. Production must be idle (one short box session)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
os.environ.setdefault('DS41_TREE', str(HOME/'src/wt/ds41-attn'))
os.environ.setdefault('PROF_OUT', str(HOME/'llm/ds41/attn/fwd_real.jsonl'))
os.environ.setdefault('PROF_LABEL', 'fwd-real')
os.environ.setdefault('PROF_LEASE_S', '900')
sys.path.insert(0, str(HOME/'src/wt/ds41-prof/benchmarks/og'))
import prof_mac as P  # noqa: E402
import mlx.core as mx  # noqa: E402
from fwd_ab import config  # noqa: E402


def main():
    lm = P.load('partial')
    n = int(os.environ.get('REAL_CTX', '8192'))
    tokens = json.loads((HOME/f'llm/ds41/split-wire/ids-{n}.json').read_text())
    s = P.Session(lm, tokens)
    P.emit(dict(event='session', context=n, open_s=round(s.open_s, 2)))
    from omlx.patches.deepseek_v41.pipe_wire import mlx_step
    rng = np.random.default_rng(0)
    bad = 0
    for rnd in range(int(os.environ.get('REAL_ROUNDS', '3'))):
        for rows in (1, 2, 3, 4, 5):
            start = s.cache[0].size()
            ids = [s.anchor] + [int(t) for t in rng.choice(tokens, rows - 1)]
            s.enc.send_step(ids, start)
            raw, _ = s.enc.recv_step()
            arrays = mlx_step(raw, rows)
            snap = [(list(item.cache), item.left_padding, item.lengths) for item in s.cache]
            outs = {}
            for new in (False, True):
                config(new)
                logits, hidden = lm.forward_boundary(**arrays, cache=s.cache, start=start, verify=True)
                mx.eval(logits, hidden)
                outs[new] = (np.array(logits.astype(mx.float32)), np.array(hidden.astype(mx.float32)))
                s.cache[0]._pipe1_verify = None
                for item, (c, lp, ln) in zip(s.cache, snap):
                    item.cache, item.left_padding, item.lengths = list(c), lp, ln
            same = all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(outs[False], outs[True]))
            diff = float(np.abs(outs[False][0] - outs[True][0]).max())
            bad += not same
            P.emit(dict(test='real_bitwise', round=rnd, rows=rows, start=start, identical=same, logits_maxabs=diff))
            # Commit one real token so the next round runs at a new position.
            config(True)
            s.step([s.anchor], verify=False)
    s.close()
    P.emit(dict(event='done', mismatches=bad))


if __name__ == '__main__':
    main()
