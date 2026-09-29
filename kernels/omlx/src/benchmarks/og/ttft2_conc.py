"""ds41-ttft2: decode steps of a running request while an admission replay runs:
  sequential  replay (blocking) then steps          -- today's admission stall
  slice:L     L replay layers (async) before each step, same stream
  sside:L     L replay layers per step on a side GPU stream
Partial load; synthetic 5-row boundaries on a real imported 8600-token cache. Gap = time between step ends."""
import os, sys, time, json, statistics
sys.argv = [sys.argv[0]]
os.environ['DS41_OG_SEG_CACHE'] = '0'
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
from omlx.patches.deepseek_v41.quantization import pack_activation
lm = load()
c = lm._config
dec = run_import(lm, *case_state('8600@8600')[:3], case_state('8600@8600')[3])
ROWS = int(os.environ.get('RP_ROWS', '5'))

def boundary(seed):
    mx.random.seed(seed)
    b = dict(h=mx.random.normal((1, ROWS, 4, c.dim)).astype(mx.bfloat16),
             pre=mx.random.uniform(0.2, 1.0, (1, ROWS, 4)).astype(mx.float32),
             kv=pack_activation(mx.random.normal((1, ROWS, c.head_dim)).astype(mx.bfloat16), 4, 16, True),
             index=pack_activation(mx.random.normal((1, ROWS, c.index_head_dim)).astype(mx.bfloat16), 4))
    mx.eval(list(b.values()))
    return b
B = [boundary(i) for i in range(4)]

def step(k):
    logits, hidden = lm.forward_boundary(**B[k % 4], cache=dec, start=dec[0].size(), verify=True)
    mx.eval(mx.argmax(logits[0], -1), hidden)
    lm.rollback_boundary(dec, ROWS)

def run(mode, new, per):
    side = mx.new_stream(mx.gpu) if mode == 'sside' else None
    ends, t0 = [], time.perf_counter()
    for k in range(3):
        step(k); ends.append(time.perf_counter())
    t_start = time.perf_counter()
    gen = lm.import_state_steps(new[1], new[2], new[0], identity=new[3])
    done, t_done, k = False, None, 3
    if mode == 'sequential':
        for _ in gen:
            pass
        done, t_done = True, time.perf_counter()
    while k < 40 and (not done or k < 3 + 12):
        if not done:
            n = 0
            while n < per:
                try:
                    if side is not None:
                        with mx.stream(side):
                            next(gen)
                    else:
                        next(gen)
                    n += 1
                except StopIteration as fin:
                    done, t_done = True, time.perf_counter()
                    cache, rows = fin.value
                    break
        step(k); ends.append(time.perf_counter()); k += 1
    gaps = [1000 * (b - a) for a, b in zip(ends, ends[1:])]
    emit(dict(kind='conc', mode=f'{mode}:{per}', case=os.environ.get('RP_NEW', '8217@8217'),
              replay_done_ms=round(1000 * (t_done - t_start), 1), max_gap_ms=round(max(gaps), 1),
              gaps=[round(g, 1) for g in gaps[:20]]))
    mx.clear_cache()

for spec in os.environ.get('RP_NEWS', '8217@8217,8600@8600').split(','):
    new = case_state(spec)
    os.environ['RP_NEW'] = spec
    for w in range(3):
        step(w)
    for mode, per in (('sequential', 0), ('slice', 4), ('slice', 8), ('slice', 12), ('sside', 4), ('sside', 8), ('sequential', 0)):
        run(mode, new, per)
