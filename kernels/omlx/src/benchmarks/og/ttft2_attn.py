"""ds41-ttft2: attention-side attribution in the replay (synced wrappers around the pieces), 128/24 rows."""
import os, sys, time, json, statistics, functools
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttft2_replay_prof.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'ttft2_replay_prof.py', 'exec'))
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
lm = load()
T = {}
ROWS = [0]
def wrap(owner, name, label, rows_of=None):
    fn = getattr(owner, name)
    @functools.wraps(fn)
    def inner(*a, **k):
        arrs = [x for x in list(a) + list(k.values()) if isinstance(x, mx.array)]
        mx.eval(arrs)
        t = time.perf_counter()
        out = fn(*a, **k)
        mx.eval(out)
        T.setdefault((label, ROWS[0]), []).append(time.perf_counter() - t)
        return out
    setattr(owner, name, inner)
blk = language.Block.__call__
def blk_rows(self, h, *a, **k):
    ROWS[0] = h.shape[1]
    return blk(self, h, *a, **k)
language.Block.__call__ = blk_rows
wrap(glm_fast, 'deepseek_v41_packed_attention', 'packed_attention')
wrap(language, 'rope_range', 'rope_range')
wrap(language, 'pack_activation', 'pack_activation')
wrap(language.Indexer, '__call__', 'indexer')
wrap(language.Attention, '_input_projections', 'in_proj')
wrap(language, 'hc_mixes', 'hc_mixes')
wrap(language, 'hc_pre_norm', 'hc_pre_norm')
wrap(language, 'hc_post', 'hc_post')
wrap(language.MoE, '__call__', 'moe')
wrap(language.Attention, '__call__', 'attn_total')
wrap(language.RMSNorm, '__call__', 'rmsnorm')
wrap(language, 'packed_index_topk', 'index_topk')
wrap(language, 'packed_index_scores', 'index_scores')
for spec in os.environ.get('RP_CASES', '8217@8217,131162@8290,524400@8600').split(','):
    st = case_state(spec)
    for rep in range(3):
        T.clear()
        drop(lm, run_import(lm, *st[:3], st[3]))
    emit(dict(kind='attn_parts', case=spec, parts={f'{k[0]}@{k[1]}': dict(n=len(v), sum_ms=round(1000 * sum(v), 2)) for k, v in sorted(T.items())}))
