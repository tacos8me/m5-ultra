"""Compare replay digests + times of two ttft2_replay_prof labels."""
import json, sys
from pathlib import Path
D = Path.home()/'llm/ds41/ttft2'
def load(label):
    dig, tim = {}, {}
    for line in (D/(label + '.jsonl')).read_text().splitlines():
        r = json.loads(line)
        if r.get('kind') == 'digest':
            dig[r['case']] = (r['state'], r['prime'])
        elif r.get('kind') == 'time':
            tim[r['case']] = r
    return dig, tim
(da, ta), (db, tb) = load(sys.argv[1]), load(sys.argv[2])
same = sum(1 for c in da if c in db and da[c] == db[c])
print(f'digests identical {same}/{len(da)}' + ('' if same == len(da) else '  DIFF: ' + ', '.join(c for c in da if da[c] != db.get(c))))
for c in ta:
    if c in tb:
        a, b = ta[c], tb[c]
        print(f"{c:>14}  plan {len(a['plan'])} seg  total {a['total_ms']:7.1f} -> {b['total_ms']:7.1f} ms ({b['total_ms'] - a['total_ms']:+.1f})"
              f"   segs {a['seg_layers_prime_ms']} -> {b['seg_layers_prime_ms']}")
