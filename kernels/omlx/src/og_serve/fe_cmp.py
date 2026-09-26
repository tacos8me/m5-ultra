"""Compare fe_bench identity outputs of two or more labels (full content, reasoning and tool calls).

  python og_serve/fe_cmp.py prod-id2 post-deploy     # no output + "12/12 identical" = bit-identical text

Labels are ~/llm/ds41/fe/<label>.jsonl files written by fe_bench.py --identity (same FE_DOC_TREE).
"""
import json
import os
from pathlib import Path
import sys

FE = Path(os.environ.get('FE_LOGS', str(Path.home()/'llm/ds41/fe')))


def main(labels):
    out = {}
    for label in labels:
        for line in (FE/f'{label}.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r.get('kind') == 'identity':
                out.setdefault(r['case'], {})[label] = r
    same = 0
    for case, per in out.items():
        ref, ok = per.get(labels[0]), True
        for label in labels[1:]:
            r = per.get(label)
            if ref is None or r is None or ref.get('error') or r.get('error'):
                print(f'{case:16s} {label:12s} missing or error')
                ok = False
                continue
            for field in ('content', 'reasoning', 'tools'):
                a, b = ref.get(field) or '', r.get(field) or ''
                if a != b:
                    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                    print(f'{case:16s} {label:12s} {field} differs at char {i}: {a[max(0, i - 30):i + 30]!r} || '
                          f'{b[max(0, i - 30):i + 30]!r}')
                    ok = False
        same += ok
    print(f'{same}/{len(out)} identical')
    return same == len(out)


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1:]) else 1)
