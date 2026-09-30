"""Per-round GPU idle split from a Metal System Trace of the og worker (engine-loop-pipeline go/no-go).

  python benchmarks/og/xt_rounds.py TRACE --c {1,2,4} [--proc python] [--merge-us 150] [--fwd-ms 8] [--skip-ms 300]

Reads the metal-gpu-intervals table (xcrun xctrace export; CPU only), takes the worker's depth-0 GPU
intervals, merges them into busy runs, and joins runs separated by idle gaps < --merge-us into blocks.
Blocks of at least --fwd-ms are verify forwards (c4 fused pair ~22 ms, c2 one request ~15 ms, c1 ~21 ms);
the shorter blocks after a forward are its draft phase (DSpark stages + head + Markov, ~4-6 ms). Gaps:
  fwd_to_draft  forward end -> next block (accept, rollback, DSpark build): THREADS verdict class (c)
  in_draft      between draft-phase blocks (stage async_evals, draft finish sync)
  handoff       draft phase end -> next forward start: within-step (a) or cross-step (b)
c2 and c4 run two verify passes per scheduler step (A then B, or pair 1 then pair 2), so handoffs alternate
a, b. The GPU alone cannot tell which is which: both alternating series are reported, the smaller median is
taken as within-step (a) and the larger as cross-step (b), which also pays replace_rows + emit + the
scheduler gap. At c1 the handoff is the box round trip (one series).
Decision (THREADS 2a / verify:engine-loop-pipeline): GO if median(a) >= 1.5 ms per round. Check the printed
block sequence first: a forward split by an internal gap >= --merge-us shows up as two long blocks in a row.
"""
import argparse
import json
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


def intervals(trace, proc):
    xml = subprocess.run(['xcrun', 'xctrace', 'export', '--input', trace, '--xpath',
                          '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'],
                         capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(xml)
    ids = {}

    def val(el):
        if el is None:
            return None
        if 'ref' in el.attrib:
            return ids.get(el.attrib['ref'])
        v = (el.text, el.attrib.get('fmt', el.text))
        if 'id' in el.attrib:
            ids[el.attrib['id']] = v
        return v

    schema = [c.find('mnemonic').text for c in root.iter('col')]
    out = []
    for row in root.iter('row'):
        rec = {}
        for name, el in zip(schema, list(row)):
            rec[name] = val(el)
            for sub in el.iter():
                if 'id' in sub.attrib and sub.attrib['id'] not in ids:
                    ids[sub.attrib['id']] = (sub.text, sub.attrib.get('fmt', sub.text))
        p = rec.get('process')
        if proc not in ((p or ('', ''))[1] or ''):
            continue
        depth = rec.get('event-depth')
        if depth and depth[1] not in (None, '0'):
            continue
        start = int(rec['start'][0])
        out.append((start, start + int(rec['duration'][0])))
    return sorted(out)


def blocks_of(ivs, merge_ns):
    runs = []
    for s, e in ivs:
        if runs and s <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    blocks = []
    for s, e in runs:
        if blocks and s - blocks[-1][1] < merge_ns:
            blocks[-1][1] = max(blocks[-1][1], e)
        else:
            blocks.append([s, e])
    return runs, blocks


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--c', type=int, required=True, choices=(1, 2, 4), help='concurrency the trace was taken at')
    ap.add_argument('--proc', default='python')
    ap.add_argument('--merge-us', type=float, default=150)
    ap.add_argument('--fwd-ms', type=float, default=8)
    ap.add_argument('--skip-ms', type=float, default=300, help='drop the first ms of the trace (attach transient)')
    ap.add_argument('--show', type=int, default=24, help='print the first N blocks')
    a = ap.parse_args()
    ivs = intervals(a.trace, a.proc)
    if not ivs:
        print(json.dumps(dict(error='no GPU intervals for process', proc=a.proc)))
        return 1
    t0 = ivs[0][0] + a.skip_ms * 1e6
    ivs = [(s, e) for s, e in ivs if s >= t0]
    runs, blocks = blocks_of(ivs, a.merge_us * 1e3)
    span = blocks[-1][1] - blocks[0][0]
    busy = sum(e - s for s, e in runs)
    kind = ['F' if (e - s) >= a.fwd_ms * 1e6 else 'd' for s, e in blocks]
    gaps = {'fwd_to_draft': [], 'in_draft': [], 'handoff': []}
    handoffs = []
    for i in range(1, len(blocks)):
        g = (blocks[i][0] - blocks[i - 1][1]) / 1e6
        if kind[i - 1] == 'F':
            gaps['fwd_to_draft'].append(g)
        elif kind[i] == 'F':
            gaps['handoff'].append(g)
            handoffs.append(g)
        else:
            gaps['in_draft'].append(g)
    small_ms = sum(e - s for s, e in blocks) / 1e6 - busy / 1e6  # idle hidden inside blocks (< merge-us gaps)
    forwards = kind.count('F')
    even, odd = handoffs[0::2], handoffs[1::2]
    a_series, b_series = sorted((even, odd), key=lambda s: med(s) if s else 1e9)
    per_pass = {k: dict(n=len(v), median_ms=med(v), total_ms_per_pass=round(sum(v) / max(1, forwards), 3))
                for k, v in gaps.items()}
    print(json.dumps(dict(trace=a.trace, proc=a.proc, span_ms=round(span / 1e6, 1), busy_pct=round(100 * busy / span, 1),
                          blocks=len(blocks), forwards=forwards,
                          forward_ms_median=med([(e - s) / 1e6 for (s, e), k in zip(blocks, kind) if k == 'F']),
                          draft_phase_block_ms_median=med([(e - s) / 1e6 for (s, e), k in zip(blocks, kind) if k == 'd']),
                          sub_merge_idle_ms_per_pass=round(small_ms / max(1, forwards), 3), gaps=per_pass)))
    print(json.dumps(dict(handoff_even=dict(n=len(even), median_ms=med(even)), handoff_odd=dict(n=len(odd), median_ms=med(odd)),
                          within_step_a_ms=med(a_series) if a.c > 1 else None,
                          cross_step_b_ms=med(b_series) if a.c > 1 else None,
                          go_engine_loop_pipeline=(med(a_series) or 0) >= 1.5 if a.c > 1 else None,
                          note='a = smaller alternating handoff series (2 passes per scheduler step at c2/c4)')))
    seq = []
    for i, ((s, e), k) in enumerate(zip(blocks[:a.show], kind[:a.show])):
        gap = round((s - blocks[i - 1][1]) / 1e6, 3) if i else None
        seq.append(f'{"" if gap is None else f"[{gap}] "}{k}{round((e - s) / 1e6, 2)}')
    print('sequence (ms, [idle gap] Forward/draft block):', ' '.join(seq))
    return 0


if __name__ == '__main__':
    sys.exit(main())
