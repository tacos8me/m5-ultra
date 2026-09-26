"""Per-stage TTFT breakdown and output identity for fe_window runs.

  python og_serve/fe_report.py --window fe-w1 --phases A,B,C,D --reference prod-base1

Joins each client record (fe_bench jsonl: wall-clock t0 + ttft) with the supervisor's
"ds41-fe sup" line and the worker's "ds41-fe trace" line (same wall clock), prints the median
of every stage per phase and case kind, and compares every identity case's full output
(content, reasoning, tool calls) against the reference label; worker digests (prompt ids and
the imported Mac state) are compared across phases for the same prompt.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

HOME = Path.home()
FE = HOME/'llm/ds41/fe'

STAGES = [  # (name, start key, end key); keys prefixed 's.' come from the supervisor, 'c.' from the client
    ('client->sup', 'c.t0', 's.recv'), ('sup body+parse', 's.recv', 's.parsed'), ('sup->worker', 's.sent', 'http_start'),
    ('worker body', 'http_start', 'body_done'), ('json+pydantic', 'body_done', 'handler'),
    ('extract', 'handler', 'count_start'), ('count tokens', 'count_start', 'count_end'),
    ('->preflight', 'count_end', 'preflight_start'), ('preflight', 'preflight_start', 'preflight_end'),
    ('->engine', 'preflight_end', 'engine_start'), ('executor wait', 'engine_start', 'pcm_start'),
    ('render+tokenize', 'pcm_start', 'pcm_end'), ('->add_request', 'pcm_end', 'add_request'),
    ('->scheduler', 'add_request', 'og.arrival'), ('->defer', 'og.arrival', 'og.deferred'),
    ('->job', 'og.deferred', 'og.job_start'), ('box ack', 'og.job_start', 'og.box_ack'),
    ('box prefill+state', 'og.box_ack', 'og.box_end'), ('job tail', 'og.box_end', 'og.job_end'),
    ('->prepare', 'og.job_end', 'og.prepare_start'), ('import arrays', 'og.prepare_start', 'og.import_arrays'),
    ('replay seg0', 'og.import_arrays', 'og.replay_layers0'), ('prime seg0', 'og.replay_layers0', 'og.replay_prime0'),
    ('replay seg1', 'og.replay_prime0', 'og.replay_layers1'), ('prime seg1', 'og.replay_layers1', 'og.replay_prime1'),
    ('import eval', ('og.replay_prime1', 'og.replay_prime0'), 'og.import_eval'),
    ('store+digest', 'og.import_eval', 'og.prepare_end'), ('->first step', 'og.prepare_end', 'og.first_step'),
    ('first step->output', 'og.first_step', 'first_output'), ('output->sse', 'first_output', 'first_content'),
    ('worker->sup->client', 'first_content', 'c.first'),
    ('= pre-admission', 'c.t0', 'og.arrival'), ('= admission', 'og.arrival', 'og.first_step'),
    ('= after first step', 'og.first_step', 'c.first'), ('= TTFT', 'c.t0', 'c.first'),
]


def load_lines(path, marker):
    out = []
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            at = line.find(marker)
            if at >= 0:
                try:
                    out.append(json.loads(line[at + len(marker):].strip()))
                except ValueError:
                    pass
    return out


def value(rec, key):
    if isinstance(key, tuple):
        for k in key:
            v = value(rec, k)
            if v is not None:
                return v
        return None
    return rec.get(key)


def join(client, sups, traces):
    sups = sorted(sups, key=lambda s: s['recv'])
    traces = sorted(traces, key=lambda t: t['http_start'])
    rows = []
    for c in client:
        if c.get('t0') is None or c.get('ttft_s') is None:
            continue
        s = next((x for x in sups if x['recv'] >= c['t0'] - 0.001), None)
        if s is None or s['recv'] - c['t0'] > 1:
            continue
        w = next((x for x in traces if x['http_start'] >= s.get('sent', s['recv']) - 0.001), None)
        if w is None or w['http_start'] - s.get('sent', s['recv']) > 1:
            continue
        rec = dict(w)
        rec.update({'s.' + k: v for k, v in s.items()})
        rec.update({'c.t0': c['t0'], 'c.first': c['t0'] + c['ttft_s']})
        rec['kind'] = f"{c['kind']} {c.get('n', c.get('case'))}"
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', required=True)
    ap.add_argument('--phases', required=True)
    ap.add_argument('--reference', default='')
    args = ap.parse_args()
    wdir = FE/args.window
    sups = load_lines(wdir/'supervisor.log', 'ds41-fe sup ')
    traces = load_lines(wdir/'og-child.log', 'ds41-fe trace ')
    digests = {}
    for line in (wdir/'og-child.log').read_text(errors='replace').splitlines():
        m = re.search(r'ds41-og digest (\S+): (\d+) tokens, ids (\w+), state (\w+)', line)
        if m:
            digests[m.group(1)] = (int(m.group(2)), m.group(3), m.group(4))
    phases = args.phases.split(',')
    report = {}
    per_case = {}  # identity case -> phase -> (prompt ids digest, imported state digest)
    for phase in phases:
        client = [json.loads(x) for x in (FE/f'{args.window}-{phase}.jsonl').read_text().splitlines()]
        rows = join(client, sups, traces)
        for row in rows:
            if row['kind'].startswith('identity'):
                per_case.setdefault(row['kind'], {})[phase] = digests.get(row.get('request_id'), (None, None, None))[1:]
        kinds = {}
        for row in rows:
            kind = row['kind']
            if kind.startswith('identity') and (row.get('prompt_tokens') or 0) < 2000:
                kind = 'identity short'
            kinds.setdefault(kind, []).append(row)
        report[phase] = {}
        for kind, group in sorted(kinds.items()):
            stages = {}
            for name, a, b in STAGES:
                vals = [value(r, b) - value(r, a) for r in group if value(r, a) is not None and value(r, b) is not None]
                if vals:
                    stages[name] = round(statistics.median(vals) * 1000, 1)
            stages['n'] = len(group)
            stages['encode_ms'] = round(statistics.median([r.get('encode_s', 0) for r in group]) * 1000, 1)
            report[phase][kind] = stages
    names = [s[0] for s in STAGES] + ['encode_ms', 'n']
    for kind in sorted({k for p in report.values() for k in p}):
        print(f'\n## {kind} (ms, median)')
        print('stage'.ljust(22) + ''.join(p.rjust(9) for p in phases))
        for name in names:
            vals = [report[p].get(kind, {}).get(name) for p in phases]
            if any(v is not None for v in vals):
                print(name.ljust(22) + ''.join(('-' if v is None else str(v)).rjust(9) for v in vals))

    # Output identity: every phase and the reference, per identity case.
    labels = ([args.reference] if args.reference else []) + [f'{args.window}-{p}' for p in phases]
    outputs = {}
    for label in labels:
        for line in (FE/f'{label}.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r.get('kind') == 'identity':
                outputs.setdefault(r['case'], {})[label] = (r.get('content'), r.get('reasoning'), r.get('tools'),
                                                            r.get('error'))
    print('\n## identity (full streamed output vs', labels[0], ')')
    same = 0
    for case, per in outputs.items():
        ref = per.get(labels[0])
        ok = all(per.get(l) == ref for l in labels) and ref is not None and ref[3] is None
        same += ok
        print(f'{case:18s}', 'IDENTICAL' if ok else 'DIFFERS', {l: (per.get(l) or ('',))[0][:40] for l in labels}
              if not ok else '')
    print(f'{same}/{len(outputs)} identical across {len(labels)} runs')

    same = sum(len(set(v.values())) == 1 and (None, None) not in v.values() for v in per_case.values())
    print(f'\n## identity cases: prompt-id and imported-state digests equal in every phase: {same}/{len(per_case)}')
    for case, per in per_case.items():
        print(f'{case:28s}', per)

    # Digests: same prompt ids -> same imported state in every phase.
    by_ids = {}
    for rid, (n, ids, state) in digests.items():
        by_ids.setdefault(ids, set()).add(state)
    diff = {k: v for k, v in by_ids.items() if len(v) > 1}
    print(f'\n## digests: {len(digests)} imports, {len(by_ids)} distinct prompts, '
          f'{len(by_ids) - len(diff)} with one state digest, {len(diff)} with more: {diff}')
    (FE/args.window/'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
