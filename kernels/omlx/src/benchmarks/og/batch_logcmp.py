"""Per-request DSpark stats from an og worker log, grouped by admission window: tokens/cycle vs Mac ms/cycle.

  python3 batch_logcmp.py ~/llm/ds41/og/logs/og-child.log 21:37:20-21:38:45 21:41:48-21:42:15 [--min-tokens 128]
Windows are UTC HH:MM:SS of the admission's first step (same day as the log's last admission). A c2 A/B with fresh
nonce prompts mixes two effects: tokens per verify cycle (acceptance: prompt-dependent, not code-dependent at
temperature 0) and time per cycle (the code). This splits them.
"""
import datetime, re, statistics, sys

path, windows = sys.argv[1], [a for a in sys.argv[2:] if '-' in a and ':' in a]
min_tokens = int(sys.argv[sys.argv.index('--min-tokens') + 1]) if '--min-tokens' in sys.argv else 128
adm, fin, order = {}, [], []
for line in open(path, errors='replace'):
    m = re.search(r'admission (\S+): (\d+) tokens.*first_step_wall ([\d.]+)', line)
    if m:
        order.append(('adm', float(m.group(3))))
        continue
    m = re.search(r'MTP\[(\d+)\] finish=\w+ tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+).*timing\[backbone=([\d.]+)ms mtp=([\d.]+)ms', line)
    if m:
        order.append(('fin', tuple(float(x) for x in m.groups())))
# a request's FIN follows its admission; pair them in order (one worker, uids restart per process)
walls, recs = [], []
for kind, v in order:
    if kind == 'adm':
        walls.append(v)
    elif walls:
        recs.append((walls.pop(0), v))
day = datetime.datetime.fromtimestamp(recs[-1][0], datetime.timezone.utc).date()
for w in windows:
    a, b = (datetime.datetime.combine(day, datetime.time.fromisoformat(x), datetime.timezone.utc).timestamp() for x in w.split('-'))
    sel = [v for t, v in recs if a <= t <= b and v[1] >= min_tokens and v[2] > 0]
    if not sel:
        print(w, 'no requests'); continue
    tpc = statistics.mean(v[1] / v[2] for v in sel)
    bb = statistics.mean(v[4] / v[2] for v in sel)
    mtp = statistics.mean(v[5] / v[2] for v in sel)
    print(f'{w}: n={len(sel)} tok/cycle={tpc:.2f} backbone_ms/cycle={bb:.1f} draft_ms/cycle={mtp:.1f}')
