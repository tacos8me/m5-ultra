"""Build summary.json + SUMMARY.md from the Mac + RTX split leg JSONL files and the historical q3 baseline."""
import calendar, csv, json, statistics as st, time
from pathlib import Path

D = Path.home() / 'llm/ds41/mac-rtx-split-bench'
BASE = Path.home() / 'llm/ds41/coord/final'


def rows(name):
    return [json.loads(l) for l in open(D / name) if l.strip()]


def agg(xs):
    xs = [x for x in xs if x is not None]
    return {'n': len(xs), 'mean': round(st.mean(xs), 2), 'min': min(xs), 'max': max(xs),
            'sd': round(st.stdev(xs), 2) if len(xs) > 1 else 0.0} if xs else None


def label(n):
    return {8192: '8K', 16384: '16K', 32768: '32K', 65536: '64K', 131072: '128K', 262144: '256K', 524288: '512K',
            786432: '768K', 1040000: '1M'}[n]


S = {'deployment': {'model': 'ds41 (Mac + RTX split)', 'box_engine': None, 'numerics': None, 'client': 'Mac localhost -> llama-swap :8080, streaming, temperature 0'}}
contention = []

# ---------- baseline q3 (historical, read-only) ----------
bl = {}
for f in ('depth-short.jsonl', 'depth.jsonl'):
    for r in (json.loads(l) for l in open(BASE / f)):
        bl.setdefault('prefill', {})[r['prompt_tokens']] = {'ttft_s': r['ttft_s'], 'prefill_tok_s': r['prefill_tok_s'], 'decode_tok_s': r['decode_tok_s']}
bmap = {8192: 8217, 16384: 16410, 32768: 32793, 65536: 65561, 131072: 130026, 262144: 262169, 524288: 524314, 786432: 786458, 1040000: 1040026}
b8 = [json.loads(l)['decode_tok_s'] for l in open(BASE / '8k-rerun.jsonl')] + [bl['prefill'][8217]['decode_tok_s']] + \
     [r['decode_tok_s'] for r in (json.loads(l) for l in open(BASE / 'decode.jsonl')) if r.get('prompt') == 'sum8k'] + \
     [r['decode_tok_s'] for r in (json.loads(l) for l in open(BASE / 'decode.cf7ad4b3.jsonl')) if r.get('prompt') == 'sum8k']
essay = [r['decode_tok_s'] for f in ('decode.jsonl', 'decode.cf7ad4b3.jsonl') for r in (json.loads(l) for l in open(BASE / f)) if r.get('prompt') == 'essay']
bdec = {'8K code-summary': agg(b8), 'short essay prompt (nearest to 8K prose)': agg(essay),
        '128K': agg([bl['prefill'][130026]['decode_tok_s']]), '256K': agg([bl['prefill'][262169]['decode_tok_s']])}
for n, f in ((524288, 'decode-512k-samples.jsonl'), (1040000, 'decode-1m-samples.jsonl'), (786432, 'decode-768k-samples.jsonl')):
    bdec[label(n)] = agg([json.loads(l)['decode_tok_s'] for l in open(BASE / f)])
bc2 = [json.loads(l) for l in open(BASE / 'c2.jsonl')]
bpre = json.loads(open(BASE / 'prefix.jsonl').readline())
S['baseline_q3'] = {'source': str(BASE), 'note': 'historical Mac-only q3g128 build, single samples unless n shown; not rerun (q3 deleted)',
                    'prefill': {label(n): bl['prefill'][bmap[n]] | {'prompt_tokens': bmap[n]} for n in bmap},
                    'decode': bdec,
                    'c2_short_prompts': {'aggregate_tok_s': agg([r['aggregate_tok_s'] for r in bc2 if 'pair' in r]),
                                         'per_stream_tok_s': agg([x for r in bc2 if 'pair' in r for x in r['per_request_tok_s']])},
                    'resume_128k': bpre}

# ---------- leg 1 ----------
L1 = [r for r in rows('leg1_prefill.jsonl') if r.get('event') == 'prefill']
S['deployment']['box_engine'] = L1[0]['pre_idle']['box_version']; S['deployment']['numerics'] = L1[0]['pre_idle']['numerics']
# additional fresh samples from other legs (leg2 q0, leg4 turn1)
extra = [r for r in rows('leg2_decode.jsonl') if r.get('event') == 'decode' and r['q'] == 0 and r['workload'] == 'code'] + \
        [r for r in rows('leg4_resume.jsonl') if r.get('event') == 'turn1']
p1 = {}
for n in bmap:
    rs = [r for r in L1 if r['target'] == n]
    ex = [r for r in extra if r['target'] == n]
    allr = rs + ex
    p1[label(n)] = {'prompt_tokens': agg([r['prompt_tokens'] for r in rs]), 'ttft_s': agg([r['ttft_s'] for r in rs]),
                    'prefill_tok_s': agg([r['prefill_tok_s'] for r in rs]),
                    'box_prefill_s': agg([r['og_open']['box_prefill_s'] for r in rs]),
                    'box_only_tok_s': agg([round(r['prompt_tokens'] / r['og_open']['box_prefill_s'], 1) for r in rs]),
                    'box_resumed_tokens': [r['og_open']['box_resumed'] for r in rs],
                    'all_fresh_samples_ttft_s': agg([r['ttft_s'] for r in allr]),
                    'all_fresh_samples_prefill_tok_s': agg([round(r['prompt_tokens'] / r['ttft_s'], 1) for r in allr]),
                    'first_decode_tok_s': agg([r['decode_tok_s'] for r in allr])}
S['leg1_prefill'] = p1
contention += [('leg1', r['foreign_requests'], r['pre_idle']['box_sessions']) for r in L1]

# ---------- leg 2 ----------
L2 = [r for r in rows('leg2_decode.jsonl') if r.get('event') == 'decode']
p2 = {}
for (n, w) in [(8192, 'code'), (8192, 'prose'), (131072, 'code'), (262144, 'code'), (524288, 'code'), (1040000, 'code')]:
    rs = [r for r in L2 if r['target'] == n and r['workload'] == w]
    k = f'{label(n)} {"code-summary" if w == "code" else "free-prose"}'
    p2[k] = {'prompt_tokens': agg([r['prompt_tokens'] for r in rs]), 'decode_tok_s': agg([r['decode_tok_s'] for r in rs]),
             'decode_tok_s_followups_only': agg([r['decode_tok_s'] for r in rs if r['q'] > 0]),
             'completion_tokens': [r['completion_tokens'] for r in rs], 'accept_rate': agg([r.get('accept_rate') for r in rs]),
             'tok_per_cycle': agg([r['mtp']['tok_per_cycle'] for r in rs if r.get('mtp')]),
             'box_ms_per_step': agg([r.get('box_ms_per_step') for r in rs]),
             'roundtrip_ms_per_step': agg([r.get('roundtrip_ms_per_step') for r in rs]),
             'followup_ttft_s': agg([r['ttft_s'] for r in rs if r['q'] > 0])}
S['leg2_decode'] = p2
contention += [('leg2', r['foreign_requests'], r['pre_idle']['box_sessions']) for r in L2]

# ---------- leg 3 ----------
L3 = rows('leg3_concurrency.jsonl')
p3 = {}
for n, c in [(8192, 2), (131072, 2), (8192, 4)]:
    warm = [r for r in L3 if r.get('event') == 'warm_decode' and r['target'] == n and r['c'] == c]
    cold = [r for r in L3 if r.get('event') == 'cold_arrival' and r['target'] == n and r['c'] == c]
    p3[f'c{c} {label(n)}'] = {
        'rounds': len(warm), 'per_stream_tok_s': agg([x for r in warm for x in r['per_stream_tok_s']]),
        'aggregate_tok_s': agg([r['aggregate_tok_s'] for r in warm]),
        'ttft_s_by_arrival_order': [agg([r['ttft_s'][j] for r in warm]) for j in range(c)],
        'streams_overlapping': [not r['serialized'] for r in warm],
        'cold_arrival': [{'ttft_s': r['ttft_s'], 'per_stream_tok_s': r['per_stream_tok_s'], 'aggregate_tok_s': r['aggregate_tok_s']} for r in cold]}
    contention += [('leg3', r.get('foreign_requests'), r['pre_idle']['box_sessions']) for r in warm + cold]
S['leg3_concurrency'] = p3

# ---------- leg 4 ----------
L4 = rows('leg4_resume.jsonl')
p4 = {}
for n in (8192, 131072, 524288):
    t1 = [r for r in L4 if r.get('event') == 'turn1' and r['target'] == n][0]
    t2 = [r for r in L4 if r.get('event') == 'turn2' and r['target'] == n]
    p4[label(n)] = {'turn1_prompt': t1['prompt_tokens'], 'turn1_ttft_s': t1['ttft_s'],
                    'turn2_prompt': agg([r['prompt_tokens'] for r in t2]), 'turn2_ttft_s': agg([r['ttft_s'] for r in t2]),
                    'turn2_box_resumed': [r['og_open']['box_resumed'] for r in t2],
                    'turn2_new_tokens': [r['prompt_tokens'] - r['og_open']['box_resumed'] for r in t2]}
    rg = [r for r in L4 if r.get('event') == 'regenerate' and r['target'] == n]
    if rg:
        p4[label(n)]['regenerate_ttft_s'] = agg([r['ttft_s'] for r in rg])
        p4[label(n)]['regenerate_same_answer'] = [r['same_answer'] for r in rg]
    contention += [('leg4', r['foreign_requests'], r['pre_idle']['box_sessions']) for r in [t1] + t2 + rg]
S['leg4_resume'] = p4

# ---------- leg 5 ----------
L5 = [r for r in rows('leg5_vision.jsonl') if r.get('event') in ('vision', 'vision_diagnostic')]
S['leg5_vision'] = [{'task': r['task'], 'rep': r['rep'], 'images': r['images'], 'prompt_tokens': r['prompt_tokens'], 'ttft_s': r['ttft_s'],
                     'answer': r['text'], 'expected': r['expected'], 'correct': r['correct'], 'event': r['event']} for r in L5]
v = [r for r in L5 if r['event'] == 'vision']
S['leg5_vision_summary'] = {'1 image': {'ttft_s': agg([r['ttft_s'] for r in v if r['images'] == 1]), 'correct': f"{sum(r['correct'] for r in v if r['images'] == 1)}/{sum(1 for r in v if r['images'] == 1)}"},
                            '2 images': {'ttft_s': agg([r['ttft_s'] for r in v if r['images'] == 2]), 'correct': f"{sum(r['correct'] for r in v if r['images'] == 2)}/{sum(1 for r in v if r['images'] == 2)}"}}
contention += [('leg5', r['foreign_requests'], r['pre_idle']['box_sessions']) for r in L5]

# ---------- leg 6 ----------
nv = list(csv.reader(open(D / 'box/nvsmi.csv')))
def nvts(s):
    return calendar.timegm(time.strptime(s.split('.')[0], '%Y/%m/%d %H:%M:%S')) + float('0.' + s.split('.')[1])
gpu = {}
for i in ('0', '1'):
    rr = [r for r in nv if r[1].strip() == i]
    mem = [int(r[2].split()[0]) for r in rr]
    gpu[f'gpu{i}'] = {'vram_used_mib_start': mem[0], 'vram_used_mib_peak': max(mem), 'vram_total_mib': int(rr[0][3].split()[0]),
                      'util_peak_pct': max(int(r[4].split()[0]) for r in rr), 'power_peak_w': max(float(r[5].split()[0]) for r in rr)}
one_m = [r for r in L1 if r['target'] == 1040000][0]
w0, w1 = one_m['wall_start'], one_m['wall_start'] + one_m['ttft_s']
for i in ('0', '1'):
    rr = [r for r in nv if r[1].strip() == i and w0 <= nvts(r[0]) <= w1]
    gpu[f'gpu{i}']['during_1M_prefill'] = {'util_mean_pct': round(st.mean(int(r[4].split()[0]) for r in rr), 1),
                                           'power_mean_w': round(st.mean(float(r[5].split()[0]) for r in rr), 1)}
nic = [tuple(map(float, l.split())) for l in open(D / 'box/nic.log')]
def link(t0, t1):
    sel = [r for r in nic if t0 <= r[0] <= t1]; a, b = sel[0], sel[-1]; dt = b[0] - a[0]
    pk = max((y[2] - x[2]) / (y[0] - x[0]) for x, y in zip(sel, sel[1:]))
    return {'window_s': round(dt, 1), 'box_to_mac_MBps_mean': round((b[2] - a[2]) / dt / 1e6, 1), 'box_to_mac_MBps_peak_0.5s': round(pk / 1e6, 1),
            'box_to_mac_GB_total': round((b[2] - a[2]) / 1e9, 3), 'mac_to_box_MBps_mean': round((b[1] - a[1]) / dt / 1e6, 2)}
r512 = [r for r in L1 if r['target'] == 524288][0]
dec = [r for r in L2 if r['target'] == 1040000 and r['q'] > 0]
mac_after = (D / 'system_mac_after.txt').read_text()
S['leg6_system'] = {
    'mac_og_worker': {'text_after_bench': mac_after, 'text_before_bench': (D / 'system_mac_before.txt').read_text()},
    'box_gpus': gpu,
    'box_step_ms (from Mac og/stats deltas, per decode step, all leg-2 requests)': {
        'box_compute': agg([r.get('box_ms_per_step') for r in L2]), 'roundtrip_incl_link': agg([r.get('roundtrip_ms_per_step') for r in L2]),
        'by_depth': {k: {'box': p2[k]['box_ms_per_step']['mean'], 'roundtrip': p2[k]['roundtrip_ms_per_step']['mean']} for k in p2}},
    'link_10GbE': {'512K prefill': link(r512['wall_start'], r512['wall_start'] + r512['ttft_s']),
                   '1M prefill': link(w0, w1),
                   '1M follow-up window (5 cached questions incl. tail re-prefills)': link(dec[0]['wall_start'] + dec[0]['ttft_s'], dec[-1]['wall_start'] + dec[-1]['total_s'])}}
S['contention'] = {'requests_checked': len(contention), 'foreign_requests_total': sum(c[1] or 0 for c in contention),
                   'box_sessions_nonzero_before_request': sum(1 for c in contention if c[2]),
                   'method': 'before every measured request: wait until box /health sessions==0 and Mac /health inflight==0; after: og/stats opened delta must equal own requests'}
json.dump(S, open(D / 'summary.json', 'w'), indent=1)

# ---------- SUMMARY.md ----------
f = lambda a, k='mean', d=1: '-' if not a else (f"{a[k]:,.{d}f}")
rng = lambda a, d=1: '-' if not a else f"{a['min']:,.{d}f}-{a['max']:,.{d}f}"
M = []
M.append('# Mac + RTX split benchmark: DeepSeek-V4.1-Flash ORIGINAL FP4/FP8, RTX PRO 6000 pair (layers 0-19) + M5 Ultra (layers 20-39 + head + DSpark)\n')
M.append(f"Run 2026-09-26, box engine {S['deployment']['box_engine']}, numerics {S['deployment']['numerics']}, served by llama-swap as `ds41`. "
         'All numbers are through the production OpenAI chat-completions API (streaming, temperature 0), client on the Mac (localhost:8080), warm server '
         '(warm-up request discarded at the start of every leg). Fresh prompts start with a unique nonce; box log confirms `resumed 0` for every fresh prompt. '
         f"Contention: {S['contention']['foreign_requests_total']} foreign requests and {S['contention']['box_sessions_nonzero_before_request']} non-idle starts across {S['contention']['requests_checked']} measured requests.\n")
M.append('Baseline = historical Mac-only q3g128 build (`~/llm/ds41/coord/final/*.jsonl`), read not rerun; mostly single samples.\n')
M.append('## 1. Prefill (fresh prompts, TTFT includes the first token)\n')
M.append('| Context | prompt_tokens | TTFT s (mean, range) | Prefill tok/s | Box-only tok/s | n | q3 TTFT s | q3 tok/s | Speedup |')
M.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for n in bmap:
    a = p1[label(n)]; b = S['baseline_q3']['prefill'][label(n)]
    rg = f" ({rng(a['ttft_s'], 2)})" if a['ttft_s']['n'] > 1 else ''
    M.append(f"| {label(n)} | {a['prompt_tokens']['mean']:,.0f} | {f(a['ttft_s'], d=2)}{rg} | {f(a['prefill_tok_s'], d=0)} | {f(a['box_only_tok_s'], d=0)} | {a['ttft_s']['n']} | {b['ttft_s']:.2f} | {b['prefill_tok_s']:,.0f} | {a['prefill_tok_s']['mean'] / b['prefill_tok_s']:.1f}x |")
M.append('\nRepeat fresh samples from legs 2/4 agree within 1%: ' + ', '.join(
    f"{k} {v['all_fresh_samples_ttft_s']['min']:.2f}-{v['all_fresh_samples_ttft_s']['max']:.2f} s (n={v['all_fresh_samples_ttft_s']['n']})"
    for k, v in p1.items() if v['all_fresh_samples_ttft_s']['n'] > v['ttft_s']['n']) + '.\n')
M.append('## 2. Decode (c1, 256 tokens, mean of 6 samples: 1 fresh + 5 prefix-cached follow-up questions on the same document)\n')
M.append('| Point | prompt_tokens | Decode tok/s mean | min-max | DSpark accept | tok/cycle | Box ms/step | Round-trip ms/step | q3 tok/s (n) |')
M.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
bkey = {'8K code-summary': '8K code-summary', '8K free-prose': 'short essay prompt (nearest to 8K prose)', '128K code-summary': '128K',
        '256K code-summary': '256K', '512K code-summary': '512K', '1M code-summary': '1M'}
for k, a in p2.items():
    b = bdec[bkey[k]]
    bs = f"{b['mean']:.1f} ({b['n']})" + (' short essay' if 'prose' in k else '')
    M.append(f"| {k} | {a['prompt_tokens']['mean']:,.0f} | **{f(a['decode_tok_s'])}** | {rng(a['decode_tok_s'])} | {a['accept_rate']['mean'] * 100:.0f}% | {f(a['tok_per_cycle'], d=2)} | {f(a['box_ms_per_step'], d=2)} | {f(a['roundtrip_ms_per_step'], d=2)} | {bs} |")
M.append('\nDecode is set by DSpark acceptance, not depth: box compute stays 7.2-7.8 ms per step from 8K to 1M. Free prose decodes about 10 tok/s slower than code summaries at 8K because fewer drafts are accepted. q3 768K for reference: '
         f"{bdec['768K']['mean']:.1f} tok/s (n={bdec['768K']['n']}).\n")
M.append('## 3. Concurrency (warm decode: documents primed first, then concurrent cached follow-ups, 4 rounds x 256 tokens)\n')
M.append('| Config | Per-stream tok/s mean (range) | Aggregate tok/s mean (range) | TTFT s by arrival order | Cold arrival (fresh prompts at once) TTFT s |')
M.append('|---|---:|---:|---|---|')
for k, a in p3.items():
    tt = ' / '.join(f"{x['mean']:.2f}" for x in a['ttft_s_by_arrival_order'])
    ca = '; '.join(' / '.join(f'{t:.2f}' for t in c['ttft_s']) for c in a['cold_arrival'])
    M.append(f"| {k} | {f(a['per_stream_tok_s'])} ({rng(a['per_stream_tok_s'])}) | {f(a['aggregate_tok_s'])} ({rng(a['aggregate_tok_s'])}) | {tt} | {ca} |")
M.append(f"\nq3 baseline c2 (short prompts): aggregate {S['baseline_q3']['c2_short_prompts']['aggregate_tok_s']['mean']:.1f} tok/s, per-stream {S['baseline_q3']['c2_short_prompts']['per_stream_tok_s']['mean']:.1f}. "
         'c2 streams decode concurrently; prefills are serialized on the box, so the second stream waits for the first prefill '
         '(at 8K the cached follow-ups also re-prefill; see caveats). c4 is accepted but runs 2 at a time: streams 3-4 queue until the first pair finishes (~5 s at 8K), so the aggregate stays around 97 tok/s.\n')
M.append('## 4. Resume (prefix cache on box + Mac)\n')
M.append('| Context | Turn-1 TTFT s (fresh) | Turn-2 TTFT s mean (range, n=3) | New tokens in turn 2 | Regenerate TTFT s | q3 turn-2 s |')
M.append('|---|---:|---:|---:|---:|---:|')
for k, a in p4.items():
    rgs = f"{a['regenerate_ttft_s']['mean']:.2f} ({rng(a['regenerate_ttft_s'], 2)}), identical output {sum(a['regenerate_same_answer'])}/3" if 'regenerate_ttft_s' in a else '-'
    q3 = f"{bpre['turn2_ttft_s']:.2f}" if k == '128K' else '-'
    M.append(f"| {k} | {a['turn1_ttft_s']:.2f} | {f(a['turn2_ttft_s'], d=2)} ({rng(a['turn2_ttft_s'], 2)}) | {min(a['turn2_new_tokens'])}-{max(a['turn2_new_tokens'])} | {rgs} | {q3} |")
M.append('\n## 5. Vision (generated PNGs, 32 max tokens, nonce text part first)\n')
M.append('| Task | Images | TTFT s | Answer | Expected | OK |')
M.append('|---|---:|---:|---|---|---|')
for r in S['leg5_vision']:
    ans = r['answer'].replace('\n', ' ')[:70]
    tag = ' (diagnostic re-ask)' if r['event'] == 'vision_diagnostic' else f" rep{r['rep']}"
    M.append(f"| {r['task']}{tag} | {r['images']} | {r['ttft_s']:.2f} | {ans} | {', '.join(r['expected'])} | {'yes' if r['correct'] else 'NO'} |")
vs = S['leg5_vision_summary']
M.append(f"\n1 image: TTFT {vs['1 image']['ttft_s']['mean']:.2f} s, {vs['1 image']['correct']} correct. 2 images: TTFT {vs['2 images']['ttft_s']['mean']:.2f} s, {vs['2 images']['correct']} correct. "
         'The one miss echoed the answer template ("first, second"). Re-asked without the template, the same images gave "red ... green". q3 had no vision.\n')
M.append('## 6. System\n')
M.append('| Item | Value |'); M.append('|---|---|')
for line in mac_after.splitlines():
    if 'Footprint' in line:
        M.append(f'| Mac omlx-server footprint (after bench) | {line.split("Footprint:")[1].strip()} |')
rss = [l for l in mac_after.splitlines() if l.strip().isdigit()]
M.append(f'| Mac omlx-server RSS (after bench) | {int(rss[0]) / 1048576:.1f} GiB |')
mj = json.loads([l for l in mac_after.splitlines() if l.startswith('{')][0])
M.append(f"| Mac og worker peak RSS / footprint (watcher) | {mj['peak_rss_gib']:.1f} / {mj['peak_footprint_gib']:.1f} GiB of 256 GB |")
for g, a in gpu.items():
    M.append(f"| Box {g} VRAM used (start / peak / total) | {a['vram_used_mib_start']:,} / {a['vram_used_mib_peak']:,} / {a['vram_total_mib']:,} MiB |")
    M.append(f"| Box {g} during 1M prefill | util {a['during_1M_prefill']['util_mean_pct']}% mean, {a['during_1M_prefill']['power_mean_w']} W mean (peak {a['power_peak_w']} W) |")
bs = S['leg6_system']['box_step_ms (from Mac og/stats deltas, per decode step, all leg-2 requests)']
M.append(f"| Box compute per decode step (layers 0-19, 8K-1M) | {bs['box_compute']['mean']:.2f} ms mean ({rng(bs['box_compute'], 2)}) |")
M.append(f"| Mac-observed box round trip per step (incl. 10GbE) | {bs['roundtrip_incl_link']['mean']:.2f} ms mean ({rng(bs['roundtrip_incl_link'], 2)}) |")
for k, l in S['leg6_system']['link_10GbE'].items():
    M.append(f"| Link box->Mac during {k} | {l['box_to_mac_MBps_mean']} MB/s mean, {l['box_to_mac_MBps_peak_0.5s']} MB/s peak, {l['box_to_mac_GB_total']} GB total |")
M.append('\nThe link is nowhere near saturated: a prefill ships about 0.43 KB per token of layer-20 state. Step times are Mac-side og/stats deltas per request; the box engine log records only per-session prefill times (box/engine_sessions.log).\n')
M.append('## Caveats\n')
M.append('- Decode rates depend on content through DSpark acceptance (per-sample range about +-15%). Every point is a mean of 6 samples. Follow-ups reuse the same document with different questions.')
M.append('- The q3 baseline is mostly single samples taken on a different build. The ratios show the order of magnitude, not a controlled A/B.')
M.append('- TTFT is client-measured on the Mac through llama-swap. Box-only prefill time comes from the og worker log (`box open ... prefill Xs`).')
M.append('- Follow-up questions on an 8K document got no box-cache resume (box log `resumed 0`), so their TTFT is a full 8K re-prefill (~1.0 s). From 128K up, follow-ups resume in 8,192-token blocks: all but the last partial block is reused, and TTFT is 1.5 s at 128K and 5.8 s at 1M. Turn-2 continuations (leg 4) resume the whole turn-1 prompt at every size.')
M.append('- Turn-2 resume at 128K takes 1.25 s, slower than q3 (0.85 s). Box prefill is only 0.05 s; the Mac import of box state takes about 0.55 s, and the Mac layers 20-39 take the rest.')
M.append('- The llama-swap description still says "text only", but vision is live (box /health vision:true) and was tested here.')
(D / 'SUMMARY.md').write_text('\n'.join(M) + '\n')
print('\n'.join(M))
