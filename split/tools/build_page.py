"""Build index.html for the Mac + RTX split page from bench/summary.json (static SVG, no JS)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "bench/summary.json").read_text())
Q3 = S["baseline_q3"]
PRE, DEC = S["leg1_prefill"], S["leg2_decode"]
CTX = ["8K", "16K", "32K", "64K", "128K", "256K", "512K", "768K", "1M"]
DEC_KEYS = [("8K", "8K code-summary"), ("128K", "128K code-summary"), ("256K", "256K code-summary"),
            ("512K", "512K code-summary"), ("1M", "1M code-summary")]


def q3_decode(k):
    d = Q3["decode"]
    key = {"8K": "8K code-summary"}.get(k, k)
    return d[key]["mean"] if key in d else Q3["prefill"][k]["decode_tok_s"]


def fmt_int(x):
    return f"{x:,.0f}"


def fmt_s(x):
    return f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


# ---------------------------------------------------------------- charts
W, H, L, R, T, B = 820, 300, 56, 104, 20, 40
XMAX = 1_100_000
TICKS_X = [(0, "0"), (131072, "128K"), (262144, "256K"), (524288, "512K"), (786432, "768K"), (1048576, "1M")]


def sx(v):
    return L + (W - L - R) * v / XMAX


def chart(series, ymax, yticks, yfmt, label):
    sy = lambda v: T + (H - T - B) * (1 - v / ymax)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{label}">']
    for v in yticks:
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        out.append(f'<text x="{L - 10}" y="{sy(v) + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for v, t in TICKS_X:
        out.append(f'<text x="{sx(v):.1f}" y="{H - B + 22}" text-anchor="middle">{t}</text>')
    out.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    for s in series:
        pts = s["pts"]
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y, *_r) in enumerate(pts))
        out.append(f'<path class="{s["cls"]}" d="{path}"/>')
        for x, y, *rng in [(p[0], p[1], *p[2:]) for p in pts]:
            if rng:
                lo, hi = rng
                out.append(f'<line class="{s["cls"]} whisk" x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}"/>')
            out.append(f'<circle class="{s["cls"]}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{3.5 if s["cls"] == "ic" else 2.5}"/>')
        lx, ly = pts[-1][0], pts[-1][1]
        out.append(f'<text class="lbl {s["cls"]}" x="{sx(lx) + 8:.1f}" y="{sy(ly) + 4:.1f}">{s["name"]}</text>')
    out.append("</svg>")
    return "\n".join(out)


pre_ic = [(PRE[k]["prompt_tokens"]["mean"], PRE[k]["prefill_tok_s"]["mean"]) for k in CTX]
pre_q3 = [(Q3["prefill"][k]["prompt_tokens"], Q3["prefill"][k]["prefill_tok_s"]) for k in CTX]
prefill_svg = chart(
    [{"pts": pre_q3, "cls": "q3", "name": "3-bit, Mac only"}, {"pts": pre_ic, "cls": "ic", "name": "Mac + RTX"}],
    20000, [0, 5000, 10000, 15000, 20000], lambda v: f"{v // 1000:.0f}K" if v else "0",
    "Prefill throughput by prompt length")

dec_ic = [(DEC[k]["prompt_tokens"]["mean"], DEC[k]["decode_tok_s"]["mean"], DEC[k]["decode_tok_s"]["min"],
           DEC[k]["decode_tok_s"]["max"]) for _, k in DEC_KEYS]
dec_q3 = [(Q3["prefill"][c]["prompt_tokens"], q3_decode(c)) for c, _ in DEC_KEYS]
decode_svg = chart(
    [{"pts": dec_q3, "cls": "q3", "name": "3-bit, Mac only"}, {"pts": dec_ic, "cls": "ic", "name": "Mac + RTX"}],
    120, [0, 30, 60, 90, 120], lambda v: f"{v:.0f}", "Decode speed by context depth")

# ---------------------------------------------------------------- table
rows = []
for k in CTX:
    p, q = PRE[k], Q3["prefill"][k]
    dk = dict(DEC_KEYS).get(k)
    dec = f'{DEC[dk]["decode_tok_s"]["mean"]:.1f}<span>{q3_decode(k):.1f}</span>' if dk else '<i class="na">—</i>'
    rows.append(
        f'<tr><td>{k}<span>{fmt_int(p["prompt_tokens"]["mean"])} tokens</span></td>'
        f'<td>{fmt_s(p["ttft_s"]["mean"])} s<span>{fmt_s(q["ttft_s"])} s</span></td>'
        f'<td>{fmt_int(p["prefill_tok_s"]["mean"])}<span>{fmt_int(q["prefill_tok_s"])}</span></td>'
        f'<td class="x">{p["prefill_tok_s"]["mean"] / q["prefill_tok_s"]:.1f}×</td>'
        f'<td>{dec}</td></tr>')
table = "\n".join(rows)

c3, r4, v5 = S["leg3_concurrency"], S["leg4_resume"], S["leg5_vision_summary"]
q3c2 = Q3["c2_short_prompts"]["aggregate_tok_s"]["mean"]
t128, q128 = PRE["128K"]["ttft_s"]["mean"], Q3["prefill"]["128K"]["ttft_s"]
t1m, q1m = PRE["1M"]["ttft_s"]["mean"], Q3["prefill"]["1M"]["ttft_s"]
d512 = DEC["512K code-summary"]["decode_tok_s"]["mean"]

html = f"""<title>DeepSeek-V4.1-Flash, Mac + RTX</title>
<meta name="description" content="DeepSeek-V4.1-Flash on its original weights across a Mac Studio M5 Ultra and two RTX PRO 6000 GPUs.">
<style>
  :root {{
    color-scheme: dark;
    --bg: oklch(0.141 0.005 285.8); --fg: oklch(0.92 0.004 286); --dim: oklch(0.62 0.014 286);
    --rule: oklch(0.26 0.006 286); --accent: oklch(0.82 0.1 215); --ghost: oklch(0.5 0.012 286);
    --panel: oklch(0.175 0.006 286);
  }}
  @media (prefers-color-scheme: light) {{
    :root:not([data-theme="dark"]) {{
      color-scheme: light;
      --bg: oklch(0.99 0.002 286); --fg: oklch(0.2 0.006 286); --dim: oklch(0.5 0.014 286);
      --rule: oklch(0.9 0.004 286); --accent: oklch(0.52 0.13 235); --ghost: oklch(0.7 0.01 286);
      --panel: oklch(0.965 0.003 286);
    }}
  }}
  :root[data-theme="light"] {{
    color-scheme: light;
    --bg: oklch(0.99 0.002 286); --fg: oklch(0.2 0.006 286); --dim: oklch(0.5 0.014 286);
    --rule: oklch(0.9 0.004 286); --accent: oklch(0.52 0.13 235); --ghost: oklch(0.7 0.01 286);
    --panel: oklch(0.965 0.003 286);
  }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, "Segoe UI", Roboto, sans-serif;
    font-variant-numeric: tabular-nums; padding-inline: 24px; padding-block: 56px 72px; }}
  main {{ max-width: 820px; margin: 0 auto; }}
  h1 {{ font-size: 24px; font-weight: 600; letter-spacing: -0.015em; margin: 0; text-wrap: balance; }}
  h2 {{ font-size: 18px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 4px; text-wrap: balance; }}
  .meta {{ color: var(--dim); margin: 6px 0 36px; }}
  .meta b {{ color: var(--fg); font-weight: 500; }}
  .meta .links {{ font-size: 13px; }}
  .meta a {{ color: var(--accent); text-decoration: none; }}
  .meta a:hover, .meta a:focus-visible {{ text-decoration: underline; }}
  .lead {{ color: var(--dim); font-size: 13px; margin: 0 0 18px; }}
  section {{ margin-bottom: 52px; }}
  .figs {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--rule);
    border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; margin-bottom: 52px; }}
  .fig {{ background: var(--bg); padding: 16px 16px 14px; }}
  .fig .k {{ color: var(--dim); font-size: 12px; letter-spacing: 0.02em; }}
  .fig .v {{ font-size: 26px; font-weight: 600; letter-spacing: -0.02em; color: var(--accent); margin: 2px 0 0; }}
  .fig .v small {{ font-size: 14px; font-weight: 500; color: var(--fg); margin-left: 2px; }}
  .fig .was {{ color: var(--ghost); font-size: 12.5px; }}
  .chart {{ overflow-x: auto; margin: 0 -8px 8px; }}
  svg {{ display: block; width: 100%; min-width: 520px; height: auto; }}
  svg text {{ font: 11px -apple-system, BlinkMacSystemFont, system-ui, sans-serif; fill: var(--dim); }}
  svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
  svg .axis {{ stroke: var(--ghost); stroke-width: 1; }}
  svg path {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
  svg path.ic {{ stroke: var(--accent); }}
  svg path.q3 {{ stroke: var(--ghost); stroke-dasharray: 5 5; stroke-width: 1.5; }}
  svg circle.ic {{ fill: var(--accent); }} svg circle.q3 {{ fill: var(--ghost); }}
  svg line.whisk.ic {{ stroke: var(--accent); stroke-width: 1.5; opacity: 0.55; }}
  svg text.lbl.ic {{ fill: var(--accent); font-weight: 600; }} svg text.lbl.q3 {{ fill: var(--ghost); }}
  .pipe {{ display: grid; grid-template-columns: max-content 156px max-content; justify-content: center; align-items: center; gap: 12px; }}
  .node {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; max-width: 280px; }}
  .node .nt {{ font-weight: 600; }}
  .node .hw {{ color: var(--dim); font-size: 13px; margin-bottom: 10px; }}
  .node ul {{ list-style: none; margin: 0; padding: 0; font-size: 13.5px; }}
  .node ul li {{ padding: 1px 0; }}
  .node ul.do {{ border-top: 1px solid var(--rule); margin-top: 10px; padding-top: 10px; color: var(--dim); }}
  .wire {{ display: grid; justify-items: center; gap: 4px; text-align: center; }}
  .wire .lk {{ color: var(--accent); font-size: 12px; line-height: 1.3; white-space: nowrap; }}
  .wire .cable {{ color: var(--dim); font-size: 12px; margin: 6px 0; }}
  .wire svg {{ width: 100%; min-width: 0; height: 10px; overflow: visible; }}
  .wire svg path {{ stroke: var(--accent); stroke-width: 1.5; fill: none; vector-effect: non-scaling-stroke; }}
  .wire svg path.hd {{ fill: var(--accent); stroke: none; }}
  .tablewrap {{ overflow-x: auto; }}
  table {{ width: 100%; min-width: 600px; border-collapse: collapse; }}
  th, td {{ padding: 9px 0 9px 16px; text-align: right; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  th:first-child, td:first-child {{ padding-left: 0; text-align: left; white-space: nowrap; }}
  th {{ font-weight: 500; color: var(--dim); font-size: 13px; }}
  tr:last-child td {{ border-bottom: 0; }}
  td span {{ display: block; color: var(--ghost); font-size: 12.5px; }}
  td.x {{ color: var(--accent); font-weight: 600; }}
  .na {{ color: var(--ghost); font-style: normal; }}
  .facts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px 40px; }}
  .facts h3 {{ font-size: 13px; font-weight: 600; color: var(--dim); letter-spacing: 0.04em; text-transform: uppercase; margin: 0 0 6px; }}
  .facts dl {{ margin: 0; display: grid; grid-template-columns: 1fr auto; gap: 4px 16px; }}
  .facts dt {{ color: var(--dim); }} .facts dd {{ margin: 0; text-align: right; }}
  .facts dd span {{ color: var(--ghost); font-size: 12.5px; margin-left: 6px; }}
  .prose p {{ margin: 0 0 12px; }}
  .prose code {{ font: 13px ui-monospace, "SF Mono", Menlo, monospace; color: var(--fg); }}
  .foot {{ color: var(--dim); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 18px; }}
  .foot p {{ margin: 0 0 8px; }}
  .foot a {{ color: inherit; text-decoration-color: var(--ghost); text-underline-offset: 2px; }}
  @media (max-width: 640px) {{
    .figs {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .facts {{ grid-template-columns: 1fr; }}
    .pipe {{ grid-template-columns: 1fr; justify-content: stretch; }}
    .node {{ max-width: none; }}
    .wire {{ grid-template-columns: 1fr auto 1fr; align-items: center; padding: 2px 8px; }}
    .wire svg {{ display: none; }}
    .wire .lk {{ white-space: normal; }}
    .wire .cable::before {{ content: "⇅ "; color: var(--accent); }}
  }}
  @media print {{
    :root {{ color-scheme: light; --bg: #fff; --fg: #1d1d22; --dim: #62626c; --rule: #e3e3e8;
      --accent: oklch(0.5 0.13 235); --ghost: #9a9aa3; --panel: #f6f6f8; }}
    body {{ padding-block: 0; font-size: 12.5px; }}
    section, .figs, .facts > div, .node {{ break-inside: avoid; }}
    @page {{ size: A4; margin: 16mm 14mm; }}
  }}
</style>
<main>
  <h1>DeepSeek-V4.1-Flash on a Mac and two RTX PRO 6000s</h1>
  <p class="meta"><b>DeepSeek-V4.1-Flash</b> on its original FP4/FP8 weights, served as one model across a
  <b>Mac Studio M5 Ultra</b> (256&nbsp;GB) and <b>two RTX PRO 6000 Blackwell</b> GPUs over a 10GbE cable.
  Measured through the production API, September 26, 2026.<br>
  <span class="links"><a href="https://github.com/tacos8me/m5-ultra/tree/main/kernels">Code on GitHub</a> · <a href="../">All M5 Ultra results</a></span></p>

  <div class="figs">
    <div class="fig"><div class="k">128K prompt, first token</div><div class="v">{t128:.1f}<small>s</small></div><div class="was">was {q128:.1f} s</div></div>
    <div class="fig"><div class="k">1M prompt, first token</div><div class="v">{t1m:.0f}<small>s</small></div><div class="was">was {q1m:.0f} s</div></div>
    <div class="fig"><div class="k">Decode at 512K</div><div class="v">{d512:.0f}<small>tok/s</small></div><div class="was">was {q3_decode("512K"):.0f} tok/s</div></div>
    <div class="fig"><div class="k">Two requests at once</div><div class="v">{c3["c2 8K"]["aggregate_tok_s"]["mean"]:.0f}<small>tok/s</small></div><div class="was">was {q3c2:.0f} tok/s</div></div>
  </div>

  <section>
    <h2>One model, two machines</h2>
    <p class="lead">Only four of the model's 40 layers produce KV, and the upper half reuses layer 20's. The prompt state is
    about 0.9&nbsp;KB per token, so the model splits cleanly at layer 20.</p>
    <div class="pipe" role="img" aria-label="Pipeline: RTX box runs layers 0 to 19, Mac Studio runs layers 20 to 39, joined by 10GbE">
      <div class="node">
        <div class="nt">RTX box</div>
        <div class="hw">2× RTX PRO 6000 Blackwell</div>
        <ul><li>Embeddings, Engram, layers 0–19</li><li>Layer-20 KV rows, vision tower</li></ul>
        <ul class="do"><li>Prefill the whole prompt</li><li>1–5 verify rows a step, ~7&nbsp;ms</li></ul>
      </div>
      <div class="wire">
        <span class="lk">prompt state, step rows</span>
        <svg viewBox="0 0 100 10" aria-hidden="true"><path d="M2 5 H92"/><path class="hd" d="M90 1 L98 5 L90 9 z"/></svg>
        <span class="cable">10GbE</span>
        <svg viewBox="0 0 100 10" aria-hidden="true"><path d="M8 5 H98"/><path class="hd" d="M10 1 L2 5 L10 9 z"/></svg>
        <span class="lk">draft tokens</span>
      </div>
      <div class="node">
        <div class="nt">Mac Studio</div>
        <div class="hw">M5 Ultra, 256&nbsp;GB</div>
        <ul><li>Layers 20–39, output head</li><li>DSpark drafter, up to 4 tokens</li></ul>
        <ul class="do"><li>Accept, stream, draft the next rows</li><li>Prefix cache for resumed turns</li></ul>
      </div>
    </div>
  </section>

  <section>
    <h2>Prefill</h2>
    <p class="lead">Prompt tokens per second of time to first token, fresh uncached prompts. Dashed: the previous Mac-only 3-bit build.</p>
    <div class="chart">{prefill_svg}</div>
  </section>

  <section>
    <h2>Decode at depth</h2>
    <p class="lead">Output tokens per second, mean of six 256-token samples with the full range. Speed follows speculative
    acceptance, not context length: the box step is 7.2–7.8&nbsp;ms from 8K to 1M.</p>
    <div class="chart">{decode_svg}</div>
  </section>

  <section>
    <h2>By prompt length</h2>
    <p class="lead">Mac + RTX first, the 3-bit build beneath.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>Prompt</th><th>First token</th><th>Prefill tok/s</th><th>Speedup</th><th>Decode tok/s</th></tr></thead>
      <tbody>
{table}
      </tbody>
    </table></div>
  </section>

  <section class="facts">
    <div><h3>Concurrency</h3><dl>
      <dt>2 requests, 8K</dt><dd>{c3["c2 8K"]["aggregate_tok_s"]["mean"]:.1f} tok/s<span>{c3["c2 8K"]["per_stream_tok_s"]["mean"]:.0f} each</span></dd>
      <dt>2 requests, 128K</dt><dd>{c3["c2 128K"]["aggregate_tok_s"]["mean"]:.1f} tok/s<span>{c3["c2 128K"]["per_stream_tok_s"]["mean"]:.0f} each</span></dd>
      <dt>4 requests, 8K</dt><dd>{c3["c4 8K"]["aggregate_tok_s"]["mean"]:.1f} tok/s<span>2 at a time</span></dd>
    </dl></div>
    <div><h3>Resuming a conversation</h3><dl>
      <dt>Turn 2 at 8K</dt><dd>{r4["8K"]["turn2_ttft_s"]["mean"]:.2f} s</dd>
      <dt>Turn 2 at 128K</dt><dd>{r4["128K"]["turn2_ttft_s"]["mean"]:.2f} s<span>3-bit {Q3["resume_128k"]["turn2_ttft_s"]:.2f} s</span></dd>
      <dt>Turn 2 at 512K</dt><dd>{r4["512K"]["turn2_ttft_s"]["mean"]:.2f} s</dd>
    </dl></div>
    <div><h3>Images</h3><dl>
      <dt>1 image, first token</dt><dd>{v5["1 image"]["ttft_s"]["mean"]:.2f} s<span>{v5["1 image"]["correct"]} correct</span></dd>
      <dt>2 images, first token</dt><dd>{v5["2 images"]["ttft_s"]["mean"]:.2f} s<span>{v5["2 images"]["correct"]} correct</span></dd>
    </dl></div>
    <div><h3>Quality, 27 cases</h3><dl>
      <dt>Code continuation NLL</dt><dd>0.058<span>3-bit 0.062</span></dd>
      <dt>Needles at 128K and 1M</dt><dd>recalled</dd>
      <dt>Draft acceptance</dt><dd>66%<span>3-bit 65%</span></dd>
    </dl></div>
  </section>

  <section class="prose">
    <h2>How it runs</h2>
    <p>The box prefills layers 0–20 over the whole prompt and streams the state to the Mac in 8K chunks while it works.
    The Mac replays the last rows through layers 20–39, then each decode step crosses the link twice: draft tokens go to
    the box, which returns 41&nbsp;KB of hidden state per row. Two requests pipeline, so each machine works on one while the
    other finishes the next.</p>
    <p>A step on the box is bit-identical to prefilling the same rows, including rejected drafts, so a box engine restart
    mid-answer rebuilds the session and the text continues unchanged. If the box stays down, clients get a retryable
    error; no other model ever stands in.</p>
  </section>

  <footer class="foot">
    <p>{S["contention"]["requests_checked"]} measured requests through llama-swap on the Mac, temperature 0, streaming, warm server, no other
    load. Prefill at 8K–128K is the mean of three fresh prompts, longer points single runs that repeat within 1%. Decode
    samples swing about ±15% with speculative acceptance.</p>
    <p>The 3-bit baseline is the previous production build (Mac only, LSQ 3-bit experts), measured earlier with a similar
    harness; ratios are indicative, not a controlled A/B. Box engine c75a0a2, numerics og-s4.4.</p>
    <p>Hardware: Mac Studio M5 Ultra, 80-core GPU, 256&nbsp;GB · 2× NVIDIA RTX PRO 6000 Blackwell, 96&nbsp;GB each ·
    direct 10GbE. Model: <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">deepseek-ai/DeepSeek-V4.1-Flash</a>, original weights.</p>
  </footer>
</main>
"""

fav = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' "
       "fill='%2318181b'/%3E%3Cpath d='M9 16h14' stroke='%235fc4dc' stroke-width='3' stroke-linecap='round'/%3E"
       "%3Ccircle cx='8' cy='16' r='4' fill='%235fc4dc'/%3E%3Ccircle cx='24' cy='16' r='4' fill='%235fc4dc'/%3E%3C/svg%3E")
desc = "DeepSeek-V4.1-Flash on its original weights, split across a Mac Studio M5 Ultra and two RTX PRO 6000 GPUs over 10GbE."
site = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta property="og:title" content="DeepSeek-V4.1-Flash on a Mac and two RTX PRO 6000s">\n<meta property="og:description" content="{desc}">\n'
        f'<meta name="twitter:card" content="summary">\n<link rel="icon" href="{fav}">\n'
        + html.replace("<main>", "</head>\n<body>\n<main>", 1) + "</body>\n</html>\n")
(ROOT / "index.html").write_text(site)
print("wrote", ROOT / "index.html", len(site))
