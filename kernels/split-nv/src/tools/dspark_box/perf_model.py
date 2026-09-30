"""DSpark-on-box performance model (DSPARK-BOX.md s4): c1/c2/c4 and the fairness interaction from measured inputs.
M = measured (THREADS-2026-09-30 s2a/s2c, C4-ANALYSIS, box-perf ab2 steplog), E = estimates. X = box drafter job
(drafter_bench.py job_ms). No I/O: python3 tools/dspark_box/perf_model.py
"""
M = dict(
  c1_mac=15.43, c1_box=6.99, c1_dspark=3.80, c1_link=1.37, c1_glue=0.10, c1_tok=3.10,        # 8K code, leg2 (THREADS 2a)
  c1m_mac=17.25, c1m_box=7.13, c1m_dspark=3.85, c1m_link=1.31, c1m_glue=0.16, c1m_tok=3.14,  # 1M
  c2_mac=20.5, c2_tok=3.23, c2_win=157.8,                                                    # c4ab window (THREADS 2c)
  c4_pass_c4ab=32.4, c4_pass_part=30.3, c4_draft=6.0, c4_tok=3.19, c4_win=197.0,
  box_host=0.60,                                                                             # box host per step (p50 0.50-0.56 + bcast)
  share=0.5,
)
E = dict(accept_off=0.30, tap_mac=0.10, upload=0.20)  # accept moved behind the box; Mac tap readback+pack; taps wire + r1 pipe

def c1(X, mac, box, dspark, link, glue, f=1.0):
    old = mac + box + dspark + link + glue
    new = (mac - E['accept_off'] + E['tap_mac']) + link + E['upload'] + X + box + glue
    return old, new, old / new * f - 1

print("c1 (per cycle; X = box drafter job incl. H2D, D2H sync, width choice, next-launch bubble)")
for X in (1.4, 1.8, 2.2, 3.0):
    for tag, k in (("8K", "c1"), ("1M", "c1m")):
        old, new, g = c1(X, M[k+'_mac'], M[k+'_box'], M[k+'_dspark'], M[k+'_link'], M[k+'_glue'])
        print(f"  X={X:.1f} {tag}: {old:.2f} -> {new:.2f} ms  {100*g:+.1f}%   (acceptance -1%: {100*((1+g)*0.99-1):+.1f}%)")
be = M['c1_mac']+M['c1_box']+M['c1_dspark']+M['c1_link']+M['c1_glue'] - ((M['c1_mac']-E['accept_off']+E['tap_mac'])+M['c1_link']+E['upload']+M['c1_box']+M['c1_glue'])
print(f"  c1 break-even X = {be:.2f} ms")
# production c1 coverage: >=1024 ctx 93% of output tokens; copy cycles 2.6%; T>0/processors share unknown (assume 0-20%)
for X in (1.4, 1.8, 2.2):
    _, _, g = c1(X, M['c1_mac'], M['c1_box'], M['c1_dspark'], M['c1_link'], M['c1_glue'])
    for tshare in (0.0, 0.2):
        cov = 0.93 * (1 - 0.026) * (1 - tshare)
        # time-weighted: covered cycles faster by g; speedup of the mix
        mix = 1 / ((1 - cov) + cov / (1 + g)) - 1
        print(f"  production c1 X={X} T>0/proc share {tshare:.0%}: coverage {cov:.2f} -> {100*mix:+.1f}%")

print("c2 (Mac-bound; per request-cycle Mac time)")
for X in (1.4, 1.8, 2.2):
    new_mac = M['c2_mac'] - 3.90 + E['tap_mac']
    rtt = M['c1_link'] + E['upload'] + X + M['c1_box']
    boxbusy = (X + M['c1_box'] + M['box_host']) / new_mac
    print(f"  X={X}: Mac {M['c2_mac']:.1f} -> {new_mac:.2f} ms, gain {100*(M['c2_mac']/new_mac-1):+.1f}% (window {M['c2_win']*M['c2_mac']/new_mac:.0f} tok/s); "
          f"box RTT {rtt:.1f} < other's Mac slot {new_mac:.1f}; box busy {100*boxbusy:.0f}%")

print("c4 (fused pairs; per pair pass)")
for X in (1.4, 1.8, 2.2):
    for base in ('c4ab', 'part'):
        p = M['c4_pass_' + base]
        new = p - M['c4_draft'] + 2 * E['tap_mac']
        box = 2 * (X + M['c1_box'] + M['box_host'])
        second = M['c1_link'] + E['upload'] + box            # 2nd request of the pair: queued behind the 1st on the box
        slack = new - second
        print(f"  X={X} {base}: pass {p:.1f} -> {new:.1f} ms ceiling {100*(p/new-1):+.1f}%; box {box:.1f} ms/pass = {100*box/new:.0f}% busy; "
              f"2nd reply after {second:.1f} ms vs other pair's pass {new:.1f}: slack {slack:+.1f}")
    # box-side pair batching of the drafter (one pass for both sessions ~ X + 0.4)
    box_b = 2 * (M['c1_box'] + M['box_host']) + X + 0.4
    print(f"    with pair-batched box drafting: box {box_b:.1f} ms/pass = {100*box_b/(M['c4_pass_c4ab']-6+0.2):.0f}% busy")

print("Fairness during a concurrent prefill (PREEMPT_SHARE 0.5; box share needed by decode)")
X = 1.8
step_today = M['c1_box'] + M['box_host']
step_new = X + M['c1_box'] + M['box_host']
for c, mac_today, mac_new in ((2, 20.5, 16.7), (4, 32.4/2, (32.4-6+0.2)/2)):
    need_today = step_today / mac_today   # per request-cycle; requests share the Mac serially
    need_new = step_new / mac_new
    for name, need in (("today", need_today), ("box drafts", need_new)):
        dec_share = min(need, M['share'])
        pf_share = 1 - dec_share
        print(f"  c{c} {name:10s}: decode needs {100*need:.0f}% of box -> gets {100*dec_share:.0f}%, prefill {100*pf_share:.0f}% "
              f"(prefill time x{1/pf_share:.2f}); decode rate {100*min(1,dec_share/need):.0f}% of its unthrottled rate")
    # TTFT effect of box drafts vs today
    t_today = 1 / (1 - min(need_today, M['share'])); t_new = 1 / (1 - min(need_new, M['share']))
    print(f"     -> new-arrival prefill time {100*(t_new/t_today-1):+.0f}% vs today; decode during prefill {100*(min(1,M['share']/need_new)*mac_today/mac_new-1):+.0f}% vs today")
