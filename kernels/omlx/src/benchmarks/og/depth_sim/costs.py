# Cost models, ms, index 0..3 = L=2..5. Tiers keyed by context floor.
OLD = {0:(29.6,32.6,34.4,36.7), 131072:(29.9,32.9,34.7,37.1), 524288:(30.2,33.4,35.4,37.8)}
# Mac forward L=1..5
MAC = {'prod':(10.88,13.14,14.73,16.11,17.89), 'ffn':(10.27,12.55,14.11,15.46,17.38)}
# fused pair 3+3 and 5+5
PAIR = {'prod':(20.75,28.24), 'ffn':(19.88,27.52)}
# Box STEP RTT from the Mac, 25 ms gaps, decode-findings L=1/2/3/5 minus 0.7 (box-perf); L=4 interpolated
OLD_RTT = {1:7.693,2:8.318,3:8.791,5:10.259}
OLD_RTT[4] = (OLD_RTT[3]+OLD_RTT[5])/2
RTT = {L:OLD_RTT[L]-0.7 for L in OLD_RTT}
DRAFT = 3.5      # width-4 DSpark, fixed
HOST = 0.4       # accept/host per stream cycle
DRAFT_PAIR = 4.76  # batched drafting per fused pair (c4)
# long-context Mac attention deltas (Sep 25 table shape), per L=2..5
MAC_TIER = {0:(0,0,0,0), 131072:(0.3,0.3,0.3,0.4), 524288:(0.6,0.8,1.0,1.1)}
# long-context c1 RTT delta (calib-runs L=5 medians: 10.60/11.18/11.52 at 8K/128K/512K), assumed flat in L
RTT_TIER = {0:0.0, 131072:0.5, 524288:0.9}

def c1(build, extra=0.0):
    out={}
    for t in MAC_TIER:
        out[t]=tuple(round(RTT[L]+RTT_TIER[t]+MAC[build][L-1]+MAC_TIER[t][L-2]+DRAFT+HOST+extra,2) for L in range(2,6))
    return out

def c2(build, extra=0.0):
    # Mac-bound: per-stream slot = Mac(L) + drafter + host (+ unmodelled scheduler overhead)
    return {t:tuple(round(MAC[build][L-1]+MAC_TIER[t][L-2]+DRAFT+HOST+extra,2) for L in range(2,6)) for t in MAC_TIER}

def c4(build, extra=0.0):
    a,b = PAIR[build]
    slope=(b-a)/4; icpt=a-6*slope
    # per-stream share of a fused pair: half the fixed pass + its own rows + half the batched draft + host
    return {t:tuple(round(icpt/2+slope*L+MAC_TIER[t][L-2]+DRAFT_PAIR/2+HOST+extra,2) for L in range(2,6)) for t in MAC_TIER}
