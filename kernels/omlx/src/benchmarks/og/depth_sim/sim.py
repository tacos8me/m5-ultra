import json, random, statistics as St
import costs as C
R=[json.loads(l) for l in open('og-speed/calib.jsonl')]
def tier(ctx): return 524288 if ctx>=524288 else 131072 if ctx>=131072 else 0
for r in R: r['tier']=tier(r['context'])
# runs + rolling-window stratum (mean accepted of previous 16 cycles in the same run; first 4 cycles use what exists)
runs=[];prev=None
for r in R:
    if prev is None or r['context']<prev: runs.append([])
    runs[-1].append(r); prev=r['context']
for run in runs:
    for i,r in enumerate(run):
        w=run[max(0,i-16):i] or run[i+1:i+5]
        r['roll']=sum(x['accepted'] for x in w)/len(w)
med=St.median(r['roll'] for r in R)
for r in R: r['stratum']='low' if r['roll']<med else 'high'

def evict(table):
    def pol(r):
        costs=table[r['tier']]; cum=1.0; exp=1.0; best=1; bu=-1
        for i,p in enumerate(r['probs'][:4]):
            cum*=max(0.,min(1.,p)); exp+=cum; u=exp/costs[i]
            if u>bu: best,bu=i+1,u
        return best
    return pol
def dinkel(table, lam):
    def pol(r):
        costs=table[r['tier']]; cum=1.0; exp=1.0; best=1; bu=-1e9
        for i,p in enumerate(r['probs'][:4]):
            cum*=max(0.,min(1.,p)); exp+=cum; u=exp-lam*costs[i]
            if u>bu: best,bu=i+1,u
        return best
    return pol
fixed=lambda d:(lambda r:d)

def run(rows, pol, true):
    tok=cost=0.0
    for r in rows:
        d=pol(r); tok+=min(r['accepted'],d)+1; cost+=true[r['tier']][d-1]
    n=len(rows); return tok/n, cost/n, 1000*tok/cost
def dinkel_fit(rows,table):
    lam=0.1
    for _ in range(30):
        t,c,_=run(rows,dinkel(table,lam),table); lam=t/c
    return dinkel(table,lam)
def oracle(rows,true):
    lam=0.1
    for _ in range(40):
        tok=cost=0
        for r in rows:
            cs=true[r['tier']]; d=max(range(1,5),key=lambda d:(min(r['accepted'],d)+1)-lam*cs[d-1])
            tok+=min(r['accepted'],d)+1; cost+=cs[d-1]
        lam=tok/cost
    return len(rows) and (tok/len(rows), cost/len(rows), 1000*lam)
