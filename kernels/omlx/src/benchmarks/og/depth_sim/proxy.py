import math, sim, costs as C
R=sim.R
for r in R:
    cum=1; e=0
    for p in r['probs']: cum*=p; e+=cum
    r['conf']=e   # predicted accepted drafts (0..4)
def wrun(rows,w,pol,true):
    tok=cost=W=0.0; ds=[0]*4
    for r,wi in zip(rows,w):
        d=pol(r); tok+=wi*(min(r['accepted'],d)+1); cost+=wi*true[r['tier']][d-1]; W+=wi; ds[d-1]+=wi
    return tok/W, cost/W, 1000*tok/cost, [x/W for x in ds]
def wdn(rows,w,pol):
    D=[[0,0] for _ in range(4)]
    for r,wi in zip(rows,w):
        k=pol(r); m=min(r['accepted'],k)
        for j in range(k):
            D[j][1]+=wi
            if j<m: D[j][0]+=wi
            else: break
    return [a/p for a,p in D]
def weights(beta): return [math.exp(beta*r['conf']) for r in R]
def ess(w): return sum(w)**2/sum(x*x for x in w)
TARGETS={'prose-like (log low tercile)':(.71,.65,.62,.66),'log mid tercile':(.79,.74,.71,.73),
         'code-like (log high tercile)':(.85,.81,.79,.80),'current worker pool':(.77,.72,.71,.74)}
def fit(target):
    best=None
    for i in range(-300,301):
        b=i/100; d=wdn(R,weights(b),sim.evict(C.OLD)); err=sum((x-y)**2 for x,y in zip(d,target))
        if best is None or err<best[0]: best=(err,b,d)
    return best
if __name__=='__main__':
    for k,t in TARGETS.items():
        err,b,d=fit(t); w=weights(b)
        print('%-30s beta %+.2f  fitted d %s  target %s  ESS %.0f  E[acc@4] %.2f'%(k,b,'/'.join('%.2f'%x for x in d),'/'.join('%.2f'%x for x in t),ess(w),sum(wi*r['accepted'] for r,wi in zip(R,w))/sum(w)))
