import pickle, sim, costs as C
reqs=pickle.load(open('reqs.pkl','rb'))
def pooled(rs):
    D=[[0,0] for _ in range(4)]; cyc=tok=0
    for r in rs:
        for i,(a,p) in enumerate(r['d'][:4]): D[i][0]+=a; D[i][1]+=p
        cyc+=r['cycles']
    return [a/p if p else float('nan') for a,p in D], cyc
def fmt(x): return '/'.join('%.2f'%v for v in x)
# selection-matched dN of the calib data under a policy
def sim_d(rows,pol):
    D=[[0,0] for _ in range(4)]; tok=0
    for r in rows:
        k=pol(r); m=min(r['accepted'],k); tok+=m+1
        for j in range(k):
            D[j][1]+=1
            if j<m: D[j][0]+=1
            else: break
    return [a/p for a,p in D], tok/len(rows)
if __name__=='__main__':
    last=reqs[-1]['seg']
    for name,sel in [('all segments',lambda r:True),('current worker '+last,lambda r:r['seg']==last)]:
        rs=[r for r in reqs if sel(r) and r['cycles']>=8]
        big=[r for r in rs if (r['prompt'] or 0)>=1024]
        nocopy=[r for r in big if r['copy'][0]==0]
        print('==',name,'requests',len(rs),'prompt>=1K',len(big),'of which no copy',len(nocopy))
        for lab,g in [('all',rs),('prompt>=1K',big),('>=1K no-copy',nocopy)]:
            d,c=pooled(g); print('  %-14s cycles %6d  d1..d4 %s'%(lab,c,fmt(d)))
        # per-request terciles by DSpark conditional d1*d2 rate among no-copy >=1K
        key=lambda r: r['acc']/max(1,r['prop'])
        g=sorted(nocopy,key=key); n=len(g)
        for lab,part in [('low tercile',g[:n//3]),('mid',g[n//3:2*n//3]),('high tercile',g[2*n//3:])]:
            d,c=pooled(part); print('  %-14s reqs %4d cycles %6d  d %s  tok/cyc %.2f'%(lab,len(part),c,fmt(d),sum(r['tokens'] for r in part)/max(1,c)))
    for lab,rows in [('calib all',sim.R),('calib low-roll',[r for r in sim.R if r['stratum']=='low'])]:
        d,t=sim_d(rows,sim.evict(C.OLD)); print('%s under served policy (old table): d %s tok/cyc %.2f'%(lab,fmt(d),t))
        d,t=sim_d(rows,sim.fixed(4)); print('%s at forced depth 4: d %s tok/cyc %.2f'%(lab,fmt(d),t))
