import sim, proxy, costs as C
R=proxy.R
X={'c1':0.0,'c2':1.04,'c4':2.26}
def true_model(conc,build,extra=True):
    f={'c1':C.c1,'c2':C.c2,'c4':C.c4}[conc]; return f(build, X[conc] if extra else 0.0)
def dinkel_w(rows,w,table):
    lam=0.1
    for _ in range(30):
        t,c,_,_=proxy.wrun(rows,w,sim.dinkel(table,lam),table); lam=t/c
    return sim.dinkel(table,lam)
def oracle_w(rows,w,true):
    lam=0.1
    for _ in range(40):
        tok=cost=0
        for r,wi in zip(rows,w):
            cs=true[r['tier']]; d=max(range(1,5),key=lambda d:(min(r['accepted'],d)+1)-lam*cs[d-1])
            tok+=wi*(min(r['accepted'],d)+1); cost+=wi*cs[d-1]
        lam=tok/cost
    return 1000*lam
STRATA={'calib all (code)':0.0,'code-like proxy':0.11,'current-mix proxy':-0.37,'prose-like proxy':-0.77}
if __name__=='__main__':
    import sys
    extra = '--noextra' not in sys.argv
    for build in ('prod','ffn'):
        for conc in ('c1','c2','c4'):
            true=true_model(conc,build,extra)
            pols=[('old Sep25',sim.evict(C.OLD)),('new c1 table',sim.evict(C.c1(build))),('matched table',sim.evict(true)),
                  ('fixed d4 (L5)',sim.fixed(4)),('fixed d3',sim.fixed(3))]
            print('\n### %s %s  true C(L2..5)@8K %s%s'%(build,conc,true[0],'' if extra else ' (no extra overhead)'))
            for sname,beta in STRATA.items():
                w=proxy.weights(beta); base=None; line=[]
                for pname,pol in pols+[('dinkelbach',dinkel_w(R,w,true))]:
                    t,c,tps,ds=proxy.wrun(R,w,pol,true)
                    if base is None: base=tps
                    line.append('%s: %.3f tok/cyc %.2f ms %.2f (%+.2f%%) L-mix %s'%(pname,t,c,tps,100*(tps/base-1),'/'.join('%.2f'%x for x in ds)))
                orc=oracle_w(R,w,true)
                print('  [%s]  hindsight oracle %.2f (%+.1f%%)'%(sname,orc,100*(orc/base-1)))
                for l in line: print('     '+l)
