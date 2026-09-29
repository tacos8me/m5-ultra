import random, sim, proxy, costs as C, main
R=proxy.R
blocks=[]
for run in sim.runs:
    blocks.append([run[i:i+16] for i in range(0,len(run),16)])
def rate(rows,w,pol,true):
    return proxy.wrun(rows,w,pol,true)[2]
def dk(rows,w,table):
    return main.dinkel_w(rows,w,table)
random.seed(1)
cmp=[]
for build in ('ffn',):
    t1=main.true_model('c1',build); t2=main.true_model('c2',build); t4=main.true_model('c4',build)
    cmp=[('c1 new vs old',t1,sim.evict(C.OLD),sim.evict(t1)),
         ('c2 c1-table vs old',t2,sim.evict(C.OLD),sim.evict(t1)),
         ('c2 matched vs c1-table',t2,sim.evict(t1),sim.evict(t2)),
         ('c4 c1-table vs old',t4,sim.evict(C.OLD),sim.evict(t1)),
         ('c4 fused vs c1-table',t4,sim.evict(t1),sim.evict(t4))]
    for beta,sname in [(0.0,'calib'),(-0.37,'current-mix'),(-0.77,'prose-proxy')]:
        for name,true,a,b in cmp+[('c1 dinkel vs new',t1,sim.evict(t1),'dk'),('c4 dinkel vs fused',t4,sim.evict(t4),'dk')]:
            ds=[]
            for _ in range(400):
                rows=[r for bl in blocks for r in random.choice([random.choice(bl) for _ in bl]) ] if False else None
                rows=[]
                for bl in blocks:
                    for _ in bl: rows.extend(random.choice(bl))
                w=[proxy.math.exp(beta*r['conf']) for r in rows]
                pb = dk(rows,w,true) if b=='dk' else b
                ds.append(100*(rate(rows,w,pb,true)/rate(rows,w,a,true)-1))
            ds.sort(); m=sum(ds)/len(ds)
            print('%-12s %-26s mean %+.2f%%  90%% CI [%+.2f, %+.2f]'%(sname,name,m,ds[20],ds[379]))
