import re,collections
seg=None; segs=collections.OrderedDict()
mtp=re.compile(r'MTP\[(\S+)\] finish=(\S+) tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+).*?(?:depth\[([^\]]*)\])?(?: d0=(\d+))?(?: copy\[cycles=(\d+) accept=(\d+)/(\d+)\])? emits')
chat=re.compile(r'Chat completion: model=(\S+), (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), prompt: (\d+)')
for line in open('og-child.log',errors='replace'):
    if 'start og:' in line and line.startswith('==='):
        seg=line[4:23]; segs[seg]=dict(mtp=[],chat=[],path=line.split('/src/wt/')[1].split('/')[0] if '/src/wt/' in line else '?'); continue
    if seg is None: continue
    m=mtp.search(line)
    if m:
        d=[tuple(map(int,x.split('=')[1].split('/'))) for x in (m.group(8) or '').split(',') if x]
        segs[seg]['mtp'].append(dict(tokens=int(m.group(3)),cycles=int(m.group(4)),acc=int(m.group(6)),prop=int(m.group(7)),d=d,copy=(int(m.group(10) or 0),int(m.group(11) or 0),int(m.group(12) or 0))))
        continue
    m=chat.search(line)
    if m: segs[seg]['chat'].append(dict(model=m.group(1),tokens=int(m.group(2)),s=float(m.group(3)),tps=float(m.group(4)),prompt=int(m.group(5))))
import pickle; pickle.dump(segs,open('segs.pkl','wb'))
for k,v in segs.items():
    ms=v['mtp']
    if not ms and not v['chat']: continue
    D=[[0,0] for _ in range(4)]
    for r in ms:
        for i,(a,p) in enumerate(r['d'][:4]): D[i][0]+=a; D[i][1]+=p
    tok=sum(r['tokens'] for r in ms); cyc=sum(r['cycles'] for r in ms)
    cp=[c['prompt'] for c in v['chat']]
    big=sum(1 for p in cp if p>=1024)
    print(k,v['path'],'req',len(ms),'tok/cyc %.2f'%(tok/cyc if cyc else 0),'d:',' '.join('%d/%d=%.2f'%(a,p,a/p if p else 0) for a,p in D),'chats',len(cp),'prompt>=1k',big, 'max prompt',max(cp) if cp else None)
