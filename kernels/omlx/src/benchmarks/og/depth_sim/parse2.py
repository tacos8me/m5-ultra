import re,collections,pickle
mtp=re.compile(r'MTP\[(\S+)\] finish=(\S+) tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) accept=(\d+)/(\d+).*?(?:depth\[([^\]]*)\])?(?: d0=(\d+))?(?: copy\[cycles=(\d+) accept=(\d+)/(\d+)\])? emits\[init=(\d+),draft=(\d+),bonus=(\d+),verify=(\d+)\]')
chat=re.compile(r'Chat completion: model=(\S+), (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\), prompt: (\d+)')
seg=None; reqs=[]; pending=[]
for line in open('og-child.log',errors='replace'):
    if line.startswith('===') and 'start og:' in line:
        seg=line[4:23]; pending=[]; continue
    m=mtp.search(line)
    if m:
        d=[tuple(map(int,x.split('=')[1].split('/'))) for x in (m.group(8) or '').split(',') if x]
        r=dict(seg=seg,tokens=int(m.group(3)),cycles=int(m.group(4)),acc=int(m.group(6)),prop=int(m.group(7)),d=d,
               copy=(int(m.group(10) or 0),int(m.group(11) or 0),int(m.group(12) or 0)),bonus=int(m.group(15)),prompt=None)
        reqs.append(r); pending.append(r); continue
    m=chat.search(line)
    if m:
        t=int(m.group(2)); p=int(m.group(5))
        for r in pending:
            if abs(r['tokens']-t)<=2 and r['prompt'] is None:
                r['prompt']=p; pending.remove(r); break
pickle.dump(reqs,open('reqs.pkl','wb'))
print(len(reqs),'linked',sum(r['prompt'] is not None for r in reqs))
