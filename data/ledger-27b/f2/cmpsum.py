#!/usr/bin/env python3
# summarize <tag>-cmp.txt: per-call DEC=2 vs stock difference, map call -> chunk, eval, layer; args: tag [chunk eval]
import sys,re
X='/media/avg-usr/Models/kld-f2-20261108/'
tag=sys.argv[1]; R=[]
for l in open(X+tag+'-cmp.txt'):
    f=l.split(); c=int(f[0]); d={k:v for k,v in (x.split('=') for x in f[1:])}
    e=(c-16)//16; R.append((c,e//171,e%171,(c-16)%16,int(d['nkv']),float(d['wrel']),float(d['rms']),float(d['wabs']),int(d['r']),int(d['h']),int(d['e']),int(d['nf'])))
print(tag,'calls',len(R),'nonfinite',sum(r[11] for r in R))
import statistics as S
for name,idx in (('wrel',5),('rms',6),('wabs',7)):
    v=sorted(r[idx] for r in R); print(name,'median %.3g p99 %.3g p99.9 %.3g max %.3g'%(v[len(v)//2],v[int(.99*len(v))],v[int(.999*len(v))],v[-1]))
for nkv in sorted(set(r[4] for r in R)):
    a=[r for r in R if r[4]==nkv]; print('nkv',nkv,'calls',len(a),'wrel max %.3g median %.3g rms median %.3g'%(max(r[5] for r in a),S.median(r[5] for r in a),S.median(r[6] for r in a)))
print('top by wrel (call chunk eval layer nkv wrel rms wabs r h e):')
for r in sorted(R,key=lambda r:-r[5])[:8]: print(' ',r[:11])
if len(sys.argv)>3:
    ch,ev=int(sys.argv[2]),int(sys.argv[3])
    print('target eval chunk',ch,'eval',ev,'(call chunk eval layer nkv wrel rms wabs r h e)')
    for r in R:
        if r[1]==ch and r[2]==ev: print(' ',r[:11])
