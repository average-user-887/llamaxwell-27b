#!/usr/bin/env python3
# per-token paired stats from kldshim -tok.txt files: pair.py a b [...]; first file is the reference for pairing
import sys
X='/media/avg-usr/Models/kld-f2-20261108/'
def L(t):
    d={}
    for l in open(X+t+'-tok.txt'):
        f=l.split(); d[(int(f[0]),int(f[1]))]=(float(f[3]),float(f[4]),float(f[5]),int(f[6]),int(f[7]),int(f[9]))
    return d
R={t:L(t) for t in sys.argv[1:]}
ref=sys.argv[1]
print("run n mean max >0.5 >1 >3 top% nonfinite | worst tokens (chunk:pos=kld)")
for t,d in R.items():
    v=sorted(((x[0],k) for k,x in d.items()),reverse=True)
    n=len(v); m=sum(x for x,_ in v)/n
    print(t,n,'%.6f %.3f %d %d %d %.3f %d'%(m,v[0][0],sum(x>.5 for x,_ in v),sum(x>1 for x,_ in v),sum(x>3 for x,_ in v),100*sum(x[3]==x[4] for x in d.values())/n,sum(x[5] for x in d.values())),' '.join('%d:%d=%.2f'%(k[0],k[1],x) for x,k in v[:6]))
keys=sorted(set(k for d in R.values() for k,x in d.items() if x[0]>1))
print("tokens with kld>1 in any run:", len(keys))
print('chunk:pos '+' '.join(R))
for k in keys: print('%d:%d '%k+' '.join('%.3f'%R[t][k][0] if k in R[t] else '-' for t in R))
