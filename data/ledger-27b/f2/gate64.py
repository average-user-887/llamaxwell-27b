#!/usr/bin/env python3
# gate64.py A B: paired gate on kldshim -tok.txt. Limits set by owner 9 Nov (raised from |dmean|<=.0005, |dtop|<=.3): dmean<=+.0010, same-top no lower than -.3 pt; improvements never fail.
import sys,math
X='/media/avg-usr/Models/kld-f2-20261108/'
def L(t):
    d={}
    for l in open(X+t+'-tok.txt'):
        f=l.split(); d[(int(f[0]),int(f[1]))]=(float(f[3]),int(f[6]),int(f[7]))
    return d
a,b=L(sys.argv[1]),L(sys.argv[2]); K=sorted(set(a)&set(b)); n=len(K)
D=[b[k][0]-a[k][0] for k in K]; m=sum(D)/n; s=math.sqrt(sum((x-m)**2 for x in D)/(n-1)/n)
top=lambda d:100*sum(d[k][1]==d[k][2] for k in K)/n
ma,mb=sum(a[k][0] for k in K)/n,sum(b[k][0] for k in K)/n
dt=top(b)-top(a); ok=(mb-ma)<=.0010 and dt>=-.3
print(f"n={n} (a {len(a)}, b {len(b)}) mean {ma:.6f} -> {mb:.6f} d={mb-ma:+.6f} se={s:.6f} t={m/s:.2f} top {top(a):.3f} -> {top(b):.3f} d={dt:+.3f}")
print(f"max {max(a[k][0] for k in K):.3f} -> {max(b[k][0] for k in K):.3f}; >1 {sum(a[k][0]>1 for k in K)} -> {sum(b[k][0]>1 for k in K)}; >3 {sum(a[k][0]>3 for k in K)} -> {sum(b[k][0]>3 for k in K)}")
print('10:416 %s'%' -> '.join('%.3f'%d[(10,416)][0] if (10,416) in d else '-' for d in (a,b)))
print('GATE',"PASS" if ok else "FAIL")
