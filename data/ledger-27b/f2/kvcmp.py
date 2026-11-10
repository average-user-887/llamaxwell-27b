#!/usr/bin/env python3
# dequantize q4_0 K/V in two runs' dumps and compare: kvcmp.py A B a b
import sys,struct
import numpy as np
X='/media/avg-usr/Models/kld-f2-20261108/'
A,B,a,b=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
def rd(f):
    h=struct.unpack('9q',f.read(72)); return h,np.frombuffer(f.read(h[8]),np.uint8)
def f16(u):
    e=(u>>10)&31; m=u&1023; s=np.where(u>>15,-1.0,1.0)
    return s*np.where(e==0,m*2.0**-24,(1+m/1024.0)*2.0**(e.astype(float)-15))
def deq(raw):
    r=raw.reshape(-1,18)  # order: head then cell handled below
    d=f16(r[:,0].astype(np.uint16)|(r[:,1].astype(np.uint16)<<8))
    n=r[:,2:]; lo=(n&15).astype(np.int16)-8; hi=(n>>4).astype(np.int16)-8
    out=np.empty((len(r),32),np.float32); out[:,0::2]=lo*d[:,None]; out[:,1::2]=hi*d[:,None]
    return out.reshape(512,4,8,32).reshape(512,4,256) # cell, head, dim
def load(t,c):
    f=open(f'{X}{t}-dump-{c}.bin','rb'); struct.unpack('f',f.read(4)); Q=rd(f);K=rd(f);V=rd(f);M=rd(f); r=np.frombuffer(f.read(),np.float32); n=len(r)//2
    return np.frombuffer(Q[1],np.float32),deq(K[1]),deq(V[1]),r[:n],r[n:],K[1],V[1]
def rel(x,y): return np.sqrt(((x-y)**2).sum()/max((y**2).sum(),1e-30))
print('L  Qrel  | Krel(used rows) Vrel | Knib%% Vnib%% | kernel diff same input: rms in A | run out rel(A dec vs B dec)  | worst V row rel @row')
for c in range(a,b):
    qa,ka,va,oa,ob,kra,vra=load(A,c); qb,kb,vb,oc,od,krb,vrb=load(B,c)
    used=np.any(kb!=0,axis=(1,2)); nu=int(used.sum())
    vr=np.sqrt(((va-vb)**2).sum((1,2))/np.maximum((vb**2).sum((1,2)),1e-30)); vr[~used]=0
    knib=(kra!=krb).mean()*100; vnib=(vra!=vrb).mean()*100
    print((c-16)%16,'%.5f'%rel(qa,qb),'| %.5f %.5f'%(rel(ka[used],kb[used]),rel(va[used],vb[used])),'| %.1f %.1f'%(knib,vnib),'| %.2e'%rel(oa,ob),'| %.4f'%rel(oa,oc),'| %.4f @%d'%(vr.max(),int(vr.argmax())),'used',nu)
