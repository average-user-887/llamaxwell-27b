#!/usr/bin/env python3
# compare dumps for call range across two trajectories: snapcmp.py tagA tagB a b
import sys,struct
import numpy as np
X='/media/avg-usr/Models/kld-f2-20261108/'
A,B,a,b=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
def rd(f):
    h=struct.unpack('9q',f.read(72)); return h,np.frombuffer(f.read(h[8]),np.uint8)
def load(t,c):
    f=open(f'{X}{t}-dump-{c}.bin','rb'); s=struct.unpack('f',f.read(4))[0]
    Q=rd(f);K=rd(f);V=rd(f);M=rd(f); r=np.frombuffer(f.read(),np.float32); n=len(r)//2
    return dict(s=s,Q=Q,K=K,V=V,M=M,a=r[:n],b=r[n:])
def deq(h,d):
    # rows of 144 bytes (8 blocks of 18): dequantize
    nb1=h[4+1]; b=d
    return None
def kv(hd,xa,xb):
    ne=hd[:4]; nb=hd[4:8]; rows=int(ne[1])*int(ne[2])*int(ne[3])
    ra=xa[:rows*144].reshape(rows,-1) if (nb[1]==144) else None
    return ra
print(f'call layer nkv Qrel Qrms(%%) maskdiff outA/B | K: rows_diff scale_diff byte_diff | V: rows_diff scale_diff byte_diff')
for c in range(a,b):
    x=load(A,c);y=load(B,c)
    q1=np.frombuffer(x['Q'][1],np.float32);q2=np.frombuffer(y['Q'][1],np.float32)
    n=len(q1); qr=np.abs(q1-q2).max()/max(np.abs(q2).max(),1e-30); qrms=np.sqrt(((q1-q2)**2).sum()/max((q2**2).sum(),1e-30))
    md=int((x['M'][1]!=y['M'][1]).sum())
    def bytes_(key):
        h,u=x[key];_,v=y[key]; row=h[5]; nr=len(u)//row
        ub=u.reshape(nr,row)[:,:144];vb=v.reshape(nr,row)[:,:144]
        dif=(ub!=vb); rows=int(dif.any(1).sum()); sc=int((dif[:,0::18].sum()+dif[:,1::18].sum())) ; tot=int(dif.sum())
        ub2=ub.reshape(nr,8,18);vb2=vb.reshape(nr,8,18)
        dN=dif.reshape(nr,8,18); s=int(dN[:,:,:2].any(2).sum()); nib=int(dN[:,:,2:].sum())
        return rows,s,nib,nr
    kk=bytes_('K');vv=bytes_('V')
    print(c,(c-16)%16,x['K'][0][1],'%.3g %.4f'%(qr,qrms),md,'%.3g'%(np.abs(x['a']-y['a']).max()),'K',kk,'V',vv)
