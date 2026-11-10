import csv,sys,collections,re
rows=list(csv.reader(open(sys.argv[1])))
h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
K=[]
for r in rows[h+2:]:
    if len(r)!=len(hd): continue
    n=r[ci["Name"]]
    if n.startswith("["): continue
    K.append((float(r[ci["Start"]]),float(r[ci["Duration"]])*1000,(r[ci["Grid X"]],r[ci["Grid Y"]],r[ci["Grid Z"]]),r[ci["Block X"]],re.sub(r'\(.*','',n.replace("void ",""))[:90]))
# token boundary: head kernel = max grid X mmvq
heads=[i for i,k in enumerate(K) if 'head_rowpack' in k[4]]
print("heads",len(heads),[K[i][2] for i in heads[-2:]])
a,b=heads[-2]+1,heads[-1]
tok=K[a:b+1]; busy=sum(k[1] for k in tok); wall=(tok[-1][0]+tok[-1][1]/1e6-tok[0][0])*1e6
print("kernels",len(tok),"busy us",round(busy),"wall us",round(wall))
ag=collections.OrderedDict()
for k in tok:
    e=ag.setdefault((k[4],k[2],k[3]),[0,0.0]); e[0]+=1; e[1]+=k[1]
for key,(n,t) in sorted(ag.items(),key=lambda x:-x[1][1])[:26]:
    print(f"{t/busy*100:5.1f}% {t:7.0f}us n={n:3d} {t/n:6.1f}us/call grid={'x'.join(key[1])} blk={key[2]} {key[0]}")
