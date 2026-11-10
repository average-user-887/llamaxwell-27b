#!/usr/bin/env python3
"""9 Nov power-cap sweep on live prod C4. Same text at each cap (temperature 0). Restores 250 W on any exit."""
import sys,json,time,random,signal,subprocess,threading,re
sys.path.insert(0,"/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
P=q.PROD_PORT; U=q.GPU; RES="/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/power/sweep-20261109.jsonl"
src=open("/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/f2/stage1.py").read()
PR=eval(re.search(r'^PR=q\.PROBES\+(\[.*\])$',src,re.M).group(1)); PR=q.PROBES+PR
CAPS=[int(x) for x in sys.argv[1:]] or [250,200,175,225,150,200,250]
def out(**k):
    k["t"]=time.strftime("%H:%M:%S"); s=json.dumps(k); print(s,flush=True); open(RES,"a").write(s+"\n")
def pl(w): return subprocess.run(["sudo","-n","/usr/bin/nvidia-smi","-i",U,"-pl",str(w)],capture_output=True,text=True).returncode
def smi(f): return subprocess.run(["nvidia-smi","-i",U,"--query-gpu="+f,"--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
def bye(*a): pl(250); sys.exit(9)
signal.signal(signal.SIGTERM,bye); signal.signal(signal.SIGINT,bye)
class Samp:
    def __enter__(s):
        s.r=[]; s.stop=False; s.t0=time.time(); s.th=threading.Thread(target=s.run,daemon=True); s.th.start(); return s
    def run(s):
        while not s.stop:
            try:
                v=[float(x) for x in smi("temperature.gpu,power.draw,clocks.gr").split(",")]; v.append(time.time()); s.r.append(v)
                if v[0]>=83: pl(250); out(sec="abort",T=v[0]); import os; os._exit(8)
            except ValueError: pass
            time.sleep(1)
    def __exit__(s,*a): s.stop=True; s.dt=time.time()-s.t0
    def sum(s):
        r=s.r or [[0,0,0,0]]; W=sum(x[1] for x in r)/len(r)
        return dict(W=round(W,1),Wmax=max(x[1] for x in r),MHz=round(sum(x[2] for x in r)/len(r)),maxT=max(x[0] for x in r),J=round(W*s.dt))
post=lambda b,t=7200:q.post("/completion",b,port=P,timeout=t)["timings"]
tk=lambda x:len(q.post("/tokenize",{"content":x},port=P)["tokens"])
fill=lambda r:" ".join("word%d alpha beta gamma delta, the quick brown fox %d."%(i,i*7) for i in range(r))+"\n\n"+q.TG
per=tk(fill(1000))/1000; LONG=fill(int(51900/per))
random.seed(11); T4=[random.randint(2000,150000) for _ in range(4096)]
try:
    if any(s["is_processing"] for s in q.get("/slots",port=P)): out(sec="abort",why="prod busy"); sys.exit(7)
    pl(250)
    with Samp() as S: t=post({"prompt":LONG,"n_predict":128,"ignore_eos":True,"cache_prompt":True,"temperature":0,"id_slot":0})
    out(sec="prefill52k",prompt_n=t["prompt_n"],pp=round(t["prompt_per_second"],2),**S.sum())
    for cap in CAPS:
        busy=any(s["is_processing"] for s in q.get("/slots",port=P))
        rc=pl(cap); time.sleep(5); lim=smi("power.limit")
        with Samp() as S: t=post({"prompt":T4,"n_predict":1,"cache_prompt":False,"temperature":0,"id_slot":1})
        out(sec="pp4k",cap=cap,rc=rc,lim=lim,busy=busy,pp=round(t["prompt_per_second"],2),**S.sum())
        n=d=a=dn=0
        with Samp() as S:
            for p in PR:
                t=post({"prompt":p,"n_predict":128,"ignore_eos":True,"cache_prompt":False,"temperature":0,"id_slot":1})
                n+=t["predicted_n"]; d+=t["predicted_ms"]/1000; a+=t.get("draft_n_accepted") or 0; dn+=t.get("draft_n") or 0
        out(sec="short",cap=cap,tg=round(n/d,3),n=n,accept=f"{a}/{dn}",**S.sum())
        L=[]
        with Samp() as S:
            for rep in (0,1):
                t=post({"prompt":LONG,"n_predict":128,"ignore_eos":True,"cache_prompt":True,"temperature":0,"id_slot":0}); L.append(t)
        out(sec="52k",cap=cap,tg=round(sum(x["predicted_n"] for x in L)/sum(x["predicted_ms"]/1000 for x in L),3),n=sum(x["predicted_n"] for x in L),prompt_n=[x["prompt_n"] for x in L],accept=[f'{x.get("draft_n_accepted")}/{x.get("draft_n")}' for x in L],thr=smi("clocks_throttle_reasons.active"),**S.sum())
finally:
    pl(250); out(sec="end",lim=smi("power.limit"),ac=smi("clocks.applications.memory,clocks.applications.graphics"))
