#!/usr/bin/env python3
"""9 Nov Scout 3 Stage 1 speed. Usage: stage1.py short|long ARM... where ARM is C (C4) or F (C4+FP32MV). Exact live prod env + prod lib on the quick.py base. Caller holds lock, prod stopped."""
import sys,json,time,copy,threading,subprocess
sys.path.insert(0,"/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
SO="/media/avg-usr/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-fadec2-20261109.so"
RES="/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/fp32mv-stage1-20261109.jsonl"
LIVE={"CONCAT_T":"1","FA_DEC":"2","FA_GEMM":"1","GATE_COPY":"1","GDN_COLS":"2,4","HEAD_ROWPACK_SM52":"1","MMQ_COMPACT_SM52":"1","MMQ_TAIL_SM52":"1","MMQ_XMAD_SM52":"1","NORM_Q8":"1","OCCAMY_FA":"1","OCCAMY_FUSED":"1","OCCAMY_GDN":"1","OCCAMY_GDN_STATE_DIRECT":"1","Q8_REUSE":"1","RMS_WARP":"0","SGEMM_NN":"1"}
LIVE={"GGML_CUDA_"+k:v for k,v in LIVE.items()}
PR=q.PROBES+[q.TG,"Explain how a B-tree insert splits a full node, with a small worked example:\n","def parse_iso_duration(s: str) -> int:\n    \"\"\"Return seconds for an ISO 8601 duration.\"\"\"\n","Summarise the causes of the 1929 stock market crash in plain language.\n","Translate to Polish and explain the grammar: The cat sat on the warm windowsill all afternoon.\n"]
def out(**k):
    k["t"]=time.strftime("%H:%M:%S"); s=json.dumps(k); print(s,flush=True); open(RES,"a").write(s+"\n")
def cfg(arm):
    c=copy.deepcopy(q.load()["base"]); c["so"]=SO; c["env"]={k:None for k in q.ENV}; c["env"].update(LIVE)
    if arm=="F": c["env"]["GGML_CUDA_FP32MV"]="1"
    return c
class Samp:
    def __enter__(s):
        s.r=[]; s.stop=False; s.th=threading.Thread(target=s.run,daemon=True); s.th.start(); return s
    def run(s):
        while not s.stop:
            o=subprocess.run(["nvidia-smi","--id="+q.GPU,"--query-gpu=temperature.gpu,power.draw,clocks.gr","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
            try:
                v=[float(x) for x in o.split(",")]; s.r.append(v)
                if v[0]>=83: subprocess.run(["docker","stop","-t","2",q.NAME])
            except ValueError: pass
            time.sleep(5)
    def __exit__(s,*a): s.stop=True
    def sum(s):
        b=[x for x in s.r if x[1]>150] or s.r or [[0,0,0]]
        return dict(maxT=max(x[0] for x in s.r) if s.r else 0,W=round(sum(x[1] for x in b)/len(b),1),MHz=round(sum(x[2] for x in b)/len(b)))
def short(arm):
    m=q.measure(cfg_c)
    n=d=a=dn=0
    with Samp() as S:
        for i,p in enumerate(PR):
            t=q.post("/completion",{"prompt":p,"n_predict":128,"ignore_eos":True,"cache_prompt":False,"temperature":0})["timings"]
            n+=t["predicted_n"]; d+=t["predicted_ms"]/1000; a+=t.get("draft_n_accepted") or 0; dn+=t.get("draft_n") or 0
            out(sec="p",arm=arm,i=i,tg=round(t["predicted_per_second"],2))
    out(sec="short",arm=arm,pp4k=m.get("pp4k"),tg1=m.get("tg1"),tg2agg=m.get("tg2agg"),pooled=round(n/d,3),accept=f"{a}/{dn}",err=m.get("error"),**S.sum())
def long(arm):
    tk=lambda s:len(q.post("/tokenize",{"content":s})["tokens"])
    fill=lambda r:" ".join("word%d alpha beta gamma delta, the quick brown fox %d."%(i,i*7) for i in range(r))+"\n\n"+q.TG
    per=tk(fill(1000))/1000
    for tag,tgt in (("52k",51900),("100k",100000)):
        p=fill(int(tgt/per))
        for rep in (0,1):
            with Samp() as S:
                t=q.post("/completion",{"prompt":p,"n_predict":128,"ignore_eos":True,"cache_prompt":True,"temperature":0,"id_slot":0},timeout=7200)["timings"]
            out(sec=tag,arm=arm,rep=rep,prompt_n=t["prompt_n"],cached=t.get("cache_n"),pp=round(t["prompt_per_second"],2),tg=round(t["predicted_per_second"],2),accept=f'{t.get("draft_n_accepted")}/{t.get("draft_n")}',**S.sum())
sec=sys.argv[1]
for arm in sys.argv[2:]:
    cfg_c=cfg(arm); ok,err=q.start(cfg_c)
    if not ok: out(sec="error",arm=arm,err=err[-300:]); q.sh("docker","rm","-f",q.NAME); sys.exit(2)
    try: (short if sec=="short" else long)(arm)
    finally: q.sh("docker","rm","-f",q.NAME)
