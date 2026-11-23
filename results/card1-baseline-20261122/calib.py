import sys,json
sys.path.insert(0,"/media/$USER/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q, agentic2 as a
c=q.load()["base"]; ok,err=q.start(c)
if not ok: print("ERR",err); sys.exit(2)
try:
    n1=len(q.post("/tokenize",{"content":a.filler("seed1234_0_s0",1000)})["tokens"])
    n2=len(q.post("/tokenize",{"content":a.tool_chunk("seed1234_0_s0",90)})["tokens"])
    per=n1/1000; fill=int((90000-n2-200)/per)
    print(json.dumps({"per_rep":per,"tool_tok":n2,"fill":fill}),flush=True); open(__import__("os").environ["FILLOUT"],"w").write(str(fill))
finally: q.sh("docker","rm","-f",q.NAME)
