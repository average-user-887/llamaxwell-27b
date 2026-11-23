import sys,threading,copy
sys.path.insert(0,"/media/$USER/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
c=copy.deepcopy(q.load()["base"]); c["extra"]=["-lv","4"]
ok,err=q.start(c)
try:
    if ok:
        def one(i): q.post("/completion",{"prompt":q.TG+f" [{i}]","n_predict":96,"ignore_eos":True,"cache_prompt":False,"temperature":0})
        th=[threading.Thread(target=one,args=(i,)) for i in (1,2)]; [t.start() for t in th]; [t.join() for t in th]
        open("verb-docker.log","w").write(q.sh("docker","logs",q.NAME).stderr)
    else: print("ERR",err)
finally: q.sh("docker","rm","-f",q.NAME)
