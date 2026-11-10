#!/usr/bin/env python3
"""7 Nov: A/B long-context prefill for GEMM FA (GGML_CUDA_FA_GEMM). usage: fa_eval.py <so> <reps...>
Same base config as quick.py state; only GGML_CUDA_FA_GEMM differs. Prod stop/restore is done by the caller."""
import sys, json, copy, time, urllib.request
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113")
import quick as q

def pp(reps):
    pr = " ".join("word%d alpha beta gamma delta, the quick brown fox %d." % (i, i * 7) for i in range(reps))
    body = json.dumps({"prompt": pr, "n_predict": 4, "temperature": 0, "cache_prompt": False}).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{q.PORT}/completion", body, {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(rq, timeout=3600))
    return d["timings"]["prompt_n"], round(d["timings"]["prompt_per_second"], 2), d["content"]

so = sys.argv[1]; reps = [int(x) for x in sys.argv[2:]] or [130, 1200]
base = q.load()["base"]
import os
for var in os.environ.get("VARS", "0 1").split():
    fg, _, kc = var.partition(":")
    c = copy.deepcopy(base); c["so"] = so; c["env"]["GGML_CUDA_FA_GEMM"] = fg
    if kc: c["env"]["GGML_CUDA_FA_GEMM_KC"] = kc
    ok, err = q.start(c)
    if not ok: print(json.dumps({"fa_gemm": fg, "error": err[-400:]})); continue
    for r in reps:
        try: n, t, out = pp(r); print(json.dumps({"var": var, "prompt_n": n, "pp": t, "out": out}), flush=True)
        except Exception as e: print(json.dumps({"fa_gemm": fg, "reps": r, "error": repr(e)}), flush=True)
    q.sh("docker", "rm", "-f", q.NAME)
