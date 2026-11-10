#!/usr/bin/env python3
"""7 Nov card C: A/B GGML_CUDA_MMVQ_SM52 0 vs 1 on the quick.py base (new .so). 3 reps each, interleaved. Caller holds gpu-bench.lock.
Also checks bit-exactness: 128-token temp-0 seed-1 completion and the quick probes must be identical across flags."""
import sys, json, copy, statistics
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
SO = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-sm52.so"
base = q.load()["base"]; res = {"0": [], "1": []}; outs = {"0": [], "1": []}
for rep in range(3):
    for fl in ("0", "1"):
        c = copy.deepcopy(base); c["so"] = SO; c["env"]["GGML_CUDA_MMVQ_SM52"] = fl
        ok, err = q.start(c)
        if not ok: print(json.dumps({"flag": fl, "rep": rep, "error": err[-400:]}), flush=True); continue
        m = q.measure(c)
        try:
            t = q.post("/completion", {"prompt": q.TG, "n_predict": 128, "temperature": 0, "seed": 1, "ignore_eos": True, "cache_prompt": False})
            m["exact128"] = t["content"]
        except Exception as e: m["exact128"] = "ERR " + repr(e)
        q.sh("docker", "rm", "-f", q.NAME)
        res[fl].append(m); outs[fl].append((m.get("exact128"), m.get("probe_outs")))
        print(json.dumps({"flag": fl, "rep": rep, **{k: m.get(k) for k in ("pp4k", "tg1", "tg2agg", "accept", "vram_peak", "error")}}), flush=True)
def med(fl, k):
    v = [m[k] for m in res[fl] if m.get(k)]; return round(statistics.median(v), 2) if v else None
print("SUMMARY (median of reps)")
for k in ("pp4k", "tg1", "tg2agg"):
    a, b = med("0", k), med("1", k); print(k, a, b, f"{(b/a-1):+.2%}" if a and b else "n/a")
print("accept flag0", [m.get("accept") for m in res["0"]], "flag1", [m.get("accept") for m in res["1"]])
allo = outs["0"] + outs["1"]
print("bit-exact (128-tok + probes identical across all runs):", len(set(json.dumps(o) for o in allo)) == 1 if allo else "n/a")
