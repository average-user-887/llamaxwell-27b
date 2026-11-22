#!/usr/bin/env python3
"""22 Nov: two-slot agentic makespan fixture on the quick.py base (model: tg_long.py). Caller holds gpu-bench.lock.
Usage: agentic2.py [reps=3] [filler_reps=2000] [tool_reps=90]
Server config comes from q.load()["base"]; set base np=2 (and a c >= 2x~100K) beforehand, else the run errors out.
Per rep: unique seed token at the start of each slot's prompt (no cross-rep/cross-slot prefix reuse).
Phase 1 (logged, not in metric): both slots prefill distinct ~90K-token filler, n_predict 1, cache_prompt true.
Phase 2 (timed makespan): same prompt + ~4K-token distinct tool-result chunk + TG instruction, n_predict 128 ignore_eos, temp 0."""
import sys, json, time, threading, os
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FILL = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
TOOL = int(sys.argv[3]) if len(sys.argv) > 3 else 90

def filler(tag, n):
    return " ".join("word%d alpha beta gamma delta, the quick brown fox %d %s." % (i, i * 7, tag) for i in range(n))

def tool_chunk(tag, n):
    return "\n\n[tool_result %s]\n" % tag + "\n".join("row %d: status ok, latency %d ms, path /srv/%s/item%d, checksum %x" % (i, (i * 37) % 997, tag, i, (i * 2654435761) & 0xffffff) for i in range(n)) + "\n\n"

def go(slot, prompt, n_pred, out):
    t0 = time.time()
    r = q.post("/completion", {"prompt": prompt, "n_predict": n_pred, "ignore_eos": True, "cache_prompt": True, "temperature": 0, "id_slot": slot})
    out[slot] = (time.time() - t0, r.get("timings", {}))

def phase(prompts, n_pred):
    out = {}; th = [threading.Thread(target=go, args=(s, p, n_pred, out)) for s, p in enumerate(prompts)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    return time.time() - t0, out

def slotinfo(out):
    return [{"slot": s, "wall": round(w, 2), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "pp": round(t.get("prompt_per_second", 0), 2),
             "tg": round(t.get("predicted_per_second", 0), 2), "accept": f'{t.get("draft_n_accepted")}/{t.get("draft_n")}'} for s, (w, t) in sorted(out.items())]

if __name__ == "__main__":
    c = q.load()["base"]
    print(json.dumps({"identity": {"so": c.get("so"), "env": {**q.ENV, **c.get("env", {})}, "np": c.get("np"), "c": c.get("c"), "kv": c.get("kv"),
                                     "ub": c.get("ub"), "model": c.get("model"), "extra": c.get("extra")}}), flush=True)
    ok, err = q.start(c)
    if not ok: print(json.dumps({"error": err[-300:]})); sys.exit(2)
    spans = []
    try:
        for rep in range(REPS):
            seed = "seed%d_%d" % (os.getpid(), rep)
            base = [("S%d %s " % (s, seed)) + filler("%s_s%d" % (seed, s), FILL) for s in range(2)]
            w1, o1 = phase(base, 1)
            print(json.dumps({"rep": rep, "phase": "prefill", "wall": round(w1, 2), "slots": slotinfo(o1)}), flush=True)
            ext = [base[s] + tool_chunk("%s_s%d" % (seed, s), TOOL) + "\n" + q.TG for s in range(2)]
            w2, o2 = phase(ext, 128)
            spans.append(w2)
            print(json.dumps({"rep": rep, "phase": "makespan", "makespan_s": round(w2, 2), "slots": slotinfo(o2)}), flush=True)
        print(json.dumps({"summary": "agentic2", "reps": REPS, "makespan_s": [round(x, 2) for x in spans], "makespan_mean_s": round(sum(spans) / len(spans), 2),
                          "makespan_min_s": round(min(spans), 2)}), flush=True)
    finally:
        q.sh("docker", "rm", "-f", q.NAME)
