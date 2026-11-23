#!/usr/bin/env python3
"""22 Nov: two-slot agentic makespan fixture on the quick.py base (model: tg_long.py). Caller holds gpu-bench.lock.
Usage: agentic2.py [reps=2] [filler_reps=auto(0)|N, fallback 2800 ~32 tok/rep] [tool_reps=90] [target_tok=90000]
Each rep needs fresh filler (unique seed per rep/slot, so no cross-rep prefix reuse).
Server config comes from q.load()["base"]; set base np=2 (and a c >= 2x~100K) beforehand, else the run errors out.
Per rep: unique seed token at the start of each slot's prompt (no cross-rep/cross-slot prefix reuse).
Phase 1 (logged, not in metric): slots prefilled SEQUENTIALLY (slot 0 then 1, timeout 3600), each distinct ~90K-token filler, n_predict 1, cache_prompt true.
Phase 2 (timed makespan): same prompt + ~4K-token distinct tool-result chunk + TG instruction, n_predict 128 ignore_eos, temp 0."""
import sys, json, time, threading, os
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
FILL = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = auto-calibrate via /tokenize, fallback 2800
TOOL = int(sys.argv[3]) if len(sys.argv) > 3 else 90
TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 90000

def filler(tag, n):
    return " ".join("word%d alpha beta gamma delta, the quick brown fox %d %s." % (i, i * 7, tag) for i in range(n))

def tool_chunk(tag, n):
    return "\n\n[tool_result %s]\n" % tag + "\n".join("row %d: status ok, latency %d ms, path /srv/%s/item%d, checksum %x" % (i, (i * 37) % 997, tag, i, (i * 2654435761) & 0xffffff) for i in range(n)) + "\n\n"

def calibrate():
    try:
        n = 200; r = q.post("/tokenize", {"content": filler("probe", n)}, timeout=120)
        per = len(r["tokens"]) / n
        return max(1, int(TARGET / per)), round(per, 2)
    except Exception as e:
        print(json.dumps({"calibrate_failed": str(e)[:200]}), flush=True); return 2800, None

def go(slot, prompt, n_pred, out, timeout=1800):
    t0 = time.time()
    r = q.post("/completion", {"prompt": prompt, "n_predict": n_pred, "ignore_eos": True, "cache_prompt": True, "temperature": 0, "id_slot": slot}, timeout=timeout)
    out[slot] = (time.time() - t0, r.get("timings", {}))

def phase(prompts, n_pred):
    out = {}; th = [threading.Thread(target=go, args=(s, p, n_pred, out)) for s, p in enumerate(prompts)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    return time.time() - t0, out

def prefill_seq(prompts):
    out = {}; t0 = time.time()
    for s, p in enumerate(prompts): go(s, p, 1, out, 3600)
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
    spans = []; tgs = []
    fill, per = (FILL, None) if FILL else calibrate()
    print(json.dumps({"filler_reps": fill, "tok_per_rep": per, "target_tok": TARGET}), flush=True)
    try:
        for rep in range(REPS):
            seed = "seed%d_%d" % (os.getpid(), rep)
            base = [("S%d %s " % (s, seed)) + filler("%s_s%d" % (seed, s), fill) for s in range(2)]
            w1, o1 = prefill_seq(base)
            print(json.dumps({"rep": rep, "phase": "prefill", "wall": round(w1, 2), "slots": slotinfo(o1)}), flush=True)
            ext = [base[s] + tool_chunk("%s_s%d" % (seed, s), TOOL) + "\n" + q.TG for s in range(2)]
            w2, o2 = phase(ext, 128)
            valid = all((o2[s][1].get("cache_n") or 0) >= 0.9 * (o1[s][1].get("prompt_n") or 0) > 0 for s in range(2))
            tgs.append([round(o2[s][1].get("predicted_per_second", 0), 2) for s in range(2)])
            if valid: spans.append(w2)
            print(json.dumps({"rep": rep, "phase": "makespan", "valid": valid, "makespan_s": round(w2, 2), "slots": slotinfo(o2)}), flush=True)
        print(json.dumps({"summary": "agentic2", "reps": REPS, "valid_reps": len(spans), "makespan_s": [round(x, 2) for x in spans],
                          "makespan_mean_s": round(sum(spans) / len(spans), 2) if spans else None, "makespan_min_s": round(min(spans), 2) if spans else None,
                          "tg_per_slot_p2": tgs}), flush=True)
    finally:
        q.sh("docker", "rm", "-f", q.NAME)
