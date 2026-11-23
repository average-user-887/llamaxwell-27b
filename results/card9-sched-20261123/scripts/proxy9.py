#!/usr/bin/python3 -I
"""Card 7 proxy for agentic2 makespan: 2 slots at ~92K depth (restored from one saved slot), timed parallel
+3873-token tool chunk + 128 tok decode per slot (agentic2 phase 2 shape). Caller holds gpu-bench.lock.
Usage: proxy.py prep            -> prefill slot 0 (~92K, base lib), save c7base.bin
       proxy.py arms A B B A    -> per arm: start container, 2 samples of (restore both slots, timed phase)"""
import sys, json, time, os
sys.path.insert(0, "$M/opt27b-20261113"); import quick as q
_av = sys.argv; sys.argv = sys.argv[:1]; import agentic2 as a2; sys.argv = _av
q.SLOTS = "$MODELS/c7-slots"
SRV = "$M/prod-libs/sched-20261123/libllama-server-impl.so"
import hashlib
_sh = q.sh
def _sh2(*a, **k):
    a = list(a)
    if a[:2] == ["docker", "run"] and "-d" in a:
        i = a.index(q.IMG); a[i:i] = ["-v", SRV + ":/app/libllama-server-impl.so:ro"]
    return _sh(*a, **k)
q.sh = _sh2
def _go(slot, prompt, n_pred, out, timeout=1800):
    t0 = time.time()
    r = q.post("/completion", {"prompt": prompt, "n_predict": n_pred, "ignore_eos": True, "cache_prompt": True, "temperature": 0, "id_slot": slot}, timeout=timeout)
    t = dict(r.get("timings", {})); t["sha"] = hashlib.sha256(r.get("content", "").encode()).hexdigest()[:12]
    out[slot] = (time.time() - t0, t)
a2.go = _go
_si = a2.slotinfo
a2.slotinfo = lambda o: [dict(d, sha=o[d["slot"]][1].get("sha")) for d in _si(o)]
SEED = "seedc7_0"; FILL = 2879
base = "S0 %s " % SEED + a2.filler("%s_s0" % SEED, FILL)
ENVS = {"A": {}, "B": {"LLAMA_SRV_PREFILL_FIRST": "1"}, "C": {"LLAMA_SRV_MIXED_CAP": "1024"}, "D": {"LLAMA_SRV_MIXED_CAP": "1024", "LLAMA_SRV_PREFILL_FIRST": "1"}}
def cfg(arm):
    c = json.loads(json.dumps(q.load()["base"])); c["env"].update(ENVS.get(arm, {})); return c
def P(o): print(json.dumps(o), flush=True)
if sys.argv[1] == "prep":
    ok, err = q.start(cfg("P")); P({"start": ok, "err": err[-200:]})
    try:
        w, o = a2.prefill_seq([base]); P({"prefill_wall": round(w, 1), "slots": a2.slotinfo(o)})
        P({"save": q.post("/slots/0?action=save", {"filename": "c7base.bin"})})
    finally: q.sh("docker", "rm", "-f", q.NAME)
elif sys.argv[1] == "quick":
    os.environ["QUICK_LONG"] = "1"
    for i, arm in enumerate(sys.argv[2:]):
        ok, err = q.start(cfg(arm)); P({"arm": arm, "i": i, "start": ok, "err": err[-200:]})
        if not ok: break
        try: P({"arm": arm, "i": i, "quick": q.measure(cfg(arm))})
        finally: q.sh("docker", "rm", "-f", q.NAME)
else:
    for i, arm in enumerate(sys.argv[2:]):
        ok, err = q.start(cfg(arm)); P({"arm": arm, "i": i, "start": ok, "err": err[-200:]})
        if not ok: break
        try:
            for smp in range(2):
                for s in range(2): q.post("/slots/%d?action=restore" % s, {"filename": "c7base.bin"})
                ext = [base + a2.tool_chunk("%s_s%d_%d_%d" % (SEED, s, i, smp), 90) + "\n" + q.TG for s in range(2)]
                w, o = a2.phase(ext, 128)
                P({"arm": arm, "i": i, "smp": smp, "makespan_s": round(w, 2), "slots": a2.slotinfo(o)})
        finally: q.sh("docker", "rm", "-f", q.NAME)
