#!/usr/bin/env python3
"""8 Nov card E: tg at long context on the C3 base, MTP draft length from env DRAFT (2, 1, or 0 = MTP off).
Copy of tg_long.py; only the server args differ. Filler prompt (as fa_eval) + TG instruction, n_predict 128. Caller holds lock."""
import os, sys, json
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
DRAFT = os.environ.get("DRAFT", "2")
if DRAFT not in ("0", "1", "2"): print(json.dumps({"error": "DRAFT must be 0,1,2"})); sys.exit(2)
_orig_args_of = q.args_of
def _args_of(c):
    c = dict(c); c["draftn"] = None if DRAFT == "0" else int(DRAFT)
    a = _orig_args_of(c)
    if DRAFT == "0":  # drop MTP: --spec-type draft-mtp and --spec-draft-n-max
        for flag, n in (("--spec-type", 2), ("--spec-draft-n-max", 2)):
            while flag in a:
                i = a.index(flag); del a[i:i + n]
    return a
q.args_of = _args_of
c = q.load()["base"]; ok, err = q.start(c)
if not ok: print(json.dumps({"draft": DRAFT, "error": err[-300:]})); sys.exit(2)
for reps in [0] + [int(x) for x in sys.argv[1:]]:
    pr = " ".join("word%d alpha beta gamma delta, the quick brown fox %d." % (i, i * 7) for i in range(reps)) + "\n\n" + q.TG
    t = q.post("/completion", {"prompt": pr, "n_predict": 128, "ignore_eos": True, "cache_prompt": False, "temperature": 0})["timings"]
    print(json.dumps({"draft": DRAFT, "reps": reps, "prompt_n": t["prompt_n"], "pp": round(t["prompt_per_second"], 2), "tg": round(t["predicted_per_second"], 2),
                      "accept": f'{t.get("draft_n_accepted")}/{t.get("draft_n")}'}), flush=True)
q.sh("docker", "rm", "-f", q.NAME)
