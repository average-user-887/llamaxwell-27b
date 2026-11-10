#!/usr/bin/env python3
"""8 Nov fadec2: A/B GGML_CUDA_FA_DEC 0/2 with the fadec2 lib; tg (and pp) at long context on the quick.py base. Filler prompt + TG, n_predict 128.
Usage: tg_long_fadec2.py reps... ; env ORDER (default "0,2,0,2"), XENV (json, extra server env applied to all runs, e.g. {"GGML_CUDA_FA_DEC_MINQ":"1"}),
DRAFTN ("none" = MTP off), TAG (label). Caller holds lock. Removes only the exp container."""
import sys, os, json, copy
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
SO = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so"
XENV = json.loads(os.environ.get("XENV", "{}")); TAG = os.environ.get("TAG", "")
for flag in os.environ.get("ORDER", "0,2,0,2").split(","):
  c = copy.deepcopy(q.load()["base"]); c["so"] = SO; c["env"]["GGML_CUDA_FA_DEC"] = flag; c["env"].update(XENV)
  if os.environ.get("DRAFTN") == "none": c["draftn"] = None; c["extra"] = [x for x in c.get("extra", [])] + ["--spec-type", "none"]
  ok, err = q.start(c)
  if not ok: print(json.dumps({"error": err[-300:]})); q.sh("docker", "rm", "-f", q.NAME); sys.exit(2)
  try:
    for reps in [0] + [int(x) for x in sys.argv[1:]]:
      pr = " ".join("word%d alpha beta gamma delta, the quick brown fox %d." % (i, i * 7) for i in range(reps)) + "\n\n" + q.TG
      t = q.post("/completion", {"prompt": pr, "n_predict": 128, "ignore_eos": True, "cache_prompt": False, "temperature": 0})["timings"]
      print(json.dumps({"tag": TAG, "flag": flag, "reps": reps, "prompt_n": t["prompt_n"], "pp": round(t["prompt_per_second"], 2), "tg": round(t["predicted_per_second"], 2),
                        "accept": f'{t.get("draft_n_accepted")}/{t.get("draft_n")}'}), flush=True)
  finally:
    q.sh("docker", "rm", "-f", q.NAME)
