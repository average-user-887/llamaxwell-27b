#!/usr/bin/env python3
"""8 Nov fadec: A/B GGML_CUDA_FA_DEC 0/1 with the fadec lib; tg at long context on the quick.py base. Filler prompt (as fa_eval) + TG instruction, n_predict 128. Caller holds lock."""
import sys, json, time, urllib.request
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113"); import quick as q
import copy
SO = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec.so"
for flag in ("0", "1", "0", "1"):  # order 0,1,0,1 ; each start = 1 run per rep (2 runs/flag total)
  c = copy.deepcopy(q.load()["base"]); c["so"] = SO; c["env"]["GGML_CUDA_FA_DEC"] = flag; ok, err = q.start(c)
  if not ok: print(json.dumps({"error": err[-300:]})); sys.exit(2)
  for reps in [0] + [int(x) for x in sys.argv[1:]]:
      pr = " ".join("word%d alpha beta gamma delta, the quick brown fox %d." % (i, i * 7) for i in range(reps)) + "\n\n" + q.TG
      t = q.post("/completion", {"prompt": pr, "n_predict": 128, "ignore_eos": True, "cache_prompt": False, "temperature": 0})["timings"]
      print(json.dumps({"flag": flag, "reps": reps, "prompt_n": t["prompt_n"], "pp": round(t["prompt_per_second"], 2), "tg": round(t["predicted_per_second"], 2),
                        "accept": f'{t.get("draft_n_accepted")}/{t.get("draft_n")}'}), flush=True)
  q.sh("docker", "rm", "-f", q.NAME)
