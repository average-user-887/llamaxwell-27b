#!/usr/bin/python3 -I
"""opt27b frozen benchmark + gates (frozen 6 Nov 2026 for the 7-13 Nov loop).
Usage: bench.py <port> <outdir> <variant> <args-string>
Score: pp4k/pp32k (random tokens, as exp-27b-serving), pp at needle length (real text), tg1, tg2 agg,
warm-resume TTFT (cache-ram path and slot-file path), peak VRAM.
Gates: needle (120k, or slot ctx-1.5k if smaller), temp-0 probes vs variant P reference, 3-task coding slice.
Appends one JSON line to LEDGER.jsonl next to this file."""
import sys, os, json, time, random, threading, subprocess, urllib.request, glob, re, tempfile
port, out, var, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
U = f"http://127.0.0.1:{port}"
HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "LEDGER.jsonl"); REF = os.path.join(HERE, "probe-ref-P.json")

def post(path, body, timeout=7200):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def get(path): return json.load(urllib.request.urlopen(U + path, timeout=30))
def vram(): return int(subprocess.run(["nvidia-smi", "--id=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip() or 0)
peak = [0]; stop = [False]
def sampler():
    while not stop[0]:
        peak[0] = max(peak[0], vram()); time.sleep(2)
threading.Thread(target=sampler, daemon=True).start()
res = {"variant": var, "args": args, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "vram_idle_mib": vram()}
def step(name, fn):
    t = time.time()
    try: res[name] = fn()
    except Exception as e: res[name] = {"error": repr(e)[:300]}
    res.setdefault("step_s", {})[name] = round(time.time() - t, 1)
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)

slots = get("/slots"); res["slots"] = len(slots); nctx = slots[0].get("n_ctx", 0); res["slot_ctx"] = nctx
try: res["build"] = get("/props").get("build_info")
except Exception: pass

def pp_rand(n, seed):
    random.seed(seed); toks = [random.randint(2000, 150000) for _ in range(n)]
    r = post("/completion", {"prompt": toks, "n_predict": 1, "cache_prompt": False, "temperature": 0})
    return {"n": r["timings"]["prompt_n"], "pps": round(r["timings"]["prompt_per_second"], 1)}
TG = ("Write a Python module implementing an LRU cache with TTL, thread safety, type hints and docstrings, "
      "then explain its complexity in detail.")
def tg_once(seed, o):
    r = post("/completion", {"prompt": TG + f" (variant {seed})", "n_predict": 256, "ignore_eos": True, "cache_prompt": False, "temperature": 0})
    t = r["timings"]; o.append({"tps": round(t["predicted_per_second"], 2), "n": t["predicted_n"], "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")})
def tg1():
    o = []; tg_once(1, o); return o[0]
def tg2():
    o = []; t = time.time(); th = [threading.Thread(target=tg_once, args=(i, o)) for i in (2, 3)]
    [x.start() for x in th]; [x.join() for x in th]; w = time.time() - t
    return {"streams": o, "agg_tps": round(sum(x["n"] for x in o) / w, 2)}

# --- real-text corpus (stdlib sources; local only) ---
def corpus_tokens(n):
    txt = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3.12/*.py")))
    toks = post("/tokenize", {"content": txt[: n * 6]})["tokens"]
    return toks[:n]
NEEDLE = "\n# NOTE FOR AUDITORS: the depot access phrase is PURPLE-MANGO-4719. Remember it.\n"
def needle():
    n = min(118000, nctx - 1500)
    body = corpus_tokens(n - 200)
    ntok = post("/tokenize", {"content": NEEDLE})["tokens"]
    q = post("/tokenize", {"content": "\n\nQuestion: what is the depot access phrase mentioned in the auditors' note above? Answer with the phrase only.\nAnswer:"})["tokens"]
    mid = int(len(body) * 0.45); prompt = body[:mid] + ntok + body[mid:] + q
    r = post("/completion", {"prompt": prompt, "n_predict": 16, "cache_prompt": False, "temperature": 0})
    return {"n": r["timings"]["prompt_n"], "pps": round(r["timings"]["prompt_per_second"], 1),
            "wall_s": round(r["timings"]["prompt_ms"] / 1000, 1), "answer": r["content"][:80],
            "pass": "PURPLE-MANGO-4719" in r["content"]}

PROBES = ["The three laws of thermodynamics, explained for a twelve-year-old:\n",
          "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
          "Q: A train leaves at 14:35 and the trip takes 2 h 50 min. When does it arrive?\nA: Let's think step by step.",
          "Translate to Polish: 'The server restarts every night at three, and nobody knows why.'\nPolish:",
          "List five failure modes of a lithium-ion battery pack and one mitigation for each:\n1."]
def probes():
    outs = []
    for p in PROBES:
        r = post("/completion", {"prompt": p, "n_predict": 160, "cache_prompt": False, "temperature": 0, "seed": 1, "n_probs": 0})
        outs.append(r["content"])
    if var == "P" and not os.path.exists(REF): json.dump(outs, open(REF, "w"), indent=1)
    if not os.path.exists(REF): return {"outs": outs, "note": "no reference"}
    ref = json.load(open(REF)); sc = []
    for a, b in zip(outs, ref):
        ta = post("/tokenize", {"content": a})["tokens"]; tb = post("/tokenize", {"content": b})["tokens"]
        k = 0
        while k < min(len(ta), len(tb)) and ta[k] == tb[k]: k += 1
        sc.append(round(k / max(1, len(tb)), 3))
    return {"prefix_match": sc, "mean": round(sum(sc) / len(sc), 3), "identical": sum(1 for a, b in zip(outs, ref) if a == b), "outs": outs}

CODE = [("Write a Python function `is_balanced(s: str) -> bool` that returns True when the brackets ()[]{} in s are balanced. Reply with only a python code block.",
         "assert is_balanced('([]{})') and not is_balanced('([)]') and is_balanced('') and not is_balanced('((')"),
        ("Write a Python function `roman_to_int(s: str) -> int` converting a Roman numeral to an integer. Reply with only a python code block.",
         "assert roman_to_int('MCMXCIV')==1994 and roman_to_int('LVIII')==58 and roman_to_int('IV')==4"),
        ("Write a Python function `top_k_words(text: str, k: int) -> list[str]` returning the k most frequent lowercase words (ties broken alphabetically). Words are runs of letters. Reply with only a python code block.",
         "assert top_k_words('b a b c a b', 2)==['b','a'] and top_k_words('x y z', 2)==['x','y']")]
def coding():
    ok = 0; det = []
    for q, t in CODE:
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 700,
                                          "chat_template_kwargs": {"enable_thinking": False}})
        txt = r["choices"][0]["message"]["content"] or ""
        m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S); code = m.group(1) if m else txt
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "t.py"); open(f, "w").write(code + "\n" + t + "\nprint('OK')\n")
            p = subprocess.run([sys.executable, "-I", f], capture_output=True, text=True, timeout=20, cwd=d)
        good = p.stdout.strip().endswith("OK"); ok += good; det.append(good)
    return {"pass": ok, "of": len(CODE), "detail": det}

def resume():
    """Session switch: 16k real-text context on slot 0, a different session uses the server, then resume."""
    ctx = corpus_tokens(16000); tail = post("/tokenize", {"content": "\n\nSummarise the code above in one line:"})["tokens"]
    a = post("/completion", {"prompt": ctx + tail, "n_predict": 8, "cache_prompt": True, "temperature": 0, "id_slot": 0})
    cold_ms = a["timings"]["prompt_ms"]
    # other session takes slot 0 (on 1-slot servers this evicts the context; cache-ram may keep it)
    post("/completion", {"prompt": corpus_tokens(2000)[::-1], "n_predict": 4, "cache_prompt": True, "temperature": 0, "id_slot": 0})
    t = time.time(); b = post("/completion", {"prompt": ctx + tail, "n_predict": 1, "cache_prompt": True, "temperature": 0, "id_slot": 0})
    warm = {"ttft_s": round(time.time() - t, 2), "prompt_n_recomputed": b["timings"]["prompt_n"]}
    # slot-file path: save, evict, erase, restore, resume
    post(f"/slots/0?action=save", {"filename": f"r-{var}.bin"})
    post("/completion", {"prompt": corpus_tokens(2000)[::-1], "n_predict": 4, "cache_prompt": True, "temperature": 0, "id_slot": 0})
    t = time.time(); rr = post(f"/slots/0?action=restore", {"filename": f"r-{var}.bin"})
    c = post("/completion", {"prompt": ctx + tail, "n_predict": 1, "cache_prompt": True, "temperature": 0, "id_slot": 0})
    return {"cold_prefill_s": round(cold_ms / 1000, 1), "after_switch_cache_ram": warm,
            "slot_restore": {"ttft_s": round(time.time() - t, 2), "restore_ms": rr.get("timings", {}).get("restore_ms"),
                             "prompt_n_recomputed": c["timings"]["prompt_n"]}}

step("tg1", tg1)
if res["slots"] >= 2: step("tg2", tg2)
step("pp4k", lambda: pp_rand(4096, 11))
step("pp32k", lambda: pp_rand(32768, 12))
step("probes", probes)
step("coding", coding)
step("resume", resume)
step("needle", needle)
stop[0] = True; res["vram_peak_mib"] = peak[0]; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
slim = {k: v for k, v in res.items() if k != "probes"}; slim["probes"] = {k: v for k, v in (res.get("probes") or {}).items() if k != "outs"}
with open(LEDGER, "a") as f: f.write(json.dumps(slim) + "\n")
print(json.dumps(slim))
