#!/usr/bin/python3 -I
"""Quick keep-or-revert loop for Qwen3.8-27B on the M6000 (occamy-style ledger rows). 6 Nov 2026.
Usage: quick.py <round 1..4>. Holds the M6000 for <=85 min, then restores prod (llamacpp-m6000-27b).
State: state.json (base config + metrics + probe ref). Rows: LEDGER.md + ledger.jsonl.
Keep rule: tg1 or pp4k >= +3% vs base, the other >= -3%, probes prefix-match mean >= 0.95 vs base outputs."""
import sys, os, json, time, random, subprocess, threading, urllib.request, copy
HERE = os.path.dirname(os.path.abspath(__file__)); M = "/media/avg-usr/Storage/m6000-llama-cpp"
ST = os.path.join(HERE, "state.json"); LMD = os.path.join(HERE, "LEDGER.md"); LJS = os.path.join(HERE, "ledger.jsonl")
GPU = "GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e"; IMG = "sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d"
NAME, PORT, PROD, PROD_PORT = "llamacpp-m6000-27b-exp", 11448, "llamacpp-m6000-27b", 11437
SLOTS = "/home/avg-usr/.claude/jobs/ee277f80/tmp/slots"; CAP_S = 6 * 3600; T0 = time.time()
DEADLINE = time.mktime(time.strptime("2026-11-13 08:00", "%Y-%m-%d %H:%M"))
ENV = dict(GGML_CUDA_RMS_WARP="0", GGML_CUDA_HEAD_ROWPACK_SM52="1", GGML_CUDA_MMQ_COMPACT_SM52="1", GGML_CUDA_Q8_REUSE="1",
           GGML_CUDA_NORM_Q8="1", GGML_CUDA_CONCAT_T="1", GGML_CUDA_OCCAMY_GDN_STATE_DIRECT="1", GGML_CUDA_GATE_COPY="1",
           GGML_CUDA_OCCAMY_GDN="1", GGML_CUDA_MMQ_XMAD_SM52="1", GGML_CUDA_GDN_COLS="2,4", GGML_CUDA_OCCAMY_FA="1",
           GGML_CUDA_OCCAMY_FUSED="1", GGML_CUDA_MMQ_TAIL_SM52="1")
PROD_CFG = dict(model="/models27/Qwen3.8-27B-Q4_0-head6-requant-SPEEDONLY.gguf", np=1, c=131072, kv="q8_0", unified=False,
                ub=512, b=2048, draftn=None, ctkd=None, mmproj=None, md=None, env={}, extra=[])
MMPROJ = "/m2/qwen38-27b-mmproj/mmproj-F16.gguf"; MTPF = "/m2/qwen38-27b/mtp-Qwen3.8-27B-Q4_0.gguf"

def sh(*a, **k): return subprocess.run(list(a), capture_output=True, text=True, **k)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def post(path, body, port=PORT, timeout=1800):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def get(path, port=PORT): return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=20))
def healthy(port, wait):
    end = time.time() + wait
    while time.time() < end:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200: return True
        except Exception: pass
        if port == PORT and sh("docker", "inspect", "-f", "{{.State.Running}}", NAME).stdout.strip() != "true": return False
        time.sleep(3)
    return False
def vram(): return int(sh("nvidia-smi", f"--id={GPU}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits").stdout.split(",")[0] or 0)

def args_of(c):
    a = ["-m", c["model"], "--alias", "exp", "-ngl", "999", "--fit", "off", "-fa", "on", "--jinja", "--metrics",
         "-np", str(c["np"]), "-c", str(c["c"]), "-ctk", c["kv"], "-ctv", c["kv"], "-b", str(c["b"]), "-ub", str(c["ub"]),
         "--cache-ram", "4096", "--ctx-checkpoints", "8", "--checkpoint-min-step", "2048", "--spec-type", "draft-mtp",
         "--host", "0.0.0.0", "--port", "8080", "--slot-save-path", "/slots", "--no-mmproj"]
    if c["unified"]: a += ["--kv-unified"]
    elif c["np"] > 1: a += ["-no-kvu"]
    if c["draftn"]: a += ["--spec-draft-n-max", str(c["draftn"])]
    if c["ctkd"]: a += ["-ctkd", c["ctkd"], "-ctvd", c["ctkd"]]
    if c["md"]: a += ["-md", c["md"]]
    if c["mmproj"]: a.remove("--no-mmproj"); a += ["--mmproj", c["mmproj"]]
    return a + c["extra"]

def start(c):
    sh("docker", "rm", "-f", NAME); os.makedirs(SLOTS, exist_ok=True)
    env = dict(ENV); env.update(c["env"]); ev = []
    for k, v in env.items():
        if v is not None: ev += ["-e", f"{k}={v}"]
    r = sh("docker", "run", "-d", "--name", NAME, "--network", "ai-net", "--restart", "no", "-p", f"127.0.0.1:{PORT}:8080",
           "--device", f"nvidia.com/gpu={GPU}", "-v", f"{M}/bench27b-20261105/dl:/models27:ro", "-v", "/media/avg-usr/Models/gguf:/m2:ro",
           "-v", f"{SLOTS}:/slots", "-v", (c.get("so") or f"{M}/prod-libs/libggml-cuda-gate-copy-20261013.so") + ":/app/libggml-cuda.so:ro", *ev, IMG, *args_of(c))
    if r.returncode: return False, r.stderr[-300:]
    if healthy(PORT, 240): return True, ""
    tail = sh("docker", "logs", "--tail", "25", NAME).stderr[-1200:]
    return False, tail

PROBES = ["def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
          "Q: A train leaves at 14:35 and the trip takes 2 h 50 min. When does it arrive?\nA: Let's think step by step.",
          "List five failure modes of a lithium-ion battery pack and one mitigation for each:\n1."]
TG = "Write a Python module implementing an LRU cache with TTL, thread safety, type hints and docstrings, then explain it."
def measure(c):
    pk = [0]; stop = [False]
    def samp():
        while not stop[0]: pk[0] = max(pk[0], vram()); time.sleep(1.5)
    threading.Thread(target=samp, daemon=True).start()
    m = {}
    try:
        sl = get("/slots"); m["slots"] = len(sl); m["n_ctx"] = sl[0].get("n_ctx")
        random.seed(11); toks = [random.randint(2000, 150000) for _ in range(4096)]
        t = post("/completion", {"prompt": toks, "n_predict": 1, "cache_prompt": False, "temperature": 0})["timings"]
        m["pp4k"] = round(t["prompt_per_second"], 1)
        if os.environ.get("QUICK_LONG"):
            pr = " ".join("word%d alpha beta gamma delta, the quick brown fox %d." % (i, i * 7) for i in range(1200))
            m["pp23k"] = round(post("/completion", {"prompt": pr, "n_predict": 1, "cache_prompt": False, "temperature": 0})["timings"]["prompt_per_second"], 2)
        tg = []
        for i in range(2):
            t = post("/completion", {"prompt": TG + f" ({i})", "n_predict": 256, "ignore_eos": True, "cache_prompt": False, "temperature": 0})["timings"]
            tg.append(t["predicted_per_second"]); m["accept"] = f'{t.get("draft_n_accepted")}/{t.get("draft_n")}'
        m["tg1"] = round(max(tg), 2)
        if m["slots"] >= 2:
            o = []; t0 = time.time()
            def one(i): o.append(post("/completion", {"prompt": TG + f" [{i}]", "n_predict": 192, "ignore_eos": True, "cache_prompt": False, "temperature": 0})["timings"]["predicted_n"])
            th = [threading.Thread(target=one, args=(i,)) for i in (5, 6)]; [x.start() for x in th]; [x.join() for x in th]
            m["tg2agg"] = round(sum(o) / (time.time() - t0), 2)
        m["probe_outs"] = [post("/completion", {"prompt": p, "n_predict": 96, "cache_prompt": False, "temperature": 0, "seed": 1})["content"] for p in PROBES]
    except Exception as e: m["error"] = repr(e)[:200]
    stop[0] = True; m["vram_peak"] = pk[0]
    return m

def pmatch(a, b):
    if not a or not b: return None
    sc = []
    for x, y in zip(a, b):
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
        sc.append(k / max(1, len(y)))
    return round(sum(sc) / len(sc), 3)

def load(): return json.load(open(ST)) if os.path.exists(ST) else {"base": None, "base_m": None, "tried": [], "notes": {}}
def save(s): json.dump(s, open(ST, "w"), indent=1)
def row(case, c, m, ok, verdict, why=""):
    slim = {k: v for k, v in m.items() if k != "probe_outs"}
    rec = {"t": time.strftime("%m-%d %H:%M"), "case": case, "cfg": {k: v for k, v in c.items() if v not in (None, [], {}, False)}, **slim, "verdict": verdict, "why": why}
    open(LJS, "a").write(json.dumps(rec) + "\n")
    if not os.path.exists(LMD):
        open(LMD, "w").write("# opt27b quick ledger\n\n| time | case | layout | VRAM MiB | pp4k | tg1 | tg2agg | accept | probe | verdict |\n|---|---|---|---|---|---|---|---|---|---|\n")
    lay = f'{c["np"]}x{c["c"]//c["np"] if not c["unified"] else c["c"]}{"u" if c["unified"] else ""} {c["kv"]} ub{c["ub"]}'
    open(LMD, "a").write(f'| {rec["t"]} | {case} | {lay} | {m.get("vram_peak","")} | {m.get("pp4k","")} | {m.get("tg1","")} | {m.get("tg2agg","")} | '
                         f'{m.get("accept","")} | {m.get("probe","")} | {verdict} {why} |\n')
    log(case, verdict, why, json.dumps(slim))

def run_case(s, case, c):
    if time.time() - T0 > CAP_S or time.time() > DEADLINE - 360: log("cap/deadline reached, skip", case); return None
    s["tried"].append(case); save(s)
    ok, err = start(c)
    if not ok:
        m = {"error": "load: " + (err.strip().splitlines() or [""])[-1][:160]}
        row(case, c, m, False, "NOFIT/FAIL"); sh("docker", "rm", "-f", NAME); return None
    m = measure(c); sh("docker", "rm", "-f", NAME)
    if s.get("base_m"): m["probe"] = pmatch(m.get("probe_outs"), s["base_m"].get("probe_outs"))
    return m

# Gate (7 Nov, owner: above noise, but don't throw useful ideas out).
# Noise from B-reps: tg1/pp4k ~0.2%, tg2agg ~0.3%. A metric counts as moved at >= 3 sigma.
TH = {"tg1": 0.006, "pp4k": 0.006, "tg2agg": 0.008, "pp23k": 0.006}
BIG = 0.02  # a gain this size on one metric is worth an owner look even if another metric drops
def gate(m, b):
    """-> (verdict, why). KEEP: some metric >= +3 sigma, none <= -3 sigma. CONFIRM: best gain 1.5-3 sigma, none <= -3 sigma
    (re-run decides). TRADEOFF: some metric >= +2% but another <= -3 sigma (shelved for the owner, not lost). else revert."""
    if "error" in m or "tg1" not in m or "pp4k" not in m: return "error", m.get("error", "missing metrics")
    pm = m.get("probe")
    if pm is not None and pm < 0.95: return "revert", f"probe {pm}"
    d = {k: m[k] / b[k] - 1 for k in TH if m.get(k) and b.get(k)}
    z = {k: v / TH[k] for k, v in d.items()}
    why = " ".join(f"{k} {v:+.1%}" for k, v in d.items())
    best, worst = max(z.values()), min(z.values())
    if worst <= -1 and max(d.values()) >= BIG: return "TRADEOFF", why
    if worst <= -1: return "revert", why
    if best >= 1: return "KEEP", why
    if best >= 0.5: return "CONFIRM", why
    return "revert", why

def better(m, b):  # old rounds: bool view of gate()
    v, why = gate(m, b); return v == "KEEP", why

def mean_m(a, b):
    o = dict(b)
    for k in TH:
        if a.get(k) and b.get(k): o[k] = round((a[k] + b[k]) / 2, 2)
    return o

def shelve(s, key, case, c, why):
    # idea kept with its full config so it can be re-run, stacked or picked as an owner trade-off
    s.setdefault(key, []).append({"case": case, "cfg": c, "why": why, "t": time.strftime("%m-%d %H:%M")})

def lever(s, case, change):
    c = copy.deepcopy(s["base"]); change(c)
    m = run_case(s, case, c)
    if m is None: return
    v, why = gate(m, s["base_m"])
    if v == "CONFIRM":  # ambiguous band: one more rep, decide on the mean
        row(case, c, m, True, "CONFIRM", why)
        m2 = run_case(s, case + "-c", c)
        if m2 and "tg1" in m2:
            m = mean_m(m, m2); v, why = gate(m, s["base_m"]); why = "mean2 " + why
            if v == "CONFIRM": v = "SHELF"  # small but consistent: keep for stacking with other small wins
        else: v = "SHELF"
    if v == "KEEP": s["base"], s["base_m"] = c, m; s["notes"]["kept"] = s["notes"].get("kept", []) + [case]
    elif v == "TRADEOFF": shelve(s, "tradeoffs", case, c, why)
    elif v == "SHELF": shelve(s, "shelf", case, c, why)
    row(case, c, m, True, v, why); save(s)

def round1(s):
    # control + fit layouts. Layout preference: 2x128k > 2x96k > 2x64k; within a size, best tg1.
    c = copy.deepcopy(PROD_CFG); m = run_case(s, "P", c)
    if m: row("P", c, m, True, "control"); s["notes"]["P"] = {k: m.get(k) for k in ("pp4k", "tg1", "vram_peak")}; s["notes"]["P_outs"] = m.get("probe_outs"); save(s)
    lays = [("F5-2x128k-q8", dict(np=2, c=262144, kv="q8_0")), ("F1-2x128k-q4", dict(np=2, c=262144, kv="q4_0")),
            ("F2-256k-unified-q4", dict(np=2, c=262144, kv="q4_0", unified=True)), ("F3-2x96k-q8", dict(np=2, c=196608, kv="q8_0")),
            ("F6-2x64k-q8", dict(np=2, c=131072, kv="q8_0"))]
    fits = []
    for name, ch in lays:
        c = copy.deepcopy(PROD_CFG); c.update(ch); m = run_case(s, name, c)
        if m is None: continue
        if s["notes"].get("P_outs"): m["probe"] = pmatch(m.get("probe_outs"), s["notes"]["P_outs"])
        ok = "error" not in m and m.get("vram_peak", 99999) < 24000
        row(name, c, m, ok, "FITS" if ok else "FAIL");
        if ok: fits.append((c["c"] // 2, m.get("tg1", 0), name, c, m))
        if ok and c["c"] // 2 >= 131072 and len([f for f in fits if f[0] >= 131072]) >= 2: break
    if not fits: s["notes"]["layout"] = "none fits; base stays prod"; s["base"], s["base_m"] = copy.deepcopy(PROD_CFG), m or {}; save(s); return
    fits.sort(key=lambda f: (-f[0], -f[1])); _, _, name, c, m = fits[0]
    s["base"], s["base_m"] = c, m; s["notes"]["layout"] = name; save(s)
    v = copy.deepcopy(c); v["mmproj"] = MMPROJ; mv = run_case(s, "F4-vision-on-" + name, v)
    vf = bool(mv) and "error" not in mv and mv.get("vram_peak", 99999) < 23800
    s["notes"]["vision_fit"] = {"fits": vf, "vram_peak": (mv or {}).get("vram_peak"), "tg1": (mv or {}).get("tg1")}
    row("F4-vision", v, mv or {"error": "load"}, vf, "VISION-FITS" if vf else "VISION-NOFIT"); save(s)

def round2(s):
    for case, ch in [("U1-ub1024", lambda c: c.update(ub=1024)), ("U2-ub2048", lambda c: c.update(ub=2048, b=2048)),
                     ("M2-draft2", lambda c: c.update(draftn=2)), ("M4-draft4", lambda c: c.update(draftn=4)),
                     ("M6-draft6", lambda c: c.update(draftn=6)), ("K1-draftkv-q8", lambda c: c.update(ctkd="q8_0")),
                     ("G0-graphs-off", lambda c: c["env"].update(GGML_CUDA_DISABLE_GRAPHS="1"))]:
        if case not in s["tried"]: lever(s, case, ch)

def round3(s):
    flips = [("E-rms_warp1", "GGML_CUDA_RMS_WARP", "1"), ("E-mmq_compact0", "GGML_CUDA_MMQ_COMPACT_SM52", "0"),
             ("E-mmq_xmad0", "GGML_CUDA_MMQ_XMAD_SM52", "0"), ("E-mmq_tail0", "GGML_CUDA_MMQ_TAIL_SM52", "0"),
             ("E-q8reuse0", "GGML_CUDA_Q8_REUSE", "0"), ("E-normq8_0", "GGML_CUDA_NORM_Q8", "0"),
             ("E-rowpack0", "GGML_CUDA_HEAD_ROWPACK_SM52", "0"), ("E-occfa0", "GGML_CUDA_OCCAMY_FA", "0"),
             ("E-occfused0", "GGML_CUDA_OCCAMY_FUSED", "0"), ("E-force_mmq", "GGML_CUDA_FORCE_MMQ", "1"),
             ("E-force_cublas", "GGML_CUDA_FORCE_CUBLAS", "1")]
    for case, k, v in flips:
        if case not in s["tried"]: lever(s, case, lambda c, k=k, v=v: c["env"].update({k: v}))

def round4(s):
    d = "/m2/qwen38-27b/"
    for q in ["Qwen3.8-27B-UD-IQ4_XS.gguf", "Qwen3.8-27B-UD-Q3_K_XL.gguf"]:
        if not os.path.exists("/media/avg-usr/Models/gguf/qwen38-27b/" + q): log("missing", q); continue
        case = "Q-" + q.split("UD-")[1].split(".")[0]
        c = copy.deepcopy(s["base"]); c["model"] = d + q
        m = run_case(s, case, c)
        if m is None or "error" in m:  # no embedded MTP head -> separate draft file
            c["md"] = MTPF; m = run_case(s, case + "+md", c)
        if m:
            k, why = better(m, s["base_m"])
            # different weights: probe drift expected; record, never auto-keep a quant change
            row(case, c, m, True, "QUANT-CANDIDATE" if k or why.startswith("probe") else "revert", why)
    reps = []
    for i in range(3):
        m = run_case(s, f"B-rep{i}", copy.deepcopy(s["base"]))
        if m and "tg1" in m: reps.append((m["tg1"], m["pp4k"])); row(f"B-rep{i}", s["base"], m, True, "noise")
    s["notes"]["best_reps"] = reps; save(s)

def round5(s):
    # env levers missed by R3: Maxwell graphs/fusion are opt-in on sm_52 (G0 was a no-op)
    for case, ch in [("G1-maxwell-graphs", {"GGML_CUDA_MAXWELL_GRAPHS": "1"}),
                     ("G2-graph-opt", {"GGML_CUDA_MAXWELL_GRAPHS": "1", "GGML_CUDA_GRAPH_OPT": "1"}),
                     ("X1-maxwell-fusion", {"GGML_CUDA_MAXWELL_FUSION": "1"}),
                     ("X2-mmvq-pipe", {"GGML_CUDA_MMVQ_PIPE": "1"}),
                     ("A1-native-fa-sm52", {"GGML_CUDA_NATIVE_FA_SM52": "1"}),
                     ("A2-fa-vec", {"GGML_CUDA_FA_KERNEL": "vec"}),
                     ("A3-fa-waves2", {"GGML_CUDA_OCCAMY_FA_WAVES": "2"}),
                     ("A4-fa-waves4", {"GGML_CUDA_OCCAMY_FA_WAVES": "4"}),
                     ("D1-gdn-dec1", {"GGML_CUDA_GDN_COLS_DECODE": "1"}),
                     ("D4-gdn-dec4", {"GGML_CUDA_GDN_COLS_DECODE": "4"}),
                     ("X0-fusion-off", {"GGML_CUDA_DISABLE_FUSION": "1"})]:
        if case not in s["tried"]: lever(s, case, lambda c, ch=ch: c["env"].update(ch))

def round6(s):
    for case, ch in [("M1-draft1", lambda c: c.update(draftn=1)), ("M3-draft3", lambda c: c.update(draftn=3)),
                     ("S1-pmin0.6", lambda c: c.update(extra=c["extra"] + ["--spec-draft-p-min", "0.6"])),
                     ("S2-pmin0.3", lambda c: c.update(extra=c["extra"] + ["--spec-draft-p-min", "0.3"])),
                     ("U3-ub512", lambda c: c.update(ub=512)), ("U4-ub768", lambda c: c.update(ub=768)),
                     ("U5-ub1536", lambda c: c.update(ub=1536)), ("B1-b1024", lambda c: c.update(b=1024)),
                     ("K2-k8v4", lambda c: c.update(extra=c["extra"] + ["-ctk", "q8_0", "-ctv", "q4_0"]))]:
        if case not in s["tried"]: lever(s, case, ch)
    reps = []
    for i in range(2):
        m = run_case(s, f"B6-rep{i}", copy.deepcopy(s["base"]))
        if m and "tg1" in m: reps.append((m["tg1"], m["pp4k"])); row(f"B6-rep{i}", s["base"], m, True, "noise")
    s["notes"]["best_reps6"] = reps; save(s)

def round7(s):
    # vision re-fit (record only, never changes base) + layout row + runtime knobs
    for case, ch in [("V1-vision-base", lambda c: c.update(mmproj=MMPROJ)),
                     ("V2-vision-ub512", lambda c: c.update(mmproj=MMPROJ, ub=512)),
                     ("V3-vision-2x96k-q4", lambda c: c.update(mmproj=MMPROJ, c=196608)),
                     ("L1-2x96k-q8", lambda c: c.update(c=196608, kv="q8_0"))]:
        if case in s["tried"]: continue
        c = copy.deepcopy(s["base"]); ch(c); m = run_case(s, case, c)
        ok = bool(m) and "error" not in m and m.get("vram_peak", 99999) < 23800
        row(case, c, m or {"error": "load"}, ok, ("FITS" if ok else "NOFIT"), ""); save(s)
    for case, ch in [("C1-ckpt0", lambda c: c.update(extra=c["extra"] + ["--ctx-checkpoints", "0"])),
                     ("C2-cacheram0", lambda c: c.update(extra=c["extra"] + ["--cache-ram", "0"])),
                     ("T1-threads4", lambda c: c.update(extra=c["extra"] + ["-t", "4"])),
                     ("T2-prio2", lambda c: c.update(extra=c["extra"] + ["--prio", "2"]))]:
        if case not in s["tried"]: lever(s, case, ch)
    reps = []
    for i in range(3):
        m = run_case(s, f"B7-rep{i}", copy.deepcopy(s["base"]))
        if m and "tg1" in m: reps.append((m["tg1"], m["pp4k"])); row(f"B7-rep{i}", s["base"], m, True, "noise")
    s["notes"]["best_reps7"] = reps; save(s)

def round8(s):
    # stack near-misses under the 1% keep bar (noise ~0.2%)
    for case, ch in [("G1b-maxwell-graphs", {"GGML_CUDA_MAXWELL_GRAPHS": "1"}),
                     ("A2b-fa-vec", {"GGML_CUDA_FA_KERNEL": "vec"}),
                     ("X2b-mmvq-pipe", {"GGML_CUDA_MMVQ_PIPE": "1"}),
                     ("X1b-maxwell-fusion", {"GGML_CUDA_MAXWELL_FUSION": "1"})]:
        if case not in s["tried"]: lever(s, case, lambda c, ch=ch: c["env"].update(ch))
    reps = []
    for i in range(3):
        m = run_case(s, f"B8-rep{i}", copy.deepcopy(s["base"]))
        if m and "tg1" in m: reps.append((m["tg1"], m["pp4k"])); row(f"B8-rep{i}", s["base"], m, True, "noise")
    s["notes"]["best_reps8"] = reps; save(s)

def round9(s):
    # sm52 GENERIC mmvq nwarps/rpb sweep (build_b11552_mmvqtune.sh); base variant = parity check vs prod .so
    L = f"{M}/mmvqtune-libs/libggml-cuda-mmvqT-"
    for v in ["base", "w2", "w4", "r1", "w4r1", "up"]:
        case = "K-mmvq-" + v
        if case in s["tried"] or not os.path.exists(L + v + ".so"): continue
        lever(s, case, lambda c, v=v: c.update(so=L + v + ".so"))
    reps = []
    for i in range(2):
        m = run_case(s, f"B9-rep{i}", copy.deepcopy(s["base"]))
        if m and "tg1" in m: reps.append((m["tg1"], m["pp4k"])); row(f"B9-rep{i}", s["base"], m, True, "noise")
    s["notes"]["best_reps9"] = reps; save(s)

def step_ms(m):
    # accept-neutral speed: wall ms per MTP verify step (256 tokens, steps = 256 - accepted)
    try: acc = int(str(m["accept"]).split("/")[0]); return round(256 / m["tg1"] * 1000 / (256 - acc), 1)
    except Exception: return None

def round10(s):
    # smaller head quant + ncols-specific mmvq nwarps; judged on step_ms (MTP accept luck removed), recorded as candidates
    log("R10 base step_ms", step_ms(s["base_m"]))
    L = f"{M}/mmvqtune-libs/libggml-cuda-mmvqT-"
    cases = [("Q-head4", lambda c: c.update(model="/m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf"))]
    cases += [("K-mmvq-" + v, (lambda c, v=v: c.update(so=L + v + ".so"))) for v in ["n3w4", "n58w4", "n24w4"] if os.path.exists(L + v + ".so")]
    for case, ch in cases:
        if case in s["tried"]: continue
        c = copy.deepcopy(s["base"]); ch(c); m = run_case(s, case, c)
        if m is None: continue
        b = step_ms(s["base_m"]); x = step_ms(m)
        why = f"step_ms {x} vs {b} ({(x/b-1)*100:+.1f}%)" if (x and b) else "step_ms n/a"
        row(case, c, m, True, "CANDIDATE" if (x and b and x < b * 0.99) else "revert", why); s["notes"].setdefault("r10", {})[case] = [x, b]; save(s)
    m = run_case(s, "B10-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B10-rep0", s["base"], m, True, "noise", f"step_ms {step_ms(m)}")
    save(s)

def round11(s):
    # mmvq rows-per-block (activation-traffic fix): screen showed p4n1 -17% kernel time at n=1..3, numerics unchanged
    L = f"{M}/mmvqtune-libs/libggml-cuda-mmvqT-"
    for v in ["p4n1", "p4", "p4n4"]:
        case = "K-mmvq-" + v
        if case in s["tried"] or not os.path.exists(L + v + ".so"): continue
        lever(s, case, lambda c, v=v: c.update(so=L + v + ".so"))
        log(case, "step_ms", step_ms(s["base_m"]))
    m = run_case(s, "B11-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B11-rep0", s["base"], m, True, "noise", f"step_ms {step_ms(m)}")
    save(s)

def round12(s):
    # after p4n1: verify step is cheaper -> retry deeper draft; head4 combo judged on step_ms (candidate only)
    for case, ch in [("M3b-draft3-p4n1", lambda c: c.update(draftn=3)), ("M4b-draft4-p4n1", lambda c: c.update(draftn=4))]:
        if case in s["tried"]: continue
        lever(s, case, ch); log(case, "base step_ms", step_ms(s["base_m"]))
    case = "Q-head4-p4n1"
    if case not in s["tried"]:
        c = copy.deepcopy(s["base"]); c.update(model="/m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf"); m = run_case(s, case, c)
        if m:
            b = step_ms(s["base_m"]); x = step_ms(m)
            why = f"step_ms {x} vs {b} ({(x/b-1)*100:+.1f}%)" if (x and b) else "step_ms n/a"
            row(case, c, m, True, "CANDIDATE" if (x and b and x < b * 0.99) else "revert", why); s["notes"].setdefault("r12", {})[case] = [x, b]
    m = run_case(s, "B12-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B12-rep0", s["base"], m, True, "noise", f"step_ms {step_ms(m)}")
    save(s)

def round13(s):
    # adopt head4 on KLD evidence (PPL +0.07%, mean KLD 0.0022, same-top 98.6%) + step_ms -2.1%
    case = "Q-head4-adopt"
    if case not in s["tried"]:
        c = copy.deepcopy(s["base"]); c.update(model="/m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf"); m = run_case(s, case, c)
        if m and "tg1" in m:
            b = step_ms(s["base_m"]); x = step_ms(m)
            s["base"], s["base_m"] = c, m; s["notes"]["kept"] = s["notes"].get("kept", []) + ["Q-head4-kld"]
            row(case, c, m, True, "KEEP", f"KLD 0.0022 same-top 98.6%; step_ms {x} vs {b}"); save(s)
    for r in ("B13-rep0", "B13-rep1"):
        m = run_case(s, r, copy.deepcopy(s["base"]))
        if m and "tg1" in m: row(r, s["base"], m, True, "noise", f"step_ms {step_ms(m)}")
    save(s)

def round14(s):
    # draft1 on head4+p4n1 base: n=2 kernel 166us vs n=3 217us; judged on tg1 & tg2agg
    case = "M1b-draft1-head4"
    if case not in s["tried"]:
        c = copy.deepcopy(s["base"]); c.update(draftn=1); m = run_case(s, case, c)
        if m and "tg1" in m:
            b = s["base_m"]; why = f"tg1 {m['tg1']} vs {b['tg1']}; tg2agg {m.get('tg2agg')} vs {b.get('tg2agg')}; step_ms {step_ms(m)}"
            row(case, c, m, True, "CANDIDATE" if m["tg1"] >= b["tg1"]*1.03 or m.get("tg2agg",0) >= b.get("tg2agg",0)*1.03 else "revert", why)
    m = run_case(s, "B14-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B14-rep0", s["base"], m, True, "noise", f"step_ms {step_ms(m)}")
    save(s)

def round15(s):
    # FP32-FFMA q4_0 multi-col matvec (GGML_CUDA_FP32MV, n=2..4): op n=3 194 vs 214 us, n=4 234 vs 273
    L = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fp32mv.so"
    def fx(c): c.update(so=L); c["env"] = dict(c.get("env") or {}, GGML_CUDA_FP32MV="1")
    for case, ch in (("K-fp32mv", fx), ("K-fp32mv-draft3", lambda c: (fx(c), c.update(draftn=3)))):
        c = copy.deepcopy(s["base"]); ch(c); m = run_case(s, case, c)
        if m and "tg1" in m:
            b = s["base_m"]; row(case, c, m, True, "CANDIDATE" if m["tg1"] >= b["tg1"]*1.03 else "revert",
                                 f"tg1 {m['tg1']} vs {b['tg1']}; tg2agg {m.get('tg2agg')} vs {b.get('tg2agg')}; step_ms {step_ms(m)} vs {step_ms(b)}")
    m = run_case(s, "B15-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B15-rep0", s["base"], m, True, "noise", f"step_ms {step_ms(m)}")
    save(s)

def round16(s):
    # pp: transposed q4_0 dequant + cuBLAS NN algo5 (GGML_CUDA_SGEMM_NN): ubench 6.1-6.3 TF vs 4.7-5.4 TN; op n512 13.1 vs 15.2 ms
    L = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-sgemmnn.so"
    lever(s, "P-sgemmnn", lambda c: (c.update(so=L), c["env"].update(GGML_CUDA_SGEMM_NN="1")))
    m = run_case(s, "B16-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B16-rep0", s["base"], m, True, "noise", f"pp4k {m.get('pp4k')} step_ms {step_ms(m)}")
    save(s)

def round17(s):
    # adopt sgemmnn on KLD evidence (mean KLD 0.000000, max 5e-5, same-top 100%, PPL ratio 1.00005); r16 pp4k 87.0 vs 79.0
    L = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-sgemmnn.so"
    case = "P-sgemmnn-adopt"
    if case not in s["tried"]:
        c = copy.deepcopy(s["base"]); c.update(so=L); c["env"].update(GGML_CUDA_SGEMM_NN="1"); m = run_case(s, case, c)
        if m and "tg1" in m:
            b = s["base_m"]
            s["base"], s["base_m"] = c, m; s["notes"]["kept"] = s["notes"].get("kept", []) + ["P-sgemmnn-kld"]
            row(case, c, m, True, "KEEP", f"KLD ~0 same-top 100%; pp4k {m.get('pp4k')} vs {b.get('pp4k')}; tg1 {m['tg1']} vs {b['tg1']}"); save(s)
    m = run_case(s, "B17-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B17-rep0", s["base"], m, True, "noise", f"pp4k {m.get('pp4k')} step_ms {step_ms(m)}")
    save(s)

def round18(s):
    # tiled transposed dequant k_dequant_q4_0_T2: ubench 1.4-1.9x vs T (184 vs 94-130 GB/s), KLD identical to sgemmnn
    L = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-sgemmnn2.so"
    lever(s, "P-dqT2", lambda c: c.update(so=L))
    m = run_case(s, "B18-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B18-rep0", s["base"], m, True, "noise", f"pp4k {m.get('pp4k')} step_ms {step_ms(m)}")
    save(s)

def round19(s):
    # overlap dequant piece p+1 (aux stream) with cuBLAS NN GEMM piece p; KLD identical
    L = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-nnsplit.so"
    for n in ("4", "2"):
        lever(s, f"P-nnsplit{n}", lambda c, n=n: (c.update(so=L), c["env"].update(GGML_CUDA_SGEMM_NN_SPLIT=n)))
    m = run_case(s, "B19-rep0", copy.deepcopy(s["base"]))
    if m and "tg1" in m: row("B19-rep0", s["base"], m, True, "noise", f"pp4k {m.get('pp4k')} step_ms {step_ms(m)}")
    save(s)

def round20(s):
    # re-test probe-rejected tg candidate on current base; judged as owner trade-off (ckpt0 disables hybrid prompt-cache reuse)
    c = copy.deepcopy(s["base"]); c.update(extra=c["extra"] + ["--ctx-checkpoints", "0"]); m = run_case(s, "C1b-ckpt0-now", c)
    if m and "tg1" in m:
        b = s["base_m"]; row("C1b-ckpt0-now", c, m, True, "TRADEOFF", f"tg1 {m['tg1']} vs {b['tg1']}; tg2agg {m.get('tg2agg')} vs {b.get('tg2agg')}; pp4k {m.get('pp4k')} vs {b.get('pp4k')}")
    save(s)

def round21(s):
    # GEMM prefill FA (fattn-gemm.cu): KLD 0 (max 5.3e-5, top 100%), tbo 3993/3993; pp 22.7k/51.9k/100.2k +7/+15/+24%
    # decode/draft untouched (nq<64); adopt if pp4k/tg1 not worse than -3%
    L = "/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fagemm.so"; case = "P-fagemm-adopt"
    c = copy.deepcopy(s["base"]); c.update(so=L); c["env"].update(GGML_CUDA_FA_GEMM="1"); m = run_case(s, case, c)
    if m and "tg1" in m:
        b = s["base_m"]; ok = m["tg1"] >= 0.97*b["tg1"] and m["pp4k"] >= 0.97*b["pp4k"]
        if ok: s["base"], s["base_m"] = c, m; s["notes"]["kept"] = s["notes"].get("kept", []) + ["P-fagemm"]
        row(case, c, m, True, "KEEP" if ok else "revert", f"long-ctx pp +7/+15/+24% @22.7k/51.9k/100.2k; pp4k {m.get('pp4k')} vs {b.get('pp4k')}; tg1 {m['tg1']} vs {b['tg1']}"); save(s)

def round22(s):
    # 7 Nov: apps clocks 987 -> 1152 MHz (owner sudo; guard reverts at 78C). Same C3 config; A = R21 row @987.
    b = s["base_m"]; ms = []
    for i in range(2):
        m = run_case(s, f"CLK1152-rep{i}", copy.deepcopy(s["base"]))
        if not m or "tg1" not in m: continue
        k, why = better(m, b); ms.append((m, k))
        row(f"CLK1152-rep{i}", s["base"], m, True, "KEEP" if k else "revert", why + f" vs 987MHz pp4k {b['pp4k']} tg1 {b['tg1']} tg2agg {b.get('tg2agg')}")
    if ms and all(k for _, k in ms): s["base_m"] = ms[-1][0]; s["notes"]["kept"] = s["notes"].get("kept", []) + ["CLK1152"]; s["notes"]["clk"] = "3305,1152"
    save(s)

def round23(s):
    # rebase: base unchanged, base_m = mean of 2 reps (run with QUICK_LONG=1 to add pp23k to the base)
    ms = [m for m in (run_case(s, f"BASE-rep{i}", copy.deepcopy(s["base"])) for i in range(2)) if m and "tg1" in m]
    if len(ms) == 2:
        nb = mean_m(ms[0], ms[1]); row("BASE-rebase", s["base"], nb, True, "rebase", f"old pp4k {s['base_m'].get('pp4k')} tg1 {s['base_m'].get('tg1')}")
        s["base_m"] = nb
    save(s)

def restore_prod():
    sh("docker", "rm", "-f", NAME); sh("docker", "start", PROD)
    ok = healthy(PROD_PORT, 240); log("prod healthy" if ok else "PROD NOT HEALTHY")
    log(sh("bash", "-lc", "labslot check 2>&1 | tail -15").stdout)
    return ok

if __name__ == "__main__":
    rnd = int(sys.argv[1]); s = load()
    if rnd > 1 and not s.get("base"): log("no base; run round 1 first"); sys.exit(3)
    log(f"round {rnd} start; df:", sh("df", "-h", "/media/avg-usr/Storage").stdout.splitlines()[-1])
    sh("docker", "stop", "-t", "60", "llamacpp-m6000"); sh("docker", "stop", "-t", "60", PROD)
    rc = 0
    try:
        {1: round1, 2: round2, 3: round3, 4: round4, 5: round5, 6: round6, 7: round7, 8: round8, 9: round9, 10: round10, 11: round11, 12: round12, 13: round13, 14: round14, 15: round15, 16: round16, 17: round17, 18: round18, 19: round19, 20: round20, 21: round21, 22: round22, 23: round23}[rnd](s)
    except Exception as e: log("ROUND ERROR", repr(e)); rc = 2
    finally:
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NO_RESTORE")):
            sh("docker", "rm", "-f", NAME); log("NO_RESTORE: prod left stopped")
        elif not restore_prod(): rc = 5
    log(f"round {rnd} done in {round((time.time()-T0)/60)} min; base layout {s['notes'].get('layout')}; kept {s['notes'].get('kept')}")
    sys.exit(rc)
