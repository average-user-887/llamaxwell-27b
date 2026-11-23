#!/bin/bash
# card6: nvprof GPU trace of decode: one single-slot request, then two concurrent (tg2agg pattern). usage: prof_dec.sh <tag> <so> [KEY=VAL env...]
# run under: flock /media/$USER/Storage/m6000-llama-cpp/gpu-bench.lock
set -u; P=/media/$USER/Models/prof-27b-20261107; TAG=$1; SO=$2; shift 2
GU=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep -i m6000 | cut -d, -f1 | head -1)
EV=""; for kv in GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 \
  GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 \
  GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_FA_DEC=2 GGML_CUDA_SGEMM_NN=1 GGML_CUDA_FA_GEMM=1 "$@"; do EV="$EV -e $kv"; done
N=llamacpp-m6000-27b-prof; docker rm -f $N >/dev/null 2>&1
docker run -d --name $N --network ai-net --restart no -p 127.0.0.1:11449:8080 --device nvidia.com/gpu=$GU \
  -v /media/$USER/Models/gguf:/m2:ro -v $P:/prof -v $SO:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/prof/cuda/lib:/app $EV \
  --entrypoint /prof/cuda/bin/nvprof sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d \
  --print-gpu-trace --csv --normalized-time-unit us --log-file /prof/$TAG.trace.csv \
  /app/llama-server -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --alias exp -ngl 999 --fit off -fa on --jinja -np 2 -no-kvu -c 262144 \
  -ctk q4_0 -ctv q4_0 -b 2048 -ub 1024 --cache-ram 4096 --ctx-checkpoints 8 --checkpoint-min-step 2048 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080 --no-mmproj >/dev/null
for i in $(seq 120); do curl -sf 127.0.0.1:11449/health >/dev/null && break; sleep 3; done
python3 -I - <<'EOF' | tee $P/$TAG.timings.txt
import json, threading, time, urllib.request
TG = "Write a Python module implementing an LRU cache with TTL, thread safety, type hints and docstrings, then explain it."
def post(b):
    r = urllib.request.Request("http://127.0.0.1:11449/completion", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))["timings"]
def req(i, n): return post({"prompt": TG + f" [{i}]", "n_predict": n, "ignore_eos": True, "cache_prompt": False, "temperature": 0})
time.sleep(2); t0 = time.time(); t = req(5, 128); print("single", round(time.time() - t0, 3), t["predicted_n"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n"))
time.sleep(3); o = []; t0 = time.time()
th = [threading.Thread(target=lambda i=i: o.append(req(i, 128))) for i in (5, 6)]; [x.start() for x in th]; [x.join() for x in th]
w = time.time() - t0; print("LABEXP tg2agg_prof=%.2f t/s" % (sum(x["predicted_n"] for x in o) / w)); print("pair", round(w, 3), [x["predicted_n"] for x in o], round(sum(x["predicted_n"] for x in o) / w, 2), [(x.get("draft_n_accepted"), x.get("draft_n")) for x in o])
time.sleep(2)
EOF
docker exec $N sh -c 'for d in /proc/[0-9]*; do grep -q llama-server $d/cmdline 2>/dev/null && [ "${d#/proc/}" != 1 ] && kill -INT ${d#/proc/}; done'
for i in $(seq 900); do [ "$(docker inspect -f '{{.State.Running}}' $N)" = false ] && break; sleep 2; done
docker rm -f $N >/dev/null; ls -la $P/$TAG.trace.csv
