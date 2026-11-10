#!/bin/bash
# 8 Nov card F2: one KLD b3 run with per-token shim. Args: TAG VAR=VALUE ...  Env CHUNKS(16) BASE(/w/head4.kld) SO. Call under flock gpu-bench.lock.
TAG=$1; shift
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; X=/media/avg-usr/Models/kld-f2-20261108
SO=${SO:-$M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2dbg.so}
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=local/llama.cpp:server-cuda-b11552-sm52-normq8; C=kld-27b-f2
E=(); for a in "$@"; do E+=(-e "$a"); done
BASEF=${BASE:-/w/head4.kld}; CH=${CHUNKS:-16}; OUT=$X/summary.txt
trap "docker rm -f $C >/dev/null 2>&1; [ -n \"\${W:-}\" ] && kill \$W 2>/dev/null" EXIT
if [ -n "$(docker ps -q --filter name=^/$C\$)" ]; then echo "container busy"; exit 1; fi
nvidia-smi -i $GPU --query-gpu=temperature.gpu --format=csv,noheader -l 15 > >(awk -v f=$X/$TAG-temp.txt -v c=$C '{print > f; fflush(f); if ($1+0>=83) {system("docker stop -t 2 " c); print "TEMP ABORT" > f; fflush(f)}}') & W=$!
if [ -n "${KRUNMAX:-}" ]; then (sleep $KRUNMAX >/dev/null 2>&1; docker stop -t 2 $C >/dev/null 2>&1) >/dev/null 2>&1 & K=$!; fi
S=$(date +%s); echo "== $TAG $* chunks=$CH base=$BASEF $(date +%T)" | tee -a $OUT
docker run --rm --name $C --device nvidia.com/gpu=$GPU -v /media/avg-usr/Models/gguf:/m2:ro -v /media/avg-usr/Models/kld-head4-20261107:/w:ro -v $X:/x -v $SO:/app/libggml-cuda.so:ro -v $O/f2/kldshim.so:/kldshim.so:ro \
  -e LD_PRELOAD=/kldshim.so -e KLDSHIM_TAG=$TAG -e KLDSHIM_BASE=$BASEF -e GGML_CUDA_FA_DBG_TAG=$TAG -e GGML_CUDA_FA_DBG_DIR=/x -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 "${E[@]}" --entrypoint /app/llama $IMG \
  perplexity -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -c 512 -b 3 -ub 3 --chunks $CH -f /w/corpus.txt -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --kl-divergence-base $BASEF --kl-divergence > $X/$TAG.log 2>&1
rc=$?
[ -n "${K:-}" ] && { pkill -P $K sleep 2>/dev/null; kill $K 2>/dev/null; }
grep -E 'Mean    KLD|Maximum KLD|Same top|error|fatal|kldshim chunk=15|kldshim chunk=63' $X/$TAG.log | cut -c1-120 | tee -a $OUT
echo "rc=$rc secs=$(( $(date +%s)-S )) maxtemp=$(sort -n $X/$TAG-temp.txt | tail -1)" | tee -a $OUT
