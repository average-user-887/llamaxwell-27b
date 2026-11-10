#!/bin/bash
# ppbench.sh <model:27|35> <tag> <so> <pp-list> <reps> [extra env K=V ...]
# llama bench (pp only) in a throwaway container on the M6000. Run under flock gpu-bench.lock.
set -u
M=$1; TAG=$2; SO=$3; PL=$4; REPS=$5; shift 5
J=/home/avg-usr/.claude/jobs/fc533454/tmp/pp; mkdir -p $J/out
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
ENV="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1"
if [ "$M" = 27 ]; then
  ENV="$ENV -e GGML_CUDA_FA_DEC=2 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1"
  MV="-v /media/avg-usr/Models/gguf:/m:ro"; MF=/m/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf; ARGS="-ub ${UB:-1024} -ctk q4_0 -ctv q4_0"
else
  MV="-v /media/avg-usr/Storage/uroboros-models:/m:ro"; MF=/m/occamy-1.0-Q4_0-headq40b.gguf; ARGS="-ub ${UB:-512}"
fi
for kv in "$@"; do ENV="$ENV -e $kv"; done
PRE=""; [ -n "${PROF:-}" ] && PRE="--entrypoint /prof/cuda/bin/nvprof"
C=ppprobe-bench; docker rm -f $C >/dev/null 2>&1
if [ -n "${PROF:-}" ]; then
  docker run --rm --name $C --device nvidia.com/gpu=$GPU $MV -v /media/avg-usr/Models/prof-27b-20261107/cuda:/prof/cuda:ro -v $J/out:/out \
    -v $SO:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/prof/cuda/lib:/app $ENV $PRE $IMG \
    --print-gpu-summary --log-file /out/$TAG.nvprof.txt /app/llama bench -m $MF -p $PL -n 0 -b 2048 $ARGS -ngl 999 -fa 1 -r $REPS --no-warmup -o jsonl \
    > $J/out/$TAG.jsonl 2> $J/out/$TAG.err
else
  docker run --rm --name $C --device nvidia.com/gpu=$GPU $MV -v $SO:/app/libggml-cuda.so:ro $ENV --entrypoint /app/llama $IMG \
    bench -m $MF -p $PL -n 0 -b 2048 $ARGS -ngl 999 -fa 1 -r $REPS -o jsonl > $J/out/$TAG.jsonl 2> $J/out/$TAG.err
fi
echo "rc=$? $TAG"; python3 -I -c "
import json,sys
for l in open('$J/out/$TAG.jsonl'):
    d=json.loads(l); print('$TAG', d['n_prompt'], d['n_batch'], d['n_ubatch'], round(d['avg_ts'],2), round(d['stddev_ts'],2))
"
nvidia-smi -i $GPU --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader
