#!/bin/bash
J=/home/avg-usr/.claude/jobs/fc533454/tmp/prof35; S=/media/avg-usr/Storage/m6000-llama-cpp
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=prof35-nvprof
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1"
M=/models/occamy-1.0-Q4_0-headq40b.gguf
for d in 0 16384; do
 docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $J/app:/app:ro -v $S/prod-libs/libggml-cuda-xmad35-20261110.so:/app/libggml-cuda.so:ro -v $J/out:/out -e LD_LIBRARY_PATH=/app $E \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --print-gpu-trace --log-file /out/d$d.csv \
  /app/llama bench -m $M -ngl 999 -fa on -b 4096 -ub 4096 -p 0 -n 6 -d $d -r 1 > $J/out/d$d.bench.txt 2>&1
 echo "d$d rc=$?"
done
