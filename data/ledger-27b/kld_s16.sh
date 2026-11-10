#!/bin/bash
# 7 Nov: KLD gate for FA-GEMM f16 S scratch. usage: flock ../gpu-bench.lock ./kld_s16.sh   (prod stays stopped)
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; SO=$M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-s16.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; W=/media/avg-usr/Models/kld-head4-20261107; OUT=$O/kld-s16-20261107.txt
trap "docker rm -f kld-27b >/dev/null 2>&1" EXIT
echo "== KLD vs head4 base (FA_GEMM=1 S16=1, c=512)" | tee -a $OUT
docker run --rm --name kld-27b --device nvidia.com/gpu=$GPU -v /media/avg-usr/Models/gguf:/m2:ro -v $W:/w -v $SO:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_GEMM_S16=1 --entrypoint /app/llama local/llama.cpp:server-cuda-b11552-sm52-normq8 \
  perplexity -ngl 99 -c 512 --chunks 16 -f /w/corpus.txt -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --kl-divergence-base /w/head4.kld --kl-divergence 2>&1 \
  | grep -E 'Mean PPL\(Q\) |Mean    KLD|Maximum KLD|Same top|error|seconds per pass' | tee -a $OUT
