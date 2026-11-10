#!/bin/bash
# 7 Nov card C: KLD vs head4 base with GGML_CUDA_MMVQ_SM52=1 (note: ppl batches n=512 use MMQ, so this only guards against collateral damage). Run under flock gpu-bench.lock. Does not touch prod.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; SO=$M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-sm52.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; W=/media/avg-usr/Models/kld-head4-20261107; OUT=$O/kld-sm52-20261107.txt
trap "docker rm -f kld-sm52 >/dev/null 2>&1" EXIT
echo "== KLD vs head4 base (SM52=1, c=512)" | tee -a $OUT
docker run --rm --name kld-sm52 --device nvidia.com/gpu=$GPU -v /media/avg-usr/Models/gguf:/m2:ro -v $W:/w -v $SO:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_MMVQ_SM52=1 --entrypoint /app/llama local/llama.cpp:server-cuda-b11552-sm52-normq8 \
  perplexity -ngl 99 -c 512 --chunks 16 -f /w/corpus.txt -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --kl-divergence-base /w/head4.kld --kl-divergence 2>&1 \
  | grep -E 'Mean PPL\(Q\) |Mean    KLD|Maximum KLD|Same top|error|seconds per pass' | tee -a $OUT
