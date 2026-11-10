#!/bin/bash
# 8 Nov F2: extended base, same command as head4 base was made (assumed; checked afterwards by comparing the first 16 chunks with head4.kld).
M=/media/avg-usr/Storage/m6000-llama-cpp; X=/media/avg-usr/Models/kld-f2-20261108; GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
trap "docker rm -f kld-27b-f2gen >/dev/null 2>&1; [ -n \"\${W:-}\" ] && kill \$W" EXIT
nvidia-smi -i $GPU --query-gpu=temperature.gpu --format=csv,noheader -l 15 > >(awk -v f=$X/gen64-temp.txt '{print > f; fflush(f); if ($1+0>=83) {system("docker stop -t 2 kld-27b-f2gen"); print "TEMP ABORT" > f}}') & W=$!
docker run --rm --name kld-27b-f2gen --device nvidia.com/gpu=$GPU -v /media/avg-usr/Models/gguf:/m2:ro -v /media/avg-usr/Models/kld-head4-20261107:/w:ro -v $X:/x -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_SGEMM_NN=1 --entrypoint /app/llama local/llama.cpp:server-cuda-b11552-sm52-normq8 perplexity -ngl 99 -c 512 --chunks 64 -f /w/corpus.txt -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --kl-divergence-base /x/head4-64.kld > $X/gen64.log 2>&1
echo rc=$?; tail -3 $X/gen64.log | cut -c1-120; ls -la $X/head4-64.kld
