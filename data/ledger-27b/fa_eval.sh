#!/bin/bash
# 7 Nov: GEMM-FA eval. usage: flock ../gpu-bench.lock ./fa_eval.sh <so> [reps...]   (stops prod; trap restores it)
set -u; M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; SO=$1; shift
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
W=/media/avg-usr/Models/kld-head4-20261107; T=$(date +%H%M); OUT=$O/fa-gemm-eval-20261107-$T.txt
restore() { docker rm -f llamacpp-m6000-27b-exp kld-27b fag-tbo >/dev/null 2>&1; docker start llamacpp-m6000-27b >/dev/null
  for i in $(seq 80); do curl -sf 127.0.0.1:11437/health >/dev/null && { echo "prod-ok" | tee -a $OUT; break; }; sleep 3; done; labslot check 2>&1 | tail -3; }
[ "${RESTORE:-0}" = 1 ] && trap restore EXIT || trap "docker rm -f llamacpp-m6000-27b-exp kld-27b fag-tbo >/dev/null 2>&1" EXIT
docker stop -t 60 llamacpp-m6000-27b >/dev/null
echo "== test-backend-ops FLASH_ATTN_EXT (FA_GEMM=1 MINQ=1)" | tee -a $OUT
docker run --rm --name fag-tbo --device nvidia.com/gpu=$GPU -v $M/src-b11552-mmvqtune-20261107/build/bin:/b:ro -e LD_LIBRARY_PATH=/b \
  -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_GEMM_MINQ=1 --entrypoint /b/test-backend-ops $IMG test -o FLASH_ATTN_EXT -b CUDA0 2>&1 \
  | grep -E "hsk=256" | grep -cE "OK" | sed 's/^/hsk256 OK: /' | tee -a $OUT
docker run --rm --name fag-tbo --device nvidia.com/gpu=$GPU -v $M/src-b11552-mmvqtune-20261107/build/bin:/b:ro -e LD_LIBRARY_PATH=/b \
  -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_GEMM_MINQ=1 --entrypoint /b/test-backend-ops $IMG test -o FLASH_ATTN_EXT -b CUDA0 2>&1 \
  | grep -E "FAIL|passed|Backend CUDA0:" | head -15 | tee -a $OUT
echo "== KLD vs head4 base (FA_GEMM=1, c=512)" | tee -a $OUT
docker run --rm --name kld-27b --device nvidia.com/gpu=$GPU -v /media/avg-usr/Models/gguf:/m2:ro -v $W:/w -v $SO:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 --entrypoint /app/llama local/llama.cpp:server-cuda-b11552-sm52-normq8 \
  perplexity -ngl 99 -c 512 --chunks 16 -f /w/corpus.txt -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --kl-divergence-base /w/head4.kld --kl-divergence 2>&1 \
  | grep -E 'Mean PPL\(Q\) |Mean    KLD|Maximum KLD|Same top|error|seconds per pass' | tee -a $OUT
echo "== server pp A/B" | tee -a $OUT
python3 $O/fa_eval.py $SO "$@" 2>&1 | tee -a $OUT
