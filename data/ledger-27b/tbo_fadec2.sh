#!/bin/bash
# 8 Nov fadec2: test-backend-ops FLASH_ATTN_EXT, GGML_CUDA_FA_DEC=0 vs 2 (+DEBUG counter, MINQ=1, FADEC2_CASES extra q4_0 G=4/6/8 hsk256 cases). Run under flock gpu-bench.lock.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; OUT=$O/tbo-fadec2-20261108.txt
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
trap "docker rm -f fadec2-tbo >/dev/null 2>&1" EXIT
for cfg in "0 3 0" "2 3 0" "2 1 0" "0 3 1" "2 3 1" "2 1 1"; do set -- $cfg; echo "== FA_DEC=$1 MINQ=$2 FADEC2_CASES=$3 $(date +%T)" | tee -a $OUT
X=""; [ $3 = 1 ] && X="-e FADEC2_CASES=1"
docker run --rm --name fadec2-tbo --device nvidia.com/gpu=$GPU -v $M/src-b11552-fadec-20261108/build/bin:/b:ro -e LD_LIBRARY_PATH=/b \
  -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_DEC=$1 -e GGML_CUDA_FA_DEC_MINQ=$2 -e GGML_CUDA_FA_DEC_DEBUG=1 $X --entrypoint /b/test-backend-ops $IMG test -o FLASH_ATTN_EXT -b CUDA0 2>&1 \
  | grep -E "FAIL|passed|fa_dec2 total|fa_dec2 call=[1-3] " | head -15 | tee -a $OUT
done
