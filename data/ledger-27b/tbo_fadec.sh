#!/bin/bash
# 8 Nov fadec: test-backend-ops FLASH_ATTN_EXT with GGML_CUDA_FA_DEC=0/1 (fadec tree build/bin). Run under flock gpu-bench.lock.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; OUT=$O/tbo-fadec-20261108.txt
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
trap "docker rm -f fadec-tbo >/dev/null 2>&1" EXIT
for cfg in "0 0" "1 0" "1 1"; do set -- $cfg; echo "== FA_DEC=$1 GEMM_MINQ1=$2" | tee -a $OUT
X=""; [ $2 = 1 ] && X="-e GGML_CUDA_FA_GEMM_MINQ=1"
docker run --rm --name fadec-tbo --device nvidia.com/gpu=$GPU -v $M/src-b11552-fadec-20261108/build/bin:/b:ro -e LD_LIBRARY_PATH=/b \
  -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_DEC=$1 $X --entrypoint /b/test-backend-ops $IMG test -o FLASH_ATTN_EXT -b CUDA0 2>&1 \
  | grep -E "FAIL|passed|Backend CUDA0:" | head -15 | tee -a $OUT
done
