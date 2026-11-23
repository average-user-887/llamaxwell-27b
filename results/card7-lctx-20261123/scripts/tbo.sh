#!/bin/bash
# Card 7: test-backend-ops FLASH_ATTN_EXT with FA_GEMM=1, SMAX1 off/on, MINQ 1 (force FA_GEMM on small cases) and default. Under gpu-bench.lock via wrap.sh.
B=$WT/build-c7/bin; OUT=$M/opt27b-20261113/card7-lctx/tbo.txt
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
for cfg in "0 1" "1 1" "1 64"; do set -- $cfg; echo "== FA_GEMM=1 SMAX1=$1 MINQ=$2 $(date +%T)" | tee -a $OUT
docker run --rm --name c7-tbo --device nvidia.com/gpu=$GU -v $B:/b:ro -e LD_LIBRARY_PATH=/b -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_GEMM_SMAX1=$1 -e GGML_CUDA_FA_GEMM_MINQ=$2 \
  -e GGML_CUDA_FA_DEC=2 --entrypoint /b/test-backend-ops $IMG test -o FLASH_ATTN_EXT -b CUDA0 2>&1 | grep -E "FAIL|passed" | tail -6 | tee -a $OUT
done
