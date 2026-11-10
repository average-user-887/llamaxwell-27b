#!/bin/bash
M=/media/avg-usr/Storage/m6000-llama-cpp; D=$M/tmp-combo/sm52/bin; O=$M/opt27b-20261113; OUT=$O/tbo-sm52-perf-20261108.txt
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
trap "docker rm -f sm52-tbo >/dev/null 2>&1" EXIT
for s in 0 1 0 1; do echo "== SM52=$s" | tee -a $OUT
docker run --rm --name sm52-tbo --device nvidia.com/gpu=$GPU -v $D:/b:ro -e LD_LIBRARY_PATH=/b -e GGML_CUDA_MMVQ_SM52=$s --entrypoint /b/test-backend-ops $IMG perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0,type_b=f32,m=4096,n=[1-3],k=14336" 2>&1 | grep -E "n=[1-3],k" | sed -E 's/.*n=([0-9]),k=.* ([0-9.]+) us\/run.*/n=\1 \2 us/' | tee -a $OUT
done
