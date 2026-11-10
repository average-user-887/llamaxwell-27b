#!/bin/bash
M=/media/avg-usr/Storage/m6000-llama-cpp; D=$M/tmp-combo/sm52/bin; O=$M/opt27b-20261113
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
trap "docker rm -f sm52-tbo >/dev/null 2>&1" EXIT
docker run --rm --name sm52-tbo --device nvidia.com/gpu=$GPU -v $D:/b:ro -e LD_LIBRARY_PATH=/b -e GGML_CUDA_MMVQ_SM52=1 --entrypoint /b/test-backend-ops $IMG test -o MUL_MAT -b CUDA0 > $O/tbo-sm52-raw-test1.txt 2>&1
docker run --rm --name sm52-tbo --device nvidia.com/gpu=$GPU -v $D:/b:ro -e LD_LIBRARY_PATH=/b -e GGML_CUDA_MMVQ_SM52=1 --entrypoint /b/test-backend-ops $IMG perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0,type_b=f32,m=5120,n=[1-3],k=17408" > $O/tbo-sm52-raw-perf.txt 2>&1
