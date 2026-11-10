#!/bin/bash
# 7 Nov card C: MUL_MAT correctness + q4_0 perf (n=1..3, model shapes), GGML_CUDA_MMVQ_SM52=0/1. Run under flock gpu-bench.lock.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; OUT=$O/tbo-sm52-20261107-b.txt
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
D=$M/tmp-combo/sm52/bin; mkdir -p $D; B=$M/src-b11552-mmvqtune-20261107/build/bin
cp -f $B/test-backend-ops $B/libggml-base.so* $B/libggml.so* $B/libggml-cpu*.so $B/libllama*.so* $D/ 2>/dev/null
cp -f $M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-sm52.so $D/libggml-cuda.so
trap "docker rm -f sm52-tbo >/dev/null 2>&1" EXIT
for s in 1; do echo "== SM52=$s test" | tee -a $OUT
docker run --rm --name sm52-tbo --device nvidia.com/gpu=$GPU -v $D:/b:ro -e LD_LIBRARY_PATH=/b -e GGML_CUDA_MMVQ_SM52=$s \
  --entrypoint /b/test-backend-ops $IMG test -o MUL_MAT -b CUDA0 2>&1 | grep -E "FAIL|passed|Backend CUDA0:" | head -30 | tee -a $OUT
done
for s in 0 1 0 1; do for sh in "m=5120,n=[1-3],k=17408" "m=17408,n=[1-3],k=5120"; do echo "== SM52=$s perf $sh" | tee -a $OUT
docker run --rm --name sm52-tbo --device nvidia.com/gpu=$GPU -v $D:/b:ro -e LD_LIBRARY_PATH=/b -e GGML_CUDA_MMVQ_SM52=$s \
  --entrypoint /b/test-backend-ops $IMG perf -o MUL_MAT -b CUDA0 -p "type_a=q4_0,type_b=f32,$sh" 2>&1 | grep -E "n=[1-3],k" | sed -E 's/.*n=([0-9]),k=([0-9]+).* ([0-9.]+) us\/run.*/n=\1 k=\2 \3 us/' | tee -a $OUT
done; done
