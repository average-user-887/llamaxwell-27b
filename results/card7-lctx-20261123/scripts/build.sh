#!/bin/bash
# Card 7 build: CPU only (no --gpus), sm_52, flags as llamaxwell-27b build-n8 (prod C6 lib).
W=$WT; mkdir -p $W/build-c7
taskset -c 2-11 docker run --rm --cpuset-cpus 2-11 --name c7-build -u "$(id -u):$(id -g)" --network none \
  -v $W:/src:ro -v $W/build-c7:/b -w /b -e TMPDIR=/tmp -e CC=gcc-14 -e CXX=g++-14 -e CUDAHOSTCXX=g++-14 local/m6000-build-tools:20261013 bash -c '
set -e
[ -f /b/CMakeCache.txt ] || cmake -S /src -B /b -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" -DGGML_CUDA_COMPRESSION_MODE=size
cmake --build /b --config Release -j10 --target ggml-cuda test-backend-ops
ls -la /b/bin/libggml-cuda.so /b/bin/test-backend-ops'
