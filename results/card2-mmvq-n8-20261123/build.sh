#!/usr/bin/env bash
# card2 build: ggml-cuda + test-backend-ops, sm52, same flags as build_b11552_* scripts
set -uo pipefail
R=$HOME/.claude/jobs/f65c1077/tmp/llamaxwell-27b; A=/media/$USER/Storage/m6000-llama-cpp/tmp-combo
nice -n 19 docker run --rm --cpuset-cpus 2-11 --name build-card2-n8 -v $R:/src -v $A:/apt -w /src nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
set -e
APTO="-o Dir::Cache::archives=/apt/apt-archives -o Dir::State::Lists=/apt/apt-lists"
apt-get $APTO update -qq && DEBIAN_FRONTEND=noninteractive apt-get $APTO install -y -qq --no-install-recommends gcc-14 g++-14 cmake git libssl-dev >/dev/null
export CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
[ -f build-n8/CMakeCache.txt ] || cmake -B build-n8 -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_TYPE=Release . >/dev/null
cmake --build build-n8 --config Release -j10 --target ggml-cuda '"${TARGETS:-}"' 2>&1 | grep -E "error|warning: .*mmvq|Linking|Built target" | tail -20
chown -R '$(id -u):$(id -g)' build-n8
'
echo "rc=$?"; sha256sum $R/build-n8/bin/libggml-cuda.so
