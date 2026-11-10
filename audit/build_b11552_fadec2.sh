#!/usr/bin/env bash
# 8 Nov fadec2: GGML_CUDA_FA_DEC=2 design-2 GQA q4_0 decode (fattn-dec.cu). Builds the fadec tree copy mounted at the old path (CMake cache is path-bound).
set -uo pipefail
ROOT=/media/avg-usr/Storage/m6000-llama-cpp; LOG=$ROOT/build-b11552-fadec2-$(date +%Y%m%dT%H%M%S).log
nice -n 19 docker run --rm --cpuset-cpus 6-11 --name build-fadec2-20261108 -v $ROOT:/w -v $ROOT/src-b11552-fadec-20261108:/w/src-b11552-mmvqtune-20261107 -w /w/src-b11552-mmvqtune-20261107 \
  -e TMPDIR=/w/tmp-combo nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
set -e
APTO="-o Dir::Cache::archives=/w/tmp-combo/apt-archives -o Dir::State::Lists=/w/tmp-combo/apt-lists"
mkdir -p /w/tmp-combo/apt-archives/partial /w/tmp-combo/apt-lists/partial
apt-get $APTO update -qq && DEBIAN_FRONTEND=noninteractive apt-get $APTO install -y -qq --no-install-recommends gcc-14 g++-14 cmake git libssl-dev >/dev/null
export CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_TYPE=Release . >/dev/null
H=ggml/src/ggml-cuda/mmvq_tune.h
for v in "fagemm 1 2 2 2 4 4 1 0"; do
  set -- $v
  printf "#pragma once\n#define TUNE_NW1 %s\n#define TUNE_NW24 %s\n#define TUNE_NW58 %s\n#define TUNE_RPB1 %s\n#define TUNE_RPB24 %s\n#define TUNE_RPB58 %s\n#define TUNE_NW3 %s\n#define TUNE_XCOLS %s\n" $2 $3 $4 $5 $6 $7 $8 $9 > $H
  echo "== variant $1 $(date +%T)"
  nice -n 19 cmake --build build --config Release -j6 --target ggml-cuda test-backend-ops 2>&1 | grep -E "error|Built target" | tail -5
  cp build/bin/libggml-cuda.so /w/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so
done
printf "#pragma once\n#define TUNE_NW1 1\n#define TUNE_NW24 2\n#define TUNE_NW58 2\n#define TUNE_RPB1 1\n#define TUNE_RPB24 2\n#define TUNE_RPB58 2\n#define TUNE_NW3 2\n" > $H
' > $LOG 2>&1; rc=$?
echo "done rc=$rc log=$LOG"; ls -la $ROOT/mmvqtune-libs
