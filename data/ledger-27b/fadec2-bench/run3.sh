#!/bin/bash
# edge-case checks (env CNKV CVALID CSPLIT CQS). call under flock gpu-bench.lock
B=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/fadec2-bench
S=/media/avg-usr/Storage/m6000-llama-cpp/src-b11552-fadec-20261108
docker run --rm --name fadec2-bench --gpus device=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e -v $S:/w/src:ro -v $B:/b nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
cd /b; nvcc -O3 -arch=sm_52 -include cstdlib -DTAG="\"chk\"" -o bin_chk bench.cu || exit 1
for c in "256 3 1 1" "256 3 2 1" "512 130 4 1" "512 300 4 1" "512 300 1 1" "256 3 2 6" "512 300 4 6" "512 300 1 6" "1024 987 3 6" "1024 987 8 6"; do set -- $c
 CNKV=$1 CVALID=$2 CSPLIT=$3 CQS=$4 ./bin_chk 256 6 3 4 1 | grep check | sed "s/$/ qs=$4/"; done' 2>&1 | grep -v "^time"
