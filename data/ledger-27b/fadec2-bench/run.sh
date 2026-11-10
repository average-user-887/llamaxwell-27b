#!/bin/bash
# compile both AV variants and run check+timing in a CUDA devel container; call under flock gpu-bench.lock
B=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/fadec2-bench
S=/media/avg-usr/Storage/m6000-llama-cpp/src-b11552-fadec-20261108
docker run --rm --name fadec2-bench --gpus device=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e -v $S:/w/src:ro -v $B:/b nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
cd /b && for av in 1 2; do nvcc -O3 -arch=sm_52 -DFD_AV=$av -Xptxas -v -o bench_av$av bench.cu 2> ptxas_av$av.txt || { cat ptxas_av$av.txt; exit 1; }; grep -A1 "fd_kernelILi6ELi3E" ptxas_av$av.txt | grep -o "Used [0-9]* registers.*" ; done
for av in 1 2; do ./bench_av$av "$@"; done' -- "$@"
