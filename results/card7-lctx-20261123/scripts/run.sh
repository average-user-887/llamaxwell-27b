#!/bin/bash
docker run --rm --name c7-fagbench --gpus device=$GU -v $M/opt27b-20261113/card7-lctx/fagbench:/b nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
cd /b && nvcc -O3 -arch=sm_52 -o fag fag.cu -lcublas 2>/dev/null || exit 1
for m in 5 6; do for a in -1 99 $(seq 0 23) $(seq 100 115); do ./fag 92416 512 2 4096 $m $a 2>&1 | grep -v "^err"; done; done'
