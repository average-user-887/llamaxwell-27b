#!/bin/bash
# variants: name "defines"; compile in parallel, run sequentially. call under flock gpu-bench.lock
B=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/fadec2-bench
S=/media/avg-usr/Storage/m6000-llama-cpp/src-b11552-fadec-20261108
VARS="$1"; shift
docker run --rm --name fadec2-bench --gpus device=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e -v $S:/w/src:ro -v $B:/b nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
cd /b; V="$1"; shift; for v in $V; do n=${v%%:*}; d=${v#*:}; nvcc -O3 -arch=sm_52 ${d//,/ } -DTAG="\"$n\"" -Xptxas -v -o bin_$n bench.cu 2> ptx_$n.txt & done; wait
for v in $V; do n=${v%%:*}; echo "$n $(grep -A2 fd_kernelILi6ELi3E ptx_$n.txt | grep -o "Used [0-9]* registers")"; ./bin_$n "$@"; done' -- "$VARS" "$@" 2>&1 | grep -E "^check|^time|registers|error"
