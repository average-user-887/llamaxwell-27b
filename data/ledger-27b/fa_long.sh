#!/bin/bash
# 7 Nov: long-ctx GEMM-FA A/B + KC sweep, no prod restore (owner: GPU time is a resource)
set -u; O=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113; SO=/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fagemm.so
OUT=$O/fa-gemm-long-20261107-$(date +%H%M).txt; trap "docker rm -f llamacpp-m6000-27b-exp >/dev/null 2>&1" EXIT
docker stop -t 60 llamacpp-m6000-27b >/dev/null 2>&1
echo "== KC sweep at ~22.7k" | tee -a $OUT; VARS="1:2048 1:1024" python3 $O/fa_eval.py $SO 1200 2>&1 | tee -a $OUT
echo "== long A/B" | tee -a $OUT; VARS="0 1" python3 $O/fa_eval.py $SO 2600 4900 2>&1 | tee -a $OUT
