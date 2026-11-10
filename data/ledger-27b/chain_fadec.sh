#!/bin/bash
# 8 Nov fadec chain: wait for build, then tbo, KLD, tg A/B each under the GPU lock. Hard stop 07:30.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113
while pgrep -f build_b11552_fadec.sh >/dev/null; do sleep 20; done
ls -la $M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec.so || { echo BUILD-FAILED; tail -20 $M/build-b11552-fadec-*.log; exit 1; }
flock $M/gpu-bench.lock $O/tbo_fadec.sh
flock $M/gpu-bench.lock $O/kld_fadec.sh
flock $M/gpu-bench.lock python3 $O/tg_long_fadec.py 2600 4900 2>&1 | tee $O/fadec-ab-20261108.txt
echo CHAIN-DONE
