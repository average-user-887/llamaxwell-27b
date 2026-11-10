#!/bin/bash
# no-q8 route costs about 1-1.6 s/eval: gain test moved to call 336 (eval 20) so the dump is reached.
M=/media/avg-usr/Storage/m6000-llama-cpp; F=$M/opt27b-20261113/f2
B="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DBG_AUTH=0 GGML_CUDA_FA_DBG_DUMP=336:338"
N="GGML_CUDA_FA_DBG_NOISERANGE=336:337"
KRUNMAX=35 $F/kldrun.sh k1r $B
KRUNMAX=35 $F/kldrun.sh k1n6 $B $N GGML_CUDA_FA_DBG_NOISE=1e-6
KRUNMAX=35 $F/kldrun.sh k1n4 $B $N GGML_CUDA_FA_DBG_NOISE=1e-4
export SO=$M/tmp-combo/libggml-cuda-fadec2dbg-v3.so
KRUNMAX=70 $F/kldrun.sh w1r $B GGML_CUDA_FA_DBG_NOQ8=1
KRUNMAX=70 $F/kldrun.sh w1n6 $B GGML_CUDA_FA_DBG_NOQ8=1 $N GGML_CUDA_FA_DBG_NOISE=1e-6
KRUNMAX=70 $F/kldrun.sh w1n4 $B GGML_CUDA_FA_DBG_NOQ8=1 $N GGML_CUDA_FA_DBG_NOISE=1e-4
