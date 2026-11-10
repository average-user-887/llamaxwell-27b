#!/bin/bash
# chain6 lib install failed (dbg.so root-owned), so its z*/za* runs were the v2 lib without NOQ8. Rerun with v3 lib directly via SO.
M=/media/avg-usr/Storage/m6000-llama-cpp; F=$M/opt27b-20261113/f2
export SO=$M/tmp-combo/libggml-cuda-fadec2dbg-v3.so
B="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DBG_AUTH=0 GGML_CUDA_FA_DBG_DUMP=1616:1618 GGML_CUDA_FA_DBG_NOQ8=1"
N="GGML_CUDA_FA_DBG_NOISERANGE=1616:1617"
KRUNMAX=90 $F/kldrun.sh y1r $B
KRUNMAX=90 $F/kldrun.sh y1n6 $B $N GGML_CUDA_FA_DBG_NOISE=1e-6
KRUNMAX=90 $F/kldrun.sh y1n4 $B $N GGML_CUDA_FA_DBG_NOISE=1e-4
C="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DBG_NOQ8=1"
KRUNMAX=300 $F/kldrun.sh ya0 $C GGML_CUDA_FA_DBG_AUTH=0
KRUNMAX=300 $F/kldrun.sh ya2 $C GGML_CUDA_FA_DBG_AUTH=2
