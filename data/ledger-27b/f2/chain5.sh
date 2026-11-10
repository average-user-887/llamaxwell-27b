#!/bin/bash
F=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/f2; X=/media/avg-usr/Models/kld-f2-20261108
$F/genbase64.sh
[ -s $X/head4-64.kld ] || exit 1
export CHUNKS=64 BASE=/x/head4-64.kld
B="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99"
$F/kldrun.sh a064 $B GGML_CUDA_FA_DBG_AUTH=0
$F/kldrun.sh a264 $B GGML_CUDA_FA_DBG_AUTH=2
