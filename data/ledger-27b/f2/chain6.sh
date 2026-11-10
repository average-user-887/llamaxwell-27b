#!/bin/bash
# Card F2 amplifier test: q8_1 activation rounding as source of the step response. Run under gpu-bench.lock after chain5.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; F=$O/f2; X=/media/avg-usr/Models/kld-f2-20261108
L=$M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2dbg.so
[ -f $X/libggml-cuda-fadec2dbg-v2.so ] || cp $L $X/libggml-cuda-fadec2dbg-v2.so
cp $M/tmp-combo/libggml-cuda-fadec2dbg-v3.so $L
B="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DBG_AUTH=0 GGML_CUDA_FA_DBG_DUMP=1616:1618"
N="GGML_CUDA_FA_DBG_NOISERANGE=1616:1617"
KRUNMAX=60 $F/kldrun.sh c1r $B
KRUNMAX=60 $F/kldrun.sh c1n6 $B $N GGML_CUDA_FA_DBG_NOISE=1e-6
for t in "z1r" "z1n7 $N GGML_CUDA_FA_DBG_NOISE=1e-7" "z1n6 $N GGML_CUDA_FA_DBG_NOISE=1e-6" "z1n4 $N GGML_CUDA_FA_DBG_NOISE=1e-4"; do
  set -- $t; tag=$1; shift
  KRUNMAX=120 $F/kldrun.sh $tag $B GGML_CUDA_FA_DBG_NOQ8=1 "$@"
done
C="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DBG_NOQ8=1"
KRUNMAX=540 $F/kldrun.sh za0 $C GGML_CUDA_FA_DBG_AUTH=0
KRUNMAX=540 $F/kldrun.sh za2 $C GGML_CUDA_FA_DBG_AUTH=2
