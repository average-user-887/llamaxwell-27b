#!/bin/bash
# 9 Nov: no-q8 pair (chunks 0-10) then 64-chunk re-gate on the real fadec2.so. Whole chain under gpu-bench.lock; prod restored on exit.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; F=$O/f2; X=/media/avg-usr/Models/kld-f2-20261108; R=$O/fadec2-noq8-regate-20261109.txt
restore(){ docker start llamacpp-m6000-27b >/dev/null; mv -f $O/NO_RESTORE $O/NO_RESTORE.done-20261109 2>/dev/null; echo "prod start $(date +%T)" >> $R; }
trap restore EXIT
{ echo "9 Nov no-q8 + 64-chunk re-gate. Gate fixed before results: |dmean|<=.0005, same-top within .3 pt. Start $(date +%T)"; md5sum $M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so; } >> $R
touch $O/NO_RESTORE; docker stop -t 30 llamacpp-m6000-27b >/dev/null
N="GGML_CUDA_FA_DBG=1 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DBG_NOQ8=1"
export SO=$X/libggml-cuda-fadec2dbg-v3.so CHUNKS=11 KRUNMAX=5400
$F/kldrun.sh nq0 $N GGML_CUDA_FA_DBG_AUTH=0 2>&1 | tee -a $R
$F/kldrun.sh nq2 $N GGML_CUDA_FA_DBG_AUTH=2 2>&1 | tee -a $R
echo "== NO-Q8 chunks 0-10: stock vs DEC=2" >> $R; python3 $F/gate64.py nq0 nq2 >> $R 2>&1
export SO=$M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so CHUNKS=64 BASE=/x/head4-64.kld KRUNMAX=3600
$F/kldrun.sh g0 GGML_CUDA_FA_DEC=0 KLDSHIM_RAWTHR=99 2>&1 | tee -a $R
$F/kldrun.sh g2 GGML_CUDA_FA_DEC=2 KLDSHIM_RAWTHR=99 2>&1 | tee -a $R
echo "== 64-CHUNK GATE real lib: DEC=0 vs DEC=2" >> $R; python3 $F/gate64.py g0 g2 >> $R 2>&1
echo "== real vs debug lib (F2 a064/a264)" >> $R; python3 $F/gate64.py a064 g0 >> $R 2>&1; python3 $F/gate64.py a264 g2 >> $R 2>&1
