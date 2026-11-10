#!/bin/bash
# 9 Nov: Scout 3 first test on the full prod env. Arms: p0 = prod env DEC=0, p2 = C4, pf = C4 + FP32MV. Call under flock gpu-bench.lock; C4 restored on exit.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; F=$O/f2; R=$O/fp32mv-screen-20261109.txt
restore(){ docker start llamacpp-m6000-27b >/dev/null; mv -f $O/NO_RESTORE $O/NO_RESTORE.done-20261109b 2>/dev/null; echo "prod start $(date +%T)" >> $R; }
trap restore EXIT
P="GGML_CUDA_CONCAT_T=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_RMS_WARP=0 KLDSHIM_RAWTHR=99"
{ echo "9 Nov FP32MV screen. Fixed before results (F vs C): d mean <= +.0005; same-top >= -.3 pt; F mean <= C mean - .0030; no nonfinite; no new token > 3. Start $(date +%T)"; md5sum $M/prod-libs/libggml-cuda-fadec2-20261109.so; } >> $R
touch $O/NO_RESTORE; docker stop -t 30 llamacpp-m6000-27b >/dev/null
export SO=$M/prod-libs/libggml-cuda-fadec2-20261109.so CHUNKS=16 KRUNMAX=1800
$F/kldrun.sh p0 $P GGML_CUDA_FA_DEC=0 2>&1 | tee -a $R
$F/kldrun.sh p2 $P GGML_CUDA_FA_DEC=2 2>&1 | tee -a $R
$F/kldrun.sh pf $P GGML_CUDA_FA_DEC=2 GGML_CUDA_FP32MV=1 2>&1 | tee -a $R
echo "== design 2 cost on prod env: p0 vs p2" >> $R; python3 $F/gate64.py p0 p2 >> $R 2>&1
echo "== FP32MV screen: p2 (C) vs pf (F)" >> $R; python3 $F/gate64.py p2 pf >> $R 2>&1
