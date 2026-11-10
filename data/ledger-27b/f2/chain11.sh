#!/bin/bash
# 9 Nov Scout 3 Stage 1. Short speed, then 64-chunk KLD, then long speed unless F short tg is under .93x. Call under flock gpu-bench.lock; C4 restored on exit.
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; F=$O/f2; R=$O/fp32mv-stage1-20261109.txt
restore(){ docker rm -f llamacpp-m6000-27b-exp >/dev/null 2>&1; docker start llamacpp-m6000-27b >/dev/null; mv -f $O/NO_RESTORE $O/NO_RESTORE.done-20261109c 2>/dev/null; echo "prod start $(date +%T)" >> $R; }
trap restore EXIT
P="GGML_CUDA_CONCAT_T=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_RMS_WARP=0 KLDSHIM_RAWTHR=99"
echo "9 Nov FP32MV Stage 1. Start $(date +%T)" >> $R
touch $O/NO_RESTORE; docker stop -t 30 llamacpp-m6000-27b >/dev/null
python3 $F/stage1.py short C F C F >> $R 2>&1 || { echo "short rc=$?" >> $R; exit 1; }
python3 $F/stage1eval.py >> $R 2>&1; SKIP=$?
export SO=$M/prod-libs/libggml-cuda-fadec2-20261109.so CHUNKS=64 BASE=/x/head4-64.kld KRUNMAX=3600
$F/kldrun.sh c64 $P GGML_CUDA_FA_DEC=2 2>&1 | tee -a $R
$F/kldrun.sh f64 $P GGML_CUDA_FA_DEC=2 GGML_CUDA_FP32MV=1 2>&1 | tee -a $R
echo "== KLD 64: C4 vs C4+FP32MV" >> $R; python3 $F/gate64.py c64 f64 >> $R 2>&1
echo "== KLD 64: 3-flag DEC=0 (g0) vs C4+FP32MV" >> $R; python3 $F/gate64.py g0 f64 >> $R 2>&1
if [ "$SKIP" = 3 ]; then echo "long skipped: F short tg under .93x" >> $R; exit 0; fi
python3 $F/stage1.py long C F >> $R 2>&1 || echo "long rc=$?" >> $R
echo "== final" >> $R; python3 $F/stage1eval.py >> $R 2>&1
