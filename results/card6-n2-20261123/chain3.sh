#!/bin/bash
# card6 chain (run under flock gpu-bench.lock): quick.py r26 ABBA of GGML_CUDA_MMVQ_N2SPLIT=3, then KLD -b 6 A/B.
H=/media/$USER/Storage/m6000-llama-cpp/opt27b-20261113; C=$(dirname $(readlink -f $0)); cd $H
SO=/media/$USER/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-n2split-v2-20261123.so
touch $H/NO_RESTORE
GU=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep -i m6000 | cut -d, -f1 | head -1)
(while true; do T=$(nvidia-smi --id=$GU --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits|tr -d ' '); echo "$(date +%T),$T" >> $C/temps.csv; t=${T%%,*}; if [ "$t" -ge 80 ]; then echo "THERMAL ABORT $t" >> $C/chain.log; docker rm -f llamacpp-m6000-27b-exp kld-27b-f2; pkill -f card6-n2-20261123/chain.sh; fi; sleep 10; done) & W=$!
trap 'kill $W 2>/dev/null; rm -f $H/NO_RESTORE; echo CHAIN-END $(date +%T) >> $C/chain.log' EXIT
L(){ echo "== $(date +%T) $*" >> $C/chain.log; }
LX=~/.local/bin/labexp

L abba; QUICK_SO=$SO QUICK_OUT=$C/abba QUICK_ENVK=GGML_CUDA_MMVQ_N2SPLIT QUICK_ENVV=3 QUICK_HIST=1 \
  $LX run --title card6-abba-n2split3-v2-hist --gpu ryzen:1 --agent m6k-coder -- python3 -I $H/quick.py 26 > $C/quick26.log 2>&1
KE="GGML_CUDA_CONCAT_T=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_RMS_WARP=0 KLDSHIM_RAWTHR=99 GGML_CUDA_FA_DEC=2 GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_MMVQ_SM52_N8=1"
for S in; do :
  SO=$SO $LX run --title card6-kld6-v2-n2split$S --gpu ryzen:1 --agent m6k-coder -- bash $C/kldrun6.sh c6n2v2s$S $KE GGML_CUDA_MMVQ_N2SPLIT=$S > $C/kld-$S.txt 2>&1
done
L done
