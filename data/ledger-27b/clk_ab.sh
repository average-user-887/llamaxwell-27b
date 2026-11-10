#!/bin/bash
# 7 Nov: clock A/B 987->1152 MHz on C3. A = R21 (4k) + fa-gemm-long-20261107-2001 (long). Temp/power/throttle sampled every 2s.
set -u; O=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113; G=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
SO=/media/avg-usr/Storage/m6000-llama-cpp/mmvqtune-libs/libggml-cuda-mmvqT-fagemm.so; TS=$(date +%Y%m%d-%H%M)
OUT=$O/clk1152-long-$TS.txt; TMP=$O/clk1152-temps-$TS.csv
exec 9>$O/../gpu-bench.lock; flock 9
nvidia-smi -i $G --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active -l 2 --format=csv > $TMP & SP=$!
trap "kill $SP 2>/dev/null; docker rm -f llamacpp-m6000-27b-exp >/dev/null 2>&1" EXIT
docker stop -t 60 llamacpp-m6000-27b >/dev/null 2>&1
echo "== long pp @1152 (FA_GEMM=1, C3)" | tee -a $OUT
VARS="1" python3 $O/fa_eval.py $SO 1200 2600 4900 2>&1 | tee -a $OUT
python3 $O/quick.py 22 2>&1 | tee $O/r22.log
echo "== peak temp/power"; awk -F', ' 'NR>1{t=$2+0;p=$3+0; if(t>mt)mt=t; if(p>mp)mp=p} END{print "peak_temp", mt, "peak_power", mp}' $TMP | tee -a $OUT
nvidia-smi -i $G --query-gpu=clocks.applications.graphics --format=csv,noheader | sed 's/^/apps_clock_end /' | tee -a $OUT
