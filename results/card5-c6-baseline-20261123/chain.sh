#!/bin/bash
H=/media/$USER/Storage/m6000-llama-cpp/opt27b-20261113; D=$H/card5-20261123; cd $H
touch $H/NO_RESTORE
GU=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep -i m6000 | cut -d, -f1 | head -1)
(while true; do T=$(nvidia-smi --id=$GU --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader,nounits|tr -d ' '); echo "$(date +%T),$T" >> $D/temps.csv; t=${T%%,*}; if [ "$t" -ge 80 ]; then echo "THERMAL ABORT $t" >> $D/chain.log; docker rm -f llamacpp-m6000-27b-exp; pkill -f 'card5-20261123/chain.sh'; fi; sleep 10; done) & W=$!
trap 'kill $W 2>/dev/null; rm -f $H/NO_RESTORE; docker rm -f llamacpp-m6000-27b-exp >/dev/null 2>&1; echo CHAIN-END >> $D/chain.log' EXIT
export QUICK_LONG=1
L(){ echo "== $(date +%T) $*" >> $D/chain.log; }
L quick23; python3 -I $H/quick.py 23 > $D/quick23.log 2>&1
L tg_long; python3 $H/tg_long.py 1200 2600 4900 > $D/tglong.log 2>&1
for S in A B; do L "agentic2 $S"; python3 $H/agentic2.py 2 > $D/agentic2-$S.log 2>&1; done
L done
