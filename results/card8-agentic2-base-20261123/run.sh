#!/bin/bash
H=<H>; D=$H/card8-20261123; M=$H/..; cd $H
GU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
cleanup(){ kill $W 2>/dev/null; docker rm -f llamacpp-m6000-27b-exp >/dev/null 2>&1
  if [ -e $H/NO_RESTORE ]; then echo "NO_RESTORE present, prod not restarted" >> $D/run.log
  else docker start llamacpp-m6000-27b >/dev/null 2>&1; for i in $(seq 1 90); do curl -sf localhost:11437/health >/dev/null && { echo "prod healthy" >> $D/run.log; break; }; sleep 5; done; fi
  echo END >> $D/run.log; }
(while true; do echo "$(date +%T),$(nvidia-smi --id=$GU --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader,nounits|tr -d ' ')" >> $D/temps.csv; sleep 10; done) & W=$!
trap cleanup EXIT
echo "lock acquired $(date +%T)" >> $D/run.log
curl -s -X POST http://172.21.0.1:11445/api/models/unload >> $D/run.log 2>&1
docker stop -t 60 llamacpp-m6000-27b >> $D/run.log 2>&1
python3 $H/agentic2.py 2 > $D/agentic2.log 2>&1
