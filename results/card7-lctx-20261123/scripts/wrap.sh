#!/bin/bash
# Card 7 GPU wrapper: caller holds gpu-bench.lock. Unload swap, stop prod, run "$@", restore prod (unless NO_RESTORE).
H=$M/opt27b-20261113; D=$H/card7-lctx
GU=<GPU_UUID>; export GU
cleanup(){ kill $W 2>/dev/null; docker rm -f llamacpp-m6000-27b-exp c7-fagbench c7-tbo >/dev/null 2>&1
  if [ -e $H/NO_RESTORE ]; then echo "NO_RESTORE present" >> $D/run.log
  else docker start llamacpp-m6000-27b >/dev/null 2>&1; for i in $(seq 1 90); do curl -sf localhost:11437/health >/dev/null && { echo "prod healthy $(date +%T)" >> $D/run.log; break; }; sleep 5; done; fi; }
(while true; do T=$(nvidia-smi --id=$GU --query-gpu=temperature.gpu --format=csv,noheader,nounits); echo "$(date +%T),$T" >> $D/temps.csv; [ "$T" -ge 80 ] && { echo "THERMAL ABORT $T" >> $D/run.log; pkill -f c7-; docker rm -f llamacpp-m6000-27b-exp c7-fagbench c7-tbo; }; sleep 10; done) & W=$!
trap cleanup EXIT
echo "start $(date +%T): $*" >> $D/run.log
curl -s -X POST http://<HOST>:11445/api/models/unload >> $D/run.log 2>&1; echo >> $D/run.log
docker stop -t 60 llamacpp-m6000-27b >> $D/run.log 2>&1
"$@"
