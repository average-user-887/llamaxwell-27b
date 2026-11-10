#!/bin/bash
# 7 Nov day chain: R5 (env levers) + R6 (spec/layout levers), hard deadline 17:50 inside quick.py.
cd "$(dirname "$0")"
L=day-$(date +%Y%m%d-%H%M).log
for r in 5 6; do
  flock /media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock /usr/bin/python3 -I quick.py $r >>"$L" 2>&1
  echo "ROUND $r rc=$?" >>"$L"
  [ $r -eq 5 ] && sleep 900
done
echo "DAY DONE $(date)" >>"$L"
