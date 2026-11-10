#!/bin/bash
# Overnight chain: 4 short rounds (<=85 min each) with 15-min prod-up gaps. Every round restores prod itself.
cd "$(dirname "$0")"
L=night-$(date +%Y%m%d-%H%M).log
for r in 1 2 3 4; do
  flock /media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock /usr/bin/python3 -I quick.py $r >>"$L" 2>&1
  echo "ROUND $r rc=$?" >>"$L"
  [ $r -lt 4 ] && sleep 900
done
echo "NIGHT DONE $(date)" >>"$L"
