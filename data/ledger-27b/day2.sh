#!/bin/bash
# back-to-back rounds, no prod gaps (owner 7 Nov: full GPU time). Waits on the lock held by the running R5.
cd "$(dirname "$0")"; L=$(ls -t day-*.log | head -1)
for r in 6 7; do
  flock /media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock /usr/bin/python3 -I quick.py $r >>"$L" 2>&1
  echo "ROUND $r rc=$?" >>"$L"
done
echo "DAY DONE $(date)" >>"$L"
