#!/bin/bash
D=$M/opt27b-20261113/card9-sched; M=$M
exec flock $M/gpu-bench.lock ~/.local/bin/labexp run --title "card9 sched prefill-first" --gpu ryzen:1 --agent m6k-patcher -- $D/wrap.sh $D/chain.sh
