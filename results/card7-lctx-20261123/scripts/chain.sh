#!/bin/bash
# Card 7 chain (inside wrap.sh, under gpu-bench.lock): tbo, proxy ABBA, quick A B.
D=$M/opt27b-20261113/card7-lctx; cd $D
[ $(date +%H%M) -lt 1530 ] && ./tbo.sh
[ $(date +%H%M) -lt 1530 ] && python3 -I proxy.py arms A B B A > arms.log 2>&1
[ $(date +%H%M) -lt 1520 ] && python3 -I proxy.py quick A B > quick.log 2>&1
echo "chain done $(date +%T)" >> run.log
