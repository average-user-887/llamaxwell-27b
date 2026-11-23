#!/bin/bash
D=$M/opt27b-20261113/card9-sched; cd $D
for a in B A A B C D; do [ $(date +%H%M) -lt 1600 ] && python3 -I proxy9.py arms $a >> arms.log 2>&1; done
for a in A B; do [ $(date +%H%M) -lt 1612 ] && python3 -I proxy9.py quick $a >> quick.log 2>&1; done
echo "chain done $(date +%T)" >> run.log
