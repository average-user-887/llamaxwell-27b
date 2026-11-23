#!/bin/bash
cd $M/opt27b-20261113/card7-lctx
[ $(date +%H%M) -lt 1445 ] && python3 -I proxy.py arms B A A B > arms2.log 2>&1
echo "chain2 done $(date +%T)" >> run.log
