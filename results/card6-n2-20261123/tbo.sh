#!/usr/bin/env bash
# card6: test-backend-ops MUL_MAT q4_0 CUDA vs CPU with N2SPLIT off / 3 / 2 on the n2split lib (runtime image)
T=$HOME/.claude/jobs/f65c1077/tmp/tbo-card6
GU=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep -i m6000 | cut -d, -f1 | head -1)
for S in ${TBO_S:-0 3 2}; do
  echo "== N2SPLIT=$S"
  docker run --rm --gpus device=$GU -v $T:/t -w /t --entrypoint bash \
    -e LD_LIBRARY_PATH=/t:/app -e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_MMVQ_HIST=1 -e GGML_CUDA_MMVQ_N2SPLIT=$S \
    sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d \
    -c './test-backend-ops test -o MUL_MAT -b CUDA0 -p q4_0 2>&1' > /tmp/tbo-card6-$S.log 2>&1
  echo "LABEXP tbo_n2split${S}_rc=$? rc"
  grep -E "mmvq-hist|tests passed|backends passed|FAIL|CUDA0" /tmp/tbo-card6-$S.log | sort | uniq -c | sort -rn | head -20
done
