#!/bin/bash
# 8 Nov fadec2: KLD vs head4 base, FA_DEC=$DEC (arg1, default 2). (a) standard c=512 prefill; (b) -b 3 -ub 3 so every FA call has 3 query cols; q4_0 KV (design 2 needs it). Run under flock gpu-bench.lock.
DEC=${1:-2}; shift; EXS=("$@"); [ ${#EXS[@]} = 0 ] && EXS=("-c 512 -b 3 -ub 3" "-c 512")
M=/media/avg-usr/Storage/m6000-llama-cpp; O=$M/opt27b-20261113; SO=$M/mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
W=/media/avg-usr/Models/kld-head4-20261107; OUT=$O/kld-fadec2-20261108.txt
trap "docker rm -f kld-27b-fadec2 >/dev/null 2>&1" EXIT
for ex in "${EXS[@]}"; do echo "== KLD vs head4 base (FA_DEC=$DEC ${XE:-}) $ex" | tee -a $OUT
docker run --rm --name kld-27b-fadec2 --device nvidia.com/gpu=$GPU -v /media/avg-usr/Models/gguf:/m2:ro -v $W:/w -v $SO:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 -e GGML_CUDA_FA_DEC=$DEC -e GGML_CUDA_FA_DEC_DEBUG=1 ${XE:-} --entrypoint /app/llama local/llama.cpp:server-cuda-b11552-sm52-normq8 \
  perplexity -ngl 99 -fa on -ctk q4_0 -ctv q4_0 $ex --chunks 16 -f /w/corpus.txt -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --kl-divergence-base /w/head4.kld --kl-divergence 2>&1 \
  | grep -E 'Mean PPL\(Q\) |Mean    KLD|Maximum KLD|Same top|error|seconds per pass|fa_dec2 (total|verify)' | tee -a $OUT
done
