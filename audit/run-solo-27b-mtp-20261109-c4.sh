#!/bin/bash
# solo-27b C4 (9 Nov 2026): C3 + design 2 q4_0 decode FA (fadec2 lib, GGML_CUDA_FA_DEC=2). Owner accepted despite KLD gate FAIL (64 chunks +.000627 vs .0005).
# Measured: tg@100k 8.07->14.39, tg@52k 11.59->16.78, short tg -0.39%, pp flat. opt27b-20261113/LEDGER.md.
# Rollback: docker stop llamacpp-m6000-27b; docker rename llamacpp-m6000-27b llamacpp-m6000-27b-c4-off; docker rename llamacpp-m6000-27b-c3-20261109 llamacpp-m6000-27b; docker start llamacpp-m6000-27b
docker run -d --name llamacpp-m6000-27b --network ai-net --restart no -p 127.0.0.1:11437:8080 \
  --device nvidia.com/gpu=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e \
  -v /media/avg-usr/Models/gguf:/m2:ro \
  -v /media/avg-usr/Storage/llama-slots-m6000:/slots \
  -v /media/avg-usr/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-fadec2-20261109.so:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_FA_DEC=2 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 \
  sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d \
  -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --alias solo-27b -ngl 999 --fit off -np 2 -no-kvu -c 262144 \
  -ctk q4_0 -ctv q4_0 -b 2048 -ub 1024 --cache-ram 4096 --ctx-checkpoints 8 --checkpoint-min-step 2048 -fa on \
  --host 0.0.0.0 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 --no-mmproj --jinja --metrics --slot-save-path /slots -lv 4
