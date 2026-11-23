#!/bin/bash
# solo-27b C6 (23 Nov 2026): C5 + XMAD Q4_0 mmvq for ncols 5-8 (lib xmadn8-v2, GGML_CUDA_MMVQ_SM52_N8 default on). tg2agg 22.1->24.3 (+9.6%), tg1/pp4k flat, KLD bit-identical to C5 (b6, 16 chunks).
# Rollback: docker stop llamacpp-m6000-27b; docker rename llamacpp-m6000-27b llamacpp-m6000-27b-c6-off; docker rename llamacpp-m6000-27b-c5-20261123 llamacpp-m6000-27b; docker start llamacpp-m6000-27b; labslot check
docker create --name llamacpp-m6000-27b --network ai-net --restart no -p 127.0.0.1:11437:8080 \
  --device nvidia.com/gpu=GPU-<m6000-uuid> \
  -v /media/$USER/Models/gguf:/m2:ro \
  -v /media/$USER/Storage/llama-slots-m6000:/slots \
  -v /media/$USER/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-xmadn8-v2-20261123.so:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_FA_DEC=2 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1 -e GGML_CUDA_SGEMM_NN=1 -e GGML_CUDA_FA_GEMM=1 \
  sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d \
  -m /m2/qwen38-27b/Qwen3.8-27B-Q4_0-head4.gguf --alias solo-27b -ngl 999 --fit off -np 2 -no-kvu -c 262144 \
  -ctk q4_0 -ctv q4_0 -b 2048 -ub 1024 --cache-ram 4096 --ctx-checkpoints 8 --checkpoint-min-step 2048 -fa on \
  --host 0.0.0.0 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 --no-mmproj --jinja --metrics --slot-save-path /slots -lv 4
