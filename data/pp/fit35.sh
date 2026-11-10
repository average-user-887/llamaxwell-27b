# fit35.sh: load 35B with prod args (bench container, no ports) at several ubatch sizes; report VRAM. Run under gpu-bench.lock.
J=/home/avg-usr/.claude/jobs/fc533454/tmp/pp; C=ppprobe-fit
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
ENV="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1"
for UB in ${UBS:-512 1024 1536 2048}; do
  docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
    -v /media/avg-usr/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-xmad35-20261110.so:/app/libggml-cuda.so:ro $ENV $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b.gguf --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub $UB \
    --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 127.0.0.1 --port 8080 >/dev/null
  timeout 300 docker logs -f $C 2>&1 | grep -m1 -E 'server is listening|failed|error|out of memory' > $J/out/fit35-$UB.state
  docker logs $C 2>&1 | grep -E 'compute buffer size|KV buffer size|model buffer size' | grep -i cuda > $J/out/fit35-$UB.buf
  echo "ub=$UB state=$(cut -c1-80 $J/out/fit35-$UB.state) vram=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
  cat $J/out/fit35-$UB.buf
  docker rm -f $C >/dev/null 2>&1
done
