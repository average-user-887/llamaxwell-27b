# conc35.sh: prod-args 35B server at UB, 2 concurrent ~30k prompts. Run under gpu-bench.lock.
J=/home/avg-usr/.claude/jobs/fc533454/tmp/pp; C=ppprobe-conc; P=18391
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
ENV="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1"
python3 -I - <<PY
t=open('/media/avg-usr/Storage/m6000-llama-cpp/native-occamy/p3/raw/text.txt').read()
open('$J/out/conc-a.txt','w').write(t[:112000]); open('$J/out/conc-b.txt','w').write(t[92000:204000])
PY
for UB in ${UBS:-1024 512}; do
  docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C --device nvidia.com/gpu=$GPU -p 127.0.0.1:$P:8080 -v /media/avg-usr/Storage/uroboros-models:/models:ro \
    -v /media/avg-usr/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-xmad35-20261110.so:/app/libggml-cuda.so:ro $ENV $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b.gguf --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub $UB \
    --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --cache-ram 0 >/dev/null
  timeout 300 docker logs -f $C 2>&1 | grep -m1 -E 'server is listening|failed|error|out of memory' >/dev/null
  for s in a b; do
    python3 -I -c "
import json;print(json.dumps({'prompt':open('$J/out/conc-$s.txt').read(),'n_predict':8,'temperature':0,'cache_prompt':False}))" > $J/out/conc-$s.json
    curl -s -m 900 localhost:$P/completion -d @$J/out/conc-$s.json > $J/out/conc-$s-$UB.res &
  done
  wait
  echo "== ub=$UB vram=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader) running=$(docker inspect -f '{{.State.Running}}' $C)"
  for s in a b; do python3 -I -c "
import json;d=json.load(open('$J/out/conc-$s-$UB.res'));t=d['timings'];print('$s',d['tokens_evaluated'],round(t['prompt_per_second'],1),'pp t/s', d['content'][:30].replace('\n',' '))"; done
  docker logs $C 2>&1 | grep -ciE 'out of memory|cuda error|abort'
  docker rm -f $C >/dev/null 2>&1
done
