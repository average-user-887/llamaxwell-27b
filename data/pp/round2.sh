cd /home/avg-usr/.claude/jobs/fc533454/tmp/pp
R=/media/avg-usr/Storage/m6000-llama-cpp/prod-libs; L27=$R/libggml-cuda-xmad27-20261110.so; L35=$R/libggml-cuda-xmad35-20261110.so
PROF=1 bash ppbench.sh 27 p27 $L27 4096 1
PROF=1 bash ppbench.sh 35 p35 $L35 4096 1
bash ppbench.sh 35 s35nn $L35 4096,16384 2 GGML_CUDA_SGEMM_NN=1
bash ppbench.sh 35 s35fa $L35 4096,16384 2 GGML_CUDA_FA_GEMM=1
bash ppbench.sh 35 s35both $L35 4096,16384,65536 1 GGML_CUDA_SGEMM_NN=1 GGML_CUDA_FA_GEMM=1
UB=1536 bash ppbench.sh 27 u27 $L27 4096,16384 2
UB=2048 bash ppbench.sh 27 u27b $L27 4096 2
UB=1024 bash ppbench.sh 35 u35 $L35 4096,16384 2
