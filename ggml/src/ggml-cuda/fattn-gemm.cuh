#pragma once
#include "common.cuh"
int  ggml_cuda_fa_gemm_mode();
bool ggml_cuda_fa_gemm_applicable(const ggml_tensor * dst);
void ggml_cuda_fa_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
