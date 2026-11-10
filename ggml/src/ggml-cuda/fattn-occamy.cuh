#pragma once
// Track F (native-occamy): GQA split-K FA decode kernel for sm_52, env-gated (see fattn-occamy.cu).
#include "common.cuh"

int  ggml_cuda_occamy_fa_mode();
bool ggml_cuda_occamy_fa_applicable(const ggml_tensor * dst);
void ggml_cuda_occamy_fa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_occamy_fa_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
