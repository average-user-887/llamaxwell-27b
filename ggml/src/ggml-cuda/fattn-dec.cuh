#pragma once
// 8 Nov fadec design 2 (GGML_CUDA_FA_DEC=2): GQA-grouped q4_0 decode flash attention for sm_52 (see fattn-dec.cu).
#include "common.cuh"

int  ggml_cuda_fa_dec_mode();                              // value of GGML_CUDA_FA_DEC (cached)
bool ggml_cuda_fa_dec_applicable(const ggml_tensor * dst);
void ggml_cuda_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
