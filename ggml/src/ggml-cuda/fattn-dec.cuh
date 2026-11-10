#pragma once
// 8 Nov fadec design 2 (GGML_CUDA_FA_DEC=2): GQA-grouped q4_0 decode flash attention for sm_52 (see fattn-dec.cu).
#include "common.cuh"

int  ggml_cuda_fa_dec_mode();                              // value of GGML_CUDA_FA_DEC (cached)
bool ggml_cuda_fa_dec_applicable(const ggml_tensor * dst);
void ggml_cuda_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// 8 Nov card F2: env-gated in-process comparison with the stock kernel (GGML_CUDA_FA_DBG=1), diagnostics only.
typedef void (*ggml_cuda_fa_dbg_stock_fn)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_fa_dbg_on();
void ggml_cuda_fa_dbg_pre(ggml_tensor * dst);
void ggml_cuda_fa_dbg_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_fa_dbg_stock_fn stock);
