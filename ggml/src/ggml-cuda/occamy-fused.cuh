#pragma once
// native-occamy Track D (12 Oct 2026): fused qwen35moe MoE-block decode kernel for sm_52.
// Env-gated: GGML_CUDA_OCCAMY_FUSED=1 (default off). n_tokens == 1 only; any mismatch falls back to the normal path.
//   GGML_CUDA_OCCAMY_3L=1     use the 3-launch variant (no grid barrier) instead of the persistent single launch
//   GGML_CUDA_OCCAMY_DEBUG=1  log the first fused window and the first reason a candidate window was rejected
//   GGML_CUDA_OCCAMY_CHECK=N  on the first N fused windows also run the unfused path and print max abs/rel diff
#include "common.cuh"

struct ggml_cuda_occamy_moe_match {
    int n_nodes = 0;                 // window length (first = RMS_NORM, last = residual ADD)
    const ggml_tensor * h = nullptr; // residual stream input [2048]
    const ggml_tensor * norm_w = nullptr;
    float eps = 1e-6f;
    const ggml_tensor * gate_inp = nullptr, * gate_inp_sh = nullptr;
    const ggml_tensor * gate_exps = nullptr, * up_exps = nullptr, * down_exps = nullptr;
    const ggml_tensor * gate_sh = nullptr, * up_sh = nullptr, * down_sh = nullptr;
    ggml_tensor * out = nullptr;
};

bool ggml_cuda_occamy_enabled();
int  ggml_cuda_occamy_check_budget();   // remaining CHECK windows (decrements)
bool ggml_cuda_occamy_moe_match_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, ggml_cuda_occamy_moe_match & m);
void ggml_cuda_occamy_moe_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_moe_match & m, float * out_override);
