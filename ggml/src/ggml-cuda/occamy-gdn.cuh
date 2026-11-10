#pragma once
// native-occamy Track G (13 Oct 2026): fused qwen35moe GDN (gated delta net) layer decode segment for sm_52.
// Window = attn_norm RMS_NORM .. attention residual ADD (qkv/gate/alpha/beta GEMV, conv1d(4) + state, l2norm,
// softplus/exp gate, sigmoid beta, delta rule state update, gated rmsnorm * silu(z), ssm_out GEMV + residual).
// Env-gated: GGML_CUDA_OCCAMY_GDN=1 (default off). n_tokens == 1, n_seqs == 1, n_rs == 1 only; mismatch -> normal path.
//   GGML_CUDA_OCCAMY_GDN_3L=1     3-launch variant (no grid barrier)
//   GGML_CUDA_OCCAMY_DEBUG=1      log the first fused window, rejected GDN windows (with the op list once) and counters
//   GGML_CUDA_OCCAMY_GDN_CHECK=N  on the first N fused windows also run the stock ops and print max abs/rel diff
//                                 of the layer output, the written SSM state and the written conv state
#include "common.cuh"

struct ggml_cuda_occamy_gdn_match {
    int n_nodes = 0;
    int n_seqs = 1;
    const ggml_tensor *gr_ssm=nullptr,*gdn=nullptr,*cpy_ssm=nullptr;
    const ggml_tensor * h = nullptr;         // residual stream in [2048]
    const ggml_tensor * norm_w = nullptr;    // attn_norm [2048]
    float eps_attn = 1e-6f, eps_l2 = 1e-6f / 128, scale_l2 = 0.088388f, eps_gated = 1e-6f;
    const ggml_tensor * wqkv = nullptr, * wgate = nullptr, * walpha = nullptr, * wbeta = nullptr, * wout = nullptr; // Q4_0
    const ggml_tensor * conv_w = nullptr;    // f32 [4, 8192]
    const ggml_tensor * dt_bias = nullptr;   // f32 [32]
    const ggml_tensor * ssm_a = nullptr;     // f32 [32]
    const ggml_tensor * ssm_norm = nullptr;  // f32 [128]
    const ggml_tensor * s_copy = nullptr;    // I32 [1] source row of both recurrent states
    const float * conv_base = nullptr; size_t conv_stride = 0;   // get_rows source (row = s_copy[0]) and row stride in floats
    const float * ssm_base  = nullptr; size_t ssm_stride  = 0;
    float * conv_out = nullptr;              // cpy destination of the new conv state [3, 8192]
    float * state_out = nullptr;             // cpy destination of the new SSM state [128,128,32]
    float * zero_conv = nullptr;             // build_rs state_zero rows (rs_zero >= 0): zeroed before the kernel reads the states
    float * zero_ssm = nullptr;
    ggml_tensor * out = nullptr;             // residual ADD output [2048]
};

bool ggml_cuda_occamy_gdn_enabled();
int  ggml_cuda_occamy_gdn_check_budget();   // remaining CHECK windows (decrements)
bool ggml_cuda_occamy_gdn_match_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, ggml_cuda_occamy_gdn_match & m);
void ggml_cuda_occamy_gdn_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m, float * out_override);
