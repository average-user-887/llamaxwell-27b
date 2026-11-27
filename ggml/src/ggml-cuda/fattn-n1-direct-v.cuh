// Standalone single-shape implementation. Source generation pins are in stage.py.
// Included after the stock QK definitions in the D256 tile TU.

// Pinned SM52 SASS rounds weight*value before fusing alpha*old on each tile's first key.
static __device__ __forceinline__ float direct_v_first(float value, float weight, float alpha, float old) {
    float first;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(first) : "f"(weight), "f"(value));
    float out;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(alpha), "f"(old), "f"(first));
    return out;
}

static __device__ __forceinline__ void direct_v_tile(
        float *Q_tmp, const half2 *__restrict__ K_h2, const half2 *__restrict__ V_h2,
        const half *__restrict__ mask, uint3 ne01, float logit_softcap, float slope,
        float (*weights)[2], float *alphas, float *KV_tmp, int stride_K2, int stride_V2,
        int stride_mask, float *KQ_max, float *KQ_sum, float (&acc)[2][4],
        int k_VKQ_0, int k_VKQ_max, int col_Q_0) {
    constexpr int ncols1=1, ncols2=2, nwarps=4, warp_size=32;
    constexpr int DKQ=256, DV=256, nbatch_fa=64, nbatch_K=64;
    constexpr bool use_logit_softcap=false, oob_check=false;
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

    // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
#ifdef FAST_FP16_AVAILABLE
    constexpr int KQ_cs = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
#else
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
#endif // FAST_FP16_AVAILABLE
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int j = fastmodulo(col_Q_0 + (jc0 + (threadIdx.y / np)*cpw)/ncols2, ne01);

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
            // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
            // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
            KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0] *= 4.0f;
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)

            if (use_logit_softcap) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * tanhf(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        KQ_max_new[jc0] = warp_reduce_max<warp_size>(KQ_max_new[jc0]);
    }

    if constexpr (np == 1) {
        __syncthreads();
    } else {
        static_assert(cpw == 1, "bad cpw");
        __shared__ float KQ_max_new_shared[nwarps];
        if (threadIdx.x == 0) {
            KQ_max_new_shared[threadIdx.y] = KQ_max_new[0];
        }
        __syncthreads();
        KQ_max_new[0] = KQ_max_new_shared[(threadIdx.y & ~(np-1)) + threadIdx.x % np];
        KQ_max_new[0] = warp_reduce_max<np>(KQ_max_new[0]);
    }

    {
            constexpr int jc = 0;
            const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                    expf(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                KQ_sum_add += val;
                weights[(threadIdx.y % np)*warp_size + threadIdx.x][threadIdx.y/np] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

            if (threadIdx.x == 0 && (threadIdx.y & 1) == 0) {
                alphas[threadIdx.y/2] = KQ_max_scale;
            }
    }
    __syncthreads();

    const int parity = threadIdx.y & 1;
    const int d = 128*(threadIdx.y >> 1) + 4*threadIdx.x;
    const half2 *v_base = V_h2 + int64_t(k_VKQ_0 + parity)*stride_V2 + d/2;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        __align__(8) half2 frag[2];
        ggml_cuda_memcpy_1<8>(frag, v_base + (2*i)*stride_V2);
        const float2 v01 = __half22float2(frag[0]);
        const float2 v23 = __half22float2(frag[1]);
        float2 weight;
        ggml_cuda_memcpy_1<8>(&weight, weights[2*i + parity]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float w = h == 0 ? weight.x : weight.y;
            if (i == 0) {
                acc[h][0] = direct_v_first(v01.x, w, alphas[h], acc[h][0]);
                acc[h][1] = direct_v_first(v01.y, w, alphas[h], acc[h][1]);
                acc[h][2] = direct_v_first(v23.x, w, alphas[h], acc[h][2]);
                acc[h][3] = direct_v_first(v23.y, w, alphas[h], acc[h][3]);
            } else {
                acc[h][0] += v01.x * w;
                acc[h][1] += v01.y * w;
                acc[h][2] += v23.x * w;
                acc[h][3] += v23.y * w;
            }
        }
    }
}

extern "C" __attribute__((visibility("hidden"))) __global__ __launch_bounds__(128,3) void n1_direct_v(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    constexpr int ncols1=1, ncols2=2, DKQ=256, DV=256, tune=0;
    static_assert(ggml_cuda_fattn_tile_get_nthreads(DKQ,DV,2,tune) == 128);
    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2, tune) / warp_size; // local patch 20261012 fattn: tune
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2, tune);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2, tune);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int col_Q_0 = blockIdx.x * ncols1; // Index of the first Q column for this CUDA block to work on.

    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0 = blockIdx.z*ncols2 - sequence*ne02; // == blockIdx.z % (ne02/ncols2)
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33)) : nullptr;

    const int stride_K2   = nb11 / sizeof(half2);
    const int stride_V2   = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(cpw == 1 || np == 1, "bad cpw / np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    static_assert(nbatch_fa == 64 && nbatch_K == 64 && cpy_ne == 2);
    __shared__ float Q_tmp[ncols*DKQ];
    __shared__ float KV_tmp[nbatch_fa*(nbatch_K+cpy_ne)];
    __shared__ __align__(8) float weights[64][2];
    __shared__ float alphas[2];
    float acc[2][4] = {};
    __align__(16) float2 VKQ[4];
    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    ggml_cuda_pdl_sync();

    // Load Q data, convert to FP16 if fast:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            if (i0 + np*warp_size*cpy_ne_D <= DKQ || i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D < DKQ) {
                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                ggml_cuda_memcpy_1<sizeof(tmp_f)>
                    (tmp_f, &Q_f[c*(nb02/sizeof(float)) + fastmodulo(col_Q_0 + j, ne01)*(nb01/sizeof(float))
                                 + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

#ifdef FAST_FP16_AVAILABLE
                __align__(16) half2 tmp_h2[cpy_ne_D/2];
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                    tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                    // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
                    // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
                    tmp_h2[i1/2] *= make_half2(0.25f, 0.25f);
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
                }
                ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                    &Q_tmp[jc*(DKQ/2) + i0/2 + (threadIdx.y % np)*(warp_size*cpy_ne_D/2) + threadIdx.x*(cpy_ne_D/2)],
                    tmp_h2);
#else
                ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (threadIdx.y % np)*(warp_size*cpy_ne_D)   + threadIdx.x* cpy_ne_D],
                    tmp_f);
#endif // FAST_FP16_AVAILABLE
            }
        }
    }

    __syncthreads();

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int base = blockIdx.y*nbatch_fa; base < k_VKQ_max; base += gridDim.y*nbatch_fa) {
        direct_v_tile(Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope,
                      weights, alphas, KV_tmp, stride_K2, stride_V2, stride_mask,
                      KQ_max, KQ_sum, acc, base, k_VKQ_max, col_Q_0);
    }

    const int parity = threadIdx.y & 1;
    const int d = 128*(threadIdx.y >> 1) + 4*threadIdx.x;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            KV_tmp[(2*h+parity)*DV + d+j] = acc[h][j];
        }
    }
    KQ_sum[0] = warp_reduce_sum<warp_size>(KQ_sum[0]);
    if (parity && threadIdx.x == 0) {
        Q_tmp[threadIdx.y] = KQ_sum[0];
    }
    __syncthreads();
    if (parity) {
        return;
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int pos = threadIdx.y*DV + 64*b + 2*threadIdx.x;
        VKQ[b].x = KV_tmp[pos] + KV_tmp[pos+DV];
        VKQ[b].y = KV_tmp[pos+1] + KV_tmp[pos+DV+1];
    }
    KQ_sum[0] += Q_tmp[threadIdx.y+1];

    // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y/np)*cpw;
            const float sink = ((const float *) sinks)[head0 + jc % ncols2];

            float KQ_max_new_j = fmaxf(KQ_max[jc0], sink);
            const float KQ_max_scale = expf(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = expf(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0]*KQ_max_scale + val;

#ifdef FAST_FP16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        if (ncols1 > 1 && col_Q_0 + j >= int(ne01.z)) {
            return;
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*int(ne01.z) + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (DVp/2)/warp_size ? cpy_ne/2 : (DVp/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp/2; i0 += warp_size*cpy_ne_D) {
            __align__(16) float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = __half22float2(VKQ[jc0*((DVp/2)/warp_size) + i0/warp_size + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + warp_size*cpy_ne_D <= DV/2 || i0 + threadIdx.x*cpy_ne_D < DV/2) {
                ggml_cuda_memcpy_1<sizeof(tmp)>(&dst[j_dst_unrolled*DV + 2*i0 + threadIdx.x*(2*cpy_ne_D)], tmp);
            }
        }
#else
        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + threadIdx.x*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + threadIdx.x*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }
#endif // FAST_FP16_AVAILABLE

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

