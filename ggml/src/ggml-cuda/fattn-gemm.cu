// 7 Nov 2026: prefill flash attention via cuBLAS GEMMs (sm_52 long shot). Env GGML_CUDA_FA_GEMM=1 (default off).
// Per KV head g, per key chunk of KC: S = scale*K^T Q (f16 in, f32 acc) -> fused mask+online softmax -> P f16 -> O += V P.
// ubench (native-occamy/fa-gemm-20261107): 1.8x vs flash_attn_tile<256> at 4k-64k keys, nq=1024.
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-gemm.cuh"
#include <cstdlib>

static int fag_env(const char * n, int d) { const char * e = getenv(n); return e ? atoi(e) : d; }
int ggml_cuda_fa_gemm_mode() { static const int m = fag_env("GGML_CUDA_FA_GEMM", 0); return m; }
static int fag_minq() { static const int m = fag_env("GGML_CUDA_FA_GEMM_MINQ", 64); return m; }
static int fag_s16() { static const int m = fag_env("GGML_CUDA_FA_GEMM_S16", 0); return m; }
static int fag_kc()   { static const int m = fag_env("GGML_CUDA_FA_GEMM_KC", 4096); return m; }

constexpr int FAG_D = 256;

bool ggml_cuda_fa_gemm_applicable(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if ((K->type != GGML_TYPE_F16 && !ggml_get_to_fp16_cuda(K->type)) || (V->type != GGML_TYPE_F16 && !ggml_get_to_fp16_cuda(V->type))) return false;
    if (Q->ne[0] != FAG_D || K->ne[0] != FAG_D || V->ne[0] != FAG_D) return false;
    if (Q->ne[1] < fag_minq()) return false;
    if (K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0) return false;
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || K->ne[1] != V->ne[1]) return false;
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || Q->ne[3] % mask->ne[3] != 0)) return false;
    if (dst->src[4] || max_bias != 0.0f || softcap != 0.0f) return false;
    if (!ggml_is_contiguous(dst)) return false;
    const int kc = fag_kc(); if (kc < 256 || kc > 4096 || kc % 256) return false;
    return true;
}

// Q f32 strided -> Qh [q][h][d] half
static __global__ void fag_q2h(const char * __restrict__ Q, half * __restrict__ Qh, int H, size_t nb1, size_t nb2) {
    const int q = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    Qh[((size_t) q*H + h)*FAG_D + d] = __float2half(((const float *) (Q + q*nb1 + h*nb2))[d]);
}

static __device__ __forceinline__ float fag_ld(float x) { return x; }
static __device__ __forceinline__ float fag_ld(half x) { return __half2float(x); }

// one block per (query, head-in-group); S/P rows [hg][q][KC]; O = dst [q][H][D]
template <typename ST>
static __global__ void __launch_bounds__(256) fag_smax(const ST * __restrict__ S, half * __restrict__ P, const half * __restrict__ mask,
        size_t mrow, float * __restrict__ O, float2 * __restrict__ ml, int nq, int H, int h0, int KCs, int kc, int c0, int first) {
    const int q = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x;
    const ST * s = S + ((size_t) hg*nq + q)*KCs;
    half * p = P + ((size_t) hg*nq + q)*KCs;
    const half * mk = mask ? mask + (size_t) q*mrow + c0 : nullptr;
    __shared__ float red[8];
    float v[16]; float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) { const int k = i*256 + tid; float x = -INFINITY;
        if (k < kc) x = fag_ld(s[k]) + (mk ? __half2float(mk[k]) : 0.0f); v[i] = x; mx = fmaxf(mx, x); }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if ((tid & 31) == 0) red[tid >> 5] = mx; __syncthreads();
    mx = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float2 * mlp = ml + (size_t) hg*nq + q;
    const float2 mlo = first ? make_float2(-INFINITY, 0.0f) : *mlp;
    const float mn = fmaxf(mlo.x, mx);
    const float alpha = mn == -INFINITY ? 1.0f : __expf(mlo.x - mn);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { const int k = i*256 + tid; if (k < KCs) { const float e = (k < kc && mn != -INFINITY) ? __expf(v[i] - mn) : 0.0f;
        sum += e; p[k] = __float2half(e); } }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) red[tid >> 5] = sum; __syncthreads();
    if (tid == 0) { float t = 0; for (int i = 0; i < 8; ++i) t += red[i]; *mlp = make_float2(mn, mlo.y*alpha + t); }
    if (!first) O[((size_t) q*H + h0 + hg)*FAG_D + tid] *= alpha;
}

static __global__ void fag_norm(float * O, const float2 * ml, int nq, int H, int h0) {
    const int q = blockIdx.x, hg = blockIdx.y; const float l = ml[(size_t) hg*nq + q].y;
    O[((size_t) q*H + h0 + hg)*FAG_D + threadIdx.x] *= (l > 0.0f ? 1.0f/l : 0.0f);
}

static const half * fag_to_f16(const ggml_tensor * T, uintptr_t buf, size_t nb[4], cudaStream_t st) {
    nb[1] = T->nb[1]; nb[2] = T->nb[2]; nb[3] = T->nb[3];
    if (T->type == GGML_TYPE_F16) return (const half *) T->data;
    const size_t bs = ggml_blck_size(T->type), ts = ggml_type_size(T->type);
    GGML_ASSERT(buf != 0);
    half * o = (half *) buf;
    if (ggml_is_contiguously_allocated(T)) {
        ggml_get_to_fp16_cuda(T->type)(T->data, o, ggml_nelements(T), st);
        for (int i = 1; i < 4; ++i) nb[i] = nb[i]*bs*sizeof(half)/ts;
    } else {
        GGML_ASSERT(T->nb[0] == ts);
        ggml_get_to_fp16_nc_cuda(T->type)(T->data, o, T->ne[0], T->ne[1], T->ne[2], T->ne[3], nb[1]/ts, nb[2]/ts, nb[3]/ts, st);
        nb[1] = T->ne[0]*sizeof(half); nb[2] = T->ne[1]*nb[1]; nb[3] = T->ne[2]*nb[2];
    }
    return o;
}

void ggml_cuda_fa_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t st = ctx.stream();
    cublasHandle_t hb = ctx.cublas_handle();
    float scale; memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // f16 K/V in the stock extra region after dst (sized for the tile kernel's need_f16_K/V)
    const ggml_cuda_flash_attn_ext_f16_extra_data ex = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    size_t kb[4], vb[4];
    const half * Kh = fag_to_f16(K, ex.K, kb, st);
    const half * Vh;
    if (V_is_K_view && V->type != GGML_TYPE_F16) { Vh = Kh; vb[1] = kb[1]; vb[2] = kb[2]; vb[3] = kb[3]; }
    else Vh = fag_to_f16(V, ex.V, vb, st);
    GGML_ASSERT(kb[1] % 2 == 0 && vb[1] % 2 == 0 && kb[2] % 2 == 0 && vb[2] % 2 == 0);

    const int nq = Q->ne[1], H = Q->ne[2], HK = K->ne[2], G = H/HK, nk = K->ne[1], ns = Q->ne[3];
    const int KC = std::min(fag_kc(), GGML_PAD(nk, 256));
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half>   Qh(pool, (size_t) nq*H*FAG_D);
    const bool s16 = fag_s16() != 0;
    ggml_cuda_pool_alloc<float>  S (pool);
    ggml_cuda_pool_alloc<half>   S16(pool);
    if (s16) S16.alloc((size_t) G*nq*KC); else S.alloc((size_t) G*nq*KC);
    ggml_cuda_pool_alloc<half>   P (pool, (size_t) G*nq*KC);
    ggml_cuda_pool_alloc<float2> ml(pool, (size_t) G*nq);
    const float one = 1.0f, zero = 0.0f;
    const size_t mrow = mask ? mask->nb[1]/sizeof(half) : 0;

    for (int s3 = 0; s3 < ns; ++s3) {
        fag_q2h<<<dim3(nq, H), FAG_D, 0, st>>>((const char *) Q->data + s3*Q->nb[3], Qh.get(), H, Q->nb[1], Q->nb[2]);
        float * O = (float *) ((char *) dst->data + s3*dst->nb[3]);
        const half * mk = mask ? (const half *) ((const char *) mask->data + (s3 % mask->ne[3])*mask->nb[3]) : nullptr;
        const char * Kb = (const char *) Kh + s3*kb[3];
        const char * Vb = (const char *) Vh + s3*vb[3];
        for (int g = 0; g < HK; ++g) {
            const int h0 = g*G;
            for (int c0 = 0, ci = 0; c0 < nk; c0 += KC, ++ci) {
                const int kc = std::min(KC, nk - c0);
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, kc, nq, FAG_D, &scale,
                    Kb + g*kb[2] + (size_t) c0*kb[1], CUDA_R_16F, kb[1]/2, 0,
                    Qh.get() + (size_t) h0*FAG_D, CUDA_R_16F, H*FAG_D, FAG_D, &zero,
                    s16 ? (void *) S16.get() : (void *) S.get(), s16 ? CUDA_R_16F : CUDA_R_32F, KC, (long long) nq*KC, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                if (s16) fag_smax<half><<<dim3(nq, G), 256, 0, st>>>(S16.get(), P.get(), mk, mrow, O, ml.get(), nq, H, h0, KC, kc, c0, ci == 0);
                else fag_smax<float><<<dim3(nq, G), 256, 0, st>>>(S.get(), P.get(), mk, mrow, O, ml.get(), nq, H, h0, KC, kc, c0, ci == 0);
                const float beta = ci == 0 ? 0.0f : 1.0f;
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, FAG_D, nq, kc, &one,
                    Vb + g*vb[2] + (size_t) c0*vb[1], CUDA_R_16F, vb[1]/2, 0,
                    P.get(), CUDA_R_16F, KC, (long long) nq*KC, &beta,
                    O + (size_t) h0*FAG_D, CUDA_R_32F, H*FAG_D, FAG_D, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
            fag_norm<<<dim3(nq, G), FAG_D, 0, st>>>(O, ml.get(), nq, H, h0);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
