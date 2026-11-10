// Track F (native-occamy, 12 Oct 2026): GQA-aware split-K flash-attention DECODE kernel for sm_52.
// Shape it is built for: D = 256, 8 q-heads per kv-head, f16 K/V, f32 Q, one query token per stream.
// Env gate (default off): GGML_CUDA_OCCAMY_FA=1 (or GGML_CUDA_NATIVE_FA_SM52=1) -> use this kernel when applicable;
//                         GGML_CUDA_OCCAMY_FA=2 -> run this kernel AND the stock kernel, log max abs/rel diff, keep stock output.
// Knobs: GGML_CUDA_OCCAMY_FA_WAVES (blocks = waves * resident blocks, default 1).
// Design: one block = (kv-head, stream, KV split); 4 warps; each warp streams 4 KV tokens per step,
// lane l owns dims [8l, 8l+8) of a 256-dim row (one 16-B LDG per row), all 8 q-heads of that kv-head live
// in registers (64 fp32) and so does the output accumulator (64 fp32). K/V are read exactly once per block,
// converted f16 -> f32 (no half math). QK partials (4 tokens x 8 heads = 32 per lane) are reduce-scattered
// across the warp so lane l ends with score (token l>>3, head l&7). Partials (m, l, o) per block -> combine.
#include "common.cuh"
#include "fattn-occamy.cuh"

#include <cstdio>
#include <cstdlib>
#include <cfloat>

#define OFA_D   256
#define OFA_G   8
#define OFA_NW  4
#define OFA_T   4
#define OFA_NT  (OFA_NW*32)

static __device__ __forceinline__ void ofa_cvt8(const uint4 w, float * f) {
    const half2 * h = (const half2 *) &w;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h[i]);
        f[2*i + 0] = t.x;
        f[2*i + 1] = t.y;
    }
}

template <int n>
static __device__ __forceinline__ void ofa_rs_step(float * s, const int lane) {
    const bool up = lane & n;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const float send = up ? s[i] : s[i + n];
        const float keep = up ? s[i + n] : s[i];
        s[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, n, 32);
    }
}

struct ofa_params {
    const char * Q; const char * K; const char * V; const char * mask;
    float  * part_o;   // [nblk][8][256]
    float2 * part_ml;  // [nblk][8]
    int n_kv, nsplit, nkvh, mask_ne1, mask_ne3;
    float scale_log2;  // scale * log2(e)
    size_t nbq1, nbq2, nbq3;
    size_t nbk1, nbk2, nbk3;
    size_t nbv1, nbv2, nbv3;
    size_t nbm1, nbm3;
};

__global__ void __launch_bounds__(OFA_NT, 2) ofa_decode_kernel(const ofa_params p) {
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;

    __shared__ float  s_p[OFA_NW][32];
    __shared__ float  s_a[OFA_NW][OFA_G];
    __shared__ float  s_o[OFA_NW][OFA_G][OFA_D];
    __shared__ float  s_m[OFA_NW][OFA_G];
    __shared__ float  s_l[OFA_NW][OFA_G];

    // q for the 8 heads, my 8 dims, pre-scaled by scale*log2(e)
    float q[OFA_G][8];
#pragma unroll
    for (int h = 0; h < OFA_G; ++h) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(kvh*OFA_G + h)*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[h][0] = a.x*p.scale_log2; q[h][1] = a.y*p.scale_log2; q[h][2] = a.z*p.scale_log2; q[h][3] = a.w*p.scale_log2;
        q[h][4] = b.x*p.scale_log2; q[h][5] = b.y*p.scale_log2; q[h][6] = b.z*p.scale_log2; q[h][7] = b.w*p.scale_log2;
    }
    float o[OFA_G][8];
#pragma unroll
    for (int h = 0; h < OFA_G; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }
    const int   hl = lane & 7;       // head whose score this lane owns after the reduce-scatter
    const int   jl = lane >> 3;      // token (within the step) whose score this lane owns
    float m_run = -FLT_MAX/2.0f;     // running max for head hl (log2 domain), same on the 4 lanes of a head
    float l_run = 0.0f;              // per-lane partial of the softmax denominator for head hl

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3);

    const int nunits = p.n_kv / OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    for (int u = u0 + warp; u < u1; u += OFA_NW) {
        const int t0 = u*OFA_T;
        uint4 kr[OFA_T], vr[OFA_T];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) kr[j] = __ldg((const uint4 *) (Kb + (size_t)(t0 + j)*p.nbk1));
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) vr[j] = __ldg((const uint4 *) (Vb + (size_t)(t0 + j)*p.nbv1));
        const float mk = __half2float(Mb[t0 + jl]);

        float s[OFA_T*OFA_G];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float k[8];
            ofa_cvt8(kr[j], k);
#pragma unroll
            for (int h = 0; h < OFA_G; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*OFA_G + h] = acc;
            }
        }
        // reduce-scatter 32 values over 32 lanes: lane l ends with the full sum of value index l
        ofa_rs_step<16>(s, lane);
        ofa_rs_step< 8>(s, lane);
        ofa_rs_step< 4>(s, lane);
        ofa_rs_step< 2>(s, lane);
        ofa_rs_step< 1>(s, lane);
        const float sc = s[0] + mk*1.4426950408889634f;   // mask is -inf for masked cells
        float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
        const float m_new = fmaxf(m_run, gm);
        const float alpha = exp2f(m_run - m_new);
        const float pr    = exp2f(sc - m_new);
        m_run = m_new;
        l_run = l_run*alpha + pr;
        s_p[warp][lane] = pr;
        if (lane < OFA_G) s_a[warp][lane] = alpha;
        __syncwarp();
        float a[OFA_G];
        {
            const float4 a0 = *(const float4 *) &s_a[warp][0];
            const float4 a1 = *(const float4 *) &s_a[warp][4];
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w; a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
        }
#pragma unroll
        for (int h = 0; h < OFA_G; ++h) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float v[8];
            ofa_cvt8(vr[j], v);
            const float4 p0 = *(const float4 *) &s_p[warp][j*OFA_G + 0];
            const float4 p1 = *(const float4 *) &s_p[warp][j*OFA_G + 4];
            const float pj[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int h = 0; h < OFA_G; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    // warp -> smem
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < OFA_G) { s_m[warp][lane] = m_run; s_l[warp][lane] = l_run; }
#pragma unroll
    for (int h = 0; h < OFA_G; ++h) {
        float4 * dp = (float4 *) &s_o[warp][h][8*lane];
        dp[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
        dp[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
    }
    __syncthreads();

    // block combine across warps -> one partial per block
    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
    float * po = p.part_o + (size_t) blk*OFA_G*OFA_D;
    for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += OFA_NT) {
        const int h = idx / OFA_D, d = idx % OFA_D;
        float M = s_m[0][h];
#pragma unroll
        for (int w = 1; w < OFA_NW; ++w) M = fmaxf(M, s_m[w][h]);
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < OFA_NW; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][h] - M);
        po[idx] = acc;
        if (d == 0) {
            float L = 0.0f;
#pragma unroll
            for (int w = 0; w < OFA_NW; ++w) L += s_l[w][h]*exp2f(s_m[w][h] - M);
            p.part_ml[(size_t) blk*OFA_G + h] = make_float2(M, L);
        }
    }
}

// grid (nkvh*8 q-heads, ns), block 256 = one thread per dim
__global__ void __launch_bounds__(OFA_D) ofa_combine_kernel(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
        char * __restrict__ dst, const int nsplit, const int nkvh, const size_t nbd1, const size_t nbd3) {
    const int qh = blockIdx.x, i3 = blockIdx.y, d = threadIdx.x;
    const int kvh = qh / OFA_G, h = qh % OFA_G;
    const int blk0 = (i3*nkvh + kvh)*nsplit;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, part_ml[(size_t)(blk0 + s)*OFA_G + h].x);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float2 ml = part_ml[(size_t)(blk0 + s)*OFA_G + h];
        const float  w  = exp2f(ml.x - M);
        L   += ml.y*w;
        acc += part_o[((size_t)(blk0 + s)*OFA_G + h)*OFA_D + d]*w;
    }
    float * out = (float *) (dst + i3*nbd3 + qh*nbd1);
    out[d] = L > 0.0f ? acc/L : 0.0f;
}

// compare kernel: res[0] = max|a-b| (as uint bits), res[1] = max|b|, res[2] = max rel (|a-b|/max(|b|,1e-2)), res[3] = nan count
__global__ void ofa_compare_kernel(const float * a, const float * b, const int n, unsigned int * res) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const float x = a[i], y = b[i];
        if (!(x == x) || !(y == y)) { atomicAdd(&res[3], 1u); continue; }
        const float e = fabsf(x - y);
        atomicMax(&res[0], __float_as_uint(e));
        atomicMax(&res[1], __float_as_uint(fabsf(y)));
        atomicMax(&res[2], __float_as_uint(e/fmaxf(fabsf(y), 1e-2f)));
    }
}

int ggml_cuda_occamy_fa_mode() {
    static const int v = []() {
        const char * s = getenv("GGML_CUDA_OCCAMY_FA");
        int r = s ? atoi(s) : 0;
        if (r == 0) {
            const char * s2 = getenv("GGML_CUDA_NATIVE_FA_SM52");
            r = (s2 && atoi(s2) != 0) ? 1 : 0;
        }
        if (r) fprintf(stderr, "ggml_cuda: occamy FA decode kernel mode %d\n", r);
        return r;
    }();
    return v;
}

bool ggml_cuda_occamy_fa_applicable(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) return false;
    if (K->ne[0] != OFA_D || V->ne[0] != OFA_D || Q->ne[0] != OFA_D) return false;
    if (K->ne[2] != V->ne[2] || Q->ne[2] != OFA_G*K->ne[2]) return false;
    if (Q->ne[1] != 1) return false;
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || K->ne[1] != V->ne[1]) return false;
    if (K->ne[1] % OFA_T != 0) return false;
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] < K->ne[1]) return false;
    if (Q->ne[3] % mask->ne[3] != 0) return false;
    if (sinks || max_bias != 0.0f || softcap != 0.0f) return false;
    if (Q->nb[0] != 4 || K->nb[0] != 2 || V->nb[0] != 2 || dst->nb[0] != 4) return false;
    if ((uintptr_t) Q->data % 16 || (uintptr_t) K->data % 16 || (uintptr_t) V->data % 16 || (uintptr_t) mask->data % 8) return false;
    for (int i = 1; i < 4; ++i) {
        if (Q->nb[i] % 16 || K->nb[i] % 16 || V->nb[i] % 16) return false;
    }
    return true;
}

static void ofa_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, char * out) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();

    static int resident = 0;
    static int waves = 1;
    if (resident == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ofa_decode_kernel, OFA_NT, 0));
        resident = std::max(1, nb) * ggml_cuda_info().devices[id].nsm;
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_WAVES");
        if (s) waves = std::max(1, atoi(s));
    }
    const int nkvh = (int) K->ne[2];
    const int ns   = (int) Q->ne[3];
    const int n_kv = (int) K->ne[1];
    int nsplit = (resident*waves + nkvh*ns - 1) / (nkvh*ns);
    nsplit = std::max(1, std::min(nsplit, n_kv / (OFA_T*OFA_NW*4)));  // >= 64 tokens per block

    const int nblk = nsplit*nkvh*ns;
    ggml_cuda_pool_alloc<float>  part_o (ctx.pool(), (size_t) nblk*OFA_G*OFA_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) nblk*OFA_G);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ofa_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data; p.mask = (const char *) mask->data;
    p.part_o = part_o.get(); p.part_ml = part_ml.get();
    p.n_kv = n_kv; p.nsplit = nsplit; p.nkvh = nkvh; p.mask_ne1 = (int) mask->ne[1]; p.mask_ne3 = (int) mask->ne[3];
    p.scale_log2 = scale*1.4426950408889634f;
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];

    ofa_decode_kernel<<<dim3(nsplit, nkvh, ns), OFA_NT, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
    ofa_combine_kernel<<<dim3(nkvh*OFA_G, ns), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), out, nsplit, nkvh, dst->nb[1], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_occamy_fa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ofa_launch(ctx, dst, (char *) dst->data);
}

// mode 2: our result into a temp buffer, stock result already in dst; compare and log.
void ggml_cuda_occamy_fa_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static int ncalls = 0;
    ++ncalls;
    const bool log_it = ncalls <= 40 || ncalls % 200 == 0;
    if (!log_it) return;
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int n = (int) ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n);
    // dst is [D, nh, 1, ns] contiguous: our launch writes with dst strides into tmp
    ofa_launch(ctx, dst, (char *) tmp.get());
    ggml_cuda_pool_alloc<unsigned int> res(ctx.pool(), 4);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(res.get(), 0, 4*sizeof(unsigned int), stream));
    ofa_compare_kernel<<<16, 256, 0, stream>>>(tmp.get(), (const float *) dst->data, n, res.get());
    unsigned int h[4];
    CUDA_CHECK(cudaMemcpyAsync(h, res.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float f[3];
    memcpy(f, h, sizeof(f));
    fprintf(stderr, "occamy_fa_cmp call=%d n_kv=%lld ns=%lld max_abs=%.3e max_ref=%.3e max_rel=%.3e nan=%u\n",
        ncalls, (long long) dst->src[1]->ne[1], (long long) dst->src[0]->ne[3], f[0], f[1], f[2], h[3]);
}
