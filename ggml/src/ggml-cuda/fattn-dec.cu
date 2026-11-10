// 8 Nov fadec design 2 (GGML_CUDA_FA_DEC=2): GQA-grouped split-K decode flash attention, q4_0 K/V, D=256, sm_52.
// One block = (KV split, kv-head, stream) and serves all G q-heads x NR query rows (NQ = G*NR <= 24) of that kv-head,
// so every K/V byte is read once per step and dequantized in registers (no f16 copy of the cache).
// Tiles of 128 KV tokens are interleaved across splits. Per tile:
//   A: QK. AV=1: thread = 1 token x all NQ queries; AV=2: thread = 2 tokens x NQ/2 queries (half the LDS per FMA).
//      K row nibbles -> fp32 once, dot against Q (pre-scaled by scale*log2e) broadcast from smem, x fp16 scale per block.
//   B: warp-per-query online softmax over the tile (log2 domain) -> P, m, l, alpha in smem.
//   C: PV. thread owns dims {b*32+j, b*32+16+j} (one q4_0 byte per token), acc[NQ][2] in registers.
// Epilogue: per-block partials (o, m, l) -> combine kernel writes dst.
// Env: GGML_CUDA_FA_DEC=2 enables; GGML_CUDA_FA_DEC_MINQ (default 3) min query rows; GGML_CUDA_FA_DEC_SPLIT forces nsplit;
//      GGML_CUDA_FA_DEC_DEBUG=1 counts/logs launches (stderr, and the total at exit).
#ifndef FD_STANDALONE
#include "common.cuh"
#include "fattn-dec.cuh"
#endif
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#define FD_D    256
#define FD_NT   128
#define FD_TILE 128
#ifndef FD_AV
#define FD_AV   3
#endif
#ifndef FD_MINB
#define FD_MINB 2
#endif
#ifndef FD_QOFF
#define FD_QOFF(q) (q)
#endif
#ifndef FD_ILP
#define FD_ILP  0
#endif

struct fd_params {
    const char * Q; const char * K; const char * V; const char * mask;
    float  * part_o;   // [nblk][NQ][256]
    float2 * part_ml;  // [nblk][NQ]
    int ntiles, nsplit, nkvh, mask_ne3;
    float scale_log2;
    size_t nbq1, nbq2, nbq3;
    size_t nbk1, nbk2, nbk3;
    size_t nbv1, nbv2, nbv3;
    size_t nbm1, nbm3;
};

// w = the 9 words (36 B) holding q4_0 blocks 2p and 2p+1 of a row; ODD selects the second block.
// kf[e] = q_e - 8 for the block's 32 elements in natural order; returns the block scale.
template <bool ODD>
static __device__ __forceinline__ float fd_unpack(const uint32_t * w, float * kf) {
    uint32_t qw[4];
    float d;
    if (ODD) {
        d = __half2float(__ushort_as_half((unsigned short) (w[4] >> 16)));
#pragma unroll
        for (int k = 0; k < 4; ++k) qw[k] = w[5 + k];
    } else {
        d = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFFu)));
#pragma unroll
        for (int k = 0; k < 4; ++k) qw[k] = __byte_perm(w[k], w[k + 1], 0x5432);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t lo = qw[k] & 0x0F0F0F0Fu;
        const uint32_t hi = (qw[k] >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kf[4*k + i]      = __int_as_float(__byte_perm(lo, 0x4B000000u, 0x7540 | i)) - 8388616.0f;
            kf[16 + 4*k + i] = __int_as_float(__byte_perm(hi, 0x4B000000u, 0x7540 | i)) - 8388616.0f;
        }
    }
    return d;
}

static __device__ __forceinline__ float fd_dot32(const float * kf, const float4 * qp) {
    float s = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float4 f = qp[e];
        s = fmaf(kf[4*e + 0], f.x, s);
        s = fmaf(kf[4*e + 1], f.y, s);
        s = fmaf(kf[4*e + 2], f.z, s);
        s = fmaf(kf[4*e + 3], f.w, s);
    }
    return s;
}

template <int G, int NR>
static __global__ void __launch_bounds__(FD_NT, FD_MINB) fd_kernel(const fd_params p) {
    constexpr int NQ  = G*NR;
    constexpr int NQH = FD_AV == 2 ? NQ/2 : NQ;   // queries per thread in phase A
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int seq   = blockIdx.z;
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;

    __shared__ __align__(16) float sQ[NQ][FD_D];
    __shared__ __align__(16) float sS[NQ][FD_TILE];
    __shared__ float sM[NQ], sL[NQ], sA[NQ];

    for (int idx = tid; idx < NQ*FD_D; idx += FD_NT) {
        const int q = idx / FD_D, d = idx % FD_D;
        const int r = q / G, g = q % G;
        const float * qr = (const float *) (p.Q + seq*p.nbq3 + r*p.nbq1 + (size_t) (kvh*G + g)*p.nbq2);
        sQ[q][d] = qr[d]*p.scale_log2;
    }
    if (tid < NQ) { sM[tid] = -FLT_MAX/2.0f; sL[tid] = 0.0f; }

    const char * Kb = p.K + seq*p.nbk3 + kvh*p.nbk2;
    const int    vb = tid >> 4, vj = tid & 15;          // phase C: q4_0 block and byte owned
    const char * Vb = p.V + seq*p.nbv3 + kvh*p.nbv2 + 18*vb;
    const half * Mb = (const half *) (p.mask + (seq % p.mask_ne3)*p.nbm3);

    // phase A ownership
    const int q0 = FD_AV == 2 ? (tid >> 6)*NQH : 0;
    const int tA = FD_AV == 2 ? (tid & 63) : tid;

    float o[NQ][2];
#pragma unroll
    for (int q = 0; q < NQ; ++q) { o[q][0] = 0.0f; o[q][1] = 0.0f; }
    __syncthreads();

    for (int tile = split; tile < p.ntiles; tile += p.nsplit) {
        const int t0 = tile*FD_TILE;
        // ---- mask + skip
        float mkA[NR], mkB[NR];
        bool any = false;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const half * mr = (const half *) ((const char *) Mb + r*p.nbm1);
            mkA[r] = __half2float(mr[t0 + tA]);
            any |= mkA[r] != -INFINITY;
            if (FD_AV == 2) {
                mkB[r] = __half2float(mr[t0 + tA + 64]);
                any |= mkB[r] != -INFINITY;
            }
        }
        if (!__syncthreads_or(any)) continue;

        // ---- phase A: scores
#ifdef FD_SKIP_A
#pragma unroll
        for (int q = 0; q < NQ; ++q) sS[q][tA] = mkA[q / G];
        if (FD_AV == 2) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) sS[q][tA + 64] = mkB[q / G];
        }
#else
        if (FD_AV == 3) {
            // low-register variant: one block pair (9 words) at a time, double-buffered; one qs word (8 elements) unpacked at a time
            const uint32_t * kr = (const uint32_t *) (Kb + (size_t) (t0 + tA)*p.nbk1);
            float acc[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) acc[q] = 0.0f;
            uint32_t wn[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) wn[i] = __ldg(kr + i);
#pragma unroll 1
            for (int pp = 0; pp < 4; ++pp) {
                uint32_t w[9];
#pragma unroll
                for (int i = 0; i < 9; ++i) w[i] = wn[i];
                if (pp < 3) {
#pragma unroll
                    for (int i = 0; i < 9; ++i) wn[i] = __ldg(kr + 9*(pp + 1) + i);
                }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float d = __half2float(__ushort_as_half((unsigned short) (h ? (w[4] >> 16) : (w[0] & 0xFFFFu))));
                    const float * qb = &sQ[0][32*(2*pp + h)];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t qw = h ? w[5 + k] : __byte_perm(w[k], w[k + 1], 0x5432);
                        const uint32_t lo = qw & 0x0F0F0F0Fu, hi = (qw >> 4) & 0x0F0F0F0Fu;
                        float kf[8];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            kf[i]     = (__int_as_float(__byte_perm(lo, 0x4B000000u, 0x7540 | i)) - 8388616.0f)*d;
                            kf[4 + i] = (__int_as_float(__byte_perm(hi, 0x4B000000u, 0x7540 | i)) - 8388616.0f)*d;
                        }
#pragma unroll
                        for (int q = 0; q < NQ; ++q) {
                            const float4 f0 = *(const float4 *) (qb + q*FD_D + 4*k);
                            const float4 f1 = *(const float4 *) (qb + q*FD_D + 16 + 4*k);
                            float a = fmaf(kf[0], f0.x, acc[q]);
                            float c = kf[4]*f1.x;
                            a = fmaf(kf[1], f0.y, a); c = fmaf(kf[5], f1.y, c);
                            a = fmaf(kf[2], f0.z, a); c = fmaf(kf[6], f1.z, c);
                            a = fmaf(kf[3], f0.w, a); c = fmaf(kf[7], f1.w, c);
                            acc[q] = a + c;
                        }
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) sS[q][tA] = acc[q] + mkA[q / G]*1.4426950408889634f;
        } else if (FD_AV == 1) {
            const uint4 * kp = (const uint4 *) (Kb + (size_t) (t0 + tA)*p.nbk1);
            uint32_t w[36];
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                const uint4 u = __ldg(kp + i);
                w[4*i + 0] = u.x; w[4*i + 1] = u.y; w[4*i + 2] = u.z; w[4*i + 3] = u.w;
            }
            float acc[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) acc[q] = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                float kf[32];
                const float d = (b & 1) ? fd_unpack<true>(w + 9*(b >> 1), kf) : fd_unpack<false>(w + 9*(b >> 1), kf);
#if FD_ILP
                // scale folded into K, element-outer: NQ independent FMA chains
#pragma unroll
                for (int i = 0; i < 32; ++i) kf[i] *= d;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#pragma unroll
                    for (int q = 0; q < NQ; ++q) {
                        const float4 f = *(const float4 *) &sQ[FD_QOFF(q)][32*b + 4*e];
                        acc[q] = fmaf(kf[4*e + 0], f.x, acc[q]);
                        acc[q] = fmaf(kf[4*e + 1], f.y, acc[q]);
                        acc[q] = fmaf(kf[4*e + 2], f.z, acc[q]);
                        acc[q] = fmaf(kf[4*e + 3], f.w, acc[q]);
                    }
                }
#else
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    acc[q] = fmaf(fd_dot32(kf, (const float4 *) &sQ[q][32*b]), d, acc[q]);
                }
#endif
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) sS[q][tA] = acc[q] + mkA[q / G]*1.4426950408889634f;
        } else {
            const uint32_t * kpA = (const uint32_t *) (Kb + (size_t) (t0 + tA)*p.nbk1);
            const uint32_t * kpB = (const uint32_t *) (Kb + (size_t) (t0 + tA + 64)*p.nbk1);
            float accA[NQH], accB[NQH];
#pragma unroll
            for (int q = 0; q < NQH; ++q) { accA[q] = 0.0f; accB[q] = 0.0f; }
#pragma unroll
            for (int pp = 0; pp < 4; ++pp) {
                uint32_t wA[9], wB[9];
#pragma unroll
                for (int i = 0; i < 9; ++i) { wA[i] = __ldg(kpA + 9*pp + i); wB[i] = __ldg(kpB + 9*pp + i); }
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float kA[32], kB[32];
                    const float dA = h ? fd_unpack<true>(wA, kA) : fd_unpack<false>(wA, kA);
                    const float dB = h ? fd_unpack<true>(wB, kB) : fd_unpack<false>(wB, kB);
                    const int b = 2*pp + h;
#pragma unroll
                    for (int q = 0; q < NQH; ++q) {
                        const float4 * qp = (const float4 *) &sQ[q0 + q][32*b];
                        float sa = 0.0f, sb = 0.0f;
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const float4 f = qp[e];
                            sa = fmaf(kA[4*e + 0], f.x, sa); sb = fmaf(kB[4*e + 0], f.x, sb);
                            sa = fmaf(kA[4*e + 1], f.y, sa); sb = fmaf(kB[4*e + 1], f.y, sb);
                            sa = fmaf(kA[4*e + 2], f.z, sa); sb = fmaf(kB[4*e + 2], f.z, sb);
                            sa = fmaf(kA[4*e + 3], f.w, sa); sb = fmaf(kB[4*e + 3], f.w, sb);
                        }
                        accA[q] = fmaf(sa, dA, accA[q]);
                        accB[q] = fmaf(sb, dB, accB[q]);
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < NQH; ++q) {
                const int r = (q0 + q) / G;
                float ma = mkA[0], mb = mkB[0];
#pragma unroll
                for (int rr = 1; rr < NR; ++rr) if (rr == r) { ma = mkA[rr]; mb = mkB[rr]; }
                sS[q0 + q][tA]      = accA[q] + ma*1.4426950408889634f;
                sS[q0 + q][tA + 64] = accB[q] + mb*1.4426950408889634f;
            }
        }
#endif
        __syncthreads();

        // ---- phase B: online softmax, warp per query
        for (int q = warp; q < NQ; q += FD_NT/32) {
            float s[4];
            float mx = -FLT_MAX/2.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) { s[k] = sS[q][lane + 32*k]; mx = fmaxf(mx, s[k]); }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, off, 32));
            const float m_old = sM[q];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) { const float e = exp2f(s[k] - m_new); sS[q][lane + 32*k] = e; sum += e; }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, off, 32);
            if (lane == 0) {
                const float alpha = exp2f(m_old - m_new);
                sM[q] = m_new; sL[q] = sL[q]*alpha + sum; sA[q] = alpha;
            }
        }
        __syncthreads();

        // ---- phase C: O += P V
#ifndef FD_SKIP_C
#pragma unroll
        for (int q = 0; q < NQ; ++q) { const float a = sA[q]; o[q][0] *= a; o[q][1] *= a; }
        const char * vt = Vb + (size_t) t0*p.nbv1;
#pragma unroll 2
        for (int tt = 0; tt < FD_TILE; tt += 4) {
            float v0[4], v1[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const char * vr = vt + (size_t) (tt + u)*p.nbv1;
                const float d  = __half2float(__ldg((const half *) vr));
                const int   by = __ldg((const unsigned char *) vr + 2 + vj);
                v0[u] = (float) ((by & 15) - 8)*d;
                v1[u] = (float) ((by >> 4) - 8)*d;
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float4 pr = *(const float4 *) &sS[q][tt];
                o[q][0] = fmaf(pr.x, v0[0], o[q][0]); o[q][1] = fmaf(pr.x, v1[0], o[q][1]);
                o[q][0] = fmaf(pr.y, v0[1], o[q][0]); o[q][1] = fmaf(pr.y, v1[1], o[q][1]);
                o[q][0] = fmaf(pr.z, v0[2], o[q][0]); o[q][1] = fmaf(pr.z, v1[2], o[q][1]);
                o[q][0] = fmaf(pr.w, v0[3], o[q][0]); o[q][1] = fmaf(pr.w, v1[3], o[q][1]);
            }
        }
#endif
        __syncthreads();
    }

    const int blk = (seq*p.nkvh + kvh)*p.nsplit + split;
    float * po = p.part_o + (size_t) blk*NQ*FD_D;
    const int d0 = vb*32 + vj;
#pragma unroll
    for (int q = 0; q < NQ; ++q) { po[q*FD_D + d0] = o[q][0]; po[q*FD_D + d0 + 16] = o[q][1]; }
    if (tid < NQ) p.part_ml[(size_t) blk*NQ + tid] = make_float2(sM[tid], sL[tid]);
}

// grid (NQ, nkvh, ns), 256 threads (one per dim). dst is [D, H, N, S] contiguous.
static __global__ void __launch_bounds__(FD_D) fd_combine(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
        float * __restrict__ dst, const int G, const int NR, const int nsplit) {
    const int q = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z, d = threadIdx.x;
    const int NQ = G*NR, nkvh = gridDim.y;
    const int blk0 = (seq*nkvh + kvh)*nsplit;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, part_ml[(size_t) (blk0 + s)*NQ + q].x);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float2 ml = part_ml[(size_t) (blk0 + s)*NQ + q];
        const float  w  = exp2f(ml.x - M);
        L   += ml.y*w;
        acc += part_o[((size_t) (blk0 + s)*NQ + q)*FD_D + d]*w;
    }
    const int r = q / G, g = q % G, H = nkvh*G;
    dst[(((size_t) seq*NR + r)*H + kvh*G + g)*FD_D + d] = L > 0.0f ? acc/L : 0.0f;
}

template <int G, int NR>
static void fd_launch_t(const fd_params & p, int nseq, float * dst, cudaStream_t stream) {
    fd_kernel<G, NR><<<dim3(p.nsplit, p.nkvh, nseq), FD_NT, 0, stream>>>(p);
    fd_combine<<<dim3(G*NR, p.nkvh, nseq), FD_D, 0, stream>>>(p.part_o, p.part_ml, dst, G, NR, p.nsplit);
}

// returns false if (G, NR) has no instance
static bool fd_launch(const fd_params & p, int G, int NR, int nseq, float * dst, cudaStream_t stream) {
#define FD_CASE(g, nr) if (G == g && NR == nr) { fd_launch_t<g, nr>(p, nseq, dst, stream); return true; }
    FD_CASE(6, 1) FD_CASE(6, 2) FD_CASE(6, 3) FD_CASE(6, 4)
    FD_CASE(4, 1) FD_CASE(4, 2) FD_CASE(4, 3) FD_CASE(4, 4)
    FD_CASE(8, 1) FD_CASE(8, 2) FD_CASE(8, 3)
#undef FD_CASE
    return false;
}

static int fd_env_int(const char * name, int def) {
    const char * s = getenv(name);
    return (s == nullptr || *s == '\0') ? def : atoi(s);
}

#ifndef FD_STANDALONE
int ggml_cuda_fa_dec_mode() {
    static const int v = fd_env_int("GGML_CUDA_FA_DEC", 0);
    return v;
}

static long long fd_ncalls = 0;
static void fd_report_exit() { fprintf(stderr, "ggml_cuda: fa_dec2 total calls=%lld\n", fd_ncalls); }
static bool fd_debug() {
    static const bool v = []() { const bool d = fd_env_int("GGML_CUDA_FA_DEC_DEBUG", 0) != 0; if (d) atexit(fd_report_exit); return d; }();
    return v;
}

bool ggml_cuda_fa_dec_applicable(const ggml_tensor * dst) {
    static const int minq = std::max(1, fd_env_int("GGML_CUDA_FA_DEC_MINQ", 3));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || dst->type != GGML_TYPE_F32) return false;
    if (Q->ne[0] != FD_D || K->ne[0] != FD_D || V->ne[0] != FD_D) return false;
    if (K->ne[2] != V->ne[2] || K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0) return false;
    const int G = (int) (Q->ne[2] / K->ne[2]);
    const int NR = (int) Q->ne[1];
    if (G != 4 && G != 6 && G != 8) return false;
    if (NR < minq || NR > 4 || G*NR > 24) return false;
    if (K->ne[1] != V->ne[1] || K->ne[1] % FD_TILE != 0 || K->ne[1] <= 0) return false;
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) return false;
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1]) return false;
    if (Q->ne[3] % mask->ne[3] != 0) return false;
    if (sinks || max_bias != 0.0f || softcap != 0.0f) return false;
    if (!ggml_is_contiguous(dst) || dst->ne[0] != FD_D || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1]) return false;
    if (Q->nb[0] != 4 || mask->nb[0] != 2) return false;
    if (K->nb[0] != 18 || V->nb[0] != 18) return false;  // q4_0 block stride
    if ((uintptr_t) K->data % 16 || (uintptr_t) V->data % 2 || (uintptr_t) mask->data % 2 || (uintptr_t) Q->data % 4) return false;
    for (int i = 1; i < 4; ++i) {
        if (K->nb[i] % 16 || V->nb[i] % 2 || Q->nb[i] % 4) return false;
    }
    return true;
}

void ggml_cuda_fa_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    const int nkvh = (int) K->ne[2];
    const int ns   = (int) Q->ne[3];
    const int G    = (int) (Q->ne[2] / K->ne[2]);
    const int NR   = (int) Q->ne[1];
    const int NQ   = G*NR;
    const int ntiles = (int) (K->ne[1] / FD_TILE);
    static const int env_split = fd_env_int("GGML_CUDA_FA_DEC_SPLIT", 0);
    int nsplit = env_split > 0 ? env_split : (96 + nkvh*ns - 1) / (nkvh*ns);
    nsplit = std::max(1, std::min(nsplit, ntiles));

    const int nblk = nsplit*nkvh*ns;
    ggml_cuda_pool_alloc<float>  part_o (ctx.pool(), (size_t) nblk*NQ*FD_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) nblk*NQ);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    fd_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data; p.mask = (const char *) mask->data;
    p.part_o = part_o.get(); p.part_ml = part_ml.get();
    p.ntiles = ntiles; p.nsplit = nsplit; p.nkvh = nkvh; p.mask_ne3 = (int) mask->ne[3];
    p.scale_log2 = scale*1.4426950408889634f;
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];

    GGML_ASSERT(fd_launch(p, G, NR, ns, (float *) dst->data, stream));
    CUDA_CHECK(cudaGetLastError());

    // diagnostic: GGML_CUDA_FA_DEC_VERIFY=1 re-runs each call with nsplit=1 on the same inputs and logs output differences (slow)
    static const int env_verify = fd_env_int("GGML_CUDA_FA_DEC_VERIFY", 0);
    if (env_verify == 2) {
        // CPU double reference on the real inputs of this call, compared per (seq, row, head)
        static long long c_calls = 0, c_flag = 0;
        static double c_worst = 0.0, c_worst_abs = 0.0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int64_t nkv = K->ne[1]; const int H = (int) Q->ne[2];
        const auto span = [](const ggml_tensor * t, size_t row) { size_t sz = row; for (int i = 1; i < 4; ++i) sz += (size_t) (t->ne[i] - 1)*t->nb[i]; return sz; };
        const size_t nO = (size_t) ns*NR*H*FD_D;
        std::vector<uint8_t> hK(span(K, 144)), hV(span(V, 144)), hQ(span(Q, FD_D*4)), hM(span(mask, (size_t) mask->ne[0]*2));
        std::vector<float> hO(nO);
        CUDA_CHECK(cudaMemcpy(hK.data(), K->data, hK.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hV.data(), V->data, hV.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hQ.data(), Q->data, hQ.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hM.data(), mask->data, hM.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hO.data(), dst->data, nO*4, cudaMemcpyDeviceToHost));
        const auto deq = [](const uint8_t * row, double * out) {
            for (int e = 0; e < FD_D; ++e) {
                const uint8_t * blk = row + 18*(e/32);
                unsigned short h; memcpy(&h, blk, 2);
                const double d = __half2float(__ushort_as_half(h));
                const int j = e % 32; const uint8_t by = blk[2 + (j & 15)];
                out[e] = (double) ((j < 16 ? (by & 15) : (by >> 4)) - 8)*d;
            }
        };
        std::vector<double> dK((size_t) nkv*FD_D), dV((size_t) nkv*FD_D), sc(nkv), o(FD_D);
        double call_worst = 0.0, call_abs = 0.0; int wr = 0, wh = 0;
        for (int sq = 0; sq < ns; ++sq) for (int kh = 0; kh < nkvh; ++kh) {
            for (int64_t t = 0; t < nkv; ++t) {
                deq(hK.data() + sq*K->nb[3] + kh*K->nb[2] + t*K->nb[1], &dK[t*FD_D]);
                deq(hV.data() + sq*V->nb[3] + kh*V->nb[2] + t*V->nb[1], &dV[t*FD_D]);
            }
            for (int r = 0; r < NR; ++r) for (int g = 0; g < G; ++g) {
                const int h = kh*G + g;
                const float * q = (const float *) (hQ.data() + sq*Q->nb[3] + r*Q->nb[1] + h*Q->nb[2]);
                const uint8_t * mr = hM.data() + (sq % mask->ne[3])*mask->nb[3] + r*mask->nb[1];
                double mx = -HUGE_VAL;
                for (int64_t t = 0; t < nkv; ++t) {
                    unsigned short mh; memcpy(&mh, mr + t*2, 2);
                    const double m = __half2float(__ushort_as_half(mh));
                    if (std::isinf(m) && m < 0) { sc[t] = -HUGE_VAL; continue; }
                    double a = 0.0; const double * kr = &dK[t*FD_D];
                    for (int e = 0; e < FD_D; ++e) a += q[e]*kr[e];
                    sc[t] = a*scale + m; mx = std::max(mx, sc[t]);
                }
                double L = 0.0; std::fill(o.begin(), o.end(), 0.0);
                for (int64_t t = 0; t < nkv; ++t) {
                    if (sc[t] == -HUGE_VAL) continue;
                    const double w = std::exp(sc[t] - mx); L += w;
                    const double * vr = &dV[t*FD_D];
                    for (int e = 0; e < FD_D; ++e) o[e] += w*vr[e];
                }
                const float * got = &hO[(((size_t) sq*NR + r)*H + h)*FD_D];
                double md = 0.0, mref = 0.0;
                for (int e = 0; e < FD_D; ++e) { const double ref = L > 0.0 ? o[e]/L : 0.0; md = std::max(md, std::fabs(ref - got[e])); mref = std::max(mref, std::fabs(ref)); }
                if (!(md == md)) md = HUGE_VAL;
                const double rel = md/std::max(mref, 1e-30);
                if (mref > 0.0 && rel > call_worst) { call_worst = rel; wr = r; wh = h; }
                call_abs = std::max(call_abs, md);
            }
        }
        ++c_calls; c_worst = std::max(c_worst, call_worst); c_worst_abs = std::max(c_worst_abs, call_abs);
        if (call_worst > 1e-3 && ++c_flag <= 200) {
            fprintf(stderr, "ggml_cuda: fa_dec2 verify cpu flag call=%lld n_kv=%lld nsplit=%d rel=%.3g abs=%.3g r=%d h=%d\n", c_calls, (long long) nkv, nsplit, call_worst, call_abs, wr, wh);
        }
        if (c_calls % 5000 == 0) {
            fprintf(stderr, "ggml_cuda: fa_dec2 verify cpu calls=%lld flagged=%lld worst_rel=%.3g worst_abs=%.3g\n", c_calls, c_flag, c_worst, c_worst_abs);
        }
    }
    if (env_verify == 1 && nsplit > 1) {
        static long long v_calls = 0, v_flag = 0, v_nonfinite = 0;
        static double v_worst = 0.0;
        const size_t n = (size_t) ns*NR*Q->ne[2]*FD_D;
        ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n);
        fd_params p1 = p; p1.nsplit = 1;
        GGML_ASSERT(fd_launch(p1, G, NR, ns, tmp.get(), stream));
        std::vector<float> a(n), b(n);
        CUDA_CHECK(cudaMemcpyAsync(a.data(), dst->data, n*4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), tmp.get(), n*4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double mxd = 0.0, mxb = 0.0; size_t at = 0; long long nf = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nf; continue; }
            const double d = std::fabs((double) a[i] - b[i]);
            if (d > mxd) { mxd = d; at = i; }
            mxb = std::max(mxb, (double) std::fabs(b[i]));
        }
        const double rel = mxd/std::max(mxb, 1e-30);
        ++v_calls; v_nonfinite += nf; v_worst = std::max(v_worst, rel);
        if (rel > 1e-3 || nf) {
            if (++v_flag <= 200) {
                fprintf(stderr, "ggml_cuda: fa_dec2 verify flag call=%lld n_kv=%lld nsplit=%d rel=%.3g maxd=%.3g maxb=%.3g idx(r=%zu h=%zu d=%zu) a=%.6g b=%.6g nonfinite=%lld\n",
                    v_calls, (long long) K->ne[1], nsplit, rel, mxd, mxb, at/FD_D/Q->ne[2], (at/FD_D) % Q->ne[2], at % FD_D, a[at], b[at], nf);
            }
        }
        if (v_calls % 5000 == 0) {
            fprintf(stderr, "ggml_cuda: fa_dec2 verify progress calls=%lld flagged=%lld nonfinite=%lld worst_rel=%.3g\n", v_calls, v_flag, v_nonfinite, v_worst);
        }
    }

    if (fd_debug()) {
        ++fd_ncalls;
        if (fd_ncalls <= 8 || fd_ncalls % 1000 == 0) {
            fprintf(stderr, "ggml_cuda: fa_dec2 call=%lld G=%d NR=%d nkvh=%d ns=%d n_kv=%lld nsplit=%d\n",
                fd_ncalls, G, NR, nkvh, ns, (long long) K->ne[1], nsplit);
        }
    }
}

// ---- 8 Nov card F2 diagnostics (GGML_CUDA_FA_DBG=1). Nothing below runs unless that env is set. ----
// GGML_CUDA_FA_DBG_AUTH   0: stock output goes downstream (default), 2: design-2 output goes downstream
// GGML_CUDA_FA_DBG_FLIP   a:b calls in [a,b) use the other kernel as authority
// GGML_CUDA_FA_DBG_CMP    1: run both kernels on the same inputs each call and log the difference
// GGML_CUDA_FA_DBG_CPU    1: with CMP, also compare both with a CPU double reference
// GGML_CUDA_FA_DBG_NOISERANGE a:b,... restrict noise to these calls (default all)
// GGML_CUDA_FA_DBG_NOISE  eps: multiply each downstream element by (1 + eps*u), u uniform in [-1,1]; GGML_CUDA_FA_DBG_SEED
// GGML_CUDA_FA_DBG_SNAP   a:b,... calls whose raw K,V are dumped; GGML_CUDA_FA_DBG_DUMP a:b,... also Q, mask and both outputs
// GGML_CUDA_FA_DBG_DIR, GGML_CUDA_FA_DBG_TAG  output location
static bool fdbg_in_list(const char * s, long long c) {
    while (s && *s) {
        long long a = 0, b = 0;
        if (sscanf(s, "%lld:%lld", &a, &b) == 2 && c >= a && c < b) return true;
        s = strchr(s, ',');
        if (s) ++s;
    }
    return false;
}
static const char * fdbg_str(const char * n) { const char * s = getenv(n); return s ? s : ""; }
static long long fdbg_call = -1;

bool ggml_cuda_fa_dbg_on() {
    static const bool v = fd_env_int("GGML_CUDA_FA_DBG", 0) != 0;
    return v;
}

static size_t fdbg_span(const ggml_tensor * t, size_t row) {
    size_t sz = row;
    for (int i = 1; i < 4; ++i) sz += (size_t) (t->ne[i] - 1)*t->nb[i];
    return sz;
}
static std::vector<uint8_t> fdbg_host(const ggml_tensor * t, size_t row) {
    std::vector<uint8_t> h(fdbg_span(t, row));
    CUDA_CHECK(cudaMemcpy(h.data(), t->data, h.size(), cudaMemcpyDeviceToHost));
    return h;
}
static FILE * fdbg_open(const char * kind, long long c) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s-%s-%lld.bin", fdbg_str("GGML_CUDA_FA_DBG_DIR"), fdbg_str("GGML_CUDA_FA_DBG_TAG"), kind, c);
    return fopen(path, "wb");
}
static void fdbg_put(FILE * f, const ggml_tensor * t, const std::vector<uint8_t> & h) {
    int64_t hdr[9];
    for (int i = 0; i < 4; ++i) { hdr[i] = t->ne[i]; hdr[4 + i] = (int64_t) t->nb[i]; }
    hdr[8] = (int64_t) h.size();
    fwrite(hdr, 8, 9, f);
    fwrite(h.data(), 1, h.size(), f);
}
static uint64_t fdbg_hash(const void * d, size_t n) {
    uint64_t x = 1469598103934665603ULL;
    const uint8_t * p = (const uint8_t *) d;
    for (size_t i = 0; i < n; ++i) { x ^= p[i]; x *= 1099511628211ULL; }
    return x;
}

// every FA call: count it, and dump raw K/V for the listed calls
void ggml_cuda_fa_dbg_pre(ggml_tensor * dst) {
    ++fdbg_call;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (fdbg_call < 34) {
        fprintf(stderr, "ggml_cuda: fa_dbg call=%lld name=%s NR=%lld nkv=%lld K=%s V=%s\n", fdbg_call, dst->name, (long long) Q->ne[1], (long long) K->ne[1], K->name, V->name);
    }
    if (fdbg_in_list(fdbg_str("GGML_CUDA_FA_DBG_SNAP"), fdbg_call) && K->type == GGML_TYPE_Q4_0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        FILE * f = fdbg_open("snap", fdbg_call);
        if (f) { fdbg_put(f, K, fdbg_host(K, 144)); fdbg_put(f, V, fdbg_host(V, 144)); fclose(f); }
    }
}

// exact reference for this call, [ns][NR][H][D] as dst
static void fdbg_cpu_ref(const ggml_tensor * dst, std::vector<double> & out) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int ns = (int) Q->ne[3], NR = (int) Q->ne[1], H = (int) Q->ne[2], nkvh = (int) K->ne[2], G = H/nkvh;
    const int64_t nkv = K->ne[1];
    const std::vector<uint8_t> hK = fdbg_host(K, 144), hV = fdbg_host(V, 144), hQ = fdbg_host(Q, FD_D*4), hM = fdbg_host(mask, (size_t) mask->ne[0]*2);
    out.assign((size_t) ns*NR*H*FD_D, 0.0);
    const auto deq = [](const uint8_t * row, double * o) {
        for (int e = 0; e < FD_D; ++e) {
            const uint8_t * blk = row + 18*(e/32);
            unsigned short h; memcpy(&h, blk, 2);
            const double d = __half2float(__ushort_as_half(h));
            const int j = e % 32; const uint8_t by = blk[2 + j/2];
            o[e] = (double) (((j & 1) == 0 ? (by & 15) : (by >> 4)) - 8)*d;
        }
    };
    std::vector<double> dK((size_t) nkv*FD_D), dV((size_t) nkv*FD_D), sc(nkv), o(FD_D);
    for (int sq = 0; sq < ns; ++sq) for (int kh = 0; kh < nkvh; ++kh) {
        for (int64_t t = 0; t < nkv; ++t) {
            deq(hK.data() + sq*K->nb[3] + kh*K->nb[2] + t*K->nb[1], &dK[t*FD_D]);
            deq(hV.data() + sq*V->nb[3] + kh*V->nb[2] + t*V->nb[1], &dV[t*FD_D]);
        }
        for (int r = 0; r < NR; ++r) for (int g = 0; g < G; ++g) {
            const int h = kh*G + g;
            const float * q = (const float *) (hQ.data() + sq*Q->nb[3] + r*Q->nb[1] + h*Q->nb[2]);
            const uint8_t * mr = hM.data() + (sq % mask->ne[3])*mask->nb[3] + r*mask->nb[1];
            double mx = -HUGE_VAL;
            for (int64_t t = 0; t < nkv; ++t) {
                unsigned short mh; memcpy(&mh, mr + t*2, 2);
                const double m = __half2float(__ushort_as_half(mh));
                if (std::isinf(m) && m < 0) { sc[t] = -HUGE_VAL; continue; }
                double a = 0.0;
                for (int e = 0; e < FD_D; ++e) a += q[e]*dK[t*FD_D + e];
                sc[t] = a*scale + m; mx = std::max(mx, sc[t]);
            }
            double L = 0.0; std::fill(o.begin(), o.end(), 0.0);
            for (int64_t t = 0; t < nkv; ++t) {
                if (sc[t] == -HUGE_VAL) continue;
                const double w = std::exp(sc[t] - mx); L += w;
                for (int e = 0; e < FD_D; ++e) o[e] += w*dV[t*FD_D + e];
            }
            double * d = &out[(((size_t) sq*NR + r)*H + h)*FD_D];
            for (int e = 0; e < FD_D; ++e) d[e] = L > 0.0 ? o[e]/L : 0.0;
        }
    }
}

// per (row, head) max-norm relative difference: worst over the call, with its location; also worst abs and rms rel
struct fdbg_cmp { double wrel, wabs, rms; int r, h, e; long long nf; };
template <typename A, typename B>
static fdbg_cmp fdbg_compare(const A * a, const B * b, size_t n, int H) {
    fdbg_cmp c = {0.0, 0.0, 0.0, 0, 0, 0, 0};
    double s2 = 0.0, b2 = 0.0;
    for (size_t row = 0; row*H*FD_D < n; ++row) for (int h = 0; h < H; ++h) {
        const A * pa = a + (row*H + h)*FD_D; const B * pb = b + (row*H + h)*FD_D;
        double md = 0.0, mb = 0.0; int me = 0;
        for (int e = 0; e < FD_D; ++e) {
            const double x = pa[e], y = pb[e];
            if (!std::isfinite(x) || !std::isfinite(y)) { ++c.nf; continue; }
            const double d = std::fabs(x - y);
            if (d > md) { md = d; me = e; }
            mb = std::max(mb, std::fabs(y));
            s2 += d*d; b2 += y*y;
        }
        const double rel = md/std::max(mb, 1e-30);
        if (rel > c.wrel) { c.wrel = rel; c.r = (int) row; c.h = h; c.e = me; }
        c.wabs = std::max(c.wabs, md);
    }
    c.rms = std::sqrt(s2/std::max(b2, 1e-60));
    return c;
}

void ggml_cuda_fa_dbg_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_fa_dbg_stock_fn stock) {
    static const int auth0 = fd_env_int("GGML_CUDA_FA_DBG_AUTH", 0);
    static const int cmp   = fd_env_int("GGML_CUDA_FA_DBG_CMP", 0);
    static const int cpu   = fd_env_int("GGML_CUDA_FA_DBG_CPU", 0);
    static const double eps = []() { const char * s = getenv("GGML_CUDA_FA_DBG_NOISE"); return s ? atof(s) : 0.0; }();
    static const unsigned long long seed = (unsigned long long) fd_env_int("GGML_CUDA_FA_DBG_SEED", 1);
    static FILE * logf = []() { char p[512]; snprintf(p, sizeof(p), "%s/%s-cmp.txt", fdbg_str("GGML_CUDA_FA_DBG_DIR"), fdbg_str("GGML_CUDA_FA_DBG_TAG")); return cmp ? fopen(p, "w") : (FILE *) nullptr; }();
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * Q = dst->src[0];
    const long long c = fdbg_call;
    const int auth = fdbg_in_list(fdbg_str("GGML_CUDA_FA_DBG_FLIP"), c) ? (auth0 == 2 ? 0 : 2) : auth0;
    const int H = (int) Q->ne[2];
    const size_t n = (size_t) Q->ne[3]*Q->ne[1]*H*FD_D;
    const bool dump = fdbg_in_list(fdbg_str("GGML_CUDA_FA_DBG_DUMP"), c);
    if (cmp || dump) {
        std::vector<float> a(n), b(n);
        ggml_cuda_fa_dec(ctx, dst);
        CUDA_CHECK(cudaMemcpyAsync(a.data(), dst->data, n*4, cudaMemcpyDeviceToHost, stream));
        stock(ctx, dst);
        CUDA_CHECK(cudaMemcpyAsync(b.data(), dst->data, n*4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const fdbg_cmp d = fdbg_compare(a.data(), b.data(), n, H);
        if (logf) {
            fprintf(logf, "%lld NR=%lld nkv=%lld auth=%d wrel=%.4g rms=%.4g wabs=%.4g r=%d h=%d e=%d nf=%lld qhash=%016llx ohash=%016llx",
                c, (long long) Q->ne[1], (long long) dst->src[1]->ne[1], auth, d.wrel, d.rms, d.wabs, d.r, d.h, d.e, d.nf,
                (unsigned long long) fdbg_hash(fdbg_host(Q, FD_D*4).data(), fdbg_span(Q, FD_D*4)),
                (unsigned long long) fdbg_hash(auth == 2 ? a.data() : b.data(), n*4));
            if (cpu) {
                std::vector<double> ref;
                fdbg_cpu_ref(dst, ref);
                const fdbg_cmp da = fdbg_compare(a.data(), ref.data(), n, H), db = fdbg_compare(b.data(), ref.data(), n, H);
                fprintf(logf, " dec_cpu=%.4g dec_cpu_rms=%.4g stock_cpu=%.4g stock_cpu_rms=%.4g", da.wrel, da.rms, db.wrel, db.rms);
            }
            fprintf(logf, "\n");
            if (c % 2736 == 0) fflush(logf);
        }
        if (dump) {
            FILE * f = fdbg_open("dump", c);
            if (f) {
                const ggml_tensor * mask = dst->src[3];
                float scale; memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
                fwrite(&scale, 4, 1, f);
                fdbg_put(f, Q, fdbg_host(Q, FD_D*4)); fdbg_put(f, dst->src[1], fdbg_host(dst->src[1], 144)); fdbg_put(f, dst->src[2], fdbg_host(dst->src[2], 144));
                fdbg_put(f, mask, fdbg_host(mask, (size_t) mask->ne[0]*2));
                fwrite(a.data(), 4, n, f); fwrite(b.data(), 4, n, f);
                fclose(f);
            }
        }
        if (auth == 2) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, a.data(), n*4, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    } else if (auth == 2) {
        ggml_cuda_fa_dec(ctx, dst);
    } else {
        stock(ctx, dst);
    }
    const char * nrange = fdbg_str("GGML_CUDA_FA_DBG_NOISERANGE");
    if (eps != 0.0 && (!*nrange || fdbg_in_list(nrange, c))) {
        std::vector<float> o(n);
        CUDA_CHECK(cudaMemcpyAsync(o.data(), dst->data, n*4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        unsigned long long x = (seed*0x9E3779B97F4A7C15ULL) ^ ((unsigned long long) (c + 1)*0xC2B2AE3D27D4EB4FULL);
        for (size_t i = 0; i < n; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            const double u = (double) (x >> 11)*(1.0/9007199254740992.0)*2.0 - 1.0;
            o[i] = (float) ((double) o[i]*(1.0 + eps*u));
        }
        CUDA_CHECK(cudaMemcpyAsync(dst->data, o.data(), n*4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
#endif
