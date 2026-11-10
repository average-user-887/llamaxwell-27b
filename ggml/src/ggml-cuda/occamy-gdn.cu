// native-occamy Track G (13 Oct 2026): fused qwen35moe GDN layer DECODE segment (batch 1, one sequence), sm_52.
// Kernel = native-occamy/decode/gdn_fused.cu gdn_fused<PHASE, V=1> (d2 staged uint4 row loads), adapted to the
// ggml graph: the recurrent states are read from the get_rows source row (s_copy[0]) and written to the cpy
// destinations (kv_head rows); the q/k conv-state write moves to phase C because q/k channels are read by 16
// phase-B items (conv in/out alias in-graph). Activations stay fp32 (no q8_1 requantisation).
//   A: xn = rmsnorm(h)*attn_norm (per block); 12352 GEMV rows (qkv 8192, gate 4096, alpha 32, beta 32), K=2048
//   B: per (v head, 16-column group): conv1d(4)+silu, l2norm q/k, gate/beta, delta rule on 2 columns per warp
//   C: per-head rmsnorm * ssm_norm * silu(z) -> ssm_out GEMV (K=4096) + residual; q/k conv state write
#include "occamy-gdn.cuh"
#include "gated_delta_net.cuh"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

namespace occamy_gdn {

constexpr int H = 2048, NQKV = 8192, DI = 4096, NVH = 32, NKH = 16, SD = 128;
constexpr int QB = 18;
constexpr int ROW_H = H / 32 * QB;     // 1152 B (K=2048)
constexpr int ROW_O = DI / 32 * QB;    // 2304 B (K=4096)
constexpr int NT = 256, NW = NT / 32;
constexpr int CH = 68;
constexpr int NCG = 8, CGC = SD / NCG; // column groups per head (16 cols each, 2 per warp)
constexpr int NROWS_A = NQKV + DI + 2 * NVH;
constexpr int CONV_N = 3 * NQKV;       // conv state floats per row
constexpr int SSM_N  = NVH * SD * SD;  // ssm state floats per row
constexpr unsigned SPIN_LIMIT = 1u << 26;

struct Layer {
    const float * attn_norm, * conv_w, * ssm_a, * dt_bias, * ssm_norm;
    const uint8_t * qkv, * gate, * alpha, * beta, * out;
};
struct Scratch {
    float qkv[NQKV], z[DI], ab[2 * NVH], attn[DI], ssq[NVH * NCG];
    unsigned bar_count, bar_gen;
};
struct Params {
    Layer L;
    const float * h;          // residual stream in [2048]
    float * out;              // [2048] = h + ssm_out(...)
    const int32_t * s_copy;   // source row index for both states
    const float * conv_base; size_t conv_stride;   // [row][8192][3] (time-fastest, oldest first)
    float * conv_out;                              // [8192][3]
    const float * ssm_base;  size_t ssm_stride;    // [row][32][128 col][128 i] (transposed, as ggml-cuda)
    float * S_out;                                 // [32][128][128]
    Scratch * s;
    float eps_attn, eps_l2, scale_l2, eps_gated;
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ void grid_barrier(Scratch * s) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned * genp = &s->bar_gen;
        unsigned gen = *genp;
        __threadfence();
        if (atomicAdd(&s->bar_count, 1) == gridDim.x - 1) {
            s->bar_count = 0;
            __threadfence();
            atomicAdd(&s->bar_gen, 1);
        } else {
            unsigned k = 0;
            while (*genp == gen) { if (++k > SPIN_LIMIT) { __trap(); } }
        }
        __threadfence();
    }
    __syncthreads();
}
__device__ __forceinline__ float nib(unsigned v, int k) {
    return __int_as_float(__byte_perm(v, 0x3F000000u, 0x7044u | (k << 8)));
}
// dot of 2 consecutive Q4_0 blocks staged as 9 smem words (lane stride 9) with a 64-float smem chunk and its block sums
__device__ __forceinline__ float dot2w(const unsigned * w, const float * xc, float2 sx) {
    unsigned w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7], w8 = w[8];
    float d0 = __half2float(__ushort_as_half((unsigned short)(w0 & 0xFFFF)));
    float d1 = __half2float(__ushort_as_half((unsigned short)(w4 >> 16)));
    unsigned q0[4] = { __byte_perm(w0, w1, 0x5432), __byte_perm(w1, w2, 0x5432), __byte_perm(w2, w3, 0x5432), __byte_perm(w3, w4, 0x5432) };
    unsigned q1[4] = { w5, w6, w7, w8 };
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned lo = (q0[i] & 0x0F0F0F0Fu) | 0x80808080u, hi = ((q0[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        float4 xl = *(const float4 *)(xc + 4 * i), xh = *(const float4 *)(xc + 16 + 4 * i);
        a0 = fmaf(nib(lo, 0), xl.x, a0); a0 = fmaf(nib(lo, 1), xl.y, a0); a0 = fmaf(nib(lo, 2), xl.z, a0); a0 = fmaf(nib(lo, 3), xl.w, a0);
        a0 = fmaf(nib(hi, 0), xh.x, a0); a0 = fmaf(nib(hi, 1), xh.y, a0); a0 = fmaf(nib(hi, 2), xh.z, a0); a0 = fmaf(nib(hi, 3), xh.w, a0);
        lo = (q1[i] & 0x0F0F0F0Fu) | 0x80808080u; hi = ((q1[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        xl = *(const float4 *)(xc + 32 + 4 * i); xh = *(const float4 *)(xc + 48 + 4 * i);
        a1 = fmaf(nib(lo, 0), xl.x, a1); a1 = fmaf(nib(lo, 1), xl.y, a1); a1 = fmaf(nib(lo, 2), xl.z, a1); a1 = fmaf(nib(lo, 3), xl.w, a1);
        a1 = fmaf(nib(hi, 0), xh.x, a1); a1 = fmaf(nib(hi, 1), xh.y, a1); a1 = fmaf(nib(hi, 2), xh.z, a1); a1 = fmaf(nib(hi, 3), xh.w, a1);
    }
    return d0 * (128.f * a0 - 136.f * sx.x) + d1 * (128.f * a1 - 136.f * sx.y);
}
__device__ __forceinline__ const uint8_t * rowA(const Layer & L, int t) {
    return t < NQKV ? L.qkv + (size_t)t * ROW_H : t < NQKV + DI ? L.gate + (size_t)(t - NQKV) * ROW_H
         : t < NQKV + DI + NVH ? L.alpha + (size_t)(t - NQKV - DI) * ROW_H : L.beta + (size_t)(t - NQKV - DI - NVH) * ROW_H;
}
__device__ __forceinline__ void block_sums(const float * xs, float2 * sums, int n) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < n / 32; b += NW) {
        float s = warp_sum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
        if (lane == 0) ((float *)sums)[b] = s;
    }
    __syncthreads();
}

constexpr int SMEM_FLOATS = DI / 64 * CH + DI / 32 + 64;
constexpr int STW = ROW_O / 4;   // 576 stage words per warp
constexpr int SMEM2_FLOATS = SMEM_FLOATS + NW * STW;

template <int PHASE>
__global__ void __launch_bounds__(NT, 2) gdn_fused(Params P) {
    extern __shared__ float smem[];
    __shared__ float qn[SD], kn[SD], vv[CGC], red[NW], fac[NVH];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + DI / 64 * CH);
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + (threadIdx.x >> 5) * STW; uint4 * stg4 = (uint4 *)stg;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * NW + warp, W = gridDim.x * NW;
    Scratch * s = P.s;
    const Layer & L = P.L;
    const int row = __ldg(P.s_copy);
    const float * conv_in = P.conv_base + (size_t)row * P.conv_stride;
    const float * S_in    = P.ssm_base  + (size_t)row * P.ssm_stride;

    if (PHASE == 0 || PHASE == 1) {
        // ---- phase A: xn = rmsnorm(h)*w (per block), then all K=2048 GEMV rows ----
        float ss = 0.f;
        for (int i = threadIdx.x; i < H; i += NT) { float v = P.h[i]; xs[(i >> 6) * CH + (i & 63)] = v; ss += v * v; }
        ss = warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < NW; ++i) tot += red[i];
        const float sc = rsqrtf(tot / H + P.eps_attn);
        for (int i = threadIdx.x; i < H; i += NT) xs[(i >> 6) * CH + (i & 63)] *= sc * __ldg(L.attn_norm + i);
        __syncthreads();
        block_sums(xs, sums, H);
        const float * xc = xs + lane * CH; const float2 sx = sums[lane];
        uint4 c0, c1, c2;
        int t = gw;
        if (t < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
        for (; t < NROWS_A; t += W) {
            stg4[lane] = c0; stg4[lane + 32] = c1; if (lane < 8) stg4[lane + 64] = c2;
            __syncwarp();
            const int tn = t + W;
            if (tn < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, tn); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
            float a = warp_sum(dot2w(stg + lane * 9, xc, sx));
            if (lane == 0) { float * dst = t < NQKV ? &s->qkv[t] : t < NQKV + DI ? &s->z[t - NQKV] : &s->ab[t - NQKV - DI]; *dst = a; }
            __syncwarp();
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 2) {
        // ---- phase B: conv + l2norm + gates + delta rule, item = (v head, column group) ----
        for (int it = blockIdx.x; it < NVH * NCG; it += gridDim.x) {
            const int hv = it / NCG, cg = it % NCG, kh = hv % NKH;   // ggml-cuda: q/k head = v head mod n_k_heads
            {
                const int t = threadIdx.x;
                const int c = t < SD ? kh * SD + t : H + kh * SD + (t - SD);
                const float * ci = conv_in + c * 3; const float * w = L.conv_w + c * 4;
                float xnew = __ldcg(&s->qkv[c]);
                float y = ci[0] * w[0] + ci[1] * w[1] + ci[2] * w[2] + xnew * w[3];
                y = silu(y);
                float sq = warp_sum(y * y);
                if (lane == 0) red[warp] = sq;
                if (t < SD) qn[t] = y; else kn[t - SD] = y;
                if (t < CGC) {
                    const int cv = 2 * H + hv * SD + cg * CGC + t;   // unique owner -> conv state written here
                    const float * cvi = conv_in + cv * 3; const float * wv = L.conv_w + cv * 4;
                    float xv = __ldcg(&s->qkv[cv]);
                    float c1 = cvi[1], c2 = cvi[2];
                    float yv = cvi[0] * wv[0] + c1 * wv[1] + c2 * wv[2] + xv * wv[3];
                    vv[t] = silu(yv);
                    float * co = P.conv_out + cv * 3; co[0] = c1; co[1] = c2; co[2] = xv;
                }
            }
            __syncthreads();
            // l2 norm as ggml: rms_norm(x, eps/n) * 1/sqrt(n)
            const float qs = rsqrtf((red[0] + red[1] + red[2] + red[3]) / SD + P.eps_l2) * P.scale_l2;
            const float ks = rsqrtf((red[4] + red[5] + red[6] + red[7]) / SD + P.eps_l2) * P.scale_l2;
            const float al = __ldcg(&s->ab[hv]) + __ldg(L.dt_bias + hv);
            const float sp = al > 20.f ? al : logf(1.f + expf(al));
            const float gexp = expf(sp * __ldg(L.ssm_a + hv));
            const float beta = 1.f / (1.f + expf(-__ldcg(&s->ab[NVH + hv])));
            const float4 q4 = *(const float4 *)(qn + 4 * lane), k4r = *(const float4 *)(kn + 4 * lane);
            const float4 qv = make_float4(q4.x * qs, q4.y * qs, q4.z * qs, q4.w * qs);
            const float4 kv4 = make_float4(k4r.x * ks, k4r.y * ks, k4r.z * ks, k4r.w * ks);
            const int col0 = cg * CGC + warp * 2;
            const float4 * si = (const float4 *)(S_in + ((size_t)hv * SD + col0) * SD) + lane;
            float4 * so = (float4 *)(P.S_out + ((size_t)hv * SD + col0) * SD) + lane;
            float4 st[2] = { si[0], si[SD / 4] };
            float ssq = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float4 a = st[j];
                float kvd = warp_sum(a.x * kv4.x + a.y * kv4.y + a.z * kv4.z + a.w * kv4.w);
                float delta = (vv[warp * 2 + j] - gexp * kvd) * beta;
                a.x = gexp * a.x + kv4.x * delta; a.y = gexp * a.y + kv4.y * delta;
                a.z = gexp * a.z + kv4.z * delta; a.w = gexp * a.w + kv4.w * delta;
                float att = warp_sum(a.x * qv.x + a.y * qv.y + a.z * qv.z + a.w * qv.w) * rsqrtf((float)SD);
                so[j * SD / 4] = a;
                if (lane == 0) s->attn[hv * SD + col0 + j] = att;
                ssq += att * att;
            }
            __syncthreads();   // everyone done with red/qn/kn/vv
            if (lane == 0) red[warp] = ssq;
            __syncthreads();
            if (threadIdx.x == 0) { float t = 0.f; for (int i = 0; i < NW; ++i) t += red[i]; s->ssq[hv * NCG + cg] = t; }
            __syncthreads();
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 3) {
        // q/k conv state: every phase-B reader is done, one writer per channel
        for (int c = blockIdx.x * NT + threadIdx.x; c < 2 * H; c += gridDim.x * NT) {
            const float * ci = conv_in + c * 3; float * co = P.conv_out + c * 3;
            const float a = ci[1], b = ci[2];
            co[0] = a; co[1] = b; co[2] = __ldcg(&s->qkv[c]);
        }
        // ---- phase C: o = rmsnorm_head(attn)*ssm_norm*silu(z) -> ssm_out GEMV + residual ----
        if (threadIdx.x < NVH) {
            float t = 0.f;
#pragma unroll
            for (int i = 0; i < NCG; ++i) t += __ldcg(&s->ssq[threadIdx.x * NCG + i]);
            fac[threadIdx.x] = rsqrtf(t / SD + P.eps_gated);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < DI; i += NT)
            xs[(i >> 6) * CH + (i & 63)] = __ldcg(&s->attn[i]) * fac[i >> 7] * __ldg(L.ssm_norm + (i & 127)) * silu(__ldcg(&s->z[i]));
        __syncthreads();
        block_sums(xs, sums, DI);
        uint4 c[5];
        int r = gw;
        if (r < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)r * ROW_O);
#pragma unroll
            for (int i = 0; i < 4; ++i) c[i] = __ldg(p + lane + 32 * i);
            if (lane < 16) c[4] = __ldg(p + lane + 128); }
        for (; r < H; r += W) {
#pragma unroll
            for (int i = 0; i < 4; ++i) stg4[lane + 32 * i] = c[i];
            if (lane < 16) stg4[lane + 128] = c[4];
            __syncwarp();
            const int rn = r + W;
            if (rn < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)rn * ROW_O);
#pragma unroll
                for (int i = 0; i < 4; ++i) c[i] = __ldg(p + lane + 32 * i);
                if (lane < 16) c[4] = __ldg(p + lane + 128); }
            float acc = dot2w(stg + lane * 9, xs + lane * CH, sums[lane]) + dot2w(stg + (lane + 32) * 9, xs + (lane + 32) * CH, sums[lane + 32]);
            acc = warp_sum(acc);
            if (lane == 0) P.out[r] = P.h[r] + acc;
            __syncwarp();
        }
    }
}

struct DevState {
    Scratch * scratch = nullptr;
    int grid = 0;
};
static DevState g_dev[GGML_CUDA_MAX_DEVICES];
static std::mutex g_mtx;

static bool env_flag(const char * name) {
    const char * e = getenv(name);
    return e != nullptr && atoi(e) != 0;
}
static bool g_debug = env_flag("GGML_CUDA_OCCAMY_DEBUG");
static long g_engaged = 0, g_rejected_gdn = 0;   // windows fused / GDN-layer windows rejected (DEBUG counter)

static void reject(const char * why) {
    if (!g_debug) { return; }
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (seen.insert(why).second) {
        fprintf(stderr, "occamy-gdn: window rejected: %s\n", why);
    }
}

static DevState & dev_state(int device) {
    DevState & d = g_dev[device];
    if (d.scratch == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **)&d.scratch, sizeof(Scratch)));
        CUDA_CHECK(cudaMemset(d.scratch, 0, sizeof(Scratch)));
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_fused<0>, NT, SMEM2_FLOATS * sizeof(float)));
        const int nsm = ggml_cuda_info().devices[device].nsm;
        d.grid = std::max(1, per_sm) * nsm;   // all blocks co-resident -> grid barrier is safe
        if (g_debug) {
            fprintf(stderr, "occamy-gdn: device %d: %d SMs x %d blocks/SM -> grid %d, smem %zu B\n",
                    device, nsm, per_sm, d.grid, SMEM2_FLOATS * sizeof(float));
        }
        if (per_sm < 1) { d.grid = -1; }
    }
    return d;
}

} // namespace occamy_gdn

using namespace occamy_gdn;

bool ggml_cuda_occamy_gdn_enabled() {
    static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN");
    return on;
}

int ggml_cuda_occamy_gdn_check_budget() {
    static int left = getenv("GGML_CUDA_OCCAMY_GDN_CHECK") ? atoi(getenv("GGML_CUDA_OCCAMY_GDN_CHECK")) : 0;
    if (left > 0) { return left--; }
    return 0;
}

static bool is_q4_mat(const ggml_tensor * t, int64_t ne0, int64_t ne1) {
    return t && t->op == GGML_OP_NONE && t->type == GGML_TYPE_Q4_0 && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 &&
           t->ne[3] == 1 && ggml_is_contiguous(t) && ((uintptr_t)t->data & 15) == 0;
}
static bool is_f32_leaf(const ggml_tensor * t, int64_t ne0, int64_t ne1) {
    return t && t->op == GGML_OP_NONE && t->type == GGML_TYPE_F32 && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 &&
           t->ne[3] == 1 && ggml_is_contiguous(t) && ((uintptr_t)t->data & 15) == 0;
}
// root of a view chain (ggml keeps view_src pointing at the non-view base and view_offs cumulative)
static const ggml_tensor * vroot(const ggml_tensor * t) { return t && t->view_src ? t->view_src : t; }
static bool on_device(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    return t->buffer != nullptr && t->buffer->buft == ggml_backend_cuda_buffer_type(ctx.device);
}

bool ggml_cuda_occamy_gdn_match_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, ggml_cuda_occamy_gdn_match & m) {
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM) { return false; }
    const ggml_tensor * h = norm->src[0];
    const int ns=(int)ggml_nrows(h);
    if (h->type != GGML_TYPE_F32 || h->ne[0] != H || !ggml_is_contiguous(h)) { return false; }
    if (ns!=1 && !(ns==2 && env_flag("GGML_CUDA_OCCAMY_GDN_STATE_DIRECT") && ggml_cuda_info().devices[ctx.device].cc==520 && ggml_cuda_gdn_direct_pair_supported())) { return false; }
    m.n_seqs=ns;

    // window end: first ADD after i that takes h directly (the attention residual add)
    constexpr int MAXW = 192;
    int end = -1;
    bool has_gdn = false;
    const int lim = std::min(cgraph->n_nodes, i + MAXW);
    for (int j = i + 1; j < lim; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_GATED_DELTA_NET) { has_gdn = true; }
        if (t->op == GGML_OP_ADD && (t->src[0] == h || t->src[1] == h)) { end = j; break; }
    }
    if (!has_gdn) { return false; }   // not a GDN layer: stay silent, the MoE matcher owns those windows
    static std::set<std::string> dumped;
    int cur = i;
    auto rej = [&](const char * why) {
        g_rejected_gdn++;
        reject(why);
        if (g_debug && dumped.insert(why).second) {
            const int e = end < 0 ? lim - 1 : end;
            fprintf(stderr, "occamy-gdn: first rejected GDN window %d..%d (%s at node %d):\n", i, e, why, cur);
            for (int j = i; j <= e; ++j) {
                const ggml_tensor * t = cgraph->nodes[j];
                fprintf(stderr, "  [%d] %s %s %s ne=[%ld,%ld,%ld,%ld] view_offs=%zu\n", j, ggml_op_name(t->op),
                        t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : "", t->name,
                        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3], t->view_offs);
            }
        }
        return false;
    };
    if (end < 0) { return rej("no residual ADD within 192 nodes"); }
    const int n = end - i + 1;

    const ggml_tensor * win[MAXW];
    for (int j = 0; j < n; ++j) { win[j] = cgraph->nodes[i + j]; }
    std::sort(win, win + n);
    auto in_win = [&](const ggml_tensor * t) { return std::binary_search(win, win + n, t); };

    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return rej("window node flagged as output"); }
        switch (t->op) {
            case GGML_OP_RMS_NORM: case GGML_OP_MUL: case GGML_OP_MUL_MAT: case GGML_OP_ADD: case GGML_OP_UNARY:
            case GGML_OP_SCALE: case GGML_OP_CPY: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE: case GGML_OP_CONT: case GGML_OP_CONCAT: case GGML_OP_GET_ROWS: case GGML_OP_SSM_CONV:
            case GGML_OP_GATED_DELTA_NET:
                break;
            default: return rej(ggml_op_name(t->op));
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = t->src[k];
            if (s == nullptr || s == h || in_win(s)) { continue; }
            const ggml_tensor * r = vroot(s);
            if (r->op == GGML_OP_NONE) {
                if (!on_device(ctx, r)) { return rej("leaf input not in this device's plain CUDA buffer"); }
                continue;
            }
            if (in_win(r)) { continue; }   // view of a window member created outside the window order (not expected)
            return rej("external computed input inside window");
        }
    }
    // every node except the last is consumed only inside the window
    for (int j = 0; j < n - 1; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        int uses = 0;
        for (int q = 0; q < n; ++q) {
            const ggml_tensor * u = cgraph->nodes[i + q];
            for (int k = 0; k < GGML_MAX_SRC; ++k) { if (u->src[k] == t) { uses++; } }
        }
        if (uses != ggml_node_get_use_count(cgraph, i + j)) { return rej("intermediate used outside the window"); }
    }

    // ---- structure ----
    const ggml_tensor * xn = nullptr;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->op == GGML_OP_MUL && t->src[0] == norm && is_f32_leaf(t->src[1], H, 1)) { xn = t; m.norm_w = t->src[1]; }
    }
    if (!xn) { return rej("no rms_norm*w"); }

    const ggml_tensor * mm_qkv = nullptr, * mm_gate = nullptr, * mm_ab[2] = { nullptr, nullptr }, * mm_out = nullptr;
    const ggml_tensor * gr_conv = nullptr, * gr_ssm = nullptr, * conv = nullptr, * silu_conv = nullptr, * gdn = nullptr;
    const ggml_tensor * zero_conv_root = nullptr, * zero_ssm_root = nullptr;
    const ggml_tensor * concat = nullptr;   // conv_input = concat(conv_state[3,8192], qkv^T[1,8192]) along dim 0
    const ggml_tensor * add_dt = nullptr, * softplus = nullptr, * mul_a = nullptr, * sigm = nullptr, * silu_z = nullptr;
    const ggml_tensor * rms_q = nullptr, * rms_k = nullptr, * rms_g = nullptr, * sc_q = nullptr, * sc_k = nullptr;
    const ggml_tensor * mul_norm = nullptr, * mul_z = nullptr, * cpy_conv = nullptr, * cpy_ssm = nullptr;
    int n_ab = 0;
    for (int j = 1; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        cur = i + j;
        switch (t->op) {
            case GGML_OP_MUL_MAT: {
                const ggml_tensor * w = t->src[0];
                if (vroot(t->src[1]) == xn && t->ne[1] == ns && t->ne[2] == 1) {
                    if (is_q4_mat(w, H, NQKV)) { mm_qkv = t; }
                    else if (is_q4_mat(w, H, DI)) { mm_gate = t; }
                    else if (is_q4_mat(w, H, NVH) && n_ab < 2) { mm_ab[n_ab++] = t; }
                    else { return rej("unexpected mul_mat on xn"); }
                } else if (is_q4_mat(w, DI, H) && t->ne[1] == 1 && t->ne[2] == ns) { mm_out = t; }
                else { return rej("unexpected mul_mat"); }
            } break;
            case GGML_OP_GET_ROWS: {
                if (t->src[1]->type != GGML_TYPE_I32 || vroot(t->src[1])->op != GGML_OP_NONE) { return rej("get_rows ids not an I32 leaf"); }
                if (ggml_nelements(t->src[1]) != ns || !ggml_is_contiguous(t->src[1])) { return rej("n_rs != n_seqs or noncontiguous ids"); }
                if (vroot(t->src[0])->op != GGML_OP_NONE || t->src[0]->type != GGML_TYPE_F32) { return rej("get_rows source not a leaf"); }
                if (m.s_copy && m.s_copy != t->src[1]) { return rej("two different s_copy tensors"); }
                m.s_copy = t->src[1];
                if (t->ne[0] == CONV_N && t->ne[1] == ns) { gr_conv = t; m.conv_base = (const float *) t->src[0]->data; m.conv_stride = t->src[0]->nb[1] / 4; }
                else if (t->ne[0] == SSM_N && t->ne[1] == ns) { gr_ssm = t; m.ssm_base = (const float *) t->src[0]->data; m.ssm_stride = t->src[0]->nb[1] / 4; }
                else { return rej("get_rows shape"); }
            } break;
            case GGML_OP_CONCAT: {
                if (concat || !gr_conv || vroot(t->src[0]) != gr_conv || !mm_qkv || vroot(t->src[1]) != mm_qkv || t->ne[0] != 4 || t->ne[1] != NQKV ||
                    t->ne[2] != ns || t->ne[3] != 1 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
                    return rej("concat not concat(conv_state, qkv^T)");
                }
                concat = t;
            } break;
            case GGML_OP_SSM_CONV: {
                if (t->src[2] != nullptr || !is_f32_leaf(t->src[1], 4, NQKV) || !concat || t->src[0] != concat) { return rej("ssm_conv shape"); }
                if (t->ne[0] != NQKV || t->ne[1] != 1 || t->ne[2] != ns) { return rej("ssm_conv output shape"); }
                conv = t; m.conv_w = t->src[1];
            } break;
            case GGML_OP_UNARY: {
                const ggml_unary_op u = ggml_get_unary_op(t);
                if (u == GGML_UNARY_OP_SILU && t->src[0] == conv) { silu_conv = t; }
                else if (u == GGML_UNARY_OP_SILU && mm_gate && vroot(t->src[0]) == mm_gate) { silu_z = t; }
                else if (u == GGML_UNARY_OP_SOFTPLUS && t->src[0] == add_dt) { softplus = t; }
                else if (u == GGML_UNARY_OP_SIGMOID && n_ab == 2 && (vroot(t->src[0]) == mm_ab[0] || vroot(t->src[0]) == mm_ab[1])) {
                    sigm = t; m.wbeta = vroot(t->src[0])->src[0];
                }
                else { return rej("unexpected unary"); }
            } break;
            case GGML_OP_ADD: {
                if (t == cgraph->nodes[end]) { break; }
                const ggml_tensor * a = t->src[0], * b = t->src[1];
                if (!is_f32_leaf(b, NVH, 1) || n_ab == 0 || !(vroot(a) == mm_ab[0] || (n_ab == 2 && vroot(a) == mm_ab[1]))) { return rej("unexpected add"); }
                add_dt = t; m.dt_bias = b; m.walpha = vroot(a)->src[0];
            } break;
            case GGML_OP_MUL: {
                if (t == xn) { break; }
                if (t->src[0] == softplus && is_f32_leaf(t->src[1], NVH, 1)) { mul_a = t; m.ssm_a = t->src[1]; }
                else if (t->src[0] == rms_g && is_f32_leaf(t->src[1], SD, 1)) { mul_norm = t; m.ssm_norm = t->src[1]; }
                else if (mul_norm && silu_z && ((t->src[0] == mul_norm && vroot(t->src[1]) == silu_z) || (t->src[1] == mul_norm && vroot(t->src[0]) == silu_z))) { mul_z = t; }
                else { return rej("unexpected mul"); }
            } break;
            case GGML_OP_RMS_NORM: {
                float eps = 0.f; memcpy(&eps, t->op_params, sizeof(float));
                const ggml_tensor * src = t->src[0];
                if (silu_conv && vroot(src) == silu_conv && src->ne[0] == SD && src->ne[1] == NKH && src->ne[2] == 1 && src->ne[3] == ns) {
                    if (src->view_offs == 0) { rms_q = t; m.eps_l2 = eps; }
                    else if (src->view_offs == (size_t) H * 4) { rms_k = t; if (eps != m.eps_l2) { return rej("q/k l2 eps differ"); } }
                    else { return rej("l2 rms_norm view offset"); }
                } else if (gdn && vroot(src) == gdn && src->view_offs == 0 && src->ne[0] == SD && src->ne[1] == NVH && src->ne[2] == 1 && src->ne[3] == ns) {
                    rms_g = t; m.eps_gated = eps;
                } else { return rej("unexpected rms_norm"); }
            } break;
            case GGML_OP_SCALE: {
                if (ggml_nelements(t) == 0) { break; }   // build_rs state_zero (rs_zero < 0)
                float sc = 0.f, bias = 0.f; memcpy(&sc, t->op_params, sizeof(float)); memcpy(&bias, (const float *) t->op_params + 1, sizeof(float));
                if (bias != 0.f) { return rej("scale with bias"); }
                if (t->src[0] == rms_q) { sc_q = t; m.scale_l2 = sc; }
                else if (t->src[0] == rms_k) { sc_k = t; if (sc != m.scale_l2) { return rej("q/k l2 scale differ"); } }
                else if (sc == 0.f && t->view_src && vroot(t)->op == GGML_OP_NONE && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
                         t->src[0]->view_src && t->data == t->src[0]->data) {
                    // build_rs state_zero (rs_zero >= 0): in-place scale by 0 of one state row; applied as a memset before the kernel
                    if (ggml_nelements(t) == CONV_N && !m.zero_conv) { m.zero_conv = (float *) t->data; zero_conv_root = vroot(t); }
                    else if (ggml_nelements(t) == SSM_N && !m.zero_ssm) { m.zero_ssm = (float *) t->data; zero_ssm_root = vroot(t); }
                    else { return rej("state_zero scale shape"); }
                }
                else { return rej("unexpected scale"); }
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                if (ggml_get_op_params_i32(t, 0) != 1) { return rej("gdn K != 1"); }
                const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2], * g = t->src[3], * b = t->src[4], * st = t->src[5];
                if (!sc_q || !sc_k || !silu_conv || !mul_a || !sigm || !gr_ssm) { return rej("gdn inputs appear before their producers"); }
                if (vroot(q) != sc_q || vroot(k) != sc_k || vroot(v) != silu_conv || v->view_offs != (size_t) 2 * H * 4 ||
                    vroot(g) != mul_a || vroot(b) != sigm || vroot(st) != gr_ssm) { return rej("gdn wiring"); }
                if (v->ne[0] != SD || v->ne[1] != NVH || v->ne[2] != 1 || v->ne[3] != ns || q->ne[1] != NKH || g->ne[0] != 1 ||
                    st->ne[0] != SD || st->ne[1] != SD || st->ne[2] != NVH || st->ne[3] != ns) { return rej("gdn shapes"); }
                gdn = t;
            } break;
            case GGML_OP_CPY: {
                if (ggml_nelements(t) == 0) { break; }   // build_rs states_extra (n_rs == n_seqs)
                const ggml_tensor * src = t->src[0];
                if (concat && vroot(src) == concat && src->view_offs == 4 && src->ne[0] == 3 && src->ne[1] == NQKV && src->ne[2] == ns &&
                    gr_conv && vroot(t) == vroot(gr_conv->src[0]) && ggml_nelements(t) == CONV_N*ns && ggml_is_contiguous(t) && t->type == GGML_TYPE_F32) {
                    cpy_conv = t; m.conv_out = (float *) t->data;
                } else if (gdn && vroot(src) == gdn && src->view_offs == (size_t) SD * NVH * ns * 4 && ggml_nelements(src) == SSM_N*ns && ggml_is_contiguous(src) &&
                    gr_ssm && vroot(t) == vroot(gr_ssm->src[0]) && ggml_nelements(t) == SSM_N*ns && ggml_is_contiguous(t) && t->type == GGML_TYPE_F32) {
                    cpy_ssm = t; m.state_out = (float *) t->data;
                } else { return rej("unexpected cpy"); }
            } break;
            default: break;   // views / reshape / transpose / permute / cont / concat: checked through their consumers
        }
    }
    if (!mm_qkv || !mm_gate || n_ab != 2 || !mm_out || !gr_conv || !gr_ssm || !conv || !silu_conv || !gdn || !add_dt || !softplus ||
        !mul_a || !sigm || !silu_z || !rms_q || !rms_k || !rms_g || !sc_q || !sc_k || !mul_norm || !mul_z || !cpy_conv || !cpy_ssm) {
        return rej("missing GDN block component");
    }
    if ((zero_conv_root && zero_conv_root != vroot(gr_conv->src[0])) || (zero_ssm_root && zero_ssm_root != vroot(gr_ssm->src[0]))) {
        return rej("state_zero row not in the get_rows source");
    }
    if (m.walpha == m.wbeta) { return rej("alpha/beta resolve to the same weight"); }
    if (vroot(mm_out->src[1]) != mul_z) { return rej("ssm_out input is not the gated norm output"); }
    ggml_tensor * out = cgraph->nodes[end];
    if (!((vroot(out->src[0]) == mm_out && out->src[1] == h) || (vroot(out->src[1]) == mm_out && out->src[0] == h))) { return rej("residual add wiring"); }
    if (out->type != GGML_TYPE_F32 || out->ne[0] != H || ggml_nrows(out) != ns || !ggml_is_contiguous(out)) { return rej("output shape"); }
    if (m.conv_stride < CONV_N || m.ssm_stride < SSM_N || ((uintptr_t) m.conv_base & 15) || ((uintptr_t) m.ssm_base & 15) ||
        ((uintptr_t) m.conv_out & 15) || ((uintptr_t) m.state_out & 15)) { return rej("state layout/alignment"); }

    if(ns==2 && (m.conv_stride!=CONV_N || m.ssm_stride!=SSM_N ||
        ((uintptr_t)m.conv_out-(uintptr_t)m.conv_base)%(CONV_N*sizeof(float)) ||
        ((uintptr_t)m.state_out-(uintptr_t)m.ssm_base)%(SSM_N*sizeof(float)))) { return rej("paired state rows must be contiguous and aligned"); }
    if(ns==2) {
        if(!ggml_are_same_stride(gdn->src[0],gdn->src[1]) || !ggml_is_contiguous_rows(gdn->src[0]) ||
           !ggml_is_contiguous_rows(gdn->src[2]) || !ggml_is_contiguous(gdn->src[3]) || !ggml_is_contiguous(gdn->src[4]) ||
           gdn->src[0]->ne[0]!=SD || gdn->src[0]->ne[3]!=2 || gdn->src[3]->ne[0]!=1) return rej("direct state input stride/shape");
        for(int j=0;j<n;++j){
            const ggml_tensor*t=cgraph->nodes[i+j];
            for(int k=0;k<GGML_MAX_SRC;++k){
                if(!t->src[k] || vroot(t->src[k])!=gr_ssm)continue;
                if((t==gdn && k==5) || t->op==GGML_OP_VIEW || t->op==GGML_OP_RESHAPE ||
                   (t->op==GGML_OP_CPY && ggml_nelements(t)==0))continue;
                return rej("direct state gather has another consumer");
            }
        }
        m.gr_ssm=gr_ssm;m.gdn=gdn;m.cpy_ssm=cpy_ssm;
    }
    memcpy(&m.eps_attn, norm->op_params, sizeof(float));
    m.n_nodes = n; m.h = h; m.out = out;
    m.wqkv = mm_qkv->src[0]; m.wgate = mm_gate->src[0]; m.wout = mm_out->src[0];
    for (const ggml_tensor * w : { m.wqkv, m.wgate, m.walpha, m.wbeta, m.wout, m.conv_w, m.dt_bias, m.ssm_a, m.ssm_norm, m.norm_w }) {
        if (!on_device(ctx, w)) { return rej("weight not in plain CUDA buffer"); }
    }
    DevState & d = dev_state(ctx.device);
    if (d.grid <= 0) { return rej("occupancy 0"); }
    if(ns==2 && g_debug) { static bool once=false;if(!once){fprintf(stderr,"occamy-state-direct: matched two-sequence window, n_rs=2\n");once=true;} }
    g_engaged++;
    if (g_debug && ns==1 && (g_engaged == 1 || g_engaged % 3000 == 0)) {
        fprintf(stderr, "occamy-gdn: fused GDN window %d..%d (%d nodes) %s -> %s; engaged %ld, GDN windows rejected %ld\n",
                i, end, n, norm->name, out->name, g_engaged, g_rejected_gdn);
    }
    return true;
}

void ggml_cuda_occamy_gdn_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m, float * out_override) {
    static const bool three = env_flag("GGML_CUDA_OCCAMY_GDN_3L");
    DevState & d = dev_state(ctx.device);
    Params P;
    P.L.attn_norm = (const float *) m.norm_w->data;
    P.L.conv_w    = (const float *) m.conv_w->data;
    P.L.ssm_a     = (const float *) m.ssm_a->data;
    P.L.dt_bias   = (const float *) m.dt_bias->data;
    P.L.ssm_norm  = (const float *) m.ssm_norm->data;
    P.L.qkv   = (const uint8_t *) m.wqkv->data;
    P.L.gate  = (const uint8_t *) m.wgate->data;
    P.L.alpha = (const uint8_t *) m.walpha->data;
    P.L.beta  = (const uint8_t *) m.wbeta->data;
    P.L.out   = (const uint8_t *) m.wout->data;
    P.h   = (const float *) m.h->data;
    P.out = out_override ? out_override : (float *) m.out->data;
    P.s_copy = (const int32_t *) m.s_copy->data;
    P.conv_base = m.conv_base; P.conv_stride = m.conv_stride; P.conv_out = m.conv_out;
    P.ssm_base  = m.ssm_base;  P.ssm_stride  = m.ssm_stride;  P.S_out    = m.state_out;
    P.s = d.scratch;
    P.eps_attn = m.eps_attn; P.eps_l2 = m.eps_l2; P.scale_l2 = m.scale_l2; P.eps_gated = m.eps_gated;
    cudaStream_t st = ctx.stream();
    if (m.zero_conv) { CUDA_CHECK(cudaMemsetAsync(m.zero_conv, 0, CONV_N * sizeof(float), st)); }
    if (m.zero_ssm)  { CUDA_CHECK(cudaMemsetAsync(m.zero_ssm,  0, SSM_N  * sizeof(float), st)); }
    const size_t smem = SMEM2_FLOATS * sizeof(float);
    if (three) {
        gdn_fused<1><<<d.grid, NT, smem, st>>>(P);
        gdn_fused<2><<<d.grid, NT, smem, st>>>(P);
        gdn_fused<3><<<d.grid, NT, smem, st>>>(P);
    } else {
        gdn_fused<0><<<d.grid, NT, smem, st>>>(P);
    }
    CUDA_CHECK(cudaGetLastError());
}
