// native-occamy Track D (12 Oct 2026): fused qwen35moe MoE-block DECODE kernel (batch 1), sm_52.
// Kernel = native-occamy/decode/moe_fused.cu moe_v2 (d2): coalesced uint4 loads staged through per-warp smem.
//   A: rmsnorm(h)*w (per block) + router logits 256x2048 F32 and the shared-expert gate scalar, quarter-row partials
//   B: softmax + top-8 + renormalise (per block, redundant); shexp gate/up + 8x512 routed gate/up rows -> silu*mul
//   C: per output row: 8 routed down rows weighted + shared down row * sigmoid(gate) + residual h
// One persistent launch with an occupancy-sized grid and a bounded-spin grid barrier (__trap on timeout),
// or 3 launches (GGML_CUDA_OCCAMY_3L=1). Activations stay fp32 (no q8_1 requantisation).
#include "occamy-fused.cuh"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

namespace occamy {

constexpr int H = 2048, NE = 256, TOPK = 8, FF = 512, FFS = 512;
constexpr int QB = 18;
constexpr int ROW_GU = H / 32 * QB;    // 1152 B
constexpr int ROW_D  = FF / 32 * QB;   // 288 B
constexpr int NT = 256, NW = NT / 32;
constexpr int CH = 68;
constexpr int NQ = 4, QK = H / NQ;
constexpr int NTA = NQ * (NE + 1), NTB = FFS + TOPK * FF;
constexpr int STW = 648;
constexpr int SMEM_FLOATS  = (TOPK * FF + FFS) / 64 * CH + (TOPK * FF + FFS) / 32 * 2 / 2 * 2 + 64;
constexpr int SMEM2_FLOATS = SMEM_FLOATS + NW * STW;
constexpr unsigned SPIN_LIMIT = 1u << 26;   // ~seconds of spinning -> a non-co-resident grid traps instead of hanging

struct Layer {
    const float * gate_inp;
    const float * gate_inp_sh;
    const uint8_t * gate_exps, * up_exps, * down_exps;
    const uint8_t * gate_sh, * up_sh, * down_sh;
};
struct Scratch {
    float logits[NE];
    float h_sh[FFS];
    float gsh;
    float h_exp[TOPK * FF];
    float lpart[NQ][NE], gpart[NQ];
    unsigned bar_count, bar_gen;
};
struct Params {
    Layer L;
    const float * h;
    const float * norm;
    float * out;
    Scratch * s;
    float eps;
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float wsum(float v) {
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
            while (*genp == gen) {
                if (++k > SPIN_LIMIT) { __trap(); }
            }
        }
        __threadfence();
    }
    __syncthreads();
}
__device__ __forceinline__ float nib(unsigned v, int k) {
    return __int_as_float(__byte_perm(v, 0x3F000000u, 0x7044u | (k << 8)));
}
struct Q9 { unsigned w[9]; };
__device__ __forceinline__ Q9 lds9(const unsigned * p) { Q9 q;
#pragma unroll
    for (int i = 0; i < 9; ++i) q.w[i] = p[i];
    return q; }
__device__ __forceinline__ float dot2c(const Q9 & q, const float * xc, float2 sx) {
    const unsigned w0 = q.w[0], w1 = q.w[1], w2 = q.w[2], w3 = q.w[3], w4 = q.w[4];
    float d0 = __half2float(__ushort_as_half((unsigned short)(w0 & 0xFFFF)));
    float d1 = __half2float(__ushort_as_half((unsigned short)(w4 >> 16)));
    unsigned q0[4] = { __byte_perm(w0, w1, 0x5432), __byte_perm(w1, w2, 0x5432), __byte_perm(w2, w3, 0x5432), __byte_perm(w3, w4, 0x5432) };
    unsigned q1[4] = { q.w[5], q.w[6], q.w[7], q.w[8] };
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
__device__ __forceinline__ void load_norm(float * xs, float2 * sums, const float * h, const float * w, float * red, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += NT) { float v = h[i]; xs[(i >> 6) * CH + (i & 63)] = v; ss += v * v; }
    ss = wsum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += red[i];
    const float sc = rsqrtf(tot / H + eps);
    for (int i = threadIdx.x; i < H; i += NT) xs[(i >> 6) * CH + (i & 63)] *= sc * __ldg(w + i);
    __syncthreads();
    for (int b = warp; b < H / 32; b += NW) {
        float s = wsum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
        if (lane == 0) ((float *)sums)[b] = s;
    }
    __syncthreads();
}
__device__ __forceinline__ void ldA(const Layer & L, int t, int lane, float4 * w) {
    const int row = t >> 2, q = t & 3;
    const float4 * p = (const float4 *)((row < NE ? L.gate_inp + (size_t)row * H : L.gate_inp_sh) + q * QK);
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = __ldg(p + lane + 32 * i);
}
__device__ __forceinline__ void rowB(const Layer & L, const int * sel, int t, const uint8_t *& g, const uint8_t *& u) {
    if (t < FFS) { g = L.gate_sh + (size_t)t * ROW_GU; u = L.up_sh + (size_t)t * ROW_GU; }
    else { const int k = (t - FFS) / FF, r = (t - FFS) % FF; const size_t off = ((size_t)sel[k] * FF + r) * ROW_GU; g = L.gate_exps + off; u = L.up_exps + off; }
}
__device__ __forceinline__ void ldB(const uint8_t * g, const uint8_t * u, int lane, uint4 * v) {
    const uint4 * g4 = (const uint4 *)g, * u4 = (const uint4 *)u;
    v[0] = __ldg(g4 + lane); v[1] = __ldg(g4 + lane + 32); v[3] = __ldg(u4 + lane); v[4] = __ldg(u4 + lane + 32);
    if (lane < 8) { v[2] = __ldg(g4 + lane + 64); v[5] = __ldg(u4 + lane + 64); }
}
__device__ __forceinline__ void route(const Scratch * s, int lane, int * sel, float * selw) {
    float v[NE / 32];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) {
        float a = 0.f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) a += __ldcg(&s->lpart[q][lane + 32 * i]);
        v[i] = a; m = fmaxf(m, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) { v[i] = expf(v[i] - m); sum += v[i]; }
    sum = wsum(sum);
    const float inv = 1.f / sum;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) v[i] *= inv;
    float ws = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < NE / 32; ++i) if (v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
#pragma unroll
        for (int i = 0; i < NE / 32; ++i) if (bi == lane + 32 * i) v[i] = -2.f;
        if (lane == 0) { sel[k] = bi; selw[k] = bv; }
        ws += bv;
    }
    ws = fmaxf(ws, 6.103515625e-5f);
    __syncwarp();
    if (lane < TOPK) selw[lane] /= ws;
}

template <int PHASE>
__global__ void __launch_bounds__(NT, 2) moe_v2(Params P) {
    extern __shared__ float smem[];
    __shared__ int sel[TOPK];
    __shared__ float selw[TOPK];
    __shared__ float red[NW];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + (TOPK * FF + FFS) / 64 * CH);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + warp * STW;
    uint4 * stg4 = (uint4 *)stg;
    const int gw = blockIdx.x * NW + warp, W = gridDim.x * NW;
    Scratch * s = P.s;
    const Layer & L = P.L;

    if (PHASE == 0 || PHASE == 1) {
        float4 cw[4], nw[4];
        int t = gw;
        if (t < NTA) ldA(L, t, lane, cw);
        load_norm(xs, sums, P.h, P.norm, red, P.eps);
        for (; t < NTA; t += W) {
            const int tn = t + W;
            if (tn < NTA) ldA(L, tn, lane, nw);
            const int row = t >> 2, q = t & 3;
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int e = q * (QK / 4) + lane + 32 * i;
                const float4 xv = *(const float4 *)(xs + ((4 * e) >> 6) * CH + ((4 * e) & 63));
                a = fmaf(cw[i].x, xv.x, a); a = fmaf(cw[i].y, xv.y, a); a = fmaf(cw[i].z, xv.z, a); a = fmaf(cw[i].w, xv.w, a);
            }
            a = wsum(a);
            if (lane == 0) { if (row < NE) s->lpart[q][row] = a; else s->gpart[q] = a; }
#pragma unroll
            for (int i = 0; i < 4; ++i) cw[i] = nw[i];
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 2) {
        uint4 cv[6];
        bool have = false;
        const uint8_t * g, * u;
        if (gw < FFS) { rowB(L, sel, gw, g, u); ldB(g, u, lane, cv); have = true; }
        if (PHASE == 2) load_norm(xs, sums, P.h, P.norm, red, P.eps);
        if (warp == 0) route(s, lane, sel, selw);
        __syncthreads();
        const float * xc = xs + lane * CH; const float2 sx = sums[lane];
        for (int t = gw; t < NTB; t += W) {
            if (!have) { rowB(L, sel, t, g, u); ldB(g, u, lane, cv); }
            stg4[lane] = cv[0]; stg4[lane + 32] = cv[1]; stg4[72 + lane] = cv[3]; stg4[104 + lane] = cv[4];
            if (lane < 8) { stg4[lane + 64] = cv[2]; stg4[136 + lane] = cv[5]; }
            __syncwarp();
            const int tn = t + W; have = tn < NTB;
            if (have) { rowB(L, sel, tn, g, u); ldB(g, u, lane, cv); }
            float gg = wsum(dot2c(lds9(stg + lane * 9), xc, sx));
            float uu = wsum(dot2c(lds9(stg + 288 + lane * 9), xc, sx));
            if (lane == 0) { if (t < FFS) s->h_sh[t] = silu(gg) * uu; else s->h_exp[t - FFS] = silu(gg) * uu; }
            __syncwarp();
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 3) {
        if (PHASE == 3) { if (warp == 0) route(s, lane, sel, selw); __syncthreads(); }
        const uint8_t * cb[6];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int uidx = lane + 32 * i, m = uidx / 18, w = uidx % 18;
            cb[i] = (m < TOPK ? L.down_exps + (size_t)sel[m < TOPK ? m : 0] * H * ROW_D : L.down_sh) + w * 16;
        }
        uint4 cv[6];
        int r = gw;
        if (r < H) {
#pragma unroll
            for (int i = 0; i < 5; ++i) cv[i] = __ldg((const uint4 *)(cb[i] + (size_t)r * ROW_D));
            if (lane < 2) cv[5] = __ldg((const uint4 *)(cb[5] + (size_t)r * ROW_D));
        }
        for (int i = threadIdx.x; i < TOPK * FF + FFS; i += NT) {
            float val = i < TOPK * FF ? __ldcg(&s->h_exp[i]) : __ldcg(&s->h_sh[i - TOPK * FF]);
            xs[(i >> 6) * CH + (i & 63)] = val;
        }
        __syncthreads();
        for (int b = warp; b < (TOPK * FF + FFS) / 32; b += NW) {
            float sm = wsum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
            if (lane == 0) ((float *)sums)[b] = sm;
        }
        __syncthreads();
        float gl = 0.f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) gl += __ldcg(&s->gpart[q]);
        const float gs = 1.f / (1.f + expf(-gl));
        const float w0 = selw[lane >> 3], w1 = selw[(lane >> 3) + 4];
        for (; r < H; r += W) {
#pragma unroll
            for (int i = 0; i < 5; ++i) stg4[lane + 32 * i] = cv[i];
            if (lane < 2) stg4[160 + lane] = cv[5];
            __syncwarp();
            const int rn = r + W;
            if (rn < H) {
#pragma unroll
                for (int i = 0; i < 5; ++i) cv[i] = __ldg((const uint4 *)(cb[i] + (size_t)rn * ROW_D));
                if (lane < 2) cv[5] = __ldg((const uint4 *)(cb[5] + (size_t)rn * ROW_D));
            }
            float acc = w0 * dot2c(lds9(stg + lane * 9), xs + lane * CH, sums[lane])
                      + w1 * dot2c(lds9(stg + (lane + 32) * 9), xs + (lane + 32) * CH, sums[lane + 32]);
            if (lane < 8) acc += gs * dot2c(lds9(stg + (lane + 64) * 9), xs + (lane + 64) * CH, sums[lane + 64]);
            acc = wsum(acc);
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

static void reject(const char * why) {
    if (!g_debug) { return; }
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (seen.insert(why).second) {
        fprintf(stderr, "occamy-fused: window rejected: %s\n", why);
    }
}

static DevState & dev_state(int device) {
    DevState & d = g_dev[device];
    if (d.scratch == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **)&d.scratch, sizeof(Scratch)));
        CUDA_CHECK(cudaMemset(d.scratch, 0, sizeof(Scratch)));
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, moe_v2<0>, NT, SMEM2_FLOATS * sizeof(float)));
        const int nsm = ggml_cuda_info().devices[device].nsm;
        d.grid = std::max(1, per_sm) * nsm;   // all blocks co-resident -> grid barrier is safe
        if (g_debug) {
            fprintf(stderr, "occamy-fused: device %d: %d SMs x %d blocks/SM -> grid %d, smem %zu B\n",
                    device, nsm, per_sm, d.grid, SMEM2_FLOATS * sizeof(float));
        }
        if (per_sm < 1) {
            d.grid = -1;
        }
    }
    return d;
}

} // namespace occamy

using namespace occamy;

bool ggml_cuda_occamy_enabled() {
    static const bool on = env_flag("GGML_CUDA_OCCAMY_FUSED");
    return on;
}

int ggml_cuda_occamy_check_budget() {
    static int left = getenv("GGML_CUDA_OCCAMY_CHECK") ? atoi(getenv("GGML_CUDA_OCCAMY_CHECK")) : 0;
    if (left > 0) { return left--; }
    return 0;
}

static bool is_q4_mat(const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2) {
    return t && t->op == GGML_OP_NONE && t->type == GGML_TYPE_Q4_0 && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 &&
           t->ne[3] == 1 && ggml_is_contiguous(t) && ((uintptr_t)t->data & 15) == 0;
}
// Track T 13 Oct 2026: build_moe_ffn feeds mul_mat_id with ggml_reshape_3d(xn, H, 1, n_tokens), so src[1] is a
// RESHAPE node aliasing xn, never xn itself; the matcher rejected every window as "unexpected mul_mat_id".
static const ggml_tensor * strip_views(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->src[0] && t->view_offs == 0 &&
           ggml_nelements(t) == ggml_nelements(t->src[0])) { t = t->src[0]; }
    return t;
}
static bool is_f32_mat(const ggml_tensor * t, int64_t ne0, int64_t ne1) {
    return t && t->op == GGML_OP_NONE && t->type == GGML_TYPE_F32 && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 &&
           t->ne[3] == 1 && ggml_is_contiguous(t) && ((uintptr_t)t->data & 15) == 0;
}

bool ggml_cuda_occamy_moe_match_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, ggml_cuda_occamy_moe_match & m) {
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM) { return false; }
    const ggml_tensor * h = norm->src[0];
    if (h->type != GGML_TYPE_F32 || h->ne[0] != H || ggml_nrows(h) != 1 || !ggml_is_contiguous(h)) { return false; }

    // window end: first ADD after i that takes h directly (the FFN residual add)
    int end = -1;
    const int lim = std::min(cgraph->n_nodes, i + 128);
    for (int j = i + 1; j < lim; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_ADD && (t->src[0] == h || t->src[1] == h)) { end = j; break; }
    }
    if (end < 0) { reject("no residual ADD within 128 nodes"); return false; }
    const int n = end - i + 1;
    if (n < 20) { reject("window too short"); return false; }

    // sorted window members for membership tests
    const ggml_tensor * win[128];
    for (int j = 0; j < n; ++j) { win[j] = cgraph->nodes[i + j]; }
    std::sort(win, win + n);
    auto in_win = [&](const ggml_tensor * t) { return std::binary_search(win, win + n, t); };

    int c_mm = 0, c_mmid = 0, c_glu = 0, c_sm = 0, c_sort = 0, c_un = 0, c_div = 0, c_sum = 0, c_gr = 0, c_norm = 0;
    const ggml_tensor * xn = nullptr;
    const ggml_tensor * router = nullptr, * shgate = nullptr, * sh_gate_mm = nullptr, * sh_up_mm = nullptr, * sh_down = nullptr;
    const ggml_tensor * glu_r = nullptr, * glu_s = nullptr, * down_id = nullptr;

    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { reject("window node flagged as output"); return false; }
        if (t->ne[2] != 1 && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) {
            // mul_mat_id outputs are [n, 8, 1]; anything with a token dim > 1 fails here
            if (t->ne[2] != 1) { reject("n_tokens != 1"); return false; }
        }
        switch (t->op) {
            case GGML_OP_RMS_NORM: c_norm++; break;
            case GGML_OP_MUL_MAT:  c_mm++;   break;
            case GGML_OP_MUL_MAT_ID: c_mmid++; break;
            case GGML_OP_GLU:      c_glu++;  break;
            case GGML_OP_SOFT_MAX: c_sm++;   break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K:    c_sort++; break;
            case GGML_OP_UNARY:    c_un++;   break;
            case GGML_OP_DIV:      c_div++;  break;
            case GGML_OP_SUM_ROWS: c_sum++;  break;
            case GGML_OP_GET_ROWS: c_gr++;   break;
            case GGML_OP_ADD: case GGML_OP_MUL: case GGML_OP_CLAMP:
            case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: case GGML_OP_CONT:
                break;
            default: reject(ggml_op_name(t->op)); return false;
        }
        // inputs: window members, h, or leaf tensors (weights)
        int inner_uses_dummy = 0; (void)inner_uses_dummy;
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = t->src[k];
            if (s == nullptr) { continue; }
            if (s == h || in_win(s)) { continue; }
            if (s->op == GGML_OP_NONE) {
                if (s->buffer == nullptr || s->buffer->buft != ggml_backend_cuda_buffer_type(ctx.device)) {
                    reject("leaf input not in this device's plain CUDA buffer"); return false;
                }
                continue;
            }
            reject("external computed input inside window"); return false;
        }
    }
    if (c_norm != 1 || c_mm != 5 || c_mmid != 3 || c_glu != 2 || c_sm != 1 || c_sort != 1 || c_un != 1 || c_div != 1 || c_sum != 1 || c_gr != 1) {
        reject("op counts differ from the qwen35moe MoE block"); return false;
    }
    // every node except the last is consumed only inside the window
    for (int j = 0; j < n - 1; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        int uses = 0;
        for (int q = 0; q < n; ++q) {
            const ggml_tensor * u = cgraph->nodes[i + q];
            for (int k = 0; k < GGML_MAX_SRC; ++k) { if (u->src[k] == t) { uses++; } }
        }
        if (uses != ggml_node_get_use_count(cgraph, i + j)) { reject("intermediate used outside the window"); return false; }
    }

    // structure
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->op == GGML_OP_MUL && t->src[0] == norm && is_f32_mat(t->src[1], H, 1)) { xn = t; m.norm_w = t->src[1]; }
    }
    if (!xn) { reject("no rms_norm*w"); return false; }
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->op == GGML_OP_MUL_MAT) {
            const ggml_tensor * w = t->src[0];
            if (t->src[1] == xn && is_f32_mat(w, H, NE)) { router = t; }
            else if (t->src[1] == xn && is_f32_mat(w, H, 1)) { shgate = t; }
            else if (t->src[1] == xn && is_q4_mat(w, H, FFS, 1)) { /* shexp gate or up, resolved via GLU */ }
            else if (is_q4_mat(w, FFS, H, 1)) { sh_down = t; }
            else { reject("unexpected mul_mat"); return false; }
        } else if (t->op == GGML_OP_GLU) {
            if (ggml_get_glu_op(t) != GGML_GLU_OP_SWIGLU || t->src[1] == nullptr || ggml_get_op_params_i32(t, 1) != 0) {
                reject("glu not a split, unswapped swiglu"); return false;
            }
            if (t->src[0]->op == GGML_OP_MUL_MAT_ID) { glu_r = t; } else if (t->src[0]->op == GGML_OP_MUL_MAT) { glu_s = t; }
        } else if (t->op == GGML_OP_MUL_MAT_ID) {
            if (t->src[2] == nullptr || t->src[2]->type != GGML_TYPE_I32 || t->src[2]->ne[0] != TOPK || t->src[2]->ne[1] != 1) {
                reject("mul_mat_id ids not [8,1]"); return false;
            }
            if (is_q4_mat(t->src[0], FF, H, NE)) { down_id = t; }
            else if (!(is_q4_mat(t->src[0], H, FF, NE) && strip_views(t->src[1]) == xn)) { reject("unexpected mul_mat_id"); return false; }
        } else if (t->op == GGML_OP_SOFT_MAX) {
            float scale = 1.f, max_bias = 0.f;
            memcpy(&scale, (const float *)t->op_params + 0, sizeof(float));
            memcpy(&max_bias, (const float *)t->op_params + 1, sizeof(float));
            if (t->src[1] != nullptr || scale != 1.f || max_bias != 0.f || t->src[0] != router) {
                reject("soft_max not plain softmax over router logits"); return false;
            }
        } else if (t->op == GGML_OP_UNARY) {
            if (ggml_get_unary_op(t) != GGML_UNARY_OP_SIGMOID || t->src[0] != shgate) { reject("unary not sigmoid(shexp gate)"); return false; }
        }
    }
    if (!router || !shgate || !sh_down || !glu_r || !glu_s || !down_id) { reject("missing block component"); return false; }
    const ggml_tensor * g_r = glu_r->src[0], * u_r = glu_r->src[1];
    const ggml_tensor * g_s = glu_s->src[0], * u_s = glu_s->src[1];
    if (u_r->op != GGML_OP_MUL_MAT_ID || u_s->op != GGML_OP_MUL_MAT || g_s->src[1] != xn || u_s->src[1] != xn ||
        !is_q4_mat(g_s->src[0], H, FFS, 1) || !is_q4_mat(u_s->src[0], H, FFS, 1) ||
        !is_q4_mat(g_r->src[0], H, FF, NE) || !is_q4_mat(u_r->src[0], H, FF, NE) ||
        down_id->src[1] != glu_r || sh_down->src[1] != glu_s || g_r->src[2] != u_r->src[2] || down_id->src[2] != g_r->src[2]) {
        reject("gate/up/down wiring differs"); return false;
    }
    float eps = 0.f;
    memcpy(&eps, norm->op_params, sizeof(float));
    ggml_tensor * out = cgraph->nodes[end];
    if (out->type != GGML_TYPE_F32 || out->ne[0] != H || ggml_nrows(out) != 1 || !ggml_is_contiguous(out)) { reject("output shape"); return false; }

    m.n_nodes = n; m.h = h; m.eps = eps; m.out = out;
    m.gate_inp = router->src[0]; m.gate_inp_sh = shgate->src[0];
    m.gate_exps = g_r->src[0]; m.up_exps = u_r->src[0]; m.down_exps = down_id->src[0];
    m.gate_sh = g_s->src[0]; m.up_sh = u_s->src[0]; m.down_sh = sh_down->src[0];
    for (const ggml_tensor * w : { m.gate_inp, m.gate_inp_sh, m.gate_exps, m.up_exps, m.down_exps, m.gate_sh, m.up_sh, m.down_sh, m.norm_w }) {
        if (w->buffer == nullptr || w->buffer->buft != ggml_backend_cuda_buffer_type(ctx.device)) { reject("weight not in plain CUDA buffer"); return false; }
    }
    DevState & d = dev_state(ctx.device);
    if (d.grid <= 0) { reject("occupancy 0"); return false; }
    if (g_debug) {
        static bool once = false;
        if (!once) { once = true; fprintf(stderr, "occamy-fused: fused MoE window %d..%d (%d nodes) %s -> %s\n", i, end, n, norm->name, out->name); }
    }
    return true;
}

void ggml_cuda_occamy_moe_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_moe_match & m, float * out_override) {
    static const bool three = env_flag("GGML_CUDA_OCCAMY_3L");
    DevState & d = dev_state(ctx.device);
    Params P;
    P.L.gate_inp    = (const float *) m.gate_inp->data;
    P.L.gate_inp_sh = (const float *) m.gate_inp_sh->data;
    P.L.gate_exps   = (const uint8_t *) m.gate_exps->data;
    P.L.up_exps     = (const uint8_t *) m.up_exps->data;
    P.L.down_exps   = (const uint8_t *) m.down_exps->data;
    P.L.gate_sh     = (const uint8_t *) m.gate_sh->data;
    P.L.up_sh       = (const uint8_t *) m.up_sh->data;
    P.L.down_sh     = (const uint8_t *) m.down_sh->data;
    P.h    = (const float *) m.h->data;
    P.norm = (const float *) m.norm_w->data;
    P.out  = out_override ? out_override : (float *) m.out->data;
    P.s    = d.scratch;
    P.eps  = m.eps;
    cudaStream_t st = ctx.stream();
    const size_t smem = SMEM2_FLOATS * sizeof(float);
    if (three) {
        moe_v2<1><<<d.grid, NT, smem, st>>>(P);
        moe_v2<2><<<d.grid, NT, smem, st>>>(P);
        moe_v2<3><<<d.grid, NT, smem, st>>>(P);
    } else {
        moe_v2<0><<<d.grid, NT, smem, st>>>(P);
    }
    CUDA_CHECK(cudaGetLastError());
}
