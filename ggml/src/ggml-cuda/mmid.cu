#include "common.cuh"
#include "mmid.cuh"

#include <cstdlib>
#include <cstring>

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

// Eight warps process consecutive token slices of one expert. Routing IDs
// are distinct within a token, as required by the existing mm_ids_helper.
// Concatenating the slices preserves exactly the old token/slot order.
__launch_bounds__(256, 2)
static __global__ void mm_ids_helper_route8(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_tokens, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int expert = blockIdx.x;
    const int slice = (n_tokens + 7) / 8;
    const int begin = warp * slice;
    const int end = min(begin + slice, n_tokens);
    extern __shared__ uint32_t store_route8[];
    __shared__ int matches[8];
    __shared__ int lower[8];
    int compact = 0;
    int less = 0;
    for (int it0 = begin; it0 < end; it0 += 4) {
        const int it = it0 + lane / 8;
        const int slot = lane % 8;
        const int used = it < end ? ids[it * si1 + slot] : INT_MAX;
        less += used < expert;
        const unsigned int mask = __ballot_sync(0xffffffffu, used == expert);
        if (used == expert) {
            const int pos = compact + __popc(mask & ((1u << lane) - 1u));
            store_route8[warp * slice + pos] = static_cast<uint32_t>(it) | (static_cast<uint32_t>(slot) << 22);
        }
        compact += __popc(mask);
    }
    less = warp_reduce_sum<32>(less);
    if (lane == 0) {
        matches[warp] = compact;
        lower[warp] = less;
    }
    __syncthreads();
    int base = 0;
    int preceding = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        base += lower[w];
        preceding += w < warp ? matches[w] : 0;
    }
    for (int i = lane; i < compact; i += 32) {
        const uint32_t packed = store_route8[warp * slice + i];
        const int it = packed & 0x003fffff;
        const int slot = packed >> 22;
        const int index = base + preceding + i;
        ids_dst[index] = it * 8 + slot;
        if (write_inverse) {
            ids_src1[it * 8 + slot] = index;
        } else {
            ids_src1[index] = it * sis1 + slot % nchannels_y;
        }
    }
    if (threadIdx.x == 0) {
        expert_bounds[expert] = base;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            int total = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                total += matches[w];
            }
            expert_bounds[expert + 1] = base + total;
        }
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream, const bool allow_route8) {
    static const bool route8_enabled = [] {
        const char * e = std::getenv("GGML_CUDA_MMQ_ROUTE8_SM52");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    const int id = ggml_cuda_get_device();
    if (allow_route8 && route8_enabled && ggml_cuda_info().devices[id].cc == 520 &&
            n_expert_used == 8 && n_experts >= 8 && n_experts <= 256 &&
            n_tokens >= 256 && n_tokens <= 2048 && nchannels_y > 0) {
        const size_t nbytes_shared = 8 * ((n_tokens + 7) / 8) * sizeof(uint32_t);
        mm_ids_helper_route8<<<n_experts, 256, nbytes_shared, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, nchannels_y, si1, sis1, write_inverse);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

// One CTA builds a stable tile list entirely on the caller's stream.
static __global__ void mmq_make_compact_tiles(const int32_t * bounds, int2 * tiles,
                                             const int n_experts, const int tile_width) {
    __shared__ int prefix[256];
    const int e = threadIdx.x;
    const int count = e < n_experts ? (bounds[e + 1] - bounds[e] + tile_width - 1) / tile_width : 0;
    prefix[e] = count;
    __syncthreads();
    for (int offset = 1; offset < 256; offset *= 2) {
        const int add = e >= offset ? prefix[e - offset] : 0;
        __syncthreads(); // Complete all reads before overwriting this stage.
        prefix[e] += add;
        __syncthreads();
    }
    if (e == 0) {
        tiles[0] = make_int2(prefix[255], 0);
    }
    const int first = prefix[e] - count;
    for (int j = 0; j < count; ++j) {
        tiles[1 + first + j] = make_int2(e, j);
    }
}

void ggml_cuda_mmq_make_compact_tiles(const int32_t * expert_bounds, int2 * tiles,
                                     int n_experts, int tile_width, cudaStream_t stream) {
    GGML_ASSERT(n_experts > 0 && n_experts <= 256 && tile_width > 0);
    mmq_make_compact_tiles<<<1, 256, 0, stream>>>(expert_bounds, tiles, n_experts, tile_width);
}

// GGML_CUDA_MMQ_TAIL_SM52=1 (P3): three tile maps for the J=64 compact path. Full 64-wide tiles
// (plus a remainder tile when the remainder exceeds 32 columns) go to t64; a remainder in (16,32]
// becomes one 32-wide tile in t32 and a remainder in (0,16] one 16-wide tile in t16. Entries are
// (expert, local tile index) in that map's J units, so jt*J inside mul_mat_q stays the column offset.
static __global__ void mmq_make_tail_tiles(const int32_t * bounds, int2 * t64, int2 * t32, int2 * t16, const int n_experts) {
    __shared__ int p64[256];
    __shared__ int p32[256];
    __shared__ int p16[256];
    const int e = threadIdx.x;
    int c64 = 0, c32 = 0, c16 = 0, full = 0;
    if (e < n_experts) {
        const int n = bounds[e + 1] - bounds[e];
        full = n / 64;
        const int r = n - full*64;
        c64 = full + (r > 32 ? 1 : 0);
        c32 = (r > 16 && r <= 32) ? 1 : 0;
        c16 = (r >  0 && r <= 16) ? 1 : 0;
    }
    p64[e] = c64; p32[e] = c32; p16[e] = c16;
    __syncthreads();
    for (int offset = 1; offset < 256; offset *= 2) {
        const int a64 = e >= offset ? p64[e - offset] : 0;
        const int a32 = e >= offset ? p32[e - offset] : 0;
        const int a16 = e >= offset ? p16[e - offset] : 0;
        __syncthreads();
        p64[e] += a64; p32[e] += a32; p16[e] += a16;
        __syncthreads();
    }
    if (e == 0) {
        t64[0] = make_int2(p64[255], 0);
        t32[0] = make_int2(p32[255], 0);
        t16[0] = make_int2(p16[255], 0);
    }
    const int f64 = p64[e] - c64;
    for (int j = 0; j < c64; ++j) {
        t64[1 + f64 + j] = make_int2(e, j);
    }
    if (c32) {
        t32[p32[e]] = make_int2(e, full*2);
    }
    if (c16) {
        t16[p16[e]] = make_int2(e, full*4);
    }
}

void ggml_cuda_mmq_make_tail_tiles(const int32_t * expert_bounds, int2 * t64, int2 * t32, int2 * t16,
                                   int n_experts, cudaStream_t stream) {
    GGML_ASSERT(n_experts > 0 && n_experts <= 256);
    mmq_make_tail_tiles<<<1, 256, 0, stream>>>(expert_bounds, t64, t32, t16, n_experts);
}
