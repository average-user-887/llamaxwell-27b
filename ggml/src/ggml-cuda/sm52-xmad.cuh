#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Two PRMT instructions sign-extend four signed bytes into two packed s16x2.
// Use PTX directly: CUDA __byte_perm does not expose sign-replicate selectors.
struct ggml_cuda_sm52_i16x4 { uint32_t lo, hi; };
static __device__ __forceinline__ ggml_cuda_sm52_i16x4 ggml_cuda_sm52_expand(uint32_t v) {
    ggml_cuda_sm52_i16x4 r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 520 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(r.lo) : "r"(v));
    asm("prmt.b32 %0, %1, 0, 0xb3a2;" : "=r"(r.hi) : "r"(v));
#else
    r.lo = uint32_t(uint16_t(int16_t(int8_t(v)))) |
           (uint32_t(uint16_t(int16_t(int8_t(v >> 8)))) << 16);
    r.hi = uint32_t(uint16_t(int16_t(int8_t(v >> 16)))) |
           (uint32_t(uint16_t(int16_t(int8_t(v >> 24)))) << 16);
#endif
    return r;
}

static __device__ __forceinline__ int ggml_cuda_sm52_mad2(uint32_t a, uint32_t b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 520 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // ptxas must be inspected: one XMAD per mad.wide.s16 is a target, not a guarantee.
    asm("{ .reg .b16 a0, a1, b0, b1;\n"
        "mov.b32 {a0, a1}, %1;\n"
        "mov.b32 {b0, b1}, %2;\n"
        "mad.wide.s16 %0, a0, b0, %0;\n"
        "mad.wide.s16 %0, a1, b1, %0;\n}"
        : "+r"(c) : "r"(a), "r"(b));
#else
    c += int16_t(a) * int16_t(b);
    c += int16_t(a >> 16) * int16_t(b >> 16);
#endif
    return c;
}

static __device__ __forceinline__ int ggml_cuda_sm52_dot4(
        ggml_cuda_sm52_i16x4 a, ggml_cuda_sm52_i16x4 b, int c) {
    return ggml_cuda_sm52_mad2(a.hi, b.hi, ggml_cuda_sm52_mad2(a.lo, b.lo, c));
}
