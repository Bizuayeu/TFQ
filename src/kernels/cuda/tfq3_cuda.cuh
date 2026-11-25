#pragma once
// ============================================================================
// TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128) & BF16 CUDA Kernels
//
// Reads the exact same binary `pw_v3_tables.bin` ("PWV3") and `.tffpw` ("TFPN")
// shards as Apple Silicon Metal — zero re-conversion between Mac and NVIDIA GPU.
//
// Supported architectures: sm_80+ (Ampere, Ada, Hopper, Blackwell / GB10).
// ============================================================================
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

namespace tfq3 {

struct alignas(16) PwV3Tables {
    uint32_t magic;
    uint32_t version;
    float a3, b3, a4, b4, a5, b5;
    uint32_t pad0[8];
    uint32_t lut3[256];
    uint32_t lut4[1024];
    uint32_t lut5[1024];
    uint2 c64_3[64]; // each uint2 = { half2(sx, sy), half2(tx, ty) }
    uint2 c64_4[64];
    uint2 c64_5[64];
};
static_assert(sizeof(PwV3Tables) == 10816, "PwV3Tables layout must match PWV3 v3");

struct DenseDims {
    uint32_t rows;
    uint32_t n;
    uint32_t k;
    uint32_t ldx;
    uint32_t ldo;
    uint32_t sr_split;
    uint32_t ksplit;
    uint32_t pad;
};

__device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
    return *reinterpret_cast<const __half2*>(&u);
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_u16(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ int2 pw_frag_home(uint32_t l) {
    return make_int2(static_cast<int>((l & 8u) + ((l & 1u) << 2)),
                     static_cast<int>(((l & 16u) >> 2) | ((l >> 1) & 3u)));
}

__device__ __forceinline__ float pw_sgn(uint32_t seed, uint32_t i) {
    uint32_t h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ uint32_t pw_sr0(uint32_t r0, uint32_t sr_split) {
    return (sr_split != 0u && r0 >= sr_split) ? (r0 - sr_split) : r0;
}

__device__ __forceinline__ void pw_fwht128_simd4(float4& v, uint32_t lane) {
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1) {
        float ox = __shfl_xor_sync(0xFFFFFFFFu, v.x, d);
        float oy = __shfl_xor_sync(0xFFFFFFFFu, v.y, d);
        float oz = __shfl_xor_sync(0xFFFFFFFFu, v.z, d);
        float ow = __shfl_xor_sync(0xFFFFFFFFu, v.w, d);
        bool lo = ((lane & static_cast<uint32_t>(d)) == 0u);
        v.x = lo ? (v.x + ox) : (ox - v.x);
        v.y = lo ? (v.y + oy) : (oy - v.y);
        v.z = lo ? (v.z + oz) : (oz - v.z);
        v.w = lo ? (v.w + ow) : (ow - v.w);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    constexpr float inv = 0.08838834764831845f;
    v = make_float4((a0 + a1) * inv, (b0 + b1) * inv, (a0 - a1) * inv, (b0 - b1) * inv);
}

// ----------------------------------------------------------------------------
// 1. Input & Output FWHT-128 Kernels
// ----------------------------------------------------------------------------
__global__ void tfq3_fwht_x_kernel(
    const uint16_t* __restrict__ X,
    __half* __restrict__ X_rot,
    DenseDims d
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t c0 = blockIdx.x * 128u;
    if (row >= d.rows || c0 >= d.k) return;
    const uint16_t* src = X + static_cast<uint64_t>(row) * d.ldx + c0;
    float4 v;
    v.x = bf16_to_f32(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
    v.y = bf16_to_f32(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
    v.z = bf16_to_f32(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
    v.w = bf16_to_f32(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
    pw_fwht128_simd4(v, lane);
    __half* dst = X_rot + static_cast<uint64_t>(row) * d.k + c0;
    dst[lane +  0u] = __float2half(v.x);
    dst[lane + 32u] = __float2half(v.y);
    dst[lane + 64u] = __float2half(v.z);
    dst[lane + 96u] = __float2half(v.w);
}

__global__ void tfq3_fwht_y_kernel(
    const float* __restrict__ YR,
    uint16_t* __restrict__ OUT,
    DenseDims d
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t r0 = blockIdx.x * 128u;
    if (row >= d.rows || r0 >= d.n) return;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const uint32_t ks = d.ksplit > 1u ? d.ksplit : 1u;
    for (uint32_t kp = 0u; kp < ks; ++kp) {
        const float* src = YR + (static_cast<uint64_t>(kp) * d.rows + row) * d.n + r0;
        v.x += src[lane +  0u];
        v.y += src[lane + 32u];
        v.z += src[lane + 64u];
        v.w += src[lane + 96u];
    }
    pw_fwht128_simd4(v, lane);
    const uint32_t sr0 = pw_sr0(r0, d.sr_split);
    uint16_t* dst = OUT + static_cast<uint64_t>(row) * d.ldo + r0;
    dst[lane +  0u] = f32_to_bf16_u16(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
    dst[lane + 32u] = f32_to_bf16_u16(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
    dst[lane + 64u] = f32_to_bf16_u16(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
    dst[lane + 96u] = f32_to_bf16_u16(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
}

// ----------------------------------------------------------------------------
// 2. 2D-AffineCoset/half4 Decoders (3b / 4b / 5b) -> four __half2 (q0a,q0b,q1a,q1b)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void pw_v3_q8_4b(
    uint32_t& st, uint32_t wc, const PwV3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t s0 = st;
    uint32_t h0 = s0 * 0x9E3779B1u;
    uint32_t h1 = ((s0 <<  8) | (wc >> 24)) * 0x9E3779B1u;
    uint32_t h2 = ((s0 << 16) | (wc >> 16)) * 0x9E3779B1u;
    uint32_t h3 = ((s0 << 24) | (wc >>  8)) * 0x9E3779B1u;
    st = wc;
    uint2 c0 = __ldg(&T->c64_4[(h0 >> 16) & 63u]);
    uint2 c2 = __ldg(&T->c64_4[(h2 >> 16) & 63u]);
    __half2 c0s = u32_to_h2(c0.x), c0t = u32_to_h2(c0.y);
    __half2 c2s = u32_to_h2(c2.x), c2t = u32_to_h2(c2.y);
    q0a = __hfma2(u32_to_h2(__ldg(&T->lut4[(((wc >> 24)        << 2) ^ (h0 >> 22)) & 1023u])), c0s, c0t);
    q0b = __hfma2(u32_to_h2(__ldg(&T->lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u])), c0s, c0t);
    q1a = __hfma2(u32_to_h2(__ldg(&T->lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u])), c2s, c2t);
    q1b = __hfma2(u32_to_h2(__ldg(&T->lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u])), c2s, c2t);
}

__device__ __forceinline__ void pw_v3_q8_3b(
    uint32_t& st, uint32_t m24, const PwV3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t s0 = st;
    uint32_t h0 = s0 * 0x9E3779B1u;
    uint32_t h1 = ((s0 <<  6) | (m24 >> 18)) * 0x9E3779B1u;
    uint32_t h2 = ((s0 << 12) | (m24 >> 12)) * 0x9E3779B1u;
    uint32_t h3 = ((s0 << 18) | (m24 >>  6)) * 0x9E3779B1u;
    st = (s0 << 24) | m24;
    uint2 c0 = __ldg(&T->c64_3[(h0 >> 16) & 63u]);
    uint2 c2 = __ldg(&T->c64_3[(h2 >> 16) & 63u]);
    __half2 c0s = u32_to_h2(c0.x), c0t = u32_to_h2(c0.y);
    __half2 c2s = u32_to_h2(c2.x), c2t = u32_to_h2(c2.y);
    q0a = __hfma2(u32_to_h2(__ldg(&T->lut3[(((m24 >> 16) & 0xFCu) ^ (h0 >> 24)) & 255u])), c0s, c0t);
    q0b = __hfma2(u32_to_h2(__ldg(&T->lut3[(((m24 >> 10) & 0xFCu) ^ (h1 >> 24)) & 255u])), c0s, c0t);
    q1a = __hfma2(u32_to_h2(__ldg(&T->lut3[(((m24 >>  4) & 0xFCu) ^ (h2 >> 24)) & 255u])), c2s, c2t);
    q1b = __hfma2(u32_to_h2(__ldg(&T->lut3[(((m24 <<  2) & 0xFCu) ^ (h3 >> 24)) & 255u])), c2s, c2t);
}

__device__ __forceinline__ void pw_v3_q8_5b(
    uint32_t& st, uint32_t hi8, uint32_t lo32, const PwV3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t s0 = st;
    uint32_t b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023u;
    uint32_t b1 = (lo32 >> 20) & 1023u;
    uint32_t b2 = (lo32 >> 10) & 1023u;
    uint32_t b3 = lo32 & 1023u;
    uint32_t h0 = s0 * 0x9E3779B1u;
    uint32_t h1 = ((s0 << 10) | b0) * 0x9E3779B1u;
    uint32_t h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) * 0x9E3779B1u;
    uint32_t h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) * 0x9E3779B1u;
    st = lo32;
    uint2 c0 = __ldg(&T->c64_5[(h0 >> 16) & 63u]);
    uint2 c2 = __ldg(&T->c64_5[(h2 >> 16) & 63u]);
    __half2 c0s = u32_to_h2(c0.x), c0t = u32_to_h2(c0.y);
    __half2 c2s = u32_to_h2(c2.x), c2t = u32_to_h2(c2.y);
    q0a = __hfma2(u32_to_h2(__ldg(&T->lut5[(b0 ^ (h0 >> 22)) & 1023u])), c0s, c0t);
    q0b = __hfma2(u32_to_h2(__ldg(&T->lut5[(b1 ^ (h1 >> 22)) & 1023u])), c0s, c0t);
    q1a = __hfma2(u32_to_h2(__ldg(&T->lut5[(b2 ^ (h2 >> 22)) & 1023u])), c2s, c2t);
    q1b = __hfma2(u32_to_h2(__ldg(&T->lut5[(b3 ^ (h3 >> 22)) & 1023u])), c2s, c2t);
}

template <uint32_t BITS>
__device__ __forceinline__ uint32_t pw_init_st(const uint32_t* __restrict__ w_base, uint32_t bc0, uint32_t lane) {
    if (bc0 == 0u) return 0x13579BDFu ^ (lane * 0x9E3779B1u);
    if constexpr (BITS == 4u) {
        const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
        return __ldg(&w4[(static_cast<uint64_t>(bc0) * 2ull - 1ull) * 32ull + lane]).w;
    } else {
        return __ldg(&w_base[(static_cast<uint64_t>(bc0) * (BITS * 2u) - 1ull) * 32ull + lane]);
    }
}

__device__ __forceinline__ float h2_pair_sum(__half2 a, __half2 b) {
    float2 fa = __half22float2(a), fb = __half22float2(b);
    return (fa.x + fa.y) + (fb.x + fb.y);
}

// ----------------------------------------------------------------------------
// 3. Decode Kernels: Fused Inv-FWHT & 2D Split-K=8 (B=1..4, 3b/4b/5b)
// ----------------------------------------------------------------------------
template <uint32_t BITS, uint32_t BS>
__device__ __forceinline__ void pw_dense_mv_range(
    const uint32_t* __restrict__ w_base,
    const __half* __restrict__ sc_base,
    uint32_t bc0, uint32_t bc1,
    const PwV3Tables* __restrict__ T,
    const __half* const x_ptrs[BS],
    uint32_t lane, int2 home,
    float acc0[BS], float acc1[BS]
) {
    uint32_t st = pw_init_st<BITS>(w_base, bc0, lane);
    #pragma unroll
    for (uint32_t m = 0; m < BS; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    auto dot_step = [&](__half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b, uint32_t cb,
                        __half2 ba0a[BS], __half2 ba0b[BS], __half2 ba1a[BS], __half2 ba1b[BS]) {
        #pragma unroll
        for (uint32_t m = 0; m < BS; ++m) {
            const uint2 xv = *reinterpret_cast<const uint2*>(x_ptrs[m] + cb);
            __half2 xa = u32_to_h2(xv.x), xb = u32_to_h2(xv.y);
            ba0a[m] = __hfma2(q0a, xa, ba0a[m]); ba0b[m] = __hfma2(q0b, xb, ba0b[m]);
            ba1a[m] = __hfma2(q1a, xa, ba1a[m]); ba1b[m] = __hfma2(q1b, xb, ba1b[m]);
        }
    };

    const __half2 hz = __float2half2_rn(0.0f);
    for (uint32_t bc = bc0; bc < bc1; ++bc) {
        __half2 ba0a[BS], ba0b[BS], ba1a[BS], ba1b[BS];
        #pragma unroll
        for (uint32_t m = 0; m < BS; ++m) { ba0a[m] = hz; ba0b[m] = hz; ba1a[m] = hz; ba1b[m] = hz; }
        uint32_t cb = (bc << 7) + static_cast<uint32_t>(home.x);
        __half2 q0a, q0b, q1a, q1b;

        if constexpr (BITS == 4u) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
            uint4 v = __ldg(&w4[(bc * 2u + 0u) * 32u + lane]);
            pw_v3_q8_4b(st, v.x, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +   0u, ba0a, ba0b, ba1a, ba1b);
            pw_v3_q8_4b(st, v.y, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +  16u, ba0a, ba0b, ba1a, ba1b);
            pw_v3_q8_4b(st, v.z, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +  32u, ba0a, ba0b, ba1a, ba1b);
            pw_v3_q8_4b(st, v.w, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +  48u, ba0a, ba0b, ba1a, ba1b);
            v = __ldg(&w4[(bc * 2u + 1u) * 32u + lane]);
            pw_v3_q8_4b(st, v.x, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +  64u, ba0a, ba0b, ba1a, ba1b);
            pw_v3_q8_4b(st, v.y, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +  80u, ba0a, ba0b, ba1a, ba1b);
            pw_v3_q8_4b(st, v.z, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb +  96u, ba0a, ba0b, ba1a, ba1b);
            pw_v3_q8_4b(st, v.w, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cb + 112u, ba0a, ba0b, ba1a, ba1b);
        } else if constexpr (BITS == 3u) {
            #pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 6u + h * 3u) * 32u + lane;
                uint32_t a0 = __ldg(&blk[0]), a1 = __ldg(&blk[32]), a2 = __ldg(&blk[64]);
                uint32_t cbh = cb + (h << 6);
                pw_v3_q8_3b(st, a0 >> 8,                               T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh +  0u, ba0a, ba0b, ba1a, ba1b);
                pw_v3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh + 16u, ba0a, ba0b, ba1a, ba1b);
                pw_v3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh + 32u, ba0a, ba0b, ba1a, ba1b);
                pw_v3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh + 48u, ba0a, ba0b, ba1a, ba1b);
            }
        } else {
            #pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 10u + h * 5u) * 32u + lane;
                uint32_t w0 = __ldg(&blk[0]), w1 = __ldg(&blk[32]), w2 = __ldg(&blk[64]), w3 = __ldg(&blk[96]), w4 = __ldg(&blk[128]);
                uint32_t cbh = cb + (h << 6);
                pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh +  0u, ba0a, ba0b, ba1a, ba1b);
                pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh + 16u, ba0a, ba0b, ba1a, ba1b);
                pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh + 32u, ba0a, ba0b, ba1a, ba1b);
                pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0a, q0b, q1a, q1b); dot_step(q0a, q0b, q1a, q1b, cbh + 48u, ba0a, ba0b, ba1a, ba1b);
            }
        }
        float sc = __half2float(__ldg(&sc_base[bc]));
        #pragma unroll
        for (uint32_t m = 0; m < BS; ++m) {
            acc0[m] = fmaf(h2_pair_sum(ba0a[m], ba0b[m]), sc, acc0[m]);
            acc1[m] = fmaf(h2_pair_sum(ba1a[m], ba1b[m]), sc, acc1[m]);
        }
    }
    #pragma unroll
    for (uint32_t m = 0; m < BS; ++m) {
        acc0[m] += __shfl_xor_sync(0xFFFFFFFFu, acc0[m], 1);
        acc0[m] += __shfl_xor_sync(0xFFFFFFFFu, acc0[m], 8);
        acc1[m] += __shfl_xor_sync(0xFFFFFFFFu, acc1[m], 1);
        acc1[m] += __shfl_xor_sync(0xFFFFFFFFu, acc1[m], 8);
    }
}

template <uint32_t BITS, uint32_t BS>
__global__ void tfq3_dec_fused_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    uint16_t* __restrict__ OUT,
    DenseDims d
) {
    const uint32_t rb = blockIdx.x, r0 = rb * 128u;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n_bc = d.k >> 7;
    constexpr uint32_t wpb = BITS * 2u;
    const uint32_t strip = rb * 8u + sg;
    const uint32_t* w_base = W + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sc_base = scales + static_cast<uint64_t>(rb) * n_bc;
    const int2 home = pw_frag_home(lane);
    const uint32_t sr0 = pw_sr0(r0, d.sr_split);
    __shared__ float smem[BS * 128];

    for (uint32_t m0 = 0u; m0 < d.rows; m0 += BS) {
        uint32_t nc = (d.rows - m0) < BS ? (d.rows - m0) : BS;
        const __half* x_ptrs[BS];
        #pragma unroll
        for (uint32_t m = 0; m < BS; ++m) {
            uint32_t mr = m0 + (m < nc ? m : 0u);
            x_ptrs[m] = X_rot + static_cast<uint64_t>(mr) * d.k;
        }
        float acc0[BS], acc1[BS];
        pw_dense_mv_range<BITS, BS>(w_base, sc_base, 0u, n_bc, T, x_ptrs, lane, home, acc0, acc1);
        if ((lane & 9u) == 0u) {
            uint32_t local_r = sg * 16u + static_cast<uint32_t>(home.y);
            #pragma unroll
            for (uint32_t m = 0; m < BS; ++m) {
                if (m < nc) {
                    smem[m * 128u + local_r] = acc0[m];
                    smem[m * 128u + local_r + 8u] = acc1[m];
                }
            }
        }
        __syncthreads();
        if (sg < nc) {
            const float* ys = smem + sg * 128u;
            float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
            pw_fwht128_simd4(v, lane);
            uint16_t* dst = OUT + static_cast<uint64_t>(m0 + sg) * d.ldo + r0;
            dst[lane +  0u] = f32_to_bf16_u16(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
            dst[lane + 32u] = f32_to_bf16_u16(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
            dst[lane + 64u] = f32_to_bf16_u16(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
            dst[lane + 96u] = f32_to_bf16_u16(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
        }
        if (m0 + BS < d.rows) __syncthreads();
    }
}

template <uint32_t BITS, uint32_t BS>
__global__ void tfq3_dec_2dsk8_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    float* __restrict__ YR,
    DenseDims d
) {
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t strip = blockIdx.x * 2u + sg;
    const uint32_t rb = strip >> 3;
    const uint32_t kp = blockIdx.y;
    const uint32_t n_bc = d.k >> 7;
    const uint32_t chunk = n_bc >> 3;
    const uint32_t bc0 = kp * chunk, bc1 = bc0 + chunk;
    constexpr uint32_t wpb = BITS * 2u;
    const uint32_t* w_base = W + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sc_base = scales + static_cast<uint64_t>(rb) * n_bc;
    const int2 home = pw_frag_home(lane);
    const uint32_t r_base = strip * 16u + static_cast<uint32_t>(home.y);

    for (uint32_t m0 = 0u; m0 < d.rows; m0 += BS) {
        uint32_t nc = (d.rows - m0) < BS ? (d.rows - m0) : BS;
        const __half* x_ptrs[BS];
        #pragma unroll
        for (uint32_t m = 0; m < BS; ++m) {
            uint32_t mr = m0 + (m < nc ? m : 0u);
            x_ptrs[m] = X_rot + static_cast<uint64_t>(mr) * d.k;
        }
        float acc0[BS], acc1[BS];
        pw_dense_mv_range<BITS, BS>(w_base, sc_base, bc0, bc1, T, x_ptrs, lane, home, acc0, acc1);
        if ((lane & 9u) == 0u) {
            #pragma unroll
            for (uint32_t m = 0; m < BS; ++m) {
                if (m < nc) {
                    float* dst = YR + (static_cast<uint64_t>(kp) * d.rows + m0 + m) * d.n;
                    dst[r_base] = acc0[m];
                    dst[r_base + 8u] = acc1[m];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 4. Prefill GEMM Kernel (Tensor Core WMMA 16x16x16, 3b/4b/5b)
// ----------------------------------------------------------------------------
template <uint32_t BITS>
__global__ void tfq3_dense_prefill_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    uint16_t* __restrict__ OUT,
    DenseDims d
) {
    using namespace nvcuda::wmma;
    const uint32_t rb = blockIdx.x, r0 = rb * 128u;
    const uint32_t m0 = blockIdx.y * 32u;
    if (m0 >= d.rows) return;
    const uint32_t nb = (d.rows - m0) < 32u ? (d.rows - m0) : 32u;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t n_bc = d.k >> 7;
    constexpr uint32_t wpb = BITS * 2u;
    const uint32_t strip0 = rb * 8u + sg * 2u;
    const uint32_t* w_s0 = W + static_cast<uint64_t>(strip0 + 0u) * n_bc * wpb * 32ull;
    const uint32_t* w_s1 = W + static_cast<uint64_t>(strip0 + 1u) * n_bc * wpb * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(rb) * n_bc;
    uint32_t st0 = 0x13579BDFu ^ (lane * 0x9E3779B1u);
    uint32_t st1 = st0;
    const int2 home = pw_frag_home(lane);

    // 16 KiB shared float output buffer (reused as 8 KiB half tg_a during K-loop) + 4 KiB per-warp decoded weight tiles.
    __shared__ float tg_f[32 * 128];
    __half* tg_a = reinterpret_cast<__half*>(tg_f);
    __shared__ __half tg_b[4][512]; // 2 x [16, 16] row-major weight tiles per warp

    fragment<accumulator, 16, 16, 16, float> c0_lo, c0_hi, c1_lo, c1_hi;
    fill_fragment(c0_lo, 0.0f); fill_fragment(c0_hi, 0.0f);
    fill_fragment(c1_lo, 0.0f); fill_fragment(c1_hi, 0.0f);

    const uint32_t row_l = tid >> 2, col_l = (tid & 3u) * 32u;

    auto wmma_step32 = [&](int kb, __half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b,
                           __half2 r0a, __half2 r0b, __half2 r1a, __half2 r1b,
                           fragment<accumulator, 16, 16, 16, float>& bc0_lo,
                           fragment<accumulator, 16, 16, 16, float>& bc0_hi,
                           fragment<accumulator, 16, 16, 16, float>& bc1_lo,
                           fragment<accumulator, 16, 16, 16, float>& bc1_hi) {
        __half* b0_dst = tg_b[sg];
        __half* b1_dst = tg_b[sg] + 256;
        *reinterpret_cast<__half2*>(b0_dst + home.y * 16 + home.x + 0)       = q0a;
        *reinterpret_cast<__half2*>(b0_dst + home.y * 16 + home.x + 2)       = q0b;
        *reinterpret_cast<__half2*>(b0_dst + (home.y + 8) * 16 + home.x + 0) = q1a;
        *reinterpret_cast<__half2*>(b0_dst + (home.y + 8) * 16 + home.x + 2) = q1b;
        *reinterpret_cast<__half2*>(b1_dst + home.y * 16 + home.x + 0)       = r0a;
        *reinterpret_cast<__half2*>(b1_dst + home.y * 16 + home.x + 2)       = r0b;
        *reinterpret_cast<__half2*>(b1_dst + (home.y + 8) * 16 + home.x + 0) = r1a;
        *reinterpret_cast<__half2*>(b1_dst + (home.y + 8) * 16 + home.x + 2) = r1b;
        __syncwarp();
        fragment<matrix_a, 16, 16, 16, __half, row_major> fa0;
        fragment<matrix_b, 16, 16, 16, __half, col_major> fb0, fb1;
        load_matrix_sync(fb0, b0_dst, 16);
        load_matrix_sync(fb1, b1_dst, 16);
        load_matrix_sync(fa0, tg_a + kb * 16, 128);
        mma_sync(bc0_lo, fa0, fb0, bc0_lo);
        mma_sync(bc0_hi, fa0, fb1, bc0_hi);
        if (nb > 16u) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> fa1;
            load_matrix_sync(fa1, tg_a + 16 * 128 + kb * 16, 128);
            mma_sync(bc1_lo, fa1, fb0, bc1_lo);
            mma_sync(bc1_hi, fa1, fb1, bc1_hi);
        }
    };

    for (uint32_t bc = 0u; bc < n_bc; ++bc) {
        if (row_l < nb) {
            const uint4* src4 = reinterpret_cast<const uint4*>(X_rot + static_cast<uint64_t>(m0 + row_l) * d.k + (bc << 7) + col_l);
            uint4* dst4 = reinterpret_cast<uint4*>(tg_a + row_l * 128u + col_l);
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) dst4[j] = src4[j];
        } else {
            uint4* dst4 = reinterpret_cast<uint4*>(tg_a + row_l * 128u + col_l);
            uint4 z = make_uint4(0u, 0u, 0u, 0u);
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) dst4[j] = z;
        }
        __syncthreads();

        fragment<accumulator, 16, 16, 16, float> bc0_lo, bc0_hi, bc1_lo, bc1_hi;
        fill_fragment(bc0_lo, 0.0f); fill_fragment(bc0_hi, 0.0f);
        fill_fragment(bc1_lo, 0.0f); fill_fragment(bc1_hi, 0.0f);
        __half2 q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b;

        if constexpr (BITS == 4u) {
            const uint4* p0 = reinterpret_cast<const uint4*>(w_s0 + static_cast<uint64_t>(bc) * 8u * 32u);
            const uint4* p1 = reinterpret_cast<const uint4*>(w_s1 + static_cast<uint64_t>(bc) * 8u * 32u);
            for (uint32_t h = 0u; h < 2u; ++h) {
                uint4 v0 = __ldg(&p0[h * 32u + lane]), v1 = __ldg(&p1[h * 32u + lane]);
                int kb = static_cast<int>(h * 4u);
                pw_v3_q8_4b(st0, v0.x, T, q0a, q0b, q1a, q1b); pw_v3_q8_4b(st1, v1.x, T, r0a, r0b, r1a, r1b); wmma_step32(kb + 0, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_4b(st0, v0.y, T, q0a, q0b, q1a, q1b); pw_v3_q8_4b(st1, v1.y, T, r0a, r0b, r1a, r1b); wmma_step32(kb + 1, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_4b(st0, v0.z, T, q0a, q0b, q1a, q1b); pw_v3_q8_4b(st1, v1.z, T, r0a, r0b, r1a, r1b); wmma_step32(kb + 2, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_4b(st0, v0.w, T, q0a, q0b, q1a, q1b); pw_v3_q8_4b(st1, v1.w, T, r0a, r0b, r1a, r1b); wmma_step32(kb + 3, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
            }
        } else if constexpr (BITS == 3u) {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* p0 = w_s0 + (static_cast<uint64_t>(bc) * 6u + h * 3u) * 32u + lane;
                const uint32_t* p1 = w_s1 + (static_cast<uint64_t>(bc) * 6u + h * 3u) * 32u + lane;
                uint32_t a0 = __ldg(&p0[0]), a1 = __ldg(&p0[32]), a2 = __ldg(&p0[64]);
                uint32_t b0 = __ldg(&p1[0]), b1 = __ldg(&p1[32]), b2 = __ldg(&p1[64]);
                int kb = static_cast<int>(h * 4u);
                pw_v3_q8_3b(st0, a0 >> 8,                               T, q0a, q0b, q1a, q1b); pw_v3_q8_3b(st1, b0 >> 8,                               T, r0a, r0b, r1a, r1b); wmma_step32(kb + 0, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_3b(st0, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); pw_v3_q8_3b(st1, ((b0 << 16) | (b1 >> 16)) & 0xFFFFFFu, T, r0a, r0b, r1a, r1b); wmma_step32(kb + 1, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_3b(st0, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); pw_v3_q8_3b(st1, ((b1 <<  8) | (b2 >> 24)) & 0xFFFFFFu, T, r0a, r0b, r1a, r1b); wmma_step32(kb + 2, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_3b(st0, a2 & 0xFFFFFFu,                        T, q0a, q0b, q1a, q1b); pw_v3_q8_3b(st1, b2 & 0xFFFFFFu,                        T, r0a, r0b, r1a, r1b); wmma_step32(kb + 3, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
            }
        } else {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* p0 = w_s0 + (static_cast<uint64_t>(bc) * 10u + h * 5u) * 32u + lane;
                const uint32_t* p1 = w_s1 + (static_cast<uint64_t>(bc) * 10u + h * 5u) * 32u + lane;
                uint32_t a0 = __ldg(&p0[0]), a1 = __ldg(&p0[32]), a2 = __ldg(&p0[64]), a3 = __ldg(&p0[96]), a4 = __ldg(&p0[128]);
                uint32_t b0 = __ldg(&p1[0]), b1 = __ldg(&p1[32]), b2 = __ldg(&p1[64]), b3 = __ldg(&p1[96]), b4 = __ldg(&p1[128]);
                int kb = static_cast<int>(h * 4u);
                pw_v3_q8_5b(st0, a0 >> 24,          (a0 <<  8) | (a1 >> 24), T, q0a, q0b, q1a, q1b); pw_v3_q8_5b(st1, b0 >> 24,          (b0 <<  8) | (b1 >> 24), T, r0a, r0b, r1a, r1b); wmma_step32(kb + 0, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_5b(st0, (a1 >> 16) & 255u, (a1 << 16) | (a2 >> 16), T, q0a, q0b, q1a, q1b); pw_v3_q8_5b(st1, (b1 >> 16) & 255u, (b1 << 16) | (b2 >> 16), T, r0a, r0b, r1a, r1b); wmma_step32(kb + 1, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_5b(st0, (a2 >>  8) & 255u, (a2 << 24) | (a3 >>  8), T, q0a, q0b, q1a, q1b); pw_v3_q8_5b(st1, (b2 >>  8) & 255u, (b2 << 24) | (b3 >>  8), T, r0a, r0b, r1a, r1b); wmma_step32(kb + 2, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
                pw_v3_q8_5b(st0, a3 & 255u,          a4,                     T, q0a, q0b, q1a, q1b); pw_v3_q8_5b(st1, b3 & 255u,          b4,                     T, r0a, r0b, r1a, r1b); wmma_step32(kb + 3, q0a, q0b, q1a, q1b, r0a, r0b, r1a, r1b, bc0_lo, bc0_hi, bc1_lo, bc1_hi);
            }
        }

        float fsv = __half2float(__ldg(&sc0[bc]));
        #pragma unroll
        for (int e = 0; e < c0_lo.num_elements; ++e) {
            c0_lo.x[e] = fmaf(bc0_lo.x[e], fsv, c0_lo.x[e]);
            c0_hi.x[e] = fmaf(bc0_hi.x[e], fsv, c0_hi.x[e]);
            if (nb > 16u) {
                c1_lo.x[e] = fmaf(bc1_lo.x[e], fsv, c1_lo.x[e]);
                c1_hi.x[e] = fmaf(bc1_hi.x[e], fsv, c1_hi.x[e]);
            }
        }
        __syncthreads();
    }

    store_matrix_sync(tg_f + sg * 32u, c0_lo, 128, mem_row_major);
    store_matrix_sync(tg_f + sg * 32u + 16u, c0_hi, 128, mem_row_major);
    if (nb > 16u) {
        store_matrix_sync(tg_f + 16u * 128u + sg * 32u, c1_lo, 128, mem_row_major);
        store_matrix_sync(tg_f + 16u * 128u + sg * 32u + 16u, c1_hi, 128, mem_row_major);
    }
    __syncthreads();

    const uint32_t sr0 = pw_sr0(r0, d.sr_split);
    for (uint32_t m = sg; m < nb; m += 4u) {
        const float* ys = tg_f + m * 128u;
        float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        uint16_t* dst = OUT + static_cast<uint64_t>(m0 + m) * d.ldo + r0;
        dst[lane +  0u] = f32_to_bf16_u16(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
        dst[lane + 32u] = f32_to_bf16_u16(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
        dst[lane + 64u] = f32_to_bf16_u16(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
        dst[lane + 96u] = f32_to_bf16_u16(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
    }
}

// ----------------------------------------------------------------------------
// 5. BF16 Residual Linear & Embedding Kernels
// ----------------------------------------------------------------------------
__global__ void tfq3_bf16_mv1_kernel(
    const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ X,
    uint16_t* __restrict__ OUT,
    DenseDims d
) {
    const uint32_t r = blockIdx.x, lane = threadIdx.x;
    if (r >= d.n) return;
    const uint2* wr = reinterpret_cast<const uint2*>(W + static_cast<uint64_t>(r) * d.k);
    const uint2* xv = reinterpret_cast<const uint2*>(X);
    const uint32_t n4 = d.k >> 2;
    float acc = 0.0f;
    for (uint32_t c = lane; c < n4; c += 32u) {
        uint2 w4 = __ldg(&wr[c]), x4 = __ldg(&xv[c]);
        acc += bf16_to_f32(w4.x & 0xFFFFu) * bf16_to_f32(x4.x & 0xFFFFu);
        acc += bf16_to_f32(w4.x >> 16)     * bf16_to_f32(x4.x >> 16);
        acc += bf16_to_f32(w4.y & 0xFFFFu) * bf16_to_f32(x4.y & 0xFFFFu);
        acc += bf16_to_f32(w4.y >> 16)     * bf16_to_f32(x4.y >> 16);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0u) OUT[r] = f32_to_bf16_u16(acc);
}

__global__ void tfq3_bf16_mm_kernel(
    const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ X,
    uint16_t* __restrict__ OUT,
    DenseDims d
) {
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * 4u + sg;
    const uint32_t m0 = blockIdx.y * 4u;
    if (r >= d.n || m0 >= d.rows) return;
    const uint32_t nc = (d.rows - m0) < 4u ? (d.rows - m0) : 4u;
    const uint16_t* wr = W + static_cast<uint64_t>(r) * d.k;
    const uint16_t* x0 = X + static_cast<uint64_t>(m0 + 0u) * d.ldx;
    const uint16_t* x1 = X + static_cast<uint64_t>(m0 + (nc > 1u ? 1u : 0u)) * d.ldx;
    const uint16_t* x2 = X + static_cast<uint64_t>(m0 + (nc > 2u ? 2u : 0u)) * d.ldx;
    const uint16_t* x3 = X + static_cast<uint64_t>(m0 + (nc > 3u ? 3u : 0u)) * d.ldx;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint32_t c = lane * 4u; c < d.k; c += 128u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            float wv = bf16_to_f32(__ldg(&wr[c + j]));
            a0 = fmaf(wv, bf16_to_f32(__ldg(&x0[c + j])), a0);
            if (nc > 1u) a1 = fmaf(wv, bf16_to_f32(__ldg(&x1[c + j])), a1);
            if (nc > 2u) a2 = fmaf(wv, bf16_to_f32(__ldg(&x2[c + j])), a2);
            if (nc > 3u) a3 = fmaf(wv, bf16_to_f32(__ldg(&x3[c + j])), a3);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, off);
        if (nc > 1u) a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, off);
        if (nc > 2u) a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, off);
        if (nc > 3u) a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, off);
    }
    if (lane == 0u) {
        OUT[static_cast<uint64_t>(m0 + 0u) * d.ldo + r] = f32_to_bf16_u16(a0);
        if (nc > 1u) OUT[static_cast<uint64_t>(m0 + 1u) * d.ldo + r] = f32_to_bf16_u16(a1);
        if (nc > 2u) OUT[static_cast<uint64_t>(m0 + 2u) * d.ldo + r] = f32_to_bf16_u16(a2);
        if (nc > 3u) OUT[static_cast<uint64_t>(m0 + 3u) * d.ldo + r] = f32_to_bf16_u16(a3);
    }
}

// ----------------------------------------------------------------------------
// 6. Clean Host Launch Helper
// ----------------------------------------------------------------------------
inline cudaError_t launch_dense(
    const uint32_t* W, const __half* scales, const PwV3Tables* T,
    const uint16_t* X, uint16_t* Y,
    __half* xrot_buf, float* yr_buf, size_t max_yr_elems,
    uint32_t bits, uint32_t rows, uint32_t n, uint32_t k,
    uint32_t ldx, uint32_t ldo, uint32_t sr_split,
    cudaStream_t stream = nullptr
) {
    const uint32_t n_bc = k >> 7;
    if (rows <= 16u) {
        bool use_sk8 = (n <= 34816u) && (n_bc >= 8u) && ((n_bc & 7u) == 0u) &&
                       (static_cast<size_t>(rows) * n * 8ull <= max_yr_elems);
        DenseDims d{rows, n, k, ldx, ldo, sr_split, use_sk8 ? 8u : 1u, 0u};
        tfq3_fwht_x_kernel<<<dim3(n_bc, rows, 1), 32, 0, stream>>>(X, xrot_buf, d);
        if (use_sk8) {
            dim3 g(n / 32u, 8u, 1u);
            if (rows == 1u) {
                if (bits == 3u) tfq3_dec_2dsk8_kernel<3u, 1u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
                else if (bits == 4u) tfq3_dec_2dsk8_kernel<4u, 1u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
                else tfq3_dec_2dsk8_kernel<5u, 1u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
            } else if (rows == 2u) {
                if (bits == 3u) tfq3_dec_2dsk8_kernel<3u, 2u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
                else if (bits == 4u) tfq3_dec_2dsk8_kernel<4u, 2u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
                else tfq3_dec_2dsk8_kernel<5u, 2u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
            } else {
                if (bits == 3u) tfq3_dec_2dsk8_kernel<3u, 4u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
                else if (bits == 4u) tfq3_dec_2dsk8_kernel<4u, 4u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
                else tfq3_dec_2dsk8_kernel<5u, 4u><<<g, 64, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
            }
            tfq3_fwht_y_kernel<<<dim3(n / 128u, rows, 1), 32, 0, stream>>>(yr_buf, Y, d);
        } else {
            dim3 g(n / 128u, 1u, 1u);
            if (rows == 1u) {
                if (bits == 3u) tfq3_dec_fused_kernel<3u, 1u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
                else if (bits == 4u) tfq3_dec_fused_kernel<4u, 1u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
                else tfq3_dec_fused_kernel<5u, 1u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
            } else if (rows == 2u) {
                if (bits == 3u) tfq3_dec_fused_kernel<3u, 2u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
                else if (bits == 4u) tfq3_dec_fused_kernel<4u, 2u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
                else tfq3_dec_fused_kernel<5u, 2u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
            } else {
                if (bits == 3u) tfq3_dec_fused_kernel<3u, 4u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
                else if (bits == 4u) tfq3_dec_fused_kernel<4u, 4u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
                else tfq3_dec_fused_kernel<5u, 4u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
            }
        }
        return cudaGetLastError();
    }
    DenseDims d{rows, n, k, ldx, ldo, sr_split, 1u, 0u};
    tfq3_fwht_x_kernel<<<dim3(n_bc, rows, 1), 32, 0, stream>>>(X, xrot_buf, d);
    dim3 gp(n / 128u, (rows + 31u) / 32u, 1u);
    if (bits == 3u) tfq3_dense_prefill_kernel<3u><<<gp, 128, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
    else if (bits == 4u) tfq3_dense_prefill_kernel<4u><<<gp, 128, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
    else tfq3_dense_prefill_kernel<5u><<<gp, 128, 0, stream>>>(W, scales, T, xrot_buf, Y, d);
    return cudaGetLastError();
}

} // namespace tfq3
