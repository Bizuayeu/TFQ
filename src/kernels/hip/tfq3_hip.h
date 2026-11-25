#pragma once
// ============================================================================
// TensorFold TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128)
// Standalone Header-Only AMD ROCm / HIP Inference Kernels
//
// Byte-identical checkpoint format with Apple Silicon Metal & NVIDIA CUDA:
//   - pw_v3_tables.bin ("PWV3", 16 KiB universal Gaussian codebook)
//   - .tffpw ("TFPN" dense/non-MoE & "TFPW" mixed-bit routed MoE shards)
//   - Portable across RDNA3/4 (gfx1100/gfx1200, wave32) and CDNA2/3
//     (gfx90a/gfx942, wave64) via explicit width=32 __shfl_xor sub-waves.
//   - Zero-repack Row-TP (state-hash resume) & Col-TP multi-GPU sharding.
// ============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace tfq3_hip {

struct PwV3Tables {
    uint32_t magic;
    uint32_t version;
    float a3, b3, a4, b4, a5, b5;
    uint32_t pad0[8];
    uint32_t lut3[256];
    uint32_t lut4[1024];
    uint32_t lut5[1024];
    uint2 c64_3[64];
    uint2 c64_4[64];
    uint2 c64_5[64];
};
static_assert(sizeof(PwV3Tables) == 10816, "PwV3Tables layout must match PWV3 v3");

struct TfpwExpertMeta {
    uint32_t b_bits;
    uint32_t heat_rank;
    uint64_t gate_word_off;
    uint64_t up_word_off;
    uint64_t down_word_off;
    uint32_t gate_sc_off;
    uint32_t up_sc_off;
    uint32_t down_sc_off;
    float    gate_snr_db;
    float    up_snr_db;
    float    down_snr_db;
    uint32_t reserved[2];
};
static_assert(sizeof(TfpwExpertMeta) == 64, "TfpwExpertMeta layout must match TFPW v3");

struct DenseDims {
    uint32_t rows, n, k, ldx, ldo, sr_split, row_off, k_sign_off;
};

struct TpRange {
    uint32_t strip_off, sc_row_off, bc0, bc1, n_bc_total;
};

__device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
    __half2 h;
    __builtin_memcpy(&h, &u, sizeof(h));
    return h;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_u16(float f) {
    hip_bfloat16 b(f);
    uint16_t u;
    __builtin_memcpy(&u, &b, sizeof(u));
    return u;
}

__device__ __forceinline__ void pw_store(void* out, uint64_t idx, float v, uint32_t out_f32) {
    if (out_f32) reinterpret_cast<float*>(out)[idx] = v;
    else reinterpret_cast<uint16_t*>(out)[idx] = f32_to_bf16_u16(v);
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

// Explicit width=32 sub-wave FWHT-128: identical semantics on wave32 (RDNA3/4) and wave64 (CDNA2/3).
__device__ __forceinline__ void pw_fwht128_simd4(float4& v, uint32_t lane32) {
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1) {
        float ox = __shfl_xor(v.x, d, 32);
        float oy = __shfl_xor(v.y, d, 32);
        float oz = __shfl_xor(v.z, d, 32);
        float ow = __shfl_xor(v.w, d, 32);
        bool lo = ((lane32 & static_cast<uint32_t>(d)) == 0u);
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

__device__ __forceinline__ float pw_glu(float g, float u, float lim, uint32_t act_mode) {
    float gv = bf16_to_f32(f32_to_bf16_u16(g));
    float uv = bf16_to_f32(f32_to_bf16_u16(u));
    if (act_mode == 0u && lim > 0.0f) {
        float fl = bf16_to_f32(f32_to_bf16_u16(lim));
        gv = fminf(gv, fl);
        uv = fminf(fmaxf(uv, -fl), fl);
    }
    float sg = gv / (1.0f + expf(-gv));
    if (act_mode != 0u) sg = bf16_to_f32(f32_to_bf16_u16(sg));
    return bf16_to_f32(f32_to_bf16_u16(sg * uv));
}

__global__ void tfq3_fwht_x_kernel(const uint16_t* __restrict__ X, __half* __restrict__ X_rot, DenseDims d) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.y;
    const uint32_t c0 = blockIdx.x * 128u;
    const uint32_t cs = c0 + d.k_sign_off;
    if (row >= d.rows || c0 >= d.k) return;
    const uint16_t* src = X + static_cast<uint64_t>(row) * d.ldx + c0;
    float4 v;
    v.x = bf16_to_f32(src[lane +  0u]) * pw_sgn(0x12345678u, cs + lane +  0u);
    v.y = bf16_to_f32(src[lane + 32u]) * pw_sgn(0x12345678u, cs + lane + 32u);
    v.z = bf16_to_f32(src[lane + 64u]) * pw_sgn(0x12345678u, cs + lane + 64u);
    v.w = bf16_to_f32(src[lane + 96u]) * pw_sgn(0x12345678u, cs + lane + 96u);
    pw_fwht128_simd4(v, lane);
    __half* dst = X_rot + static_cast<uint64_t>(row) * d.k + c0;
    dst[lane +  0u] = __float2half(v.x);
    dst[lane + 32u] = __float2half(v.y);
    dst[lane + 64u] = __float2half(v.z);
    dst[lane + 96u] = __float2half(v.w);
}

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
    uint2 c0 = T->c64_4[(h0 >> 16) & 63u], c2 = T->c64_4[(h2 >> 16) & 63u];
    __half2 c0s = u32_to_h2(c0.x), c0t = u32_to_h2(c0.y);
    __half2 c2s = u32_to_h2(c2.x), c2t = u32_to_h2(c2.y);
    q0a = __hfma2(u32_to_h2(T->lut4[(((wc >> 24)        << 2) ^ (h0 >> 22)) & 1023u]), c0s, c0t);
    q0b = __hfma2(u32_to_h2(T->lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u]), c0s, c0t);
    q1a = __hfma2(u32_to_h2(T->lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u]), c2s, c2t);
    q1b = __hfma2(u32_to_h2(T->lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u]), c2s, c2t);
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
    uint2 c0 = T->c64_3[(h0 >> 16) & 63u], c2 = T->c64_3[(h2 >> 16) & 63u];
    __half2 c0s = u32_to_h2(c0.x), c0t = u32_to_h2(c0.y);
    __half2 c2s = u32_to_h2(c2.x), c2t = u32_to_h2(c2.y);
    q0a = __hfma2(u32_to_h2(T->lut3[(((m24 >> 16) & 0xFCu) ^ (h0 >> 24)) & 255u]), c0s, c0t);
    q0b = __hfma2(u32_to_h2(T->lut3[(((m24 >> 10) & 0xFCu) ^ (h1 >> 24)) & 255u]), c0s, c0t);
    q1a = __hfma2(u32_to_h2(T->lut3[(((m24 >>  4) & 0xFCu) ^ (h2 >> 24)) & 255u]), c2s, c2t);
    q1b = __hfma2(u32_to_h2(T->lut3[(((m24 <<  2) & 0xFCu) ^ (h3 >> 24)) & 255u]), c2s, c2t);
}

__device__ __forceinline__ void pw_v3_q8_5b(
    uint32_t& st, uint32_t hi8, uint32_t lo32, const PwV3Tables* __restrict__ T,
    __half2& q0a, __half2& q0b, __half2& q1a, __half2& q1b
) {
    uint32_t s0 = st;
    uint32_t b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023u;
    uint32_t b1 = (lo32 >> 20) & 1023u, b2 = (lo32 >> 10) & 1023u, b3 = lo32 & 1023u;
    uint32_t h0 = s0 * 0x9E3779B1u;
    uint32_t h1 = ((s0 << 10) | b0) * 0x9E3779B1u;
    uint32_t h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) * 0x9E3779B1u;
    uint32_t h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) * 0x9E3779B1u;
    st = lo32;
    uint2 c0 = T->c64_5[(h0 >> 16) & 63u], c2 = T->c64_5[(h2 >> 16) & 63u];
    __half2 c0s = u32_to_h2(c0.x), c0t = u32_to_h2(c0.y);
    __half2 c2s = u32_to_h2(c2.x), c2t = u32_to_h2(c2.y);
    q0a = __hfma2(u32_to_h2(T->lut5[(b0 ^ (h0 >> 22)) & 1023u]), c0s, c0t);
    q0b = __hfma2(u32_to_h2(T->lut5[(b1 ^ (h1 >> 22)) & 1023u]), c0s, c0t);
    q1a = __hfma2(u32_to_h2(T->lut5[(b2 ^ (h2 >> 22)) & 1023u]), c2s, c2t);
    q1b = __hfma2(u32_to_h2(T->lut5[(b3 ^ (h3 >> 22)) & 1023u]), c2s, c2t);
}

template <uint32_t BITS>
__device__ __forceinline__ uint32_t pw_init_st(const uint32_t* __restrict__ w_base, uint32_t bc0, uint32_t lane32) {
    if (bc0 == 0u) return 0x13579BDFu ^ (lane32 * 0x9E3779B1u);
    if constexpr (BITS == 4u) {
        const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
        return w4[(static_cast<uint64_t>(bc0) * 2ull - 1ull) * 32ull + lane32].w;
    } else {
        return w_base[(static_cast<uint64_t>(bc0) * (BITS * 2u) - 1ull) * 32ull + lane32];
    }
}

__device__ __forceinline__ void pw_dot_acc(
    __half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b,
    const __half* __restrict__ xr, uint32_t cb, float& b0, float& b1
) {
    float2 fa = __half22float2(*reinterpret_cast<const __half2*>(xr + cb));
    float2 fb = __half22float2(*reinterpret_cast<const __half2*>(xr + cb + 2u));
    float2 g0a = __half22float2(q0a), g0b = __half22float2(q0b);
    float2 g1a = __half22float2(q1a), g1b = __half22float2(q1b);
    b0 = fmaf(g0a.x, fa.x, fmaf(g0a.y, fa.y, fmaf(g0b.x, fb.x, fmaf(g0b.y, fb.y, b0))));
    b1 = fmaf(g1a.x, fa.x, fmaf(g1a.y, fa.y, fmaf(g1b.x, fb.x, fmaf(g1b.y, fb.y, b1))));
}

template <uint32_t BITS>
__device__ __forceinline__ void pw_mv_range(
    const uint32_t* __restrict__ w_base, const __half* __restrict__ sc_base,
    uint32_t bc0, uint32_t bc1, const PwV3Tables* __restrict__ T,
    const __half* __restrict__ xr, uint32_t lane32, int2 home, float& acc0, float& acc1
) {
    uint32_t st = pw_init_st<BITS>(w_base, bc0, lane32);
    acc0 = 0.0f; acc1 = 0.0f;
    for (uint32_t bc = bc0; bc < bc1; ++bc) {
        float ba0 = 0.0f, ba1 = 0.0f;
        __half2 q0a, q0b, q1a, q1b;
        uint32_t x_off = ((bc - bc0) << 7) + static_cast<uint32_t>(home.x);
        if constexpr (BITS == 4u) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
            #pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                uint4 v = w4[(bc * 2u + h) * 32u + lane32];
                uint32_t ch = x_off + (h << 6);
                pw_v3_q8_4b(st, v.x, T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch +  0u, ba0, ba1);
                pw_v3_q8_4b(st, v.y, T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 16u, ba0, ba1);
                pw_v3_q8_4b(st, v.z, T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 32u, ba0, ba1);
                pw_v3_q8_4b(st, v.w, T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 48u, ba0, ba1);
            }
        } else if constexpr (BITS == 3u) {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 6ull + h * 3ull) * 32ull + lane32;
                uint32_t a0 = blk[0], a1 = blk[32], a2 = blk[64];
                uint32_t ch = x_off + (h << 6);
                pw_v3_q8_3b(st, a0 >> 8,                               T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch +  0u, ba0, ba1);
                pw_v3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 16u, ba0, ba1);
                pw_v3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 32u, ba0, ba1);
                pw_v3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 48u, ba0, ba1);
            }
        } else {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 10ull + h * 5ull) * 32ull + lane32;
                uint32_t w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
                uint32_t ch = x_off + (h << 6);
                pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch +  0u, ba0, ba1);
                pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 16u, ba0, ba1);
                pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 32u, ba0, ba1);
                pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0a, q0b, q1a, q1b); pw_dot_acc(q0a, q0b, q1a, q1b, xr, ch + 48u, ba0, ba1);
            }
        }
        float sc = __half2float(sc_base[bc]);
        acc0 = fmaf(ba0, sc, acc0);
        acc1 = fmaf(ba1, sc, acc1);
    }
    acc0 += __shfl_xor(acc0, 1, 32); acc0 += __shfl_xor(acc0, 8, 32);
    acc1 += __shfl_xor(acc1, 1, 32); acc1 += __shfl_xor(acc1, 8, 32);
}

// Dense + Multi-GPU TP Kernel (fuses inverse FWHT-128 per 128 output rows, block=256 = 8 sub-waves of 32).
template <uint32_t BITS>
__global__ void tfq3_dense_tp_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    void* __restrict__ OUT,
    DenseDims d, TpRange tp, uint32_t out_f32
) {
    const uint32_t rb = blockIdx.x, row = blockIdx.y, r0 = rb * 128u;
    if (row >= d.rows) return;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t strip = tp.strip_off + rb * 8u + sg;
    const uint32_t* w_s = W + static_cast<uint64_t>(strip) * tp.n_bc_total * (BITS * 2u) * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(tp.sc_row_off + rb) * tp.n_bc_total;
    const int2 home = pw_frag_home(lane);
    float acc0, acc1;
    pw_mv_range<BITS>(w_s, sc0, tp.bc0, tp.bc1, T, X_rot + static_cast<uint64_t>(row) * d.k, lane, home, acc0, acc1);
    __shared__ float sm_y[128];
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + static_cast<uint32_t>(home.y);
        sm_y[idx] = acc0;
        sm_y[idx + 8u] = acc1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 v = make_float4(sm_y[lane], sm_y[lane + 32u], sm_y[lane + 64u], sm_y[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        uint32_t sr0 = pw_sr0(r0 + d.row_off, d.sr_split);
        uint64_t base = static_cast<uint64_t>(row) * d.ldo + r0;
        pw_store(OUT, base + lane +  0u, v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
        pw_store(OUT, base + lane + 32u, v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
        pw_store(OUT, base + lane + 64u, v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
        pw_store(OUT, base + lane + 96u, v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
    }
}

__device__ __forceinline__ void pw_mv_dispatch(
    uint32_t bits, const uint32_t* __restrict__ w_base, const __half* __restrict__ sc_base,
    uint32_t n_bc, const PwV3Tables* __restrict__ T, const __half* __restrict__ xr,
    uint32_t lane32, int2 home, float& acc0, float& acc1
) {
    if (bits == 4u) pw_mv_range<4u>(w_base, sc_base, 0u, n_bc, T, xr, lane32, home, acc0, acc1);
    else if (bits == 3u) pw_mv_range<3u>(w_base, sc_base, 0u, n_bc, T, xr, lane32, home, acc0, acc1);
    else pw_mv_range<5u>(w_base, sc_base, 0u, n_bc, T, xr, lane32, home, acc0, acc1);
}

__global__ void tfq3_moe_gu_swiglu_kernel(
    const uint32_t* __restrict__ W, const __half* __restrict__ scales,
    const TfpwExpertMeta* __restrict__ metas, const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot, const int32_t* __restrict__ picks,
    __half* __restrict__ act_rot,
    uint32_t dims, uint32_t inter, uint32_t experts,
    uint32_t top_k, uint32_t pick_stride, float lim, uint32_t act_mode
) {
    const uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    const uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    const int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;
    const TfpwExpertMeta em = metas[eid];
    const uint32_t bits = em.b_bits, wpb = bits * 2u, n_bc = dims >> 7;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t strip = br * 8u + sg, r0 = br * 128u;
    const int2 home = pw_frag_home(lane);
    const __half* xr = X_rot + static_cast<uint64_t>(row) * dims;
    const uint32_t* wg = W + em.gate_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const uint32_t* wu = W + em.up_word_off   + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sg_p = scales + em.gate_sc_off + static_cast<uint64_t>(br) * n_bc;
    const __half* su_p = scales + em.up_sc_off   + static_cast<uint64_t>(br) * n_bc;
    __shared__ float tg_g[128], tg_u[128];
    float g0, g1, u0, u1;
    pw_mv_dispatch(bits, wg, sg_p, n_bc, T, xr, lane, home, g0, g1);
    pw_mv_dispatch(bits, wu, su_p, n_bc, T, xr, lane, home, u0, u1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + static_cast<uint32_t>(home.y);
        tg_g[idx] = g0; tg_g[idx + 8u] = g1;
        tg_u[idx] = u0; tg_u[idx + 8u] = u1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 vg = make_float4(tg_g[lane], tg_g[lane + 32u], tg_g[lane + 64u], tg_g[lane + 96u]);
        float4 vu = make_float4(tg_u[lane], tg_u[lane + 32u], tg_u[lane + 64u], tg_u[lane + 96u]);
        pw_fwht128_simd4(vg, lane); pw_fwht128_simd4(vu, lane);
        float4 act;
        act.x = pw_glu(vg.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), vu.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane +  0u);
        act.y = pw_glu(vg.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), vu.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane + 32u);
        act.z = pw_glu(vg.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), vu.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane + 64u);
        act.w = pw_glu(vg.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), vu.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane + 96u);
        pw_fwht128_simd4(act, lane);
        __half* dst = act_rot + static_cast<uint64_t>(slot_flat) * inter + r0;
        dst[lane +  0u] = __float2half(act.x); dst[lane + 32u] = __float2half(act.y);
        dst[lane + 64u] = __float2half(act.z); dst[lane + 96u] = __float2half(act.w);
    }
}

__global__ void tfq3_moe_down_kernel(
    const uint32_t* __restrict__ W, const __half* __restrict__ scales,
    const TfpwExpertMeta* __restrict__ metas, const PwV3Tables* __restrict__ T,
    const __half* __restrict__ act_rot, const int32_t* __restrict__ picks, void* __restrict__ ey,
    uint32_t dims, uint32_t inter, uint32_t experts,
    uint32_t top_k, uint32_t pick_stride, uint32_t out_slot_stride, uint32_t out_f32
) {
    const uint32_t br = blockIdx.x, slot_flat = blockIdx.y;
    const uint32_t row = slot_flat / top_k, s = slot_flat - row * top_k;
    const int32_t eid = picks[static_cast<uint64_t>(row) * pick_stride + s];
    if (eid < 0 || static_cast<uint32_t>(eid) >= experts) return;
    const TfpwExpertMeta em = metas[eid];
    const uint32_t bits = em.b_bits, wpb = bits * 2u, n_bc = inter >> 7;
    const uint32_t sg = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t strip = br * 8u + sg, r0 = br * 128u;
    const int2 home = pw_frag_home(lane);
    const __half* ar = act_rot + static_cast<uint64_t>(slot_flat) * inter;
    const uint32_t* wd = W + em.down_word_off + static_cast<uint64_t>(strip) * n_bc * wpb * 32ull;
    const __half* sd_p = scales + em.down_sc_off + static_cast<uint64_t>(br) * n_bc;
    __shared__ float tg_d[128];
    float d0, d1;
    pw_mv_dispatch(bits, wd, sd_p, n_bc, T, ar, lane, home, d0, d1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + static_cast<uint32_t>(home.y);
        tg_d[idx] = d0; tg_d[idx + 8u] = d1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 vd = make_float4(tg_d[lane], tg_d[lane + 32u], tg_d[lane + 64u], tg_d[lane + 96u]);
        pw_fwht128_simd4(vd, lane);
        uint64_t base = static_cast<uint64_t>(slot_flat) * out_slot_stride + r0;
        pw_store(ey, base + lane +  0u, vd.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), out_f32);
        pw_store(ey, base + lane + 32u, vd.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), out_f32);
        pw_store(ey, base + lane + 64u, vd.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), out_f32);
        pw_store(ey, base + lane + 96u, vd.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), out_f32);
    }
}

inline hipError_t launch_dense_tp(
    const uint32_t* W, const __half* scales, const PwV3Tables* T,
    const uint16_t* X_local, void* Y_local, __half* xrot_buf,
    uint32_t bits, uint32_t rows,
    uint32_t n_off, uint32_t n_local,
    uint32_t k_full, uint32_t k_off, uint32_t k_local,
    uint32_t ldx_local, uint32_t ldo_local,
    bool out_f32 = false, hipStream_t stream = nullptr
) {
    DenseDims d{rows, n_local, k_local, ldx_local, ldo_local, 0u, n_off, k_off};
    TpRange tp{n_off >> 4, n_off >> 7, k_off >> 7, (k_off + k_local) >> 7, k_full >> 7};
    hipLaunchKernelGGL(tfq3_fwht_x_kernel, dim3(k_local >> 7, rows, 1), dim3(32), 0, stream, X_local, xrot_buf, d);
    dim3 g(n_local >> 7, rows, 1);
    if (bits == 3u) hipLaunchKernelGGL(tfq3_dense_tp_kernel<3u>, g, dim3(256), 0, stream, W, scales, T, xrot_buf, Y_local, d, tp, out_f32 ? 1u : 0u);
    else if (bits == 4u) hipLaunchKernelGGL(tfq3_dense_tp_kernel<4u>, g, dim3(256), 0, stream, W, scales, T, xrot_buf, Y_local, d, tp, out_f32 ? 1u : 0u);
    else hipLaunchKernelGGL(tfq3_dense_tp_kernel<5u>, g, dim3(256), 0, stream, W, scales, T, xrot_buf, Y_local, d, tp, out_f32 ? 1u : 0u);
    return hipGetLastError();
}

} // namespace tfq3_hip
