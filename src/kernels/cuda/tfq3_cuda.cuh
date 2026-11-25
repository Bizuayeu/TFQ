#pragma once
// ============================================================================
// TFQ3 (3/4/5-bit 2D-AffineCoset/half4 + 2D FWHT-128) & BF16 CUDA Kernels
//
// Reads the exact same binary `pw_v3_tables.bin` ("PWV3"), dense `.tffpw`
// ("TFPN"), and MoE `.tffpw` ("TFPW") shards as Apple Silicon Metal — one
// checkpoint works byte-for-byte on both Mac (Metal) and NVIDIA GPU (CUDA).
//
// Highlights:
// - Bank-conflict-free direct PTX `mma.sync.aligned.m16n8k16` split-K=8 dense
//   kernel (3D grid.z = ceil(rows/16)) — bit-exact across rows=1..64 for
//   speculative decoding (DFlash2 / DSpine / MTP) with zero <mma.h> dependency.
// - Fused PTX m16n8k16 + inv-FWHT-128 kernel for wide-N lm_head & prefill.
// - Fused mixed-bit (3b/4b/5b) HeatAlloc routed MoE gate+up+SwiGLU & down.
//
// Supported architectures: sm_80+ (Ampere, Ada, Hopper, Blackwell / GB10).
// ============================================================================
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
    uint32_t rows;
    uint32_t n;
    uint32_t k;
    uint32_t ldx;
    uint32_t ldo;
    uint32_t sr_split;
    uint32_t ksplit;
    uint32_t row_off;
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

__device__ __forceinline__ void pw_store(void* out, uint64_t idx, float v, uint32_t out_f32) {
    if (out_f32) {
        reinterpret_cast<float*>(out)[idx] = v;
    } else {
        reinterpret_cast<uint16_t*>(out)[idx] = f32_to_bf16_u16(v);
    }
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

__device__ __forceinline__ float pw_glu(float g, float u, float lim, uint32_t act_mode) {
    float gv = __bfloat162float(__float2bfloat16(g));
    float uv = __bfloat162float(__float2bfloat16(u));
    if (act_mode == 0u && lim > 0.0f) {
        float fl = __bfloat162float(__float2bfloat16(lim));
        gv = fminf(gv, fl);
        uv = fminf(fmaxf(uv, -fl), fl);
    }
    float sg = gv / (1.0f + __expf(-gv));
    if (act_mode != 0u) sg = __bfloat162float(__float2bfloat16(sg));
    return __bfloat162float(__float2bfloat16(sg * uv));
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
    void* __restrict__ OUT,
    DenseDims d,
    uint32_t out_f32
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
    const uint32_t sr0 = pw_sr0(r0 + d.row_off, d.sr_split);
    const uint64_t base = static_cast<uint64_t>(row) * d.ldo + r0;
    pw_store(OUT, base + lane +  0u, v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
    pw_store(OUT, base + lane + 32u, v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
    pw_store(OUT, base + lane + 64u, v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
    pw_store(OUT, base + lane + 96u, v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
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

// ----------------------------------------------------------------------------
// 3. Direct PTX m16n8k16 Dense Kernels (Split-K=8 & Fused, 3b/4b/5b)
//    Zero shared-memory bank conflicts (ldm=24/136) and bit-exact rows=1..64.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void pw_stage_b24(__half* dst, __half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b, int2 home) {
    __half2* p0 = reinterpret_cast<__half2*>(dst + home.y * 24 + home.x);
    __half2* p1 = reinterpret_cast<__half2*>(dst + (home.y + 8) * 24 + home.x);
    p0[0] = q0a; p0[1] = q0b; p1[0] = q1a; p1[1] = q1b;
}

__device__ __forceinline__ void pw_mma_f16_16x16x16(
    float (&b0)[8], const __half* __restrict__ tg_a, int kb, const __half* __restrict__ tg_b, uint32_t lane
) {
    uint32_t g = lane >> 2, t = (lane & 3u) << 1;
    const __half* a0p = tg_a + g * 136u + kb * 16 + t;
    const __half* a1p = a0p + 8u * 136u;
    uint32_t a0 = *reinterpret_cast<const uint32_t*>(a0p);
    uint32_t a1 = *reinterpret_cast<const uint32_t*>(a1p);
    uint32_t a2 = *reinterpret_cast<const uint32_t*>(a0p + 8u);
    uint32_t a3 = *reinterpret_cast<const uint32_t*>(a1p + 8u);
    const __half* b0p = tg_b + g * 24u + t;
    const __half* b1p = b0p + 8u * 24u;
    uint32_t b0_lo = *reinterpret_cast<const uint32_t*>(b0p);
    uint32_t b1_lo = *reinterpret_cast<const uint32_t*>(b0p + 8u);
    uint32_t b0_hi = *reinterpret_cast<const uint32_t*>(b1p);
    uint32_t b1_hi = *reinterpret_cast<const uint32_t*>(b1p + 8u);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%8,%9,%10,%11}, {%12,%13}, {%0,%1,%2,%3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%14,%15}, {%4,%5,%6,%7};\n"
        : "+f"(b0[0]), "+f"(b0[1]), "+f"(b0[2]), "+f"(b0[3]),
          "+f"(b0[4]), "+f"(b0[5]), "+f"(b0[6]), "+f"(b0[7])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_lo), "r"(b1_lo), "r"(b0_hi), "r"(b1_hi));
}

template <uint32_t BITS>
__device__ __forceinline__ void pw_mma_bc(
    const uint32_t* __restrict__ w_s, const __half* __restrict__ sc0, uint32_t bc,
    const PwV3Tables* __restrict__ T, uint32_t& st, uint32_t lane, int2 home,
    const __half* __restrict__ tg_a, __half* __restrict__ tg_b_warp, float (&acc)[8]
) {
    float b0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    __half2 q0a, q0b, q1a, q1b;
    if constexpr (BITS == 4u) {
        const uint4* p = reinterpret_cast<const uint4*>(w_s + static_cast<uint64_t>(bc) * 8ull * 32ull);
        for (uint32_t h = 0u; h < 2u; ++h) {
            uint4 v = __ldg(&p[h * 32u + lane]);
            uint32_t uv[4] = {v.x, v.y, v.z, v.w};
            int kb = static_cast<int>(h * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                pw_v3_q8_4b(st, uv[j], T, q0a, q0b, q1a, q1b);
                pw_stage_b24(tg_b_warp, q0a, q0b, q1a, q1b, home);
                __syncwarp();
                pw_mma_f16_16x16x16(b0, tg_a, kb + j, tg_b_warp, lane);
            }
        }
    } else if constexpr (BITS == 3u) {
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t* p = w_s + (static_cast<uint64_t>(bc) * 6ull + h * 3ull) * 32ull + lane;
            uint32_t a0 = __ldg(&p[0]), a1 = __ldg(&p[32]), a2 = __ldg(&p[64]);
            uint32_t mv[4] = {a0 >> 8, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu, a2 & 0xFFFFFFu};
            int kb = static_cast<int>(h * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                pw_v3_q8_3b(st, mv[j], T, q0a, q0b, q1a, q1b);
                pw_stage_b24(tg_b_warp, q0a, q0b, q1a, q1b, home);
                __syncwarp();
                pw_mma_f16_16x16x16(b0, tg_a, kb + j, tg_b_warp, lane);
            }
        }
    } else {
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t* p = w_s + (static_cast<uint64_t>(bc) * 10ull + h * 5ull) * 32ull + lane;
            uint32_t w0 = __ldg(&p[0]), w1 = __ldg(&p[32]), w2 = __ldg(&p[64]), w3 = __ldg(&p[96]), w4 = __ldg(&p[128]);
            uint32_t hv[4] = {w0 >> 24, (w1 >> 16) & 255u, (w2 >> 8) & 255u, w3 & 255u};
            uint32_t lv[4] = {(w0 << 8) | (w1 >> 24), (w1 << 16) | (w2 >> 16), (w2 << 24) | (w3 >> 8), w4};
            int kb = static_cast<int>(h * 4u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                pw_v3_q8_5b(st, hv[j], lv[j], T, q0a, q0b, q1a, q1b);
                pw_stage_b24(tg_b_warp, q0a, q0b, q1a, q1b, home);
                __syncwarp();
                pw_mma_f16_16x16x16(b0, tg_a, kb + j, tg_b_warp, lane);
            }
        }
    }
    float fsv = __half2float(__ldg(&sc0[bc]));
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = fmaf(b0[i], fsv, acc[i]);
}

// Split-K=8 BM=16 MMA (4 warps = 64 output rows/CTA): grid=(N/64, 8, ceil(rows/16)), block=128.
template <uint32_t BITS>
__global__ void tfq3_mma_sk8_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    float* __restrict__ YR,
    DenseDims d
) {
    const uint32_t blk = blockIdx.x, kp = blockIdx.y, m0 = blockIdx.z * 16u, n_bc = d.k >> 7;
    if (m0 >= d.rows) return;
    const uint32_t nb = (d.rows - m0) < 16u ? (d.rows - m0) : 16u;
    const uint32_t chunk = n_bc >> 3, bc0 = kp * chunk, bc1 = bc0 + chunk;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u, strip = blk * 4u + sg;
    const uint32_t* w_s = W + static_cast<uint64_t>(strip) * n_bc * (BITS * 2u) * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(strip >> 3) * n_bc;
    uint32_t st = pw_init_st<BITS>(w_s, bc0, lane);
    const int2 home = pw_frag_home(lane);
    __shared__ __half sm_a[16 * 136];
    __shared__ __half sm_b[4][16 * 24];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t bc = bc0; bc < bc1; ++bc) {
        for (uint32_t idx = tid; idx < 256u; idx += 128u) {
            uint32_t rl = idx >> 4, cl = (idx & 15u) << 3;
            uint4 v = (rl < nb) ? *reinterpret_cast<const uint4*>(X_rot + static_cast<uint64_t>(m0 + rl) * d.k + (bc << 7) + cl) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(sm_a + rl * 136u + cl) = v;
        }
        __syncthreads();
        pw_mma_bc<BITS>(w_s, sc0, bc, T, st, lane, home, sm_a, sm_b[sg], acc);
        __syncthreads();
    }
    const uint32_t g = lane >> 2, t = (lane & 3u) << 1;
    const uint64_t pbase = static_cast<uint64_t>(kp) * d.rows * d.n + blk * 64u + sg * 16u + t;
    if (g < nb) {
        float* d0 = YR + pbase + static_cast<uint64_t>(m0 + g) * d.n;
        *reinterpret_cast<float2*>(d0) = make_float2(acc[0], acc[1]);
        *reinterpret_cast<float2*>(d0 + 8u) = make_float2(acc[4], acc[5]);
    }
    if (g + 8u < nb) {
        float* d1 = YR + pbase + static_cast<uint64_t>(m0 + g + 8u) * d.n;
        *reinterpret_cast<float2*>(d1) = make_float2(acc[2], acc[3]);
        *reinterpret_cast<float2*>(d1 + 8u) = make_float2(acc[6], acc[7]);
    }
}

// Fused BM=16 MMA + inv-FWHT-128 (8 warps = 128 output rows/CTA): grid=(N/128, ceil(rows/16)), block=256.
template <uint32_t BITS>
__global__ void tfq3_mma_fused_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    void* __restrict__ OUT,
    DenseDims d,
    uint32_t out_f32
) {
    const uint32_t rb = blockIdx.x, r0 = rb * 128u, m0 = blockIdx.y * 16u;
    if (m0 >= d.rows) return;
    const uint32_t nb = (d.rows - m0) < 16u ? (d.rows - m0) : 16u, n_bc = d.k >> 7;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u, strip = rb * 8u + sg;
    const uint32_t* w_s = W + static_cast<uint64_t>(strip) * n_bc * (BITS * 2u) * 32ull;
    const __half* sc0 = scales + static_cast<uint64_t>(rb) * n_bc;
    uint32_t st = pw_init_st<BITS>(w_s, 0u, lane);
    const int2 home = pw_frag_home(lane);
    __shared__ union {
        struct { __half a[16 * 136]; __half b[8][16 * 24]; } ab;
        float f[16 * 128];
    } sm;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t bc = 0u; bc < n_bc; ++bc) {
        uint32_t rl = tid >> 4, cl = (tid & 15u) << 3;
        uint4 v = (rl < nb) ? *reinterpret_cast<const uint4*>(X_rot + static_cast<uint64_t>(m0 + rl) * d.k + (bc << 7) + cl) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(sm.ab.a + rl * 136u + cl) = v;
        __syncthreads();
        pw_mma_bc<BITS>(w_s, sc0, bc, T, st, lane, home, sm.ab.a, sm.ab.b[sg], acc);
        __syncthreads();
    }
    const uint32_t g = lane >> 2, t = (lane & 3u) << 1, col = sg * 16u + t;
    *reinterpret_cast<float2*>(sm.f + g * 128u + col) = make_float2(acc[0], acc[1]);
    *reinterpret_cast<float2*>(sm.f + g * 128u + col + 8u) = make_float2(acc[4], acc[5]);
    *reinterpret_cast<float2*>(sm.f + (g + 8u) * 128u + col) = make_float2(acc[2], acc[3]);
    *reinterpret_cast<float2*>(sm.f + (g + 8u) * 128u + col + 8u) = make_float2(acc[6], acc[7]);
    __syncthreads();
    const uint32_t sr0 = pw_sr0(r0 + d.row_off, d.sr_split);
    for (uint32_t m = sg; m < nb; m += 8u) {
        const float* ys = sm.f + m * 128u;
        float4 v = make_float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        const uint64_t base = static_cast<uint64_t>(m0 + m) * d.ldo + r0;
        pw_store(OUT, base + lane +  0u, v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u), out_f32);
        pw_store(OUT, base + lane + 32u, v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u), out_f32);
        pw_store(OUT, base + lane + 64u, v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u), out_f32);
        pw_store(OUT, base + lane + 96u, v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u), out_f32);
    }
}

// ----------------------------------------------------------------------------
// 4. Fused Routed MoE Kernels (Mixed 3b / 4b / 5b HeatAlloc Experts)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void pw_dot1_acc(
    __half2 q0a, __half2 q0b, __half2 q1a, __half2 q1b,
    const __half* __restrict__ xr, uint32_t cb, float& b0, float& b1
) {
    uint2 raw = *reinterpret_cast<const uint2*>(xr + cb);
    float2 fa = __half22float2(u32_to_h2(raw.x)), fb = __half22float2(u32_to_h2(raw.y));
    float2 g0a = __half22float2(q0a), g0b = __half22float2(q0b);
    float2 g1a = __half22float2(q1a), g1b = __half22float2(q1b);
    b0 = fmaf(g0a.x, fa.x, fmaf(g0a.y, fa.y, fmaf(g0b.x, fb.x, fmaf(g0b.y, fb.y, b0))));
    b1 = fmaf(g1a.x, fa.x, fmaf(g1a.y, fa.y, fmaf(g1b.x, fb.x, fmaf(g1b.y, fb.y, b1))));
}

template <uint32_t BITS>
__device__ __forceinline__ void pw_mv1_range(
    const uint32_t* __restrict__ w_base, const __half* __restrict__ sc_base,
    uint32_t n_bc, const PwV3Tables* __restrict__ T, const __half* __restrict__ xr,
    uint32_t lane, int2 home, float& acc0, float& acc1
) {
    uint32_t st = pw_init_st<BITS>(w_base, 0u, lane);
    acc0 = 0.0f; acc1 = 0.0f;
    for (uint32_t bc = 0u; bc < n_bc; ++bc) {
        float ba0 = 0.0f, ba1 = 0.0f;
        __half2 q0a, q0b, q1a, q1b;
        if constexpr (BITS == 4u) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
            uint32_t cb = (bc << 7) + static_cast<uint32_t>(home.x);
            #pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                uint4 v = __ldg(&w4[(bc * 2u + h) * 32u + lane]);
                uint32_t ch = cb + (h << 6);
                pw_v3_q8_4b(st, v.x, T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, ch +  0u, ba0, ba1);
                pw_v3_q8_4b(st, v.y, T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, ch + 16u, ba0, ba1);
                pw_v3_q8_4b(st, v.z, T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, ch + 32u, ba0, ba1);
                pw_v3_q8_4b(st, v.w, T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, ch + 48u, ba0, ba1);
            }
        } else if constexpr (BITS == 3u) {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 6ull + h * 3ull) * 32ull + lane;
                uint32_t a0 = __ldg(&blk[0]), a1 = __ldg(&blk[32]), a2 = __ldg(&blk[64]);
                uint32_t cb = (bc << 7) + (h << 6) + static_cast<uint32_t>(home.x);
                pw_v3_q8_3b(st, a0 >> 8,                               T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb +  0u, ba0, ba1);
                pw_v3_q8_3b(st, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 16u, ba0, ba1);
                pw_v3_q8_3b(st, ((a1 <<  8) | (a2 >> 24)) & 0xFFFFFFu, T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 32u, ba0, ba1);
                pw_v3_q8_3b(st, a2 & 0xFFFFFFu,                        T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 48u, ba0, ba1);
            }
        } else {
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t* blk = w_base + (static_cast<uint64_t>(bc) * 10ull + h * 5ull) * 32ull + lane;
                uint32_t w0 = __ldg(&blk[0]), w1 = __ldg(&blk[32]), w2 = __ldg(&blk[64]), w3 = __ldg(&blk[96]), w4 = __ldg(&blk[128]);
                uint32_t cb = (bc << 7) + (h << 6) + static_cast<uint32_t>(home.x);
                pw_v3_q8_5b(st, w0 >> 24,          (w0 <<  8) | (w1 >> 24), T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb +  0u, ba0, ba1);
                pw_v3_q8_5b(st, (w1 >> 16) & 255u, (w1 << 16) | (w2 >> 16), T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 16u, ba0, ba1);
                pw_v3_q8_5b(st, (w2 >>  8) & 255u, (w2 << 24) | (w3 >>  8), T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 32u, ba0, ba1);
                pw_v3_q8_5b(st, w3 & 255u,          w4,                     T, q0a, q0b, q1a, q1b); pw_dot1_acc(q0a, q0b, q1a, q1b, xr, cb + 48u, ba0, ba1);
            }
        }
        float sc = __half2float(__ldg(&sc_base[bc]));
        acc0 = fmaf(ba0, sc, acc0);
        acc1 = fmaf(ba1, sc, acc1);
    }
    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1); acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 8);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1); acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 8);
}

__device__ __forceinline__ void pw_mv1_dispatch(
    uint32_t bits, const uint32_t* __restrict__ w_base, const __half* __restrict__ sc_base,
    uint32_t n_bc, const PwV3Tables* __restrict__ T, const __half* __restrict__ xr,
    uint32_t lane, int2 home, float& acc0, float& acc1
) {
    if (bits == 4u) pw_mv1_range<4u>(w_base, sc_base, n_bc, T, xr, lane, home, acc0, acc1);
    else if (bits == 3u) pw_mv1_range<3u>(w_base, sc_base, n_bc, T, xr, lane, home, acc0, acc1);
    else pw_mv1_range<5u>(w_base, sc_base, n_bc, T, xr, lane, home, acc0, acc1);
}

__global__ void tfq3_moe_gu_swiglu_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const TfpwExpertMeta* __restrict__ metas,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ X_rot,
    const int32_t* __restrict__ picks,
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
    pw_mv1_dispatch(bits, wg, sg_p, n_bc, T, xr, lane, home, g0, g1);
    pw_mv1_dispatch(bits, wu, su_p, n_bc, T, xr, lane, home, u0, u1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + static_cast<uint32_t>(home.y);
        tg_g[idx] = g0; tg_g[idx + 8u] = g1;
        tg_u[idx] = u0; tg_u[idx + 8u] = u1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 gv = make_float4(tg_g[lane], tg_g[lane + 32u], tg_g[lane + 64u], tg_g[lane + 96u]);
        float4 uv = make_float4(tg_u[lane], tg_u[lane + 32u], tg_u[lane + 64u], tg_u[lane + 96u]);
        pw_fwht128_simd4(gv, lane);
        pw_fwht128_simd4(uv, lane);
        float4 z;
        z.x = pw_glu(gv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane +  0u);
        z.y = pw_glu(gv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane + 32u);
        z.z = pw_glu(gv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane + 64u);
        z.w = pw_glu(gv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), lim, act_mode) * pw_sgn(0x12345678u, r0 + lane + 96u);
        pw_fwht128_simd4(z, lane);
        __half* dst = act_rot + static_cast<uint64_t>(slot_flat) * inter + r0;
        dst[lane +  0u] = __float2half(z.x);
        dst[lane + 32u] = __float2half(z.y);
        dst[lane + 64u] = __float2half(z.z);
        dst[lane + 96u] = __float2half(z.w);
    }
}

__global__ void tfq3_moe_down_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ scales,
    const TfpwExpertMeta* __restrict__ metas,
    const PwV3Tables* __restrict__ T,
    const __half* __restrict__ act_rot,
    const int32_t* __restrict__ picks,
    void* __restrict__ ey,
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
    __shared__ float tg_y[128];
    float y0, y1;
    pw_mv1_dispatch(bits, wd, sd_p, n_bc, T, ar, lane, home, y0, y1);
    if ((lane & 9u) == 0u) {
        uint32_t idx = sg * 16u + static_cast<uint32_t>(home.y);
        tg_y[idx] = y0; tg_y[idx + 8u] = y1;
    }
    __syncthreads();
    if (sg == 0u) {
        float4 v = make_float4(tg_y[lane], tg_y[lane + 32u], tg_y[lane + 64u], tg_y[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        const uint64_t base = (static_cast<uint64_t>(row) * out_slot_stride + s) * dims + r0;
        pw_store(ey, base + lane +  0u, v.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), out_f32);
        pw_store(ey, base + lane + 32u, v.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), out_f32);
        pw_store(ey, base + lane + 64u, v.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), out_f32);
        pw_store(ey, base + lane + 96u, v.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), out_f32);
    }
}

// ----------------------------------------------------------------------------
// 5. BF16 Residual Linear Kernels
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
// 6. Clean Host Launch Helpers (Dense & Routed MoE)
// ----------------------------------------------------------------------------
inline cudaError_t launch_dense(
    const uint32_t* W, const __half* scales, const PwV3Tables* T,
    const uint16_t* X, void* Y,
    __half* xrot_buf, float* yr_buf, size_t max_yr_elems,
    uint32_t bits, uint32_t rows, uint32_t n, uint32_t k,
    uint32_t ldx, uint32_t ldo, uint32_t sr_split,
    bool out_f32 = false, uint32_t row_off = 0u,
    cudaStream_t stream = nullptr
) {
    const uint32_t n_bc = k >> 7;
    const size_t yr_rows = (yr_buf != nullptr) ? (max_yr_elems / (static_cast<size_t>(n) * 8ull)) : 0ull;
    const bool use_sk8 = (rows <= 64u) && (n <= 34816u) && (n_bc >= 8u) && ((n_bc & 7u) == 0u) && (yr_rows >= rows);
    DenseDims d{rows, n, k, ldx, ldo, sr_split, use_sk8 ? 8u : 1u, row_off};
    tfq3_fwht_x_kernel<<<dim3(n_bc, rows, 1), 32, 0, stream>>>(X, xrot_buf, d);
    if (use_sk8) {
        dim3 g(n / 64u, 8u, (rows + 15u) / 16u);
        if (bits == 3u) tfq3_mma_sk8_kernel<3u><<<g, 128, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
        else if (bits == 4u) tfq3_mma_sk8_kernel<4u><<<g, 128, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
        else tfq3_mma_sk8_kernel<5u><<<g, 128, 0, stream>>>(W, scales, T, xrot_buf, yr_buf, d);
        tfq3_fwht_y_kernel<<<dim3(n / 128u, rows, 1), 32, 0, stream>>>(yr_buf, Y, d, out_f32 ? 1u : 0u);
    } else {
        dim3 g(n / 128u, (rows + 15u) / 16u, 1u);
        if (bits == 3u) tfq3_mma_fused_kernel<3u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d, out_f32 ? 1u : 0u);
        else if (bits == 4u) tfq3_mma_fused_kernel<4u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d, out_f32 ? 1u : 0u);
        else tfq3_mma_fused_kernel<5u><<<g, 256, 0, stream>>>(W, scales, T, xrot_buf, Y, d, out_f32 ? 1u : 0u);
    }
    return cudaGetLastError();
}

inline cudaError_t launch_moe(
    const uint32_t* W, const __half* scales, const TfpwExpertMeta* metas, const PwV3Tables* T,
    const uint16_t* X, const int32_t* picks, void* ey,
    __half* xrot_buf, __half* act_rot_buf,
    uint32_t rows, uint32_t dims, uint32_t inter, uint32_t experts,
    uint32_t top_k, uint32_t x_stride, uint32_t pick_stride, uint32_t out_slot_stride,
    float lim = 0.0f, uint32_t act_mode = 0u, bool out_f32 = false,
    cudaStream_t stream = nullptr
) {
    DenseDims dx{rows, dims, dims, x_stride, dims, 0u, 1u, 0u};
    tfq3_fwht_x_kernel<<<dim3(dims >> 7, rows, 1), 32, 0, stream>>>(X, xrot_buf, dx);
    tfq3_moe_gu_swiglu_kernel<<<dim3(inter >> 7, rows * top_k, 1), 256, 0, stream>>>(
        W, scales, metas, T, xrot_buf, picks, act_rot_buf, dims, inter, experts, top_k, pick_stride, lim, act_mode);
    tfq3_moe_down_kernel<<<dim3(dims >> 7, rows * top_k, 1), 256, 0, stream>>>(
        W, scales, metas, T, act_rot_buf, picks, ey, dims, inter, experts, top_k, pick_stride, out_slot_stride, out_f32 ? 1u : 0u);
    return cudaGetLastError();
}

} // namespace tfq3
