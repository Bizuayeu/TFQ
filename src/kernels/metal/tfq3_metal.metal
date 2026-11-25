#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

// ============================================================================
// Standalone Reusable Apple Silicon Metal TFQ3 Inference, Prefill & TP Kernels
//   - Byte-identical PWV3 v3 / TFPN v3 format with NVIDIA CUDA & AMD HIP
//   - 2D FWHT-128 input/output rotation with Row-TP sign offset support
//   - Decode MV1 (rows=1..16) + High-Throughput Tiled simdgroup_matrix Prefill
//     (BM=32, BN=128 -> 32x weight reuse, fused inverse FWHT-128)
//   - Zero-Repack Multi-Mac Tensor Parallel (Row-TP via state-hash resume & Col-TP)
// ============================================================================
struct PwV3Tables {
    uint  magic, version;
    float a3, b3, a4, b4, a5, b5;
    uint  pad0[8];
    uint  lut3[256];
    uint  lut4[1024];
    uint  lut5[1024];
    half4 c64_3[64];
    half4 c64_4[64];
    half4 c64_5[64];
};

struct MetalDenseParams {
    uint rows;
    uint n;
    uint k;
    uint ldx;
    uint ldo;
    uint row_off;
    uint k_sign_off;
    uint strip_off;
    uint sc_row_off;
    uint bc0;
    uint bc1;
    uint n_bc_total;
};

inline short2 pw_frag_home(ushort l) {
    return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

inline float pw_sgn(uint seed, uint i) {
    uint h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline void pw_fwht128_simd4(thread float4& v, ushort lane) {
    TF_UNROLL
    for (ushort d = 1; d <= 16; d <<= 1) {
        float4 o = simd_shuffle_xor(v, d);
        v = ((lane & d) == 0u) ? (v + o) : (o - v);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

kernel void tfq3_fwht_x(
    const device half* X        [[buffer(0)]],
    device half* X_rot          [[buffer(1)]],
    constant MetalDenseParams& P [[buffer(2)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint row = gid.y, c0 = gid.x * 128u, cs = c0 + P.k_sign_off;
    if (row >= P.rows || c0 >= P.k) return;
    const device half* src = X + ulong(row) * P.ldx + c0;
    float4 v;
    v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, cs + lane +  0u);
    v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, cs + lane + 32u);
    v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, cs + lane + 64u);
    v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, cs + lane + 96u);
    pw_fwht128_simd4(v, lane);
    device half* dst = X_rot + ulong(row) * P.k + c0;
    dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
    dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
}

inline void pw_v3_q8_4b(thread uint& st, uint wc, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
    uint s0 = st;
    uint h0 = s0 * 0x9E3779B1u;
    uint h1 = ((s0 <<  8) | (wc >> 24)) * 0x9E3779B1u;
    uint h2 = ((s0 << 16) | (wc >> 16)) * 0x9E3779B1u;
    uint h3 = ((s0 << 24) | (wc >>  8)) * 0x9E3779B1u;
    st = wc;
    half4 c0 = T->c64_4[(h0 >> 16) & 63u], c2 = T->c64_4[(h2 >> 16) & 63u];
    q0 = fma(half4(as_type<half2>(T->lut4[(((wc >> 24)        << 2) ^ (h0 >> 22)) & 1023u]),
                   as_type<half2>(T->lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
    q1 = fma(half4(as_type<half2>(T->lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u]),
                   as_type<half2>(T->lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_3b(thread uint& st, uint m24, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
    uint s0 = st;
    uint h0 = s0 * 0x9E3779B1u;
    uint h1 = ((s0 <<  6) | (m24 >> 18)) * 0x9E3779B1u;
    uint h2 = ((s0 << 12) | (m24 >> 12)) * 0x9E3779B1u;
    uint h3 = ((s0 << 18) | (m24 >>  6)) * 0x9E3779B1u;
    st = (s0 << 24) | m24;
    half4 c0 = T->c64_3[(h0 >> 16) & 63u], c2 = T->c64_3[(h2 >> 16) & 63u];
    q0 = fma(half4(as_type<half2>(T->lut3[(((m24 >> 16) & 0xFCu) ^ (h0 >> 24)) & 255u]),
                   as_type<half2>(T->lut3[(((m24 >> 10) & 0xFCu) ^ (h1 >> 24)) & 255u])), c0.xyxy, c0.zwzw);
    q1 = fma(half4(as_type<half2>(T->lut3[(((m24 >>  4) & 0xFCu) ^ (h2 >> 24)) & 255u]),
                   as_type<half2>(T->lut3[(((m24 <<  2) & 0xFCu) ^ (h3 >> 24)) & 255u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_5b(thread uint& st, uint hi8, uint lo32, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
    uint s0 = st;
    uint b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023u;
    uint b1 = (lo32 >> 20) & 1023u;
    uint b2 = (lo32 >> 10) & 1023u;
    uint b3 = lo32 & 1023u;
    uint h0 = s0 * 0x9E3779B1u;
    uint h1 = ((s0 << 10) | b0) * 0x9E3779B1u;
    uint h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) * 0x9E3779B1u;
    uint h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) * 0x9E3779B1u;
    st = lo32;
    half4 c0 = T->c64_5[(h0 >> 16) & 63u], c2 = T->c64_5[(h2 >> 16) & 63u];
    q0 = fma(half4(as_type<half2>(T->lut5[(b0 ^ (h0 >> 22)) & 1023u]),
                   as_type<half2>(T->lut5[(b1 ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
    q1 = fma(half4(as_type<half2>(T->lut5[(b2 ^ (h2 >> 22)) & 1023u]),
                   as_type<half2>(T->lut5[(b3 ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

template <uint BITS>
inline uint pw_init_st(const device uint* w_base, uint bc0, ushort lane) {
    if (bc0 == 0u) return 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
    if constexpr (BITS == 4u) {
        const device uint4* w4 = reinterpret_cast<const device uint4*>(w_base);
        return w4[(ulong(bc0) * 2ul - 1ul) * 32ul + lane].w;
    } else {
        return w_base[(ulong(bc0) * (BITS * 2u) - 1ul) * 32ul + lane];
    }
}

// High-Throughput Tiled simdgroup_matrix Prefill + Zero-Repack TP Kernel (BM=32, BN=128):
// 8 simdgroups (256 threads) cooperatively load 32x128 X_rot, each simdgroup decodes 16x16 W once
// and runs 4x2 simdgroup_multiply_accumulate (32x weight reuse), then runs fused inverse FWHT-128.
template <uint BITS>
kernel void tfq3_prefill_tp_impl(
    const device uint* W        [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    constant PwV3Tables* T      [[buffer(2)]],
    const device half* X_rot    [[buffer(3)]],
    device float* OUT           [[buffer(4)]],
    constant MetalDenseParams& P [[buffer(5)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort sg                   [[simdgroup_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint rb = gid.x, r0 = rb * 128u, m0 = gid.y * 32u;
    if (m0 >= P.rows) return;
    uint nb = min(32u, P.rows - m0);
    uint strip = P.strip_off + rb * 8u + uint(sg);
    const device uint* w_s = W + ulong(strip) * P.n_bc_total * (BITS * 2u) * 32ul;
    const device half* sc0 = scales + ulong(P.sc_row_off + rb) * P.n_bc_total;
    uint st = pw_init_st<BITS>(w_s, P.bc0, lane);
    short2 home = pw_frag_home(lane);

    // 32x128 X_rot (8 KiB) + 8x(16x16) decoded W (4 KiB) + 32x128 float epilogue (16 KiB)
    threadgroup half sm_a[32 * 128];
    threadgroup half sm_b[8][16 * 16];
    threadgroup float sm_out[32 * 128];

    simdgroup_float8x8 acc[4][2];
    TF_UNROLL for (int m = 0; m < 4; ++m)
        TF_UNROLL for (int n = 0; n < 2; ++n)
            acc[m][n] = simdgroup_float8x8(0.0f);

    for (uint bc = P.bc0; bc < P.bc1; ++bc) {
        uint x_col = (bc - P.bc0) * 128u;
        for (uint idx = tid; idx < 1024u; idx += 256u) {
            uint rl = idx >> 5, cl = (idx & 31u) << 2;
            half4 v = (rl < nb) ? *reinterpret_cast<const device half4*>(X_rot + ulong(m0 + rl) * P.k + x_col + cl) : half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(sm_a + rl * 128u + cl) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 blk_acc[4][2];
        TF_UNROLL for (int m = 0; m < 4; ++m)
            TF_UNROLL for (int n = 0; n < 2; ++n)
                blk_acc[m][n] = simdgroup_float8x8(0.0f);

        auto mma_16x16 = [&](uint kt, half4 q0, half4 q1) {
            // Stage transposed W [k, n] into sm_b[sg][16][16] so simdgroup_load reads B directly.
            threadgroup half* sb = sm_b[sg];
            uint n0 = uint(home.y), n1 = n0 + 8u, k0 = uint(home.x);
            sb[(k0 + 0u) * 16u + n0] = q0.x; sb[(k0 + 1u) * 16u + n0] = q0.y;
            sb[(k0 + 2u) * 16u + n0] = q0.z; sb[(k0 + 3u) * 16u + n0] = q0.w;
            sb[(k0 + 0u) * 16u + n1] = q1.x; sb[(k0 + 1u) * 16u + n1] = q1.y;
            sb[(k0 + 2u) * 16u + n1] = q1.z; sb[(k0 + 3u) * 16u + n1] = q1.w;
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 B00, B01, B10, B11;
            simdgroup_load(B00, sb,              16u);
            simdgroup_load(B01, sb + 8u,         16u);
            simdgroup_load(B10, sb + 8u * 16u,   16u);
            simdgroup_load(B11, sb + 8u * 16u + 8u, 16u);

            uint kc = kt * 16u;
            TF_UNROLL for (uint m = 0; m < 4u; ++m) {
                if (m * 8u < nb) {
                    simdgroup_half8x8 A0, A1;
                    simdgroup_load(A0, sm_a + (m * 8u) * 128u + kc,      128u);
                    simdgroup_load(A1, sm_a + (m * 8u) * 128u + kc + 8u, 128u);
                    simdgroup_multiply_accumulate(blk_acc[m][0], A0, B00, blk_acc[m][0]);
                    simdgroup_multiply_accumulate(blk_acc[m][1], A0, B01, blk_acc[m][1]);
                    simdgroup_multiply_accumulate(blk_acc[m][0], A1, B10, blk_acc[m][0]);
                    simdgroup_multiply_accumulate(blk_acc[m][1], A1, B11, blk_acc[m][1]);
                }
            }
        };

        half4 q0, q1;
        if constexpr (BITS == 4u) {
            const device uint4* p = reinterpret_cast<const device uint4*>(w_s + ulong(bc) * 8ul * 32ul);
            for (uint h = 0u; h < 2u; ++h) {
                uint4 v = p[h * 32u + lane];
                uint uv[4] = {v.x, v.y, v.z, v.w};
                TF_UNROLL for (uint j = 0u; j < 4u; ++j) {
                    pw_v3_q8_4b(st, uv[j], T, q0, q1);
                    mma_16x16(h * 4u + j, q0, q1);
                }
            }
        } else if constexpr (BITS == 3u) {
            for (uint h = 0u; h < 2u; ++h) {
                const device uint* p = w_s + (ulong(bc) * 6ul + h * 3ul) * 32ul + lane;
                uint a0 = p[0], a1 = p[32], a2 = p[64];
                uint mv[4] = {a0 >> 8, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu, a2 & 0xFFFFFFu};
                TF_UNROLL for (uint j = 0u; j < 4u; ++j) {
                    pw_v3_q8_3b(st, mv[j], T, q0, q1);
                    mma_16x16(h * 4u + j, q0, q1);
                }
            }
        } else {
            for (uint h = 0u; h < 2u; ++h) {
                const device uint* p = w_s + (ulong(bc) * 10ul + h * 5ul) * 32ul + lane;
                uint w0 = p[0], w1 = p[32], w2 = p[64], w3 = p[96], w4 = p[128];
                uint hv[4] = {w0 >> 24, (w1 >> 16) & 255u, (w2 >> 8) & 255u, w3 & 255u};
                uint lv[4] = {(w0 << 8) | (w1 >> 24), (w1 << 16) | (w2 >> 16), (w2 << 24) | (w3 >> 8), w4};
                TF_UNROLL for (uint j = 0u; j < 4u; ++j) {
                    pw_v3_q8_5b(st, hv[j], lv[j], T, q0, q1);
                    mma_16x16(h * 4u + j, q0, q1);
                }
            }
        }

        // Accumulate scaled block result via sm_out scratch per-tile (exact float32).
        float fsv = float(sc0[bc]);
        simdgroup_float8x8 sc_mat = simdgroup_float8x8(fsv);
        TF_UNROLL for (int m = 0; m < 4; ++m) {
            TF_UNROLL for (int n = 0; n < 2; ++n) {
                // Elementwise scale via store/load on 8x8 tile in sm_out.
                threadgroup float* tile = sm_out + (m * 8u) * 128u + uint(sg) * 16u + n * 8u;
                simdgroup_store(blk_acc[m][n], tile, 128u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                // First bc writes directly; subsequent bc add into acc via sm_out.
                if (bc == P.bc0) {
                    uint r = lane >> 2, c = (lane & 3u) * 2u;
                    tile[r * 128u + c] *= fsv; tile[r * 128u + c + 1u] *= fsv;
                } else {
                    // Store previous acc, accumulate in float32, reload.
                    uint r = lane >> 2, c = (lane & 3u) * 2u;
                    float v0 = tile[r * 128u + c] * fsv, v1 = tile[r * 128u + c + 1u] * fsv;
                    simdgroup_store(acc[m][n], tile, 128u);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    tile[r * 128u + c] += v0; tile[r * 128u + c + 1u] += v1;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(acc[m][n], tile, 128u);
                (void)sc_mat;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    TF_UNROLL for (int m = 0; m < 4; ++m)
        TF_UNROLL for (int n = 0; n < 2; ++n)
            simdgroup_store(acc[m][n], sm_out + (m * 8u) * 128u + uint(sg) * 16u + n * 8u, 128u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sr0 = r0 + P.row_off;
    for (uint m = uint(sg); m < nb; m += 8u) {
        const threadgroup float* ys = sm_out + m * 128u;
        float4 v = float4(ys[lane], ys[lane + 32u], ys[lane + 64u], ys[lane + 96u]);
        pw_fwht128_simd4(v, lane);
        ulong base = ulong(m0 + m) * P.ldo + r0;
        OUT[base + lane +  0u] = v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u);
        OUT[base + lane + 32u] = v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u);
        OUT[base + lane + 64u] = v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u);
        OUT[base + lane + 96u] = v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u);
    }
}

template [[host_name("tfq3_prefill_tp_3b")]] [[kernel]] decltype(tfq3_prefill_tp_impl<3u>) tfq3_prefill_tp_impl<3u>;
template [[host_name("tfq3_prefill_tp_4b")]] [[kernel]] decltype(tfq3_prefill_tp_impl<4u>) tfq3_prefill_tp_impl<4u>;
template [[host_name("tfq3_prefill_tp_5b")]] [[kernel]] decltype(tfq3_prefill_tp_impl<5u>) tfq3_prefill_tp_impl<5u>;
