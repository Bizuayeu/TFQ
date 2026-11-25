#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

struct Tfq4MatParams {
    uint R;
    uint C;
};

inline float sgn_bit(uint seed, uint i) {
    uint h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline void fwht128_simd4(thread float4& v, ushort lane) {
    TF_UNROLL
    for (ushort d = 1; d <= 16; d <<= 1) {
        float4 o = simd_shuffle_xor(v, d);
        v = ((lane & d) == 0u) ? (v + o) : (o - v);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y;
    float a1 = v.z + v.w, b1 = v.z - v.w;
    v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

kernel void f32_row_fwht(
    const device float* W_in  [[buffer(0)]],
    device float* W_tmp       [[buffer(1)]],
    constant Tfq4MatParams& P [[buffer(2)]],
    uint2 gid                 [[threadgroup_position_in_grid]],
    ushort lane               [[thread_index_in_simdgroup]]
) {
    uint r = gid.y, bc = gid.x, c0 = bc * 128u;
    const device float* row = W_in + ulong(r) * P.C + c0;
    float4 v;
    v.x = row[lane +  0u] * sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = row[lane + 32u] * sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = row[lane + 64u] * sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = row[lane + 96u] * sgn_bit(0x12345678u, c0 + lane + 96u);
    fwht128_simd4(v, lane);
    device float* dst = W_tmp + ulong(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

kernel void col_fwht_and_ss(
    const device float* W_tmp   [[buffer(0)]],
    device float* W_rot         [[buffer(1)]],
    device atomic_float* blk_ss [[buffer(2)]],
    constant Tfq4MatParams& P   [[buffer(3)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint c = gid.x, br = gid.y, r0 = br * 128u;
    float4 v;
    v.x = W_tmp[ulong(r0 + lane +  0u) * P.C + c] * sgn_bit(0x9E3779B9u, r0 + lane +  0u);
    v.y = W_tmp[ulong(r0 + lane + 32u) * P.C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 32u);
    v.z = W_tmp[ulong(r0 + lane + 64u) * P.C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 64u);
    v.w = W_tmp[ulong(r0 + lane + 96u) * P.C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 96u);
    fwht128_simd4(v, lane);
    W_rot[ulong(r0 + lane +  0u) * P.C + c] = v.x;
    W_rot[ulong(r0 + lane + 32u) * P.C + c] = v.y;
    W_rot[ulong(r0 + lane + 64u) * P.C + c] = v.z;
    W_rot[ulong(r0 + lane + 96u) * P.C + c] = v.w;
    float ss = simd_sum(dot(v, v));
    if (lane == 0) atomic_fetch_add_explicit(&blk_ss[br * (P.C >> 7) + (c >> 7)], ss, memory_order_relaxed);
}

kernel void finalize_scales(
    const device float* blk_ss [[buffer(0)]],
    device half* scales_f16    [[buffer(1)]],
    constant uint& n_blocks    [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]]
) {
    if (tid >= n_blocks) return;
    scales_f16[tid] = half(sqrt(blk_ss[tid] * (1.0f / 16384.0f) + 1e-12f));
}

// Exhaustive 256-entry positive-orthant 8D E8P encoder: 1 thread per uint32_t (16 weights = 2 x 8D).
// Loads 4 KiB E8P LUT into threadgroup float[256][8] once per 256-thread TG.
kernel void tfq4_e8p_encode(
    const device float* W_rot   [[buffer(0)]],
    const device half* scales   [[buffer(1)]],
    const device half* e8p_lut  [[buffer(2)]], // 256 * 8 halfs
    device uint* out_words      [[buffer(3)]],
    device atomic_float* metrics [[buffer(4)]], // [0]=sig, [1]=err
    constant Tfq4MatParams& P   [[buffer(5)]],
    uint gtid                   [[thread_position_in_grid]],
    ushort tid                  [[thread_index_in_threadgroup]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    threadgroup float sm_lut[256][8];
    for (uint d = 0u; d < 8u; ++d) sm_lut[tid][d] = float(e8p_lut[uint(tid) * 8u + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint wpr = P.C >> 4; // C / 16
    ulong total_words = ulong(P.R) * wpr;
    float my_sig = 0.0f, my_err = 0.0f;

    if (ulong(gtid) < total_words) {
        uint r = gtid / wpr;
        uint wc = gtid - r * wpr;
        uint n_bc = P.C >> 7;
        float sc = float(scales[(r >> 7) * n_bc + (wc >> 3)]);
        float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        const device float* src = W_rot + ulong(r) * P.C + ulong(wc) * 16ul;

        uint codes[2];
        TF_UNROLL for (uint h = 0u; h < 2u; ++h) {
            float v[8], a[8];
            uint sm = 0u;
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float x = src[h * 8u + d];
                v[d] = x;
                if (x < 0.0f) sm |= (1u << d);
                a[d] = fabs(x) * inv_sc;
                my_sig += x * x;
            }
            uint best_k = 0u;
            float best_d = 1e30f;
            for (uint k = 0u; k < 256u; ++k) {
                float d0 = a[0] - sm_lut[k][0], d1 = a[1] - sm_lut[k][1];
                float d2 = a[2] - sm_lut[k][2], d3 = a[3] - sm_lut[k][3];
                float d4 = a[4] - sm_lut[k][4], d5 = a[5] - sm_lut[k][5];
                float d6 = a[6] - sm_lut[k][6], d7 = a[7] - sm_lut[k][7];
                float dist = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
                if (dist < best_d) { best_d = dist; best_k = k; }
            }
            codes[h] = (sm << 8) | best_k;
            TF_UNROLL for (uint d = 0u; d < 8u; ++d) {
                float e = fabs(v[d]) - sm_lut[best_k][d] * sc;
                my_err += e * e;
            }
        }
        out_words[gtid] = codes[0] | (codes[1] << 16);
    }

    float sg_sig = simd_sum(my_sig), sg_err = simd_sum(my_err);
    if (lane == 0) {
        atomic_fetch_add_explicit(&metrics[0], sg_sig, memory_order_relaxed);
        atomic_fetch_add_explicit(&metrics[1], sg_err, memory_order_relaxed);
    }
}
