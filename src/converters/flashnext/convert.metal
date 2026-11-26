#include <metal_stdlib>
using namespace metal;

#define TF_UNROLL _Pragma("clang loop unroll(full)")

inline short2 frag_home(ushort l) {
    return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) FlashNext GPU Converter Kernels
//   - q8_32_dequant_row_fwht : MLX 8-bit/32 affine dequant + row FWHT-128
//   - col_fwht_and_ss        : column FWHT-128 + per-128x128 RMS accumulation
//   - finalize_scales_and_pack_nax : NAX fragment stream layout
//   - prism_encode_v3_b{3,4,5} : M=256 O(1) 2D-grid GPU beam encoder
//   - prism_verify_v3_snr    : bit-exact decoder SNR verification
//   - requant_ple_q8_to_qb   : MLX 8-bit/32 -> B-bit/32 PLE shard requantizer
// ============================================================================

struct MatParams {
    uint R;
    uint C;
    uint b_bits;
    float shrink;
};

struct EncParams {
    uint R;
    uint C;
    uint b_bits;
    uint stream_off;
    uint n_streams;
};

struct PleParams {
    uint rows;
    uint cols;
    uint dst_bits;
    uint pad;
};

inline float sgn_bit(uint seed, uint i) {
    uint h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline float bf16_to_f32_m(ushort b) {
    return as_type<float>(uint(b) << 16);
}

inline ushort f32_to_bf16_m(float f) {
    uint u = as_type<uint>(f);
    uint r = u + 0x7FFFu + ((u >> 16) & 1u);
    return ushort(r >> 16);
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

kernel void q8_32_dequant_row_fwht(
    const device uchar* W_q8    [[buffer(0)]],
    const device ushort* S_bf16 [[buffer(1)]],
    const device ushort* B_bf16 [[buffer(2)]],
    device float* W_tmp         [[buffer(3)]],
    constant MatParams& P       [[buffer(4)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint r = gid.y, bc = gid.x, c0 = bc * 128u, g32 = bc * 4u, n_g32 = P.C >> 5;
    ulong sb_base = ulong(r) * n_g32 + g32;
    float s0 = bf16_to_f32_m(S_bf16[sb_base + 0u]), b0 = bf16_to_f32_m(B_bf16[sb_base + 0u]);
    float s1 = bf16_to_f32_m(S_bf16[sb_base + 1u]), b1 = bf16_to_f32_m(B_bf16[sb_base + 1u]);
    float s2 = bf16_to_f32_m(S_bf16[sb_base + 2u]), b2 = bf16_to_f32_m(B_bf16[sb_base + 2u]);
    float s3 = bf16_to_f32_m(S_bf16[sb_base + 3u]), b3 = bf16_to_f32_m(B_bf16[sb_base + 3u]);
    const device uchar* row = W_q8 + ulong(r) * P.C + c0;
    float4 v;
    v.x = fma(float(row[lane +  0u]), s0, b0) * sgn_bit(0x12345678u, c0 + lane +  0u);
    v.y = fma(float(row[lane + 32u]), s1, b1) * sgn_bit(0x12345678u, c0 + lane + 32u);
    v.z = fma(float(row[lane + 64u]), s2, b2) * sgn_bit(0x12345678u, c0 + lane + 64u);
    v.w = fma(float(row[lane + 96u]), s3, b3) * sgn_bit(0x12345678u, c0 + lane + 96u);
    fwht128_simd4(v, lane);
    device float* dst = W_tmp + ulong(r) * P.C + c0;
    dst[lane +  0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

static inline float det_simd_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

kernel void col_fwht_and_ss(
    const device float* W_tmp   [[buffer(0)]],
    device float* W_rot         [[buffer(1)]],
    device float* col_ss        [[buffer(2)]],
    constant MatParams& P       [[buffer(3)]],
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
    float ss = det_simd_sum(dot(v, v));
    uint bi = br * (P.C >> 7) + (c >> 7);
    if (lane == 0) col_ss[ulong(bi) * 128ul + (c & 127u)] = ss;
}

kernel void finalize_scales_and_pack_nax(
    const device float* W_rot   [[buffer(0)]],
    const device float* col_ss  [[buffer(1)]],
    device half* scales_f16     [[buffer(2)]],
    device float* W_streams     [[buffer(3)]],
    constant MatParams& P       [[buffer(4)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
    uint strip = gid.x, bc = gid.y, br = strip >> 3, n_bc = P.C >> 7, bi = br * n_bc + bc;
    const device float* cs = col_ss + ulong(bi) * 128ul;
    float blk_sum = det_simd_sum((cs[lane + 0u] + cs[lane + 32u]) + (cs[lane + 64u] + cs[lane + 96u]));
    float rms = sqrt(blk_sum * (1.0f / 16384.0f) + 1e-12f) * P.shrink;
    half h_rms = half(rms);
    if ((strip & 7u) == 0u && lane == 0) scales_f16[bi] = h_rms;
    float inv_sc = 1.0f / float(h_rms);
    short2 home = frag_home(lane);
    uint K_len = P.C >> 1;
    uint stream_id = strip * 32u + uint(lane);
    device float* dst = W_streams + ulong(stream_id) * K_len + ulong(bc) * 64u;
    uint r_base = strip * 16u, c_base = bc * 128u;
    TF_UNROLL
    for (uint kt = 0; kt < 8u; ++kt) {
        uint cc0 = c_base + kt * 16u + uint(home.x);
        uint r0 = r_base + uint(home.y), r1 = r0 + 8u;
        const device float* p0 = W_rot + ulong(r0) * P.C + cc0;
        const device float* p1 = W_rot + ulong(r1) * P.C + cc0;
        dst[kt * 8u + 0u] = p0[0] * inv_sc; dst[kt * 8u + 1u] = p0[1] * inv_sc;
        dst[kt * 8u + 2u] = p0[2] * inv_sc; dst[kt * 8u + 3u] = p0[3] * inv_sc;
        dst[kt * 8u + 4u] = p1[0] * inv_sc; dst[kt * 8u + 5u] = p1[1] * inv_sc;
        dst[kt * 8u + 6u] = p1[2] * inv_sc; dst[kt * 8u + 7u] = p1[3] * inv_sc;
    }
}

template <uint B, uint LUT_N, uint C_SHIFT, uint H_SHIFT, uint G, uint KC, uint K_KEEP>
kernel void prism_encode_v3(
    const device float* W_streams   [[buffer(0)]],
    constant uint* lut_u32          [[buffer(1)]],
    constant half4* c64             [[buffer(2)]],
    constant ushort* grid_cands     [[buffer(3)]],
    device uint* scratch_hist       [[buffer(4)]],
    device uint* out_packed         [[buffer(5)]],
    constant EncParams& P           [[buffer(6)]],
    uint gid                        [[threadgroup_position_in_grid]],
    ushort tid                      [[thread_index_in_threadgroup]],
    ushort sg                       [[simdgroup_index_in_threadgroup]],
    ushort lane                     [[thread_index_in_simdgroup]]
) {
    if (gid >= P.n_streams) return;
    const uint stream_id = P.stream_off + gid;
    const uint strip_id  = stream_id >> 5;
    const uint my_lane   = stream_id & 31u;
    const uint K_len     = P.C >> 1;
    const uint n_pairs   = K_len >> 1;
    constexpr uint BEAM     = 256u;
    constexpr uint N_SYM    = 1u << (2u * B);
    constexpr uint SYM_MASK = N_SYM - 1u;
    constexpr uint C_MASK   = (1u << C_SHIFT) - 1u;
    constexpr float RMAX    = 3.2f;
    constexpr float INV_STEP = float(G) / (2.0f * RMAX);

    const device float* y_ptr = W_streams + ulong(stream_id) * K_len;
    device uint* hist_ptr = scratch_hist + ulong(gid) * ulong(n_pairs) * 256ul;

    threadgroup float cur_cost[256], nxt_cost[256];
    threadgroup uint  cur_state[256], nxt_state[256];
    threadgroup uint  cur_hprev[256], nxt_hprev[256];
    threadgroup float sg_min[4], sg_sum[4], cmin_tg, inv_step_tg;
    threadgroup atomic_uint bins[64];
    threadgroup uint sg_scan[4];
    threadgroup uint cut_bin_tg, lt_base_tg, eq_quota_tg;

    if (tid == 0) {
        cur_cost[0]  = 0.0f;
        cur_state[0] = 0x13579BDFu ^ (my_lane * 0x9E3779B1u);
        cur_hprev[0] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_n = 1u;

    for (uint pair = 0; pair < n_pairs; ++pair) {
        float x0 = y_ptr[pair * 2u];
        float y0 = y_ptr[pair * 2u + 1u];
        bool is_odd = (pair & 1u) != 0u;

        if (cur_n * K_KEEP <= BEAM) {
            if (tid < cur_n) {
                float bc = cur_cost[tid];
                uint st  = cur_state[tid];
                uint h   = st * 0x9E3779B1u;
                uint h_c = is_odd ? cur_hprev[tid] : h;
                half4 cs = c64[(h_c >> 16) & 63u];
                float msx = float(cs.x), msy = float(cs.y), ax = float(cs.z), ay = float(cs.w);
                uint h_lut   = h >> H_SHIFT;
                uint c_idx   = h_lut & C_MASK;
                uint sym_xor = (h_lut >> C_SHIFT) & SYM_MASK;
                float tx = (x0 - ax) / msx, ty = (y0 - ay) / msy;
                int gx = clamp(int((tx + RMAX) * INV_STEP), 0, int(G) - 1);
                int gy = clamp(int((ty + RMAX) * INV_STEP), 0, int(G) - 1);
                ulong gbase = ((ulong(c_idx) * G + uint(gy)) * G + uint(gx)) * KC;

                float bd[K_KEEP]; ushort bs[K_KEEP];
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) { bd[k] = 1e30f; bs[k] = 0; }
                TF_UNROLL for (uint k = 0; k < KC; ++k) {
                    uint i_sub = uint(grid_cands[gbase + k]);
                    uint idx = (i_sub << C_SHIFT) | c_idx;
                    half2 q = fma(as_type<half2>(lut_u32[idx]), cs.xy, cs.zw);
                    float dx = x0 - float(q.x), dy = y0 - float(q.y), d = dx * dx + dy * dy;
                    if (d < bd[K_KEEP - 1u]) {
                        bd[K_KEEP - 1u] = d; bs[K_KEEP - 1u] = ushort(i_sub ^ sym_xor);
                        TF_UNROLL for (uint j = K_KEEP - 1u; j > 0u; --j) {
                            if (bd[j] < bd[j - 1u]) {
                                float td = bd[j]; bd[j] = bd[j - 1u]; bd[j - 1u] = td;
                                ushort ts = bs[j]; bs[j] = bs[j - 1u]; bs[j - 1u] = ts;
                            }
                        }
                    }
                }
                uint st_sh = st << (2u * B);
                uint next_hp = is_odd ? 0u : h;
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) {
                    uint dst = tid * K_KEEP + k;
                    uint sym = uint(bs[k]);
                    nxt_cost[dst]  = bc + bd[k];
                    nxt_state[dst] = st_sh | sym;
                    nxt_hprev[dst] = next_hp;
                    hist_ptr[ulong(pair) * 256ul + dst] = uint(tid) | (sym << 8);
                }
            }
            cur_n *= K_KEEP;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            TF_UNROLL for (uint s = 0; s < 2u; ++s) {
                uint i = uint(tid) + s * 128u;
                if (i < cur_n) {
                    cur_cost[i] = nxt_cost[i]; cur_state[i] = nxt_state[i]; cur_hprev[i] = nxt_hprev[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        float  r_bc[2];
        uint   r_st_sh[2];
        uint   r_hp[2];
        float  r_d[2][K_KEEP];
        ushort r_sym[2][K_KEEP];
        float my_min = 1e30f;
        float my_sum = 0.0f;

        TF_UNROLL
        for (uint s = 0; s < 2u; ++s) {
            uint pi = uint(tid) + s * 128u;
            if (pi < cur_n) {
                float bc = cur_cost[pi];
                uint st  = cur_state[pi];
                uint h   = st * 0x9E3779B1u;
                uint h_c = is_odd ? cur_hprev[pi] : h;
                half4 cs = c64[(h_c >> 16) & 63u];
                float msx = float(cs.x), msy = float(cs.y), ax = float(cs.z), ay = float(cs.w);
                uint h_lut   = h >> H_SHIFT;
                uint c_idx   = h_lut & C_MASK;
                uint sym_xor = (h_lut >> C_SHIFT) & SYM_MASK;
                float tx = (x0 - ax) / msx, ty = (y0 - ay) / msy;
                int gx = clamp(int((tx + RMAX) * INV_STEP), 0, int(G) - 1);
                int gy = clamp(int((ty + RMAX) * INV_STEP), 0, int(G) - 1);
                ulong gbase = ((ulong(c_idx) * G + uint(gy)) * G + uint(gx)) * KC;

                float bd[K_KEEP]; ushort bs[K_KEEP];
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) { bd[k] = 1e30f; bs[k] = 0; }
                TF_UNROLL for (uint k = 0; k < KC; ++k) {
                    uint i_sub = uint(grid_cands[gbase + k]);
                    uint idx = (i_sub << C_SHIFT) | c_idx;
                    half2 q = fma(as_type<half2>(lut_u32[idx]), cs.xy, cs.zw);
                    float dx = x0 - float(q.x), dy = y0 - float(q.y), d = dx * dx + dy * dy;
                    if (d < bd[K_KEEP - 1u]) {
                        bd[K_KEEP - 1u] = d; bs[K_KEEP - 1u] = ushort(i_sub ^ sym_xor);
                        TF_UNROLL for (uint j = K_KEEP - 1u; j > 0u; --j) {
                            if (bd[j] < bd[j - 1u]) {
                                float td = bd[j]; bd[j] = bd[j - 1u]; bd[j - 1u] = td;
                                ushort ts = bs[j]; bs[j] = bs[j - 1u]; bs[j - 1u] = ts;
                            }
                        }
                    }
                }
                r_bc[s] = bc;
                r_st_sh[s] = st << (2u * B);
                r_hp[s] = is_odd ? 0u : h;
                my_min = min(my_min, bc + bd[0]);
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) {
                    r_d[s][k] = bd[k]; r_sym[s][k] = bs[k]; my_sum += bc + bd[k];
                }
            }
        }

        float smin = simd_min(my_min);
        float ssum = det_simd_sum(my_sum);
        if (lane == 0) { sg_min[sg] = smin; sg_sum[sg] = ssum; }
        if (tid < 64u) atomic_store_explicit(&bins[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float cmin = min(min(sg_min[0], sg_min[1]), min(sg_min[2], sg_min[3]));
            float cmean = (sg_sum[0] + sg_sum[1] + sg_sum[2] + sg_sum[3]) / float(cur_n * K_KEEP);
            float span = max(1e-4f, (cmean - cmin) * 1.15f);
            cmin_tg = cmin; inv_step_tg = 63.0f / span;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float cmin = cmin_tg, inv_step = inv_step_tg;

        uchar r_bi[2][K_KEEP];
        TF_UNROLL for (uint s = 0; s < 2u; ++s) {
            uint pi = uint(tid) + s * 128u;
            if (pi < cur_n) {
                float bc = r_bc[s] - cmin;
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) {
                    uint bi = min(63u, uint(max(0.0f, (bc + r_d[s][k]) * inv_step)));
                    r_bi[s][k] = uchar(bi);
                    atomic_fetch_add_explicit(&bins[bi], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint acc = 0u, cb = 63u, lb = 0u, eq = BEAM;
            bool found = false;
            for (uint i = 0; i < 64u; ++i) {
                uint cnt = atomic_load_explicit(&bins[i], memory_order_relaxed);
                if (!found && acc + cnt >= BEAM) { cb = i; lb = acc; eq = BEAM - acc; found = true; }
                acc += cnt;
            }
            cut_bin_tg = cb; lt_base_tg = lb; eq_quota_tg = eq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint cut_bin = cut_bin_tg, lt_base = lt_base_tg, eq_quota = eq_quota_tg;

        uint my_lt = 0u, my_eq = 0u;
        TF_UNROLL for (uint s = 0; s < 2u; ++s) {
            uint pi = uint(tid) + s * 128u;
            if (pi < cur_n) {
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) {
                    uint bi = uint(r_bi[s][k]);
                    if (bi < cut_bin) my_lt++;
                    else if (bi == cut_bin) my_eq++;
                }
            }
        }
        uint packed = (my_eq << 16) | my_lt;
        uint incl = packed;
        TF_UNROLL for (ushort d = 1; d < 32; d <<= 1) {
            uint v = simd_shuffle_up(incl, d);
            if (lane >= d) incl += v;
        }
        uint excl = incl - packed;
        if (lane == 31) sg_scan[sg] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w_base = 0u;
        if (sg > 0) w_base += sg_scan[0];
        if (sg > 1) w_base += sg_scan[1];
        if (sg > 2) w_base += sg_scan[2];
        uint base_packed = excl + w_base;
        uint pos_lt = base_packed & 0xFFFFu;
        uint pos_eq = base_packed >> 16;

        TF_UNROLL for (uint s = 0; s < 2u; ++s) {
            uint pi = uint(tid) + s * 128u;
            if (pi < cur_n) {
                float bc_raw = r_bc[s];
                uint st_sh = r_st_sh[s], next_hp = r_hp[s];
                TF_UNROLL for (uint k = 0; k < K_KEEP; ++k) {
                    uint bi = uint(r_bi[s][k]);
                    if (bi < cut_bin) {
                        uint pos = pos_lt++;
                        if (pos < lt_base) {
                            uint sym = uint(r_sym[s][k]);
                            nxt_cost[pos] = bc_raw + r_d[s][k]; nxt_state[pos] = st_sh | sym; nxt_hprev[pos] = next_hp;
                            hist_ptr[ulong(pair) * 256ul + pos] = pi | (sym << 8);
                        }
                    } else if (bi == cut_bin) {
                        uint ep = pos_eq++;
                        if (ep < eq_quota) {
                            uint pos = lt_base + ep;
                            uint sym = uint(r_sym[s][k]);
                            nxt_cost[pos] = bc_raw + r_d[s][k]; nxt_state[pos] = st_sh | sym; nxt_hprev[pos] = next_hp;
                            hist_ptr[ulong(pair) * 256ul + pos] = pi | (sym << 8);
                        }
                    }
                }
            }
        }
        cur_n = BEAM;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TF_UNROLL for (uint s = 0; s < 2u; ++s) {
            uint i = uint(tid) + s * 128u;
            cur_cost[i] = nxt_cost[i] - cmin; cur_state[i] = nxt_state[i]; cur_hprev[i] = nxt_hprev[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint best_i = 0u; float best_c = 1e30f;
        for (uint i = 0; i < BEAM; ++i) if (cur_cost[i] < best_c) { best_c = cur_cost[i]; best_i = i; }
        uint idx = best_i;
        for (int pair = int(n_pairs) - 1; pair >= 0; --pair) {
            uint rec = hist_ptr[ulong(pair) * 256ul + idx];
            hist_ptr[ulong(pair) * 256ul] = rec >> 8;
            idx = rec & 255u;
        }
        uint n_bc = P.C >> 7;
        constexpr uint wpb = B * 2u;
        ulong strip_word_base = ulong(strip_id) * n_bc * wpb * 32ul;
        for (uint bc = 0; bc < n_bc; ++bc) {
            uint words[10];
            TF_UNROLL for (uint w = 0; w < wpb; ++w) words[w] = 0u;
            uint bit_pos = 0u;
            for (uint p = 0; p < 32u; ++p) {
                uint sym = hist_ptr[ulong(bc * 32u + p) * 256ul];
                for (int k = int(2u * B) - 1; k >= 0; --k) {
                    if ((sym >> k) & 1u) words[bit_pos >> 5] |= (1u << (31u - (bit_pos & 31u)));
                    bit_pos++;
                }
            }
            device uint* base = out_packed + strip_word_base + ulong(bc) * wpb * 32ul;
            if constexpr (B == 4u) {
                base[my_lane * 4u + 0u] = words[0]; base[my_lane * 4u + 1u] = words[1];
                base[my_lane * 4u + 2u] = words[2]; base[my_lane * 4u + 3u] = words[3];
                base[128u + my_lane * 4u + 0u] = words[4]; base[128u + my_lane * 4u + 1u] = words[5];
                base[128u + my_lane * 4u + 2u] = words[6]; base[128u + my_lane * 4u + 3u] = words[7];
            } else {
                TF_UNROLL for (uint w = 0; w < wpb; ++w) base[w * 32u + my_lane] = words[w];
            }
        }
    }
}

template [[host_name("prism_encode_v3_b3")]] [[kernel]] decltype(prism_encode_v3<3u,  256u, 2u, 24u, 16u,  8u, 5u>) prism_encode_v3<3u,  256u, 2u, 24u, 16u,  8u, 5u>;
template [[host_name("prism_encode_v3_b4")]] [[kernel]] decltype(prism_encode_v3<4u, 1024u, 2u, 22u, 16u, 10u, 5u>) prism_encode_v3<4u, 1024u, 2u, 22u, 16u, 10u, 5u>;
template [[host_name("prism_encode_v3_b5")]] [[kernel]] decltype(prism_encode_v3<5u, 1024u, 0u, 22u, 32u, 12u, 6u>) prism_encode_v3<5u, 1024u, 0u, 22u, 32u, 12u, 6u>;

kernel void prism_verify_v3_snr(
    const device uint* W_packed  [[buffer(0)]],
    const device half* scales    [[buffer(1)]],
    constant uint* lut_u32       [[buffer(2)]],
    constant half4* c64          [[buffer(3)]],
    const device float* W_rot    [[buffer(4)]],
    device float* out_se        [[buffer(5)]],
    constant MatParams& P        [[buffer(6)]],
    uint strip                   [[threadgroup_position_in_grid]],
    ushort lane                  [[thread_index_in_simdgroup]]
) {
    short2 home = frag_home(lane);
    uint n_bc = P.C >> 7, b = P.b_bits, wpb = b * 2u;
    const device uint* w_strip = W_packed + ulong(strip) * n_bc * wpb * 32ul;
    const device half* sc_row = scales + ulong(strip >> 3) * n_bc;
    uint st = 0x13579BDFu ^ (uint(lane) * 0x9E3779B1u);
    float sig = 0.0f, err = 0.0f;
    for (uint bc = 0; bc < n_bc; ++bc) {
        float sc = float(sc_row[bc]);
        const device uint* base = w_strip + ulong(bc) * wpb * 32ul;
        uint words[10];
        if (b == 4u) {
            for (uint k = 0; k < 4u; ++k) words[k] = base[lane * 4u + k];
            for (uint k = 0; k < 4u; ++k) words[k + 4u] = base[128u + lane * 4u + k];
        } else {
            for (uint w = 0; w < wpb; ++w) words[w] = base[w * 32u + lane];
        }
        uint bit_pos = 0u;
        for (uint kt = 0; kt < 8u; ++kt) {
            uint syms[4];
            for (uint p = 0; p < 4u; ++p) {
                uint sym = 0u;
                for (uint k = 0; k < 2u * b; ++k) {
                    sym = (sym << 1) | ((words[bit_pos >> 5] >> (31u - (bit_pos & 31u))) & 1u);
                    bit_pos++;
                }
                syms[p] = sym;
            }
            half4 q0, q1;
            if (b == 3u) {
                uint h0 = st * 0x9E3779B1u; uint st1 = (st << 6) | syms[0];
                uint h1 = st1 * 0x9E3779B1u; uint st2 = (st1 << 6) | syms[1];
                uint h2 = st2 * 0x9E3779B1u; uint st3 = (st2 << 6) | syms[2];
                uint h3 = st3 * 0x9E3779B1u; st = (st3 << 6) | syms[3];
                half4 c0 = c64[(h0 >> 16) & 63u], c2 = c64[(h2 >> 16) & 63u];
                q0 = fma(half4(as_type<half2>(lut_u32[((syms[0] << 2) ^ (h0 >> 24)) & 255u]),
                               as_type<half2>(lut_u32[((syms[1] << 2) ^ (h1 >> 24)) & 255u])), c0.xyxy, c0.zwzw);
                q1 = fma(half4(as_type<half2>(lut_u32[((syms[2] << 2) ^ (h2 >> 24)) & 255u]),
                               as_type<half2>(lut_u32[((syms[3] << 2) ^ (h3 >> 24)) & 255u])), c2.xyxy, c2.zwzw);
            } else if (b == 4u) {
                uint h0 = st * 0x9E3779B1u; uint st1 = (st << 8) | syms[0];
                uint h1 = st1 * 0x9E3779B1u; uint st2 = (st1 << 8) | syms[1];
                uint h2 = st2 * 0x9E3779B1u; uint st3 = (st2 << 8) | syms[2];
                uint h3 = st3 * 0x9E3779B1u; st = (st3 << 8) | syms[3];
                half4 c0 = c64[(h0 >> 16) & 63u], c2 = c64[(h2 >> 16) & 63u];
                q0 = fma(half4(as_type<half2>(lut_u32[((syms[0] << 2) ^ (h0 >> 22)) & 1023u]),
                               as_type<half2>(lut_u32[((syms[1] << 2) ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
                q1 = fma(half4(as_type<half2>(lut_u32[((syms[2] << 2) ^ (h2 >> 22)) & 1023u]),
                               as_type<half2>(lut_u32[((syms[3] << 2) ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
            } else {
                uint h0 = st * 0x9E3779B1u; uint st1 = (st << 10) | syms[0];
                uint h1 = st1 * 0x9E3779B1u; uint st2 = (st1 << 10) | syms[1];
                uint h2 = st2 * 0x9E3779B1u; uint st3 = (st2 << 10) | syms[2];
                uint h3 = st3 * 0x9E3779B1u; st = (st3 << 10) | syms[3];
                half4 c0 = c64[(h0 >> 16) & 63u], c2 = c64[(h2 >> 16) & 63u];
                q0 = fma(half4(as_type<half2>(lut_u32[(syms[0] ^ (h0 >> 22)) & 1023u]),
                               as_type<half2>(lut_u32[(syms[1] ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
                q1 = fma(half4(as_type<half2>(lut_u32[(syms[2] ^ (h2 >> 22)) & 1023u]),
                               as_type<half2>(lut_u32[(syms[3] ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
            }
            float4 f0 = float4(q0) * sc, f1 = float4(q1) * sc;
            uint cc0 = bc * 128u + kt * 16u + uint(home.x);
            uint r0 = strip * 16u + uint(home.y), r1 = r0 + 8u;
            const device float* p0 = W_rot + ulong(r0) * P.C + cc0;
            const device float* p1 = W_rot + ulong(r1) * P.C + cc0;
            float4 ref0 = float4(p0[0], p0[1], p0[2], p0[3]), ref1 = float4(p1[0], p1[1], p1[2], p1[3]);
            float4 d0 = ref0 - f0, d1 = ref1 - f1;
            sig += dot(ref0, ref0) + dot(ref1, ref1);
            err += dot(d0, d0) + dot(d1, d1);
        }
    }
    sig = det_simd_sum(sig); err = det_simd_sum(err);
    if (lane == 0) {
        out_se[strip * 2u + 0u] = sig;
        out_se[strip * 2u + 1u] = err;
    }
}

// Requantize MLX 8-bit/32 affine [rows, cols] (cols = 160) -> B-bit/32 (B in {2, 3, 4})
// One thread per group of 32 values!
kernel void requant_ple_q8_to_qb(
    const device uchar* src_w   [[buffer(0)]],
    const device ushort* src_s  [[buffer(1)]],
    const device ushort* src_b  [[buffer(2)]],
    device uint* dst_w          [[buffer(3)]],
    device ushort* dst_s        [[buffer(4)]],
    device ushort* dst_b        [[buffer(5)]],
    constant PleParams& P       [[buffer(6)]],
    uint gid                    [[thread_position_in_grid]]
) {
    uint ng = P.cols >> 5;
    uint total_g = P.rows * ng;
    if (gid >= total_g) return;
    uint r = gid / ng;
    uint g = gid - r * ng;
    float s_in = bf16_to_f32_m(src_s[gid]);
    float b_in = bf16_to_f32_m(src_b[gid]);
    const device uchar* wp = src_w + ulong(gid) * 32ul;
    float vals[32];
    float vmin = 1e30f, vmax = -1e30f;
    for (uint i = 0; i < 32u; ++i) {
        float v = fma(float(wp[i]), s_in, b_in);
        vals[i] = v;
        vmin = min(vmin, v);
        vmax = max(vmax, v);
    }
    uint B = P.dst_bits;
    float qmax = float((1u << B) - 1u);
    ushort hb = f32_to_bf16_m(vmin);
    float bf = bf16_to_f32_m(hb);
    float sc = (vmax > bf) ? ((vmax - bf) / qmax) : 1.0f;
    ushort hs = f32_to_bf16_m(sc);
    float sf = bf16_to_f32_m(hs);
    float inv_s = (sf > 0.0f) ? (1.0f / sf) : 0.0f;
    dst_s[gid] = hs;
    dst_b[gid] = hb;

    // Pack 32 * B bits into B uint32 words (group_size=32 -> 32*B/32 = B u32 words per group!)
    uint words[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0; i < 32u; ++i) {
        uint q = uint(clamp(rint((vals[i] - bf) * inv_s), 0.0f, qmax));
        uint bit = i * B;
        uint widx = bit >> 5;
        uint sh = bit & 31u;
        words[widx] |= (q << sh);
        if (sh + B > 32u) words[widx + 1u] |= (q >> (32u - sh));
    }
    ulong dst_row_words = ulong(r) * ulong(ng * B) + ulong(g * B);
    for (uint w = 0; w < B; ++w) dst_w[dst_row_words + w] = words[w];
}
