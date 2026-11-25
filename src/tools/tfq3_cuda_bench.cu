// Standalone CUDA GPU parity test (dense + MoE + bit-exact speculative verify) and throughput benchmark for TFQ3.
#include "../kernels/cuda/tfq3_cuda.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static float h_bf16_to_f32(uint16_t b) { uint32_t w = (uint32_t)b << 16; float f; memcpy(&f, &w, 4); return f; }
static uint16_t h_f32_to_bf16(float f) { uint32_t w; memcpy(&w, &f, 4); return (uint16_t)((w + 0x7FFFu + ((w >> 16) & 1u)) >> 16); }
static float h_sgn(uint32_t seed, uint32_t i) { uint32_t h = (seed ^ i) * 2654435761u; return ((h >> 31) & 1u) ? -1.0f : 1.0f; }

static void h_fwht128(double* v) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += len * 2) {
            for (int j = 0; j < len; ++j) {
                double a = v[i + j], b = v[i + j + len];
                v[i + j] = a + b; v[i + j + len] = a - b;
            }
        }
    }
    double inv = 1.0 / std::sqrt(128.0);
    for (int i = 0; i < 128; ++i) v[i] *= inv;
}

static void init_synth_tables(tfq3::PwV3Tables& tbl) {
    memset(&tbl, 0, sizeof(tbl));
    tbl.magic = 0x33565750u;
    tbl.version = 3u;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.8f);
    auto pack2 = [](float a, float b) {
        __half2 h = __floats2half2_rn(a, b);
        uint32_t u; memcpy(&u, &h, 4); return u;
    };
    for (int i = 0; i < 256; ++i) tbl.lut3[i] = pack2(nd(rng), nd(rng));
    for (int i = 0; i < 1024; ++i) tbl.lut4[i] = pack2(nd(rng), nd(rng));
    for (int i = 0; i < 1024; ++i) tbl.lut5[i] = pack2(nd(rng), nd(rng));
    for (int i = 0; i < 64; ++i) {
        tbl.c64_3[i] = make_uint2(pack2(1.0f + 0.05f * nd(rng), 1.0f + 0.05f * nd(rng)), pack2(0.05f * nd(rng), 0.05f * nd(rng)));
        tbl.c64_4[i] = make_uint2(pack2(1.0f + 0.05f * nd(rng), 1.0f + 0.05f * nd(rng)), pack2(0.05f * nd(rng), 0.05f * nd(rng)));
        tbl.c64_5[i] = make_uint2(pack2(1.0f + 0.05f * nd(rng), 1.0f + 0.05f * nd(rng)), pack2(0.05f * nd(rng), 0.05f * nd(rng)));
    }
}

static void h_decode_wrot(
    const uint32_t* W, const __half* scales, const tfq3::PwV3Tables& T,
    uint32_t bits, uint32_t R, uint32_t C, std::vector<double>& W_rot
) {
    W_rot.assign((size_t)R * C, 0.0);
    uint32_t n_bc = C >> 7, wpb = bits * 2u;
    auto u2f = [](uint32_t u, float& x, float& y) { __half2 h; memcpy(&h, &u, 4); float2 f = __half22float2(h); x = f.x; y = f.y; };
    auto apply_aff = [&](uint32_t lut_u, uint2 c, float& x, float& y) {
        float lx, ly, sx, sy, tx, ty;
        u2f(lut_u, lx, ly); u2f(c.x, sx, sy); u2f(c.y, tx, ty);
        x = lx * sx + tx; y = ly * sy + ty;
    };
    for (uint32_t strip = 0; strip < R / 16u; ++strip) {
        uint32_t rb = strip >> 3;
        for (uint32_t lane = 0; lane < 32u; ++lane) {
            uint32_t hx = (lane & 8u) + ((lane & 1u) << 2);
            uint32_t hy = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
            uint32_t st = 0x13579BDFu ^ (lane * 0x9E3779B1u);
            const uint32_t* w_base = W + (size_t)strip * n_bc * wpb * 32ull;
            for (uint32_t bc = 0; bc < n_bc; ++bc) {
                double sc = (double)__half2float(scales[(size_t)rb * n_bc + bc]);
                auto put8 = [&](uint32_t kt, const float q[8]) {
                    uint32_t c0 = bc * 128u + kt * 16u + hx;
                    uint32_t r0 = strip * 16u + hy, r1 = r0 + 8u;
                    for (int j = 0; j < 4; ++j) { W_rot[(size_t)r0 * C + c0 + j] = (double)q[j] * sc; W_rot[(size_t)r1 * C + c0 + j] = (double)q[j + 4] * sc; }
                };
                if (bits == 4u) {
                    const uint4* w4 = reinterpret_cast<const uint4*>(w_base);
                    for (uint32_t h = 0; h < 2u; ++h) {
                        uint4 v = w4[(bc * 2u + h) * 32u + lane];
                        uint32_t wcs[4] = {v.x, v.y, v.z, v.w};
                        for (uint32_t k = 0; k < 4u; ++k) {
                            uint32_t wc = wcs[k], s0 = st; st = wc;
                            uint32_t h0 = s0 * 0x9E3779B1u, h1 = ((s0 << 8) | (wc >> 24)) * 0x9E3779B1u;
                            uint32_t h2 = ((s0 << 16) | (wc >> 16)) * 0x9E3779B1u, h3 = ((s0 << 24) | (wc >> 8)) * 0x9E3779B1u;
                            uint2 c0 = T.c64_4[(h0 >> 16) & 63u], c2 = T.c64_4[(h2 >> 16) & 63u];
                            float q[8];
                            apply_aff(T.lut4[(((wc >> 24) << 2) ^ (h0 >> 22)) & 1023u], c0, q[0], q[1]);
                            apply_aff(T.lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u], c0, q[2], q[3]);
                            apply_aff(T.lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u], c2, q[4], q[5]);
                            apply_aff(T.lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u], c2, q[6], q[7]);
                            put8(h * 4u + k, q);
                        }
                    }
                } else if (bits == 3u) {
                    for (uint32_t h = 0; h < 2u; ++h) {
                        const uint32_t* blk = w_base + ((size_t)bc * 6u + h * 3u) * 32u + lane;
                        uint32_t a0 = blk[0], a1 = blk[32], a2 = blk[64];
                        uint32_t ms[4] = {a0 >> 8, ((a0 << 16) | (a1 >> 16)) & 0xFFFFFFu, ((a1 << 8) | (a2 >> 24)) & 0xFFFFFFu, a2 & 0xFFFFFFu};
                        for (uint32_t k = 0; k < 4u; ++k) {
                            uint32_t m24 = ms[k], s0 = st; st = (s0 << 24) | m24;
                            uint32_t h0 = s0 * 0x9E3779B1u, h1 = ((s0 << 6) | (m24 >> 18)) * 0x9E3779B1u;
                            uint32_t h2 = ((s0 << 12) | (m24 >> 12)) * 0x9E3779B1u, h3 = ((s0 << 18) | (m24 >> 6)) * 0x9E3779B1u;
                            uint2 c0 = T.c64_3[(h0 >> 16) & 63u], c2 = T.c64_3[(h2 >> 16) & 63u];
                            float q[8];
                            apply_aff(T.lut3[(((m24 >> 16) & 0xFCu) ^ (h0 >> 24)) & 255u], c0, q[0], q[1]);
                            apply_aff(T.lut3[(((m24 >> 10) & 0xFCu) ^ (h1 >> 24)) & 255u], c0, q[2], q[3]);
                            apply_aff(T.lut3[(((m24 >>  4) & 0xFCu) ^ (h2 >> 24)) & 255u], c2, q[4], q[5]);
                            apply_aff(T.lut3[(((m24 <<  2) & 0xFCu) ^ (h3 >> 24)) & 255u], c2, q[6], q[7]);
                            put8(h * 4u + k, q);
                        }
                    }
                } else {
                    for (uint32_t h = 0; h < 2u; ++h) {
                        const uint32_t* blk = w_base + ((size_t)bc * 10u + h * 5u) * 32u + lane;
                        uint32_t w0 = blk[0], w1 = blk[32], w2 = blk[64], w3 = blk[96], w4 = blk[128];
                        uint32_t his[4] = {w0 >> 24, (w1 >> 16) & 255u, (w2 >> 8) & 255u, w3 & 255u};
                        uint32_t los[4] = {(w0 << 8) | (w1 >> 24), (w1 << 16) | (w2 >> 16), (w2 << 24) | (w3 >> 8), w4};
                        for (uint32_t k = 0; k < 4u; ++k) {
                            uint32_t hi8 = his[k], lo32 = los[k], s0 = st; st = lo32;
                            uint32_t b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023u, b1 = (lo32 >> 20) & 1023u, b2 = (lo32 >> 10) & 1023u, b3 = lo32 & 1023u;
                            uint32_t h0 = s0 * 0x9E3779B1u, h1 = ((s0 << 10) | b0) * 0x9E3779B1u;
                            uint32_t h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) * 0x9E3779B1u, h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) * 0x9E3779B1u;
                            uint2 c0 = T.c64_5[(h0 >> 16) & 63u], c2 = T.c64_5[(h2 >> 16) & 63u];
                            float q[8];
                            apply_aff(T.lut5[(b0 ^ (h0 >> 22)) & 1023u], c0, q[0], q[1]);
                            apply_aff(T.lut5[(b1 ^ (h1 >> 22)) & 1023u], c0, q[2], q[3]);
                            apply_aff(T.lut5[(b2 ^ (h2 >> 22)) & 1023u], c2, q[4], q[5]);
                            apply_aff(T.lut5[(b3 ^ (h3 >> 22)) & 1023u], c2, q[6], q[7]);
                            put8(h * 4u + k, q);
                        }
                    }
                }
            }
        }
    }
}

static void h_mv_ref(const std::vector<double>& W_rot, const double* x, uint32_t R, uint32_t C, uint32_t sr_split, double* out) {
    std::vector<double> xr(C);
    for (uint32_t bc = 0; bc < C / 128u; ++bc) {
        double blk[128];
        for (int j = 0; j < 128; ++j) blk[j] = x[bc * 128u + j] * h_sgn(0x12345678u, bc * 128u + j);
        h_fwht128(blk);
        for (int j = 0; j < 128; ++j) xr[bc * 128u + j] = blk[j];
    }
    std::vector<double> yr(R, 0.0);
    for (uint32_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (uint32_t c = 0; c < C; ++c) s += W_rot[(size_t)r * C + c] * xr[c];
        yr[r] = s;
    }
    for (uint32_t rb = 0; rb < R / 128u; ++rb) {
        double blk[128];
        for (int j = 0; j < 128; ++j) blk[j] = yr[rb * 128u + j];
        h_fwht128(blk);
        uint32_t sr0 = (sr_split != 0u && rb * 128u >= sr_split) ? (rb * 128u - sr_split) : (rb * 128u);
        for (int j = 0; j < 128; ++j) out[rb * 128u + j] = blk[j] * h_sgn(0x9E3779B9u, sr0 + j);
    }
}

static void run_parity_suite() {
    tfq3::PwV3Tables h_tbl;
    init_synth_tables(h_tbl);
    tfq3::PwV3Tables* d_tbl;
    CUDA_CHECK(cudaMalloc(&d_tbl, sizeof(h_tbl)));
    CUDA_CHECK(cudaMemcpy(d_tbl, &h_tbl, sizeof(h_tbl), cudaMemcpyHostToDevice));

    __half* d_xrot; float* d_yr;
    const size_t max_yr = 64ull * 34816ull * 8ull;
    CUDA_CHECK(cudaMalloc(&d_xrot, 128ull * 2048ull * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_yr, max_yr * sizeof(float)));

    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 0.25f);

    // 1. Dense PTX m16n8k16 parity + bit-exact speculative verify (rows=1 == rows=16 == rows=32)
    for (uint32_t bits : {3u, 4u, 5u}) {
        const uint32_t R = 256u, C = 1024u, ldx = 1024u, ldo = 384u, sr_split = 128u;
        const size_t n_words = (size_t)(R / 16u) * (C / 128u) * (bits * 2u) * 32ull;
        const size_t n_sc = (size_t)(R / 128u) * (C / 128u);
        std::vector<uint32_t> h_W(n_words);
        for (auto& w : h_W) w = rng();
        std::vector<__half> h_sc(n_sc);
        for (auto& s : h_sc) s = __float2half(0.05f + 0.02f * std::fabs(nd(rng)));

        std::vector<double> W_rot;
        h_decode_wrot(h_W.data(), h_sc.data(), h_tbl, bits, R, C, W_rot);

        uint32_t* d_W; __half* d_sc;
        CUDA_CHECK(cudaMalloc(&d_W, n_words * 4));
        CUDA_CHECK(cudaMalloc(&d_sc, n_sc * 2));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), n_words * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, h_sc.data(), n_sc * 2, cudaMemcpyHostToDevice));

        std::vector<uint16_t> h_X_all(68ull * ldx);
        for (auto& x : h_X_all) x = h_f32_to_bf16(nd(rng));
        uint16_t *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc(&d_X, h_X_all.size() * 2));
        CUDA_CHECK(cudaMalloc(&d_Y, 68ull * ldo * 2));
        CUDA_CHECK(cudaMemcpy(d_X, h_X_all.data(), h_X_all.size() * 2, cudaMemcpyHostToDevice));

        std::vector<uint16_t> row0_r1(R);
        for (uint32_t rows : {1u, 4u, 16u, 32u, 68u}) {
            std::vector<double> ref((size_t)rows * R, 0.0);
            for (uint32_t m = 0; m < rows; ++m) {
                std::vector<double> xm(C);
                for (uint32_t c = 0; c < C; ++c) xm[c] = (double)h_bf16_to_f32(h_X_all[(size_t)m * ldx + c]);
                h_mv_ref(W_rot, xm.data(), R, C, sr_split, &ref[(size_t)m * R]);
            }
            for (int mode = 0; mode < (rows <= 64u ? 2 : 1); ++mode) {
                CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)rows * ldo * 2));
                CUDA_CHECK(tfq3::launch_dense(d_W, d_sc, d_tbl, d_X, d_Y, d_xrot, d_yr, mode == 0 ? max_yr : 0ull, bits, rows, R, C, ldx, ldo, sr_split));
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<uint16_t> h_Y((size_t)rows * ldo);
                CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, (size_t)rows * ldo * 2, cudaMemcpyDeviceToHost));
                double max_err = 0.0, max_ref = 0.0;
                for (uint32_t m = 0; m < rows; ++m) {
                    for (uint32_t r = 0; r < R; ++r) {
                        double got = (double)h_bf16_to_f32(h_Y[(size_t)m * ldo + r]);
                        double exp = ref[(size_t)m * R + r];
                        max_err = std::max(max_err, std::fabs(got - exp));
                        max_ref = std::max(max_ref, std::fabs(exp));
                    }
                }
                double rel = max_err / (max_ref + 1e-6);
                if (rel > 0.025) {
                    fprintf(stderr, "FAIL dense bits=%u rows=%u mode=%d max_err=%.5f rel=%.5f\n", bits, rows, mode, max_err, rel);
                    exit(1);
                }
                if (mode == 0 && rows == 1u) {
                    memcpy(row0_r1.data(), h_Y.data(), R * 2);
                } else if (mode == 0 && rows <= 64u) {
                    if (memcmp(row0_r1.data(), h_Y.data(), R * 2) != 0) {
                        fprintf(stderr, "FAIL bit-exact speculative verify bits=%u rows=%u vs rows=1\n", bits, rows);
                        exit(1);
                    }
                }
            }
        }
        cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); cudaFree(d_sc);
    }

    // 2. Mixed 3b/4b/5b routed MoE parity + bit-exact rows=1 == rows=16 (act_mode=0 clamped & act_mode=1)
    {
        const uint32_t E = 3u, D = 256u, I = 256u, top_k = 3u, R = 16u;
        const uint32_t ebits[E] = {3u, 4u, 5u};
        std::vector<uint32_t> h_W;
        std::vector<__half> h_S;
        std::vector<tfq3::TfpwExpertMeta> h_M(E);
        std::vector<double> Wg[E], Wu[E], Wd[E];
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t b = ebits[e];
            size_t gu_w = (size_t)(I / 16u) * (D / 128u) * (b * 2u) * 32ull;
            size_t dn_w = (size_t)(D / 16u) * (I / 128u) * (b * 2u) * 32ull;
            size_t gu_s = (size_t)(I / 128u) * (D / 128u);
            size_t dn_s = (size_t)(D / 128u) * (I / 128u);
            memset(&h_M[e], 0, sizeof(h_M[e]));
            h_M[e].b_bits = b;
            auto append_mat = [&](size_t nw, size_t ns, uint64_t& woff, uint32_t& soff, uint32_t nr, uint32_t nc, std::vector<double>& dec) {
                woff = h_W.size(); soff = (uint32_t)h_S.size();
                h_W.resize(woff + nw); h_S.resize(soff + ns);
                for (size_t i = 0; i < nw; ++i) h_W[woff + i] = rng();
                for (size_t i = 0; i < ns; ++i) h_S[soff + i] = __float2half(0.05f + 0.02f * std::fabs(nd(rng)));
                h_decode_wrot(h_W.data() + woff, h_S.data() + soff, h_tbl, b, nr, nc, dec);
            };
            append_mat(gu_w, gu_s, h_M[e].gate_word_off, h_M[e].gate_sc_off, I, D, Wg[e]);
            append_mat(gu_w, gu_s, h_M[e].up_word_off,   h_M[e].up_sc_off,   I, D, Wu[e]);
            append_mat(dn_w, dn_s, h_M[e].down_word_off, h_M[e].down_sc_off, D, I, Wd[e]);
        }
        uint32_t* d_W; __half *d_S, *d_act; tfq3::TfpwExpertMeta* d_M;
        uint16_t* d_X; int32_t* d_picks; float *d_ey1, *d_ey16;
        CUDA_CHECK(cudaMalloc(&d_W, h_W.size() * 4)); CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), h_W.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_S, h_S.size() * 2)); CUDA_CHECK(cudaMemcpy(d_S, h_S.data(), h_S.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_M, E * sizeof(tfq3::TfpwExpertMeta))); CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), E * sizeof(tfq3::TfpwExpertMeta), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_act, R * top_k * I * 2));
        std::vector<uint16_t> h_X(R * D);
        for (auto& x : h_X) x = h_f32_to_bf16(nd(rng));
        CUDA_CHECK(cudaMalloc(&d_X, h_X.size() * 2)); CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), h_X.size() * 2, cudaMemcpyHostToDevice));
        std::vector<int32_t> h_picks(R * top_k);
        for (uint32_t r = 0; r < R; ++r) for (uint32_t s = 0; s < top_k; ++s) h_picks[r * top_k + s] = (int32_t)((r + s) % E);
        CUDA_CHECK(cudaMalloc(&d_picks, h_picks.size() * 4)); CUDA_CHECK(cudaMemcpy(d_picks, h_picks.data(), h_picks.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_ey1, top_k * D * 4));
        CUDA_CHECK(cudaMalloc(&d_ey16, R * top_k * D * 4));

        std::vector<double> x0(D);
        for (uint32_t c = 0; c < D; ++c) x0[c] = (double)h_bf16_to_f32(h_X[c]);
        for (uint32_t act_mode : {0u, 1u}) {
            float lim = (act_mode == 0u) ? 7.0f : 0.0f;
            CUDA_CHECK(tfq3::launch_moe(d_W, d_S, d_M, d_tbl, d_X, d_picks, d_ey1,  d_xrot, d_act, 1u, D, I, E, top_k, D, top_k, top_k, lim, act_mode, true));
            CUDA_CHECK(tfq3::launch_moe(d_W, d_S, d_M, d_tbl, d_X, d_picks, d_ey16, d_xrot, d_act, R,  D, I, E, top_k, D, top_k, top_k, lim, act_mode, true));
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> ey1(top_k * D), ey16(top_k * D);
            CUDA_CHECK(cudaMemcpy(ey1.data(),  d_ey1,  top_k * D * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ey16.data(), d_ey16, top_k * D * 4, cudaMemcpyDeviceToHost));
            if (memcmp(ey1.data(), ey16.data(), top_k * D * 4) != 0) {
                fprintf(stderr, "FAIL MoE bit-exact r1 vs r16 act_mode=%u\n", act_mode); exit(1);
            }
            for (uint32_t s = 0; s < top_k; ++s) {
                std::vector<double> gr(I), ur(I), z(I), dr(D);
                h_mv_ref(Wg[s], x0.data(), I, D, 0u, gr.data());
                h_mv_ref(Wu[s], x0.data(), I, D, 0u, ur.data());
                for (uint32_t i = 0; i < I; ++i) {
                    double gv = h_bf16_to_f32(h_f32_to_bf16((float)gr[i]));
                    double uv = h_bf16_to_f32(h_f32_to_bf16((float)ur[i]));
                    if (act_mode == 0u && lim > 0.0f) {
                        double fl = h_bf16_to_f32(h_f32_to_bf16(lim));
                        gv = std::min(gv, fl); uv = std::min(std::max(uv, -fl), fl);
                    }
                    double sg = gv / (1.0 + std::exp(-gv));
                    if (act_mode != 0u) sg = h_bf16_to_f32(h_f32_to_bf16((float)sg));
                    z[i] = h_bf16_to_f32(h_f32_to_bf16((float)(sg * uv)));
                }
                h_mv_ref(Wd[s], z.data(), D, I, 0u, dr.data());
                double max_err = 0.0, max_ref = 0.0;
                for (uint32_t c = 0; c < D; ++c) {
                    max_err = std::max(max_err, std::fabs((double)ey1[s * D + c] - dr[c]));
                    max_ref = std::max(max_ref, std::fabs(dr[c]));
                }
                if (max_err / (max_ref + 1e-6) > 0.025) {
                    fprintf(stderr, "FAIL MoE parity act_mode=%u expert=%u rel=%.5f\n", act_mode, s, max_err / (max_ref + 1e-6));
                    exit(1);
                }
            }
        }
        cudaFree(d_W); cudaFree(d_S); cudaFree(d_M); cudaFree(d_act);
        cudaFree(d_X); cudaFree(d_picks); cudaFree(d_ey1); cudaFree(d_ey16);
    }

    // 3. BF16 linear parity check (mv1 + mm)
    {
        const uint32_t N = 96u, K = 5120u, ldo = 128u;
        std::vector<uint16_t> h_W((size_t)N * K), h_X(4ull * K);
        for (auto& w : h_W) w = h_f32_to_bf16(nd(rng));
        for (auto& x : h_X) x = h_f32_to_bf16(nd(rng));
        uint16_t *d_W, *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc(&d_W, (size_t)N * K * 2));
        CUDA_CHECK(cudaMalloc(&d_X, 4ull * K * 2));
        CUDA_CHECK(cudaMalloc(&d_Y, 4ull * ldo * 2));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), (size_t)N * K * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), 4ull * K * 2, cudaMemcpyHostToDevice));
        tfq3::DenseDims d1{1u, N, K, K, N, 0u, 1u, 0u};
        tfq3::tfq3_bf16_mv1_kernel<<<N, 32>>>(d_W, d_X, d_Y, d1);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> h_Y1(N);
        CUDA_CHECK(cudaMemcpy(h_Y1.data(), d_Y, N * 2, cudaMemcpyDeviceToHost));
        for (uint32_t r = 0; r < N; ++r) {
            double ref = 0.0;
            for (uint32_t c = 0; c < K; ++c) ref += (double)h_bf16_to_f32(h_W[(size_t)r * K + c]) * h_bf16_to_f32(h_X[c]);
            if (std::fabs((double)h_bf16_to_f32(h_Y1[r]) - ref) > 0.05 * std::fabs(ref) + 0.05) {
                fprintf(stderr, "FAIL bf16_mv1 r=%u\n", r); exit(1);
            }
        }
        tfq3::DenseDims d4{4u, N, K, K, ldo, 0u, 1u, 0u};
        tfq3::tfq3_bf16_mm_kernel<<<dim3((N + 3u) / 4u, 1u, 1u), 128>>>(d_W, d_X, d_Y, d4);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> h_Y4(4ull * ldo);
        CUDA_CHECK(cudaMemcpy(h_Y4.data(), d_Y, 4ull * ldo * 2, cudaMemcpyDeviceToHost));
        for (uint32_t m = 0; m < 4u; ++m) for (uint32_t r = 0; r < N; ++r) {
            double ref = 0.0;
            for (uint32_t c = 0; c < K; ++c) ref += (double)h_bf16_to_f32(h_W[(size_t)r * K + c]) * h_bf16_to_f32(h_X[(size_t)m * K + c]);
            if (std::fabs((double)h_bf16_to_f32(h_Y4[(size_t)m * ldo + r]) - ref) > 0.05 * std::fabs(ref) + 0.05) {
                fprintf(stderr, "FAIL bf16_mm m=%u r=%u\n", m, r); exit(1);
            }
        }
        cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
    }

    // 4. Long-context prefill (BM=64) + Zero-Repack Tensor-Parallel (Row-TP & Col-TP, tp=2,4) vs fp64
    for (uint32_t bits : {3u, 4u, 5u}) {
        const uint32_t N = 512u, K = 1024u, rows = 96u;
        size_t nw = (size_t)(N / 16u) * (K / 128u) * (bits * 2u) * 32ull;
        size_t ns = (size_t)(N / 128u) * (K / 128u);
        std::vector<uint32_t> h_W(nw);
        std::vector<__half> h_sc(ns);
        std::vector<uint16_t> h_X(rows * K);
        for (auto& w : h_W) w = rng();
        for (auto& s : h_sc) s = __float2half(0.03f + 0.02f * ((float)(rng() & 255u) / 255.0f));
        for (auto& x : h_X) x = h_f32_to_bf16(((int)(rng() % 201u) - 100) * 0.01f);
        std::vector<double> W_rot;
        h_decode_wrot(h_W.data(), h_sc.data(), h_tbl, bits, N, K, W_rot);

        uint32_t* d_W; __half* d_sc; uint16_t* d_X; float* d_Y;
        CUDA_CHECK(cudaMalloc(&d_W, nw * 4)); CUDA_CHECK(cudaMalloc(&d_sc, ns * 2));
        CUDA_CHECK(cudaMalloc(&d_X, rows * K * 2)); CUDA_CHECK(cudaMalloc(&d_Y, rows * N * 4));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), nw * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, h_sc.data(), ns * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), rows * K * 2, cudaMemcpyHostToDevice));

        // 4a. Full BM=64 prefill kernel vs fp64
        CUDA_CHECK(tfq3::launch_dense(d_W, d_sc, d_tbl, d_X, d_Y, d_xrot, d_yr, 0, bits, rows, N, K, K, N, 0u, true));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h_Y_full(rows * N);
        CUDA_CHECK(cudaMemcpy(h_Y_full.data(), d_Y, rows * N * 4, cudaMemcpyDeviceToHost));
        double dot = 0.0, na = 0.0, nb = 0.0;
        std::vector<double> xm(K), ref_row(N);
        for (uint32_t m = 0; m < rows; ++m) {
            for (uint32_t c = 0; c < K; ++c) xm[c] = (double)h_bf16_to_f32(h_X[(size_t)m * K + c]);
            h_mv_ref(W_rot, xm.data(), N, K, 0u, ref_row.data());
            for (uint32_t r = 0; r < N; ++r) {
                double g = h_Y_full[(size_t)m * N + r], ref = ref_row[r];
                dot += g * ref; na += g * g; nb += ref * ref;
            }
        }
        double cos_full = dot / (std::sqrt(na * nb) + 1e-30);
        if (cos_full < 0.99999) { fprintf(stderr, "FAIL prefill %ub cos=%.8f\n", bits, cos_full); exit(1); }

        // 4b. Col-TP (tp=2,4): slice N into tp independent ranks -> concat must match full bit-exact
        for (uint32_t tp : {2u, 4u}) {
            uint32_t n_loc = N / tp;
            std::vector<float> h_cat(rows * N, 0.0f);
            for (uint32_t rank = 0; rank < tp; ++rank) {
                CUDA_CHECK(tfq3::launch_dense_tp(d_W, d_sc, d_tbl, d_X, d_Y, d_xrot, bits, rows, N, K,
                                                 rank * n_loc, n_loc, 0u, K, K, n_loc, true));
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<float> h_part(rows * n_loc);
                CUDA_CHECK(cudaMemcpy(h_part.data(), d_Y, rows * n_loc * 4, cudaMemcpyDeviceToHost));
                for (uint32_t m = 0; m < rows; ++m)
                    memcpy(&h_cat[(size_t)m * N + rank * n_loc], &h_part[(size_t)m * n_loc], n_loc * 4);
            }
            for (size_t i = 0; i < h_Y_full.size(); ++i) {
                if (h_cat[i] != h_Y_full[i]) { fprintf(stderr, "FAIL Col-TP=%u %ub mismatch at %zu\n", tp, bits, i); exit(1); }
            }
        }

        // 4c. Row-TP (tp=2,4): slice K into tp independent ranks with zero-repack state resume -> AllReduce(SUM) == full
        for (uint32_t tp : {2u, 4u}) {
            uint32_t k_loc = K / tp;
            std::vector<float> h_sum(rows * N, 0.0f);
            for (uint32_t rank = 0; rank < tp; ++rank) {
                CUDA_CHECK(tfq3::launch_dense_tp(d_W, d_sc, d_tbl, d_X + rank * k_loc, d_Y, d_xrot, bits, rows, N, K,
                                                 0u, N, rank * k_loc, k_loc, K, N, true));
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<float> h_part(rows * N);
                CUDA_CHECK(cudaMemcpy(h_part.data(), d_Y, rows * N * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < h_sum.size(); ++i) h_sum[i] += h_part[i];
            }
            double dtp = 0.0, n1 = 0.0, n2 = 0.0;
            for (size_t i = 0; i < h_sum.size(); ++i) {
                dtp += (double)h_sum[i] * h_Y_full[i]; n1 += (double)h_sum[i] * h_sum[i]; n2 += (double)h_Y_full[i] * h_Y_full[i];
            }
            double cos_tp = dtp / (std::sqrt(n1 * n2) + 1e-30);
            if (cos_tp < 0.999999) { fprintf(stderr, "FAIL Row-TP=%u %ub cos=%.8f\n", tp, bits, cos_tp); exit(1); }
        }
        cudaFree(d_W); cudaFree(d_sc); cudaFree(d_X); cudaFree(d_Y);
    }

    cudaFree(d_tbl); cudaFree(d_xrot); cudaFree(d_yr);
    printf("PASS tfq3_cuda_test (3b/4b/5b PTX sk8+fused+prefill64 + bit-exact r1==r16==r32 + Col-TP/Row-TP 2,4 + mixed-bit MoE + bf16)\n");
}

static void run_bench() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    tfq3::PwV3Tables h_tbl; init_synth_tables(h_tbl);
    tfq3::PwV3Tables* d_tbl; CUDA_CHECK(cudaMalloc(&d_tbl, sizeof(h_tbl)));
    CUDA_CHECK(cudaMemcpy(d_tbl, &h_tbl, sizeof(h_tbl), cudaMemcpyHostToDevice));

    struct Shape { const char* name; uint32_t N, K; };
    Shape shapes[] = {
        {"gate+up", 34816, 5120},
        {"down",     5120, 17408},
        {"qkv",     10240, 5120},
        {"lm_head", 248320, 5120},
    };
    __half* d_xrot; float* d_yr; uint16_t *d_X, *d_Y;
    const size_t max_yr = 64ull * 34816ull * 8ull;
    CUDA_CHECK(cudaMalloc(&d_xrot, 1024ull * 17408ull * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_yr, max_yr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, 1024ull * 17408ull * 2ull));
    CUDA_CHECK(cudaMalloc(&d_Y, 1024ull * 248320ull * 2ull));
    CUDA_CHECK(cudaMemset(d_X, 0x3C, 1024ull * 17408ull * 2ull));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    for (auto& s : shapes) {
        const uint32_t bits = 4u;
        size_t nw = (size_t)(s.N / 16u) * (s.K / 128u) * (bits * 2u) * 32ull;
        size_t ns = (size_t)(s.N / 128u) * (s.K / 128u);
        uint32_t* d_W; __half* d_sc;
        CUDA_CHECK(cudaMalloc(&d_W, nw * 4)); CUDA_CHECK(cudaMalloc(&d_sc, ns * 2));
        CUDA_CHECK(cudaMemset(d_W, 0x5A, nw * 4)); CUDA_CHECK(cudaMemset(d_sc, 0x2E, ns * 2));
        double bytes = (double)(nw * 4 + ns * 2);

        for (uint32_t rows : {1u, 16u, 64u, 256u, 1024u}) {
            if (rows >= 256u && s.N == 248320u) continue;
            for (int w = 0; w < 5; ++w) CUDA_CHECK(tfq3::launch_dense(d_W, d_sc, d_tbl, d_X, d_Y, d_xrot, d_yr, max_yr, bits, rows, s.N, s.K, s.K, s.N, 0u));
            CUDA_CHECK(cudaDeviceSynchronize());
            int iters = (rows <= 16u) ? 50 : 10;
            CUDA_CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < iters; ++it) CUDA_CHECK(tfq3::launch_dense(d_W, d_sc, d_tbl, d_X, d_Y, d_xrot, d_yr, max_yr, bits, rows, s.N, s.K, s.K, s.N, 0u));
            CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            double avg_ms = ms / iters;
            double gbps = (bytes / 1e9) / (avg_ms / 1e3);
            double tflops = (2.0 * rows * s.N * s.K / 1e12) / (avg_ms / 1e3);
            printf("  %-8s [%6u x %5u] 4b rows=%4u : %7.3f ms | %6.1f GB/s | %5.2f TFLOP/s\n", s.name, s.N, s.K, rows, avg_ms, gbps, tflops);
        }
        cudaFree(d_W); cudaFree(d_sc);
    }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_tbl); cudaFree(d_xrot); cudaFree(d_yr); cudaFree(d_X); cudaFree(d_Y);
}

int main(int argc, char** argv) {
    run_parity_suite();
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--bench") == 0) do_bench = true;
    if (do_bench) run_bench();
    return 0;
}
