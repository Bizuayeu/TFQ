// ============================================================================
// Standalone Deterministic Codebook Trainer for TFQ3 (PWV3 v3) & TFQ4 (PWV4 v4)
//
// Uses platform-independent Box-Muller over std::mt19937 (no std::normal_distribution)
// to generate canonical 16 KiB codebook binaries:
//   - reference/tables/pw_v3_tables.bin
//   - reference/tables/pw_v4_e8p_tables.bin
// and optionally emit src/converters/canonical_tables.hpp so all converters
// embed the exact canonical codebooks with zero runtime path or libm drift.
// ============================================================================
#include "../converters/tfq_constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

struct PwV3TablesBin {
    uint32_t magic, version;
    float a3, b3, a4, b4, a5, b5;
    uint32_t pad0[8];
    uint32_t lut3_u32[256];
    uint32_t lut4_u32[1024];
    uint32_t lut5_u32[1024];
    uint16_t c64_3[64][4], c64_4[64][4], c64_5[64][4];
    uint8_t pad1[16384 - 10816];
};
static_assert(sizeof(PwV3TablesBin) == 16384, "PwV3TablesBin must be 16384 bytes");

struct PwV4TablesBin {
    uint32_t magic, version, k_codes, dim;
    uint32_t pad0[12];
    uint16_t e8p_lut[256][8];
    uint8_t pad1[16384 - 64 - 4096];
};
static_assert(sizeof(PwV4TablesBin) == 16384, "PwV4TablesBin must be 16384 bytes");

struct P2 { float x, y; };

static double inv_norm_cdf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
                                3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - plow;
    double q, r;
    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

static std::vector<P2> train_2d_boxmuller(int K, uint32_t seed, int iters) {
    tfq_common::BoxMullerNormal rng(seed);
    const int N = 131072;
    std::vector<P2> pts(N);
    for (int i = 0; i < N; ++i) {
        float x = rng.next();
        float y = rng.next();
        pts[i] = {x, y};
    }
    int g = static_cast<int>(std::round(std::sqrt(static_cast<double>(K))));
    std::vector<P2> cen(K);
    for (int i = 0; i < K; ++i) {
        int gx = i % g, gy = i / g;
        float jx = rng.next(), jy = rng.next();
        cen[i] = {static_cast<float>(inv_norm_cdf((gx + 0.5) / g)) + 0.03f * jx,
                  static_cast<float>(inv_norm_cdf((gy + 0.5) / g)) + 0.03f * jy};
    }
    std::vector<double> sx(K), sy(K);
    std::vector<int> cnt(K);
    for (int it = 0; it < iters; ++it) {
        std::fill(sx.begin(), sx.end(), 0.0);
        std::fill(sy.begin(), sy.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < N; ++i) {
            float x = pts[i].x, y = pts[i].y, bd = 1e30f;
            int bi = 0;
            for (int k = 0; k < K; ++k) {
                float dx = x - cen[k].x, dy = y - cen[k].y, d = dx * dx + dy * dy;
                if (d < bd) { bd = d; bi = k; }
            }
            sx[bi] += x; sy[bi] += y; cnt[bi]++;
        }
        for (int k = 0; k < K; ++k) {
            if (cnt[k] > 0) cen[k] = {static_cast<float>(sx[k] / cnt[k]), static_cast<float>(sy[k] / cnt[k])};
        }
    }
    return cen;
}

static void fill_c64(uint16_t dst[64][4], float a, float b) {
    for (int c = 0; c < 64; ++c) {
        float t0 = static_cast<float>(((c >> 0) & 3) * 2 - 3) / 3.0f;
        float t1 = static_cast<float>(((c >> 2) & 3) * 2 - 3) / 3.0f;
        float t2 = static_cast<float>(((c >> 4) & 1) * 2 - 1);
        float t3 = static_cast<float>(((c >> 5) & 1) * 2 - 1);
        dst[c][0] = tfq_common::f32_to_f16_rne(1.0f + a * t2);
        dst[c][1] = tfq_common::f32_to_f16_rne(1.0f + a * t3);
        dst[c][2] = tfq_common::f32_to_f16_rne(b * t0);
        dst[c][3] = tfq_common::f32_to_f16_rne(b * t1);
    }
}

static void build_pwv3_canonical(PwV3TablesBin& tbl) {
    std::memset(&tbl, 0, sizeof(tbl));
    tbl.magic = 0x33565750u;
    tbl.version = 3u;
    tbl.a3 = 0.22f; tbl.b3 = 0.38f;
    tbl.a4 = 0.18f; tbl.b4 = 0.30f;
    tbl.a5 = 0.10f; tbl.b5 = 0.18f;
    for (int c = 0; c < 4; ++c) {
        auto t3 = train_2d_boxmuller(64, 10u + static_cast<uint32_t>(c), 40);
        for (int i = 0; i < 64; ++i) tbl.lut3_u32[i * 4 + c] = tfq_common::pack_h2_rne(t3[i].x, t3[i].y);
    }
    for (int c = 0; c < 4; ++c) {
        auto t4 = train_2d_boxmuller(256, 100u + static_cast<uint32_t>(c), 35);
        for (int i = 0; i < 256; ++i) tbl.lut4_u32[i * 4 + c] = tfq_common::pack_h2_rne(t4[i].x, t4[i].y);
    }
    auto t5 = train_2d_boxmuller(1024, 500u, 30);
    for (int i = 0; i < 1024; ++i) tbl.lut5_u32[i] = tfq_common::pack_h2_rne(t5[i].x, t5[i].y);
    fill_c64(tbl.c64_3, tbl.a3, tbl.b3);
    fill_c64(tbl.c64_4, tbl.a4, tbl.b4);
    fill_c64(tbl.c64_5, tbl.a5, tbl.b5);
}

static void build_pwv4_canonical(PwV4TablesBin& t) {
    std::memset(&t, 0, sizeof(t));
    t.magic   = 0x34565750u;
    t.version = 4u;
    t.k_codes = 256u;
    t.dim     = 8u;

    const float vals[5] = {0.25f, 0.75f, 1.25f, 1.75f, 2.25f};
    tfq_common::BoxMullerNormal rng(2026u);
    const int N_SAMP = 32768;
    std::vector<float> S(static_cast<size_t>(N_SAMP) * 8u);
    for (int i = 0; i < N_SAMP * 8; ++i) S[i] = std::fabs(rng.next());

    float C[256][8];
    std::vector<std::pair<float, int>> norms(N_SAMP);
    for (int i = 0; i < N_SAMP; ++i) {
        float r2 = 0.0f;
        for (int d = 0; d < 8; ++d) r2 += S[i * 8 + d] * S[i * 8 + d];
        norms[i] = {r2, i};
    }
    std::sort(norms.begin(), norms.end());
    for (int k = 0; k < 256; ++k) {
        int pick = norms[((k * 2 + 1) * N_SAMP) / 512].second;
        for (int d = 0; d < 8; ++d) {
            float v = S[pick * 8 + d];
            float snapped = vals[std::min(4, std::max(0, static_cast<int>(std::floor(v * 2.0f))))];
            C[k][d] = 0.5f * v + 0.5f * snapped;
        }
    }

    double sum[256][8];
    int cnt[256];
    for (int it = 0; it < 20; ++it) {
        std::memset(sum, 0, sizeof(sum));
        std::memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < N_SAMP; ++i) {
            const float* x = &S[i * 8];
            int best = 0;
            float bd = 1e30f;
            for (int k = 0; k < 256; ++k) {
                const float* ck = C[k];
                float d0 = x[0] - ck[0], d1 = x[1] - ck[1], d2 = x[2] - ck[2], d3 = x[3] - ck[3];
                float d4 = x[4] - ck[4], d5 = x[5] - ck[5], d6 = x[6] - ck[6], d7 = x[7] - ck[7];
                float dist = ((d0*d0 + d1*d1) + (d2*d2 + d3*d3)) + ((d4*d4 + d5*d5) + (d6*d6 + d7*d7));
                if (dist < bd) { bd = dist; best = k; }
            }
            cnt[best]++;
            for (int d = 0; d < 8; ++d) sum[best][d] += x[d];
        }
        for (int k = 0; k < 256; ++k) {
            if (cnt[k] > 0) {
                double inv = 1.0 / static_cast<double>(cnt[k]);
                for (int d = 0; d < 8; ++d) C[k][d] = static_cast<float>(sum[k][d] * inv);
            }
        }
    }

    for (int k = 0; k < 256; ++k) {
        for (int d = 0; d < 8; ++d) {
            t.e8p_lut[k][d] = tfq_common::f32_to_f16_rne(std::max(0.0f, C[k][d]));
        }
    }
}

static void write_bin(const std::string& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    f.close();
    if (f.fail()) { fprintf(stderr, "Write error on %s\n", path.c_str()); std::exit(1); }
}

static void emit_header(const std::string& path, const PwV3TablesBin& v3, const PwV4TablesBin& v4) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); std::exit(1); }
    fprintf(f, "#pragma once\n");
    fprintf(f, "// Auto-generated by src/tools/tfq3_train_tables.cpp — canonical 16 KiB PWV3 & PWV4 tables.\n");
    fprintf(f, "// Active non-zero prefixes are stored verbatim and zero-padded to 16,384 bytes.\n");
    fprintf(f, "#include <cstdint>\n#include <cstring>\n\nnamespace tfq_canonical {\n\n");
    auto dump_u32 = [&](const char* name, const void* raw, size_t n_bytes) {
        size_t n_u32 = n_bytes / 4u;
        const uint32_t* p = reinterpret_cast<const uint32_t*>(raw);
        fprintf(f, "static constexpr size_t %s_ACTIVE_BYTES = %zu;\n", name, n_bytes);
        fprintf(f, "static constexpr uint32_t %s_U32[%zu] = {\n", name, n_u32);
        for (size_t i = 0; i < n_u32; ++i) {
            if ((i % 8u) == 0u) fprintf(f, "    ");
            fprintf(f, "0x%08Xu%s", p[i], (i + 1u < n_u32) ? "," : "");
            if ((i % 8u) == 7u || i + 1u == n_u32) fprintf(f, "\n");
            else fprintf(f, " ");
        }
        fprintf(f, "};\n\n");
    };
    dump_u32("PWV3", &v3, 10816u);
    dump_u32("PWV4", &v4, 64u + 4096u);
    fprintf(f, "static inline void fill_canonical_pwv3(void* dst16k) {\n");
    fprintf(f, "    std::memset(dst16k, 0, 16384);\n");
    fprintf(f, "    std::memcpy(dst16k, PWV3_U32, PWV3_ACTIVE_BYTES);\n");
    fprintf(f, "}\n\n");
    fprintf(f, "static inline void fill_canonical_pwv4(void* dst16k) {\n");
    fprintf(f, "    std::memset(dst16k, 0, 16384);\n");
    fprintf(f, "    std::memcpy(dst16k, PWV4_U32, PWV4_ACTIVE_BYTES);\n");
    fprintf(f, "}\n\n");
    fprintf(f, "} // namespace tfq_canonical\n");
    if (fclose(f) != 0) { fprintf(stderr, "Write error on %s\n", path.c_str()); std::exit(1); }
}

int main(int argc, char** argv) {
    std::string out_v3 = "reference/tables/pw_v3_tables.bin";
    std::string out_v4 = "reference/tables/pw_v4_e8p_tables.bin";
    std::string out_hdr;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out-v3" && i + 1 < argc) out_v3 = argv[++i];
        else if (a == "--out-v4" && i + 1 < argc) out_v4 = argv[++i];
        else if (a == "--emit-header" && i + 1 < argc) out_hdr = argv[++i];
    }
    PwV3TablesBin v3{};
    PwV4TablesBin v4{};
    build_pwv3_canonical(v3);
    build_pwv4_canonical(v4);
    write_bin(out_v3, &v3, sizeof(v3));
    write_bin(out_v4, &v4, sizeof(v4));
    if (!out_hdr.empty()) emit_header(out_hdr, v3, v4);
    printf("Generated canonical tables:\n  %s (16,384 bytes)\n  %s (16,384 bytes)\n", out_v3.c_str(), out_v4.c_str());
    if (!out_hdr.empty()) printf("  %s\n", out_hdr.c_str());
    return 0;
}
