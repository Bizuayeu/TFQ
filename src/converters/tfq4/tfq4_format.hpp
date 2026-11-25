#pragma once
// ============================================================================
// TFQ4 Cross-Platform Binary Format (PWV4 v4 + TFP4 v4, 2.001 bpw 8D E8P)
//   - Byte-identical on Apple Silicon Metal and NVIDIA CUDA (16 KiB-aligned)
//   - Shares exact 2D FWHT-128 rotation & 128x128 fp16 tile scales with TFQ3
//   - Deterministic data-free 256-entry positive-orthant 8D E8P codebook (4 KiB)
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace tfq4 {

static constexpr uint32_t MAGIC_PWV4 = 0x34565750u; // "PWV4"
static constexpr uint32_t MAGIC_TFP4 = 0x34504654u; // "TFP4"
static constexpr uint32_t FORMAT_VER = 4u;
static constexpr size_t   ALIGN_16K  = 16384;

struct PwV4E8Tables {
    uint32_t magic;
    uint32_t version;
    uint32_t k_codes; // 256
    uint32_t dim;     // 8
    uint32_t pad0[12];
    uint16_t e8p_lut[256][8]; // fp16 positive-orthant 8D centroids (4,096 bytes)
    uint8_t  pad1[16384 - 64 - 4096];
};
static_assert(sizeof(PwV4E8Tables) == 16384, "PwV4E8Tables must be 16384 bytes");

struct Tfp4Header {
    uint32_t magic;
    uint32_t version;
    uint32_t num_tensors;
    uint32_t pad0;
    uint64_t off_meta;
    uint64_t off_scales;
    uint64_t off_weights;
    uint64_t total_scale_halfs;
    uint64_t total_weight_words;
    uint64_t pad1;
};
static_assert(sizeof(Tfp4Header) == 64, "Tfp4Header must be 64 bytes");

struct Tfp4TensorMeta {
    char     name[80];
    uint32_t R;          // padded multiple of 128
    uint32_t C;          // padded multiple of 128
    uint32_t b_bits;     // 2 (E8P 2.0 bpw)
    uint32_t sc_off;     // offset in fp16 scales
    uint64_t word_off;   // offset in uint32_t words
    uint64_t word_count; // (uint64_t)R * (C / 16)
    float    snr_db;
    uint32_t pad[3];     // [0]=orig_R, [1]=orig_C, [2]=flags
};
static_assert(sizeof(Tfp4TensorMeta) == 128, "Tfp4TensorMeta must be 128 bytes");

static inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

static inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// Deterministic 100% data-free 256-code positive-orthant 8D E8P Gaussian codebook.
// Initialized on spherical E8 + 1/4 shells, then polished with 20 Lloyd-Max steps
// over synthetic |N(0, 1)^8| samples (reproducible bit-for-bit on Metal and CUDA).
static inline void build_e8p_tables(PwV4E8Tables& t) {
    std::memset(&t, 0, sizeof(t));
    t.magic   = MAGIC_PWV4;
    t.version = FORMAT_VER;
    t.k_codes = 256u;
    t.dim     = 8u;

    // 1. Shifted E8 + 0.25 positive-orthant anchor grid + synthetic |N(0,1)^8| bank.
    const float vals[5] = {0.25f, 0.75f, 1.25f, 1.75f, 2.25f};
    std::mt19937 rng(2026u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Deterministic synthetic |N(0,1)^8| training bank (32,768 samples).
    const int N_SAMP = 32768;
    std::vector<float> S((size_t)N_SAMP * 8);
    for (int i = 0; i < N_SAMP * 8; ++i) S[i] = std::fabs(nd(rng));

    // Seed 256 centroids via stratified radial shells from the synthetic Gaussian bank
    // plus E8 shifted-orthant lattice anchors for clean Voronoi coverage.
    float C[256][8];
    // Deterministic farthest/quantile seed across 256 radial buckets of |N(0,1)^8|
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
            // Snap slightly toward E8 + 0.25 shell grid on init
            float v = S[pick * 8 + d];
            float snapped = vals[std::min(4, std::max(0, (int)std::floor(v * 2.0f)))];
            C[k][d] = 0.5f * v + 0.5f * snapped;
        }
    }

    // 20 iterations of exact Lloyd-Max k-means in 8D positive orthant.
    double sum[256][8];
    int    cnt[256];
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
                float dist = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
                if (dist < bd) { bd = dist; best = k; }
            }
            cnt[best]++;
            for (int d = 0; d < 8; ++d) sum[best][d] += x[d];
        }
        for (int k = 0; k < 256; ++k) {
            if (cnt[k] > 0) {
                double inv = 1.0 / (double)cnt[k];
                for (int d = 0; d < 8; ++d) C[k][d] = (float)(sum[k][d] * inv);
            }
        }
    }

    // Store canonical fp16 centroids.
    for (int k = 0; k < 256; ++k) {
        for (int d = 0; d < 8; ++d) {
            t.e8p_lut[k][d] = f32_to_f16_bits(std::max(0.0f, C[k][d]));
        }
    }
}

static inline uint64_t align_up_16k(uint64_t x) {
    return (x + (uint64_t)ALIGN_16K - 1ull) & ~((uint64_t)ALIGN_16K - 1ull);
}

static inline uint64_t tfp4_tensor_bytes(uint32_t pad_R, uint32_t pad_C) {
    uint64_t words = (uint64_t)pad_R * (pad_C / 16u);
    uint64_t sc = (uint64_t)(pad_R / 128u) * (pad_C / 128u);
    return words * 4ull + sc * 2ull;
}

static inline uint64_t write_tfp4_shard(
    const std::string& path,
    const std::vector<Tfp4TensorMeta>& metas,
    const std::vector<uint16_t>& scales,
    const std::vector<uint32_t>& words
) {
    const uint64_t off_meta = sizeof(Tfp4Header);
    const uint64_t off_sc   = align_up_16k(off_meta + (uint64_t)metas.size() * sizeof(Tfp4TensorMeta));
    const uint64_t off_wt   = align_up_16k(off_sc + (uint64_t)scales.size() * 2ull);
    const uint64_t tot_sz   = align_up_16k(off_wt + (uint64_t)words.size() * 4ull);
    Tfp4Header hdr{};
    hdr.magic = MAGIC_TFP4;
    hdr.version = FORMAT_VER;
    hdr.num_tensors = (uint32_t)metas.size();
    hdr.off_meta = off_meta;
    hdr.off_scales = off_sc;
    hdr.off_weights = off_wt;
    hdr.total_scale_halfs = scales.size();
    hdr.total_weight_words = words.size();

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.good()) {
        fprintf(stderr, "Cannot write TFP4 shard: %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<char> zeros(ALIGN_16K, 0);
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!metas.empty()) ofs.write(reinterpret_cast<const char*>(metas.data()), (std::streamsize)(metas.size() * sizeof(Tfp4TensorMeta)));
    uint64_t pos = off_meta + (uint64_t)metas.size() * sizeof(Tfp4TensorMeta);
    if (off_sc > pos) ofs.write(zeros.data(), (std::streamsize)(off_sc - pos));
    if (!scales.empty()) ofs.write(reinterpret_cast<const char*>(scales.data()), (std::streamsize)(scales.size() * 2ull));
    pos = off_sc + (uint64_t)scales.size() * 2ull;
    if (off_wt > pos) ofs.write(zeros.data(), (std::streamsize)(off_wt - pos));
    if (!words.empty()) ofs.write(reinterpret_cast<const char*>(words.data()), (std::streamsize)(words.size() * 4ull));
    pos = off_wt + (uint64_t)words.size() * 4ull;
    if (tot_sz > pos) ofs.write(zeros.data(), (std::streamsize)(tot_sz - pos));
    if (!ofs.good()) {
        fprintf(stderr, "Write error on TFP4 shard: %s\n", path.c_str());
        std::exit(1);
    }
    return tot_sz;
}

} // namespace tfq4
