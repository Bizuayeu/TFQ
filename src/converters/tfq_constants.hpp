#pragma once
// ============================================================================
// Shared TFQ3 / TFQ4 Converter Constants, Finiteness Guards, and
// Platform-Independent Box-Muller Gaussian Sampler.
//
// Shared by ALL Apple Silicon Metal (.mm) and NVIDIA CUDA (.cu) converters so
// shrink factors, table training, and NaN/Inf validation cannot drift across
// backends or C++ standard libraries.
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

namespace tfq_common {

// Canonical per-family RMS scale shrink factors (shared by Metal & CUDA).
static inline float generic_shrink(uint32_t bits) {
    return (bits == 5u) ? 0.96f : ((bits == 4u) ? 0.94f : 0.92f);
}

static inline float qwen27_shrink(uint32_t bits) {
    return (bits == 3u) ? 0.992f : ((bits == 4u) ? 0.997f : 0.999f);
}

static inline float moe_shrink(uint32_t /*bits*/) {
    return 1.0f;
}

// Fast host IEEE-754 half conversion (round-to-nearest-even, identical on all CPUs).
static inline uint16_t f32_to_f16_rne(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t aexp = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (aexp == 255u) {
        if (mant != 0u) return static_cast<uint16_t>(sign | 0x7E00u | (mant >> 13));
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    int32_t exp = static_cast<int32_t>(aexp) - 127 + 15;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        uint32_t m = mant | 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_bit = 1u << (shift - 1u);
        uint32_t rem_mask = (1u << shift) - 1u;
        uint32_t out_m = m >> shift;
        uint32_t rem = m & rem_mask;
        if (rem > half_bit || (rem == half_bit && (out_m & 1u))) out_m++;
        return static_cast<uint16_t>(sign | out_m);
    }
    uint32_t out_m = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (out_m & 1u))) {
        out_m++;
        if (out_m == 0x400u) {
            out_m = 0u;
            exp++;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | out_m);
}

static inline float f16_to_f32_exact(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0u) {
        if (mant == 0u) out = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static inline uint32_t pack_h2_rne(float x, float y) {
    return static_cast<uint32_t>(f32_to_f16_rne(x)) | (static_cast<uint32_t>(f32_to_f16_rne(y)) << 16);
}

// Fast finiteness checks for model-specific converters (Issue #11).
static inline void check_finite_bf16(const uint16_t* data, uint32_t R, uint32_t C, const std::string& name, uint32_t r_off = 0u) {
    for (uint32_t r = 0; r < R; ++r) {
        const uint16_t* row = data + static_cast<size_t>(r) * C;
        for (uint32_t c = 0; c < C; ++c) {
            if ((row[c] & 0x7F80u) == 0x7F80u) {
                bool is_nan = (row[c] & 0x007Fu) != 0u;
                fprintf(stderr, "Non-finite BF16 weight (%s) in tensor '%s' at [%u, %u]; refusing to quantize\n",
                        is_nan ? "NaN" : "Inf", name.c_str(), r_off + r, c);
                std::exit(1);
            }
        }
    }
}

static inline void check_finite_f32(const float* data, uint32_t R, uint32_t C, const std::string& name, uint32_t r_off = 0u) {
    for (uint32_t r = 0; r < R; ++r) {
        const float* row = data + static_cast<size_t>(r) * C;
        for (uint32_t c = 0; c < C; ++c) {
            if (!std::isfinite(row[c])) {
                fprintf(stderr, "Non-finite F32 value (%s) in tensor '%s' at [%u, %u]; refusing to quantize\n",
                        std::isnan(row[c]) ? "NaN" : "Inf", name.c_str(), r_off + r, c);
                std::exit(1);
            }
        }
    }
}

static inline void check_finite_fp8_e4m3(const uint8_t* data, uint32_t R, uint32_t C, const std::string& name, uint32_t r_off = 0u) {
    for (uint32_t r = 0; r < R; ++r) {
        const uint8_t* row = data + static_cast<size_t>(r) * C;
        for (uint32_t c = 0; c < C; ++c) {
            if ((row[c] & 0x7Fu) == 0x7Fu) {
                fprintf(stderr, "Non-finite FP8 E4M3 weight (NaN) in tensor '%s' at [%u, %u]; refusing to quantize\n",
                        name.c_str(), r_off + r, c);
                std::exit(1);
            }
        }
    }
}

// Standard-library-independent Box-Muller normal sampler on std::mt19937.
// Unlike std::normal_distribution (whose implementation differs between
// libc++ and libstdc++), this uses the C++ standard-specified mt19937 engine
// with an explicit Box-Muller transform.
struct BoxMullerNormal {
    std::mt19937 rng;
    bool has_spare = false;
    float spare = 0.0f;

    explicit BoxMullerNormal(uint32_t seed) : rng(seed) {}

    float next() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        // Map uint32_t uniformly into (0, 1) with no endpoint zeros.
        double u1 = (static_cast<double>(rng()) + 0.5) * (1.0 / 4294967296.0);
        double u2 = (static_cast<double>(rng()) + 0.5) * (1.0 / 4294967296.0);
        double rad = std::sqrt(-2.0 * std::log(u1));
        double theta = 6.283185307179586476925286766559 * u2;
        spare = static_cast<float>(rad * std::sin(theta));
        has_spare = true;
        return static_cast<float>(rad * std::cos(theta));
    }
};

} // namespace tfq_common
