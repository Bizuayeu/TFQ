#pragma once
// ============================================================================
// 100% Data-Free MoE Expert Sensitivity Ranking (Weight + Router Norm)
//   Replaces index-order (std::iota) fallback when no --trace routing file is
//   supplied. Uses ONLY checkpoint weights (router gate row L2 norm + expert
//   projection scale/Frobenius energy) — zero calibration data, zero forward pass.
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace tfq_moe {

static inline float bf16_to_f32_host(uint16_t u) {
    uint32_t x = (uint32_t)u << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

static inline float f16_to_f32_host(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    float v = (e == 0) ? std::ldexp((float)m, -24)
                       : (e == 31 ? 0.0f : std::ldexp(1.0f + (float)m / 1024.0f, (int)e - 15));
    return s ? -v : v;
}

static inline std::vector<double> zscore(const std::vector<double>& v) {
    const size_t n = v.size();
    std::vector<double> z(n, 0.0);
    if (n == 0) return z;
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= (double)n;
    double var = 0.0;
    for (double x : v) { double d = x - mean; var += d * d; }
    double sd = std::sqrt(var / (double)n);
    if (sd < 1e-12) return z;
    for (size_t i = 0; i < n; ++i) z[i] = (v[i] - mean) / sd;
    return z;
}

static inline std::vector<int> scores_to_ranks(const std::vector<double>& scores) {
    const int n = (int)scores.size();
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    std::vector<int> ranks(n, 0);
    for (int r = 0; r < n; ++r) ranks[order[r]] = r;
    return ranks;
}

template <typename CacheT>
static inline auto fetch_tensor_bytes(CacheT& cache, const std::string& name, size_t* out_len, int)
    -> decltype(cache.get_tensor(name, (std::string*)nullptr, out_len)) {
    return cache.get_tensor(name, (std::string*)nullptr, out_len);
}
template <typename CacheT>
static inline const uint8_t* fetch_tensor_bytes(CacheT& cache, const std::string& name, size_t* out_len, long) {
    return cache.get_tensor(name, out_len);
}

// Compute router row L2 norms for a [N_EXP, H] router weight tensor if present.
template <typename CacheT>
static inline bool compute_router_norms(
    CacheT& cache, const std::string& tensor_name, int n_exp, int H, std::vector<double>& out_norms
) {
    if (!cache.has_tensor(tensor_name)) return false;
    size_t nbytes = 0;
    const uint8_t* ptr = fetch_tensor_bytes(cache, tensor_name, &nbytes, 0);
    const size_t elems = (size_t)n_exp * (size_t)H;
    out_norms.assign(n_exp, 0.0);
    if (nbytes == elems * 4ULL) {
        const float* fp = reinterpret_cast<const float*>(ptr);
        for (int e = 0; e < n_exp; ++e) {
            double ss = 0.0;
            for (int c = 0; c < H; ++c) { double v = fp[(size_t)e * H + c]; ss += v * v; }
            out_norms[e] = std::sqrt(ss);
        }
        return true;
    }
    if (nbytes == elems * 2ULL) {
        const uint16_t* bp = reinterpret_cast<const uint16_t*>(ptr);
        for (int e = 0; e < n_exp; ++e) {
            double ss = 0.0;
            for (int c = 0; c < H; ++c) { double v = bf16_to_f32_host(bp[(size_t)e * H + c]); ss += v * v; }
            out_norms[e] = std::sqrt(ss);
        }
        return true;
    }
    return false;
}

// GLM data-free expert ranking using router L2 norm + FP8 block-scale energy.
// Supports both GLM-5.3-Flash ("model.language_model.layers") and GLM-5.3 753B ("model.layers").
template <typename CacheT>
static inline std::vector<int> rank_glm_experts_datafree(
    CacheT& cache, int lyr, int n_exp, int H, bool* used_router = nullptr,
    const char* layer_prefix = "model.language_model.layers"
) {
    static const char* projs[3] = {"gate_proj", "up_proj", "down_proj"};
    std::vector<double> w_energy(n_exp, 0.0);
    for (int e = 0; e < n_exp; ++e) {
        double acc = 0.0;
        for (int p = 0; p < 3; ++p) {
            char nm[256];
            std::snprintf(nm, sizeof(nm), "%s.%d.mlp.experts.%d.%s.weight_scale_inv", layer_prefix, lyr, e, projs[p]);
            if (!cache.has_tensor(nm)) continue;
            size_t ns = 0;
            const uint8_t* sp = fetch_tensor_bytes(cache, nm, &ns, 0);
            const size_t cnt = ns / sizeof(float);
            const float* fp = reinterpret_cast<const float*>(sp);
            double sum2 = 0.0;
            for (size_t i = 0; i < cnt; ++i) { double v = fp[i]; if (std::isfinite(v)) sum2 += v * v; }
            if (cnt > 0) acc += std::sqrt(sum2 / (double)cnt);
        }
        w_energy[e] = acc;
    }

    std::vector<double> r_norms;
    char rname[256];
    std::snprintf(rname, sizeof(rname), "%s.%d.mlp.gate.weight", layer_prefix, lyr);
    bool has_r = compute_router_norms(cache, rname, n_exp, H, r_norms);
    if (used_router) *used_router = has_r;

    auto zw = zscore(w_energy);
    if (!has_r) return scores_to_ranks(zw);
    auto zr = zscore(r_norms);
    std::vector<double> comb(n_exp, 0.0);
    for (int e = 0; e < n_exp; ++e) comb[e] = 0.5 * zr[e] + 0.5 * zw[e];
    return scores_to_ranks(comb);
}

// Qwen3.8-Flash-Next data-free expert ranking using router L2 norm + stacked MLX-Q8 scale energy.
template <typename CacheT>
static inline std::vector<int> rank_flashnext_experts_datafree(
    CacheT& cache, int lyr, bool is_mtp, int n_exp, int H,
    const uint8_t* stacked_s[3], size_t exp_sb_bytes, bool* used_router = nullptr
) {
    std::vector<double> w_energy(n_exp, 0.0);
    const size_t n_sc = exp_sb_bytes / 2ULL;
    for (int e = 0; e < n_exp; ++e) {
        double acc = 0.0;
        for (int p = 0; p < 3; ++p) {
            const uint16_t* sp = reinterpret_cast<const uint16_t*>(stacked_s[p] + (size_t)e * exp_sb_bytes);
            double sum2 = 0.0;
            for (size_t i = 0; i < n_sc; ++i) {
                double v = bf16_to_f32_host(sp[i]);
                if (std::isfinite(v)) sum2 += v * v;
            }
            if (n_sc > 0) acc += std::sqrt(sum2 / (double)n_sc);
        }
        w_energy[e] = acc;
    }

    std::vector<double> r_norms;
    char rname[256];
    if (is_mtp) std::snprintf(rname, sizeof(rname), "language_model.mtp.layers.0.mlp.gate.weight");
    else        std::snprintf(rname, sizeof(rname), "language_model.model.layers.%d.mlp.gate.weight", lyr);
    bool has_r = compute_router_norms(cache, rname, n_exp, H, r_norms);
    if (used_router) *used_router = has_r;

    auto zw = zscore(w_energy);
    if (!has_r) return scores_to_ranks(zw);
    auto zr = zscore(r_norms);
    std::vector<double> comb(n_exp, 0.0);
    for (int e = 0; e < n_exp; ++e) comb[e] = 0.5 * zr[e] + 0.5 * zw[e];
    return scores_to_ranks(comb);
}

} // namespace tfq_moe
