#pragma once
// ============================================================================
// Shared TFQ3 Codebook Loader, Canonical Verification, and 2D Spatial Grid
// Builder (used identically by all Apple Silicon Metal & NVIDIA CUDA converters).
// ============================================================================
#include "canonical_tables.hpp"
#include "tfq_constants.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace tfq_tables {

static constexpr uint32_t MAGIC_PWV3 = 0x33565750u; // "PWV3"
static constexpr uint32_t FORMAT_VER = 3u;

struct PwV3TablesFile {
    uint32_t magic;
    uint32_t version;
    float    a3, b3, a4, b4, a5, b5;
    uint32_t pad0[8];
    uint32_t lut3_u32[256];
    uint32_t lut4_u32[1024];
    uint32_t lut5_u32[1024];
    uint16_t c64_3[64][4];
    uint16_t c64_4[64][4];
    uint16_t c64_5[64][4];
    uint8_t  pad1[16384 - 10816];
};
static_assert(sizeof(PwV3TablesFile) == 16384, "PwV3TablesFile must be 16384 bytes");

// Load explicit --tables path, OR use the canonical embedded PWV3 tables and
// save them to `path`. If `explicit_tables` is false and `path` already exists
// with non-canonical bytes, refuse so checkpoints never silently mix codebooks.
static inline void load_or_write_pwv3_tables(const std::string& path, PwV3TablesFile& tbl, bool explicit_tables = false) {
    PwV3TablesFile canon{};
    tfq_canonical::fill_canonical_pwv3(&canon);

    FILE* fi = fopen(path.c_str(), "rb");
    if (fi) {
        size_t n = fread(&tbl, 1, sizeof(tbl), fi);
        fclose(fi);
        if (n != sizeof(tbl) || tbl.magic != MAGIC_PWV3 || tbl.version != FORMAT_VER) {
            fprintf(stderr, "Invalid PWV3 tables file: %s\n", path.c_str());
            std::exit(1);
        }
        if (!explicit_tables && std::memcmp(&tbl, &canon, sizeof(tbl)) != 0) {
            fprintf(stderr, "Tables mismatch at %s: differs from canonical PWV3 tables (pass --tables %s explicitly to override)\n",
                    path.c_str(), path.c_str());
            std::exit(1);
        }
        printf("Loaded %s v3 tables from %s\n", explicit_tables ? "custom" : "canonical", path.c_str());
        return;
    }
    if (explicit_tables) {
        fprintf(stderr, "Cannot open --tables file: %s\n", path.c_str());
        std::exit(1);
    }

    tbl = canon;
    FILE* fo = fopen(path.c_str(), "wb");
    if (!fo) { fprintf(stderr, "Cannot write %s\n", path.c_str()); std::exit(1); }
    if (fwrite(&tbl, 1, sizeof(tbl), fo) != sizeof(tbl) || fclose(fo) != 0) {
        fprintf(stderr, "Write error on %s\n", path.c_str());
        std::exit(1);
    }
    printf("Saved canonical PWV3 tables -> %s (16,384 bytes)\n", path.c_str());
}

// Build deterministic 2D spatial acceleration grid over lut3/lut4/lut5.
// Identical IEEE-754 host arithmetic and strict-weak tie-breaking on all CPUs.
static inline std::vector<uint16_t> build_grid(
    const uint32_t* lut_u32, int lut_size, int N_sym, int C_shift, int G, int KC
) {
    int N_cb = lut_size / N_sym;
    const float RMAX = 3.2f;
    std::vector<uint16_t> grid(static_cast<size_t>(N_cb) * G * G * KC);
    struct P2 { float x, y; };
    std::vector<P2> pts(lut_size);
    for (int i = 0; i < lut_size; ++i) {
        pts[i] = {
            tfq_common::f16_to_f32_exact(static_cast<uint16_t>(lut_u32[i] & 0xFFFFu)),
            tfq_common::f16_to_f32_exact(static_cast<uint16_t>(lut_u32[i] >> 16))
        };
    }
    for (int c = 0; c < N_cb; ++c) {
        for (int gy = 0; gy < G; ++gy) {
            float cy = -RMAX + (gy + 0.5f) * (2.0f * RMAX / G);
            for (int gx = 0; gx < G; ++gx) {
                float cx = -RMAX + (gx + 0.5f) * (2.0f * RMAX / G);
                std::vector<std::pair<float, uint16_t>> dl(N_sym);
                for (int i = 0; i < N_sym; ++i) {
                    int idx = (i << C_shift) | c;
                    float dx = pts[idx].x - cx;
                    float dy = pts[idx].y - cy;
                    dl[i] = {dx * dx + dy * dy, static_cast<uint16_t>(i)};
                }
                std::partial_sort(dl.begin(), dl.begin() + KC, dl.end());
                size_t base = ((static_cast<size_t>(c) * G + gy) * G + gx) * KC;
                for (int k = 0; k < KC; ++k) grid[base + k] = dl[k].second;
            }
        }
    }
    return grid;
}

} // namespace tfq_tables
