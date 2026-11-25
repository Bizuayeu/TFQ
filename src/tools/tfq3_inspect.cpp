// Zero-dependency TFQ3 binary shard validator and inspector.
// Supports pw_v3_tables.bin ("PWV3"), dense/non-MoE shards ("TFPN"), and MoE shards ("TFPW").
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

static constexpr uint32_t MAGIC_PWV3 = 0x33565750u;
static constexpr uint32_t MAGIC_TFPN = 0x4E504654u;
static constexpr uint32_t MAGIC_TFPW = 0x57504654u;
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr uint64_t ALIGN_16K  = 16384ULL;

struct TfpnHeader {
    uint32_t magic, version, num_tensors, pad0;
    uint64_t off_meta, off_scales, off_weights, total_scale_halfs, total_weight_words, pad1;
};

struct TfpnTensorMeta {
    char name[80];
    uint32_t R, C, b_bits, sc_off;
    uint64_t word_off, word_count;
    float snr_db;
    uint32_t pad[3];
};

struct TfpwHeader {
    uint32_t magic, version, layer_idx, num_experts, num_tensors, hidden_size, moe_inter, reserved0;
    uint64_t raw_fp8_bytes, stream_words, scale_halfs, off_meta, off_scales, off_weights;
    uint64_t reserved1[4];
    char meta_tag[32];
};
static_assert(sizeof(TfpwHeader) == 144, "TfpwHeader must be 144 bytes");

struct TfpwExpertMeta {
    uint32_t b_bits, heat_rank;
    uint64_t gate_word_off, up_word_off, down_word_off;
    uint32_t gate_sc_off, up_sc_off, down_sc_off;
    float gate_snr_db, up_snr_db, down_snr_db;
    uint32_t reserved[2];
};
static_assert(sizeof(TfpwExpertMeta) == 64, "TfpwExpertMeta must be 64 bytes");

static uint64_t file_size(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_size;
}

static bool is_dir(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool inspect_tables(const std::string& path) {
    uint64_t sz = file_size(path);
    if (sz < 10816ULL) { fprintf(stderr, "FAIL %s: truncated (%llu bytes)\n", path.c_str(), (unsigned long long)sz); return false; }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    uint32_t hdr[2];
    float ab[6];
    fread(hdr, 4, 2, f);
    fread(ab, 4, 6, f);
    fclose(f);
    if (hdr[0] != MAGIC_PWV3 || hdr[1] != FORMAT_VER) {
        fprintf(stderr, "FAIL %s: bad magic/version (0x%08x v%u)\n", path.c_str(), hdr[0], hdr[1]);
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(ab[i])) { fprintf(stderr, "FAIL %s: non-finite table scale\n", path.c_str()); return false; }
    }
    printf("OK   %-42s PWV3 v%u | %llu bytes | 3b(a=%.3f,b=%.3f) 4b(a=%.3f,b=%.3f) 5b(a=%.3f,b=%.3f)\n",
           path.c_str(), hdr[1], (unsigned long long)sz, ab[0], ab[1], ab[2], ab[3], ab[4], ab[5]);
    return true;
}

static bool inspect_tfpn(const std::string& path, bool verbose) {
    uint64_t sz = file_size(path);
    if (sz < sizeof(TfpnHeader)) { fprintf(stderr, "FAIL %s: too small\n", path.c_str()); return false; }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    TfpnHeader h{};
    fread(&h, sizeof(h), 1, f);
    if (h.magic != MAGIC_TFPN || h.version != FORMAT_VER) {
        fclose(f); fprintf(stderr, "FAIL %s: bad magic/version\n", path.c_str()); return false;
    }
    if ((h.off_scales % ALIGN_16K) != 0 || (h.off_weights % ALIGN_16K) != 0) {
        fclose(f); fprintf(stderr, "FAIL %s: scales/weights not 16 KiB aligned\n", path.c_str()); return false;
    }
    uint64_t meta_end = h.off_meta + (uint64_t)h.num_tensors * sizeof(TfpnTensorMeta);
    uint64_t sc_end   = h.off_scales + h.total_scale_halfs * 2ULL;
    uint64_t w_end    = h.off_weights + h.total_weight_words * 4ULL;
    if (meta_end > h.off_scales || sc_end > h.off_weights || w_end > sz) {
        fclose(f); fprintf(stderr, "FAIL %s: section overlap or truncation\n", path.c_str()); return false;
    }
    std::vector<TfpnTensorMeta> metas(h.num_tensors);
    fseek(f, (long)h.off_meta, SEEK_SET);
    fread(metas.data(), sizeof(TfpnTensorMeta), h.num_tensors, f);
    fclose(f);

    uint64_t params_total = 0;
    uint64_t cnt[6] = {};
    double snr_sum[6] = {};
    for (const auto& m : metas) {
        if ((m.b_bits != 3u && m.b_bits != 4u && m.b_bits != 5u) || m.R == 0 || m.C == 0 || (m.R % 128u) || (m.C % 128u)) {
            fprintf(stderr, "FAIL %s: invalid tensor %.80s [%u,%u] %ub\n", path.c_str(), m.name, m.R, m.C, m.b_bits);
            return false;
        }
        uint64_t expect_words = (uint64_t)(m.R / 16u) * (m.C / 128u) * (m.b_bits * 2u) * 32ULL;
        uint64_t expect_sc = (uint64_t)(m.R / 128u) * (m.C / 128u);
        if (m.word_count != expect_words || m.word_off + m.word_count > h.total_weight_words || (uint64_t)m.sc_off + expect_sc > h.total_scale_halfs) {
            fprintf(stderr, "FAIL %s: out-of-bounds tensor %.80s\n", path.c_str(), m.name);
            return false;
        }
        params_total += (uint64_t)m.R * m.C;
        cnt[m.b_bits]++;
        snr_sum[m.b_bits] += m.snr_db;
        if (verbose) {
            printf("  %-52.80s [%6u, %5u] %ub  SNR=%5.2f dB\n", m.name, m.R, m.C, m.b_bits, m.snr_db);
        }
    }
    double bpw = params_total ? (double)(h.total_weight_words * 32ULL + h.total_scale_halfs * 16ULL) / (double)params_total : 0.0;
    printf("OK   %-42s TFPN v%u | %3u tensors | %6.3f GiB | %.3f bpw | 3b:%llu(%.2fdB) 4b:%llu(%.2fdB) 5b:%llu(%.2fdB)\n",
           path.c_str(), h.version, h.num_tensors, (double)sz / (double)(1ULL << 30), bpw,
           (unsigned long long)cnt[3], cnt[3] ? snr_sum[3] / cnt[3] : 0.0,
           (unsigned long long)cnt[4], cnt[4] ? snr_sum[4] / cnt[4] : 0.0,
           (unsigned long long)cnt[5], cnt[5] ? snr_sum[5] / cnt[5] : 0.0);
    return true;
}

static bool inspect_tfpw(const std::string& path) {
    uint64_t sz = file_size(path);
    if (sz < sizeof(TfpwHeader)) { fprintf(stderr, "FAIL %s: too small\n", path.c_str()); return false; }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    TfpwHeader h{};
    fread(&h, sizeof(h), 1, f);
    if (h.magic != MAGIC_TFPW || h.version != FORMAT_VER) {
        fclose(f); fprintf(stderr, "FAIL %s: bad magic/version\n", path.c_str()); return false;
    }
    if ((h.off_scales % ALIGN_16K) != 0 || (h.off_weights % ALIGN_16K) != 0) {
        fclose(f); fprintf(stderr, "FAIL %s: scales/weights not 16 KiB aligned\n", path.c_str()); return false;
    }
    uint64_t meta_end = h.off_meta + (uint64_t)h.num_experts * sizeof(TfpwExpertMeta);
    uint64_t sc_end   = h.off_scales + h.scale_halfs * 2ULL;
    uint64_t w_end    = h.off_weights + h.stream_words * 4ULL;
    if (meta_end > h.off_scales || sc_end > h.off_weights || w_end > sz) {
        fclose(f); fprintf(stderr, "FAIL %s: section overlap or truncation\n", path.c_str()); return false;
    }
    std::vector<TfpwExpertMeta> metas(h.num_experts);
    fseek(f, (long)h.off_meta, SEEK_SET);
    fread(metas.data(), sizeof(TfpwExpertMeta), h.num_experts, f);
    fclose(f);

    uint64_t cnt[6] = {};
    double snr_sum[6] = {};
    for (const auto& m : metas) {
        if (m.b_bits != 3u && m.b_bits != 4u && m.b_bits != 5u) {
            fprintf(stderr, "FAIL %s: invalid expert bits (%u)\n", path.c_str(), m.b_bits);
            return false;
        }
        uint64_t gu_words = (uint64_t)(h.moe_inter / 16u) * (h.hidden_size / 128u) * (m.b_bits * 2u) * 32ULL;
        uint64_t dn_words = (uint64_t)(h.hidden_size / 16u) * (h.moe_inter / 128u) * (m.b_bits * 2u) * 32ULL;
        if (m.gate_word_off + gu_words > h.stream_words || m.up_word_off + gu_words > h.stream_words || m.down_word_off + dn_words > h.stream_words) {
            fprintf(stderr, "FAIL %s: expert weight out of bounds\n", path.c_str());
            return false;
        }
        cnt[m.b_bits]++;
        snr_sum[m.b_bits] += (m.gate_snr_db + m.up_snr_db + m.down_snr_db) / 3.0;
    }
    uint64_t params = (uint64_t)h.num_experts * 3ULL * h.hidden_size * h.moe_inter;
    double bpw = params ? (double)(h.stream_words * 32ULL + h.scale_halfs * 16ULL) / (double)params : 0.0;
    printf("OK   %-42s TFPW v%u | L%02u E=%u H=%u I=%u | %6.3f GiB | %.3f bpw | 3b:%llu(%.2fdB) 4b:%llu(%.2fdB) 5b:%llu(%.2fdB)\n",
           path.c_str(), h.version, h.layer_idx, h.num_experts, h.hidden_size, h.moe_inter,
           (double)sz / (double)(1ULL << 30), bpw,
           (unsigned long long)cnt[3], cnt[3] ? snr_sum[3] / cnt[3] : 0.0,
           (unsigned long long)cnt[4], cnt[4] ? snr_sum[4] / cnt[4] : 0.0,
           (unsigned long long)cnt[5], cnt[5] ? snr_sum[5] / cnt[5] : 0.0);
    return true;
}

static bool inspect_file(const std::string& path, bool verbose) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL %s: cannot open\n", path.c_str()); return false; }
    uint32_t magic = 0;
    size_t n = fread(&magic, 4, 1, f);
    fclose(f);
    if (n != 1) { fprintf(stderr, "FAIL %s: empty file\n", path.c_str()); return false; }
    if (magic == MAGIC_PWV3) return inspect_tables(path);
    if (magic == MAGIC_TFPN) return inspect_tfpn(path, verbose);
    if (magic == MAGIC_TFPW) return inspect_tfpw(path);
    fprintf(stderr, "FAIL %s: unknown magic 0x%08x\n", path.c_str(), magic);
    return false;
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::vector<std::string> targets;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") { targets.clear(); break; }
        else targets.push_back(a);
    }
    if (targets.empty()) {
        fprintf(stderr, "Usage: %s [-v|--verbose] <shard.tffpw | pw_v3_tables.bin | checkpoint_dir> [...]\n", argv[0]);
        return 1;
    }
    bool all_ok = true;
    for (const auto& t : targets) {
        if (is_dir(t)) {
            DIR* d = opendir(t.c_str());
            if (!d) { fprintf(stderr, "FAIL %s: cannot open dir\n", t.c_str()); return 1; }
            std::vector<std::string> files;
            while (auto* ent = readdir(d)) {
                std::string nm = ent->d_name;
                if (nm == "pw_v3_tables.bin" || (nm.size() > 6 && nm.substr(nm.size() - 6) == ".tffpw")) {
                    files.push_back(t + "/" + nm);
                }
            }
            closedir(d);
            std::sort(files.begin(), files.end());
            if (files.empty()) { fprintf(stderr, "FAIL %s: no pw_v3_tables.bin or .tffpw files found\n", t.c_str()); return 1; }
            for (const auto& fp : files) all_ok &= inspect_file(fp, verbose);
        } else {
            all_ok &= inspect_file(t, verbose);
        }
    }
    return all_ok ? 0 : 1;
}
