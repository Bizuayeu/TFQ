#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// ============================================================================
// TensorFold TFQ3 (2D-AffineCoset/half4) FlashNext GPU Converter (Apple M5)
//   - Source: Qwen3.8-Flash-Next-MLX-8bit-MTP (H=2560, I=640, E=512, 48L + MTP)
//   - Generates deterministic 16KB pw_v3_tables.bin (bit-identical to GLM TFQ3)
//   - --mode moe : Converts stacked MLX 8-bit/32 switch_mlp -> layer_XX.tffpw / mtp_00.tffpw
//   - --mode requant-ple : Fast GPU MLX 8-bit/32 -> B-bit/32 PLE shard builder (for ablation/packaging)
// ============================================================================

static constexpr uint32_t MAGIC_TFPW = 0x57504654u; // "TFPW"
static constexpr uint32_t MAGIC_PWV3 = 0x33565750u; // "PWV3"
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

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

struct PrismLayerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_idx;
    uint32_t num_experts;
    uint32_t num_tensors;
    uint32_t hidden_size;
    uint32_t moe_inter;
    uint32_t reserved0;
    uint64_t raw_fp8_bytes;
    uint64_t stream_words;
    uint64_t scale_halfs;
    uint64_t off_meta;
    uint64_t off_scales;
    uint64_t off_weights;
    uint64_t reserved1[4];
    uint8_t  meta_tag[32];
};
static_assert(sizeof(PrismLayerHeader) == 144, "Header must be 144 bytes");

struct PrismExpertMeta {
    uint32_t b_bits;
    uint32_t heat_rank;
    uint64_t gate_word_off;
    uint64_t up_word_off;
    uint64_t down_word_off;
    uint32_t gate_sc_off;
    uint32_t up_sc_off;
    uint32_t down_sc_off;
    float    gate_snr_db;
    float    up_snr_db;
    float    down_snr_db;
    uint32_t reserved[2];
};
static_assert(sizeof(PrismExpertMeta) == 64, "ExpertMeta must be 64 bytes");

struct MatParams { uint32_t R, C, b_bits; float shrink; };
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };
struct PleParams { uint32_t rows, cols, dst_bits, pad; };
struct P2 { float x, y; };

static inline uint16_t f32_to_f16(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float f16_to_f32(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
static inline uint32_t pack_h2(float x, float y) { return (uint32_t)f32_to_f16(x) | ((uint32_t)f32_to_f16(y) << 16); }

static double inv_norm_cdf(double p) {
    p = std::min(1.0 - 1e-12, std::max(1e-12, p));
    static const double a[6]={-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
    static const double b[5]={-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
    static const double c[6]={-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
    static const double d[4]={ 7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
    if (p < 0.02425) { double q=sqrt(-2*log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0); }
    else if (p <= 0.97575) { double q=p-0.5,r=q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0); }
    else { double q=sqrt(-2*log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0); }
}

static std::vector<P2> train_2d(int K, int seed, int n_iter) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int NS = 200000;
    std::vector<P2> data(NS);
    for (int i = 0; i < NS; ++i) data[i] = {nd(rng), nd(rng)};
    std::vector<P2> cen(K);
    const double phi2 = 1.3247179572447460, a1 = 1.0/phi2, a2 = 1.0/(phi2*phi2);
    for (int i = 0; i < K; ++i) {
        double u1 = fmod(0.5 + (i+0.5)*a1 + 0.031*seed, 1.0);
        double u2 = fmod(0.5 + (i+0.5)*a2 + 0.067*seed, 1.0);
        cen[i] = {(float)inv_norm_cdf(u1), (float)inv_norm_cdf(u2)};
    }
    std::vector<double> sx(K), sy(K); std::vector<int> cnt(K);
    for (int it = 0; it < n_iter; ++it) {
        std::fill(sx.begin(), sx.end(), 0.0); std::fill(sy.begin(), sy.end(), 0.0); std::fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < NS; ++i) {
            float x = data[i].x, y = data[i].y; int bi = 0; float bd = 1e30f;
            for (int k = 0; k < K; ++k) { float dx=x-cen[k].x, dy=y-cen[k].y, d=dx*dx+dy*dy; if (d < bd) { bd=d; bi=k; } }
            sx[bi] += x; sy[bi] += y; cnt[bi]++;
        }
        for (int k = 0; k < K; ++k) if (cnt[k]) { cen[k].x = (float)(sx[k]/cnt[k]); cen[k].y = (float)(sy[k]/cnt[k]); }
    }
    return cen;
}

static void fill_c64(uint16_t out[64][4], float alpha, float beta) {
    for (int i = 0; i < 64; ++i) {
        float sx = (i & 1) ? -1.0f : 1.0f;
        float sy = (i & 2) ? -1.0f : 1.0f;
        double u0 = fmod((i + 0.5) * 0.6180339887498949, 1.0) * 2.0 - 1.0;
        double u1 = fmod((i + 0.5) * 0.3819660112501051, 1.0) * 2.0 - 1.0;
        double u2 = fmod((i + 0.5) * 0.7548776662466927, 1.0) * 2.0 - 1.0;
        double u3 = fmod((i + 0.5) * 0.5698402909980532, 1.0) * 2.0 - 1.0;
        out[i][0] = f32_to_f16(sx * (1.0f + alpha * (float)u0));
        out[i][1] = f32_to_f16(sy * (1.0f + alpha * (float)u1));
        out[i][2] = f32_to_f16(beta * (float)u2);
        out[i][3] = f32_to_f16(beta * (float)u3);
    }
}

static void load_or_generate_tables(const std::string& path, PwV3TablesFile& tbl) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f) {
        size_t nr = fread(&tbl, 1, sizeof(tbl), f);
        fclose(f);
        if (nr == sizeof(tbl) && tbl.magic == MAGIC_PWV3 && tbl.version == 3u) {
            printf("Loaded existing v3 tables from %s\n", path.c_str());
            return;
        }
    }
    printf("Generating deterministic TFQ3 2D-AffineCoset tables -> %s ...\n", path.c_str());
    fflush(stdout);
    memset(&tbl, 0, sizeof(tbl));
    tbl.magic = MAGIC_PWV3; tbl.version = 3u;
    tbl.a3 = 0.22f; tbl.b3 = 0.38f;
    tbl.a4 = 0.18f; tbl.b4 = 0.30f;
    tbl.a5 = 0.10f; tbl.b5 = 0.18f;
    for (int c = 0; c < 4; ++c) {
        auto t3 = train_2d(64, 10 + c, 40);
        for (int i = 0; i < 64; ++i) tbl.lut3_u32[i * 4 + c] = pack_h2(t3[i].x, t3[i].y);
    }
    for (int c = 0; c < 4; ++c) {
        auto t4 = train_2d(256, 100 + c, 35);
        for (int i = 0; i < 256; ++i) tbl.lut4_u32[i * 4 + c] = pack_h2(t4[i].x, t4[i].y);
    }
    auto t5 = train_2d(1024, 500, 30);
    for (int i = 0; i < 1024; ++i) tbl.lut5_u32[i] = pack_h2(t5[i].x, t5[i].y);
    fill_c64(tbl.c64_3, tbl.a3, tbl.b3);
    fill_c64(tbl.c64_4, tbl.a4, tbl.b4);
    fill_c64(tbl.c64_5, tbl.a5, tbl.b5);
    FILE* fo = fopen(path.c_str(), "wb");
    if (!fo) { fprintf(stderr, "Cannot write %s\n", path.c_str()); exit(1); }
    fwrite(&tbl, 1, sizeof(tbl), fo);
    fclose(fo);
    printf("Saved %s (16,384 bytes)\n", path.c_str());
}

static std::vector<uint16_t> build_grid(const uint32_t* lut_u32, int lut_size, int N_sym, int C_shift, int G, int KC) {
    int N_cb = lut_size / N_sym;
    const float RMAX = 3.2f;
    std::vector<uint16_t> grid((size_t)N_cb * G * G * KC);
    std::vector<P2> pts(lut_size);
    for (int i = 0; i < lut_size; ++i) {
        pts[i] = {f16_to_f32((uint16_t)(lut_u32[i] & 0xFFFFu)), f16_to_f32((uint16_t)(lut_u32[i] >> 16))};
    }
    for (int c = 0; c < N_cb; ++c) {
        for (int gy = 0; gy < G; ++gy) {
            float cy = -RMAX + (gy + 0.5f) * (2.0f * RMAX / G);
            for (int gx = 0; gx < G; ++gx) {
                float cx = -RMAX + (gx + 0.5f) * (2.0f * RMAX / G);
                std::vector<std::pair<float, uint16_t>> dl(N_sym);
                for (int i = 0; i < N_sym; ++i) {
                    int idx = (i << C_shift) | c;
                    float dx = pts[idx].x - cx, dy = pts[idx].y - cy;
                    dl[i] = {dx * dx + dy * dy, (uint16_t)i};
                }
                std::partial_sort(dl.begin(), dl.begin() + KC, dl.end());
                size_t base = (((size_t)c * G + gy) * G + gx) * KC;
                for (int k = 0; k < KC; ++k) grid[base + k] = dl[k].second;
            }
        }
    }
    return grid;
}

struct ShardCache {
    struct Meta { uint64_t off, len; };
    struct Entry {
        int fd = -1; uint8_t* ptr = nullptr; size_t len = 0; uint64_t data_base = 0;
        std::map<std::string, Meta> tensors;
    };
    std::string model_dir;
    std::map<std::string, std::string> weight_map;
    std::map<std::string, Entry> shards;

    void init(const std::string& dir) {
        model_dir = dir;
        std::ifstream f(dir + "/model.safetensors.index.json");
        std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t pos = txt.find("\"weight_map\"");
        if (pos == std::string::npos) { fprintf(stderr, "No weight_map in %s\n", dir.c_str()); exit(1); }
        pos = txt.find('{', pos);
        size_t end = txt.find('}', pos);
        std::string body = txt.substr(pos + 1, end - pos - 1);
        size_t i = 0;
        while (i < body.size()) {
            size_t q1 = body.find('"', i); if (q1 == std::string::npos) break;
            size_t q2 = body.find('"', q1 + 1);
            size_t q3 = body.find('"', q2 + 1);
            size_t q4 = body.find('"', q3 + 1);
            weight_map[body.substr(q1 + 1, q2 - q1 - 1)] = body.substr(q3 + 1, q4 - q3 - 1);
            i = q4 + 1;
        }
    }

    bool has_tensor(const std::string& name) const { return weight_map.count(name) > 0; }

    Entry& get_shard(const std::string& fn) {
        auto it = shards.find(fn);
        if (it != shards.end()) return it->second;
        Entry e;
        std::string full = model_dir + "/" + fn;
        e.fd = open(full.c_str(), O_RDONLY);
        if (e.fd < 0) { fprintf(stderr, "Cannot open %s\n", full.c_str()); exit(1); }
        struct stat st{}; fstat(e.fd, &st); e.len = st.st_size;
        e.ptr = (uint8_t*)mmap(nullptr, e.len, PROT_READ, MAP_PRIVATE, e.fd, 0);
        uint64_t hlen = 0; memcpy(&hlen, e.ptr, 8);
        e.data_base = 8 + hlen;
        std::string hdr((const char*)e.ptr + 8, hlen);
        size_t p = 0;
        while (true) {
            size_t dk = hdr.find("\"data_offsets\"", p);
            if (dk == std::string::npos) break;
            size_t colon_obj = hdr.rfind(":{", dk);
            size_t q2 = hdr.rfind('"', colon_obj);
            size_t q1 = hdr.rfind('"', q2 - 1);
            std::string tname = hdr.substr(q1 + 1, q2 - q1 - 1);
            size_t lb = hdr.find('[', dk), cm = hdr.find(',', lb), rb = hdr.find(']', cm);
            uint64_t o0 = std::stoull(hdr.substr(lb + 1, cm - lb - 1));
            uint64_t o1 = std::stoull(hdr.substr(cm + 1, rb - cm - 1));
            e.tensors[tname] = {o0, o1 - o0};
            p = rb + 1;
        }
        shards[fn] = std::move(e);
        return shards[fn];
    }

    const uint8_t* get_tensor(const std::string& name, size_t* out_len = nullptr) {
        auto it = weight_map.find(name);
        if (it == weight_map.end()) { fprintf(stderr, "Missing tensor %s\n", name.c_str()); exit(1); }
        Entry& sh = get_shard(it->second);
        auto tit = sh.tensors.find(name);
        if (tit == sh.tensors.end()) { fprintf(stderr, "Missing %s in %s\n", name.c_str(), it->second.c_str()); exit(1); }
        if (out_len) *out_len = tit->second.len;
        return sh.ptr + sh.data_base + tit->second.off;
    }

    void evict_all() {
        for (auto& kv : shards) {
            if (kv.second.ptr) munmap(kv.second.ptr, kv.second.len);
            if (kv.second.fd >= 0) close(kv.second.fd);
        }
        shards.clear();
    }

    ~ShardCache() { evict_all(); }
};

static std::map<int, std::vector<int>> load_heat_ranks(const std::string& trace_path, int n_exp) {
    std::map<int, std::vector<long long>> counts;
    std::ifstream f(trace_path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("LAYER ", 0) == 0) {
            int lyr = 0, eid = 0; long long cnt = 0;
            if (sscanf(line.c_str(), "LAYER %d E%d %lld", &lyr, &eid, &cnt) == 3) {
                if (!counts.count(lyr)) counts[lyr].assign(n_exp, 0);
                if (eid >= 0 && eid < n_exp) counts[lyr][eid] += cnt;
            }
        }
    }
    std::map<int, std::vector<int>> ranks;
    for (auto& kv : counts) {
        std::vector<int> order(n_exp); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return (kv.second[a] != kv.second[b]) ? (kv.second[a] > kv.second[b]) : (a < b);
        });
        std::vector<int> r(n_exp);
        for (int idx = 0; idx < n_exp; ++idx) r[order[idx]] = idx;
        ranks[kv.first] = std::move(r);
    }
    return ranks;
}

static void write_safetensors(const std::string& path,
                              const std::vector<std::string>& names,
                              const std::vector<std::string>& dtypes,
                              const std::vector<std::vector<size_t>>& shapes,
                              const std::vector<const void*>& ptrs,
                              const std::vector<size_t>& lens) {
    std::ostringstream hdr;
    hdr << "{";
    size_t off = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) hdr << ",";
        hdr << "\"" << names[i] << "\":{\"dtype\":\"" << dtypes[i] << "\",\"shape\":[";
        for (size_t j = 0; j < shapes[i].size(); ++j) { if (j) hdr << ","; hdr << shapes[i][j]; }
        hdr << "],\"data_offsets\":[" << off << "," << (off + lens[i]) << "]}";
        off += lens[i];
    }
    hdr << "}";
    std::string hstr = hdr.str();
    while ((hstr.size() + 8) % 8 != 0) hstr.push_back(' ');
    uint64_t hlen = hstr.size();
    std::string tmp = path + ".tmp";
    FILE* fo = fopen(tmp.c_str(), "wb");
    if (!fo) { fprintf(stderr, "Cannot write %s\n", tmp.c_str()); exit(1); }
    fwrite(&hlen, 8, 1, fo);
    fwrite(hstr.data(), 1, hlen, fo);
    for (size_t i = 0; i < names.size(); ++i) fwrite(ptrs[i], 1, lens[i], fo);
    fclose(fo);
    rename(tmp.c_str(), path.c_str());
}

static std::string default_metallib(const char* name) {
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

int main(int argc, char** argv) {
    std::string model_dir   = "";
    std::string out_dir     = "";
    std::string trace_path  = "";
    std::string mode        = "moe"; // moe | requant-ple
    std::string metallib    = default_metallib("tfq3-flashnext-convert.metallib");
    std::string tables_path = "";
    int l_start = 0, l_end = 47;
    int n5 = 0, n4 = 192;
    int ple_bits = 3;
    bool do_mtp = false;
    bool skip_existing = false;
    std::string tag = "heat";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { model_dir.clear(); break; }
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--trace" && i + 1 < argc) trace_path = argv[++i];
        else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--metallib" && i + 1 < argc) metallib = argv[++i];
        else if (a == "--tables" && i + 1 < argc) tables_path = argv[++i];
        else if (a == "--layers" && i + 2 < argc) { l_start = atoi(argv[++i]); l_end = atoi(argv[++i]); }
        else if (a == "--n5" && i + 1 < argc) n5 = atoi(argv[++i]);
        else if (a == "--n4" && i + 1 < argc) n4 = atoi(argv[++i]);
        else if (a == "--ple-bits" && i + 1 < argc) ple_bits = atoi(argv[++i]);
        else if (a == "--tag" && i + 1 < argc) tag = argv[++i];
        else if (a == "--mtp") do_mtp = true;
        else if (a == "--skip-existing") skip_existing = true;
    }
    if (model_dir.empty() || out_dir.empty()) {
        fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--mode moe|requant-ple] [--layers S E] [--n5 N] [--n4 N] [--trace <file>] [--mtp] [--ple-bits 3|4] [--skip-existing] [--tables <path>] [--metallib <path>]\n", argv[0]);
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib err: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        auto get_pso = [&](const char* name) {
            NSError* e = nil;
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&e];
            if (!p) { fprintf(stderr, "pso %s err: %s\n", name, [[e localizedDescription] UTF8String]); exit(1); }
            return p;
        };

        ShardCache cache;
        cache.init(model_dir);

        // ====================================================================
        // MODE 1: REQUANT PLE (128 shards, 8-bit/32 -> B-bit/32 safetensors)
        // ====================================================================
        if (mode == "requant-ple") {
            auto pso_ple = get_pso("requant_ple_q8_to_qb");
            const uint32_t ROWS = 2500012u, COLS = 160u, NG = COLS / 32u;
            const size_t src_w_bytes = (size_t)ROWS * COLS;
            const size_t sb_bytes = (size_t)ROWS * NG * 2ULL;
            const size_t dst_w_bytes = (size_t)ROWS * NG * ple_bits * 4ULL;
            id<MTLBuffer> b_sw = [dev newBufferWithLength:src_w_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ss = [dev newBufferWithLength:sb_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sb = [dev newBufferWithLength:sb_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dw = [dev newBufferWithLength:dst_w_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ds = [dev newBufferWithLength:sb_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_db = [dev newBufferWithLength:sb_bytes options:MTLResourceStorageModeShared];

            // Pack 16 PLE shards per safetensors file -> 8 files (ple-00..07.safetensors)
            auto t0 = std::chrono::high_resolution_clock::now();
            printf("REQUANT PLE 8-bit/32 -> %d-bit/32 (128 shards -> 8 files in %s)\n", ple_bits, out_dir.c_str());
            fflush(stdout);
            for (int grp = 0; grp < 8; ++grp) {
                std::vector<std::string> names, dtypes;
                std::vector<std::vector<size_t>> shapes;
                std::vector<std::vector<uint8_t>> storage;
                storage.reserve(16 * 3);
                for (int s = grp * 16; s < grp * 16 + 16; ++s) {
                    char stem[192];
                    snprintf(stem, sizeof(stem), "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d", s);
                    size_t lw = 0, ls = 0, lb = 0;
                    const uint8_t* pw = cache.get_tensor(std::string(stem) + ".weight", &lw);
                    const uint8_t* ps = cache.get_tensor(std::string(stem) + ".scales", &ls);
                    const uint8_t* pb = cache.get_tensor(std::string(stem) + ".biases", &lb);
                    if (lw != src_w_bytes || ls != sb_bytes || lb != sb_bytes) {
                        fprintf(stderr, "Unexpected PLE shard_%d sizes: %zu %zu %zu\n", s, lw, ls, lb);
                        return 2;
                    }
                    memcpy([b_sw contents], pw, src_w_bytes);
                    memcpy([b_ss contents], ps, sb_bytes);
                    memcpy([b_sb contents], pb, sb_bytes);
                    PleParams pp = {ROWS, COLS, (uint32_t)ple_bits, 0u};
                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_ple];
                        [enc setBuffer:b_sw offset:0 atIndex:0];
                        [enc setBuffer:b_ss offset:0 atIndex:1];
                        [enc setBuffer:b_sb offset:0 atIndex:2];
                        [enc setBuffer:b_dw offset:0 atIndex:3];
                        [enc setBuffer:b_ds offset:0 atIndex:4];
                        [enc setBuffer:b_db offset:0 atIndex:5];
                        [enc setBytes:&pp length:sizeof(pp) atIndex:6];
                        uint32_t tg = (ROWS * NG + 255u) / 256u;
                        [enc dispatchThreadgroups:MTLSizeMake(tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                    storage.emplace_back((uint8_t*)[b_dw contents], (uint8_t*)[b_dw contents] + dst_w_bytes);
                    names.push_back(std::string(stem) + ".weight"); dtypes.push_back("U32"); shapes.push_back({ROWS, NG * (size_t)ple_bits});
                    storage.emplace_back((uint8_t*)[b_ds contents], (uint8_t*)[b_ds contents] + sb_bytes);
                    names.push_back(std::string(stem) + ".scales"); dtypes.push_back("BF16"); shapes.push_back({ROWS, NG});
                    storage.emplace_back((uint8_t*)[b_db contents], (uint8_t*)[b_db contents] + sb_bytes);
                    names.push_back(std::string(stem) + ".biases"); dtypes.push_back("BF16"); shapes.push_back({ROWS, NG});
                }
                std::vector<const void*> ptrs(storage.size());
                std::vector<size_t> lens(storage.size());
                for (size_t i = 0; i < storage.size(); ++i) { ptrs[i] = storage[i].data(); lens[i] = storage[i].size(); }
                char out_file[512];
                snprintf(out_file, sizeof(out_file), "%s/ple_b%d_%02d.safetensors", out_dir.c_str(), ple_bits, grp);
                write_safetensors(out_file, names, dtypes, shapes, ptrs, lens);
                cache.evict_all();
                double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
                printf("  wrote %s (%.1fs)\n", out_file, el); fflush(stdout);
            }
            return 0;
        }

        // ====================================================================
        // MODE 2: MOE LAYER CONVERSION (H=2560, I=640, E=512 -> layer_XX.tffpw / mtp_00.tffpw)
        // ====================================================================
        if (tables_path.empty()) tables_path = out_dir + "/pw_v3_tables.bin";
        PwV3TablesFile tbl;
        load_or_generate_tables(tables_path, tbl);

        auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
        auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
        auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

        auto pso_q8_row   = get_pso("q8_32_dequant_row_fwht");
        auto pso_col_fwht = get_pso("col_fwht_and_ss");
        auto pso_pack_nax = get_pso("finalize_scales_and_pack_nax");
        id<MTLComputePipelineState> pso_enc[6] = {};
        pso_enc[3] = get_pso("prism_encode_v3_b3");
        pso_enc[4] = get_pso("prism_encode_v3_b4");
        pso_enc[5] = get_pso("prism_encode_v3_b5");
        auto pso_snr = get_pso("prism_verify_v3_snr");

        id<MTLBuffer> buf_lut[6] = {}, buf_c64[6] = {}, buf_grid[6] = {};
        buf_lut[3]  = [dev newBufferWithBytes:tbl.lut3_u32 length:sizeof(tbl.lut3_u32) options:MTLResourceStorageModeShared];
        buf_lut[4]  = [dev newBufferWithBytes:tbl.lut4_u32 length:sizeof(tbl.lut4_u32) options:MTLResourceStorageModeShared];
        buf_lut[5]  = [dev newBufferWithBytes:tbl.lut5_u32 length:sizeof(tbl.lut5_u32) options:MTLResourceStorageModeShared];
        buf_c64[3]  = [dev newBufferWithBytes:tbl.c64_3    length:sizeof(tbl.c64_3)    options:MTLResourceStorageModeShared];
        buf_c64[4]  = [dev newBufferWithBytes:tbl.c64_4    length:sizeof(tbl.c64_4)    options:MTLResourceStorageModeShared];
        buf_c64[5]  = [dev newBufferWithBytes:tbl.c64_5    length:sizeof(tbl.c64_5)    options:MTLResourceStorageModeShared];
        buf_grid[3] = [dev newBufferWithBytes:grid3.data() length:grid3.size() * 2 options:MTLResourceStorageModeShared];
        buf_grid[4] = [dev newBufferWithBytes:grid4.data() length:grid4.size() * 2 options:MTLResourceStorageModeShared];
        buf_grid[5] = [dev newBufferWithBytes:grid5.data() length:grid5.size() * 2 options:MTLResourceStorageModeShared];

        constexpr int H = 2560, I = 640, N_EXP = 512;
        constexpr size_t MAX_ELEMS = (size_t)H * I; // 1,638,400 elems!
        constexpr uint32_t SC_PER_MAT = (H / 128) * (I / 128); // 20 * 5 = 100!
        // Max hist: max(1280 * 640, 5120 * 160) = 819,200 pair-slots * 256 * 4B = 838.9 MB -> 1 dispatch/matrix!
        constexpr size_t HIST_SLOTS = 819200ULL;

        id<MTLBuffer> buf_q8_w   = [dev newBufferWithLength:MAX_ELEMS options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_q8_s   = [dev newBufferWithLength:(MAX_ELEMS / 32) * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_q8_b   = [dev newBufferWithLength:(MAX_ELEMS / 32) * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_w_tmp  = [dev newBufferWithLength:MAX_ELEMS * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_w_rot  = [dev newBufferWithLength:MAX_ELEMS * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_blk_ss = [dev newBufferWithLength:SC_PER_MAT * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_streams= [dev newBufferWithLength:MAX_ELEMS * 4 options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_hist   = [dev newBufferWithLength:HIST_SLOTS * 256ULL * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_pack   = [dev newBufferWithLength:(size_t)(I / 16) * (H / 128) * 10 * 32 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sc     = [dev newBufferWithLength:SC_PER_MAT * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se     = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];

        auto heat_ranks = trace_path.empty() ? std::map<int, std::vector<int>>{} : load_heat_ranks(trace_path, N_EXP);
        const char* proj_names[3] = {"gate_proj", "up_proj", "down_proj"};
        const int proj_R[3] = {I, I, H};
        const int proj_C[3] = {H, H, I};

        std::vector<int> layer_list;
        if (do_mtp) layer_list.push_back(48);
        else for (int l = l_start; l <= l_end; ++l) layer_list.push_back(l);

        printf("============================================================================================================\n");
        printf("TFQ3 FLASHNEXT MOE CONVERTER | n5=%d n4=%d n3=%d (avg %.3f bpw) | layers=%zu | out=%s\n",
               n5, n4, N_EXP - n5 - n4, (5.0 * n5 + 4.0 * n4 + 3.0 * (N_EXP - n5 - n4)) / N_EXP,
               layer_list.size(), out_dir.c_str());
        printf("============================================================================================================\n");
        fflush(stdout);

        auto t_all = std::chrono::high_resolution_clock::now();
        for (int lyr : layer_list) {
            bool is_mtp = (lyr == 48);
            char out_path[512];
            if (is_mtp) snprintf(out_path, sizeof(out_path), "%s/mtp_00.tffpw", out_dir.c_str());
            else        snprintf(out_path, sizeof(out_path), "%s/layer_%02d.tffpw", out_dir.c_str(), lyr);
            if (skip_existing) {
                struct stat cst{};
                if (stat(out_path, &cst) == 0 && cst.st_size > 500000000LL) {
                    printf("%s SKIP (exists: %.3f GB)\n", is_mtp ? "MTP" : ("L" + std::to_string(lyr)).c_str(), (double)cst.st_size / 1e9);
                    fflush(stdout);
                    continue;
                }
            }
            auto t_lyr = std::chrono::high_resolution_clock::now();
            std::vector<int> ranks(N_EXP);
            if (heat_ranks.count(lyr)) ranks = heat_ranks[lyr];
            else std::iota(ranks.begin(), ranks.end(), 0);

            std::vector<PrismExpertMeta> metas(N_EXP);
            memset(metas.data(), 0, metas.size() * sizeof(PrismExpertMeta));
            uint64_t cur_word = 0; uint32_t cur_sc = 0;
            for (int e = 0; e < N_EXP; ++e) {
                int r = ranks[e];
                uint32_t b = (r < n5) ? 5u : ((r < n5 + n4) ? 4u : 3u);
                metas[e].b_bits = b; metas[e].heat_rank = (uint32_t)r;
                uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
                metas[e].gate_word_off = cur_word; cur_word += wpm;
                metas[e].up_word_off   = cur_word; cur_word += wpm;
                metas[e].down_word_off = cur_word; cur_word += wpm;
                metas[e].gate_sc_off = cur_sc; cur_sc += SC_PER_MAT;
                metas[e].up_sc_off   = cur_sc; cur_sc += SC_PER_MAT;
                metas[e].down_sc_off = cur_sc; cur_sc += SC_PER_MAT;
            }
            std::vector<uint16_t> all_scales(cur_sc);
            std::vector<uint32_t> all_weights(cur_word);
            double snr_sum[6] = {}; int snr_cnt[6] = {};

            // Lookup the 3 stacked tensors for this layer once
            const uint8_t* stacked_w[3] = {};
            const uint8_t* stacked_s[3] = {};
            const uint8_t* stacked_b[3] = {};
            for (int p = 0; p < 3; ++p) {
                char base[256];
                if (is_mtp) snprintf(base, sizeof(base), "language_model.mtp.layers.0.mlp.switch_mlp.%s", proj_names[p]);
                else        snprintf(base, sizeof(base), "language_model.model.layers.%d.mlp.switch_mlp.%s", lyr, proj_names[p]);
                size_t nw = 0, ns = 0, nb = 0;
                stacked_w[p] = cache.get_tensor(std::string(base) + ".weight", &nw);
                stacked_s[p] = cache.get_tensor(std::string(base) + ".scales", &ns);
                stacked_b[p] = cache.get_tensor(std::string(base) + ".biases", &nb);
                if (nw != (size_t)N_EXP * MAX_ELEMS || ns != (size_t)N_EXP * (MAX_ELEMS / 32) * 2 || nb != ns) {
                    fprintf(stderr, "Bad stacked size on %s: w=%zu s=%zu b=%zu\n", base, nw, ns, nb);
                    return 2;
                }
            }

            const size_t exp_w_stride = MAX_ELEMS;
            const size_t exp_sb_stride = (MAX_ELEMS / 32) * 2;
            for (int e = 0; e < N_EXP; ++e) {
                uint32_t b = metas[e].b_bits;
                uint64_t wpm = (uint64_t)(I / 16) * (H / 128) * (b * 2u) * 32u;
                for (int p = 0; p < 3; ++p) {
                    int R = proj_R[p], C = proj_C[p];
                    memcpy([buf_q8_w contents], stacked_w[p] + (size_t)e * exp_w_stride, exp_w_stride);
                    memcpy([buf_q8_s contents], stacked_s[p] + (size_t)e * exp_sb_stride, exp_sb_stride);
                    memcpy([buf_q8_b contents], stacked_b[p] + (size_t)e * exp_sb_stride, exp_sb_stride);
                    memset([buf_blk_ss contents], 0, SC_PER_MAT * 4);
                    memset([buf_se contents], 0, 8);

                    MatParams mp = {(uint32_t)R, (uint32_t)C, b, 1.0f};
                    uint32_t total_streams = (uint32_t)(R / 16) * 32u;

                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                        [enc setComputePipelineState:pso_q8_row];
                        [enc setBuffer:buf_q8_w offset:0 atIndex:0]; [enc setBuffer:buf_q8_s offset:0 atIndex:1];
                        [enc setBuffer:buf_q8_b offset:0 atIndex:2]; [enc setBuffer:buf_w_tmp offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(C / 128, R, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_col_fwht];
                        [enc setBuffer:buf_w_tmp offset:0 atIndex:0]; [enc setBuffer:buf_w_rot offset:0 atIndex:1];
                        [enc setBuffer:buf_blk_ss offset:0 atIndex:2]; [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(C, R / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_pack_nax];
                        [enc setBuffer:buf_w_rot offset:0 atIndex:0]; [enc setBuffer:buf_blk_ss offset:0 atIndex:1];
                        [enc setBuffer:buf_sc offset:0 atIndex:2]; [enc setBuffer:buf_streams offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(R / 16, C / 128, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_enc[b]];
                        EncParams ep = {(uint32_t)R, (uint32_t)C, b, 0u, total_streams};
                        [enc setBuffer:buf_streams offset:0 atIndex:0];
                        [enc setBuffer:buf_lut[b] offset:0 atIndex:1];
                        [enc setBuffer:buf_c64[b] offset:0 atIndex:2];
                        [enc setBuffer:buf_grid[b] offset:0 atIndex:3];
                        [enc setBuffer:buf_hist offset:0 atIndex:4];
                        [enc setBuffer:buf_pack offset:0 atIndex:5];
                        [enc setBytes:&ep length:sizeof(ep) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(total_streams, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                        [enc setComputePipelineState:pso_snr];
                        [enc setBuffer:buf_pack offset:0 atIndex:0]; [enc setBuffer:buf_sc offset:0 atIndex:1];
                        [enc setBuffer:buf_lut[b] offset:0 atIndex:2]; [enc setBuffer:buf_c64[b] offset:0 atIndex:3];
                        [enc setBuffer:buf_w_rot offset:0 atIndex:4]; [enc setBuffer:buf_se offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(R / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }

                    const float* se = (const float*)[buf_se contents];
                    float snr = (float)(10.0 * log10((double)se[0] / std::max(1e-18, (double)se[1])));
                    if (snr < 10.0f || std::isnan(snr)) {
                        fprintf(stderr, "FATAL: Bad SNR %.3f on L%d E%d %s!\n", snr, lyr, e, proj_names[p]);
                        return 2;
                    }
                    snr_sum[b] += snr; snr_cnt[b]++;
                    uint64_t w_off = (p == 0) ? metas[e].gate_word_off : (p == 1 ? metas[e].up_word_off : metas[e].down_word_off);
                    uint32_t s_off = (p == 0) ? metas[e].gate_sc_off   : (p == 1 ? metas[e].up_sc_off   : metas[e].down_sc_off);
                    if (p == 0) metas[e].gate_snr_db = snr; else if (p == 1) metas[e].up_snr_db = snr; else metas[e].down_snr_db = snr;
                    memcpy(all_scales.data() + s_off, [buf_sc contents], SC_PER_MAT * 2);
                    memcpy(all_weights.data() + w_off, [buf_pack contents], wpm * 4);
                }
            }
            cache.evict_all();

            uint64_t off_meta    = sizeof(PrismLayerHeader);
            uint64_t off_scales  = align_up(off_meta + metas.size() * sizeof(PrismExpertMeta), ALIGN_16K);
            uint64_t off_weights = align_up(off_scales + all_scales.size() * 2ULL, ALIGN_16K);
            PrismLayerHeader hdr{};
            hdr.magic = MAGIC_TFPW; hdr.version = FORMAT_VER; hdr.layer_idx = (uint32_t)lyr;
            hdr.num_experts = N_EXP; hdr.num_tensors = N_EXP * 3; hdr.hidden_size = H; hdr.moe_inter = I;
            hdr.raw_fp8_bytes = (uint64_t)N_EXP * 3ULL * H * I;
            hdr.stream_words = all_weights.size(); hdr.scale_halfs = all_scales.size();
            hdr.off_meta = off_meta; hdr.off_scales = off_scales; hdr.off_weights = off_weights;
            snprintf((char*)hdr.meta_tag, 32, "fn_v3_%s", tag.c_str());

            std::string tmp_path = std::string(out_path) + ".tmp";
            FILE* fout = fopen(tmp_path.c_str(), "wb");
            fwrite(&hdr, 1, sizeof(hdr), fout);
            fwrite(metas.data(), sizeof(PrismExpertMeta), metas.size(), fout);
            std::vector<uint8_t> zeros(ALIGN_16K, 0);
            size_t cur = off_meta + metas.size() * sizeof(PrismExpertMeta);
            if (off_scales > cur) fwrite(zeros.data(), 1, off_scales - cur, fout);
            fwrite(all_scales.data(), 2, all_scales.size(), fout);
            cur = off_scales + all_scales.size() * 2ULL;
            if (off_weights > cur) fwrite(zeros.data(), 1, off_weights - cur, fout);
            fwrite(all_weights.data(), 4, all_weights.size(), fout);
            size_t end_pos = off_weights + all_weights.size() * 4ULL;
            size_t pad_end = align_up(end_pos, ALIGN_16K) - end_pos;
            if (pad_end) fwrite(zeros.data(), 1, pad_end, fout);
            fclose(fout);
            rename(tmp_path.c_str(), out_path);

            double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_lyr).count();
            double tot_dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_all).count();
            printf("%s [n5=%d n4=%d] %.3f GB in %4.1fs (tot %5.1fs) | SNR:",
                   is_mtp ? "MTP" : (lyr < 10 ? ("L0" + std::to_string(lyr)).c_str() : ("L" + std::to_string(lyr)).c_str()),
                   n5, n4, (double)(end_pos + pad_end) / 1e9, dt, tot_dt);
            for (int b = 3; b <= 5; ++b) if (snr_cnt[b]) printf(" %db=%.2fdB", b, snr_sum[b]/snr_cnt[b]);
            printf("\n"); fflush(stdout);
        }
    }
    return 0;
}
