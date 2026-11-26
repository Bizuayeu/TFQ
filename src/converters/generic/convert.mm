// ============================================================================
// TensorFold TFQ3 Model-Agnostic Apple Silicon Metal GPU Converter
//   - Converts ANY Safetensors model (BF16, F16, F32, FP8, MLX-8bit)
//   - Zero-pads arbitrary [R, C] shapes cleanly to 128-multiples
//   - Data-free --target-gib / --fit auto sensitivity-based 3b/4b/5b allocator
//   - Optional --quantize-vision, --quantize-drafter, --quantize-embed
// ============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <chrono>
#include <random>
#include "planner.hpp"

using namespace tfq_generic;

static constexpr uint32_t MAGIC_PWV3 = 0x33565750u;
static constexpr uint32_t MAGIC_TFPN = 0x4E504654u;
static constexpr uint32_t FORMAT_VER = 3u;
static constexpr size_t   ALIGN_16K  = 16384;

struct PwV3TablesFile {
    uint32_t magic, version;
    float a3, b3, a4, b4, a5, b5;
    uint32_t pad0[8];
    uint32_t lut3_u32[256];
    uint32_t lut4_u32[1024];
    uint32_t lut5_u32[1024];
    uint16_t c64_3[64][4], c64_4[64][4], c64_5[64][4];
    uint8_t  pad1[16384 - 10816];
};
static_assert(sizeof(PwV3TablesFile) == 16384, "PwV3TablesFile must be 16384 bytes");

struct TfpnHeader {
    uint32_t magic, version, num_tensors, pad0;
    uint64_t off_meta, off_scales, off_weights, total_scale_halfs, total_weight_words, pad1;
};
struct TfpnTensorMeta {
    char name[80];
    uint32_t R, C, b_bits, sc_off;
    uint64_t word_off, word_count;
    float snr_db;
    uint32_t pad[3]; // pad[0]=orig_R, pad[1]=orig_C, pad[2]=flags
};

struct MatParams { uint32_t R, C, b_bits; float shrink; uint32_t r_off, pad0; };
struct EncParams { uint32_t R, C, b_bits, stream_off, n_streams; };
struct P2 { float x, y; };

static inline uint16_t f32_to_f16(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float f16_to_f32(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
static inline uint32_t pack_h2(float x, float y) { return (uint32_t)f32_to_f16(x) | ((uint32_t)f32_to_f16(y) << 16); }

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

static std::vector<P2> train_2d(int K, uint32_t seed, int iters) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int N = 131072;
    std::vector<P2> pts(N);
    for (int i = 0; i < N; ++i) pts[i] = {nd(rng), nd(rng)};
    int g = (int)std::round(std::sqrt((double)K));
    std::vector<P2> cen(K);
    for (int i = 0; i < K; ++i) {
        int gx = i % g, gy = i / g;
        cen[i] = {(float)inv_norm_cdf((gx + 0.5) / g) + 0.03f * nd(rng),
                  (float)inv_norm_cdf((gy + 0.5) / g) + 0.03f * nd(rng)};
    }
    std::vector<double> sx(K), sy(K);
    std::vector<int> cnt(K);
    for (int it = 0; it < iters; ++it) {
        std::fill(sx.begin(), sx.end(), 0.0);
        std::fill(sy.begin(), sy.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < N; ++i) {
            float x = pts[i].x, y = pts[i].y, bd = 1e30f; int bi = 0;
            for (int k = 0; k < K; ++k) {
                float dx = x - cen[k].x, dy = y - cen[k].y, d = dx * dx + dy * dy;
                if (d < bd) { bd = d; bi = k; }
            }
            sx[bi] += x; sy[bi] += y; cnt[bi]++;
        }
        for (int k = 0; k < K; ++k) if (cnt[k] > 0) cen[k] = {(float)(sx[k] / cnt[k]), (float)(sy[k] / cnt[k])};
    }
    return cen;
}

static void fill_c64(uint16_t dst[64][4], float a, float b) {
    for (int c = 0; c < 64; ++c) {
        float t0 = (float)(((c >> 0) & 3) * 2 - 3) / 3.0f;
        float t1 = (float)(((c >> 2) & 3) * 2 - 3) / 3.0f;
        float t2 = (float)(((c >> 4) & 1) * 2 - 1);
        float t3 = (float)(((c >> 5) & 1) * 2 - 1);
        dst[c][0] = f32_to_f16(1.0f + a * t2);
        dst[c][1] = f32_to_f16(1.0f + a * t3);
        dst[c][2] = f32_to_f16(b * t0);
        dst[c][3] = f32_to_f16(b * t1);
    }
}

static void load_or_generate_tables(const std::string& path, PwV3TablesFile& tbl) {
    FILE* fi = fopen(path.c_str(), "rb");
    if (fi) {
        size_t n = fread(&tbl, 1, sizeof(tbl), fi);
        fclose(fi);
        if (n == sizeof(tbl) && tbl.magic == MAGIC_PWV3 && tbl.version == FORMAT_VER) return;
    }
    printf("Generating deterministic TFQ3 2D-AffineCoset tables -> %s ...\n", path.c_str());
    memset(&tbl, 0, sizeof(tbl));
    tbl.magic = MAGIC_PWV3; tbl.version = FORMAT_VER;
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
}

static std::vector<uint16_t> build_grid(const uint32_t* lut_u32, int lut_size, int N_sym, int C_shift, int G, int KC) {
    int N_cb = lut_size / N_sym;
    const float RMAX = 3.2f;
    std::vector<uint16_t> grid((size_t)N_cb * G * G * KC);
    std::vector<P2> pts(lut_size);
    for (int i = 0; i < lut_size; ++i) pts[i] = {f16_to_f32((uint16_t)(lut_u32[i] & 0xFFFFu)), f16_to_f32((uint16_t)(lut_u32[i] >> 16))};
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

static void write_tfpn_shard(const std::string& path, const std::vector<TfpnTensorMeta>& metas,
                             const std::vector<uint16_t>& scales, const std::vector<uint32_t>& weights) {
    const size_t off_meta = sizeof(TfpnHeader);
    const size_t off_scales = (size_t)align16k(off_meta + metas.size() * sizeof(TfpnTensorMeta));
    const size_t off_weights = (size_t)align16k(off_scales + scales.size() * 2ULL);
    TfpnHeader hdr{};
    hdr.magic = MAGIC_TFPN; hdr.version = FORMAT_VER; hdr.num_tensors = (uint32_t)metas.size();
    hdr.off_meta = off_meta; hdr.off_scales = off_scales; hdr.off_weights = off_weights;
    hdr.total_scale_halfs = scales.size(); hdr.total_weight_words = weights.size();
    std::vector<uint8_t> zeros(ALIGN_16K, 0);
    FILE* fout = fopen(path.c_str(), "wb");
    if (!fout) { fprintf(stderr, "Cannot write %s\n", path.c_str()); exit(1); }
    fwrite(&hdr, 1, sizeof(hdr), fout);
    fwrite(metas.data(), sizeof(TfpnTensorMeta), metas.size(), fout);
    size_t p1 = off_scales - (off_meta + metas.size() * sizeof(TfpnTensorMeta));
    if (p1) fwrite(zeros.data(), 1, p1, fout);
    fwrite(scales.data(), 2, scales.size(), fout);
    size_t p2 = off_weights - (off_scales + scales.size() * 2ULL);
    if (p2) fwrite(zeros.data(), 1, p2, fout);
    fwrite(weights.data(), 4, weights.size(), fout);
    size_t end_pos = off_weights + weights.size() * 4ULL;
    size_t pe = (size_t)align16k(end_pos) - end_pos;
    if (pe) fwrite(zeros.data(), 1, pe, fout);
    fclose(fout);
}

static std::string default_metallib(const char* name) {
    uint32_t sz = 0; _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        GenericOptions opt;
        std::string metallib = default_metallib("tfq3-generic-convert.metallib");
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--metallib" && i + 1 < argc) metallib = argv[++i];
        }
        if (!parse_generic_args(argc, argv, opt)) {
            fprintf(stderr, "Usage: %s --model <dir> --out <dir> [--profile 4b|3.5b|3b|5b] [--bits 3|4|5] [--head-bits 0|4|5] [--target-gib GB | --fit auto] [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run] [--tables <path>] [--metallib <path>]\n", argv[0]);
            return 1;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Apple Metal GPU available\n"); return 1; }

        if (opt.fit_auto && opt.target_gib <= 0.0) {
            double ws_gib = (double)dev.recommendedMaxWorkingSetSize / (double)(1ULL << 30);
            double reserve = std::max(4.0, ws_gib * 0.22);
            opt.target_gib = std::max(2.0, ws_gib - reserve);
            printf("Auto-fit Metal GPU (%s): working set %.1f GiB -> weight budget %.2f GiB\n",
                   [[dev name] UTF8String], ws_gib, opt.target_gib);
        }

        SafetensorsDir st;
        st.init(opt.model_dir);
        ConversionPlan plan = build_plan(st, opt);
        print_plan_summary(opt, plan);
        if (plan.jobs.empty()) {
            fprintf(stderr, "No quantizable 2D >=128x128 tensors found in %s\n", opt.model_dir.c_str());
            return 1;
        }
        if (opt.dry_run) return 0;

        mkdir(opt.out_dir.c_str(), 0755);
        std::string dst_tables = opt.out_dir + "/pw_v3_tables.bin";
        if (opt.tables_path.empty()) opt.tables_path = dst_tables;
        PwV3TablesFile tbl;
        load_or_generate_tables(opt.tables_path, tbl);
        if (opt.tables_path != dst_tables) copy_file_if_exists(opt.tables_path, dst_tables);

        auto grid3 = build_grid(tbl.lut3_u32,  256,   64, 2, 16,  8);
        auto grid4 = build_grid(tbl.lut4_u32, 1024,  256, 2, 16, 10);
        auto grid5 = build_grid(tbl.lut5_u32, 1024, 1024, 0, 32, 12);

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib error (%s): %s\n", metallib.c_str(), [[err localizedDescription] UTF8String]); return 1; }
        auto mk_pso = [&](const char* fn) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
            if (!f) { fprintf(stderr, "Missing Metal kernel %s\n", fn); exit(1); }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
            if (!p) { fprintf(stderr, "PSO error %s: %s\n", fn, [[err localizedDescription] UTF8String]); exit(1); }
            return p;
        };
        id<MTLComputePipelineState> pso_row  = mk_pso("f32_row_fwht");
        id<MTLComputePipelineState> pso_col  = mk_pso("col_fwht_and_ss");
        id<MTLComputePipelineState> pso_pack = mk_pso("finalize_scales_and_pack_nax");
        id<MTLComputePipelineState> pso_e[6] = {};
        pso_e[3] = mk_pso("prism_encode_v3_b3");
        pso_e[4] = mk_pso("prism_encode_v3_b4");
        pso_e[5] = mk_pso("prism_encode_v3_b5");
        id<MTLComputePipelineState> pso_ver  = mk_pso("prism_verify_v3_snr");
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        id<MTLBuffer> buf_lut[6] = {}, buf_c64[6] = {}, buf_grid[6] = {};
        buf_lut[3]  = [dev newBufferWithBytes:tbl.lut3_u32  length:sizeof(tbl.lut3_u32) options:MTLResourceStorageModeShared];
        buf_lut[4]  = [dev newBufferWithBytes:tbl.lut4_u32  length:sizeof(tbl.lut4_u32) options:MTLResourceStorageModeShared];
        buf_lut[5]  = [dev newBufferWithBytes:tbl.lut5_u32  length:sizeof(tbl.lut5_u32) options:MTLResourceStorageModeShared];
        buf_c64[3]  = [dev newBufferWithBytes:tbl.c64_3     length:sizeof(tbl.c64_3)    options:MTLResourceStorageModeShared];
        buf_c64[4]  = [dev newBufferWithBytes:tbl.c64_4     length:sizeof(tbl.c64_4)    options:MTLResourceStorageModeShared];
        buf_c64[5]  = [dev newBufferWithBytes:tbl.c64_5     length:sizeof(tbl.c64_5)    options:MTLResourceStorageModeShared];
        buf_grid[3] = [dev newBufferWithBytes:grid3.data()  length:grid3.size() * 2     options:MTLResourceStorageModeShared];
        buf_grid[4] = [dev newBufferWithBytes:grid4.data()  length:grid4.size() * 2     options:MTLResourceStorageModeShared];
        buf_grid[5] = [dev newBufferWithBytes:grid5.data()  length:grid5.size() * 2     options:MTLResourceStorageModeShared];

        uint32_t max_pad_C = 128u;
        for (const auto& j : plan.jobs) max_pad_C = std::max(max_pad_C, j.pad_C);
        const uint32_t MAX_CHUNK_R = 4096u;
        const size_t MAX_ELEMS = (size_t)MAX_CHUNK_R * max_pad_C;
        const uint32_t CHUNK_STR = 512u;
        const size_t MAX_HIST = (size_t)CHUNK_STR * (max_pad_C / 4u) * 256ULL * 4ULL;
        const size_t MAX_OUTW = (size_t)(MAX_CHUNK_R / 16u) * (max_pad_C / 128u) * 10ULL * 32ULL * 4ULL;

        id<MTLBuffer> buf_in   = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_tmp  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_rot  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_str  = [dev newBufferWithLength:MAX_ELEMS * 4ULL options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_ss   = [dev newBufferWithLength:(size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u) * 128ULL * 4ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_sc16 = [dev newBufferWithLength:(size_t)(MAX_CHUNK_R / 128u) * (max_pad_C / 128u) * 2ULL options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_hist = [dev newBufferWithLength:MAX_HIST options:MTLResourceStorageModePrivate];
        id<MTLBuffer> buf_outw = [dev newBufferWithLength:MAX_OUTW options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_se   = [dev newBufferWithLength:(size_t)(MAX_CHUNK_R / 16u) * 2ULL * 4ULL options:MTLResourceStorageModeShared];

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> host_chunk;
        std::vector<std::string> shard_names;
        double snr_sum[6] = {};
        uint64_t snr_cnt[6] = {};

        for (size_t si = 0; si < plan.shards.size(); ++si) {
            const auto& idxs = plan.shards[si];
            std::vector<TfpnTensorMeta> metas;
            metas.reserve(idxs.size());
            std::vector<uint16_t> all_sc;
            std::vector<uint32_t> all_w;

            for (size_t ji : idxs) {
                auto& j = plan.jobs[ji];
                const float shrink = (j.bits == 5u) ? 0.96f : ((j.bits == 4u) ? 0.94f : 0.92f);
                const uint32_t R = j.pad_R, C = j.pad_C, B = j.bits;
                const uint32_t n_sc_tot = (R / 128u) * (C / 128u);
                const uint64_t n_w_tot  = (uint64_t)(R / 16u) * (C / 128u) * (B * 2u) * 32ULL;

                TfpnTensorMeta m{};
                strncpy(m.name, j.tfpn_name.c_str(), 79);
                m.R = R; m.C = C; m.b_bits = B;
                m.sc_off = (uint32_t)all_sc.size();
                m.word_off = all_w.size();
                m.word_count = n_w_tot;
                m.pad[0] = j.orig_R;
                m.pad[1] = j.orig_C;
                m.pad[2] = j.flags;

                all_sc.resize(all_sc.size() + n_sc_tot);
                all_w.resize(all_w.size() + n_w_tot);

                double tot_sig = 0.0, tot_err = 0.0;
                uint32_t sc_cur = 0;
                uint64_t w_cur = 0;
                for (uint32_t r0 = 0; r0 < R; r0 += MAX_CHUNK_R) {
                    uint32_t rc = std::min(MAX_CHUNK_R, R - r0);
                    decode_chunk_to_f32(st, j, r0, rc, host_chunk);
                    memcpy([buf_in contents], host_chunk.data(), host_chunk.size() * 4ULL);

                    uint32_t n_blk = (rc / 128u) * (C / 128u);
                    MatParams mp{rc, C, B, shrink, r0, 0};

                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_row];
                        [enc setBuffer:buf_in offset:0 atIndex:0]; [enc setBuffer:buf_tmp offset:0 atIndex:1];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(C / 128u, rc, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_col];
                        [enc setBuffer:buf_tmp offset:0 atIndex:0]; [enc setBuffer:buf_rot offset:0 atIndex:1];
                        [enc setBuffer:buf_ss offset:0 atIndex:2]; [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(C, rc / 128u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        [enc setComputePipelineState:pso_pack];
                        [enc setBuffer:buf_rot offset:0 atIndex:0]; [enc setBuffer:buf_ss offset:0 atIndex:1];
                        [enc setBuffer:buf_sc16 offset:0 atIndex:2]; [enc setBuffer:buf_str offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(rc / 16u, C / 128u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        uint32_t tot_str = (rc / 16u) * 32u;
                        for (uint32_t s0 = 0; s0 < tot_str; s0 += CHUNK_STR) {
                            uint32_t ns = std::min(CHUNK_STR, tot_str - s0);
                            EncParams ep{rc, C, B, s0, ns};
                            [enc setComputePipelineState:pso_e[B]];
                            [enc setBuffer:buf_str offset:0 atIndex:0];
                            [enc setBuffer:buf_lut[B] offset:0 atIndex:1];
                            [enc setBuffer:buf_c64[B] offset:0 atIndex:2];
                            [enc setBuffer:buf_grid[B] offset:0 atIndex:3];
                            [enc setBuffer:buf_hist offset:0 atIndex:4];
                            [enc setBuffer:buf_outw offset:0 atIndex:5];
                            [enc setBytes:&ep length:sizeof(ep) atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake(ns, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }

                        [enc setComputePipelineState:pso_ver];
                        [enc setBuffer:buf_outw offset:0 atIndex:0]; [enc setBuffer:buf_sc16 offset:0 atIndex:1];
                        [enc setBuffer:buf_lut[B] offset:0 atIndex:2]; [enc setBuffer:buf_c64[B] offset:0 atIndex:3];
                        [enc setBuffer:buf_rot offset:0 atIndex:4]; [enc setBuffer:buf_se offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(rc / 16u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }

                    const float* se = (const float*)[buf_se contents];
                    uint32_t n_strips = rc / 16u;
                    for (uint32_t si_s = 0; si_s < n_strips; ++si_s) {
                        tot_sig += (double)se[si_s * 2u + 0u];
                        tot_err += (double)se[si_s * 2u + 1u];
                    }
                    uint64_t cw = (uint64_t)(rc / 16u) * (C / 128u) * (B * 2u) * 32ULL;
                    memcpy(all_sc.data() + m.sc_off + sc_cur, [buf_sc16 contents], n_blk * 2ULL);
                    memcpy(all_w.data() + m.word_off + w_cur, [buf_outw contents], cw * 4ULL);
                    sc_cur += n_blk; w_cur += cw;
                }
                m.snr_db = (tot_sig > 0.0) ? (float)(10.0 * std::log10(tot_sig / std::max(1e-18, tot_err))) : 0.0f;
                j.snr_db = m.snr_db;
                snr_sum[B] += m.snr_db; snr_cnt[B]++;
                metas.push_back(m);
            }

            char fn[96];
            snprintf(fn, sizeof(fn), "tfq3-%05zu-of-%05zu.tffpw", si + 1, plan.shards.size());
            shard_names.push_back(fn);
            write_tfpn_shard(opt.out_dir + "/" + fn, metas, all_sc, all_w);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("  wrote %s (%zu tensors, %.3f GiB) [%.1fs]\n",
                   fn, metas.size(), (double)(all_w.size() * 4ULL + all_sc.size() * 2ULL) / (double)(1ULL << 30), dt);
            fflush(stdout);
        }

        package_residuals_and_manifest(st, opt, plan, shard_names);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("Done in %.1fs | SNR:", dt);
        for (int b = 3; b <= 5; ++b) if (snr_cnt[b]) printf(" %db=%.2fdB(%llu)", b, snr_sum[b] / snr_cnt[b], (unsigned long long)snr_cnt[b]);
        printf("\n");
        return 0;
    }
}
