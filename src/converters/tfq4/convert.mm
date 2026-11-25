// ============================================================================
// TFQ4-E8P (2.001 bpw) Model-Agnostic Apple Silicon Metal GPU Converter
//   Converts any Safetensors checkpoint (BF16/F16/F32/FP8/MLX-Q8, padded [R,C])
//   into 16 KiB-aligned cross-platform TFP4 v4 + PWV4 v4 shards.
// ============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../generic/planner.hpp"
#include "tfq4_format.hpp"

namespace fs = std::filesystem;

struct Tfq4MatParams { uint32_t R, C; };

static std::string default_metallib(const char* name) {
    uint32_t sz = 0; _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(&exe[0], &sz) == 0) {
        size_t sl = exe.find_last_of('/');
        if (sl != std::string::npos) return exe.substr(0, sl + 1) + name;
    }
    return name;
}

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

int main(int argc, char** argv) {
    @autoreleasepool {
        tfq_generic::GenericOptions opt;
        opt.default_bits = 3;
        opt.head_bits = 4;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if ((a == "--model" || a == "-m") && i + 1 < argc) opt.model_dir = argv[++i];
            else if ((a == "--out" || a == "-o") && i + 1 < argc) opt.out_dir = argv[++i];
            else if (a == "--quantize-vision") opt.quantize_vision = true;
            else if (a == "--quantize-drafter") opt.quantize_drafter = true;
            else if (a == "--quantize-embed") opt.quantize_embed = true;
            else if (a == "--head-bits" && i + 1 < argc) opt.head_bits = (uint32_t)std::stoul(argv[++i]);
            else if (a == "--dry-run") opt.dry_run = true;
            else if (a == "-h" || a == "--help") {
                printf("Usage: tfq4-convert --model DIR --out DIR [--quantize-vision] [--quantize-drafter] [--quantize-embed] [--dry-run]\n");
                return 0;
            }
        }
        if (opt.model_dir.empty() || opt.out_dir.empty()) {
            fprintf(stderr, "Usage: tfq4-convert --model DIR --out DIR [--quantize-vision] [--quantize-drafter] [--quantize-embed]\n");
            return 2;
        }

        tfq_generic::SafetensorsDir st;
        st.init(opt.model_dir);
        auto plan = tfq_generic::build_plan(st, opt);
        if (plan.jobs.empty()) { fprintf(stderr, "No quantizable 2D float tensors in %s\n", opt.model_dir.c_str()); return 1; }

        uint64_t est_tfq4_bytes = 0;
        for (auto& j : plan.jobs) {
            est_tfq4_bytes += (uint64_t)j.pad_R * (j.pad_C / 16u) * 4ull + (uint64_t)(j.pad_R / 128u) * (j.pad_C / 128u) * 2ull;
        }
        printf("TFQ4-E8P (2.001 bpw) Plan: %zu quantized tensors (%.3f GiB) + %zu residual (%.3f GiB) = %.3f GiB total\n",
               plan.jobs.size(), est_tfq4_bytes / (double)(1ULL << 30),
               plan.residual_names.size(), plan.residual_bytes / (double)(1ULL << 30),
               (est_tfq4_bytes + plan.residual_bytes) / (double)(1ULL << 30));
        if (opt.dry_run) return 0;

        auto t_start = std::chrono::steady_clock::now();
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Apple Metal device\n"); return 1; }
        NSError* err = nil;
        std::string mpath = default_metallib("tfq4-convert.metallib");
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:mpath.c_str()]] error:&err];
        if (!lib) { fprintf(stderr, "metallib error (%s): %s\n", mpath.c_str(), [[err localizedDescription] UTF8String]); return 1; }
        auto mk = [&](const char* nm) {
            return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:nm]] error:&err];
        };
        id<MTLComputePipelineState> p_rfwht = mk("f32_row_fwht");
        id<MTLComputePipelineState> p_cfwht = mk("col_fwht_and_ss");
        id<MTLComputePipelineState> p_fsc   = mk("finalize_scales");
        id<MTLComputePipelineState> p_enc   = mk("tfq4_e8p_encode");
        id<MTLCommandQueue> q = [dev newCommandQueue];

        fs::create_directories(opt.out_dir);
        tfq4::PwV4E8Tables tbl;
        tfq4::build_e8p_tables(tbl);
        {
            std::ofstream tf(fs::path(opt.out_dir) / "pw_v4_e8p_tables.bin", std::ios::binary);
            tf.write(reinterpret_cast<const char*>(&tbl), sizeof(tbl));
        }
        id<MTLBuffer> b_lut = [dev newBufferWithBytes:tbl.e8p_lut length:sizeof(tbl.e8p_lut) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_met = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];

        std::vector<tfq4::Tfp4TensorMeta> metas;
        std::vector<uint16_t> all_scales;
        std::vector<uint32_t> all_words;
        double snr_sum = 0.0;

        for (size_t ji = 0; ji < plan.jobs.size(); ++ji) {
            auto& j = plan.jobs[ji];
            std::vector<float> w_f32;
            tfq_generic::decode_chunk_to_f32(st, j, 0u, j.pad_R, w_f32);
            uint32_t R = j.pad_R, C = j.pad_C;
            uint32_t n_br = R / 128u, n_bc = C / 128u, n_blk = n_br * n_bc;
            uint64_t n_words = (uint64_t)R * (C / 16u);
            Tfq4MatParams mp{R, C};

            id<MTLBuffer> b_in  = [dev newBufferWithBytes:w_f32.data() length:(size_t)R * C * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_tmp = [dev newBufferWithLength:(size_t)R * C * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_rot = [dev newBufferWithLength:(size_t)R * C * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_ss  = [dev newBufferWithLength:(size_t)n_blk * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sc  = [dev newBufferWithLength:(size_t)n_blk * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_out = [dev newBufferWithLength:(size_t)n_words * 4 options:MTLResourceStorageModeShared];
            memset([b_ss contents], 0, (size_t)n_blk * 4);
            memset([b_met contents], 0, 8);

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p_rfwht];
            [enc setBuffer:b_in offset:0 atIndex:0]; [enc setBuffer:b_tmp offset:0 atIndex:1];
            [enc setBytes:&mp length:sizeof(mp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(n_bc, R, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc setComputePipelineState:p_cfwht];
            [enc setBuffer:b_tmp offset:0 atIndex:0]; [enc setBuffer:b_rot offset:0 atIndex:1];
            [enc setBuffer:b_ss offset:0 atIndex:2];  [enc setBytes:&mp length:sizeof(mp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(C, n_br, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc setComputePipelineState:p_fsc];
            [enc setBuffer:b_ss offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
            [enc setBytes:&n_blk length:sizeof(n_blk) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((n_blk + 255u) / 256u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:p_enc];
            [enc setBuffer:b_rot offset:0 atIndex:0]; [enc setBuffer:b_sc offset:0 atIndex:1];
            [enc setBuffer:b_lut offset:0 atIndex:2]; [enc setBuffer:b_out offset:0 atIndex:3];
            [enc setBuffer:b_met offset:0 atIndex:4]; [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((n_words + 255ull) / 256ull, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

            float met[2]; memcpy(met, [b_met contents], 8);
            float snr = (met[0] > 0.0f) ? (10.0f * std::log10(met[0] / std::max(met[1], 1e-18f))) : 0.0f;
            j.snr_db = snr;
            j.bits = 2u;
            snr_sum += snr;

            tfq4::Tfp4TensorMeta m{};
            std::strncpy(m.name, j.tfpn_name.c_str(), sizeof(m.name) - 1);
            m.R = R; m.C = C; m.b_bits = 2u;
            m.sc_off = (uint32_t)all_scales.size();
            m.word_off = (uint64_t)all_words.size();
            m.word_count = n_words;
            m.snr_db = snr;
            m.pad[0] = j.orig_R; m.pad[1] = j.orig_C; m.pad[2] = j.flags;
            metas.push_back(m);

            const uint16_t* sc_p = reinterpret_cast<const uint16_t*>([b_sc contents]);
            const uint32_t* wd_p = reinterpret_cast<const uint32_t*>([b_out contents]);
            all_scales.insert(all_scales.end(), sc_p, sc_p + n_blk);
            all_words.insert(all_words.end(), wd_p, wd_p + n_words);
        }

        size_t off_meta = sizeof(tfq4::Tfp4Header);
        size_t off_sc   = align_up(off_meta + metas.size() * sizeof(tfq4::Tfp4TensorMeta), tfq4::ALIGN_16K);
        size_t off_wt   = align_up(off_sc + all_scales.size() * 2, tfq4::ALIGN_16K);
        size_t tot_sz   = align_up(off_wt + all_words.size() * 4, tfq4::ALIGN_16K);
        tfq4::Tfp4Header hdr{};
        hdr.magic = tfq4::MAGIC_TFP4; hdr.version = tfq4::FORMAT_VER;
        hdr.num_tensors = (uint32_t)metas.size();
        hdr.off_meta = off_meta; hdr.off_scales = off_sc; hdr.off_weights = off_wt;
        hdr.total_scale_halfs = all_scales.size();
        hdr.total_weight_words = all_words.size();
        const std::string shard_fn = "tfq4-00001-of-00001.tffp4";
        {
            std::ofstream ofs(fs::path(opt.out_dir) / shard_fn, std::ios::binary);
            std::vector<char> zeros(tfq4::ALIGN_16K, 0);
            ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
            ofs.write(reinterpret_cast<const char*>(metas.data()), metas.size() * sizeof(tfq4::Tfp4TensorMeta));
            size_t pos = off_meta + metas.size() * sizeof(tfq4::Tfp4TensorMeta);
            if (off_sc > pos) ofs.write(zeros.data(), off_sc - pos);
            ofs.write(reinterpret_cast<const char*>(all_scales.data()), all_scales.size() * 2);
            pos = off_sc + all_scales.size() * 2;
            if (off_wt > pos) ofs.write(zeros.data(), off_wt - pos);
            ofs.write(reinterpret_cast<const char*>(all_words.data()), all_words.size() * 4);
            pos = off_wt + all_words.size() * 4;
            if (tot_sz > pos) ofs.write(zeros.data(), tot_sz - pos);
        }

        tfq_generic::package_residuals_and_manifest(st, opt, plan, {shard_fn});
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        printf("TFQ4-E8P Metal DONE: %zu tensors -> %.3f GiB (2.001 bpw) | mean SNR=%.2f dB | %.1fs -> %s\n",
               metas.size(), tot_sz / (double)(1ULL << 30), snr_sum / metas.size(), secs, opt.out_dir.c_str());
    }
    return 0;
}
