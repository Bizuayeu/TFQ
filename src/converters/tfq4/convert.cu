// ============================================================================
// TFQ4-E8P (2.001 bpw) Model-Agnostic NVIDIA CUDA GPU Converter
//   Byte-identical PWV4 v4 + TFP4 v4 cross-platform output with Apple Metal.
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ float sgn_bit(uint32_t seed, uint32_t i) {
    uint32_t h = (seed ^ i) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ void fwht128_warp4(float4& v, uint32_t lane) {
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1) {
        float4 o;
        o.x = __shfl_xor_sync(0xFFFFFFFFu, v.x, d);
        o.y = __shfl_xor_sync(0xFFFFFFFFu, v.y, d);
        o.z = __shfl_xor_sync(0xFFFFFFFFu, v.z, d);
        o.w = __shfl_xor_sync(0xFFFFFFFFu, v.w, d);
        bool keep = ((lane & d) == 0u);
        v.x = keep ? (v.x + o.x) : (o.x - v.x);
        v.y = keep ? (v.y + o.y) : (o.y - v.y);
        v.z = keep ? (v.z + o.z) : (o.z - v.z);
        v.w = keep ? (v.w + o.w) : (o.w - v.w);
    }
    float a0 = v.x + v.y, b0 = v.x - v.y, a1 = v.z + v.w, b1 = v.z - v.w;
    v = make_float4((a0 + a1) * 0.08838834764831845f, (b0 + b1) * 0.08838834764831845f,
                    (a0 - a1) * 0.08838834764831845f, (b0 - b1) * 0.08838834764831845f);
}

__global__ void f32_row_fwht_k(const float* __restrict__ W_in, float* __restrict__ W_tmp, uint32_t R, uint32_t C) {
    uint32_t r = blockIdx.y, bc = blockIdx.x, c0 = bc * 128u, lane = threadIdx.x;
    if (r >= R) return;
    const float* row = W_in + (uint64_t)r * C + c0;
    float4 v = make_float4(row[lane + 0u] * sgn_bit(0x12345678u, c0 + lane + 0u),
                           row[lane + 32u] * sgn_bit(0x12345678u, c0 + lane + 32u),
                           row[lane + 64u] * sgn_bit(0x12345678u, c0 + lane + 64u),
                           row[lane + 96u] * sgn_bit(0x12345678u, c0 + lane + 96u));
    fwht128_warp4(v, lane);
    float* dst = W_tmp + (uint64_t)r * C + c0;
    dst[lane + 0u] = v.x; dst[lane + 32u] = v.y; dst[lane + 64u] = v.z; dst[lane + 96u] = v.w;
}

__global__ void col_fwht_ss_k(const float* __restrict__ W_tmp, float* __restrict__ W_rot, float* __restrict__ blk_ss, uint32_t R, uint32_t C) {
    uint32_t c = blockIdx.x, br = blockIdx.y, r0 = br * 128u, lane = threadIdx.x;
    if (c >= C) return;
    float4 v = make_float4(W_tmp[(uint64_t)(r0 + lane + 0u) * C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 0u),
                           W_tmp[(uint64_t)(r0 + lane + 32u) * C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 32u),
                           W_tmp[(uint64_t)(r0 + lane + 64u) * C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 64u),
                           W_tmp[(uint64_t)(r0 + lane + 96u) * C + c] * sgn_bit(0x9E3779B9u, r0 + lane + 96u));
    fwht128_warp4(v, lane);
    W_rot[(uint64_t)(r0 + lane + 0u) * C + c] = v.x;
    W_rot[(uint64_t)(r0 + lane + 32u) * C + c] = v.y;
    W_rot[(uint64_t)(r0 + lane + 64u) * C + c] = v.z;
    W_rot[(uint64_t)(r0 + lane + 96u) * C + c] = v.w;
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, d);
    if (lane == 0) atomicAdd(&blk_ss[br * (C >> 7) + (c >> 7)], ss);
}

__global__ void finalize_scales_k(const float* __restrict__ blk_ss, __half* __restrict__ sc, uint32_t n_blk) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_blk) sc[i] = __float2half(sqrtf(blk_ss[i] * (1.0f / 16384.0f) + 1e-12f));
}

__global__ void tfq4_e8p_encode_k(
    const float* __restrict__ W_rot, const __half* __restrict__ scales,
    const __half* __restrict__ e8p_lut, uint32_t* __restrict__ out_words,
    double* __restrict__ metrics, uint32_t R, uint32_t C
) {
    __shared__ float sm_lut[256][8];
    uint32_t tid = threadIdx.x;
    #pragma unroll
    for (uint32_t d = 0u; d < 8u; ++d) sm_lut[tid][d] = __half2float(e8p_lut[tid * 8u + d]);
    __syncthreads();

    uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + tid;
    uint32_t wpr = C >> 4;
    uint64_t total_words = (uint64_t)R * wpr;
    double my_sig = 0.0, my_err = 0.0;

    if (gtid < total_words) {
        uint32_t r = (uint32_t)(gtid / wpr), wc = (uint32_t)(gtid - (uint64_t)r * wpr);
        uint32_t n_bc = C >> 7;
        float sc = __half2float(scales[(r >> 7) * n_bc + (wc >> 3)]);
        float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        const float* src = W_rot + (uint64_t)r * C + (uint64_t)wc * 16ull;

        uint32_t codes[2];
        #pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            float v[8], a[8];
            uint32_t sm = 0u;
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                float x = src[h * 8u + d];
                v[d] = x;
                if (x < 0.0f) sm |= (1u << d);
                a[d] = fabsf(x) * inv_sc;
                my_sig += (double)x * (double)x;
            }
            uint32_t best_k = 0u;
            float best_d = 1e30f;
            for (uint32_t k = 0u; k < 256u; ++k) {
                float d0 = a[0] - sm_lut[k][0], d1 = a[1] - sm_lut[k][1];
                float d2 = a[2] - sm_lut[k][2], d3 = a[3] - sm_lut[k][3];
                float d4 = a[4] - sm_lut[k][4], d5 = a[5] - sm_lut[k][5];
                float d6 = a[6] - sm_lut[k][6], d7 = a[7] - sm_lut[k][7];
                float dist = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
                if (dist < best_d) { best_d = dist; best_k = k; }
            }
            codes[h] = (sm << 8) | best_k;
            #pragma unroll
            for (uint32_t d = 0u; d < 8u; ++d) {
                double e = fabs((double)v[d]) - (double)sm_lut[best_k][d] * (double)sc;
                my_err += e * e;
            }
        }
        out_words[gtid] = codes[0] | (codes[1] << 16);
    }

    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
        my_sig += __shfl_xor_sync(0xFFFFFFFFu, my_sig, d);
        my_err += __shfl_xor_sync(0xFFFFFFFFu, my_err, d);
    }
    if ((tid & 31u) == 0u) {
        atomicAdd(&metrics[0], my_sig);
        atomicAdd(&metrics[1], my_err);
    }
}

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

int main(int argc, char** argv) {
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
    }
    if (opt.model_dir.empty() || opt.out_dir.empty()) {
        fprintf(stderr, "Usage: tfq4-convert-cuda --model DIR --out DIR [--quantize-vision] [--quantize-drafter] [--quantize-embed]\n");
        return 2;
    }

    tfq_generic::SafetensorsDir st;
    st.init(opt.model_dir);
    auto plan = tfq_generic::build_plan(st, opt);
    if (plan.jobs.empty()) { fprintf(stderr, "No quantizable 2D float tensors in %s\n", opt.model_dir.c_str()); return 1; }
    if (opt.dry_run) return 0;

    auto t_start = std::chrono::steady_clock::now();
    fs::create_directories(opt.out_dir);
    tfq4::PwV4E8Tables tbl;
    tfq4::build_e8p_tables(tbl);
    {
        std::ofstream tf(fs::path(opt.out_dir) / "pw_v4_e8p_tables.bin", std::ios::binary);
        tf.write(reinterpret_cast<const char*>(&tbl), sizeof(tbl));
    }

    __half* d_lut = nullptr; double* d_met = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_lut, sizeof(tbl.e8p_lut)));
    CUDA_CHECK(cudaMemcpy(d_lut, tbl.e8p_lut, sizeof(tbl.e8p_lut), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_met, 2 * sizeof(double)));

    std::vector<tfq4::Tfp4TensorMeta> metas;
    std::vector<uint16_t> all_scales;
    std::vector<uint32_t> all_words;
    double snr_sum = 0.0;

    for (auto& j : plan.jobs) {
        std::vector<float> w_f32;
        tfq_generic::decode_chunk_to_f32(st, j, 0u, j.pad_R, w_f32);
        uint32_t R = j.pad_R, C = j.pad_C;
        uint32_t n_br = R / 128u, n_bc = C / 128u, n_blk = n_br * n_bc;
        uint64_t n_words = (uint64_t)R * (C / 16u);

        float *d_in = nullptr, *d_tmp = nullptr, *d_rot = nullptr, *d_ss = nullptr;
        __half* d_sc = nullptr; uint32_t* d_out = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_in, (size_t)R * C * 4));
        CUDA_CHECK(cudaMalloc((void**)&d_tmp, (size_t)R * C * 4));
        CUDA_CHECK(cudaMalloc((void**)&d_rot, (size_t)R * C * 4));
        CUDA_CHECK(cudaMalloc((void**)&d_ss, (size_t)n_blk * 4));
        CUDA_CHECK(cudaMalloc((void**)&d_sc, (size_t)n_blk * 2));
        CUDA_CHECK(cudaMalloc((void**)&d_out, (size_t)n_words * 4));
        CUDA_CHECK(cudaMemcpy(d_in, w_f32.data(), (size_t)R * C * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_ss, 0, (size_t)n_blk * 4));
        CUDA_CHECK(cudaMemset(d_met, 0, 2 * sizeof(double)));

        f32_row_fwht_k<<<dim3(n_bc, R), 32>>>(d_in, d_tmp, R, C);
        col_fwht_ss_k<<<dim3(C, n_br), 32>>>(d_tmp, d_rot, d_ss, R, C);
        finalize_scales_k<<<(n_blk + 255u) / 256u, 256>>>(d_ss, d_sc, n_blk);
        tfq4_e8p_encode_k<<<(uint32_t)((n_words + 255ull) / 256ull), 256>>>(d_rot, d_sc, d_lut, d_out, d_met, R, C);
        CUDA_CHECK(cudaDeviceSynchronize());

        double met[2];
        CUDA_CHECK(cudaMemcpy(met, d_met, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        float snr = (met[0] > 0.0) ? (float)(10.0 * std::log10(met[0] / std::max(met[1], 1e-18))) : 0.0f;
        j.snr_db = snr;
        j.bits = 2u;
        snr_sum += snr;

        std::vector<uint16_t> h_sc(n_blk);
        std::vector<uint32_t> h_wd(n_words);
        CUDA_CHECK(cudaMemcpy(h_sc.data(), d_sc, (size_t)n_blk * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_wd.data(), d_out, (size_t)n_words * 4, cudaMemcpyDeviceToHost));
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_rot); cudaFree(d_ss); cudaFree(d_sc); cudaFree(d_out);

        tfq4::Tfp4TensorMeta m{};
        std::strncpy(m.name, j.tfpn_name.c_str(), sizeof(m.name) - 1);
        m.R = R; m.C = C; m.b_bits = 2u;
        m.sc_off = (uint32_t)all_scales.size();
        m.word_off = (uint64_t)all_words.size();
        m.word_count = n_words;
        m.snr_db = snr;
        m.pad[0] = j.orig_R; m.pad[1] = j.orig_C; m.pad[2] = j.flags;
        metas.push_back(m);
        all_scales.insert(all_scales.end(), h_sc.begin(), h_sc.end());
        all_words.insert(all_words.end(), h_wd.begin(), h_wd.end());
    }
    cudaFree(d_lut); cudaFree(d_met);

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
    printf("TFQ4-E8P CUDA DONE: %zu tensors -> %.3f GiB (2.001 bpw) | mean SNR=%.2f dB | %.1fs -> %s\n",
           metas.size(), tot_sz / (double)(1ULL << 30), snr_sum / metas.size(), secs, opt.out_dir.c_str());
    return 0;
}
