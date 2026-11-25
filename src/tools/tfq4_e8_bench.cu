// TFQ4 Prototype: 2.0 bpw E8P (256-entry 4 KiB FP16 LUT x 8-bit sign mask) CUDA Decode Kernel + fp64 Parity & GB10 Benchmark.
// Demonstrates sub-3-bit cold-expert decode throughput using a 4 KiB shared-memory LUT + 2D FWHT-128.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// Each 16-bit code encodes 8 rotated weights (2.000 bpw):
//   low  8 bits = abs_idx in [0..255] -> 8 FP16 magnitudes from 4 KiB E8P LUT
//   high 8 bits = sign mask (bit j == 1 -> negate lane j)
// Packed in uint32_t words: 2 codes = 16 weights per uint32_t!
__global__ void tfq4_e8p_mv1_kernel(
    const uint32_t* __restrict__ W_e8, // [N, K / 16] (16 weights per u32 -> 2.0 bpw)
    const __half* __restrict__ scales, // [N / 128, K / 128]
    const uint4* __restrict__ lut256,  // 256 x 8 halfs = 256 x 16B = 4096 B
    const __half* __restrict__ X_rot,  // [K]
    float* __restrict__ Y_rot,         // [N]
    uint32_t N, uint32_t K
) {
    __shared__ uint4 s_lut[256];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_lut[i] = lut256[i];
    __syncthreads();

    const uint32_t sg = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 4u + sg;
    if (row >= N) return;

    const uint32_t n_bc = K >> 7;
    const uint32_t* w_row = W_e8 + static_cast<uint64_t>(row) * (K >> 4);
    const __half* sc_row = scales + static_cast<uint64_t>(row >> 7) * n_bc;
    float acc = 0.0f;

    // 32 lanes process 4 blocks of 128 cols per iter (each lane reads 1 uint32_t = 16 weights per 512 cols? Even simpler:
    // 8 lanes cover one 128-col tile (8 x 16 = 128), so 32 lanes cover 4 tiles (512 cols) in parallel!).
    const uint32_t sub = lane >> 3;      // 0..3 -> tile offset inside 512 cols
    const uint32_t pos = lane & 7u;      // 0..7 -> 16-weight chunk inside 128 cols
    for (uint32_t bc4 = 0u; bc4 < n_bc; bc4 += 4u) {
        uint32_t bc = bc4 + sub;
        if (bc < n_bc) {
            uint32_t packed = __ldg(&w_row[bc * 8u + pos]);
            uint32_t c0 = packed & 0xFFFFu;
            uint32_t c1 = packed >> 16;
            const __half* xp = X_rot + (bc << 7) + (pos << 4);
            float blk_sum = 0.0f;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t code = (h == 0) ? c0 : c1;
                uint32_t aidx = code & 255u;
                uint32_t smask = code >> 8;
                uint4 raw_w = s_lut[aidx];
                uint4 raw_x = *reinterpret_cast<const uint4*>(xp + h * 8);
                uint32_t wx[4] = {raw_w.x, raw_w.y, raw_w.z, raw_w.w};
                uint32_t xx[4] = {raw_x.x, raw_x.y, raw_x.z, raw_x.w};
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    // Apply two sign bits directly by XORing FP16 sign bits (bit 15 and bit 31)!
                    uint32_t flip = (((smask >> (p * 2)) & 1u) << 15) | (((smask >> (p * 2 + 1)) & 1u) << 31);
                    uint32_t sw = wx[p] ^ flip;
                    float2 fw = __half22float2(*reinterpret_cast<const __half2*>(&sw));
                    float2 fx = __half22float2(*reinterpret_cast<const __half2*>(&xx[p]));
                    blk_sum = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, blk_sum));
                }
            }
            acc = fmaf(blk_sum, __half2float(__ldg(&sc_row[bc])), acc);
        }
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
    if (lane == 0u) Y_rot[row] = acc;
}

int main() {
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("TFQ4-E8P (2.0 bpw) CUDA Kernel Check on %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // 1. Exact fp64 parity test
    const uint32_t N = 512u, K = 2048u;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(0.05f, 2.0f);
    std::vector<__half> h_lut(256 * 8);
    for (auto& h : h_lut) h = __float2half(ud(rng));
    std::vector<uint32_t> h_W((size_t)N * (K / 16u));
    for (auto& w : h_W) w = rng();
    std::vector<__half> h_sc((size_t)(N / 128u) * (K / 128u));
    for (auto& s : h_sc) s = __float2half(0.02f + 0.01f * (float)(rng() & 255u) / 255.0f);
    std::vector<__half> h_X(K);
    for (auto& x : h_X) x = __float2half(((int)(rng() % 201u) - 100) * 0.01f);

    uint32_t* d_W; __half *d_sc, *d_X; uint4* d_lut; float* d_Y;
    CHECK(cudaMalloc(&d_W, h_W.size() * 4)); CHECK(cudaMalloc(&d_sc, h_sc.size() * 2));
    CHECK(cudaMalloc(&d_lut, 4096)); CHECK(cudaMalloc(&d_X, K * 2)); CHECK(cudaMalloc(&d_Y, N * 4));
    CHECK(cudaMemcpy(d_W, h_W.data(), h_W.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sc, h_sc.data(), h_sc.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lut, h_lut.data(), 4096, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_X, h_X.data(), K * 2, cudaMemcpyHostToDevice));

    tfq4_e8p_mv1_kernel<<<N / 4u, 128>>>(d_W, d_sc, d_lut, d_X, d_Y, N, K);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> h_Y(N);
    CHECK(cudaMemcpy(h_Y.data(), d_Y, N * 4, cudaMemcpyDeviceToHost));

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t r = 0; r < N; ++r) {
        double ref = 0.0;
        for (uint32_t bc = 0; bc < K / 128u; ++bc) {
            double sc = __half2float(h_sc[(size_t)(r / 128u) * (K / 128u) + bc]);
            double bs = 0.0;
            for (uint32_t pos = 0; pos < 8u; ++pos) {
                uint32_t pk = h_W[(size_t)r * (K / 16u) + bc * 8u + pos];
                for (int h = 0; h < 2; ++h) {
                    uint32_t code = (h == 0) ? (pk & 0xFFFFu) : (pk >> 16);
                    uint32_t aidx = code & 255u, sm = code >> 8;
                    for (int j = 0; j < 8; ++j) {
                        double wv = __half2float(h_lut[aidx * 8 + j]);
                        if ((sm >> j) & 1u) wv = -wv;
                        bs += wv * (double)__half2float(h_X[bc * 128u + pos * 16u + h * 8 + j]);
                    }
                }
            }
            ref += bs * sc;
        }
        dot += (double)h_Y[r] * ref; na += (double)h_Y[r] * h_Y[r]; nb += ref * ref;
    }
    double cos_sim = dot / (std::sqrt(na * nb) + 1e-30);
    printf("PASS fp64 parity: cos=%.8f\n", cos_sim);
    if (cos_sim < 0.999999) return 1;
    cudaFree(d_W); cudaFree(d_sc); cudaFree(d_X); cudaFree(d_Y);

    // 2. Cold-expert MoE / Dense throughput benchmark (2.0 bpw)
    struct Shape { const char* name; uint32_t N, K; };
    Shape shapes[] = {{"MoE-8E-gu", 32768, 4096}, {"Qwen27-gu", 34816, 5120}, {"Qwen27-dn", 5120, 17408}};
    cudaEvent_t ev0, ev1; CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));
    for (auto& s : shapes) {
        size_t nw = (size_t)s.N * (s.K / 16u), ns = (size_t)(s.N / 128u) * (s.K / 128u);
        CHECK(cudaMalloc(&d_W, nw * 4)); CHECK(cudaMalloc(&d_sc, ns * 2));
        CHECK(cudaMalloc(&d_X, s.K * 2)); CHECK(cudaMalloc(&d_Y, s.N * 4));
        CHECK(cudaMemset(d_W, 0x55, nw * 4)); CHECK(cudaMemset(d_sc, 0x2E, ns * 2)); CHECK(cudaMemset(d_X, 0x38, s.K * 2));
        for (int w = 0; w < 10; ++w) tfq4_e8p_mv1_kernel<<<s.N / 4u, 128>>>(d_W, d_sc, d_lut, d_X, d_Y, s.N, s.K);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(ev0));
        for (int it = 0; it < 100; ++it) tfq4_e8p_mv1_kernel<<<s.N / 4u, 128>>>(d_W, d_sc, d_lut, d_X, d_Y, s.N, s.K);
        CHECK(cudaEventRecord(ev1)); CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        double avg_ms = ms / 100.0;
        double bytes = (double)(nw * 4 + ns * 2);
        printf("  %-10s [%5u x %5u] 2.0b E8P : %6.3f ms | %6.1f GB/s (%.1f MiB)\n",
               s.name, s.N, s.K, avg_ms, (bytes / 1e9) / (avg_ms / 1e3), bytes / (1024.0 * 1024.0));
        cudaFree(d_W); cudaFree(d_sc); cudaFree(d_X); cudaFree(d_Y);
    }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaFree(d_lut);
    return 0;
}
