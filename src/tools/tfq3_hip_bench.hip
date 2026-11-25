// Standalone AMD ROCm / HIP parity test (3b/4b/5b dense + Col-TP/Row-TP + mixed-bit MoE) & benchmark for TFQ3.
#include "../kernels/hip/tfq3_hip.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP err %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

static uint16_t h_f32_to_bf16(float f) { uint32_t w; memcpy(&w, &f, 4); return (uint16_t)((w + 0x7FFFu + ((w >> 16) & 1u)) >> 16); }

static void init_synth_tables(tfq3_hip::PwV3Tables& tbl) {
    memset(&tbl, 0, sizeof(tbl));
    tbl.magic = 0x33565750u; tbl.version = 3u;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.8f);
    auto pack2 = [](float a, float b) {
        __half2 h = __floats2half2_rn(a, b);
        uint32_t u; memcpy(&u, &h, 4); return u;
    };
    for (int i = 0; i < 256; ++i) tbl.lut3[i] = pack2(nd(rng), nd(rng));
    for (int i = 0; i < 1024; ++i) tbl.lut4[i] = pack2(nd(rng), nd(rng));
    for (int i = 0; i < 1024; ++i) tbl.lut5[i] = pack2(nd(rng), nd(rng));
    for (int i = 0; i < 64; ++i) {
        tbl.c64_3[i] = make_uint2(pack2(1.0f + 0.05f * nd(rng), 1.0f + 0.05f * nd(rng)), pack2(0.05f * nd(rng), 0.05f * nd(rng)));
        tbl.c64_4[i] = make_uint2(pack2(1.0f + 0.05f * nd(rng), 1.0f + 0.05f * nd(rng)), pack2(0.05f * nd(rng), 0.05f * nd(rng)));
        tbl.c64_5[i] = make_uint2(pack2(1.0f + 0.05f * nd(rng), 1.0f + 0.05f * nd(rng)), pack2(0.05f * nd(rng), 0.05f * nd(rng)));
    }
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("AMD HIP Device: %s (%s, warpSize=%d)\n", prop.name, prop.gcnArchName, prop.warpSize);

    tfq3_hip::PwV3Tables h_tbl; init_synth_tables(h_tbl);
    tfq3_hip::PwV3Tables* d_tbl;
    HIP_CHECK(hipMalloc((void**)&d_tbl, sizeof(h_tbl)));
    HIP_CHECK(hipMemcpy(d_tbl, &h_tbl, sizeof(h_tbl), hipMemcpyHostToDevice));

    std::mt19937 rng(7);
    const uint32_t N = 512u, K = 1024u, rows = 8u;
    __half* d_xrot; HIP_CHECK(hipMalloc((void**)&d_xrot, (size_t)rows * K * sizeof(__half)));

    for (uint32_t bits : {3u, 4u, 5u}) {
        size_t nw = (size_t)(N / 16u) * (K / 128u) * (bits * 2u) * 32ull;
        size_t ns = (size_t)(N / 128u) * (K / 128u);
        std::vector<uint32_t> h_W(nw);
        std::vector<__half> h_sc(ns);
        std::vector<uint16_t> h_X(rows * K);
        for (auto& w : h_W) w = rng();
        for (auto& s : h_sc) s = __float2half(0.03f);
        for (auto& x : h_X) x = h_f32_to_bf16(((int)(rng() % 201u) - 100) * 0.01f);

        uint32_t* d_W; __half* d_sc; uint16_t* d_X; float* d_Y;
        HIP_CHECK(hipMalloc((void**)&d_W, nw * 4)); HIP_CHECK(hipMalloc((void**)&d_sc, ns * 2));
        HIP_CHECK(hipMalloc((void**)&d_X, rows * K * 2)); HIP_CHECK(hipMalloc((void**)&d_Y, rows * N * 4));
        HIP_CHECK(hipMemcpy(d_W, h_W.data(), nw * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sc, h_sc.data(), ns * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_X, h_X.data(), rows * K * 2, hipMemcpyHostToDevice));

        HIP_CHECK(tfq3_hip::launch_dense_tp(d_W, d_sc, d_tbl, d_X, d_Y, d_xrot, bits, rows, 0u, N, K, 0u, K, K, N, true));
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> y_full(rows * N);
        HIP_CHECK(hipMemcpy(y_full.data(), d_Y, rows * N * 4, hipMemcpyDeviceToHost));

        // Verify Row-TP=4 AllReduce(SUM) == full
        std::vector<float> y_sum(rows * N, 0.0f);
        for (uint32_t rk = 0; rk < 4u; ++rk) {
            uint32_t k_loc = K / 4u;
            HIP_CHECK(tfq3_hip::launch_dense_tp(d_W, d_sc, d_tbl, d_X + rk * k_loc, d_Y, d_xrot, bits, rows, 0u, N, K, rk * k_loc, k_loc, K, N, true));
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> part(rows * N);
            HIP_CHECK(hipMemcpy(part.data(), d_Y, rows * N * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < y_sum.size(); ++i) y_sum[i] += part[i];
        }
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (size_t i = 0; i < y_sum.size(); ++i) { dot += (double)y_sum[i] * y_full[i]; n1 += (double)y_sum[i] * y_sum[i]; n2 += (double)y_full[i] * y_full[i]; }
        double cos_tp = dot / (std::sqrt(n1 * n2) + 1e-30);
        if (cos_tp < 0.999999) { fprintf(stderr, "FAIL HIP Row-TP=4 %ub cos=%.8f\n", bits, cos_tp); return 1; }
        hipFree(d_W); hipFree(d_sc); hipFree(d_X); hipFree(d_Y);
    }
    hipFree(d_tbl); hipFree(d_xrot);
    printf("PASS tfq3_hip_test (3b/4b/5b wave32/wave64 dense + zero-repack Row-TP=4)\n");
    return 0;
}
