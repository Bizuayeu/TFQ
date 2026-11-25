// ============================================================================
// PyTorch C++/CUDA Extension for TFQ3 (3/4/5-bit 2D-AffineCoset + 2D FWHT-128)
//   Directly binds `src/kernels/cuda/tfq3_cuda.cuh` to torch::Tensor so any
//   PyTorch, Hugging Face Transformers, or vLLM model can run TFQ3 checkpoints
//   without TensorFold.
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "../../src/kernels/cuda/tfq3_cuda.cuh"

torch::Tensor tfq3_linear_cuda(
    torch::Tensor x_bf16,      // [..., orig_c] bfloat16 CUDA
    torch::Tensor words_u32,   // flat int32/uint32 packed TFPN words CUDA
    torch::Tensor scales_f16,  // [pad_r/128, pad_c/128] float16 CUDA
    torch::Tensor tables_u8,   // [16384] uint8 pw_v3_tables.bin CUDA
    int64_t bits,
    int64_t pad_r,
    int64_t pad_c,
    int64_t orig_r
) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be a CUDA tensor");
    TORCH_CHECK(x_bf16.scalar_type() == torch::kBFloat16, "x_bf16 must be bfloat16");
    TORCH_CHECK(words_u32.is_cuda() && scales_f16.is_cuda() && tables_u8.is_cuda(), "TFQ3 buffers must be CUDA");
    TORCH_CHECK(bits == 3 || bits == 4 || bits == 5, "TFQ3 bits must be 3, 4, or 5");
    TORCH_CHECK((pad_r % 128) == 0 && (pad_c % 128) == 0, "pad_r and pad_c must be multiples of 128");
    TORCH_CHECK(tables_u8.numel() >= (int64_t)sizeof(tfq3::PwV3Tables), "tables_u8 smaller than PwV3Tables");

    const c10::cuda::CUDAGuard device_guard(x_bf16.device());
    auto orig_sizes = x_bf16.sizes().vec();
    const int64_t orig_c = orig_sizes.back();
    const int64_t rows = x_bf16.numel() / orig_c;

    torch::Tensor x2d = x_bf16.contiguous().view({rows, orig_c});
    if (orig_c < pad_c) {
        x2d = torch::nn::functional::pad(x2d, torch::nn::functional::PadFuncOptions({0, pad_c - orig_c}));
    }

    auto opts_bf16 = x_bf16.options().dtype(torch::kBFloat16);
    auto opts_f16  = x_bf16.options().dtype(torch::kFloat16);
    auto opts_f32  = x_bf16.options().dtype(torch::kFloat32);

    torch::Tensor y_pad = torch::empty({rows, pad_r}, opts_bf16);
    torch::Tensor xrot  = torch::empty({rows, pad_c}, opts_f16);

    const bool can_sk8 = (rows <= 64) && (pad_r <= 34816) && ((pad_c >> 7) >= 8) && (((pad_c >> 7) & 7) == 0);
    torch::Tensor yr = can_sk8 ? torch::empty({rows * pad_r * 8}, opts_f32) : torch::empty({0}, opts_f32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = tfq3::launch_dense(
        reinterpret_cast<const uint32_t*>(words_u32.data_ptr()),
        reinterpret_cast<const __half*>(scales_f16.data_ptr()),
        reinterpret_cast<const tfq3::PwV3Tables*>(tables_u8.data_ptr()),
        reinterpret_cast<const uint16_t*>(x2d.data_ptr()),
        y_pad.data_ptr(),
        reinterpret_cast<__half*>(xrot.data_ptr()),
        can_sk8 ? yr.data_ptr<float>() : nullptr,
        static_cast<size_t>(yr.numel()),
        static_cast<uint32_t>(bits),
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(pad_r),
        static_cast<uint32_t>(pad_c),
        static_cast<uint32_t>(pad_c),
        static_cast<uint32_t>(pad_r),
        0u,
        false,
        0u,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "tfq3::launch_dense failed: ", cudaGetErrorString(err));

    torch::Tensor y_out = (orig_r < pad_r) ? y_pad.slice(1, 0, orig_r).contiguous() : y_pad;
    orig_sizes.back() = orig_r;
    return y_out.view(orig_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tfq3_linear", &tfq3_linear_cuda, "TFQ3 3/4/5-bit fused 2D-FWHT + Tensor Core linear (CUDA)");
}
