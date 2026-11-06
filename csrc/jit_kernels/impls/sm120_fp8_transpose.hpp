#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

class SM120TransposeFP8Runtime final {
public:
    struct Args {
        deep_jit::cuda::LaunchOptions options;
        const void* in;
        void* out;
        int mn, k;
        int64_t ld_in, in_batch_stride;
    };

    static constexpr int kNumThreads = 256, kBlockMN = 128, kBlockK = 64;

    static void compile_and_launch(const Args& args) {
        const auto kernel = jit->compile("sm120_transpose_fp8", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_transpose.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_transpose_fp8<{}, {}, {}>);
}};
)", kNumThreads, kBlockMN, kBlockK));
        jit->launch(kernel, args.options, args.in, args.out,
                    static_cast<uint32_t>(args.mn), static_cast<uint32_t>(args.k),
                    static_cast<uint64_t>(args.ld_in), static_cast<uint64_t>(args.in_batch_stride));
    }
};

// Whether the JIT byte transpose can move this MN-major operand (`[(G,) MN, K]` with unit MN stride)
static bool sm120_can_transpose_fp8(const torch::Tensor& t) {
    const int mn = static_cast<int>(t.size(-2)), k = static_cast<int>(t.size(-1));
    return t.element_size() == 1 and (t.dim() == 2 or t.dim() == 3) and t.stride(-2) == 1 and
           mn % 16 == 0 and k % 16 == 0 and t.stride(-1) % 4 == 0 and
           reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0;
}

// Compact K-major copy `[(G,) MN, K]` of an MN-major FP8 operand (the explicit transpose that precedes the K-major
// kernel). Falls back to PyTorch's copy for shapes the byte-transpose kernel does not handle
static torch::Tensor sm120_transpose_fp8_to_k_major(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.stride(-2) == 1);
    if (not sm120_can_transpose_fp8(t))
        return t.contiguous();
    const int mn = static_cast<int>(t.size(-2)), k = static_cast<int>(t.size(-1));
    const int num_batches = t.dim() == 3 ? static_cast<int>(t.size(0)) : 1;
    auto out = torch::empty(t.sizes(), t.options());
    using Runtime = SM120TransposeFP8Runtime;
    Runtime::compile_and_launch({
        .options = {
            .num_smem_bytes = 0,
            .grid_dim = dim3(ceil_div(mn, Runtime::kBlockMN), ceil_div(k, Runtime::kBlockK), num_batches),
            .block_dim = dim3(Runtime::kNumThreads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .in = t.data_ptr(), .out = out.data_ptr(),
        .mn = mn, .k = k,
        .ld_in = t.stride(-1), .in_batch_stride = t.dim() == 3 ? t.stride(0) : 0,
    });
    return out;
}

} // namespace deep_gemm
