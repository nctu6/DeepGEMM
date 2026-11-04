#pragma once

#include <cstdio>
#include <format>
#include <torch/python.h>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// SM120 hyper-connection pre-norm GEMM (see `sm120_tf32_hc_prenorm_gemm.cuh`). A is BF16, B is FP32, D is FP32, both
// operands K-major. D and `sqr_sum` are written straight from registers (no TMA store), so N is not padded
static void sm120_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& d,
                                       const torch::Tensor& sqr_sum,
                                       const int& m, const int& n, const int& k,
                                       const int& num_splits) {
    constexpr int block_m = 64;
    constexpr int block_k = 64;
    constexpr int num_math_threads = 128;
    constexpr int num_tma_threads = 128;

    DG_HOST_ASSERT(n % 8 == 0 and n <= 128 and "SM120 hc pre-norm GEMM supports N a multiple of 8, up to 128");
    DG_HOST_ASSERT(k % block_k == 0);
    // D rows store 2 FP32 columns (8 bytes) at a time
    DG_HOST_ASSERT(d.stride(-2) % 2 == 0 and reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0);

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              block_m, block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, a.element_size()), 0, true);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                              n, block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, b.element_size()), 0, true);

    // Stages: as many as the ~99 KB smem fits (A 8 KB + B n*256 B per stage)
    const int smem_a_per_stage = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_b_per_stage = n * block_k * static_cast<int>(sizeof(float));
    int num_stages = std::min(SM120ArchSpec::kNumMaxStages,
                              (SM120ArchSpec::get_smem_capacity() - SM120ArchSpec::kNumMaxStages * 2 * 8) /
                                  (smem_a_per_stage + smem_b_per_stage));
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages + num_stages * 2 * 8;

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS"))
        printf("SM120 hc pre-norm: M %d, N %d, K %d, splits %d, stages %d, smem %d\n", m, n, k, num_splits, num_stages, smem_size);

    const auto kernel = jit->compile("sm120_tf32_hc_prenorm_gemm", std::format(R"(
#include <deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_tf32_hc_prenorm_gemm_impl<
        {}, {},
        {}, {},
        {},
        {},
        {}, {}
    >);
}};
)",
        n, k,
        block_m, block_k,
        num_splits,
        num_stages,
        num_math_threads, num_tma_threads));

    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_splits * ceil_div(m, block_m), 1, 1),
            .block_dim = dim3(num_math_threads + num_tma_threads, 1, 1),
        },
        static_cast<uint32_t>(m), tensor_map_a, tensor_map_b,
        d.data_ptr<float>(), static_cast<uint32_t>(d.stride(-2)),
        num_splits == 1 ? 0u : static_cast<uint32_t>(d.stride(-3)),
        sqr_sum.data_ptr<float>()
    );
}

} // namespace deep_gemm
