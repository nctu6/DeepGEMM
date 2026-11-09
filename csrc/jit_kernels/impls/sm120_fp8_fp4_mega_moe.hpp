#pragma once
#include <format>
#include <torch/python.h>
#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"
#include "sm120_fp8_fp4_gemm_1d1d.hpp"
namespace deep_gemm {
// SM120 `fp8_fp4_mega_moe` for one rank: dispatch, the two m-grouped FP8 x FP4 GEMMs, SwiGLU and combine as a sequence
// of kernels with the SM100 numerics (see `sm120_fp8_fp4_mega_moe.cuh` for the steps and why)
class SM120MegaMoERuntime final {
public:
    static constexpr int kDispatchThreads = 1024, kNumWarps = 8, kCombineThreads = 128;
    static deep_jit::cuda::LaunchOptions options(const int& num_blocks, const int& num_threads) {
        return {.num_smem_bytes = 0, .grid_dim = dim3(num_blocks, 1, 1), .block_dim = dim3(num_threads, 1, 1),
                .cluster_dim = dim3(1, 1, 1)};
    }
    static void dispatch(const int& num_experts, const int& num_topk, const int& block_m,
                         const torch::Tensor& topk_idx, const torch::Tensor& pair_rows, const torch::Tensor& psum_layout,
                         int* expert_stats, const int& num_tokens) {
        const auto kernel = jit->compile("sm120_mega_moe_dispatch", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_mega_moe_dispatch<{}, {}, {}, {}>);
}};
)", num_experts, num_topk, block_m, kDispatchThreads));
        jit->launch(kernel, options(1, kDispatchThreads), topk_idx.data_ptr(), pair_rows.data_ptr(), psum_layout.data_ptr(),
                    expert_stats, static_cast<uint32_t>(num_tokens));
    }
    static void gather(const int& hidden, const int& num_topk,
                       const torch::Tensor& x, const torch::Tensor& x_sf, const torch::Tensor& topk_weights,
                       const torch::Tensor& pair_rows, const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
                       const torch::Tensor& row_weights, const int& num_tokens, const int& sf_stride) {
        const auto kernel = jit->compile("sm120_mega_moe_gather", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_mega_moe_gather<{}, {}, {}>);
}};
)", hidden, num_topk, kNumWarps));
        jit->launch(kernel, options(ceil_div(num_tokens * num_topk, kNumWarps), kNumWarps * 32),
                    x.data_ptr(), x_sf.data_ptr(), topk_weights.data_ptr(), pair_rows.data_ptr(),
                    l1_acts.data_ptr(), l1_acts_sf.data_ptr(), row_weights.data_ptr(),
                    static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(sf_stride));
    }
    static void swiglu(const int& intermediate_hidden, const int& num_topk, const bool& fast_math,
                       const torch::Tensor& l1_out, const torch::Tensor& row_weights, const torch::Tensor& pair_rows,
                       const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf, const float& activation_clamp,
                       const int& num_tokens, const int& sf_stride) {
        const auto kernel = jit->compile("sm120_mega_moe_swiglu", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_mega_moe_swiglu<{}, {}, {}, {}>);
}};
)", intermediate_hidden, num_topk, kNumWarps, fast_math ? "true" : "false"));
        const int num_tasks = num_tokens * num_topk * (intermediate_hidden / 128);
        jit->launch(kernel, options(ceil_div(num_tasks, kNumWarps), kNumWarps * 32),
                    l1_out.data_ptr(), row_weights.data_ptr(), pair_rows.data_ptr(), l2_acts.data_ptr(), l2_acts_sf.data_ptr(),
                    activation_clamp, static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(sf_stride));
    }
    static void combine(const int& hidden, const int& num_topk,
                        const torch::Tensor& l2_out, const torch::Tensor& pair_rows, const torch::Tensor& y, const int& num_tokens) {
        const auto kernel = jit->compile("sm120_mega_moe_combine", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_mega_moe_combine<{}, {}, {}>);
}};
)", hidden, num_topk, kCombineThreads));
        jit->launch(kernel, options(num_tokens, kCombineThreads), l2_out.data_ptr(), pair_rows.data_ptr(), y.data_ptr());
    }
};

// Rows per expert run (the GEMM's `BLOCK_M`): the smallest tile covering the expected tokens of a routed expert
static int sm120_get_block_m_for_mega_moe(const int& num_tokens, const int& num_topk, const int& num_experts) {
    const int num_pairs = num_tokens * num_topk;
    const int expected = ceil_div(num_pairs, std::max(1, std::min(num_experts, num_pairs)));
    for (const int block_m: {16, 32, 64})
        if (expected <= block_m)
            return block_m;
    return 128;
}

// Rows that the expert runs can take: every pair, plus the tail of each non-empty run
static int sm120_get_num_mega_moe_rows(const int& num_tokens, const int& num_topk, const int& num_experts, const int& block_m) {
    const int num_pairs = num_tokens * num_topk;
    return align(num_pairs + std::min(num_experts, num_pairs) * (block_m - 1), block_m);
}

// The m-grouped FP8 x FP4 GEMM of one layer over the expert runs (PSUM layout)
static void sm120_mega_moe_grouped_gemm(const torch::Tensor& a, const torch::Tensor& sfa,
                                        const torch::Tensor& b, const torch::Tensor& sfb,
                                        const torch::Tensor& d, const torch::Tensor& psum_layout,
                                        const int& num_experts, const int& m, const int& n, const int& k,
                                        const int& block_m, const int& num_tokens_per_expert) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedContiguousWithPsumLayout,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_experts,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        // NOTES: N and K stay runtime values: with both compiled in, ptxas unrolls the producer's K loop and spills there
        .compiled_dims = "",
        .sf_gran_k = 32,
        .expected_m = std::max(1, num_tokens_per_expert), .expected_n = n, .expected_k = k,
        .expected_num_groups = num_experts
    };
    // FP8 x FP4 takes 64-byte B rows; 128-column tiles (one weight read per expert run and N tile)
    const int block_n = deep_jit::get_env<int>("DG_SM120_MEGA_BLOCK_N", 128);
    DG_HOST_ASSERT(SM120ArchSpec::is_warp_tiling_legal(desc.kernel_type, block_m, block_n));
    const auto choice = SM120BlockScaledChoice{Layout{0, block_m, block_n, 64, 1, 1}, false};
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_mega_moe_grouped_gemm", desc, a, sfa, b, sfb, d,
                                   psum_layout.data_ptr(), 1, num_experts, 1, num_experts, 32, 32, 1.0f, choice);
}

static void sm120_fp8_fp4_mega_moe(const torch::Tensor& y,
                                   const torch::Tensor& x, const torch::Tensor& x_sf,
                                   const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                                   const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
                                   const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
                                   const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                                   const int& num_experts, const int& num_tokens, const int& num_topk,
                                   const int& hidden, const int& intermediate_hidden,
                                   const float& activation_clamp, const bool& fast_math) {
    if (num_tokens == 0)
        return;
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 128 == 0);
    const int forced_block_m = deep_jit::get_env<int>("DG_SM120_MEGA_BLOCK_M", 0);
    const int block_m = forced_block_m > 0 ? forced_block_m : sm120_get_block_m_for_mega_moe(num_tokens, num_topk, num_experts);
    const int num_rows = sm120_get_num_mega_moe_rows(num_tokens, num_topk, num_experts, block_m);
    // Scales are MN-major (`[K / 128][sf_stride]` int32), with 16-byte aligned columns for TMA
    const int sf_stride = align(num_rows, 4);
    const auto options = x.options();
    const auto int_options = options.dtype(torch::kInt);
    const auto pair_rows = torch::empty({num_tokens * num_topk}, int_options);
    const auto psum_layout = torch::empty({num_experts}, int_options);
    const auto row_weights = torch::empty({num_rows}, options.dtype(torch::kFloat));
    const auto make_sf = [&](const int& k) {
        return torch::empty({k / 128, sf_stride}, int_options).t().slice(0, 0, num_rows);
    };
    const auto l1_acts = torch::empty({num_rows, hidden}, options.dtype(torch::kFloat8_e4m3fn));
    const auto l1_acts_sf = make_sf(hidden);
    const auto l1_out = torch::empty({num_rows, intermediate_hidden * 2}, options.dtype(torch::kBFloat16));
    const auto l2_acts = torch::empty({num_rows, intermediate_hidden}, options.dtype(torch::kFloat8_e4m3fn));
    const auto l2_acts_sf = make_sf(intermediate_hidden);
    const auto l2_out = torch::empty({num_rows, hidden}, options.dtype(torch::kBFloat16));
    const int num_tokens_per_expert = ceil_div(num_tokens * num_topk, num_experts);

    using Runtime = SM120MegaMoERuntime;
    Runtime::dispatch(num_experts, num_topk, block_m, topk_idx, pair_rows, psum_layout,
                      cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                      num_tokens);
    Runtime::gather(hidden, num_topk, x, x_sf, topk_weights, pair_rows, l1_acts, l1_acts_sf, row_weights, num_tokens, sf_stride);
    sm120_mega_moe_grouped_gemm(l1_acts, l1_acts_sf, l1_weights, l1_weights_sf, l1_out, psum_layout,
                                num_experts, num_rows, intermediate_hidden * 2, hidden, block_m, num_tokens_per_expert);
    Runtime::swiglu(intermediate_hidden, num_topk, fast_math, l1_out, row_weights, pair_rows, l2_acts, l2_acts_sf,
                    activation_clamp, num_tokens, sf_stride);
    sm120_mega_moe_grouped_gemm(l2_acts, l2_acts_sf, l2_weights, l2_weights_sf, l2_out, psum_layout,
                                num_experts, num_rows, hidden, intermediate_hidden, block_m, num_tokens_per_expert);
    Runtime::combine(hidden, num_topk, l2_out, pair_rows, y, num_tokens);
}

} // namespace deep_gemm
