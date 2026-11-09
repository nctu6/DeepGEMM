#pragma once
#include <format>
#include <functional>
#include <torch/python.h>
#include <deep_gemm/layout/sym_buffer.cuh>
#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"
#include "sm120_fp8_fp4_gemm_1d1d.hpp"
namespace deep_gemm {
// SM120 `fp8_fp4_mega_moe` on one or more ranks: push, barrier, dispatch, gather, the two m-grouped FP8 x FP4 GEMMs,
// SwiGLU, return, barrier and combine as a sequence of kernels with the SM100 numerics (see `sm120_fp8_fp4_mega_moe.cuh`
// for the steps and why). Ranks write into each other's symmetric buffers with peer stores over CUDA IPC (PCIe P2P on
// SM120 workstations). Calls that share a buffer must be ordered on one stream per rank (the barrier epoch is in it).
// With several ranks the intermediate rows are sized for the worst routing of `num_max_tokens_per_rank` on every rank

// Symmetric buffer of the SM120 path. The control block is first, so the barrier state does not move when a buffer is
// reused for another configuration. Several ranks add what peers write: the receive area (every rank's top-k lists and
// the FP8 rows and scales that reach this rank's experts) and the combine buffer (the L2 row of every (token, slot) pair)
struct SM120MegaMoESymLayout {
    static constexpr int kMaxRanks = 8, kCtrlBytes = 256, kMaxBlockM = 128, kMaxPairBits = 27, kMaxRankBits = 23;
    int64_t ctrl = 0, x = 0, x_sf = 0, topk_idx = 0, topk_weights = 0;
    int64_t recv_x = 0, recv_x_sf = 0, recv_topk_idx = 0, recv_topk_weights = 0, combine = 0, num_bytes = 0;

    SM120MegaMoESymLayout(const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank,
                          const int& num_topk, const int& hidden) {
        DG_HOST_ASSERT(num_ranks >= 1 and num_ranks <= kMaxRanks and num_experts % num_ranks == 0);
        DG_HOST_ASSERT(static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank < (1 << kMaxRankBits));
        DG_HOST_ASSERT(static_cast<int64_t>(num_max_tokens_per_rank) * num_topk < (1 << kMaxPairBits));
        const auto place = [&](int64_t& offset, const int64_t& size) {
            offset = num_bytes;
            num_bytes += align<int64_t>(size, 256);
        };
        const int64_t tokens = num_max_tokens_per_rank;
        place(ctrl, kCtrlBytes);
        place(x, tokens * hidden);
        place(x_sf, tokens * (hidden / 128) * 4);
        place(topk_idx, tokens * num_topk * 8);
        place(topk_weights, tokens * num_topk * 4);
        if (num_ranks > 1) {
            place(recv_x, num_ranks * tokens * hidden);
            place(recv_x_sf, num_ranks * tokens * (hidden / 128) * 4);
            place(recv_topk_idx, num_ranks * tokens * num_topk * 8);
            place(recv_topk_weights, num_ranks * tokens * num_topk * 4);
            place(combine, tokens * num_topk * hidden * 2);
        }
    }

    // Rows that the runs of the local experts can take for at most `num_pairs` routed pairs: every pair, plus the tail of
    // each non-empty run
    static int max_num_rows(const int64_t& num_pairs, const int& num_local_experts, const int& block_m) {
        return static_cast<int>(align<int64_t>(num_pairs + std::min<int64_t>(num_local_experts, num_pairs) * (block_m - 1), block_m));
    }
};

using SM120MegaMoESlices = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

static std::tuple<int64_t, std::function<SM120MegaMoESlices(const torch::Tensor&)>>
sm120_get_symm_buffer_size_for_mega_moe(const int& num_ranks, const int& num_experts,
                                        const int& num_max_tokens_per_rank, const int& num_topk,
                                        const int& hidden, const int& intermediate_hidden,
                                        const std::string& mma_type, const int& num_shared_experts) {
    DG_HOST_ASSERT(mma_type == "fp8xfp4" and "SM120 mega MoE takes FP8 activations and MXFP4 weights");
    DG_HOST_ASSERT(num_shared_experts == 0 and "SM120 mega MoE has no fused shared experts");
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 128 == 0);
    const SM120MegaMoESymLayout symm(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden);
    // The inputs have the SM100 views; the rest of the SM100 buffers (rings, shared experts) do not exist here
    const auto slice = [=](const torch::Tensor& buffer) {
        const auto view = [&](const int64_t& offset, const std::vector<int64_t>& shape, const torch::ScalarType& dtype) {
            return torch::from_blob(math::advance_ptr(buffer.data_ptr(), offset), shape,
                                    torch::TensorOptions().dtype(dtype).device(buffer.device()));
        };
        const auto x = view(symm.x, {num_max_tokens_per_rank, hidden}, torch::kFloat8_e4m3fn);
        const auto x_sf = view(symm.x_sf, {num_max_tokens_per_rank, hidden / 128}, torch::kInt);
        const auto topk_idx = view(symm.topk_idx, {num_max_tokens_per_rank, num_topk}, torch::kInt64);
        const auto topk_weights = view(symm.topk_weights, {num_max_tokens_per_rank, num_topk}, torch::kFloat);
        const torch::Tensor none;
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, none, none, none, none, none, none, none, none);
    };
    return {symm.num_bytes, slice};
}

class SM120MegaMoERuntime final {
public:
    static constexpr int kDispatchThreads = 1024, kNumWarps = 8, kCombineThreads = 128;
    static deep_jit::cuda::LaunchOptions options(const int& num_blocks, const int& num_threads) {
        return {.num_smem_bytes = 0, .grid_dim = dim3(num_blocks, 1, 1), .block_dim = dim3(num_threads, 1, 1),
                .cluster_dim = dim3(1, 1, 1)};
    }
    // A grid-stride kernel: any grid gives the same result, so size it for the expected work
    static int grid_for(const int64_t& num_tasks, const int& tasks_per_block) {
        const int64_t max_blocks = static_cast<int64_t>(runtime->get_num_sms()) * 16;
        return static_cast<int>(std::max<int64_t>(1, std::min(max_blocks, ceil_div<int64_t>(num_tasks, tasks_per_block))));
    }
    static std::string include() {
        return "#include <deep_gemm/impls/sm120_fp8_fp4_mega_moe.cuh>\nusing namespace deep_gemm;\n";
    }
    template <typename... Args>
    static void launch(const std::string& name, const std::string& instantiation, const int& num_blocks, const int& num_threads,
                       Args&&... args) {
        const auto kernel = jit->compile(name, std::format(R"({}
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}};
)", include(), instantiation));
        jit->launch(kernel, options(num_blocks, num_threads), std::forward<Args>(args)...);
    }
    static void push(const int& num_ranks, const int& num_local_experts, const int& hidden, const int& num_topk, const int& num_blocks,
                     const layout::SymBuffer<>& sym, const void* x, const void* x_sf, const void* topk_idx, const void* topk_weights,
                     void* recv_x, void* recv_x_sf, void* recv_topk_idx, void* recv_topk_weights,
                     const int& num_tokens, const int& num_max_tokens) {
        launch("sm120_mega_moe_push", std::format("sm120_mega_moe_push<{}, {}, {}, {}, {}>", num_ranks, num_local_experts, hidden, num_topk, kNumWarps),
               num_blocks, kNumWarps * 32, sym, x, x_sf, topk_idx, topk_weights, recv_x, recv_x_sf, recv_topk_idx, recv_topk_weights,
               static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(num_max_tokens));
    }
    static void barrier(const int& num_ranks, const layout::SymBuffer<>& sym, void* ctrl, const int& num_tokens) {
        // A rank that never arrives would hang its peers: trap after DG_SM120_COMM_TIMEOUT_S seconds (at most 3 GHz)
        const uint64_t timeout_cycles = static_cast<uint64_t>(deep_jit::get_env<int>("DG_SM120_COMM_TIMEOUT_S", 120)) * 3000000000ull;
        launch("sm120_mega_moe_barrier", std::format("sm120_mega_moe_barrier<{}>", num_ranks), 1, 32,
               sym, ctrl, static_cast<uint32_t>(num_tokens), timeout_cycles);
    }
    static void dispatch(const int& num_ranks, const int& num_local_experts, const int& num_topk, const int& block_m,
                         const layout::SymBuffer<>& sym, const void* topk_idx, const void* recv_topk_idx, const void* ctrl,
                         void* pair_rows, const int& pair_stride, const torch::Tensor& row_pair, const torch::Tensor& psum_layout,
                         int* expert_stats, const torch::Tensor& total_rows, const int& num_tokens, const int& num_max_tokens) {
        launch("sm120_mega_moe_dispatch",
               std::format("sm120_mega_moe_dispatch<{}, {}, {}, {}, {}>", num_ranks, num_local_experts, num_topk, block_m, kDispatchThreads),
               1, kDispatchThreads, sym, topk_idx, recv_topk_idx, ctrl, pair_rows, static_cast<uint32_t>(pair_stride),
               row_pair.data_ptr(), psum_layout.data_ptr(), expert_stats, total_rows.data_ptr(),
               static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(num_max_tokens));
    }
    static void gather(const int& num_ranks, const int& hidden, const int& num_topk, const int& num_blocks,
                       const layout::SymBuffer<>& sym, const void* x, const void* x_sf, const void* topk_weights,
                       const void* recv_x, const void* recv_x_sf, const void* recv_topk_weights,
                       const torch::Tensor& row_pair, const torch::Tensor& total_rows,
                       const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf, const torch::Tensor& row_weights,
                       const int& sf_stride, const int& num_max_tokens) {
        launch("sm120_mega_moe_gather", std::format("sm120_mega_moe_gather<{}, {}, {}, {}>", num_ranks, hidden, num_topk, kNumWarps),
               num_blocks, kNumWarps * 32, sym, x, x_sf, topk_weights, recv_x, recv_x_sf, recv_topk_weights,
               row_pair.data_ptr(), total_rows.data_ptr(), l1_acts.data_ptr(), l1_acts_sf.data_ptr(), row_weights.data_ptr(),
               static_cast<uint32_t>(sf_stride), static_cast<uint32_t>(num_max_tokens));
    }
    static void swiglu(const int& intermediate_hidden, const bool& fast_math, const int& num_blocks,
                       const torch::Tensor& l1_out, const torch::Tensor& row_weights, const torch::Tensor& row_pair,
                       const torch::Tensor& total_rows, const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
                       const float& activation_clamp, const int& sf_stride) {
        launch("sm120_mega_moe_swiglu", std::format("sm120_mega_moe_swiglu<{}, {}, {}>", intermediate_hidden, kNumWarps, fast_math ? "true" : "false"),
               num_blocks, kNumWarps * 32, l1_out.data_ptr(), row_weights.data_ptr(), row_pair.data_ptr(), total_rows.data_ptr(),
               l2_acts.data_ptr(), l2_acts_sf.data_ptr(), activation_clamp, static_cast<uint32_t>(sf_stride));
    }
    static void send_rows(const int& num_ranks, const int& hidden, const int& num_topk, const int& num_blocks,
                          const layout::SymBuffer<>& sym, const torch::Tensor& l2_out, const torch::Tensor& row_pair,
                          const torch::Tensor& total_rows, void* combine_rows) {
        launch("sm120_mega_moe_return", std::format("sm120_mega_moe_return<{}, {}, {}, {}>", num_ranks, hidden, num_topk, kNumWarps),
               num_blocks, kNumWarps * 32, sym, l2_out.data_ptr(), row_pair.data_ptr(), total_rows.data_ptr(), combine_rows);
    }
    static void combine(const int& num_ranks, const int& num_local_experts, const int& hidden, const int& num_topk,
                        const void* rows_base, const void* topk_idx, const void* pair_rows, const torch::Tensor& y, const int& num_tokens) {
        launch("sm120_mega_moe_combine",
               std::format("sm120_mega_moe_combine<{}, {}, {}, {}, {}>", num_ranks, num_local_experts, hidden, num_topk, kCombineThreads),
               num_tokens, kCombineThreads, rows_base, topk_idx, pair_rows, y.data_ptr());
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

// `num_experts` is the global count; this rank owns the experts `[rank_idx * E / R, (rank_idx + 1) * E / R)` and holds
// their weights. `y` has this rank's tokens (a rank may have none; it still serves its peers)
static void sm120_fp8_fp4_mega_moe(const torch::Tensor& y, const torch::Tensor& sym_buffer,
                                   const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
                                   const int& num_max_tokens_per_rank,
                                   const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
                                   const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
                                   const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                                   const int& num_experts, const int& num_tokens, const int& num_topk,
                                   const int& hidden, const int& intermediate_hidden,
                                   const float& activation_clamp, const bool& fast_math) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    if (num_ranks == 1 and num_tokens == 0)
        return;
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 128 == 0);
    const int num_local_experts = num_experts / num_ranks;
    const SM120MegaMoESymLayout symm(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(symm.num_bytes));
    DG_HOST_ASSERT(num_local_experts <= 256);
    const auto base = static_cast<uint8_t*>(sym_buffer.data_ptr());
    const bool is_multi_rank = num_ranks > 1;

    // Sizing hints: the host knows only its own tokens, so grids and the tile height assume the peers have as many. The
    // kernels read the real counts on the device, so a different routing or token count changes speed, not results
    const int num_est_tokens = num_tokens * num_ranks;
    const int forced_block_m = deep_jit::get_env<int>("DG_SM120_MEGA_BLOCK_M", 0);
    const int block_m = forced_block_m > 0 ? forced_block_m : sm120_get_block_m_for_mega_moe(std::max(1, num_est_tokens), num_topk, num_experts);
    // Rows the runs can take at most: exact for one rank, the worst routing of the largest batch for several
    const int64_t num_pair_bound = static_cast<int64_t>(is_multi_rank ? num_ranks * num_max_tokens_per_rank : num_tokens) *
                                   std::min(num_topk, num_local_experts);
    const int num_rows = SM120MegaMoESymLayout::max_num_rows(num_pair_bound, num_local_experts, block_m);
    const int num_est_rows = std::min<int64_t>(num_rows, SM120MegaMoESymLayout::max_num_rows(
        static_cast<int64_t>(num_est_tokens) * std::min(num_topk, num_local_experts), num_local_experts, block_m));
    // Scales are MN-major (`[K / 128][sf_stride]` int32), with 16-byte aligned columns for TMA
    const int sf_stride = align(num_rows, 4);
    const auto options = y.options();
    const auto int_options = options.dtype(torch::kInt);
    const int pair_stride = (is_multi_rank ? num_max_tokens_per_rank : num_tokens) * num_topk;
    const auto pair_rows_tensor = torch::empty({std::max<int64_t>(1, static_cast<int64_t>(num_ranks) * pair_stride)}, int_options);
    void* pair_rows = pair_rows_tensor.data_ptr();
    const auto row_pair = torch::empty({num_rows}, int_options);
    const auto total_rows = torch::empty({1}, int_options);
    const auto psum_layout = torch::empty({num_local_experts}, int_options);
    const auto row_weights = torch::empty({num_rows}, options.dtype(torch::kFloat));
    const auto make_sf = [&](const int& k) {
        return torch::empty({k / 128, sf_stride}, int_options).t().slice(0, 0, num_rows);
    };
    const auto l1_acts = torch::empty({num_rows, hidden}, options.dtype(torch::kFloat8_e4m3fn));
    const auto l1_acts_sf = make_sf(hidden);
    const auto l1_out = torch::empty({num_rows, intermediate_hidden * 2}, options.dtype(torch::kBFloat16));
    const auto l2_acts = torch::empty({num_rows, intermediate_hidden}, options.dtype(torch::kFloat8_e4m3fn));
    const auto l2_acts_sf = make_sf(intermediate_hidden);
    // With several ranks the return kernel sends these rows to the combine buffer of each pair's source rank
    const auto l2_out = torch::empty({num_rows, hidden}, options.dtype(torch::kBFloat16));
    const int num_tokens_per_expert = ceil_div(num_est_tokens * num_topk, num_experts);

    const layout::SymBuffer<> sym(sym_buffer_ptrs, rank_idx);
    const auto ctrl = base + symm.ctrl;
    const auto x = base + symm.x, x_sf = base + symm.x_sf, topk_idx = base + symm.topk_idx, topk_weights = base + symm.topk_weights;
    // Null with one rank
    const auto recv_x = is_multi_rank ? base + symm.recv_x : nullptr, recv_x_sf = is_multi_rank ? base + symm.recv_x_sf : nullptr;
    const auto recv_topk_idx = is_multi_rank ? base + symm.recv_topk_idx : nullptr;
    const auto recv_topk_weights = is_multi_rank ? base + symm.recv_topk_weights : nullptr;
    const auto combine_rows = is_multi_rank ? base + symm.combine : nullptr;
    const auto stats = cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
    const int num_push_tasks = num_tokens * (num_ranks - 1);

    using Runtime = SM120MegaMoERuntime;
    // Peers write into this rank's receive area and combine buffer: the barrier makes the pushes visible, and proves that
    // the peers are done with the previous call's rows before the next call overwrites them
    if (is_multi_rank) {
        if (num_tokens > 0)
            Runtime::push(num_ranks, num_local_experts, hidden, num_topk, Runtime::grid_for(num_push_tasks, Runtime::kNumWarps), sym,
                          x, x_sf, topk_idx, topk_weights, recv_x, recv_x_sf, recv_topk_idx, recv_topk_weights,
                          num_tokens, num_max_tokens_per_rank);
        Runtime::barrier(num_ranks, sym, ctrl, num_tokens);
    }
    Runtime::dispatch(num_ranks, num_local_experts, num_topk, block_m, sym, topk_idx, recv_topk_idx, ctrl, pair_rows,
                      pair_stride, row_pair, psum_layout, stats, total_rows, num_tokens, num_max_tokens_per_rank);
    Runtime::gather(num_ranks, hidden, num_topk, Runtime::grid_for(num_est_rows, Runtime::kNumWarps), sym, x, x_sf, topk_weights,
                    recv_x, recv_x_sf, recv_topk_weights, row_pair, total_rows, l1_acts, l1_acts_sf, row_weights, sf_stride,
                    num_max_tokens_per_rank);
    sm120_mega_moe_grouped_gemm(l1_acts, l1_acts_sf, l1_weights, l1_weights_sf, l1_out, psum_layout,
                                num_local_experts, num_rows, intermediate_hidden * 2, hidden, block_m, num_tokens_per_expert);
    Runtime::swiglu(intermediate_hidden, fast_math,
                    Runtime::grid_for(static_cast<int64_t>(num_est_rows) * (intermediate_hidden / 128), Runtime::kNumWarps),
                    l1_out, row_weights, row_pair, total_rows, l2_acts, l2_acts_sf, activation_clamp, sf_stride);
    sm120_mega_moe_grouped_gemm(l2_acts, l2_acts_sf, l2_weights, l2_weights_sf, l2_out, psum_layout,
                                num_local_experts, num_rows, hidden, intermediate_hidden, block_m, num_tokens_per_expert);
    if (is_multi_rank) {
        Runtime::send_rows(num_ranks, hidden, num_topk, Runtime::grid_for(num_est_rows, Runtime::kNumWarps), sym, l2_out, row_pair,
                           total_rows, combine_rows);
        Runtime::barrier(num_ranks, sym, ctrl, num_tokens);
    }
    if (num_tokens > 0)
        Runtime::combine(num_ranks, num_local_experts, hidden, num_topk, is_multi_rank ? static_cast<const void*>(combine_rows) : l2_out.data_ptr(),
                         topk_idx, pair_rows, y, num_tokens);
}

} // namespace deep_gemm
