#pragma once

#include <format>
#include <unordered_map>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <mutex>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/sm120.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

class SM120FP8Gemm1D2DRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;

        cute::UMMA::Major major_sfb;
        void *sfb, *grouped_layout, *d;
        int ld_d;
        bool use_stream_k;
        void *workspace, *counters;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        const float* sfa;
        int sfa_stride_g, sfa_stride_m, sfa_stride_k;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm120_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_gemm_1d2d_impl<
        {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}
    >);
}};
)",
        to_string(args.major_sfb),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        args.use_stream_k ? "true" : "false",
        to_string(args.gemm_desc.cd_dtype)));

        jit->launch(
            kernel, args.options,
            args.sfb, args.grouped_layout, args.d, static_cast<uint32_t>(args.ld_d),
            args.workspace, args.counters,
            args.sfa, static_cast<uint32_t>(args.sfa_stride_g), static_cast<uint32_t>(args.sfa_stride_m),
            static_cast<uint32_t>(args.sfa_stride_k),
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b
        );
    }
};

// Per-stream stream-K arrival counters (one per tile)
//  - every shared tile resets its own counter after reducing, so counters stay zero between launches,
//    including across CUDA graph replays
//  - separate streams get separate counters, as their GEMMs may run concurrently
//  - the memory is a plain `cudaMalloc` that is never freed (not from a graph-private pool), so a first call
//    inside a CUDA graph capture is safe: allocation runs in relaxed capture mode, and the zeroing memset is
//    issued on the current stream (it becomes a harmless memset node if that stream is capturing)
static int* get_stream_k_counters() {
    static constexpr int kNumCounters = SM120ArchSpec::kNumMaxStreamKTiles;
    static std::mutex mutex;
    static std::unordered_map<cudaStream_t, int*> counters;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = counters.find(stream); it != counters.end())
        return it->second;
    int* ptr = nullptr;
    {
        c10::cuda::CUDAStreamCaptureModeGuard guard(cudaStreamCaptureModeRelaxed);
        DG_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, kNumCounters * sizeof(int)));
    }
    DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(ptr, 0, kNumCounters * sizeof(int), stream));
    counters[stream] = ptr;
    return ptr;
}

static void sm120_fp8_gemm_1d2d_launch(const std::string& tag,
                                       const GemmDesc& desc,
                                       const torch::Tensor& a, const torch::Tensor& sfa,
                                       const torch::Tensor& b, const torch::Tensor& sfb,
                                       const torch::Tensor& d,
                                       void* grouped_layout,
                                       const int& num_groups_a, const int& num_groups_b, const int& num_groups_sfa,
                                       const cute::UMMA::Major& major_sfb) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K and desc.major_b == cute::UMMA::Major::K);
    // The epilogue stores 4 BF16 columns (8 bytes) at a time
    DG_HOST_ASSERT(desc.n % 8 == 0 and d.stride(-2) % 4 == 0);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0);

    const auto config = get_best_config<SM120ArchSpec>(desc);
    const auto stream_k = SM120ArchSpec::get_stream_k_config(desc, config.layout);

    // Stream-K buffers: FP32 partial slots, and per-tile arrival counters that kernels reset themselves
    std::optional<torch::Tensor> workspace;
    int* counters = nullptr;
    if (stream_k.num_ctas > 0) {
        workspace = torch::empty({stream_k.num_slots, desc.m, desc.n}, d.options().dtype(torch::kFloat));
        counters = get_stream_k_counters();
    }

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto tensor_map_a = make_tma_a_desc(desc.major_a, a, desc.m, desc.k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(desc.major_a))), num_groups_a,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(desc.major_b, b, desc.n, desc.k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(desc.major_b))), num_groups_b,
                                              config.storage_config.swizzle_b_mode);
    // SFA is read with plain loads at any strides: `[(G,) M, K / 128]` K-major or MN-major (TMA-aligned)
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.dim() == (num_groups_sfa > 1 ? 3 : 2));
    const int sfa_stride_g = num_groups_sfa > 1 ? static_cast<int>(sfa.stride(0)) : 0;

    SM120FP8Gemm1D2DRuntime::compile_and_launch(tag, {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(stream_k.num_ctas > 0 ? stream_k.num_ctas : config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = grouped_layout,
        .d = d.data_ptr(),
        .ld_d = static_cast<int>(d.stride(-2)),
        .use_stream_k = stream_k.num_ctas > 0,
        .workspace = workspace.has_value() ? workspace->data_ptr() : nullptr,
        .counters = counters,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .sfa = sfa.data_ptr<float>(),
        .sfa_stride_g = sfa_stride_g,
        .sfa_stride_m = static_cast<int>(sfa.stride(-2)),
        .sfa_stride_k = static_cast<int>(sfa.stride(-1)),
    });
}

static void sm120_fp8_gemm_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                const std::string& compiled_dims) {
    DG_HOST_ASSERT(not c.has_value() and "SM120 FP8 GEMM does not support accumulation");
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    sm120_fp8_gemm_1d2d_launch("sm120_fp8_gemm_1d2d", desc, a, sfa, b, sfb, d, nullptr, 1, 1, 1, major_sfb);
}

static void sm120_m_grouped_fp8_gemm_contiguous_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& m_indices,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                                     const std::string& compiled_dims,
                                                     const bool& use_psum_layout,
                                                     const std::optional<int>& expected_m_for_psum_layout) {
    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);
    const auto desc = GemmDesc {
        .gemm_type = use_psum_layout ? GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    sm120_fp8_gemm_1d2d_launch("sm120_m_grouped_fp8_gemm_contiguous_1d2d", desc, a, sfa, b, sfb, d,
                               m_indices.data_ptr(), 1, num_groups, 1, major_sfb);
}

static void sm120_m_grouped_fp8_gemm_masked_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                 const torch::Tensor& b, const torch::Tensor& sfb,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& masked_m,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const int& expected_m,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                                 const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    sm120_fp8_gemm_1d2d_launch("sm120_fp8_m_grouped_gemm_masked_1d2d", desc, a, sfa, b, sfb, d,
                               masked_m.data_ptr(), num_groups, num_groups, num_groups, major_sfb);
}

} // namespace deep_gemm
