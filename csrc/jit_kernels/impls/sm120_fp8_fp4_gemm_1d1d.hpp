#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/sm120.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

// SM120 block-scaled GEMM with packed UE8M0 scales (FP4, or FP8 with 32/128-element scales)
class SM120FP8FP4Gemm1D1DRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;

        bool is_fp4;
        int gran_k;
        void *grouped_layout, *d;
        int ld_d;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        const void *sfa, *sfb;
        int sfa_stride_g, sfa_stride_m, sfa_stride_k;
        int sfb_stride_g, sfb_stride_n, sfb_stride_k;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
        args.is_fp4 ? "true" : "false", args.gran_k,
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        to_string(args.gemm_desc.cd_dtype)));

        jit->launch(
            kernel, args.options,
            args.grouped_layout, args.d, static_cast<uint32_t>(args.ld_d),
            args.sfa, static_cast<uint32_t>(args.sfa_stride_g), static_cast<uint32_t>(args.sfa_stride_m),
            static_cast<uint32_t>(args.sfa_stride_k),
            args.sfb, static_cast<uint32_t>(args.sfb_stride_g), static_cast<uint32_t>(args.sfb_stride_n),
            static_cast<uint32_t>(args.sfb_stride_k),
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b
        );
    }
};

static void sm120_fp8_fp4_gemm_1d1d_launch(const std::string& tag,
                                           const GemmDesc& desc,
                                           const torch::Tensor& a, const torch::Tensor& sfa,
                                           const torch::Tensor& b, const torch::Tensor& sfb,
                                           const torch::Tensor& d,
                                           void* grouped_layout,
                                           const int& num_groups_a, const int& num_groups_b,
                                           const int& gran_k_a, const int& gran_k_b) {
    // `desc.k` counts logical elements, so FP4 (packed two per byte) has `a.size(-1) * 2 == k`
    const bool is_fp4 = a.scalar_type() == kPackedFP4;
    DG_HOST_ASSERT(is_fp4 == (b.scalar_type() == kPackedFP4));
    DG_HOST_ASSERT(gran_k_a == gran_k_b);
    DG_HOST_ASSERT(is_fp4 ? gran_k_a == 32 : (gran_k_a == 32 or gran_k_a == 128));
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K and desc.major_b == cute::UMMA::Major::K);
    // K is whole scale groups (the packed scale tensors are padded, and TMA zero-fills the tail K block), and the
    // epilogue stores 4 BF16 columns (8 bytes) at a time
    DG_HOST_ASSERT(desc.k % gran_k_a == 0);
    DG_HOST_ASSERT(desc.n % 8 == 0 and d.stride(-2) % 4 == 0);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt);

    const auto config = get_best_config<SM120ArchSpec>(desc);

    // Requires no TMA splits: a tile row is exactly one 128-byte swizzle atom
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto tensor_map_a = make_tma_a_desc(desc.major_a, a, desc.m, desc.k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(desc.major_a))), num_groups_a,
                                              config.storage_config.swizzle_a_mode, 0, false,
                                              /* fp4_unpacked_smem */ false);
    const auto tensor_map_b = make_tma_b_desc(desc.major_b, b, desc.n, desc.k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(desc.major_b))), num_groups_b,
                                              config.storage_config.swizzle_b_mode, 0, false,
                                              /* fp4_unpacked_smem */ false);

    // Scales are read with plain loads at any strides: `[(G,) MN, K / (gran_k * 4)]` int32, MN-major (TMA-aligned)
    const bool sfa_grouped = sfa.dim() == 3, sfb_grouped = sfb.dim() == 3;
    SM120FP8FP4Gemm1D1DRuntime::compile_and_launch(tag, {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .is_fp4 = is_fp4,
        .gran_k = gran_k_a,
        .grouped_layout = grouped_layout,
        .d = d.data_ptr(),
        .ld_d = static_cast<int>(d.stride(-2)),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .sfa = sfa.data_ptr(),
        .sfb = sfb.data_ptr(),
        .sfa_stride_g = sfa_grouped ? static_cast<int>(sfa.stride(0)) : 0,
        .sfa_stride_m = static_cast<int>(sfa.stride(-2)),
        .sfa_stride_k = static_cast<int>(sfa.stride(-1)),
        .sfb_stride_g = sfb_grouped ? static_cast<int>(sfb.stride(0)) : 0,
        .sfb_stride_n = static_cast<int>(sfb.stride(-2)),
        .sfb_stride_k = static_cast<int>(sfb.stride(-1)),
    });
}

static void sm120_fp8_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                    const std::optional<torch::Tensor>& c,
                                    const torch::Tensor& d,
                                    const int& m, const int& n, const int& k,
                                    const int& gran_k_a, const int& gran_k_b,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims) {
    DG_HOST_ASSERT(not c.has_value() and "SM120 block-scaled GEMM does not support accumulation");
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_fp8_fp4_gemm_1d1d", desc, a, sfa, b, sfb, d, nullptr, 1, 1, gran_k_a, gran_k_b);
}

static void sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                                         const torch::Tensor& d,
                                                         const torch::Tensor& m_indices,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout) {
    DG_HOST_ASSERT(not use_psum_layout and "SM120 block-scaled GEMM does not support the PSUM layout");
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = k,
        .expected_num_groups = 1
    };
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d", desc, a, sfa, b, sfb, d,
                                   m_indices.data_ptr(), 1, num_groups, gran_k_a, gran_k_b);
}

static void sm120_m_grouped_fp8_fp4_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& masked_m,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const int& expected_m,
                                                     const int& gran_k_a, const int& gran_k_b,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D1D,
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
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_m_grouped_fp8_fp4_gemm_masked_1d1d", desc, a, sfa, b, sfb, d,
                                   masked_m.data_ptr(), num_groups, num_groups, gran_k_a, gran_k_b);
}

} // namespace deep_gemm
