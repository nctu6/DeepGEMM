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

#include "kernel_lookup.hpp"
#include "runtime_utils.hpp"
#include "sm120_fp8_transpose.hpp"

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
        // Batched GEMMs: elements between the D (or floats between the SFB) of consecutive batches
        int64_t d_stride_g;
        int sfb_stride_g;
        bool use_stream_k;
        void *workspace, *counters;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        const float* sfa;
        int sfa_stride_g, sfa_stride_m, sfa_stride_k;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        // Every value that fills the template below is in the key, so the source is built on the first call only
        const auto& desc = args.gemm_desc;
        const auto& config = args.gemm_config;
        const auto kernel = get_cached_kernel(tag, std::array<int64_t, 19>{
            static_cast<int64_t>(desc.major_a), static_cast<int64_t>(desc.major_b),
            static_cast<int64_t>(args.major_sfb),
            get_compiled_dim(desc.m, 'm', desc.compiled_dims),
            get_compiled_dim(desc.n, 'n', desc.compiled_dims),
            get_compiled_dim(desc.k, 'k', desc.compiled_dims),
            desc.num_groups,
            config.layout.block_m, config.layout.block_n, config.layout.block_k,
            config.storage_config.swizzle_a_mode, config.storage_config.swizzle_b_mode,
            config.pipeline_config.num_stages,
            config.launch_config.num_tma_threads, config.launch_config.num_math_threads,
            config.launch_config.num_sms, static_cast<int64_t>(desc.gemm_type),
            args.use_stream_k, static_cast<int64_t>(desc.cd_dtype)}, [&] {
        return std::format(R"(
#include <deep_gemm/impls/sm120_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_gemm_1d2d_impl<
        {}, {}, {},
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
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
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
        to_string(args.gemm_desc.cd_dtype));
        });

        jit->launch(
            kernel, args.options,
            args.sfb, static_cast<uint32_t>(args.sfb_stride_g), args.grouped_layout,
            args.d, static_cast<uint32_t>(args.ld_d), static_cast<uint64_t>(args.d_stride_g),
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

// Per-stream stream-K partial workspace (FP32), reused by every GEMM on the stream
//  - one buffer instead of a fresh allocation per call: under CUDA graphs every captured GEMM would otherwise own its
//    own partials, so a replay cycles through all of them in DRAM instead of rewriting one L2-resident buffer
//    (measured: block-scaled FP4 o_proj M=128 with stream-K, 19 us with per-call buffers)
//  - GEMMs of one stream run in order (with PDL, a kernel still waits for its predecessor before touching memory),
//    so they can share it; separate streams get separate buffers
//  - growing allocates a new buffer and keeps the old one alive, as graphs captured earlier still point to it; the
//    allocation runs in relaxed capture mode and is never freed, like the counters
static float* get_stream_k_workspace(const size_t& num_floats) {
    static std::mutex mutex;
    static std::unordered_map<cudaStream_t, std::pair<float*, size_t>> buffers;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    std::lock_guard<std::mutex> lock(mutex);
    auto& [ptr, capacity] = buffers[stream];
    if (ptr == nullptr or capacity < num_floats) {
        // Round up generously so a few shapes settle on one buffer
        const size_t new_capacity = std::max<size_t>(num_floats, capacity * 2);
        c10::cuda::CUDAStreamCaptureModeGuard guard(cudaStreamCaptureModeRelaxed);
        DG_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, new_capacity * sizeof(float)));
        capacity = new_capacity;
    }
    return ptr;
}

// MN-major FP8 operands. The kernels read an MN-major operand in-kernel with the 8-bit transposing `ldmatrix`
// (`mma::sm120::ldmatrix_b8_x2_trans`); operands the kernels do not cover, and the cases where an explicit byte transpose
// followed by the K-major kernel is faster, take that route on the host instead (`sm120_transpose_fp8_to_k_major`).
// `DG_SM120_MN_MAJOR_FP8` (developer override): 0 auto, 1 always transpose on the host, 2 in-kernel whenever supported
//  - Supported in-kernel A: normal GEMMs only (no group offset along M), `M % 16 == 0` (TMA row stride)
//  - Supported in-kernel B: `N % 16 == 0`; grouped B (`[G, K, N]`) needs whole K blocks, as TMA bounds-checks only the
//    stacked rows
//  - Auto policy (measured, CUDA graph with cold operands, DeepSeek-V3 shapes and squares, `deepgemm-analysis.md`):
//    an MN-major B always runs in-kernel (1.4-2.0x faster than the explicit transpose at M <= 256, tied from M = 1024 up).
//    An MN-major A runs in-kernel from M = 256: the kernel's MN-major A tiles are at least 64 rows, so at M <= 64 it computes
//    mostly padding and loses to the transpose of the small activation (0.3-1.1x), while at M = 256 and up it ties or wins
//    (0.86-1.4x, 1.0-1.16x for all but one shape). M = 128 was not measured, so it keeps the transpose
static bool sm120_fp8_mn_major_supported_in_kernel(const GemmDesc& desc, const bool& is_a, const torch::Tensor& t) {
    // TMA: 16-byte aligned base and row stride
    if (t.stride(-1) % 16 != 0 or reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0)
        return false;
    if (is_a)
        return desc.gemm_type == GemmType::Normal and desc.m % 16 == 0;
    return desc.n % 16 == 0 and (desc.gemm_type == GemmType::Normal or desc.k % 128 == 0);
}

static bool sm120_fp8_mn_major_use_in_kernel(const GemmDesc& desc, const bool& is_a, const torch::Tensor& t) {
    const int mode = deep_jit::get_env<int>("DG_SM120_MN_MAJOR_FP8", 0);
    if (mode == 1 or not sm120_fp8_mn_major_supported_in_kernel(desc, is_a, t))
        return false;
    return mode == 2 or not is_a or desc.m >= 256;
}

// Returns the operands the kernel will read: an MN-major operand that does not run in-kernel is replaced by a compact
// K-major copy (and `desc` is updated)
static void sm120_resolve_mn_major_fp8(GemmDesc& desc, torch::Tensor& a, torch::Tensor& b) {
    if (desc.major_a == cute::UMMA::Major::MN and not sm120_fp8_mn_major_use_in_kernel(desc, true, a)) {
        a = sm120_transpose_fp8_to_k_major(a);
        desc.major_a = cute::UMMA::Major::K;
    }
    if (desc.major_b == cute::UMMA::Major::MN and not sm120_fp8_mn_major_use_in_kernel(desc, false, b)) {
        b = sm120_transpose_fp8_to_k_major(b);
        desc.major_b = cute::UMMA::Major::K;
    }
}

static void sm120_fp8_gemm_1d2d_launch(const std::string& tag,
                                       const GemmDesc& desc_in,
                                       const torch::Tensor& a_in, const torch::Tensor& sfa,
                                       const torch::Tensor& b_in, const torch::Tensor& sfb,
                                       const torch::Tensor& d,
                                       void* grouped_layout,
                                       const int& num_groups_a, const int& num_groups_b, const int& num_groups_sfa,
                                       const cute::UMMA::Major& major_sfb) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a_in.scalar_type() == torch::kFloat8_e4m3fn and b_in.scalar_type() == torch::kFloat8_e4m3fn);
    auto desc = desc_in;
    auto a = a_in, b = b_in;
    // A batched launch reads its (possibly MN-major) operands in-kernel: the caller has checked `sm120_fp8_bmm_1d2d_supported` / `sm120_fp8_bmm_1d1d_supported`
    const bool is_batched = desc.gemm_type == GemmType::Batched;
    if (not is_batched)
        sm120_resolve_mn_major_fp8(desc, a, b);
    // The epilogue stores 4 BF16 columns (8 bytes) at a time
    DG_HOST_ASSERT(desc.n % 8 == 0 and d.stride(-2) % 4 == 0);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0);

    const auto config = get_best_config<SM120ArchSpec>(desc);
    const auto stream_k = SM120ArchSpec::get_stream_k_config(desc, config.layout);

    // Stream-K buffers: FP32 partial slots, and per-tile arrival counters that kernels reset themselves
    float* workspace = nullptr;
    int* counters = nullptr;
    if (stream_k.num_ctas > 0) {
        workspace = get_stream_k_workspace(static_cast<size_t>(stream_k.num_slots) * desc.m * desc.n);
        counters = get_stream_k_counters();
    }

    // Requires no TMA splits: a K-major row is one swizzle atom, an MN-major atom is 64 or 128 MN bytes
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == (desc.major_a == cute::UMMA::Major::K ? config.layout.block_k :
                                                            std::min(128, config.layout.block_m)));
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == (desc.major_b == cute::UMMA::Major::K ? config.layout.block_k :
                                                            std::min(128, config.layout.block_n)));
    const auto tensor_map_a = is_batched ?
        make_tma_batched_desc(desc.major_a, a, desc.m, desc.k, config.storage_config.load_block_m, config.layout.block_k,
                              config.storage_config.swizzle_a_mode) :
        make_tma_a_desc(desc.major_a, a, desc.m, desc.k,
                        config.storage_config.load_block_m,
                        config.layout.block_k,
                        static_cast<int>(a.stride(get_non_contiguous_dim(desc.major_a))), num_groups_a,
                        config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = is_batched ?
        make_tma_batched_desc(desc.major_b, b, desc.n, desc.k, config.storage_config.load_block_n, config.layout.block_k,
                              config.storage_config.swizzle_b_mode) :
        make_tma_b_desc(desc.major_b, b, desc.n, desc.k,
                        config.storage_config.load_block_n,
                        config.layout.block_k,
                        static_cast<int>(b.stride(get_non_contiguous_dim(desc.major_b))), num_groups_b,
                        config.storage_config.swizzle_b_mode);
    // SFA is read with plain loads at any strides: `[(G,) M, K / 128]` K-major or MN-major (TMA-aligned)
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfa.dim() == (num_groups_sfa > 1 or is_batched ? 3 : 2));
    const int sfa_stride_g = num_groups_sfa > 1 or is_batched ? static_cast<int>(sfa.stride(0)) : 0;
    // SFB is read per batch as a compact `[N / 128, K / 128]` block (either major), batches `sfb_stride_g` floats apart
    if (is_batched)
        DG_HOST_ASSERT(sfb.dim() == 3 and sfb.size(0) == desc.num_groups);
    const int sfb_stride_g = is_batched ? static_cast<int>(sfb.stride(0)) : 0;

    SM120FP8Gemm1D2DRuntime::compile_and_launch(tag, {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            // Data-parallel normal GEMMs launch one CTA per tile when there are fewer tiles than SMs (idle persistent
            // CTAs still pay for dispatch and their prologue, and sit between this kernel and the next under PDL)
            .grid_dim = dim3(stream_k.num_ctas > 0 ? stream_k.num_ctas :
                             (desc.gemm_type == GemmType::Normal or is_batched ?
                                  std::min(config.launch_config.num_sms, SM120ArchSpec::get_num_tiles(desc, config.layout)) :
                                  config.launch_config.num_sms), 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = grouped_layout,
        .d = d.data_ptr(),
        .ld_d = static_cast<int>(d.stride(-2)),
        .d_stride_g = is_batched ? d.stride(0) : 0,
        .sfb_stride_g = sfb_stride_g,
        .use_stream_k = stream_k.num_ctas > 0,
        .workspace = workspace,
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

// `DG_SM120_BATCHED_LAUNCH=0` (developer override) makes every batched caller run one GEMM per batch instead, for A/B timing
// and to check that both give identical results
static bool sm120_batched_launch_enabled() {
    return deep_jit::get_env<int>("DG_SM120_BATCHED_LAUNCH", 1) != 0;
}

// Batched GEMM `D[g] = A[g] @ B[g]^T` in one launch (`a`: `[G, M, K]`, `b`: `[G, N, K]`, `d`: `[G, M, N]`, with FP32
// scales `sfa` `[G, M, K / 128]` and `sfb` `[G, N / 128, K / 128]`), on strided views of any layout the 3D tensor maps can
// describe. Operands are read in either major in-kernel. Callers check `sm120_fp8_bmm_1d2d_supported` and run a GEMM per
// batch when the views are not supported
static bool sm120_fp8_bmm_1d2d_supported(const torch::Tensor& a, const torch::Tensor& sfa,
                                         const torch::Tensor& b, const torch::Tensor& sfb, const torch::Tensor& d,
                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b) {
    // SFB is read as a compact `[N / 128, K / 128]` block per batch, K-major or MN-major
    const bool is_sfb_compact = sfb.dim() == 3 and
        ((sfb.stride(2) == 1 and sfb.stride(1) == sfb.size(2)) or (sfb.stride(1) == 1 and sfb.stride(2) == sfb.size(1)));
    // The epilogue stores 8 bytes at a time, of rows `ld_d` apart in batches `d.stride(0)` apart
    return sm120_batched_launch_enabled() and is_tma_batched_operand(major_a, a) and is_tma_batched_operand(major_b, b) and
           sfa.dim() == 3 and is_sfb_compact and
           d.dim() == 3 and d.stride(-1) == 1 and d.stride(-2) % 4 == 0 and d.stride(0) % 4 == 0 and
           d.size(-1) % 8 == 0 and reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0;
}

static void sm120_fp8_bmm_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const torch::Tensor& d,
                               const int& batch_size, const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                               const cute::UMMA::Major& major_sfb, const std::string& compiled_dims) {
    DG_HOST_ASSERT(sm120_fp8_bmm_1d2d_supported(a, sfa, b, sfb, d, major_a, major_b));
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    sm120_fp8_gemm_1d2d_launch("sm120_fp8_bmm_1d2d", desc, a, sfa, b, sfb, d, nullptr, 1, 1, batch_size, major_sfb);
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
