#pragma once

#include <format>
#include <functional>
#include <algorithm>
#include <limits>
#include <vector>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include <deep_gemm/impls/sm120_mega_moe_epilogue.cuh>
#include "../heuristics/sm120.hpp"

#include "runtime_utils.hpp"
#include "sm120_fp8_gemm_1d2d.hpp"

namespace deep_gemm {

// SM120 `mma.sync` 1D1D GEMM: block-scaled with packed UE8M0 scales (FP4, or FP8 with 32/128-element scales), or
// BF16 without scales
class SM120FP8FP4Gemm1D1DRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;

        bool is_fp4, is_bf16;
        bool is_mixed_fp4_b;   // FP8 A against packed FP4 B (W4A8)
        int gran_k;
        bool use_stream_k;
        bool early_pdl_trigger;
        void *grouped_layout, *d;
        int ld_d;
        int64_t d_stride_g;   // Batched: elements between the D of consecutive batches
        float alpha;
        void *workspace, *counters;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        // Fused epilogue of the mega MoE: its kind, math mode and what it reads and writes
        SM120MegaMoEEpilogue epilogue = SM120MegaMoEEpilogue::None;
        bool epilogue_fast_math = false;
        SM120MegaMoEEpilogueArgs epilogue_args = {};
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_gemm_1d1d_impl<
        {}, {}, {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        SM120MegaMoEEpilogue({}), {},
        {}
    >);
}};
)",
        args.is_fp4 ? "true" : "false", args.is_bf16 ? "true" : "false", args.is_mixed_fp4_b ? "true" : "false", args.gran_k,
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        args.use_stream_k ? "true" : "false", args.gemm_desc.with_accumulation ? "true" : "false",
        args.early_pdl_trigger ? "true" : "false",
        static_cast<uint32_t>(args.epilogue), args.epilogue_fast_math ? "true" : "false",
        to_string(args.gemm_desc.cd_dtype)));

        jit->launch(
            kernel, args.options,
            args.grouped_layout, args.d, static_cast<uint32_t>(args.ld_d), static_cast<uint64_t>(args.d_stride_g), args.alpha,
            args.workspace, args.counters,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_sfa, args.tensor_map_sfb,
            args.epilogue_args
        );
    }
};

// A fused epilogue of the mega MoE (`kind` `None`: store D)
struct SM120GemmEpilogue {
    SM120MegaMoEEpilogue kind = SM120MegaMoEEpilogue::None;
    bool fast_math = false;
    SM120MegaMoEEpilogueArgs args = {};
};

// A tile layout plus a stream-K choice: what the autotuner selects between
struct SM120BlockScaledChoice {
    Layout layout;
    bool use_stream_k;
};

static GemmConfig sm120_make_config(const GemmDesc& desc, const Layout& layout) {
    const auto storage_config = SM120ArchSpec::get_storage_config(desc, layout);
    return GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = SM120ArchSpec::get_pipeline_config(desc, layout, storage_config),
        .launch_config = SM120ArchSpec::get_launch_config(desc, layout)
    };
}

// Launch one explicit choice
static void sm120_fp8_fp4_gemm_1d1d_run(const std::string& tag,
                                        const GemmDesc& desc, const SM120BlockScaledChoice& choice,
                                        const torch::Tensor& a, const torch::Tensor& sfa,
                                        const torch::Tensor& b, const torch::Tensor& sfb,
                                        const torch::Tensor& d,
                                        void* grouped_layout,
                                        const int& num_groups_a, const int& num_groups_b,
                                        const int& num_groups_sfa, const int& num_groups_sfb,
                                        const int& gran_k, const float& alpha, const SM120GemmEpilogue& epilogue) {
    const bool is_fp4 = a.scalar_type() == kPackedFP4;
    const bool is_bf16 = a.scalar_type() == torch::kBFloat16;
    const bool is_mixed_fp4_b = not is_fp4 and b.scalar_type() == kPackedFP4;
    const auto config = sm120_make_config(desc, choice.layout);
    auto stream_k = SM120ArchSpec::StreamKConfig{};
    if (choice.use_stream_k)
        stream_k = SM120ArchSpec::get_stream_k_split(desc, choice.layout);

    // Stream-K buffers: FP32 partial slots, and per-tile arrival counters that kernels reset themselves
    float* workspace = nullptr;
    int* counters = nullptr;
    if (stream_k.num_ctas > 0) {
        workspace = get_stream_k_workspace(static_cast<size_t>(stream_k.num_slots) * desc.m * desc.n);
        counters = get_stream_k_counters();
    }

    // Requires no TMA splits: a K-major tile row is exactly one swizzle atom of `block_k` bytes (for packed FP4 the TMA
    // descriptor converts it into elements itself; BF16 rows hold `block_k / 2` elements). An MN-major atom is 64 or 128
    // MN bytes
    const bool is_batched = desc.gemm_type == GemmType::Batched;
    const int elem_bytes = is_bf16 ? 2 : 1;
    // FP8 x FP4: `block_k` is the B row (64 bytes); the A row holds as many FP8 elements, twice the bytes
    const int block_k_a = is_mixed_fp4_b ? config.layout.block_k * 2 : config.layout.block_k;
    DG_HOST_ASSERT(not is_mixed_fp4_b or config.layout.block_k == 64);
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == (desc.major_a == cute::UMMA::Major::K ? block_k_a :
                                                            std::min(128, config.layout.block_m * elem_bytes)));
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == (desc.major_b == cute::UMMA::Major::K ? config.layout.block_k :
                                                            std::min(128, config.layout.block_n * elem_bytes)));
    const int tma_block_k = is_bf16 ? config.layout.block_k / 2 : block_k_a;
    // Batched: 3D maps (inner, outer, batch) over `[G, MN, K]` views, instead of groups stacked along the outer dimension
    const auto tensor_map_a = is_batched ?
        make_tma_batched_desc(desc.major_a, a, desc.m, desc.k, config.storage_config.load_block_m, tma_block_k,
                              config.storage_config.swizzle_a_mode, /* fp4_unpacked_smem */ false) :
        make_tma_a_desc(desc.major_a, a, desc.m, desc.k,
                        config.storage_config.load_block_m,
                        tma_block_k,
                        static_cast<int>(a.stride(get_non_contiguous_dim(desc.major_a))), num_groups_a,
                        config.storage_config.swizzle_a_mode, 0, false,
                        /* fp4_unpacked_smem */ false);
    DG_HOST_ASSERT(not is_mixed_fp4_b or not is_batched);
    const auto tensor_map_b = is_batched ?
        make_tma_batched_desc(desc.major_b, b, desc.n, desc.k, config.storage_config.load_block_n, tma_block_k,
                              config.storage_config.swizzle_b_mode, /* fp4_unpacked_smem */ false) :
        make_tma_b_desc(desc.major_b, b, desc.n, desc.k,
                        config.storage_config.load_block_n,
                        tma_block_k,
                        static_cast<int>(b.stride(get_non_contiguous_dim(desc.major_b))), num_groups_b,
                        config.storage_config.swizzle_b_mode, 0, false,
                        /* fp4_unpacked_smem */ false);

    // Scales are TMA-loaded with each stage: `[(G,) MN, ceil(K / (gran_k * 4))]` int32, MN-major (TMA-aligned). An FP4
    // K block of `block_k` bytes holds `block_k * 2` elements, i.e. `block_k * 2 / (gran_k * 4)` words (MXFP4: 2 for 128
    // bytes, 1 for 64; NVFP4: 4 and 2); an FP8 one needs 1 (128 elements, a quarter of a 128-granularity word
    // 64). BF16 has no scales: the kernel never reads the SF descriptors, which just repeat A's
    const int num_sf_words_per_block = (is_fp4 or is_mixed_fp4_b) ? config.layout.block_k * 2 / (gran_k * 4) : 1;
    const auto tensor_map_sfa = is_bf16 ? tensor_map_a :
                                make_tma_sf_desc(cute::UMMA::Major::MN, sfa, desc.m, desc.k,
                                                 config.storage_config.load_block_m, gran_k, num_groups_sfa, 0, 0, false,
                                                 num_sf_words_per_block, static_cast<int>(sfa.stride(-1)));
    const auto tensor_map_sfb = is_bf16 ? tensor_map_a :
                                make_tma_sf_desc(cute::UMMA::Major::MN, sfb, desc.n, desc.k,
                                                 config.storage_config.load_block_n, gran_k, num_groups_sfb, 0, 0, false,
                                                 num_sf_words_per_block, static_cast<int>(sfb.stride(-1)));

    // Data-parallel normal GEMMs launch one CTA per tile when there are fewer tiles than SMs: idle persistent CTAs still
    // pay for dispatch and their prologue, and under PDL they sit between this kernel and the next. The scheduler strides
    // by the compile-time SM count, so each launched CTA still gets exactly its own tile
    const int num_ctas = stream_k.num_ctas > 0 ? stream_k.num_ctas :
                         (desc.gemm_type == GemmType::Normal or is_batched ?
                              std::min(config.launch_config.num_sms, SM120ArchSpec::get_num_tiles(desc, config.layout)) :
                              config.launch_config.num_sms);
    const bool early_pdl_trigger = SM120ArchSpec::use_early_pdl_trigger(desc, num_ctas);
    SM120FP8FP4Gemm1D1DRuntime::compile_and_launch(tag, {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(num_ctas, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .is_fp4 = is_fp4,
        .is_bf16 = is_bf16,
        .is_mixed_fp4_b = is_mixed_fp4_b,
        .gran_k = gran_k,
        .use_stream_k = stream_k.num_ctas > 0,
        .early_pdl_trigger = early_pdl_trigger,
        .grouped_layout = grouped_layout,
        .d = d.data_ptr(),
        .ld_d = static_cast<int>(d.stride(-2)),
        .d_stride_g = is_batched ? d.stride(0) : 0,
        .alpha = alpha,
        .workspace = workspace,
        .counters = counters,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .epilogue = epilogue.kind,
        .epilogue_fast_math = epilogue.fast_math,
        .epilogue_args = epilogue.args,
    });
}

// Opt-in autotuning (`DG_SM120_AUTOTUNE=1`): on the first call of each shape, time the model's top candidates on the
// real tensors and cache the fastest (in-process, keyed by the full GEMM descriptor). Off by default: CUDA events
// resolve only ~2 us here and eager timing cannot see the PDL overlap with neighboring kernels, so it mispicked as
// often as the fitted model (see `get_1d1d_cost`). It never runs during CUDA-graph capture: capture falls back to a
// cached choice, or the model's. `DG_SM120_FORCE_LAYOUT` / `DG_SM120_FORCE_STREAM_K` override it
static SM120BlockScaledChoice sm120_block_scaled_choice(const GemmDesc& desc, const std::function<void(const SM120BlockScaledChoice&)>& run) {
    // Forced (developer sweeps) or model choice
    const auto model_layout = get_best_config<SM120ArchSpec>(desc).layout;
    const SM120BlockScaledChoice model_choice = {model_layout, SM120ArchSpec::get_stream_k_config(desc, model_layout).num_ctas > 0};
    const bool forced = not deep_jit::get_env<std::string>("DG_SM120_FORCE_LAYOUT").empty() or
                        deep_jit::get_env<int>("DG_SM120_FORCE_STREAM_K", -1) >= 0;
    if (forced or deep_jit::get_env<int>("DG_SM120_AUTOTUNE", 0) == 0)
        return model_choice;

    std::stringstream key_stream;
    key_stream << desc;
    const auto key = key_stream.str();
    static std::mutex mutex;
    static std::unordered_map<std::string, SM120BlockScaledChoice> cache;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (const auto it = cache.find(key); it != cache.end())
            return it->second;
    }
    if (c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None)
        return model_choice;

    // Candidates: every legal layout, with and without stream-K, ranked by the model; time the best few
    std::vector<std::pair<int64_t, SM120BlockScaledChoice>> ranked;
    for (const auto& layout: SM120ArchSpec::get_layout_candidates(desc)) {
        const auto cost = SM120ArchSpec::get_layout_info(desc, layout).num_cycles;
        ranked.push_back({cost, {layout, false}});
        if (SM120ArchSpec::get_stream_k_split(desc, layout).num_ctas > 0)
            ranked.push_back({cost, {layout, true}});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    const int num_candidates = std::min<int>(static_cast<int>(ranked.size()), deep_jit::get_env<int>("DG_SM120_AUTOTUNE_TOP", 12));

    // Time each candidate the way it runs when serving: weights cold in L2. A 256 MB scratch (2x the L2) is read
    // (summed) before every timed launch, which leaves L2 full of clean lines, like the previous layer's weights; a
    // memset would leave dirty lines whose write-back competes with the GEMM for DRAM and penalizes the
    // bandwidth-bound decode tiles. Only the GEMM sits between the events, and the median of a few launches is kept,
    // as the GPU may be shared
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    static torch::Tensor flush_buffer, flush_sum;
    if (not flush_buffer.defined()) {
        flush_buffer = torch::ones({64 << 20}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        flush_sum = torch::empty({}, flush_buffer.options());
    }
    constexpr int kNumIters = 7;
    std::vector<cudaEvent_t> events(kNumIters * 2);
    for (auto& event: events)
        DG_CUDA_RUNTIME_CHECK(cudaEventCreate(&event));
    float best_us = std::numeric_limits<float>::infinity();
    auto best = model_choice;
    for (int i = 0; i < num_candidates; ++ i) {
        const auto& choice = ranked[i].second;
        run(choice);   // compile + warm up
        for (int j = 0; j < kNumIters; ++ j) {
            at::sum_out(flush_sum, flush_buffer);
            DG_CUDA_RUNTIME_CHECK(cudaEventRecord(events[j * 2], stream));
            run(choice);
            DG_CUDA_RUNTIME_CHECK(cudaEventRecord(events[j * 2 + 1], stream));
        }
        DG_CUDA_RUNTIME_CHECK(cudaEventSynchronize(events.back()));
        std::vector<float> times(kNumIters);
        for (int j = 0; j < kNumIters; ++ j)
            DG_CUDA_RUNTIME_CHECK(cudaEventElapsedTime(&times[j], events[j * 2], events[j * 2 + 1]));
        std::nth_element(times.begin(), times.begin() + kNumIters / 2, times.end());
        const float us = times[kNumIters / 2] * 1000;
        if (us < best_us)
            best_us = us, best = choice;
    }
    for (auto& event: events)
        DG_CUDA_RUNTIME_CHECK(cudaEventDestroy(event));
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        std::cout << "SM120 autotune " << key << ": " << best.layout.block_m << "x" << best.layout.block_n
                  << (best.use_stream_k ? " + stream-K" : "") << ", " << best_us << " us (cold L2)" << std::endl;
    }
    std::lock_guard<std::mutex> lock(mutex);
    cache[key] = best;
    return best;
}

static void sm120_fp8_fp4_gemm_1d1d_launch(const std::string& tag,
                                           const GemmDesc& desc_in,
                                           const torch::Tensor& a_in, const torch::Tensor& sfa,
                                           const torch::Tensor& b_in, const torch::Tensor& sfb,
                                           const torch::Tensor& d,
                                           void* grouped_layout,
                                           const int& num_groups_a, const int& num_groups_b,
                                           const int& num_groups_sfa, const int& num_groups_sfb,
                                           const int& gran_k_a, const int& gran_k_b,
                                           const float& alpha,
                                           const std::optional<SM120BlockScaledChoice>& forced_choice = std::nullopt,
                                           const SM120GemmEpilogue& epilogue = {}) {
    // `desc.k` counts logical elements, so FP4 (packed two per byte) has `a.size(-1) * 2 == k`
    const bool is_fp4 = a_in.scalar_type() == kPackedFP4;
    const bool is_bf16 = a_in.scalar_type() == torch::kBFloat16;
    // Operands share a dtype, except for FP8 activations against MXFP4 weights (W4A8)
    const bool is_mixed_fp4_b = not is_fp4 and b_in.scalar_type() == kPackedFP4;
    DG_HOST_ASSERT(a_in.scalar_type() == b_in.scalar_type() or (is_mixed_fp4_b and a_in.scalar_type() == torch::kFloat8_e4m3fn));
    // MN-major operands: BF16 is read in-kernel with the 16-bit `ldmatrix.trans`; FP8 either in-kernel (the 8-bit
    // transposing `ldmatrix`) or through a host transpose (see `sm120_resolve_mn_major_fp8`); FP4 must be K-major, as there
    // is no sub-byte transposing load. An MN-major A has no groups, and a grouped MN-major B (`[G, K, N]`) must have whole K
    // blocks: TMA bounds-checks only the stacked `G * K` rows, so a tail block would read the next group's rows
    if (is_fp4 or is_mixed_fp4_b)
        DG_HOST_ASSERT(desc_in.major_a == cute::UMMA::Major::K and desc_in.major_b == cute::UMMA::Major::K and
                       "SM120 FP4 operands must be K-major (there is no sub-byte transposing load)");
    auto desc = desc_in;
    auto a = a_in, b = b_in;
    const bool is_batched = desc.gemm_type == GemmType::Batched;
    if (not is_bf16 and not is_fp4 and not is_mixed_fp4_b and not is_batched)
        sm120_resolve_mn_major_fp8(desc, a, b);
    DG_HOST_ASSERT(is_bf16 == (desc.kernel_type == KernelType::KernelNoSF));
    DG_HOST_ASSERT(gran_k_a == gran_k_b);
    // FP4: 32-element UE8M0 (MXFP4) or 16-element UE4M3 (NVFP4) scales. BF16 has no scales and passes a nominal 32
    if (is_fp4)
        DG_HOST_ASSERT(gran_k_a == 32 or gran_k_a == 16);
    else if (is_mixed_fp4_b)
        DG_HOST_ASSERT(gran_k_a == 32);
    else
        DG_HOST_ASSERT(is_bf16 ? gran_k_a == 32 : (gran_k_a == 32 or gran_k_a == 128));
    DG_HOST_ASSERT(is_bf16 or desc.sf_gran_k == gran_k_a);
    DG_HOST_ASSERT(d.scalar_type() == desc.cd_dtype and (d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat));
    if (desc.major_a == cute::UMMA::Major::MN)
        DG_HOST_ASSERT(desc.gemm_type == GemmType::Normal or is_batched);
    if (desc.major_b == cute::UMMA::Major::MN and desc.gemm_type != GemmType::Normal and not is_batched)
        DG_HOST_ASSERT(desc.k % 64 == 0 and "Grouped MN-major B needs K to be a multiple of 64");
    // K is whole scale groups (the packed scale tensors are padded, and TMA zero-fills the tail K block), and the
    // epilogue stores 8 bytes at a time (4 BF16 or 2 FP32 columns)
    // (BF16: 16-byte aligned rows)
    DG_HOST_ASSERT(is_bf16 ? desc.k % 8 == 0 : desc.k % gran_k_a == 0);
    DG_HOST_ASSERT(desc.n % 8 == 0 and d.stride(-2) % 4 == 0);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0);
    DG_HOST_ASSERT(is_bf16 or (sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt));

    const auto run = [&](const SM120BlockScaledChoice& choice) {
        sm120_fp8_fp4_gemm_1d1d_run(tag, desc, choice, a, sfa, b, sfb, d, grouped_layout,
                                    num_groups_a, num_groups_b, num_groups_sfa, num_groups_sfb, gran_k_a, alpha, epilogue);
    };
    // NOTES: tuning runs the GEMM several times into `d`, which is overwritten by the final launch below
    run(forced_choice.has_value() ? forced_choice.value() : sm120_block_scaled_choice(desc, run));
}

static void sm120_fp8_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                    const std::optional<torch::Tensor>& c,
                                    const torch::Tensor& d,
                                    const int& m, const int& n, const int& k,
                                    const int& gran_k_a, const int& gran_k_b,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims, const float& alpha) {
    // With `c`, D already holds C (see `early_return`), and the kernel adds to it
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .sf_gran_k = gran_k_a
    };
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_fp8_fp4_gemm_1d1d", desc, a, sfa, b, sfb, d, nullptr, 1, 1, 1, 1, gran_k_a, gran_k_b, alpha);
}

// Batched block-scaled GEMM `D[g] (+)= A[g] @ B[g]^T` in one launch (`a`: `[G, M, K]`, `b`: `[G, N, K]`, `d`: `[G, M, N]`;
// `sfa` / `sfb`: the packed UE8M0 scales of each batch stacked like a grouped GEMM's, `[G, packed_k, MN]` MN-major and
// TMA-aligned), on strided views of any layout the 3D tensor maps can describe. With `accumulate`, D already holds C.
// Callers check `sm120_fp8_bmm_1d1d_supported` and run a GEMM per batch when the views are not supported
static bool sm120_fp8_bmm_1d1d_supported(const torch::Tensor& a, const torch::Tensor& sfa,
                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                         const torch::Tensor& d, const int& k,
                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b) {
    // `[G, MN, words]` with MN contiguous: the batches follow each other as `words` rows of `stride(2)` elements
    const auto stacked = [](const torch::Tensor& sf) {
        return sf.dim() == 3 and sf.stride(1) == 1 and sf.stride(0) == sf.size(2) * sf.stride(2);
    };
    return sm120_batched_launch_enabled() and is_tma_batched_operand(major_a, a) and is_tma_batched_operand(major_b, b) and
           stacked(sfa) and stacked(sfb) and
           d.dim() == 3 and d.stride(-1) == 1 and d.stride(-2) % 4 == 0 and d.stride(0) % 4 == 0 and
           d.size(-1) % 8 == 0 and reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0 and k % 128 == 0;
}

static void sm120_fp8_bmm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const bool& accumulate, const torch::Tensor& d,
                               const int& batch_size, const int& m, const int& n, const int& k,
                               const int& gran_k, const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                               const std::string& compiled_dims) {
    DG_HOST_ASSERT(sm120_fp8_bmm_1d1d_supported(a, sfa, b, sfb, d, k, major_a, major_b));
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = accumulate,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .sf_gran_k = gran_k
    };
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_fp8_bmm_1d1d", desc, a, sfa, b, sfb, d, nullptr, 1, 1, batch_size, batch_size,
                                   gran_k, gran_k, 1.0f);
}

static void sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                                         const torch::Tensor& d,
                                                         const torch::Tensor& m_indices,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout,
                                                         const std::optional<int>& expected_m_for_psum_layout) {
    // Only the PSUM layout can use an expected M
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);
    const auto desc = GemmDesc {
        .gemm_type = use_psum_layout ? GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .sf_gran_k = gran_k_a,
        .expected_m = expected_m_for_psum_layout.value_or(m), .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d", desc, a, sfa, b, sfb, d,
                                   m_indices.data_ptr(), 1, num_groups, 1, num_groups, gran_k_a, gran_k_b, 1.0f);
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
        .sf_gran_k = gran_k_a,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_m_grouped_fp8_fp4_gemm_masked_1d1d", desc, a, sfa, b, sfb, d,
                                   masked_m.data_ptr(), num_groups, num_groups, num_groups, num_groups, gran_k_a, gran_k_b, 1.0f);
}

// BF16 (no scales) on the same kernel; the scale tensors are unused
static GemmDesc sm120_bf16_desc(const GemmType& gemm_type, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                const int& num_groups, const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims,
                                const int& expected_m, const int& expected_num_groups,
                                const bool& with_accumulation) {
    return GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = with_accumulation,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = expected_num_groups
    };
}

static void sm120_bf16_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& b,
                                 const std::optional<torch::Tensor>& c,
                                 const torch::Tensor& d,
                                 const int& m, const int& n, const int& k,
                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                 const std::string& compiled_dims, const float& alpha,
                                 const std::optional<SM120BlockScaledChoice>& forced_choice = std::nullopt) {
    // With `c`, D already holds C (see `early_return`), and the kernel adds to it
    const auto desc = sm120_bf16_desc(GemmType::Normal, a, b, d, 1, m, n, k, major_a, major_b, compiled_dims, m, 1,
                                      c.has_value());
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_bf16_gemm_1d1d", desc, a, torch::Tensor(), b, torch::Tensor(), d, nullptr,
                                   1, 1, 1, 1, 32, 32, alpha, forced_choice);
}

// Batched BF16 GEMM `D[g] (+)= A[g] @ B[g]^T` in one launch (`a`: `[G, M, K]`, `b`: `[G, N, K]`, `d`: `[G, M, N]`), on strided
// views of any layout the 3D tensor maps can describe (the heads of a `[b, h, d]` tensor, an MN-major `[G, N, K]` view of a
// `[G, K, N]` weight). With `accumulate`, D already holds C. Callers check `sm120_bf16_bmm_supported` and run a GEMM per
// batch when the views are not supported; both routes give identical bits
static bool sm120_bf16_bmm_supported(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d, const int& k,
                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b) {
    return sm120_batched_launch_enabled() and is_tma_batched_operand(major_a, a) and is_tma_batched_operand(major_b, b) and
           d.dim() == 3 and d.stride(-1) == 1 and d.stride(-2) % 4 == 0 and d.stride(0) % 4 == 0 and
           d.size(-1) % 8 == 0 and reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0 and k % 8 == 0;
}

// The tile layout of the batched launch, data-parallel. A GEMM's bits depend on its layout (the warps split each K block,
// and stream-K reduces partial sums), so a launch per batch that wants the same bits runs this layout as well
static SM120BlockScaledChoice sm120_bf16_bmm_choice(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                                                    const int& batch_size, const int& m, const int& n, const int& k,
                                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                    const std::string& compiled_dims, const bool& accumulate) {
    const auto desc = sm120_bf16_desc(GemmType::Batched, a, b, d, batch_size, m, n, k, major_a, major_b, compiled_dims, m,
                                      batch_size, accumulate);
    return {get_best_config<SM120ArchSpec>(desc).layout, false};
}

static void sm120_bf16_bmm_1d1d(const torch::Tensor& a, const torch::Tensor& b,
                                const bool& accumulate, const torch::Tensor& d,
                                const int& batch_size, const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims, const float& alpha) {
    DG_HOST_ASSERT(sm120_bf16_bmm_supported(a, b, d, k, major_a, major_b));
    const auto desc = sm120_bf16_desc(GemmType::Batched, a, b, d, batch_size, m, n, k, major_a, major_b, compiled_dims, m,
                                      batch_size, accumulate);
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_bf16_bmm_1d1d", desc, a, torch::Tensor(), b, torch::Tensor(), d, nullptr,
                                   1, 1, 1, 1, 32, 32, alpha,
                                   sm120_bf16_bmm_choice(a, b, d, batch_size, m, n, k, major_a, major_b, compiled_dims, accumulate));
}

static void sm120_m_grouped_bf16_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& b,
                                                      const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                                      const int& num_groups, const int& m, const int& n, const int& k,
                                                      const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                      const std::string& compiled_dims,
                                                      const bool& use_psum_layout,
                                                      const std::optional<int>& expected_m_for_psum_layout) {
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);
    const auto desc = sm120_bf16_desc(use_psum_layout ? GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous,
                                      a, b, d, num_groups, m, n, k, major_a, major_b, compiled_dims,
                                      expected_m_for_psum_layout.value_or(m),
                                      expected_m_for_psum_layout.has_value() ? num_groups : 1, false);
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_m_grouped_bf16_gemm_contiguous_1d1d", desc, a, torch::Tensor(), b, torch::Tensor(), d,
                                   grouped_layout.data_ptr(), 1, num_groups, 1, num_groups, 32, 32, 1.0f);
}

static void sm120_m_grouped_bf16_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& b,
                                                  const torch::Tensor& d, const torch::Tensor& masked_m,
                                                  const int& num_groups, const int& m, const int& n, const int& k,
                                                  const int& expected_m,
                                                  const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                  const std::string& compiled_dims) {
    const auto desc = sm120_bf16_desc(GemmType::MGroupedMasked, a, b, d, num_groups, m, n, k, major_a, major_b, compiled_dims,
                                      expected_m, num_groups, false);
    sm120_fp8_fp4_gemm_1d1d_launch("sm120_m_grouped_bf16_gemm_masked_1d1d", desc, a, torch::Tensor(), b, torch::Tensor(), d,
                                   masked_m.data_ptr(), num_groups, num_groups, num_groups, num_groups, 32, 32, 1.0f);
}

// K-grouped FP8 GEMM (weight gradients): `D[g] = A[K_g]^T @ B[K_g]` (+ C[g], already copied into D by `early_return`), one
// block-scaled GEMM per group over its K slice, with the group sizes from the host like SM90. `sfa` / `sfb` are the packed
// UE8M0 scales of `transform_k_grouped_sf_into_required_layout`, `[sum(ceil(K_g / (gran_k * 4))), MN]` with MN contiguous:
// a group's rows, transposed, are exactly the `[MN, packed_k]` MN-major scales of a normal GEMM. Operands are either MN-major
// `[sum_k, MN]` (TN: `major` MN) or K-major `[MN, K_g]` blocks laid end to end (NT: `major` K)
static void sm120_k_grouped_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                          const torch::Tensor& b, const torch::Tensor& sfb,
                                          const bool& accumulate, const torch::Tensor& d,
                                          const int& m, const int& n, const std::vector<int>& ks,
                                          const int& gran_k, const cute::UMMA::Major& major,
                                          const std::string& compiled_dims) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn and b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    const bool is_mn_major = major == cute::UMMA::Major::MN;
    int64_t k_start = 0, sf_start = 0;
    for (size_t i = 0; i < ks.size(); ++ i) {
        const int k = ks[i];
        const auto d_group = d[i];
        if (k == 0) {
            if (not accumulate)
                d_group.zero_();
            continue;
        }
        const int packed_k = ceil_div(k / gran_k, 4);
        const auto slice = [&](const torch::Tensor& t, const int& mn) {
            return is_mn_major ? t.narrow(0, k_start, k).transpose(0, 1) :
                                 t.narrow(0, k_start * mn, static_cast<int64_t>(k) * mn).view({mn, k});
        };
        sm120_fp8_fp4_gemm_1d1d(slice(a, m), sfa.narrow(0, sf_start, packed_k).transpose(0, 1),
                                slice(b, n), sfb.narrow(0, sf_start, packed_k).transpose(0, 1),
                                accumulate ? std::make_optional(d_group) : std::nullopt, d_group,
                                m, n, k, gran_k, gran_k, major, major, compiled_dims, 1.0f);
        k_start += k;
        sf_start += packed_k;
    }
}
} // namespace deep_gemm
