#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "utils.hpp"
#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

// SM120 (RTX PRO 6000 / RTX 50 series Blackwell): `mma.sync` tensor cores, TMA, no clusters used,
// and ~99 KB of opt-in shared memory per block (vs. 227 KB on SM90/SM100).
struct SM120ArchSpec {
    static constexpr int kNumMaxStages = 16;
    static constexpr int kNumMaxStreamKTiles = 1024;
    // The stream-K reduction costs about two K blocks of time
    static constexpr int kStreamKFixupCost = 2;
    // Fitted cost model parameters (see `get_layout_info`)
    static constexpr double kDRAMBytesPerUnit = 1.6e6;
    static constexpr double kL2BytesPerUnit = 6e6;
    static constexpr double kCTACostScale = 1.2;
    static constexpr double kWaveOverhead = 2.0;
    static constexpr int kNumMaxAccumPerThread = 64;

    static int get_smem_capacity() {
        return jit->device.get_num_smem_bytes();
    }

    // The `mma.sync` 1D1D kernel serves both block-scaled operands (`Kernel1D1D`) and BF16 (`KernelNoSF`); its
    // `block_k` counts bytes of an operand row (one swizzle atom), while the FP32-scale 1D2D kernel's counts elements
    static bool is_1d1d_kernel(const KernelType& kernel_type) {
        return kernel_type == KernelType::Kernel1D1D or kernel_type == KernelType::KernelNoSF;
    }

    // Math threads per CTA. The FP32 1D2D kernel uses 4 warps for `block_m <= 64` and 8 above. The block-scaled 1D1D
    // kernel also gives 64x128 / 64x256 tiles 8 warps (as a 2 x 4 grid of 32-row warps): with 4 warps, `mma.sync` and
    // `ldmatrix` latencies are not hidden (ncu: 5 active warps/SM, `stall_wait` 2x that of FlashInfer's b12x, which
    // runs 8 warps on the same 64x128 tile)
    static int get_num_math_threads(const KernelType& kernel_type, const int& block_m, const int& block_n) {
        if (is_1d1d_kernel(kernel_type) and block_m == 64 and block_n >= 128)
            return 256;
        return block_m <= 64 ? 128 : 256;
    }

    // Warp rows of the grid, mirroring `kNumWarpsM` in the kernels: one 64-row warp per 64 rows, except the 8-warp
    // 64-row block-scaled tiles, which use two 32-row warps
    static int get_num_warps_m(const int& num_math_warps, const int& block_m) {
        if (block_m == 64 and num_math_warps == 8)
            return 2;
        return block_m >= 64 ? block_m / 64 : 1;
    }

    // Must stay in sync with the warp tiling in `sm120_fp8_gemm_1d2d.cuh`
    // Narrow tiles (`block_n == 32`, block-scaled kernel only) keep 4 math warps (a whole warpgroup for `setmaxnreg`)
    // by splitting each K block's MMA steps over 2 K-warps, whose accumulators are summed through shared memory
    static int get_num_k_warps(const int& block_n) {
        return block_n == 32 ? 2 : 1;
    }

    // Shared memory of the K-warp reduction scratch: `[accumulators per thread][threads of one K-warp group]` floats
    static int get_k_warp_reduce_smem(const int& block_m, const int& block_n) {
        if (get_num_k_warps(block_n) == 1)
            return 0;
        const int num_accum = (block_m / 16) * (block_n / 2 / 8) * 4;
        return align(num_accum * (get_num_math_threads(KernelType::Kernel1D1D, block_m, block_n) / get_num_k_warps(block_n)) * 4, 128);
    }

    static bool is_warp_tiling_legal(const KernelType& kernel_type, const int& block_m, const int& block_n) {
        const int num_math_warps = get_num_math_threads(kernel_type, block_m, block_n) / 32;
        const int num_warps_m = get_num_warps_m(num_math_warps, block_m);
        const int num_k_warps = get_num_k_warps(block_n);
        if (num_k_warps > 1 and block_m >= 64)
            return false;
        if (num_math_warps % (num_warps_m * num_k_warps) != 0)
            return false;
        const int num_warps_n = num_math_warps / (num_warps_m * num_k_warps);
        if (block_n % num_warps_n != 0)
            return false;
        const int warp_m = block_m / num_warps_m, warp_n = block_n / num_warps_n;
        if (warp_m % 16 != 0 or warp_n % 16 != 0 or 128 % warp_n != 0)
            return false;
        return (warp_m / 16) * (warp_n / 8) * 4 <= kNumMaxAccumPerThread;
    }

    // PDL trigger point of the block-scaled FP4 kernel (see the kernel's producer). Triggering as soon as the last load is
    // issued lets the next kernel's launch and weight pre-issue overlap with our last K blocks. On the 63-point DeepSeek-V3
    // TP8 sweep (`scripts/sm120_fp4_vs_flashinfer.py`) that wins 4-20% when the grid leaves the SMs mostly free (<= 60% of
    // them: q_b M = 128-512, o_proj / dense down M = 32-256, decode GEMMs), and loses 4-11% on grids that fill 68% or
    // more of them (q_a M = 128-512, shared gate_up M = 512 / 2048, dense gate_up M = 256, o_proj / dense down M = 512),
    // where the next kernel's early weight loads compete with our last stages for DRAM
    static bool use_early_pdl_trigger(const GemmDesc& desc, const int& num_ctas) {
        return desc.kernel_type == KernelType::Kernel1D1D and desc.a_dtype == kPackedFP4 and desc.gemm_type == GemmType::Normal and
               num_ctas * 10 <= desc.num_sms * 6;
    }
    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // 1D2D: FP32 scales promoted on CUDA cores; 1D1D: packed UE8M0 scales applied by the block-scaled MMA
        // NoSF: BF16 operands on the 1D1D kernel
        DG_HOST_ASSERT(desc.kernel_type == KernelType::Kernel1D2D or is_1d1d_kernel(desc.kernel_type));
        // MN-major operands (BF16 with the 16-bit `ldmatrix.trans`, FP8 with its 8-bit form) are whole 64-byte swizzle atoms
        // in MN (FP4 is K-major only: there is no sub-byte transposing load)
        const bool is_mn_major_a = desc.major_a == cute::UMMA::Major::MN;
        const bool is_mn_major_b = desc.major_b == cute::UMMA::Major::MN;
        DG_HOST_ASSERT((desc.a_dtype != kPackedFP4 and desc.b_dtype != kPackedFP4) or (not is_mn_major_a and not is_mn_major_b));

        // Block M candidates
        std::vector<int> block_m_candidates;
        if (desc.gemm_type == GemmType::MGroupedContiguous or
            desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
            block_m_candidates = {heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
        } else {
            // Small M blocks help memory-bound (decoding) shapes fill the GPU; skip blocks much larger than M
            for (const int block_m: {16, 32, 64, 128}) {
                if (is_mn_major_a ? (block_m == 64 or (block_m == 128 and block_m < 2 * desc.get_expected_m())) :
                                    (block_m == 16 or block_m < 2 * desc.get_expected_m()))
                    block_m_candidates.push_back(block_m);
            }
        }

        // Block N candidates
        std::vector<int> block_n_candidates;
        // BLOCK_N = 32 gives thin-N decode GEMMs (e.g. N = 2112) twice the CTAs, which they need to keep enough bytes in
        // flight; only the 1D1D kernel implements the K-warp split it requires
        for (const int block_n: {32, 64, 128, 256}) {
            if (block_n == 32 and not is_1d1d_kernel(desc.kernel_type))
                continue;
            // The K-warp split sums two partial accumulators, an order the wider tiles do not use: with deterministic
            // (batch invariant) algorithms every output must be one in-order K loop whatever the tile
            if (block_n == 32 and heuristics_runtime->get_deterministic_algorithms())
                continue;
            if (is_mn_major_b and block_n % 64 != 0)
                continue;
            if (block_n % heuristics_runtime->get_block_n_multiple_of() == 0)
                block_n_candidates.push_back(block_n);
        }

        // K block sizes, in bytes of an operand row (one swizzle atom): 128 everywhere, and for the block-scaled FP4
        // kernel also 64 (128 elements), which halves the stage so twice as many stages are in flight
        // FP8 x FP4 (W4A8): only 64-byte B rows (128 elements, one 128-byte FP8 A row), so A fragments of both MMAs of a
        // step fit in registers
        std::vector<int> block_k_candidates = {128};
        if (desc.kernel_type == KernelType::Kernel1D1D and desc.a_dtype == kPackedFP4)
            block_k_candidates.push_back(64);
        if (is_mixed_fp4_b(desc))
            block_k_candidates = {64};

        // Developer-only override for tuning sweeps, e.g. `DG_SM120_FORCE_LAYOUT=128x128` or `64x128x64`
        const auto forced = deep_jit::get_env<std::string>("DG_SM120_FORCE_LAYOUT");
        if (not forced.empty()) {
            int block_m = 0, block_n = 0, block_k = 128;
            DG_HOST_ASSERT(std::sscanf(forced.c_str(), "%dx%dx%d", &block_m, &block_n, &block_k) >= 2);
            DG_HOST_ASSERT(is_warp_tiling_legal(desc.kernel_type, block_m, block_n));
            DG_HOST_ASSERT(std::find(block_k_candidates.begin(), block_k_candidates.end(), block_k) != block_k_candidates.end());
            return {Layout{0, block_m, block_n, block_k, 1, 1}};
        }

        std::vector<Layout> candidates;
        for (const int block_m: block_m_candidates) {
            if (block_m % heuristics_runtime->get_block_m_multiple_of() != 0)
                continue;
            for (const int block_n: block_n_candidates) {
                if (not is_warp_tiling_legal(desc.kernel_type, block_m, block_n))
                    continue;
                for (const int block_k: block_k_candidates) {
                    const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
                    const auto storage_config = get_storage_config(desc, layout);
                    if (get_pipeline_config(desc, layout, storage_config).num_stages < 2)
                        continue;
                    candidates.push_back(layout);
                }
            }
        }
        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        DG_HOST_ASSERT(layout.swap_ab == 0);
        // The 1D1D kernel's `block_k` is already in bytes. An MN-major operand is swizzled along MN: 128-byte atoms, or
        // 64-byte ones for 64-wide FP8 tiles
        const auto swizzle_mode_a = desc.major_a == cute::UMMA::Major::MN ?
            get_swizzle_mode(layout.block_m, static_cast<int>(c10::elementSize(desc.a_dtype))) :
            get_swizzle_mode(get_block_k_a(desc, layout), get_smem_elem_size(desc, desc.a_dtype));
        const auto swizzle_mode_b = desc.major_b == cute::UMMA::Major::MN ?
            get_swizzle_mode(layout.block_n, static_cast<int>(c10::elementSize(desc.b_dtype))) :
            get_swizzle_mode(layout.block_k, get_smem_elem_size(desc, desc.b_dtype));
        // D is stored directly from registers
        return {
            layout.block_m, layout.block_n,
            layout.block_m, layout.block_n,
            swizzle_mode_a, swizzle_mode_b, 0
        };
    }

    // FP8 activations against MXFP4 weights (W4A8)
    static bool is_mixed_fp4_b(const GemmDesc& desc) {
        return desc.a_dtype == torch::kFloat8_e4m3fn and desc.b_dtype == kPackedFP4;
    }
    // Bytes of an A row: `block_k` counts the bytes of a B row, and the FP8 A of W4A8 has twice as many
    static int get_block_k_a(const GemmDesc& desc, const Layout& layout) {
        return is_mixed_fp4_b(desc) ? layout.block_k * 2 : layout.block_k;
    }
    // Bytes per `block_k` unit of a shared memory operand row
    static int get_smem_elem_size(const GemmDesc& desc, const at::ScalarType& dtype) {
        return is_1d1d_kernel(desc.kernel_type) ? 1 : static_cast<int>(c10::elementSize(dtype));
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        // Barriers, plus the split-K "last CTA" flag
        const int smem_barriers = kNumMaxStages * 8 * 2 + 16;
        const int smem_a_per_stage = storage_config.load_block_m * get_block_k_a(desc, layout) * get_smem_elem_size(desc, desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * get_smem_elem_size(desc, desc.b_dtype);
        // NOTES: the FP32 kernel reads SFA from global memory (L1/L2 cached), not staged. The block-scaled 1D1D
        // kernel TMA-loads the packed scale words of both operands with each stage (FP4 128-byte K block: 2 words per
        // row for MXFP4, 4 for NVFP4; FP8: 1), each tile rounded up to the 128B TMA destination alignment
        int smem_sf_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            // FP4: `block_k * 2` elements per row, one byte per `sf_gran_k` (MXFP4 32, NVFP4 16), 4 bytes per word
            DG_HOST_ASSERT(desc.sf_gran_k > 0);
            const int num_sf_words = desc.b_dtype == kPackedFP4 ? layout.block_k * 2 / desc.sf_gran_k / 4 : 1;
            smem_sf_per_stage = align(num_sf_words * storage_config.load_block_m * 4, 128) +
                                align(num_sf_words * storage_config.load_block_n * 4, 128);
        }
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sf_per_stage;
        const int smem_reduce = is_1d1d_kernel(desc.kernel_type) ? get_k_warp_reduce_smem(layout.block_m, layout.block_n) : 0;
        const int num_stages = std::min((get_smem_capacity() - smem_barriers - smem_reduce) / smem_per_stage, kNumMaxStages);
        return {smem_barriers + smem_reduce + num_stages * smem_per_stage, num_stages, 0};
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_tma_threads = 128;
        const int num_math_threads = get_num_math_threads(desc.kernel_type, layout.block_m, layout.block_n);
        return {
            desc.num_sms, 1,
            num_tma_threads + num_math_threads,
            num_tma_threads, num_math_threads,
            0, 0
        };
    }

    static int get_num_tiles(const GemmDesc& desc, const Layout& layout) {
        return ceil_div(desc.get_expected_m(), layout.block_m) *
               ceil_div(desc.get_expected_n(), layout.block_n) *
               desc.get_expected_num_groups();
    }

    struct StreamKConfig {
        int num_ctas = 0;   // 0 means data-parallel (one tile per task)
        int num_slots = 0;  // Max CTAs contributing to one tile, i.e. workspace partial slots
    };

    // Bytes along K of an operand row: packed FP4 holds two elements per byte, BF16 one per two bytes (the 1D1D
    // kernel's `block_k` counts bytes; for the FP8 1D2D kernel bytes and elements coincide)
    static int get_k_bytes(const GemmDesc& desc, const int& k) {
        if (desc.b_dtype == kPackedFP4)
            return k / 2;
        return desc.a_dtype == torch::kBFloat16 ? k * 2 : k;
    }

    // Stream-K geometry for a layout: spread `num_tiles x num_k_blocks` iterations evenly over CTAs. Returns no split
    // when there is nothing to gain (normal GEMMs only, and only with spare CTAs and enough K per CTA)
    static StreamKConfig get_stream_k_split(const GemmDesc& desc, const Layout& layout) {
        if (desc.gemm_type != GemmType::Normal)
            return {};
        const int64_t num_tiles = get_num_tiles(desc, layout);
        const int64_t num_k_blocks = ceil_div(get_k_bytes(desc, desc.k), layout.block_k);
        // Counters are per tile
        if (num_tiles > kNumMaxStreamKTiles)
            return {};

        // Each CTA should own enough K blocks to amortize its pipeline fill and the reduction
        const int min_iters = deep_jit::get_env<int>("DG_SM120_SK_MIN_ITERS", 4);
        const int64_t num_iters = num_tiles * num_k_blocks;
        const int num_ctas = static_cast<int>(std::min<int64_t>(desc.num_sms, num_iters / min_iters));
        if (num_ctas <= num_tiles)
            return {};

        // A tile spans at most `ceil(num_k_blocks / min_range) + 1` CTAs
        const int64_t min_range = std::max<int64_t>(1, num_iters / num_ctas);
        return {num_ctas, static_cast<int>(ceil_div<int64_t>(num_k_blocks, min_range) + 1)};
    }

    // Stream-K policy of the model (used without autotuning, and for the FP32-scale kernel):
    //  - The FP32-scale kernel is opt-in (`DG_SM120_FORCE_STREAM_K=1`): under CUDA graphs with cold weights data-parallel
    //    tiles were faster or equal on all measured DeepSeek shapes (`scripts/sm120_sweep_graph.py --sk 0,1`)
    //  - The 1D1D kernel streams half the bytes per K element, and on the few-tile decode shapes its per-CTA K loop is
    //    the critical path, so it splits K when there are very few tiles (<= 1/8 of the SMs) and the split clearly
    //    shortens that loop. With more tiles each CTA pays two pipeline fills: e.g. 64x128 at M = 128, N = 4608 ran
    //    23.9 vs. 16.4 us. (Folding stream-K into the fitted cost model mispicked more, as few measured shapes split)
    // `DG_SM120_FORCE_STREAM_K`: 0 never, 1 whenever it is a clear win, 2 (tests) whenever there are spare CTAs
    // Deterministic algorithms are batch invariant: a split changes the order a partial sum is added in, and the split
    // follows the number of tiles, i.e. the batch. Without it each output element is one CTA's in-order K loop
    static StreamKConfig get_stream_k_config(const GemmDesc& desc, const Layout& layout) {
        if (heuristics_runtime->get_deterministic_algorithms())
            return {};
        const auto split = get_stream_k_split(desc, layout);
        if (split.num_ctas == 0)
            return {};
        const int force = deep_jit::get_env<int>("DG_SM120_FORCE_STREAM_K", -1);
        const bool is_block_scaled = is_1d1d_kernel(desc.kernel_type);
        const int64_t num_tiles = get_num_tiles(desc, layout);
        if (force == 0)
            return {};
        if (force == 2)
            return split;
        if (force < 0 and not (is_block_scaled and num_tiles * 8 <= desc.num_sms))
            return {};

        // Compare with data-parallel waves (in K blocks per CTA), requiring a clear win
        const int64_t num_k_blocks = ceil_div(get_k_bytes(desc, desc.k), layout.block_k);
        const int64_t dp_cost = ceil_div<int64_t>(num_tiles, desc.num_sms) * num_k_blocks;
        const int64_t sk_cost = ceil_div<int64_t>(num_tiles * num_k_blocks, split.num_ctas) + kStreamKFixupCost;
        return sk_cost * 10 > dp_cost * 9 ? StreamKConfig{} : split;
    }

    // 1D1D kernel cost model of data-parallel layouts, fitted jointly on FP4 and UE8M0 FP8 CUDA-graph sweeps of DeepSeek
    // shapes with PDL and cold weights (`scripts/sm120_sweep_blockscale.py --mode fp4 / fp8g32` +
    // `scripts/sm120_fit_blockscale.py`, keep both in sync). Everything counts bytes, so one model covers every mode.
    // Together with the stream-K rule of `get_stream_k_config`, its picks are within 1.3% of the fastest layout on
    // average over 63 shapes per mode (worst 14%). Fitting on FP4 alone over-penalized shallow pipelines: FP4 always had a
    // deeper 64-byte-K alternative, while FP8's 128x128 tile has 2 stages, so FP8 fell to 64x64 at large M (1.4-1.6x
    // slower). Units: one 128x128 K block of 128 bytes.
    // Every wave costs the max of
    //  - its share of DRAM traffic, inflated for small weight boxes (`block_n x block_k` bytes per TMA stream DRAM less
    //    efficiently),
    //  - its share of L2 -> SM traffic,
    //  - the K loop of one CTA: per K block a layout cost, inflated by `(kFullDepth / stages) ^ kDepthExp` for pipelines
    //    shallower than `kFullDepth` stages (latency-bound)
    static constexpr double k1D1DDRAMBytesPerUnit = 1.8e6;
    static constexpr double k1D1DL2BytesPerUnit = 4e6;
    static constexpr double k1D1DCTACostScale = 1.5;
    static constexpr double k1D1DFullDepth = 3.15;
    static constexpr double k1D1DDepthExp = 0.5;
    static constexpr double k1D1DBoxRef = 8192;
    static constexpr double k1D1DBoxExp = 0.75;
    static constexpr double k1D1DWaveOverhead = 0.5;

    // Per-K-block CTA cost of the 1D1D kernel (relative, 128x128 = 1000)
    static int get_1d1d_block_cost(const Layout& layout) {
        const int bm = layout.block_m, bn = layout.block_n;
        if (bm ==  16 and bn ==  32) return 31;
        if (bm ==  16 and bn ==  64) return 62;
        if (bm ==  16 and bn == 128) return 250;
        if (bm ==  16 and bn == 256) return 250;
        if (bm ==  32 and bn ==  32) return 62;
        if (bm ==  32 and bn ==  64) return 105;
        if (bm ==  32 and bn == 128) return 275;
        if (bm ==  32 and bn == 256) return 500;
        if (bm ==  64 and bn ==  64) return 352;
        if (bm ==  64 and bn == 128) return 476;
        if (bm ==  64 and bn == 256) return 1100;
        if (bm == 128 and bn ==  64) return 525;
        if (bm == 128 and bn == 128) return 1000;
        return bm * bn * 1000 / (128 * 128);
    }

    static double get_1d1d_cost(const GemmDesc& desc, const Layout& layout) {
        const double m = desc.get_expected_m(), n = desc.get_expected_n();
        const double num_groups_a = desc.gemm_type == GemmType::MGroupedMasked or desc.gemm_type == GemmType::Batched ?
                                    desc.get_expected_num_groups() : 1;
        const double num_groups_b = desc.gemm_type == GemmType::Normal ? 1 : desc.num_groups;
        const double k_bytes = get_k_bytes(desc, desc.get_expected_k());
        const int64_t num_tiles = get_num_tiles(desc, layout);
        const double num_k_blocks = ceil_div(static_cast<int>(k_bytes), layout.block_k);
        const double box_penalty = std::pow(std::max(1.0, k1D1DBoxRef / (layout.block_n * layout.block_k)), k1D1DBoxExp);
        // D bytes per element: written, and also read when accumulating
        const double cd_bytes = static_cast<double>(c10::elementSize(desc.cd_dtype)) * (desc.with_accumulation ? 2 : 1);
        const double dram = (num_groups_b * n * k_bytes + num_groups_a * (m * k_bytes + cd_bytes * m * n)) / k1D1DDRAMBytesPerUnit * box_penalty;
        const double l2 = static_cast<double>(num_tiles) * num_k_blocks * (layout.block_m + layout.block_n) * layout.block_k / k1D1DL2BytesPerUnit;
        const int num_stages = get_pipeline_config(desc, layout, get_storage_config(desc, layout)).num_stages;
        const double depth = std::pow(std::max(1.0, k1D1DFullDepth / num_stages), k1D1DDepthExp);
        const double cta = num_k_blocks * layout.block_k / 128.0 * get_1d1d_block_cost(layout) / 1000.0 * k1D1DCTACostScale * depth + k1D1DWaveOverhead;
        const int64_t num_waves = ceil_div<int64_t>(num_tiles, desc.num_sms);
        const int64_t num_last_blocks = num_tiles % desc.num_sms;
        double cost = 0;
        for (int64_t i = 0; i < num_waves; ++ i) {
            const int64_t num_wave_tiles = (i == num_waves - 1 and num_last_blocks != 0) ? num_last_blocks : desc.num_sms;
            const double share = static_cast<double>(num_wave_tiles) / num_tiles;
            cost += std::max({dram * share, l2 * share, cta});
        }
        return cost;
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_tiles = get_num_tiles(desc, layout);
        if (is_1d1d_kernel(desc.kernel_type)) {
            // Data-parallel cost; the chosen layout may then be split by `get_stream_k_config`
            const double cost = get_1d1d_cost(desc, layout);
            const int num_waves = ceil_div(num_tiles, desc.num_sms);
            const int num_last_blocks = num_tiles % desc.num_sms;
            return {num_waves, num_last_blocks == 0 ? desc.num_sms : num_last_blocks, static_cast<int64_t>(cost * 1000), layout};
        }
        const auto num_waves = ceil_div(num_tiles, desc.num_sms);
        const auto num_last_blocks = num_tiles % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        // Cost model fitted on CUDA-graph, cold-weight sweeps of DeepSeek shapes (the serving condition), with
        // `scripts/sm120_sweep_graph.py` and `scripts/sm120_fit_heuristic.py` (keep the two in sync).
        // Every wave costs the max of
        //  - its share of DRAM traffic (weights, activations and output),
        //  - its share of L2 -> SM traffic (each tile reloads its A and B rows),
        //  - the time one CTA needs for its tile (`get_relative_wave_cost`), plus a fixed overhead,
        // in units of one 128x128 K block of a full wave
        const double m = desc.get_expected_m(), n = desc.get_expected_n();
        const double num_groups_a = desc.gemm_type == GemmType::MGroupedMasked or desc.gemm_type == GemmType::Batched ?
                                    desc.get_expected_num_groups() : 1;
        const double num_groups_b = desc.gemm_type == GemmType::Normal ? 1 : desc.num_groups;
        const double k_bytes = get_k_bytes(desc, desc.get_expected_k());
        const double num_k_blocks = ceil_div(static_cast<int>(k_bytes), layout.block_k);
        const double dram = (num_groups_b * n * k_bytes + num_groups_a * (m * k_bytes + 2 * m * n)) / kDRAMBytesPerUnit;
        const double l2 = static_cast<double>(num_tiles) * num_k_blocks * (layout.block_m + layout.block_n) * layout.block_k / kL2BytesPerUnit;
        const double cta = num_k_blocks * layout.block_k / 128.0 * get_relative_wave_cost(layout) / 1000.0 * kCTACostScale + kWaveOverhead;
        double cost = 0;
        for (int i = 0; i < num_waves; ++ i) {
            const int num_wave_tiles = (i == num_waves - 1 and num_last_blocks != 0) ? num_last_blocks : desc.num_sms;
            const double share = static_cast<double>(num_wave_tiles) / num_tiles;
            cost += std::max({dram * share, l2 * share, cta});
        }
        return {num_waves, last_wave_util, static_cast<int64_t>(cost * 1000), layout};
    }

    // Relative time of one full wave per unit K, normalized to 128x128 = 1000
    // NOTES: measured on RTX PRO 6000 (188 SMs) with DeepSeek-V3 shapes at M = 4096 (`scripts/sm120_sweep.py`);
    // small tiles are cheaper per wave but do much less work, as `mma.sync` tiles are bound by `ldmatrix` traffic
    static int get_relative_wave_cost(const Layout& layout) {
        const int bm = layout.block_m, bn = layout.block_n;
        if (bm == 128 and bn == 128) return 1000;
        if (bm == 128 and bn == 64)  return 590;
        if (bm == 64  and bn == 128) return 570;
        if (bm == 64  and bn == 64)  return 400;
        if (bm == 32  and bn == 256) return 840;
        if (bm == 32  and bn == 128) return 470;
        if (bm == 32  and bn == 64)  return 310;
        if (bm == 16  and bn == 256) return 850;
        if (bm == 16  and bn == 128) return 450;
        if (bm == 16  and bn == 64)  return 250;
        // Narrow tiles (block-scaled kernel): provisional, to be fitted with `scripts/sm120_fit_heuristic.py`
        if (bm == 16  and bn == 32)  return 170;
        if (bm == 32  and bn == 32)  return 230;
        // Fallback: proportional to the tile area
        return bm * bn * 1000 / (128 * 128);
    }

    // NOTES: ties keep the earlier candidate (smaller tiles first), as the fitting script does
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

} // namespace deep_gemm
