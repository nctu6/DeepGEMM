#pragma once

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
    static constexpr int kNumMaxStages = 8;
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

    static int get_num_math_threads(const int& block_m) {
        return block_m <= 64 ? 128 : 256;
    }

    // Must stay in sync with the warp tiling in `sm120_fp8_gemm_1d2d.cuh`
    static bool is_warp_tiling_legal(const int& block_m, const int& block_n) {
        const int num_math_warps = get_num_math_threads(block_m) / 32;
        const int num_warps_m = block_m >= 64 ? block_m / 64 : 1;
        if (num_math_warps % num_warps_m != 0)
            return false;
        const int num_warps_n = num_math_warps / num_warps_m;
        if (block_n % num_warps_n != 0)
            return false;
        const int warp_m = block_m / num_warps_m, warp_n = block_n / num_warps_n;
        if (warp_m % 16 != 0 or warp_n % 16 != 0 or 128 % warp_n != 0)
            return false;
        return (warp_m / 16) * (warp_n / 8) * 4 <= kNumMaxAccumPerThread;
    }

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // 1D2D: FP32 scales promoted on CUDA cores; 1D1D: packed UE8M0 scales applied by the block-scaled MMA
        DG_HOST_ASSERT(desc.kernel_type == KernelType::Kernel1D2D or desc.kernel_type == KernelType::Kernel1D1D);
        DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K and desc.major_b == cute::UMMA::Major::K);

        // Block M candidates
        std::vector<int> block_m_candidates;
        if (desc.gemm_type == GemmType::MGroupedContiguous or
            desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
            block_m_candidates = {heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
        } else {
            // Small M blocks help memory-bound (decoding) shapes fill the GPU; skip blocks much larger than M
            for (const int block_m: {16, 32, 64, 128}) {
                if (block_m == 16 or block_m < 2 * desc.get_expected_m())
                    block_m_candidates.push_back(block_m);
            }
        }

        // Block N candidates
        std::vector<int> block_n_candidates;
        for (const int block_n: {64, 128, 256}) {
            if (block_n % heuristics_runtime->get_block_n_multiple_of() == 0)
                block_n_candidates.push_back(block_n);
        }

        const int block_k = 128;

        // Developer-only override for tuning sweeps, e.g. `DG_SM120_FORCE_LAYOUT=128x128`
        const auto forced = deep_jit::get_env<std::string>("DG_SM120_FORCE_LAYOUT");
        if (not forced.empty()) {
            int block_m = 0, block_n = 0;
            DG_HOST_ASSERT(std::sscanf(forced.c_str(), "%dx%d", &block_m, &block_n) == 2);
            DG_HOST_ASSERT(is_warp_tiling_legal(block_m, block_n));
            return {Layout{0, block_m, block_n, block_k, 1, 1}};
        }

        std::vector<Layout> candidates;
        for (const int block_m: block_m_candidates) {
            if (block_m % heuristics_runtime->get_block_m_multiple_of() != 0)
                continue;
            for (const int block_n: block_n_candidates) {
                if (not is_warp_tiling_legal(block_m, block_n))
                    continue;
                const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
                const auto storage_config = get_storage_config(desc, layout);
                if (get_pipeline_config(desc, layout, storage_config).num_stages < 2)
                    continue;
                candidates.push_back(layout);
            }
        }
        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        DG_HOST_ASSERT(layout.swap_ab == 0);
        const auto swizzle_mode_a = get_swizzle_mode(layout.block_k, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(layout.block_k, c10::elementSize(desc.b_dtype));
        // D is stored directly from registers
        return {
            layout.block_m, layout.block_n,
            layout.block_m, layout.block_n,
            swizzle_mode_a, swizzle_mode_b, 0
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        // Barriers, plus the split-K "last CTA" flag
        const int smem_barriers = kNumMaxStages * 8 * 2 + 16;
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);
        // NOTES: the FP32 kernel reads SFA from global memory (L1/L2 cached), not staged. The block-scaled 1D1D
        // kernel TMA-loads the packed UE8M0 words of both operands with each stage: 2 words per row for FP4 (a K block
        // is 256 elements), 1 for FP8, each tile rounded up to the 128B TMA destination alignment
        int smem_sf_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            const int num_sf_words = desc.a_dtype == kPackedFP4 ? 2 : 1;
            smem_sf_per_stage = align(num_sf_words * storage_config.load_block_m * 4, 128) +
                                align(num_sf_words * storage_config.load_block_n * 4, 128);
        }
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sf_per_stage;
        const int num_stages = std::min((get_smem_capacity() - smem_barriers) / smem_per_stage, kNumMaxStages);
        return {smem_barriers + num_stages * smem_per_stage, num_stages, 0};
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_tma_threads = 128;
        const int num_math_threads = get_num_math_threads(layout.block_m);
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

    // Stream-K: spread `num_tiles x num_k_blocks` iterations evenly over CTAs, for normal GEMMs whose
    // tiles do not fill whole waves (decoding and wave-quantized shapes)
    static StreamKConfig get_stream_k_config(const GemmDesc& desc, const Layout& layout) {
        if (desc.gemm_type != GemmType::Normal)
            return {};
        const int64_t num_tiles = get_num_tiles(desc, layout);
        const int64_t num_k_blocks = ceil_div(desc.k, layout.block_k);
        // Counters are per tile
        if (num_tiles > kNumMaxStreamKTiles)
            return {};

        // Each CTA should own enough K blocks to amortize its pipeline fill and the reduction
        const int min_iters = deep_jit::get_env<int>("DG_SM120_SK_MIN_ITERS", 4);
        const int64_t num_iters = num_tiles * num_k_blocks;
        const int num_ctas = static_cast<int>(std::min<int64_t>(desc.num_sms, num_iters / min_iters));
        if (num_ctas <= num_tiles)
            return {};

        // Compare with data-parallel waves (in K blocks per CTA), requiring a clear win
        const int64_t dp_cost = ceil_div<int64_t>(num_tiles, desc.num_sms) * num_k_blocks;
        const int64_t sk_cost = ceil_div<int64_t>(num_iters, num_ctas) + kStreamKFixupCost;
        // NOTES: opt-in only: under CUDA graphs with cold weights (the serving condition) data-parallel tiles
        // were faster or equal on all measured DeepSeek shapes (`scripts/sm120_sweep_graph.py --sk 0,1`)
        if (deep_jit::get_env<int>("DG_SM120_FORCE_STREAM_K", 0) != 1 or sk_cost * 10 > dp_cost * 9)
            return {};

        // A tile spans at most `ceil(num_k_blocks / min_range) + 1` CTAs
        const int64_t min_range = std::max<int64_t>(1, num_iters / num_ctas);
        return {num_ctas, static_cast<int>(ceil_div<int64_t>(num_k_blocks, min_range) + 1)};
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_tiles = get_num_tiles(desc, layout);
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
        const double m = desc.get_expected_m(), n = desc.get_expected_n(), k = desc.get_expected_k();
        const double num_groups_a = desc.gemm_type == GemmType::MGroupedMasked ? desc.get_expected_num_groups() : 1;
        const double num_groups_b = desc.gemm_type == GemmType::Normal ? 1 : desc.num_groups;
        const double num_k_blocks = ceil_div(desc.get_expected_k(), layout.block_k);
        const double dram = (num_groups_b * n * k + num_groups_a * (m * k + 2 * m * n)) / kDRAMBytesPerUnit;
        const double l2 = static_cast<double>(num_tiles) * num_k_blocks * (layout.block_m + layout.block_n) * layout.block_k / kL2BytesPerUnit;
        const double cta = num_k_blocks * get_relative_wave_cost(layout) / 1000.0 * kCTACostScale + kWaveOverhead;
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
        // Fallback: proportional to the tile area
        return bm * bn * 1000 / (128 * 128);
    }

    // NOTES: ties keep the earlier candidate (smaller tiles first), as the fitting script does
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

} // namespace deep_gemm
