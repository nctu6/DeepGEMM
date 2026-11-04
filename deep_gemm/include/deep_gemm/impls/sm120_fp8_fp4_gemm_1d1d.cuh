#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

// SM120 block-scaled GEMM with packed UE8M0 scale factors ("1D1D"), for FP4 (E2M1) and FP8 (E4M3) operands.
//
// Same host contract as `sm100_fp8_fp4_gemm_1d1d_impl`: SFA/SFB are int32 tensors holding 4 consecutive-K UE8M0 bytes
// per word (K granularity `kGranK` of 32 or 128), laid out `[(G,) MN, ceil(K / (kGranK * 4))]`, MN-major and
// TMA-aligned. Unlike SM100 there is no TMEM: the scales are TMA-loaded into shared memory with each stage (on the
// same barrier as A/B, so their latency is hidden by the pipeline like the operands') and read into registers by the
// math warps:
//  - The tensor core applies the scales (`mma.sync ... block_scale`), so unlike the FP32 "1D2D" kernel there is no
//    second accumulator and no per-K-block promotion on CUDA cores
//  - FP4 (`kGranK == 32`): a TMA tile row is 128 bytes = 256 FP4 elements, consumed as 4 `m16n8k64` MMAs, each taking
//    the 2 scale bytes of its 64 K
//  - FP8: a tile row is 128 bytes = 128 elements, consumed as 4 `m16n8k32` MMAs. With `kGranK == 32` MMA `j` takes
//    byte `j` of the word, with `kGranK == 128` all MMAs of a K block share one byte
//  - A lane supplies the SFA scale of row `g` (lane `t == 0`) or `g + 8` (lane `t == 1`) and the SFB scale of column `g`
//    (lane `t == 0`), where `g = lane / 4`, `t = lane % 4`
// The TMA pipeline, scheduler, warp tiling, stream-K and epilogue are shared with `sm120_fp8_gemm_1d2d_impl`: with
// `kUseStreamK` (normal GEMMs whose tiles fill the GPU poorly, e.g. decoding with a long K), the
// `num_tiles x num_k_blocks` iteration space is split evenly over the CTAs, and a tile covered by several CTAs is
// reduced in split order through an FP32 `workspace` by the last one to finish (tracked by self-resetting `counters`)
template <bool kIsFP4, uint32_t kGranK,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs, GemmType kGemmType,
          bool kUseStreamK,
          typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm120_fp8_fp4_gemm_1d1d_impl(int* grouped_layout, cd_dtype_t* d, uint32_t ld_d,
                             float* workspace, int* counters,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using MMA = cute::conditional_t<kIsFP4, mma::sm120::FP4BlockScaledMMA, mma::sm120::FP8BlockScaledMMA>;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Checks. `BLOCK_K` is the byte width of a tile row (one 128B swizzle atom): 128 FP8 or 256 FP4 elements
    DG_STATIC_ASSERT(BLOCK_K == 128, "A K block is one 128-byte swizzle atom");
    DG_STATIC_ASSERT(kSwizzleAMode == 128 and kSwizzleBMode == 128, "Only 128B swizzling (one atom per K block) is supported");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedContiguous or
                     kGemmType == GemmType::MGroupedMasked, "Unsupported GEMM type");
    DG_STATIC_ASSERT(not kUseStreamK or kGemmType == GemmType::Normal, "Stream-K is only for normal GEMMs");
    DG_STATIC_ASSERT(kGranK == 32 or (not kIsFP4 and kGranK == 128), "FP4 needs 32-element scales; FP8 accepts 32 or 128");
    static constexpr uint32_t kElemsPerBlock = kIsFP4 ? BLOCK_K * 2 : BLOCK_K;
    static constexpr uint32_t kNumKSteps = BLOCK_K / 32;   // 32 bytes of A/B per MMA, for both FP8 and FP4
    // Packed scale words covering one K block
    static constexpr uint32_t kNumSFWords = kIsFP4 ? 2 : 1;

    // Warp tiling
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kNumWarpsM = BLOCK_M >= 64 ? BLOCK_M / 64 : 1;
    static constexpr uint32_t kNumWarpsN = kNumMathWarps / kNumWarpsM;
    static constexpr uint32_t WARP_M = BLOCK_M / kNumWarpsM;
    static constexpr uint32_t WARP_N = BLOCK_N / kNumWarpsN;
    static constexpr uint32_t kNumMTiles = WARP_M / 16;
    static constexpr uint32_t kNumNTiles = WARP_N / 8;
    static constexpr uint32_t kNumAccum = kNumMTiles * kNumNTiles * 4;
    DG_STATIC_ASSERT(kNumWarpsM * kNumWarpsN == kNumMathWarps, "Invalid warp grid");
    DG_STATIC_ASSERT(WARP_M % 16 == 0 and WARP_N % 16 == 0, "Invalid warp tile");
    // The scales ride the MMA, so only one accumulator set is live
    DG_STATIC_ASSERT(kNumAccum <= 128, "Too many accumulators (register spills)");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K;
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K;
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Swizzle-128B needs 1024B alignment");
    // Scales of one stage, `[kNumSFWords][BLOCK_MN]` words each (TMA destinations are 128B aligned)
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(kNumSFWords * BLOCK_M * 4u, 128u);
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(kNumSFWords * BLOCK_N * 4u, 128u);
    static constexpr uint32_t kNumSFTMABytes = kNumSFWords * (BLOCK_M + BLOCK_N) * 4;

    // Configs
    const uint32_t num_k_blocks = math::ceil_div(shape_k, kElemsPerBlock);
    // Number of packed words per row along K (a word covers `kGranK * 4` elements)
    const uint32_t num_sf_words = math::ceil_div(shape_k, kGranK * 4);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + i * SMEM_A_SIZE_PER_STAGE;
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE;
    });
    constexpr uint32_t SMEM_SFA_OFFSET = kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    constexpr uint32_t SMEM_SFB_OFFSET = SMEM_SFA_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + SMEM_SFA_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + SMEM_SFB_OFFSET + i * SMEM_SFB_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_BARRIER_OFFSET = SMEM_SFB_OFFSET + kNumStages * SMEM_SFB_SIZE_PER_STAGE;
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_BARRIER_OFFSET);
    auto full_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto smem_is_last_split = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2);
    if (warp_idx == kNumMathWarps + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Task iteration: a task is a tile, or with stream-K, a K range of a tile
    uint32_t tile_idx = 0, k_block_begin = 0, k_block_end = num_k_blocks;
    uint64_t sk_iter = 0, sk_iter_end = 0, sk_num_iters = 0;
    if constexpr (kUseStreamK) {
        sk_num_iters = static_cast<uint64_t>(scheduler.num_blocks) * num_k_blocks;
        sk_iter = sk_num_iters * blockIdx.x / gridDim.x;
        sk_iter_end = sk_num_iters * (blockIdx.x + 1) / gridDim.x;
    }
    // The CTA whose range contains iteration `x`
    const auto get_sk_owner = [&](const uint64_t& x) -> uint32_t {
        return static_cast<uint32_t>(((x + 1) * gridDim.x + sk_num_iters - 1) / sk_num_iters - 1);
    };
    auto get_next_task = [&](uint32_t& m_block_idx, uint32_t& n_block_idx) -> bool {
        if constexpr (not kUseStreamK) {
            return scheduler.get_next_block(m_block_idx, n_block_idx);
        } else {
            if (sk_iter >= sk_iter_end)
                return false;
            tile_idx = static_cast<uint32_t>(sk_iter / num_k_blocks);
            k_block_begin = static_cast<uint32_t>(sk_iter % num_k_blocks);
            k_block_end = static_cast<uint32_t>(cute::min<uint64_t>(num_k_blocks, k_block_begin + (sk_iter_end - sk_iter)));
            sk_iter += k_block_end - k_block_begin;
            scheduler.get_swizzled_block_idx(tile_idx, m_block_idx, n_block_idx);
            return true;
        }
    };

    // Pipeline phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathWarps) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
            constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
            while (get_next_task(m_block_idx, n_block_idx)) {
                for (uint32_t k_block_idx = k_block_begin; k_block_idx < k_block_end; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    auto& full_barrier = *full_barriers[stage_idx];
                    // NOTES: for FP4 the TMA K coordinate counts elements (16U4 data type), and the packed
                    // dtype makes one 256-element box a single 128B swizzle atom
                    using ab_dtype_t = cute::conditional_t<kIsFP4, cutlass::float_e2m1_t, uint8_t>;
                    const uint32_t k_idx = k_block_idx * kElemsPerBlock;
                    tma::copy<kElemsPerBlock, BLOCK_M, kSwizzleAMode, ab_dtype_t>(&tensor_map_a, &full_barrier,
                             reinterpret_cast<ab_dtype_t*>(smem_a[stage_idx]), k_idx,
                             scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx));
                    tma::copy<kElemsPerBlock, BLOCK_N, kSwizzleBMode, ab_dtype_t>(&tensor_map_b, &full_barrier,
                             reinterpret_cast<ab_dtype_t*>(smem_b[stage_idx]), k_idx,
                             scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx));
                    // Scales: inner coordinate is the row/column, outer is the packed word (plus the group offset).
                    // A 128-element FP8 K block uses a quarter of a 128-granularity word, so that word is re-fetched
                    const uint32_t sf_block_idx = (kIsFP4 or kGranK == 32) ? k_block_idx : k_block_idx / 4;
                    tma::copy<BLOCK_M, kNumSFWords, 0, uint32_t>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx],
                             m_block_idx * BLOCK_M,
                             scheduler.get_global_idx<kWithGroupOffsetA>(num_sf_words, kNumSFWords, sf_block_idx));
                    tma::copy<BLOCK_N, kNumSFWords, 0, uint32_t>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx],
                             n_block_idx * BLOCK_N,
                             scheduler.get_global_idx<true>(num_sf_words, kNumSFWords, sf_block_idx, m_block_idx));
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + kNumSFTMABytes);
                }
            }
        }
    } else {
        // Math warps
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const uint32_t warp_m_idx = warp_idx / kNumWarpsN;
        const uint32_t warp_n_idx = warp_idx % kNumWarpsN;
        const uint32_t group_row = lane_idx / 4, group_col = lane_idx % 4;

        // `ldmatrix` source rows/chunks of this lane (same pattern as the FP32 kernel; an MMA consumes 2 chunks of 16B)
        const uint32_t a_ld_row = warp_m_idx * WARP_M + ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
        const uint32_t a_ld_chunk = lane_idx >> 4;
        const uint32_t b_ld_row = warp_n_idx * WARP_N + (lane_idx >> 4) * 8 + (lane_idx & 7);
        const uint32_t b_ld_chunk = (lane_idx >> 3) & 1;

        while (get_next_task(m_block_idx, n_block_idx)) {
            // Scale rows/columns of this lane inside the stage's scale tile
            //  - SFA: lanes `t == 0` and `t == 1` supply rows `g` and `g + 8` of every M tile
            //  - SFB: every lane supplies column `g` of every N tile (only `t == 0` is read by the MMA)
            uint32_t sfa_rows[kNumMTiles], sfb_cols[kNumNTiles];
            #pragma unroll
            for (uint32_t i = 0; i < kNumMTiles; ++ i)
                sfa_rows[i] = warp_m_idx * WARP_M + i * 16 + group_row + (group_col & 1) * 8;
            #pragma unroll
            for (uint32_t j = 0; j < kNumNTiles; ++ j)
                sfb_cols[j] = warp_n_idx * WARP_N + j * 8 + group_row;

            float accum[kNumAccum];
            #pragma unroll
            for (uint32_t i = 0; i < kNumAccum; ++ i)
                accum[i] = 0;

            auto empty_barrier_arrive = [&]() {
                // `ldmatrix` reads the stage through the generic proxy, while the next TMA load writes it through
                // the async proxy: the write must not overtake the reads
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_idx == 0)
                    empty_barriers[stage_idx]->arrive();
            };

            if (scheduler.is_computation_valid(m_block_idx, warp_m_idx * WARP_M)) {
                #pragma unroll 1
                for (uint32_t k_block_idx = k_block_begin; k_block_idx < k_block_end; advance_pipeline(k_block_idx)) {
                    // Wait TMA arrivals (operands and scales)
                    full_barriers[stage_idx]->wait(phase);

                    // The packed scale words of this K block
                    uint32_t sfa_words[kNumMTiles][kNumSFWords], sfb_words[kNumNTiles][kNumSFWords];
                    #pragma unroll
                    for (uint32_t w = 0; w < kNumSFWords; ++ w) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i)
                            sfa_words[i][w] = smem_sfa[stage_idx][w * BLOCK_M + sfa_rows[i]];
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j)
                            sfb_words[j][w] = smem_sfb[stage_idx][w * BLOCK_N + sfb_cols[j]];
                    }

                    const uint32_t a_base = mma::sm120::to_smem_addr(smem_a[stage_idx]);
                    const uint32_t b_base = mma::sm120::to_smem_addr(smem_b[stage_idx]);

                    #pragma unroll
                    for (uint32_t k = 0; k < kNumKSteps; ++ k) {
                        uint32_t a_frag[kNumMTiles][4], b_frag[kNumNTiles][2];
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            mma::sm120::ldmatrix_x4(a_frag[i][0], a_frag[i][1], a_frag[i][2], a_frag[i][3],
                                                    a_base + mma::sm120::get_swizzle_128b_offset(a_ld_row + i * 16, k * 2 + a_ld_chunk));
                        }
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles / 2; ++ j) {
                            mma::sm120::ldmatrix_x4(b_frag[j * 2][0], b_frag[j * 2][1], b_frag[j * 2 + 1][0], b_frag[j * 2 + 1][1],
                                                    b_base + mma::sm120::get_swizzle_128b_offset(b_ld_row + j * 16, k * 2 + b_ld_chunk));
                        }

                        // The scale operand of this K step, in the low bits of a register
                        const auto get_scale = [&](const uint32_t (&words)[kNumSFWords]) -> uint32_t {
                            if constexpr (kIsFP4)
                                return (words[k / 2] >> (16 * (k % 2))) & 0xffffu;
                            else if constexpr (kGranK == 32)
                                return (words[0] >> (8 * k)) & 0xffu;
                            else
                                return (words[0] >> (8 * (k_block_idx % 4))) & 0xffu;
                        };
                        uint32_t scale_a[kNumMTiles], scale_b[kNumNTiles];
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i)
                            scale_a[i] = get_scale(sfa_words[i]);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j)
                            scale_b[j] = get_scale(sfb_words[j]);

                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumNTiles; ++ j)
                                MMA::mma(a_frag[i], b_frag[j], accum + (i * kNumNTiles + j) * 4, scale_a[i], scale_b[j]);
                        }
                    }

                    // Release the stage
                    empty_barrier_arrive();
                }
            } else {
                #pragma unroll 1
                for (uint32_t k_block_idx = k_block_begin; k_block_idx < k_block_end; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // Stream-K: publish this CTA's FP32 partial of a shared tile, and let the last contributor reduce
            if constexpr (kUseStreamK) {
                const uint32_t first_owner = get_sk_owner(static_cast<uint64_t>(tile_idx) * num_k_blocks);
                const uint32_t num_contributors = get_sk_owner(static_cast<uint64_t>(tile_idx + 1) * num_k_blocks - 1) - first_owner + 1;
                const uint32_t k_split_idx = blockIdx.x - first_owner;
                // A tile computed by a single CTA is stored directly
                if (num_contributors > 1) {
                    const uint64_t split_stride = static_cast<uint64_t>(shape_m) * shape_n;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                        #pragma unroll
                        for (uint32_t half = 0; half < 2; ++ half) {
                            const uint32_t row = m_block_idx * BLOCK_M + warp_m_idx * WARP_M + i * 16 + group_row + half * 8;
                            if (row >= shape_m)
                                continue;
                            auto partial_row = workspace + k_split_idx * split_stride + static_cast<uint64_t>(row) * shape_n;
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                                const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + j * 8 + group_col * 2;
                                if (col >= shape_n)
                                    continue;
                                const uint32_t idx = (i * kNumNTiles + j) * 4 + half * 2;
                                __stcg(reinterpret_cast<float2*>(partial_row + col), make_float2(accum[idx], accum[idx + 1]));
                            }
                        }
                    }
                    __threadfence();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                    if (threadIdx.x == 0) {
                        const bool is_last = atomicAdd(counters + tile_idx, 1) == static_cast<int>(num_contributors) - 1;
                        // Reset for the next launch (all splits of this tile have arrived)
                        if (is_last)
                            counters[tile_idx] = 0;
                        *smem_is_last_split = is_last;
                    }
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                    if (not *smem_is_last_split)
                        continue;
                    __threadfence();

                    // Deterministic reduction in split order. The splits are summed in order, but unrolled so that the
                    // loads of consecutive splits overlap instead of each waiting out an L2 round trip
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        accum[i] = 0;
                    #pragma unroll 4
                    for (uint32_t split = 0; split < num_contributors; ++ split) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            #pragma unroll
                            for (uint32_t half = 0; half < 2; ++ half) {
                                const uint32_t row = m_block_idx * BLOCK_M + warp_m_idx * WARP_M + i * 16 + group_row + half * 8;
                                if (row >= shape_m)
                                    continue;
                                auto partial_row = workspace + split * split_stride + static_cast<uint64_t>(row) * shape_n;
                                #pragma unroll
                                for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                                    const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + j * 8 + group_col * 2;
                                    if (col >= shape_n)
                                        continue;
                                    const uint32_t idx = (i * kNumNTiles + j) * 4 + half * 2;
                                    const auto value = __ldcg(reinterpret_cast<const float2*>(partial_row + col));
                                    accum[idx] += value.x, accum[idx + 1] += value.y;
                                }
                            }
                        }
                    }
                }
            }

            // Write back BF16 results directly from registers
            // NOTES: lanes `t` and `t ^ 1` exchange halves so that each stores 4 consecutive columns (8 bytes), and
            // the 4 lanes of a row cover 16 consecutive columns, i.e. one full 32-byte sector per row and store
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumNTiles % 2 == 0, "Paired stores need an even number of N tiles");
            const uint32_t m_global_base = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t m_local_base = m_block_idx * BLOCK_M;
            const bool is_odd_lane = group_col & 1;
            #pragma unroll
            for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                #pragma unroll
                for (uint32_t half = 0; half < 2; ++ half) {
                    const uint32_t row = warp_m_idx * WARP_M + i * 16 + group_row + half * 8;
                    const bool is_row_valid = m_local_base + row < shape_m;
                    auto d_row = reinterpret_cast<__nv_bfloat16*>(d) + static_cast<uint64_t>(m_global_base + row) * ld_d;
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumNTiles; j += 2) {
                        const uint32_t idx_lo = (i * kNumNTiles + j) * 4 + half * 2;
                        const uint32_t idx_hi = idx_lo + 4;
                        const auto lo = __float22bfloat162_rn({accum[idx_lo], accum[idx_lo + 1]});
                        const auto hi = __float22bfloat162_rn({accum[idx_hi], accum[idx_hi + 1]});
                        const uint32_t lo_u32 = *reinterpret_cast<const uint32_t*>(&lo);
                        const uint32_t hi_u32 = *reinterpret_cast<const uint32_t*>(&hi);
                        // Even lanes keep tile `j` (and get the odd lane's half), odd lanes keep tile `j + 1`
                        const uint32_t recv = __shfl_xor_sync(0xffffffff, is_odd_lane ? lo_u32 : hi_u32, 1);
                        const uint2 value = is_odd_lane ? make_uint2(recv, hi_u32) : make_uint2(lo_u32, recv);
                        const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + (j + is_odd_lane) * 8 + (group_col & 2) * 2;
                        if (is_row_valid and col < shape_n)
                            *reinterpret_cast<uint2*>(d_row + col) = value;
                    }
                }
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
