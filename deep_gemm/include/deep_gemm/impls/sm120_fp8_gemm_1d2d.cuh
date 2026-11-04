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

// SM120 FP8 GEMM with SM90-style "1D2D" scaling: FP32 per-token (1x128) SFA and per-block (128x128) SFB.
//
// Same contract as `sm90_fp8_gemm_1d2d_impl`, but built on `mma.sync` because SM120 has no WGMMA:
//  - One TMA warp-group loads A/B (128B swizzled) into a `kNumStages` ring guarded by mbarriers
//  - SFA is read by the math warps straight from global memory with runtime strides, so both the activation
//    quantizer's K-major `[M, K / 128]` scales and MN-major (TMA-aligned) scales work without a transpose kernel
//  - Math warps form a `kNumWarpsM x kNumWarpsN` grid; each warp owns a `WARP_M x WARP_N` tile,
//    accumulates one 128-K block with FP8 MMAs, then promotes it into FP32 with `sfa[row] * sfb[n / 128]`
//  - `WARP_N` divides 128, so a warp always sees a single SFB value per K block
//  - Results are stored straight from registers (no D staging buffer), which leaves the ~99 KB of
//    shared memory to the pipeline
//  - With `kUseStreamK` (normal GEMMs whose tiles fill the GPU poorly, e.g. decoding), the `num_tiles x num_k_blocks`
//    iteration space is split evenly into `gridDim.x` contiguous ranges, one per CTA. A tile covered by several
//    CTAs is reduced through `workspace`: each contributor writes an FP32 partial into its slot, and the last one
//    to finish (tracked by self-resetting `counters`) sums all slots in order, so results are deterministic
template <cute::UMMA::Major kMajorSFB,
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
sm120_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout, cd_dtype_t* d, uint32_t ld_d,
                         float* workspace, int* counters,
                         const float* sfa, uint32_t sfa_stride_g, uint32_t sfa_stride_m, uint32_t sfa_stride_k,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using MMA = mma::sm120::FP8MMA;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Scaling and layout checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(kSwizzleAMode == 128 and kSwizzleBMode == 128, "Only 128B swizzling (one atom per K block) is supported");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType != GemmType::Batched and not is_k_grouped_contiguous(kGemmType), "Unsupported GEMM type");
    DG_STATIC_ASSERT(not kUseStreamK or kGemmType == GemmType::Normal, "Stream-K is only for normal GEMMs");

    // Warp tiling
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kNumWarpsM = BLOCK_M >= 64 ? BLOCK_M / 64 : 1;
    static constexpr uint32_t kNumWarpsN = kNumMathWarps / kNumWarpsM;
    static constexpr uint32_t WARP_M = BLOCK_M / kNumWarpsM;
    static constexpr uint32_t WARP_N = BLOCK_N / kNumWarpsN;
    static constexpr uint32_t kNumMTiles = WARP_M / MMA::M;
    static constexpr uint32_t kNumNTiles = WARP_N / MMA::N;
    static constexpr uint32_t kNumAccum = kNumMTiles * kNumNTiles * 4;
    DG_STATIC_ASSERT(kNumWarpsM * kNumWarpsN == kNumMathWarps, "Invalid warp grid");
    DG_STATIC_ASSERT(WARP_M % MMA::M == 0, "Invalid warp M");
    DG_STATIC_ASSERT(WARP_N % 16 == 0 and 128 % WARP_N == 0, "Warp N must be a multiple of 16 that divides 128");
    DG_STATIC_ASSERT(kNumAccum <= 64, "Too many accumulators (register spills)");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Swizzle-128B needs 1024B alignment");
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    // Barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_SF_OFFSET);
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

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Task iteration: a task is a tile, or with stream-K, a K range of a tile
    const uint32_t num_k_blocks = num_total_k_blocks;
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

    // PDL: the weights (B) do not depend on the previous kernel. Before waiting for it, the producer issues the first
    // `kNumStages` K blocks of B of its first tile straight into their pipeline stages, so a CTA resident while the
    // previous kernel finishes starts with a full pipeline of weights (same scheme as `sm120_fp8_fp4_gemm_1d1d_impl`).
    // Only normal data-parallel GEMMs know their first tile without reading `grouped_layout`
    static constexpr bool kPreissueWeights = kGemmType == GemmType::Normal and not kUseStreamK;
    const uint32_t num_preissued_k_blocks = kPreissueWeights and blockIdx.x < scheduler.num_blocks ?
                                            cute::min(num_k_blocks, kNumStages) : 0;

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
            if constexpr (kPreissueWeights) {
                if (num_preissued_k_blocks > 0) {
                    uint32_t first_m_block_idx, first_n_block_idx;
                    scheduler.get_swizzled_block_idx(blockIdx.x, first_m_block_idx, first_n_block_idx);
                    for (uint32_t k_block_idx = 0; k_block_idx < num_preissued_k_blocks; ++ k_block_idx) {
                        // The stages start empty. The transaction is expected without an arrival, so the B bytes landing
                        // cannot complete the phase: the single arrival comes with A, after the wait
                        auto& full_barrier = *full_barriers[k_block_idx];
                        full_barrier.expect_transaction(SMEM_B_SIZE_PER_STAGE);
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3>(&tensor_map_b, &full_barrier,
                                 smem_b[k_block_idx], k_block_idx * BLOCK_K, first_n_block_idx * BLOCK_N);
                    }
                }
            }

            // Wait for the previous kernel before loading activations (and before reading `grouped_layout`)
            cudaGridDependencySynchronize();

            bool is_first_task = true;
            while (get_next_task(m_block_idx, n_block_idx)) {
                for (uint32_t k_block_idx = k_block_begin; k_block_idx < k_block_end; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx));
                    // B of the first stages of the first task was issued before the wait
                    if (is_first_task and k_block_idx < num_preissued_k_blocks) {
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                    } else {
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3>(&tensor_map_b, &full_barrier,
                                 smem_b[stage_idx], k_idx, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx));
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                    }
                }
                is_first_task = false;
            }
            // PDL: let the next kernel launch once its early weight loads no longer compete with ours for DRAM: here for
            // short K (every load is issued), at the end of the math warps' mainloop otherwise
            if (num_k_blocks <= kNumStages)
                cudaTriggerProgrammaticLaunchCompletion();
        }
    } else {
        // Math warps
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Wait for the previous kernel before reading its results (SFA) or writing memory it may still read
        cudaGridDependencySynchronize();

        const uint32_t warp_m_idx = warp_idx / kNumWarpsN;
        const uint32_t warp_n_idx = warp_idx % kNumWarpsN;
        const uint32_t group_row = lane_idx / 4, group_col = lane_idx % 4;

        // `ldmatrix` source rows/chunks of this lane
        //  - A x4: matrices are (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15, k 16-31)
        //  - B x4: matrices are (n 0-7, k 0-15), (n 0-7, k 16-31), (n 8-15, k 0-15), (n 8-15, k 16-31)
        const uint32_t a_ld_row = warp_m_idx * WARP_M + ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
        const uint32_t a_ld_chunk = lane_idx >> 4;
        const uint32_t b_ld_row = warp_n_idx * WARP_N + (lane_idx >> 4) * 8 + (lane_idx & 7);
        const uint32_t b_ld_chunk = (lane_idx >> 3) & 1;

        while (get_next_task(m_block_idx, n_block_idx)) {
            // SFB of this warp: `WARP_N` divides 128, so a single row (or column) of SFB per K block
            const auto group_sfb_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
            const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
            const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
            const uint32_t n_sfb_idx = cute::min((n_block_idx * BLOCK_N + warp_n_idx * WARP_N) / BLOCK_K, shape_n_sfb - 1);
            const float* local_sfb = sfb + group_sfb_offset + n_sfb_idx * stride_n_sfb;

            // SFA rows of this thread (clamped, as out-of-bound rows are never stored)
            const float* local_sfa = sfa + (kGemmType == GemmType::MGroupedMasked ? scheduler.current_group_idx * sfa_stride_g : 0);
            uint32_t sfa_row_offsets[kNumMTiles][2];
            #pragma unroll
            for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                #pragma unroll
                for (uint32_t half = 0; half < 2; ++ half) {
                    const uint32_t row = m_block_idx * BLOCK_M + warp_m_idx * WARP_M + i * MMA::M + group_row + half * 8;
                    sfa_row_offsets[i][half] = cute::min(row, shape_m - 1) * sfa_stride_m;
                }
            }

            float final_accum[kNumAccum] = {0};
            auto empty_barrier_arrive = [&]() {
                // `ldmatrix` reads the stage through the generic proxy, while the next TMA load writes it through
                // the async proxy: without this fence the write may overtake the reads (measured: random wrong
                // tiles with 8 math warps)
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_idx == 0)
                    empty_barriers[stage_idx]->arrive();
            };

            if (scheduler.is_computation_valid(m_block_idx, warp_m_idx * WARP_M)) {
                float scale_b = __ldg(local_sfb + k_block_begin * stride_k_sfb);
                float scale_a[kNumMTiles][2];
                #pragma unroll
                for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                    #pragma unroll
                    for (uint32_t half = 0; half < 2; ++ half)
                        scale_a[i][half] = __ldg(local_sfa + sfa_row_offsets[i][half] + k_block_begin * sfa_stride_k);
                }
                #pragma unroll 1
                for (uint32_t k_block_idx = k_block_begin; k_block_idx < k_block_end; advance_pipeline(k_block_idx)) {
                    // Prefetch the next K block's SFA and SFB before waiting for TMA
                    const uint32_t next_k_block_idx = cute::min(k_block_idx + 1, k_block_end - 1);
                    const float next_scale_b = __ldg(local_sfb + next_k_block_idx * stride_k_sfb);
                    float next_scale_a[kNumMTiles][2];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                        #pragma unroll
                        for (uint32_t half = 0; half < 2; ++ half)
                            next_scale_a[i][half] = __ldg(local_sfa + sfa_row_offsets[i][half] + next_k_block_idx * sfa_stride_k);
                    }

                    // Wait TMA arrivals
                    full_barriers[stage_idx]->wait(phase);

                    const uint32_t a_base = mma::sm120::to_smem_addr(smem_a[stage_idx]);
                    const uint32_t b_base = mma::sm120::to_smem_addr(smem_b[stage_idx]);

                    float accum[kNumAccum];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        accum[i] = 0;

                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / MMA::K; ++ k) {
                        uint32_t a_frag[kNumMTiles][4], b_frag[kNumNTiles][2];
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            mma::sm120::ldmatrix_x4(a_frag[i][0], a_frag[i][1], a_frag[i][2], a_frag[i][3],
                                                    a_base + mma::sm120::get_swizzle_128b_offset(a_ld_row + i * MMA::M, k * 2 + a_ld_chunk));
                        }
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles / 2; ++ j) {
                            mma::sm120::ldmatrix_x4(b_frag[j * 2][0], b_frag[j * 2][1], b_frag[j * 2 + 1][0], b_frag[j * 2 + 1][1],
                                                    b_base + mma::sm120::get_swizzle_128b_offset(b_ld_row + j * 16, k * 2 + b_ld_chunk));
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumNTiles; ++ j)
                                MMA::mma(a_frag[i], b_frag[j], accum + (i * kNumNTiles + j) * 4);
                        }
                    }


                    // Promote with scales
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                        const float scale_0 = scale_a[i][0] * scale_b, scale_1 = scale_a[i][1] * scale_b;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                            const uint32_t idx = (i * kNumNTiles + j) * 4;
                            final_accum[idx + 0] += scale_0 * accum[idx + 0];
                            final_accum[idx + 1] += scale_0 * accum[idx + 1];
                            final_accum[idx + 2] += scale_1 * accum[idx + 2];
                            final_accum[idx + 3] += scale_1 * accum[idx + 3];
                        }
                    }
                    // Release the stage
                    empty_barrier_arrive();
                    scale_b = next_scale_b;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMTiles; ++ i)
                        scale_a[i][0] = next_scale_a[i][0], scale_a[i][1] = next_scale_a[i][1];
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
                            const uint32_t row = m_block_idx * BLOCK_M + warp_m_idx * WARP_M + i * MMA::M + group_row + half * 8;
                            if (row >= shape_m)
                                continue;
                            auto partial_row = workspace + k_split_idx * split_stride + static_cast<uint64_t>(row) * shape_n;
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                                const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + j * MMA::N + group_col * 2;
                                if (col >= shape_n)
                                    continue;
                                const uint32_t idx = (i * kNumNTiles + j) * 4 + half * 2;
                                __stcg(reinterpret_cast<float2*>(partial_row + col), make_float2(final_accum[idx], final_accum[idx + 1]));
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

                    // Deterministic reduction in split order
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        final_accum[i] = 0;
                    #pragma unroll 1
                    for (uint32_t split = 0; split < num_contributors; ++ split) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            #pragma unroll
                            for (uint32_t half = 0; half < 2; ++ half) {
                                const uint32_t row = m_block_idx * BLOCK_M + warp_m_idx * WARP_M + i * MMA::M + group_row + half * 8;
                                if (row >= shape_m)
                                    continue;
                                auto partial_row = workspace + split * split_stride + static_cast<uint64_t>(row) * shape_n;
                                #pragma unroll
                                for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                                    const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + j * MMA::N + group_col * 2;
                                    if (col >= shape_n)
                                        continue;
                                    const uint32_t idx = (i * kNumNTiles + j) * 4 + half * 2;
                                    const auto value = __ldcg(reinterpret_cast<const float2*>(partial_row + col));
                                    final_accum[idx] += value.x, final_accum[idx + 1] += value.y;
                                }
                            }
                        }
                    }
                }
            }

            // PDL (long K, see the producer): the mainloop of this CTA is done
            if (num_k_blocks > kNumStages)
                cudaTriggerProgrammaticLaunchCompletion();

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
                    const uint32_t row = warp_m_idx * WARP_M + i * MMA::M + group_row + half * 8;
                    const bool is_row_valid = m_local_base + row < shape_m;
                    auto d_row = reinterpret_cast<__nv_bfloat16*>(d) + static_cast<uint64_t>(m_global_base + row) * ld_d;
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumNTiles; j += 2) {
                        const uint32_t idx_lo = (i * kNumNTiles + j) * 4 + half * 2;
                        const uint32_t idx_hi = idx_lo + 4;
                        const auto lo = __float22bfloat162_rn({final_accum[idx_lo], final_accum[idx_lo + 1]});
                        const auto hi = __float22bfloat162_rn({final_accum[idx_hi], final_accum[idx_hi + 1]});
                        const uint32_t lo_u32 = *reinterpret_cast<const uint32_t*>(&lo);
                        const uint32_t hi_u32 = *reinterpret_cast<const uint32_t*>(&hi);
                        // Even lanes keep tile `j` (and get the odd lane's half), odd lanes keep tile `j + 1`
                        const uint32_t recv = __shfl_xor_sync(0xffffffff, is_odd_lane ? lo_u32 : hi_u32, 1);
                        const uint2 value = is_odd_lane ? make_uint2(recv, hi_u32) : make_uint2(lo_u32, recv);
                        const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + (j + is_odd_lane) * MMA::N + (group_col & 2) * 2;
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
