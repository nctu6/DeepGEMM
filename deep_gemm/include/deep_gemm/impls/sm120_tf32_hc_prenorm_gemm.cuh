#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// SM120 hyper-connection pre-norm GEMM: `d = a.float() @ b.T` (TF32 tensor cores) and `sqr_sum = (a.float() ** 2).sum(-1)`,
// with `a` BF16 `[m, k]`, `b` FP32 `[n, k]` (both K-major), `d` FP32. With `kNumSplits > 1`, K is split over CTAs and
// each split writes its own `d[split]` / `sqr_sum[split]` slice (the caller sums them).
//
// Same structure as `sm90_tf32_hc_prenorm_gemm_impl`, with `mma.sync m16n8k8` TF32 instead of WGMMA:
//  - One TMA warp loads 64x64 BF16 A tiles (128B swizzle) and n x 64 FP32 B tiles (two 32-float 128B swizzle atoms)
//  - 4 math warps own 16 rows each. A lane's A fragment of a K step (8 elements) is rows `g`, `g + 8` x columns `t`,
//    `t + 4` (`g = lane / 4`, `t = lane % 4`), read from shared memory, widened from BF16 to FP32 (exact in TF32) and also
//    squared and summed for `sqr_sum`. The B fragment (column `n = g`, K rows `t`, `t + 4`) is read straight from the
//    swizzled tile: the swizzle spreads a warp's 8 rows x 4 columns over 32 different banks
//  - `d` is stored straight from the accumulators (2 floats per lane and N tile)
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
sm120_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                float* d, const uint32_t ld_d, const uint32_t split_stride_d,
                                float* sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using MMA = mma::sm120::TF32MMA;

    constexpr uint32_t BLOCK_N = SHAPE_N;
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t kSwizzleBMode = 128;
    // A B tile row (64 floats) spans two 32-float swizzle atoms, each a `[BLOCK_N][32]` block
    constexpr uint32_t kNumFloatsPerAtom = kSwizzleBMode / sizeof(float);
    DG_STATIC_ASSERT(BLOCK_K == 64 and kSwizzleAMode == 128, "A tile rows must be one 128B swizzle atom");
    DG_STATIC_ASSERT(BLOCK_M == 64 and kNumMathThreads == 128, "4 math warps of 16 rows");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0 and BLOCK_N <= 128, "Invalid N");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;

    // Shared memory: A stages, B stages, barriers
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, "Swizzle-128B needs 1024B alignment");
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // K range of this CTA's split
    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t m_block_idx = blockIdx.x / kNumSplits;
    const uint32_t k_split_idx = blockIdx.x % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t num_k_blocks = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    // Wait for the previous kernel (PDL)
    cudaGridDependencySynchronize();

    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
            for (uint32_t s = 0; s < num_k_blocks; ++ s) {
                const uint32_t stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);
                const uint32_t k_idx = k_offset + s * BLOCK_K;
                tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_block_idx * BLOCK_M);
                tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<232>();

        constexpr uint32_t kNumNTiles = BLOCK_N / 8;
        constexpr uint32_t kNumKSteps = BLOCK_K / MMA::K;
        const uint32_t group_row = lane_idx / 4, group_col = lane_idx % 4;
        const uint32_t row = warp_idx * 16 + group_row;

        float accum[kNumNTiles * 4] = {};
        float sqr_sum_0 = 0, sqr_sum_1 = 0;

        for (uint32_t s = 0; s < num_k_blocks; ++ s) {
            const uint32_t stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            #pragma unroll
            for (uint32_t k = 0; k < kNumKSteps; ++ k) {
                // A: 16B chunk `k` (8 BF16) of rows `row` and `row + 8` of the 128B-swizzled tile (rows are 128 bytes)
                const auto a_base = reinterpret_cast<uint8_t*>(smem_a[stage_idx]);
                const auto a_upper = reinterpret_cast<nv_bfloat16*>(a_base + mma::sm120::get_swizzle_offset<kSwizzleAMode>(row, k));
                const auto a_lower = reinterpret_cast<nv_bfloat16*>(a_base + mma::sm120::get_swizzle_offset<kSwizzleAMode>(row + 8, k));
                const float a0 = __bfloat162float(a_upper[group_col]), a2 = __bfloat162float(a_upper[group_col + 4]);
                const float a1 = __bfloat162float(a_lower[group_col]), a3 = __bfloat162float(a_lower[group_col + 4]);
                sqr_sum_0 += a0 * a0 + a2 * a2;
                sqr_sum_1 += a1 * a1 + a3 * a3;
                const uint32_t a_frag[4] = {__float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2), __float_as_uint(a3)};

                // B: column `n`, K rows `8k + t` and `8k + t + 4`. The tile's 64-float rows are two 128B atoms, so
                // column `n` sits in atom `8k / 32`, local row `n`, 16B chunk `(8k % 32 + col) / 4`, lane `col % 4`
                const uint32_t atom = (k * MMA::K) / kNumFloatsPerAtom;
                const auto b_atom = reinterpret_cast<uint8_t*>(smem_b[stage_idx]) + atom * BLOCK_N * kSwizzleBMode;
                #pragma unroll
                for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                    const uint32_t n = j * 8 + group_row;
                    const uint32_t k0 = (k * MMA::K) % kNumFloatsPerAtom + group_col;
                    const auto b0 = b_atom + mma::sm120::get_swizzle_offset<kSwizzleBMode>(n, k0 / 4) + (k0 % 4) * 4;
                    const auto b1 = b_atom + mma::sm120::get_swizzle_offset<kSwizzleBMode>(n, (k0 + 4) / 4) + ((k0 + 4) % 4) * 4;
                    const uint32_t b_frag[2] = {*reinterpret_cast<const uint32_t*>(b0), *reinterpret_cast<const uint32_t*>(b1)};
                    MMA::mma(a_frag, b_frag, accum + j * 4);
                }
            }

            // Release the stage: the generic-proxy reads above must complete before the next TMA (async proxy) write
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane_idx == 0)
                empty_barriers[stage_idx]->arrive();
        }

        // Squared sums: lanes `t = 0..3` of a row hold 2 elements of each K step each
        sqr_sum_0 += __shfl_xor_sync(0xffffffff, sqr_sum_0, 1);
        sqr_sum_0 += __shfl_xor_sync(0xffffffff, sqr_sum_0, 2);
        sqr_sum_1 += __shfl_xor_sync(0xffffffff, sqr_sum_1, 1);
        sqr_sum_1 += __shfl_xor_sync(0xffffffff, sqr_sum_1, 2);
        const uint32_t m_idx = m_block_idx * BLOCK_M + row;
        const uint32_t sqr_sum_offset = shape_m * k_split_idx;
        if (group_col == 0) {
            if (m_idx < shape_m)
                sqr_sum[sqr_sum_offset + m_idx] = sqr_sum_0;
            if (m_idx + 8 < shape_m)
                sqr_sum[sqr_sum_offset + m_idx + 8] = sqr_sum_1;
        }

        // D: `accum[j * 4 + {0, 1}]` is row `g`, columns `8j + 2t + {0, 1}`; `{2, 3}` is row `g + 8`
        auto d_split = d + static_cast<uint64_t>(k_split_idx) * split_stride_d;
        #pragma unroll
        for (uint32_t j = 0; j < kNumNTiles; ++ j) {
            const uint32_t col = j * 8 + group_col * 2;
            if (m_idx < shape_m)
                *reinterpret_cast<float2*>(d_split + static_cast<uint64_t>(m_idx) * ld_d + col) = make_float2(accum[j * 4], accum[j * 4 + 1]);
            if (m_idx + 8 < shape_m)
                *reinterpret_cast<float2*>(d_split + static_cast<uint64_t>(m_idx + 8) * ld_d + col) = make_float2(accum[j * 4 + 2], accum[j * 4 + 3]);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
