#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// SM120 block-scaled dense MQA logits: logits[q, kv] = sum_h relu(Q[q,h,:] . KV[kv,:]) * weight[q,h], with Q and KV
// block-scaled by 32-element UE8M0 scales (MXFP8 E4M3 or MXFP4 E2M1). BF16 output. This mirrors the SM100 dense
// `fp8_fp4_mqa_logits` for arch 12: the tensor core applies the MX scales (no post-MMA float scale, unlike the FP8
// per-token-scale `sm120_fp8_mqa_logits`). Weights and logits are BF16 or FP32 (template parameters). The pipeline is that FP8 kernel's (dedicated TMA warp-group, staged Q/KV);
// the compute is the block-scaled warp MMA used by the sparse kernel (`warp_mma_16xNxHd_block_scaled_ss`).

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          bool kIsFP4, typename weights_t, typename logits_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm120_fp8_fp4_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                              const uint32_t stride_logits,
                              uint32_t* cu_seq_len_k_start,
                              uint32_t* cu_seq_len_k_end,
                              logits_t* logits,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
    constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
    constexpr uint32_t kRowBytes = kHeadDim / kPackFactor;         // 64 FP4 / 128 FP8 (bytes per token row)
    static constexpr uint32_t kNumAccum = BLOCK_Q * kNumHeads / 2;
    DG_STATIC_ASSERT(kHeadDim == 128, "SM120 block-scaled MQA logits only support head dim 128");
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory. Q/KV are block-scaled (kRowBytes per row), with a 128B (FP8) / 64B (FP4) swizzle; TMA needs the
    // tile base aligned to the swizzle atom (kRowBytes x 8 rows) and >= 1024 bytes. SF are one int32 UE8M0 word per
    // row (head dim 128 => one 4-group word). Weights are BF16 (the SM100 MX layout) or FP32.
    static constexpr uint32_t kSwizzleAlignment = cute::max(kRowBytes * 8, 1024u);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kRowBytes;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kRowBytes;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = math::constexpr_align(BLOCK_Q * kNumHeads * 4u, 128u);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = math::constexpr_align(BLOCK_KV * 4u, 128u);
    static constexpr uint32_t kWeightBytes = sizeof(weights_t);
    static constexpr uint32_t kNumWeightElems = math::constexpr_align(kNumHeads, 16u / kWeightBytes);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = math::constexpr_align(BLOCK_Q * kNumWeightElems * kWeightBytes, 128u);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + 0 + SMEM_Q_SIZE_PER_STAGE * i); });
    const uint32_t off_kv = kNumQStages * SMEM_Q_SIZE_PER_STAGE;
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + off_kv + SMEM_KV_SIZE_PER_STAGE * i); });
    const uint32_t off_sf_q = off_kv + kNumKVStages * SMEM_KV_SIZE_PER_STAGE;
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + off_sf_q + SMEM_SF_Q_SIZE_PER_STAGE * i); });
    const uint32_t off_sf_kv = off_sf_q + kNumQStages * SMEM_SF_Q_SIZE_PER_STAGE;
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + off_sf_kv + SMEM_SF_KV_SIZE_PER_STAGE * i); });
    const uint32_t off_w = off_sf_kv + kNumKVStages * SMEM_SF_KV_SIZE_PER_STAGE;
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<weights_t*>(smem_buffer + off_w + SMEM_WEIGHT_SIZE_PER_STAGE * i); });
    const uint32_t off_bar = math::constexpr_align(off_w + kNumQStages * SMEM_WEIGHT_SIZE_PER_STAGE, 16u);
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + off_bar);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) { full_q_barriers[i]->init(1); empty_q_barriers[i]->init(kNumMathThreads); }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) { full_kv_barriers[i]->init(1); empty_kv_barriers[i]->init(kNumMathThreads); }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 200;

    const auto sm_idx = blockIdx.x;
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        start = start / 4 * 4;   // SF-KV TMA alignment
        return {(q_iter_idx + q_iter_offset) % kNumQStages,
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1,
                start, math::ceil_div(end - start, BLOCK_KV)};
    };

    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {(num_total_kv_blocks + kv_block_idx) % kNumKVStages,
                ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1};
    };

    cudaGridDependencySynchronize();

    if (threadIdx.x >= kNumMathThreads) {
        // ---- Producer: dedicated TMA warp-group ----
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (not is_tma_load_warp)
            return;

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            // Q block-scaled rows (kRowBytes each), SF-Q (one int32 per row), weights (bf16)
            tma::copy<kRowBytes, BLOCK_Q * kNumHeads, kRowBytes>(&tensor_map_q, full_q_barriers[stage_idx],
                     smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_sf_q, full_q_barriers[stage_idx],
                     smem_sf_q[stage_idx], 0, block_idx * BLOCK_Q);
            tma::copy<kNumWeightElems, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx],
                     smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(
                SMEM_Q_SIZE_PER_STAGE + BLOCK_Q * kNumHeads * 4 + BLOCK_Q * kNumWeightElems * kWeightBytes);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
                    tma::copy<kRowBytes, BLOCK_KV, kRowBytes>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                             smem_sf_kv[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + BLOCK_KV * 4);
                }
                num_total_kv_blocks += num_kv_blocks;
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // ---- Consumer: math warp-groups ----
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& lane_idx = ptx::get_lane_idx();
        float accum[kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            full_q_barriers[q_stage_idx]->wait(q_phase);
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = static_cast<float>(smem_weights[q_stage_idx][i * kNumWeightElems + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2]);

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                mma::sm120::warp_mma_16xNxHd_block_scaled_ss<BLOCK_Q * kNumHeads, kHeadDim, kIsFP4>(
                    mma::sm120::to_smem_addr(smem_kv[kv_stage_idx]), warp_idx * 16,
                    mma::sm120::to_smem_addr(smem_q[q_stage_idx]),
                    smem_sf_kv[kv_stage_idx], BLOCK_KV, smem_sf_q[q_stage_idx], BLOCK_Q * kNumHeads, accum);

                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                empty_kv_barriers[kv_stage_idx]->arrive();

                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](const uint32_t& j) { return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)]; };
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j)
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; ++ k) sum[k] += transform(j * 4 + k);
                    float v_0 = sum[0] + sum[1];
                    float v_1 = sum[2] + sum[3];
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }
                    const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits);
                    if (seq_k_start[i] <= kv_offset + v_0_offset and kv_offset + v_0_offset < seq_k_end[i])
                        logits[q_offset + kv_offset + v_0_offset - seq_k_start[i]] = static_cast<logits_t>(v_0);
                    if (seq_k_start[i] <= kv_offset + v_1_offset and kv_offset + v_1_offset < seq_k_end[i])
                        logits[q_offset + kv_offset + v_1_offset - seq_k_start[i]] = static_cast<logits_t>(v_1);
                }
            }
            num_total_kv_blocks += num_kv_blocks;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            empty_q_barriers[q_stage_idx]->arrive();
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
