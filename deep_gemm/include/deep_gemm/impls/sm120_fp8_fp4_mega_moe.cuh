#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>
namespace deep_gemm {
// SM120 mega MoE (`fp8_fp4_mega_moe`, one rank): the SM100 kernel fuses dispatch, both expert GEMMs, SwiGLU and combine
// in one persistent kernel around TMEM accumulators, 2-CTA UMMA and NVLink symmetric memory. SM120 has none of those, so
// the same computation runs as a short sequence of kernels with the SM100 numerics:
//  1. `sm120_mega_moe_dispatch`: count the routed (token, slot) pairs of every local expert, give each expert a run of rows
//     starting at a multiple of `BLOCK_M` (the PSUM layout of the m-grouped GEMM), and record each pair's row
//  2. `sm120_mega_moe_gather`: copy each pair's FP8 token row, its packed UE8M0 scales and its top-k weight into its row
//  3. L1: the m-grouped FP8 x FP4 block-scaled GEMM (`sm120_fp8_fp4_gemm_1d1d`, PSUM layout), BF16 output
//  4. `sm120_mega_moe_swiglu`: clamp, SwiGLU, top-k weight, per-32 UE8M0 FP8 cast of the interleaved gate/up rows
//  5. L2: the same GEMM, BF16 output
//  6. `sm120_mega_moe_combine`: per token, the FP32 sum of its slots' BF16 rows in top-k order, cast to BF16
// Steps 4 and 6 reproduce the SM100 epilogue and combine arithmetic operation by operation (BF16 rounding of the L1
// accumulators, BF16 clamp, `__expf` and `rcp.approx` with fast math, round-to-nearest multiplies, UE8M0 scale from the
// amax of 32 outputs, FP32 accumulation of BF16 rows)

// Count the pairs of each expert, lay the experts out, and assign each pair its row. One CTA
template <uint32_t kNumExperts, uint32_t kNumTopk, uint32_t BLOCK_M, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm120_mega_moe_dispatch(const int64_t* topk_idx, int* pair_rows, int* psum_layout, int* expert_stats,
                        const uint32_t num_tokens) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumThreads % 32 == 0 and kNumExperts <= 32 * 32, "Invalid dispatch configuration");
    __shared__ int expert_counts[kNumExperts];
    __shared__ int expert_starts[kNumExperts];
    cudaGridDependencySynchronize();
    for (uint32_t i = threadIdx.x; i < kNumExperts; i += kNumThreads)
        expert_counts[i] = 0;
    __syncthreads();
    // Rank of each pair inside its expert (the order between pairs of one expert does not change any value)
    const uint32_t num_pairs = num_tokens * kNumTopk;
    for (uint32_t i = threadIdx.x; i < num_pairs; i += kNumThreads) {
        const auto expert_idx = topk_idx[i];
        pair_rows[i] = (expert_idx >= 0 and expert_idx < kNumExperts) ? atomicAdd(expert_counts + expert_idx, 1) : -1;
    }
    __syncthreads();
    // Exclusive prefix sum of the block-aligned counts, by the first warp (each lane owns a contiguous run of experts)
    if (threadIdx.x < 32) {
        constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExperts, 32u);
        const uint32_t lane_idx = threadIdx.x;
        int lane_sum = 0;
        #pragma unroll
        for (uint32_t j = 0; j < kNumExpertsPerLane; ++ j) {
            const uint32_t e = lane_idx * kNumExpertsPerLane + j;
            if (e < kNumExperts)
                lane_sum += math::align<int>(expert_counts[e], BLOCK_M);
        }
        int inclusive = lane_sum;
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffff, inclusive, offset);
            if (lane_idx >= offset)
                inclusive += other;
        }
        int start = inclusive - lane_sum;
        #pragma unroll
        for (uint32_t j = 0; j < kNumExpertsPerLane; ++ j) {
            const uint32_t e = lane_idx * kNumExpertsPerLane + j;
            if (e < kNumExperts) {
                const int count = expert_counts[e];
                expert_starts[e] = start;
                psum_layout[e] = start + count;
                if (expert_stats != nullptr)
                    expert_stats[e] += count;
                start += math::align<int>(count, BLOCK_M);
            }
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < num_pairs; i += kNumThreads) {
        const auto rank = pair_rows[i];
        if (rank >= 0)
            pair_rows[i] = expert_starts[topk_idx[i]] + rank;
    }
#endif
}

// Copy each routed pair's token row (FP8), packed scales and top-k weight into its GEMM row. One warp per pair
template <uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumWarps>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_gather(const uint8_t* x, const int* x_sf, const float* topk_weights, const int* pair_rows,
                      uint8_t* l1_acts, int* l1_acts_sf, float* row_weights,
                      const uint32_t num_tokens, const uint32_t sf_stride) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kHidden % 512 == 0, "A row is whole 16-byte chunks of every lane");
    constexpr uint32_t kNumSFWords = kHidden / 128;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t pair_idx = blockIdx.x * kNumWarps + threadIdx.x / 32;
    cudaGridDependencySynchronize();
    if (pair_idx >= num_tokens * kNumTopk)
        return;
    const int row = pair_rows[pair_idx];
    if (row < 0)
        return;
    const uint32_t token_idx = pair_idx / kNumTopk;
    const auto src = reinterpret_cast<const uint4*>(x + static_cast<uint64_t>(token_idx) * kHidden);
    const auto dst = reinterpret_cast<uint4*>(l1_acts + static_cast<uint64_t>(row) * kHidden);
    #pragma unroll
    for (uint32_t i = lane_idx; i < kHidden / 16; i += 32)
        dst[i] = __ldg(src + i);
    // Scales: `x_sf` is `[T, H / 128]` (K-major), the GEMM reads them MN-major (`[H / 128][sf_stride]`)
    for (uint32_t i = lane_idx; i < kNumSFWords; i += 32)
        l1_acts_sf[i * sf_stride + row] = __ldg(x_sf + token_idx * kNumSFWords + i);
    if (lane_idx == 0)
        row_weights[row] = __ldg(topk_weights + pair_idx);
#endif
}

// SwiGLU of the L1 output rows (gate/up interleaved in groups of 8 columns), times the pair's top-k weight, cast to FP8
// E4M3 with one UE8M0 scale per 32 outputs (MN-major packed words). One warp per (pair, 128 outputs): lane `l` loads the
// 8 BF16 columns `8l .. 8l + 7` of the 256 input columns (even lanes gate, odd lanes up), and computes outputs
// `4l .. 4l + 3` of the 128 with its neighbor's half
template <uint32_t kIntermediateHidden, uint32_t kNumTopk, uint32_t kNumWarps, bool kFastMath>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_swiglu(const nv_bfloat16* l1_out, const float* row_weights, const int* pair_rows,
                      uint8_t* l2_acts, int* l2_acts_sf, const float activation_clamp,
                      const uint32_t num_tokens, const uint32_t sf_stride) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kIntermediateHidden % 128 == 0, "Outputs are whole 128-element scale words");
    constexpr uint32_t kNumChunks = kIntermediateHidden / 128;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t task_idx = blockIdx.x * kNumWarps + threadIdx.x / 32;
    cudaGridDependencySynchronize();
    if (task_idx >= num_tokens * kNumTopk * kNumChunks)
        return;
    const uint32_t pair_idx = task_idx / kNumChunks, chunk_idx = task_idx % kNumChunks;
    const int row = pair_rows[pair_idx];
    if (row < 0)
        return;
    // Load this lane's 8 columns and its neighbor's (gate for odd lanes, up for even ones)
    const auto in_ptr = reinterpret_cast<const uint4*>(l1_out + static_cast<uint64_t>(row) * kIntermediateHidden * 2 + chunk_idx * 256);
    const uint4 mine = in_ptr[lane_idx];
    uint4 other;
    other.x = __shfl_xor_sync(0xffffffff, mine.x, 1);
    other.y = __shfl_xor_sync(0xffffffff, mine.y, 1);
    other.z = __shfl_xor_sync(0xffffffff, mine.z, 1);
    other.w = __shfl_xor_sync(0xffffffff, mine.w, 1);
    const bool is_gate_lane = lane_idx % 2 == 0;
    const uint4 gate_vec = is_gate_lane ? mine : other, up_vec = is_gate_lane ? other : mine;
    // Even lanes take the first 4 of the 8 outputs of the pair, odd lanes the last 4
    const uint32_t half = lane_idx % 2;
    const uint32_t gate_words[2] = {half ? gate_vec.z : gate_vec.x, half ? gate_vec.w : gate_vec.y};
    const uint32_t up_words[2] = {half ? up_vec.z : up_vec.x, half ? up_vec.w : up_vec.y};
    const float weight = row_weights[row];
    float values[4];
    float amax = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 2; ++ k) {
        auto bf16_gate = *reinterpret_cast<const nv_bfloat162*>(&gate_words[k]);
        auto bf16_up = *reinterpret_cast<const nv_bfloat162*>(&up_words[k]);
        // Clamp in BF16 (as SM100), then SwiGLU in FP32 with non-contracted round-to-nearest operations
        if (not isinf(activation_clamp)) {
            const auto clamp = __float2bfloat16_rn(activation_clamp);
            bf16_gate = __hmin2(bf16_gate, {clamp, clamp});
            bf16_up = __hmax2(bf16_up, {__hneg(clamp), __hneg(clamp)});
            bf16_up = __hmin2(bf16_up, {clamp, clamp});
        }
        const float2 gate = __bfloat1622float2(bf16_gate), up = __bfloat1622float2(bf16_up);
        const float gates[2] = {gate.x, gate.y}, ups[2] = {up.x, up.y};
        #pragma unroll
        for (uint32_t l = 0; l < 2; ++ l) {
            const float neg_gate_exp = kFastMath ? __expf(-gates[l]) : expf(-gates[l]);
            const float denom = __fadd_rn(1.0f, neg_gate_exp);
            const float silu = kFastMath ? __fmul_rn(gates[l], math::fast_rcp(denom)) : gates[l] / denom;
            values[k * 2 + l] = __fmul_rn(__fmul_rn(silu, ups[l]), weight);
            amax = fmaxf(amax, fabsf(values[k * 2 + l]));
        }
    }
    // Amax of 32 outputs = 8 lanes
    #pragma unroll
    for (uint32_t offset = 1; offset < 8; offset <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    const uint32_t sf_exp = math::get_ue8m0_sf_exp(amax);
    const float sf_inv = math::get_ue8m0_sf_inv<float>(sf_exp);
    const auto fp8x4 = __nv_fp8x4_e4m3(make_float4(__fmul_rn(values[0], sf_inv), __fmul_rn(values[1], sf_inv),
                                                   __fmul_rn(values[2], sf_inv), __fmul_rn(values[3], sf_inv)));
    reinterpret_cast<__nv_fp8x4_e4m3*>(l2_acts + static_cast<uint64_t>(row) * kIntermediateHidden + chunk_idx * 128)[lane_idx] = fp8x4;
    // One word of 4 scale bytes (outputs 0-31, 32-63, 64-95, 96-127) per 128 outputs
    const uint32_t sf_byte_0 = __shfl_sync(0xffffffff, sf_exp, 0), sf_byte_1 = __shfl_sync(0xffffffff, sf_exp, 8);
    const uint32_t sf_byte_2 = __shfl_sync(0xffffffff, sf_exp, 16), sf_byte_3 = __shfl_sync(0xffffffff, sf_exp, 24);
    if (lane_idx == 0)
        l2_acts_sf[chunk_idx * sf_stride + row] = static_cast<int>(sf_byte_0 | sf_byte_1 << 8 | sf_byte_2 << 16 | sf_byte_3 << 24);
#endif
}

// Per token, the FP32 sum of the BF16 L2 rows of its routed slots in top-k order, cast to BF16 (zero if none)
template <uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm120_mega_moe_combine(const nv_bfloat16* l2_out, const int* pair_rows, nv_bfloat16* y) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kHidden % 8 == 0, "Rows are whole 16-byte chunks");
    constexpr uint32_t kNumChunks = kHidden / 8;
    const uint32_t token_idx = blockIdx.x;
    cudaGridDependencySynchronize();
    int rows[kNumTopk];
    #pragma unroll
    for (uint32_t k = 0; k < kNumTopk; ++ k)
        rows[k] = pair_rows[token_idx * kNumTopk + k];
    for (uint32_t chunk_idx = threadIdx.x; chunk_idx < kNumChunks; chunk_idx += kNumThreads) {
        float2 reduced[4] = {};
        #pragma unroll
        for (uint32_t k = 0; k < kNumTopk; ++ k) {
            if (rows[k] < 0)
                continue;
            const auto values = __ldg(reinterpret_cast<const uint4*>(l2_out + static_cast<uint64_t>(rows[k]) * kHidden) + chunk_idx);
            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&values);
            #pragma unroll
            for (uint32_t l = 0; l < 4; ++ l)
                ptx::accumulate(reduced[l], bf16_values[l]);
        }
        uint4 casted;
        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
        #pragma unroll
        for (uint32_t l = 0; l < 4; ++ l)
            casted_bf16[l] = __float22bfloat162_rn(reduced[l]);
        reinterpret_cast<uint4*>(y + static_cast<uint64_t>(token_idx) * kHidden)[chunk_idx] = casted;
    }
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
