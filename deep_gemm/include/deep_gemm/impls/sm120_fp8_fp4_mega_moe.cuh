#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
namespace deep_gemm {
// SM120 mega MoE (`fp8_fp4_mega_moe`): the SM100 kernel fuses dispatch, both expert GEMMs, SwiGLU and combine in one
// persistent kernel around TMEM accumulators, 2-CTA UMMA and NVLink symmetric memory. SM120 has none of those, so the
// same computation runs as a short sequence of kernels with the SM100 numerics, on one rank or on several ranks that
// write into each other's symmetric buffers through peer stores (CUDA IPC / PCIe P2P). Peer *loads* are not used for
// data: a PCIe read takes microseconds and the number in flight is small, while posted writes stream (measured: gathering
// the rows with peer loads took 0.5 ms for 100 rows, against microseconds for the stores)
//  1. `sm120_mega_moe_push` (several ranks): every rank sends its tokens' top-k lists, and the FP8 rows and scales of the
//     tokens that reach the destination's experts, to the destination's receive area
//  2. `sm120_mega_moe_barrier` (several ranks): every peer's pushes are in this rank's memory, and the peers are done with
//     the previous call's data in it
//  3. `sm120_mega_moe_dispatch`: the owner of the experts counts the routed (token, slot) pairs of every rank that
//     target its local experts, gives each local expert a run of rows starting at a multiple of `BLOCK_M` (the PSUM
//     layout of the m-grouped GEMM), and records each row's pair
//  4. `sm120_mega_moe_gather`: copy each row's FP8 token, its packed UE8M0 scales and its top-k weight from the input
//     (or the receive area) into the GEMM row
//  5. L1: the m-grouped FP8 x FP4 block-scaled GEMM (`sm120_fp8_fp4_gemm_1d1d`, PSUM layout), BF16 output
//  6. `sm120_mega_moe_swiglu`: clamp, SwiGLU, top-k weight, per-32 UE8M0 FP8 cast of the interleaved gate/up rows
//  7. L2: the same GEMM, BF16 output
//  8. `sm120_mega_moe_return` (several ranks): write each L2 row into its source rank's combine buffer, at the slot of
//     the pair
//  9. `sm120_mega_moe_barrier` (several ranks): every owner's rows are in this rank's combine buffer
//  10. `sm120_mega_moe_combine`: per token, the FP32 sum of its slots' BF16 rows in top-k order, cast to BF16
// Steps 6 and 10 reproduce the SM100 epilogue and combine arithmetic operation by operation (BF16 rounding of the L1
// accumulators, BF16 clamp, `__expf` and `rcp.approx` with fast math, round-to-nearest multiplies, UE8M0 scale from the
// amax of 32 outputs, FP32 accumulation of BF16 rows), so a token's result does not depend on where its experts live.
// All counts that depend on the routing (rows, tokens of a peer) are read on the device, so the launches are the same
// for every routing and the sequence can be captured in a CUDA graph. Cross-rank order comes only from the two barriers:
// each is a monotonically increasing epoch that every rank writes to every peer with a system-scope release and waits
// for with a system-scope acquire (the epoch lives in the symmetric buffer, so a replayed graph continues the count).
// Peers write only into the receive area and the combine buffer of a rank, and only after the previous call's barrier
// proved the owner done reading them, so the caller's input buffers are private to its rank

// Control block at the start of each rank's synchronization area in the symmetric buffer
struct SM120MegaMoECtrl {
    uint64_t flags[8];    // flags[p]: the last epoch rank `p` has signaled (written by rank `p`)
    uint64_t epoch;       // barriers this rank has entered
    int32_t peer_tokens[8]; // peer_tokens[p]: tokens of rank `p` in the current call (written by rank `p`)
};
constexpr uint32_t kSM120MegaMoEMaxRanks = 8;
// A row's pair is `(source rank << kPairBits) | (token * top-k + slot)`
constexpr uint32_t kSM120MegaMoEPairBits = 27;
// A pair's rank inside its expert is stored below the local expert index while the dispatch assigns rows
constexpr uint32_t kSM120MegaMoERankBits = 23;

// Send this rank's tokens to the other ranks: the top-k lists of every token, and the FP8 row and scales of a token to
// each rank that owns one of its experts. One warp per (token, destination); posted peer stores
template <uint32_t kNumRanks, uint32_t kNumLocalExperts, uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumWarps>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_push(const layout::SymBuffer<kNumRanks> sym, const uint8_t* x, const int* x_sf, const int64_t* topk_idx,
                    const float* topk_weights, uint8_t* recv_x, int* recv_x_sf, int64_t* recv_topk_idx, float* recv_topk_weights,
                    const uint32_t num_tokens, const uint32_t num_max_tokens) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumRanks >= 2 and kHidden % 512 == 0 and kNumTopk <= 32, "Invalid push configuration");
    constexpr uint32_t kNumSFWords = kHidden / 128;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t rank_idx = sym.rank_idx;
    cudaGridDependencySynchronize();
    const uint32_t num_tasks = num_tokens * (kNumRanks - 1);
    for (uint32_t task_idx = blockIdx.x * kNumWarps + threadIdx.x / 32; task_idx < num_tasks; task_idx += gridDim.x * kNumWarps) {
        const uint32_t token_idx = task_idx / (kNumRanks - 1), j = task_idx % (kNumRanks - 1);
        const uint32_t dst_rank = j < rank_idx ? j : j + 1;
        // Where this rank's slot in the destination's receive area starts
        const uint64_t slot = static_cast<uint64_t>(rank_idx) * num_max_tokens + token_idx;
        // Does the token reach one of the destination's experts?
        bool is_routed = false;
        int64_t expert_idx = -1;
        float weight = 0;
        if (lane_idx < kNumTopk) {
            expert_idx = topk_idx[token_idx * kNumTopk + lane_idx];
            weight = topk_weights[token_idx * kNumTopk + lane_idx];
            is_routed = expert_idx >= 0 and static_cast<uint32_t>(expert_idx / kNumLocalExperts) == dst_rank;
        }
        is_routed = __any_sync(0xffffffff, is_routed);
        if (lane_idx < kNumTopk) {
            sym.map(recv_topk_idx, dst_rank)[slot * kNumTopk + lane_idx] = expert_idx;
            sym.map(recv_topk_weights, dst_rank)[slot * kNumTopk + lane_idx] = weight;
        }
        if (not is_routed)
            continue;
        const auto src_row = reinterpret_cast<const uint4*>(x + static_cast<uint64_t>(token_idx) * kHidden);
        const auto dst_row = reinterpret_cast<uint4*>(sym.map(recv_x, dst_rank) + slot * kHidden);
        #pragma unroll
        for (uint32_t i = lane_idx; i < kHidden / 16; i += 32)
            dst_row[i] = __ldg(src_row + i);
        for (uint32_t i = lane_idx; i < kNumSFWords; i += 32)
            sym.map(recv_x_sf, dst_rank)[slot * kNumSFWords + i] = __ldg(x_sf + token_idx * kNumSFWords + i);
    }
#endif
}

// Wait until every rank has entered the same barrier as this one, and publish this rank's token count
template <uint32_t kNumRanks>
CUTLASS_GLOBAL __launch_bounds__(32, 1) void
sm120_mega_moe_barrier(const layout::SymBuffer<kNumRanks> sym, SM120MegaMoECtrl* ctrl, const uint32_t num_tokens,
                       const uint64_t timeout_cycles) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumRanks >= 2 and kNumRanks <= kSM120MegaMoEMaxRanks, "Invalid number of ranks");
    cudaGridDependencySynchronize();
    const uint32_t lane_idx = threadIdx.x;
    uint64_t epoch = 0;
    if (lane_idx == 0) {
        epoch = ctrl->epoch + 1;
        // Everything this stream wrote before (the pushes, the previous kernels' rows) is visible to the peers before
        // the flag
        #pragma unroll
        for (uint32_t peer = 0; peer < kNumRanks; ++ peer)
            sym.map(ctrl, peer)->peer_tokens[sym.rank_idx] = static_cast<int32_t>(num_tokens);
        __threadfence_system();
        #pragma unroll
        for (uint32_t peer = 0; peer < kNumRanks; ++ peer)
            ptx::st_rel_sys(sym.map(ctrl, peer)->flags + sym.rank_idx, epoch);
    }
    epoch = __shfl_sync(0xffffffff, epoch, 0);
    if (lane_idx < kNumRanks) {
        const auto start = clock64();
        while (ptx::ld_acq_sys(ctrl->flags + lane_idx) < epoch) {
            if (static_cast<uint64_t>(clock64() - start) > timeout_cycles) {
                printf("SM120 mega MoE barrier timeout: rank %u waits for rank %u at epoch %llu\n",
                       sym.rank_idx, lane_idx, static_cast<unsigned long long>(epoch));
                asm volatile("trap;");
            }
        }
    }
    __syncwarp();
    if (lane_idx == 0)
        ctrl->epoch = epoch;
#endif
}

// Count the pairs of each local expert (from every rank), lay the experts out, and assign each pair its row. One CTA.
// `pair_rows[source][pair]` is the pair's row at this owner (-1 if it targets another rank or is masked), `row_pair[row]`
// the pair of each row (-1 for the tail of a run), and `total_rows` the rows of all runs. The top-k lists of the
// other ranks are in `recv_topk_idx` (slot `source * num_max_tokens + token`), this rank's in `topk_idx`
template <uint32_t kNumRanks, uint32_t kNumLocalExperts, uint32_t kNumTopk, uint32_t BLOCK_M, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm120_mega_moe_dispatch(const layout::SymBuffer<kNumRanks> sym, const int64_t* topk_idx, const int64_t* recv_topk_idx,
                        const SM120MegaMoECtrl* ctrl, int* pair_rows, const uint32_t pair_stride, int* row_pair,
                        int* psum_layout, int* expert_stats, int* total_rows, const uint32_t num_tokens,
                        const uint32_t num_max_tokens) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumThreads % 32 == 0 and kNumLocalExperts <= 32 * 32 and kNumLocalExperts <= 256,
                     "Invalid dispatch configuration");
    DG_STATIC_ASSERT(kNumRanks <= kSM120MegaMoEMaxRanks, "Too many ranks");
    __shared__ int expert_counts[kNumLocalExperts];
    __shared__ int expert_starts[kNumLocalExperts];
    __shared__ int num_rows;
    cudaGridDependencySynchronize();
    for (uint32_t i = threadIdx.x; i < kNumLocalExperts; i += kNumThreads)
        expert_counts[i] = 0;
    __syncthreads();
    // Tokens of every rank (the peers published theirs in the barrier)
    uint32_t rank_tokens[kNumRanks];
    #pragma unroll
    for (uint32_t src = 0; src < kNumRanks; ++ src)
        rank_tokens[src] = (kNumRanks == 1 or src == sym.rank_idx) ? num_tokens : ptx::ld_volatile(reinterpret_cast<const uint32_t*>(ctrl->peer_tokens + src));
    // Rank of each pair inside its expert (the order between pairs of one expert does not change any value)
    #pragma unroll
    for (uint32_t src = 0; src < kNumRanks; ++ src) {
        const auto src_idx = (kNumRanks == 1 or src == sym.rank_idx) ? topk_idx : recv_topk_idx + static_cast<uint64_t>(src) * num_max_tokens * kNumTopk;
        for (uint32_t i = threadIdx.x; i < rank_tokens[src] * kNumTopk; i += kNumThreads) {
            const auto expert_idx = src_idx[i];
            const auto local_idx = expert_idx - static_cast<int64_t>(sym.rank_idx) * kNumLocalExperts;
            const bool is_local = expert_idx >= 0 and local_idx >= 0 and local_idx < kNumLocalExperts;
            pair_rows[src * pair_stride + i] = is_local ?
                static_cast<int>((local_idx << kSM120MegaMoERankBits) | atomicAdd(expert_counts + local_idx, 1)) : -1;
        }
    }
    __syncthreads();
    // Exclusive prefix sum of the block-aligned counts, by the first warp (each lane owns a contiguous run of experts)
    if (threadIdx.x < 32) {
        constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumLocalExperts, 32u);
        const uint32_t lane_idx = threadIdx.x;
        int lane_sum = 0;
        #pragma unroll
        for (uint32_t j = 0; j < kNumExpertsPerLane; ++ j) {
            const uint32_t e = lane_idx * kNumExpertsPerLane + j;
            if (e < kNumLocalExperts)
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
            if (e < kNumLocalExperts) {
                const int count = expert_counts[e];
                expert_starts[e] = start;
                psum_layout[e] = start + count;
                if (expert_stats != nullptr)
                    expert_stats[e] += count;
                start += math::align<int>(count, BLOCK_M);
            }
        }
        if (lane_idx == 31) {
            num_rows = inclusive;
            *total_rows = inclusive;
        }
    }
    __syncthreads();
    for (uint32_t row = threadIdx.x; row < static_cast<uint32_t>(num_rows); row += kNumThreads)
        row_pair[row] = -1;
    __syncthreads();
    #pragma unroll
    for (uint32_t src = 0; src < kNumRanks; ++ src) {
        for (uint32_t i = threadIdx.x; i < rank_tokens[src] * kNumTopk; i += kNumThreads) {
            const auto packed = pair_rows[src * pair_stride + i];
            if (packed >= 0) {
                const auto row = expert_starts[packed >> kSM120MegaMoERankBits] + (packed & ((1 << kSM120MegaMoERankBits) - 1));
                pair_rows[src * pair_stride + i] = row;
                row_pair[row] = static_cast<int>(src << kSM120MegaMoEPairBits | i);
            }
        }
    }
#endif
}

// Copy each row's token (FP8), packed scales and top-k weight into the GEMM row, from the inputs of this rank or the
// receive area of the pair's source rank. One warp per row
template <uint32_t kNumRanks, uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumWarps>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_gather(const layout::SymBuffer<kNumRanks> sym, const uint8_t* x, const int* x_sf, const float* topk_weights,
                      const uint8_t* recv_x, const int* recv_x_sf, const float* recv_topk_weights,
                      const int* row_pair, const int* total_rows, uint8_t* l1_acts, int* l1_acts_sf, float* row_weights,
                      const uint32_t sf_stride, const uint32_t num_max_tokens) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kHidden % 512 == 0, "A row is whole 16-byte chunks of every lane");
    constexpr uint32_t kNumSFWords = kHidden / 128;
    const uint32_t lane_idx = threadIdx.x % 32;
    cudaGridDependencySynchronize();
    const uint32_t num_rows = *total_rows;
    for (uint32_t row = blockIdx.x * kNumWarps + threadIdx.x / 32; row < num_rows; row += gridDim.x * kNumWarps) {
        const int pair = row_pair[row];
        if (pair < 0)
            continue;
        const uint32_t src = static_cast<uint32_t>(pair) >> kSM120MegaMoEPairBits;
        const uint32_t pair_idx = static_cast<uint32_t>(pair) & ((1u << kSM120MegaMoEPairBits) - 1);
        const uint32_t token_idx = pair_idx / kNumTopk;
        const bool is_local = kNumRanks == 1 or src == sym.rank_idx;
        const uint64_t slot = static_cast<uint64_t>(src) * num_max_tokens + token_idx;
        const auto src_row = reinterpret_cast<const uint4*>(is_local ? x + static_cast<uint64_t>(token_idx) * kHidden : recv_x + slot * kHidden);
        const auto dst = reinterpret_cast<uint4*>(l1_acts + static_cast<uint64_t>(row) * kHidden);
        #pragma unroll
        for (uint32_t i = lane_idx; i < kHidden / 16; i += 32)
            dst[i] = src_row[i];
        // Scales: `x_sf` is `[T, H / 128]` (K-major), the GEMM reads them MN-major (`[H / 128][sf_stride]`)
        const auto src_sf = is_local ? x_sf + token_idx * kNumSFWords : recv_x_sf + slot * kNumSFWords;
        for (uint32_t i = lane_idx; i < kNumSFWords; i += 32)
            l1_acts_sf[i * sf_stride + row] = src_sf[i];
        if (lane_idx == 0)
            row_weights[row] = is_local ? topk_weights[pair_idx] : recv_topk_weights[slot * kNumTopk + pair_idx % kNumTopk];
    }
#endif
}

// SwiGLU of the L1 output rows (gate/up interleaved in groups of 8 columns), times the pair's top-k weight, cast to FP8
// E4M3 with one UE8M0 scale per 32 outputs (MN-major packed words). One warp per (row, 128 outputs): lane `l` loads the
// 8 BF16 columns `8l .. 8l + 7` of the 256 input columns (even lanes gate, odd lanes up), and computes outputs
// `4l .. 4l + 3` of the 128 with its neighbor's half
template <uint32_t kIntermediateHidden, uint32_t kNumWarps, bool kFastMath>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_swiglu(const nv_bfloat16* l1_out, const float* row_weights, const int* row_pair, const int* total_rows,
                      uint8_t* l2_acts, int* l2_acts_sf, const float activation_clamp, const uint32_t sf_stride) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kIntermediateHidden % 128 == 0, "Outputs are whole 128-element scale words");
    constexpr uint32_t kNumChunks = kIntermediateHidden / 128;
    const uint32_t lane_idx = threadIdx.x % 32;
    cudaGridDependencySynchronize();
    const uint32_t num_tasks = *total_rows * kNumChunks;
    for (uint32_t task_idx = blockIdx.x * kNumWarps + threadIdx.x / 32; task_idx < num_tasks; task_idx += gridDim.x * kNumWarps) {
    const uint32_t row = task_idx / kNumChunks, chunk_idx = task_idx % kNumChunks;
    if (row_pair[row] < 0)
        continue;
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
    }
#endif
}

// Write each L2 row into the combine buffer of its source rank, at the slot (token, top-k slot) of the pair. One warp per
// row; posted peer stores
template <uint32_t kNumRanks, uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumWarps>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_return(const layout::SymBuffer<kNumRanks> sym, const nv_bfloat16* l2_out, const int* row_pair,
                      const int* total_rows, nv_bfloat16* combine_rows) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumRanks >= 2 and kHidden % 8 == 0, "Rows are whole 16-byte chunks");
    const uint32_t lane_idx = threadIdx.x % 32;
    cudaGridDependencySynchronize();
    const uint32_t num_rows = *total_rows;
    for (uint32_t row = blockIdx.x * kNumWarps + threadIdx.x / 32; row < num_rows; row += gridDim.x * kNumWarps) {
        const int pair = row_pair[row];
        if (pair < 0)
            continue;
        const uint32_t src = static_cast<uint32_t>(pair) >> kSM120MegaMoEPairBits;
        const uint32_t pair_idx = static_cast<uint32_t>(pair) & ((1u << kSM120MegaMoEPairBits) - 1);
        const auto src_row = reinterpret_cast<const uint4*>(l2_out + static_cast<uint64_t>(row) * kHidden);
        const auto dst_row = reinterpret_cast<uint4*>(sym.map(combine_rows, src) + static_cast<uint64_t>(pair_idx) * kHidden);
        #pragma unroll
        for (uint32_t i = lane_idx; i < kHidden / 8; i += 32)
            dst_row[i] = src_row[i];
    }
#endif
}

// Per token, the FP32 sum of the BF16 L2 rows of its routed slots in top-k order, cast to BF16 (zero if none). With
// several ranks the rows are in the combine buffer, at the slot of each pair. With one rank they are the L2 rows
template <uint32_t kNumRanks, uint32_t kNumLocalExperts, uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm120_mega_moe_combine(const nv_bfloat16* rows_base, const int64_t* topk_idx, const int* pair_rows, nv_bfloat16* y) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kHidden % 8 == 0, "Rows are whole 16-byte chunks");
    constexpr uint32_t kNumChunks = kHidden / 8;
    const uint32_t token_idx = blockIdx.x;
    cudaGridDependencySynchronize();
    const nv_bfloat16* rows[kNumTopk];
    #pragma unroll
    for (uint32_t k = 0; k < kNumTopk; ++ k) {
        const auto expert_idx = topk_idx[token_idx * kNumTopk + k];
        rows[k] = nullptr;
        if (expert_idx >= 0 and expert_idx < static_cast<int64_t>(kNumRanks * kNumLocalExperts)) {
            if constexpr (kNumRanks == 1) {
                const int row = pair_rows[token_idx * kNumTopk + k];
                if (row >= 0)
                    rows[k] = rows_base + static_cast<uint64_t>(row) * kHidden;
            } else {
                rows[k] = rows_base + static_cast<uint64_t>(token_idx * kNumTopk + k) * kHidden;
            }
        }
    }
    for (uint32_t chunk_idx = threadIdx.x; chunk_idx < kNumChunks; chunk_idx += kNumThreads) {
        float2 reduced[4] = {};
        #pragma unroll
        for (uint32_t k = 0; k < kNumTopk; ++ k) {
            if (rows[k] == nullptr)
                continue;
            const auto values = __ldcg(reinterpret_cast<const uint4*>(rows[k]) + chunk_idx);
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
