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
//     layout of the m-grouped GEMM), records each row's pair, and copies each row's FP8 token, its packed UE8M0 scales
//     and its top-k weight from the input (or the receive area) into the GEMM row. One kernel: every CTA counts the
//     layout and copies its share of the rows
//  4. L1: the m-grouped FP8 x FP4 block-scaled GEMM (`sm120_fp8_fp4_gemm_1d1d`, PSUM layout) with the SwiGLU epilogue
//     (`SM120MegaMoEEpilogue::SwiGLU`): the BF16-rounded tile, clamp, SwiGLU, top-k weight and the per-32 UE8M0 FP8 cast
//     of the interleaved gate/up columns go straight to the L2 input rows; the L1 output never reaches memory
//  5. L2: the same GEMM, BF16 output
//  6. `sm120_mega_moe_return` (several ranks): write each L2 row into its source rank's combine buffer, at the slot of
//     the pair
//  7. `sm120_mega_moe_barrier` (several ranks): every owner's rows are in this rank's combine buffer
//  8. `sm120_mega_moe_combine`: per token, the FP32 sum of its slots' BF16 rows in top-k order, cast to BF16
// The L1 epilogue and step 8 reproduce the SM100 epilogue and combine arithmetic operation by operation (BF16 rounding
// of the L1 accumulators, BF16 clamp, `__expf` and `rcp.approx` with fast math, round-to-nearest multiplies, UE8M0 scale
// from the amax of 32 outputs, FP32 accumulation of BF16 rows), so a token's result does not depend on where its experts
// live. Folding the combine into the L2 epilogue (the CTA that completes a token's last row sums it, counted with
// atomics) was measured and is slower: at one rank and 2 tokens the L2 GEMM took 7 us more and the call 5 us more,
// since every tile then ends with a fence, an atomic and a dependent read
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

// Lay the local experts out and gather the GEMM rows, in one kernel. Every CTA counts the pairs of each local expert
// (from every rank) and works out the same layout: each local expert gets a run of rows starting at a multiple of
// `BLOCK_M`, and a pair's row is its rank among the pairs of its expert (in pair order, so every CTA agrees without
// talking to the others). Then a CTA copies the rows of the pairs it owns (pair index plus source rank, modulo the grid):
// the FP8 token, its packed scales and its top-k weight, from the inputs of this rank or the receive area of the pair's
// source rank, and writes the pair's row to `pair_rows[source][pair]` (-1 if it targets another rank or is masked) and the
// row's pair to `row_pair[row]` (-1 for the tail of a run). Counting is redundant on purpose: the lists are short at
// decode, where the grid is a few CTAs, and a grid-wide wait between a count and a copy would need every CTA resident.
// The row to copy does not depend on the layout, so each warp loads its first token before the layout is known and
// stores it once the row is. CTA 0 also writes `psum_layout`, `total_rows` and the expert statistics. The other ranks'
// top-k lists are in `recv_topk_idx` (slot `source * num_max_tokens + token`), this rank's in `topk_idx`
template <uint32_t kNumRanks, uint32_t kNumLocalExperts, uint32_t kHidden, uint32_t kNumTopk, uint32_t BLOCK_M, uint32_t kNumWarps>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1) void
sm120_mega_moe_dispatch(const layout::SymBuffer<kNumRanks> sym, const uint8_t* x, const int* x_sf, const float* topk_weights,
                        const int64_t* topk_idx, const uint8_t* recv_x, const int* recv_x_sf, const float* recv_topk_weights,
                        const int64_t* recv_topk_idx, const SM120MegaMoECtrl* ctrl, int* pair_rows, const uint32_t pair_stride,
                        int* row_pair, int* psum_layout, int* expert_stats, int* total_rows, uint8_t* l1_acts,
                        int* l1_acts_sf, float* row_weights, const uint32_t sf_stride, const uint32_t num_tokens,
                        const uint32_t num_max_tokens) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    constexpr uint32_t kNumThreads = kNumWarps * 32;
    // A chunk is whole tokens of top-k slots
    constexpr uint32_t kChunkPairs = kNumThreads / kNumTopk * kNumTopk;
    DG_STATIC_ASSERT(kNumTopk <= kNumThreads, "A token's slots fit in a chunk");
    constexpr uint32_t kNumSFWords = kHidden / 128;
    constexpr uint32_t kNumRowChunks = kHidden / 512;   // 16-byte chunks per lane
    constexpr uint32_t kNumLaneSFWords = (kNumSFWords + 31) / 32;
    DG_STATIC_ASSERT(kNumLocalExperts <= kNumThreads and kNumLocalExperts <= 256 and kNumLocalExperts <= 32 * 32,
                     "Invalid dispatch configuration");
    DG_STATIC_ASSERT(kNumRanks <= kSM120MegaMoEMaxRanks and kHidden % 512 == 0, "Too many ranks, or a partial row chunk");
    __shared__ int expert_counts[kNumLocalExperts];
    __shared__ int expert_starts[kNumLocalExperts];
    __shared__ int expert_base[kNumLocalExperts];
    // Pairs of each local expert in the warps of one chunk of pairs
    __shared__ int warp_offsets[kNumWarps][kNumLocalExperts];
    // The row of each pair of the chunk (-1: another rank's expert, or masked)
    __shared__ int chunk_rows[kNumThreads];
    const uint32_t lane_idx = threadIdx.x % 32, warp_idx = threadIdx.x / 32;
    // Tokens of every rank (the peers published theirs in the barrier)
    cudaGridDependencySynchronize();
    uint32_t rank_tokens[kNumRanks];
    #pragma unroll
    for (uint32_t src = 0; src < kNumRanks; ++ src)
        rank_tokens[src] = (kNumRanks == 1 or src == sym.rank_idx) ? num_tokens : ptx::ld_volatile(reinterpret_cast<const uint32_t*>(ctrl->peer_tokens + src));
    const auto get_local_expert = [&](const int64_t& expert_idx) -> int {
        const auto local_idx = expert_idx - static_cast<int64_t>(sym.rank_idx) * kNumLocalExperts;
        return (expert_idx >= 0 and local_idx >= 0 and local_idx < kNumLocalExperts) ? static_cast<int>(local_idx) : -1;
    };
    const auto get_src_idx = [&](const uint32_t& src) {
        return (kNumRanks == 1 or src == sym.rank_idx) ? topk_idx : recv_topk_idx + static_cast<uint64_t>(src) * num_max_tokens * kNumTopk;
    };
    // A warp's row in registers: its 16-byte chunks, its scale words (lane `l` holds words `l`, `l + 32`, ...) and weight
    uint4 row_data[kNumRowChunks];
    int sf_word[kNumLaneSFWords] = {};
    float row_weight = 0;
    const auto load_row = [&](const uint32_t& src, const uint32_t& owned_pair,
                              uint4 (&data)[kNumRowChunks], int (&words)[kNumLaneSFWords], float& weight) {
        const bool is_local_src = kNumRanks == 1 or src == sym.rank_idx;
        const uint32_t token_idx = owned_pair / kNumTopk;
        const uint64_t slot = static_cast<uint64_t>(src) * num_max_tokens + token_idx;
        const auto src_row = reinterpret_cast<const uint4*>(is_local_src ? x + static_cast<uint64_t>(token_idx) * kHidden : recv_x + slot * kHidden);
        #pragma unroll
        for (uint32_t j = 0; j < kNumRowChunks; ++ j)
            data[j] = src_row[lane_idx + j * 32];
        // Scales: `x_sf` is `[T, H / 128]` (K-major)
        const auto src_sf = is_local_src ? x_sf + token_idx * kNumSFWords : recv_x_sf + slot * kNumSFWords;
        #pragma unroll
        for (uint32_t j = 0; j < kNumLaneSFWords; ++ j)
            words[j] = lane_idx + j * 32 < kNumSFWords ? src_sf[lane_idx + j * 32] : 0;
        weight = is_local_src ? topk_weights[owned_pair] : recv_topk_weights[slot * kNumTopk + owned_pair % kNumTopk];
    };
    for (uint32_t i = threadIdx.x; i < kNumLocalExperts; i += kNumThreads)
        expert_counts[i] = expert_base[i] = 0;
    __syncthreads();
    // Pairs of each local expert. The first chunk of every rank stays in registers for the rows below
    int first_chunk_expert[kNumRanks];
    #pragma unroll
    for (uint32_t src = 0; src < kNumRanks; ++ src) {
        const auto src_idx = get_src_idx(src);
        const uint32_t num_pairs = rank_tokens[src] * kNumTopk;
        first_chunk_expert[src] = threadIdx.x < kChunkPairs and threadIdx.x < num_pairs ? get_local_expert(src_idx[threadIdx.x]) : -1;
        if (first_chunk_expert[src] >= 0)
            atomicAdd(expert_counts + first_chunk_expert[src], 1);
        for (uint32_t i = kChunkPairs + threadIdx.x; i < num_pairs; i += kNumThreads) {
            const int local_idx = get_local_expert(src_idx[i]);
            if (local_idx >= 0)
                atomicAdd(expert_counts + local_idx, 1);
        }
    }
    __syncthreads();
    // Exclusive prefix sum of the block-aligned counts, by the first warp (each lane owns a contiguous run of experts)
    if (warp_idx == 0) {
        constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumLocalExperts, 32u);
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
                if (blockIdx.x == 0) {
                    psum_layout[e] = start + count;
                    if (expert_stats != nullptr)
                        expert_stats[e] += count;
                }
                start += math::align<int>(count, BLOCK_M);
            }
        }
        if (lane_idx == 31 and blockIdx.x == 0)
            *total_rows = inclusive;
    }
    __syncthreads();
    // The tail of each run has no pair (the GEMM computes it, nothing reads it). One warp per expert across the grid
    for (uint32_t e = blockIdx.x * kNumWarps + warp_idx; e < kNumLocalExperts; e += gridDim.x * kNumWarps) {
        const int count = expert_counts[e];
        for (int r = count + lane_idx; r < math::align<int>(count, BLOCK_M); r += 32)
            row_pair[expert_starts[e] + r] = -1;
    }
    // Rows. A chunk is one pair per thread (whole tokens): a warp's pairs of the same expert are ranked by lane
    // (`match_any`), the warps of the chunk are stacked in warp order, and the chunks in order, so a pair's rank does not
    // depend on timing
    #pragma unroll
    for (uint32_t src = 0; src < kNumRanks; ++ src) {
        const auto src_idx = get_src_idx(src);
        const uint32_t num_pairs = rank_tokens[src] * kNumTopk;
        for (uint32_t chunk_begin = 0; chunk_begin < num_pairs; chunk_begin += kChunkPairs) {
            const uint32_t chunk_end = chunk_begin + kChunkPairs < num_pairs ? chunk_begin + kChunkPairs : num_pairs;
            const uint32_t pair_idx = chunk_begin + threadIdx.x;
            const bool is_in_chunk = pair_idx < chunk_end;
            const uint32_t num_chunk_warps = (chunk_end - chunk_begin + 31) / 32;
            // This CTA owns the pairs `first_owned + n * gridDim.x`; warp `w` copies the `w`-th, `w + kNumWarps`-th, ...
            const uint32_t first_owned = chunk_begin + (blockIdx.x + 2 * gridDim.x - (chunk_begin + src) % gridDim.x) % gridDim.x;
            // Load the first row before the layout is known. A pair that is not routed here is loaded and dropped
            const uint32_t prefetch_pair = first_owned + warp_idx * gridDim.x;
            if (prefetch_pair < chunk_end)
                load_row(src, prefetch_pair, row_data, sf_word, row_weight);
            for (uint32_t i = threadIdx.x; i < num_chunk_warps * kNumLocalExperts; i += kNumThreads)
                warp_offsets[0][i] = 0;
            __syncthreads();
            const int local_idx = chunk_begin == 0 ? first_chunk_expert[src] : (is_in_chunk ? get_local_expert(src_idx[pair_idx]) : -1);
            const uint32_t same = __match_any_sync(0xffffffff, local_idx);
            const uint32_t lane_rank = __popc(same & ((1u << lane_idx) - 1));
            if (local_idx >= 0 and lane_rank == 0)
                warp_offsets[warp_idx][local_idx] = __popc(same);
            __syncthreads();
            if (threadIdx.x < kNumLocalExperts) {
                int running = expert_base[threadIdx.x];
                for (uint32_t w = 0; w < num_chunk_warps; ++ w) {
                    const int count = warp_offsets[w][threadIdx.x];
                    warp_offsets[w][threadIdx.x] = running;
                    running += count;
                }
                expert_base[threadIdx.x] = running;
            }
            __syncthreads();
            if (is_in_chunk) {
                const int row = local_idx >= 0 ? expert_starts[local_idx] + warp_offsets[warp_idx][local_idx] + static_cast<int>(lane_rank) : -1;
                chunk_rows[threadIdx.x] = row;
                if ((pair_idx + src) % gridDim.x == blockIdx.x) {
                    pair_rows[src * pair_stride + pair_idx] = row;
                    if (row >= 0)
                        row_pair[row] = static_cast<int>(src << kSM120MegaMoEPairBits | pair_idx);
                }
            }
            __syncthreads();
            // Store the rows: the prefetched one first, then the others
            for (uint32_t owned_pair = prefetch_pair; owned_pair < chunk_end; owned_pair += kNumWarps * gridDim.x) {
                const int row = chunk_rows[owned_pair - chunk_begin];
                if (owned_pair != prefetch_pair)
                    load_row(src, owned_pair, row_data, sf_word, row_weight);
                if (row < 0)
                    continue;
                const auto dst = reinterpret_cast<uint4*>(l1_acts + static_cast<uint64_t>(row) * kHidden);
                #pragma unroll
                for (uint32_t j = 0; j < kNumRowChunks; ++ j)
                    dst[lane_idx + j * 32] = row_data[j];
                // The GEMM reads the scales MN-major (`[H / 128][sf_stride]`)
                #pragma unroll
                for (uint32_t j = 0; j < kNumLaneSFWords; ++ j)
                    if (lane_idx + j * 32 < kNumSFWords)
                        l1_acts_sf[(lane_idx + j * 32) * sf_stride + row] = sf_word[j];
                if (lane_idx == 0)
                    row_weights[row] = row_weight;
            }
            __syncthreads();
        }
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

// Per token, the FP32 sum of the BF16 L2 rows of its routed slots in top-k order, cast to BF16 (zero if none). A CTA
// serves one token and a range of its columns. With several ranks the rows are in the combine buffer, at the slot of each pair. With one rank they are the L2 rows
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
    // `gridDim.y` CTAs share a token's columns (a decode batch has a handful of tokens)
    for (uint32_t chunk_idx = blockIdx.y * kNumThreads + threadIdx.x; chunk_idx < kNumChunks; chunk_idx += gridDim.y * kNumThreads) {
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
