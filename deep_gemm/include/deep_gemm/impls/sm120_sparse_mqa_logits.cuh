#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/sparse_mqa_logits.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// SM120 sparse MQA logits: logits[q, slot] = sum_h relu(Q[q,h,:] . KV[kv,:]) * weight[q,h], with Q and KV block-scaled
// by 32-element UE8M0 scales (MXFP8 or MXFP4), BF16 output written to compressed columns via the metadata slot map.
//
// The compute (block-scaled mma.sync + ReLU x weight head reduction) is the SM120 dense MQA kernel's; the schedule is
// the SM100 sparse metadata's (KV splits of sparse blocks, with per-block slot offsets). Unlike SM100 there is no TMEM:
// gathered sparse KV lands in an ldmatrix-swizzled smem tile and the MX scale words are staged in smem, read by
// `warp_mma_16xNxHd_block_scaled_ss`.
//
// Simple, correctness-first structure (one CTA per SM, persistent over schedule entries): a single warpgroup both
// gathers (cp.async) and computes, one KV split at a time. BLOCK_Q = 2, head dim 128.

namespace deep_gemm {

using namespace layout::sparse_mqa_logits;

// Paged sparse KV: resolve a physical sparse-block index to its token base pointer + SF pointer in the fused cache
struct PagedKVAccessor {
    const uint8_t* fused_kv_cache;
    uint32_t page_stride_bytes;
    uint32_t page_kv;
    uint32_t sparse_block_kv;
    uint32_t qk_bytes_per_token;   // head dim bytes (64 FP4 / 128 FP8)

    CUTLASS_DEVICE void block_ptrs(const uint32_t physical_kv_block_idx,
                                   const uint8_t*& kv, const uint32_t*& sf) const {
        const uint32_t blocks_per_page = page_kv / sparse_block_kv;
        const uint32_t page_idx = physical_kv_block_idx / blocks_per_page;
        const uint32_t block_in_page = physical_kv_block_idx % blocks_per_page;
        const auto page = fused_kv_cache + static_cast<uint64_t>(page_idx) * page_stride_bytes;
        const uint32_t token_in_page = block_in_page * sparse_block_kv;
        kv = page + token_in_page * qk_bytes_per_token;
        sf = reinterpret_cast<const uint32_t*>(page + page_kv * qk_bytes_per_token) + token_in_page;
    }
};

template <uint32_t kNumHeads, uint32_t SPARSE_BLOCK_KV, uint32_t SPLIT_KV, uint32_t kNumSMs,
          bool kIsFP4, bool kIsPaged>
CUTLASS_GLOBAL __launch_bounds__(256, 1)
void sm120_sparse_mqa_logits(const uint32_t logits_stride, nv_bfloat16* logits,
                             const uint8_t* metadata,
                             const uint8_t* kv, const uint32_t* sf_kv, const uint32_t num_kv_tokens,
                             const uint8_t* fused_kv_cache, const uint32_t page_stride_bytes, const uint32_t page_kv,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t BLOCK_Q = 2;
    constexpr uint32_t kHeadDim = 128;
    constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
    constexpr uint32_t kRowBytes = kHeadDim / kPackFactor;         // 64 FP4 / 128 FP8
    constexpr uint32_t kNumMathThreads = 256;                      // 8 warps, each owns 16 KV rows -> 128 of SPLIT_KV
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    constexpr uint32_t kN = BLOCK_Q * kNumHeads;                   // MMA N (Q rows)
    constexpr uint32_t kNumAccum = kN / 2;                         // per-lane accumulators (dense-kernel layout)
    constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    DG_STATIC_ASSERT(SPLIT_KV == 128 * (kNumMathWarps / 2) or SPLIT_KV % 128 == 0, "KV split must tile 16-row warps");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Shared memory: Q (ldmatrix swizzle), KV (ldmatrix swizzle), SF-Q, SF-KV, weights, metadata, barriers
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_q = reinterpret_cast<uint8_t*>(smem_buffer);
    constexpr uint32_t SMEM_Q = kN * kRowBytes;
    auto smem_kv = smem_q + SMEM_Q;
    constexpr uint32_t SMEM_KV = SPLIT_KV * kRowBytes;
    // TMA destinations (SF-Q via TMA, weights via TMA) must be 128-byte aligned; pad each region up to 128
    auto smem_sf_q = reinterpret_cast<uint32_t*>(smem_kv + SMEM_KV);
    constexpr uint32_t SMEM_SF_Q = math::constexpr_align(kN * 4u, 128u);
    auto smem_sf_kv = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_sf_q) + SMEM_SF_Q);
    constexpr uint32_t SMEM_SF_KV = math::constexpr_align(SPLIT_KV * 4u, 128u);
    auto smem_weights = reinterpret_cast<nv_bfloat16*>(reinterpret_cast<uint8_t*>(smem_sf_kv) + SMEM_SF_KV);
    constexpr uint32_t kNumWeightElems = math::constexpr_align(kNumHeads, 8u);
    constexpr uint32_t SMEM_WEIGHTS = BLOCK_Q * kNumWeightElems * 2;
    auto smem_kv_infos = reinterpret_cast<KVBlockInfo*>(
        reinterpret_cast<uint8_t*>(smem_weights) + math::constexpr_align(SMEM_WEIGHTS, 16u));
    auto smem_kv_header = reinterpret_cast<KVSplitHeader*>(smem_kv_infos + kNumKVBlocksPerSplit);
    auto barrier_ptr = reinterpret_cast<Barrier*>(
        reinterpret_cast<uint8_t*>(smem_kv_header) + math::constexpr_align<uint32_t>(sizeof(KVSplitHeader), 16u));
    auto full_q_barrier = barrier_ptr;   // Q+SF-Q+weights TMA

    const auto kv_splits = reinterpret_cast<const KVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(MetadataHeader));
    const auto header = reinterpret_cast<const MetadataHeader*>(metadata);
    const auto schedule_entries = reinterpret_cast<const ScheduleEntry*>(kv_splits + header->num_kv_splits);
    const uint32_t num_waves = header->num_waves;

    const bool is_tma_warp = warp_idx == 0;
    if (is_tma_warp and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncthreads();

    const PagedKVAccessor paged{fused_kv_cache, page_stride_bytes, page_kv, SPARSE_BLOCK_KV, kRowBytes};
    cudaGridDependencySynchronize();

    for (uint32_t wave_idx = 0; wave_idx < num_waves; ++ wave_idx) {
        const auto entry = schedule_entries[wave_idx * kNumSMs + blockIdx.x];
        if (entry.kv_split_begin == entry.kv_split_end)
            continue;
        const uint32_t q_token_base = entry.q_token_base;
        const uint32_t num_q_tokens = entry.num_q_tokens;

        // Load Q, SF-Q, weights for this q-block via TMA. Fresh barrier per wave (init -> fence -> sync -> wait(0)).
        // Order the previous wave's ldmatrix (generic proxy) reads of smem_q before this wave's TMA (async proxy) write.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (is_tma_warp and cute::elect_one_sync()) {
            full_q_barrier->init(1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();
        if (is_tma_warp and cute::elect_one_sync()) {
            tma::copy<kHeadDim, kN, kRowBytes>(&tensor_map_q, full_q_barrier, smem_q, 0, q_token_base * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_sf_q, full_q_barrier, smem_sf_q, 0, q_token_base);
            tma::copy<kNumWeightElems, BLOCK_Q, 0>(&tensor_map_weights, full_q_barrier, smem_weights, 0, q_token_base);
            full_q_barrier->arrive_and_expect_tx(SMEM_Q + kN * 4 + BLOCK_Q * kNumWeightElems * 2);
        }
        full_q_barrier->wait(0);

        // Per-lane Q weights (dense-kernel layout): weights[i][j] packs heads for the reduction
        float weights[BLOCK_Q][kNumHeads / 4];
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i)
            #pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                weights[i][j] = __bfloat162float(smem_weights[i * kNumWeightElems + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2]);

        for (uint32_t kv_split_idx = entry.kv_split_begin; kv_split_idx < entry.kv_split_end; ++ kv_split_idx) {
            const auto& split = kv_splits[kv_split_idx];
            const uint32_t packed = split.header.packed_num_kv_blocks;
            const uint32_t num_kv_blocks = KVSplitHeader::get_num_kv_blocks(packed);

            // Stage the split's metadata into smem (header + block infos)
            if (threadIdx.x == 0)
                *smem_kv_header = split.header;
            for (uint32_t b = threadIdx.x; b < num_kv_blocks; b += kNumMathThreads)
                smem_kv_infos[b] = split.kv_block_infos[b];
            __syncthreads();

            // Gather sparse KV + SF into the ldmatrix-swizzled tile. Each of the 256 threads copies parts of the blocks.
            // One warp handles a contiguous group of sparse blocks; within a block, lanes stride over 16B chunks.
            constexpr uint32_t kNumChunksPerRow = kRowBytes / 16;         // 4 FP8 / 4 FP4 (64B row -> 4 chunks? no)
            // A token row is kRowBytes; 16B chunks per row
            for (uint32_t blk = warp_idx; blk < num_kv_blocks; blk += kNumMathWarps) {
                uint32_t phys = smem_kv_infos[blk].physical_kv_block_idx;
                const uint8_t* src_kv; const uint32_t* src_sf;
                if constexpr (kIsPaged) {
                    paged.block_ptrs(phys, src_kv, src_sf);
                } else {
                    src_kv = kv + static_cast<uint64_t>(phys) * kRowBytes;   // phys is a token index for contiguous
                    src_sf = sf_kv + phys;
                }
                // Copy SPARSE_BLOCK_KV tokens of this block
                for (uint32_t tok = 0; tok < SPARSE_BLOCK_KV; ++ tok) {
                    const uint32_t dst_row = blk * SPARSE_BLOCK_KV + tok;
                    // SF: one int32 word per token
                    if (lane_idx == 0)
                        smem_sf_kv[dst_row] = src_sf[tok];
                    // KV row: kRowBytes bytes = (kRowBytes/16) chunks, swizzled like ldmatrix reads
                    const uint32_t nchunks = kRowBytes / 16;
                    for (uint32_t c = lane_idx; c < nchunks; c += 32) {
                        const uint32_t off = mma::sm120::get_swizzle_offset<kRowBytes>(dst_row, c);
                        ptx::cp_async_cg<256>(reinterpret_cast<const uint4*>(src_kv + tok * kRowBytes) + c,
                                             reinterpret_cast<uint4*>(smem_kv + off));
                    }
                }
            }
            cute::cp_async_fence();
            cute::cp_async_wait<0>();
            __syncthreads();
            // Compute. 256 threads = 8 warps cover 128 KV rows per pass, so loop over the `SPLIT_KV / 128` sub-tiles
            const uint32_t q0_slot_base = split.header.q0_slot_base;
            const uint32_t q1_slot_base = split.header.q1_slot_base;
            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            static constexpr uint32_t kNumSubTiles = SPLIT_KV / (kNumMathWarps * 16);
            DG_STATIC_ASSERT(SPLIT_KV % (kNumMathWarps * 16) == 0, "SPLIT_KV must tile into 128-row passes");
            #pragma unroll
            for (uint32_t sub = 0; sub < kNumSubTiles; ++ sub) {
                const uint32_t warp_offset = sub * (kNumMathWarps * 16) + warp_idx * 16;   // this warp's first KV row
                float accum[kNumAccum];
                mma::sm120::warp_mma_16xNxHd_block_scaled_ss<kN, kHeadDim, kIsFP4>(
                    mma::sm120::to_smem_addr(smem_kv), warp_offset,
                    mma::sm120::to_smem_addr(smem_q),
                    smem_sf_kv, SPLIT_KV, smem_sf_q, kN, accum);

                const uint32_t v_0_offset = lane_idx / 4 + 0;
                const uint32_t v_1_offset = lane_idx / 4 + 8;
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    if (i >= num_q_tokens) continue;
                    auto shifted = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](const uint32_t& j) { return fmaxf(shifted[j], 0) * weights[i][(j / 4) * 2 + (j & 1)]; };
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j)
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; ++ k) sum[k] += transform(j * 4 + k);
                    float v_0 = sum[0] + sum[1];
                    float v_1 = sum[2] + sum[3];
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const int off = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, off);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, off);
                    }
                    const uint32_t kv_row_0 = warp_offset + v_0_offset;
                    const uint32_t kv_row_1 = warp_offset + v_1_offset;
                    auto store = [&](const uint32_t kv_row, const float v) {
                        const uint32_t blk = kv_row / SPARSE_BLOCK_KV;
                        const uint32_t tok = kv_row % SPARSE_BLOCK_KV;
                        if (blk >= num_kv_blocks) return;
                        const uint32_t packed_slots = smem_kv_infos[blk].packed_slot_offsets;
                        const uint32_t slot = (packed_slots >> (i * kNumSparseSlotBits)) & kInvalidSparseSlot;
                        if (slot == kInvalidSparseSlot) return;
                        const uint32_t slot_base = i == 0 ? q0_slot_base : q1_slot_base;
                        const uint32_t col = (slot_base + slot) * SPARSE_BLOCK_KV + tok;
                        logits[(q_token_base + i) * static_cast<uint64_t>(logits_stride) + col] = static_cast<nv_bfloat16>(v);
                    };
                    if (lane_idx % 4 == 0) { store(kv_row_0, v_0); store(kv_row_1, v_1); }
                }
            }
            __syncthreads();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
