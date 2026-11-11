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
// The compute (block-scaled mma.sync + ReLU x weight head reduction) is the SM120 dense MQA kernel's; the work is the
// SM100 sparse metadata's KV splits (a run of sparse blocks of one Q block, with per-block slot offsets). Unlike SM100
// there is no TMEM: gathered sparse KV lands in an ldmatrix-swizzled smem tile and the MX scale words are staged in smem,
// read by the block-scaled `mma.sync`.
//
// Schedule. One CTA per SM walks a contiguous range of the metadata's KV splits, `total / grid` of them (the splits of one
// Q block are adjacent), so the load is even to one split at any pool size and batch; the metadata's own per-SM schedule
// entries (SM100's, capped at 8 splits per entry) are not read. A Q block's Q tile, scales and weights are TMA'd into one of
// two buffers when its first split is gathered, so the next Q block loads behind the current one, and a warp keeps its
// share of the Q fragments in registers for the whole block.
//
// Pipeline. No producer warps: the 16 warps form 8 row groups of two (one warp per query), and a row group gathers the
// sparse blocks of the 16-row tiles it scores itself, so the only waits between warps are two named barriers of 64 threads
// per split inside a pair; row groups run independently ahead of or behind each other (a barrier of the whole CTA would let
// the slowest warp set the pace of every split). The gather of split i + 1 is issued while split i computes, one block
// per tile between the MMAs and the epilogue so that its instructions fill the tensor pipe's latency (kNumStages = 2 KV
// buffers; the FP8 tile is too big for two, which then run one after the other). A sparse block is `SPARSE_BLOCK_KV`
// adjacent tokens, so a warp copies one block (512 bytes of FP4 rows) with one `cp.async` per lane, and the scale words with
// one 4-byte copy per token. The Q buffers are shared: warp 0 loads a Q block once all warps have read the block two
// before it. BLOCK_Q = 2, head dim 128.

namespace deep_gemm {

using namespace layout::sparse_mqa_logits;

// Paged sparse KV: resolve a physical sparse-block index to its token base pointer + SF pointer in the fused cache
struct PagedKVAccessor {
    const uint8_t* fused_kv_cache;
    uint32_t page_stride_bytes;
    uint32_t page_kv;
    uint32_t log2_blocks_per_page;   // sparse blocks per page, a power of two
    uint32_t qk_bytes_per_token;     // head dim bytes (64 FP4 / 128 FP8)

    CUTLASS_DEVICE void block_ptrs(const uint32_t physical_kv_block_idx, const uint32_t sparse_block_kv,
                                   const uint8_t*& kv, const uint32_t*& sf) const {
        const uint32_t page_idx = physical_kv_block_idx >> log2_blocks_per_page;
        const uint32_t block_in_page = physical_kv_block_idx & ((1u << log2_blocks_per_page) - 1);
        const auto page = fused_kv_cache + static_cast<uint64_t>(page_idx) * page_stride_bytes;
        const uint32_t token_in_page = block_in_page * sparse_block_kv;
        kv = page + token_in_page * qk_bytes_per_token;
        sf = reinterpret_cast<const uint32_t*>(page + page_kv * qk_bytes_per_token) + token_in_page;
    }
};

template <uint32_t kNumHeads, uint32_t SPARSE_BLOCK_KV, uint32_t SPLIT_KV, uint32_t kNumStages,
          bool kIsFP4, bool kIsPaged>
CUTLASS_GLOBAL __launch_bounds__(512, 1)
void sm120_sparse_mqa_logits(const uint32_t logits_stride, nv_bfloat16* logits,
                             const uint8_t* metadata,
                             const uint8_t* kv, const uint32_t* sf_kv, const uint32_t num_kv_tokens,
                             const uint8_t* fused_kv_cache, const uint32_t page_stride_bytes, const uint32_t page_kv,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t BLOCK_Q = kBlockQ;
    constexpr uint32_t kHeadDim = 128;
    constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
    constexpr uint32_t kRowBytes = kHeadDim / kPackFactor;         // 64 FP4 / 128 FP8
    constexpr uint32_t kNumWarps = 16;
    constexpr uint32_t kNumRowGroups = 8;                          // a row group of 16 KV rows per warp: 128 of SPLIT_KV
    constexpr uint32_t kN = BLOCK_Q * kNumHeads;                   // Q rows of a Q block
    // With 16 or more heads a warp scores one of the two queries (N = heads), so the two halves of the CTA run the same
    // rows in parallel with half the accumulators; fewer heads keep both queries in one warp of the first half
    constexpr bool kSplitQ = kNumHeads % 16 == 0;
    constexpr uint32_t kNumQPerWarp = kSplitQ ? 1 : BLOCK_Q;
    constexpr uint32_t kWarpN = kNumQPerWarp * kNumHeads;          // MMA N of a warp
    constexpr uint32_t kNumAccum = kWarpN / 2;                     // per-lane accumulators (dense-kernel layout)
    constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    constexpr uint32_t kNumChunksPerRow = kRowBytes / 16;          // 16-byte copies per token row
    constexpr uint32_t kNumChunksPerBlock = SPARSE_BLOCK_KV * kNumChunksPerRow;
    // The sparse blocks overlapping the 16-row tiles of a row group, one entry per block of each tile; the two warps of the
    // group split the entries
    constexpr uint32_t kNumSubTiles = SPLIT_KV / (kNumRowGroups * 16);
    constexpr uint32_t kBlocksPerTile = SPARSE_BLOCK_KV >= 16 ? 1 : 16 / SPARSE_BLOCK_KV;
    constexpr uint32_t kNumEntries = kNumSubTiles * kBlocksPerTile;
    constexpr uint32_t kNumEntriesPerWarp = math::constexpr_ceil_div(kNumEntries, 2u);
    DG_STATIC_ASSERT(SPLIT_KV % (kNumRowGroups * 16) == 0, "SPLIT_KV must tile into 128-row passes");
    DG_STATIC_ASSERT(kNumStages == 1 or kNumStages == 2, "One or two KV buffers");
    DG_STATIC_ASSERT(SPARSE_BLOCK_KV == 8 or SPARSE_BLOCK_KV == 16, "Invalid sparse block size");
    DG_STATIC_ASSERT(kNumEntriesPerWarp <= 32, "A lane resolves at most one block per entry slot");
    constexpr Sm120SparseSmem Smem(kNumHeads, SPARSE_BLOCK_KV, SPLIT_KV, kNumStages, kRowBytes);
    constexpr uint32_t kNumWeightElems = Smem.num_weight_elems;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    const auto smem_q = [&](const uint32_t buf) { return smem_buffer + Smem.q_offset + buf * Smem.q_bytes; };
    const auto smem_kv = [&](const uint32_t stage) { return smem_buffer + Smem.kv_offset + stage * Smem.kv_bytes; };
    const auto smem_sf_q = [&](const uint32_t buf) {
        return reinterpret_cast<uint32_t*>(smem_buffer + Smem.sf_q_offset + buf * Smem.sf_q_bytes); };
    const auto smem_sf_kv = [&](const uint32_t stage) {
        return reinterpret_cast<uint32_t*>(smem_buffer + Smem.sf_kv_offset + stage * Smem.sf_kv_bytes); };
    const auto smem_weights = [&](const uint32_t buf) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + Smem.weight_offset + buf * Smem.weight_bytes); };
    const auto smem_kv_infos = [&](const uint32_t stage) {
        return reinterpret_cast<KVBlockInfo*>(smem_buffer + Smem.info_offset + stage * Smem.info_bytes); };
    auto full_q_barriers = reinterpret_cast<Barrier*>(smem_buffer + Smem.barrier_offset);
    auto empty_q_barriers = full_q_barriers + 2;   // 16 arrivals: every warp has read the Q buffer

    const auto header = reinterpret_cast<const MetadataHeader*>(metadata);
    const auto kv_splits = reinterpret_cast<const KVSplit<kNumKVBlocksPerSplit>*>(metadata + sizeof(MetadataHeader));

    const bool is_tma_warp = warp_idx == 0;
    if (is_tma_warp and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        full_q_barriers[0].init(1);
        full_q_barriers[1].init(1);
        empty_q_barriers[0].init(kNumWarps);
        empty_q_barriers[1].init(kNumWarps);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    const PagedKVAccessor paged{fused_kv_cache, page_stride_bytes, page_kv,
                                kIsPaged ? static_cast<uint32_t>(__ffs(page_kv / SPARSE_BLOCK_KV) - 1) : 0u, kRowBytes};
    cudaGridDependencySynchronize();

    // This CTA's even share of the splits
    const uint64_t num_splits = header->num_kv_splits;
    const uint32_t split_begin = static_cast<uint32_t>((num_splits * blockIdx.x + gridDim.x - 1) / gridDim.x);
    const uint32_t split_end = static_cast<uint32_t>((num_splits * (blockIdx.x + 1) + gridDim.x - 1) / gridDim.x);
    if (split_begin >= split_end)
        return;

    // Q blocks are numbered in the order this CTA meets them: block j uses buffer `j % 2` and its barrier's phase `j / 2`
    uint32_t num_issued_q = 0, num_waited_q = 0;
    uint32_t issued_q_token = ~0u, current_q_token = ~0u, q_buf = 0;
    const uint32_t row_group = warp_idx % kNumRowGroups;
    const uint32_t pair_idx = warp_idx / kNumRowGroups;   // the group's warp that scores query `pair_idx`
    const auto sync_pair = [&]() { cutlass::arch::NamedBarrier::sync(64, 1 + row_group); };

    // Load the Q tile, its scales and its weights for a Q block (warp 0)
    const auto issue_q = [&](const uint32_t q_token_base, const uint32_t block_idx) {
        const uint32_t buf = block_idx % 2;
        // Every warp has read the buffer's previous Q block (two blocks ago) before it is overwritten
        if (block_idx >= 2)
            empty_q_barriers[buf].wait((block_idx / 2 - 1) % 2);
        if (cute::elect_one_sync()) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            tma::copy<kRowBytes, kN, kRowBytes>(&tensor_map_q, full_q_barriers + buf, smem_q(buf), 0, q_token_base * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_sf_q, full_q_barriers + buf, smem_sf_q(buf), 0, q_token_base);
            tma::copy<kNumWeightElems, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers + buf, smem_weights(buf), 0, q_token_base);
            full_q_barriers[buf].arrive_and_expect_tx(Smem.q_bytes + kN * 4 + BLOCK_Q * kNumWeightElems * 2);
        }
    };

    // The sparse blocks of a split that this warp gathers: entry `e` is block `(e / kBlocksPerTile) * 128 / SPARSE_BLOCK_KV +
    // row_group * 16 / SPARSE_BLOCK_KV + e % kBlocksPerTile` (the blocks of the row group's tiles), and the group's two
    // warps take every other entry
    const auto entry_block = [&](const uint32_t entry) {
        return (entry / kBlocksPerTile) * (kNumRowGroups * 16 / SPARSE_BLOCK_KV) +
               row_group * 16 / SPARSE_BLOCK_KV + entry % kBlocksPerTile;
    };

    // Read the header of a split and stage its block infos in shared memory, and start the Q load of its Q block when it is a
    // new one. Lane `t` reads the info of the warp's t-th entry; returns that block's physical index
    const auto stage_split = [&](const uint32_t split_idx, const uint32_t stage, KVSplitHeader& split_header) {
        const auto& split = kv_splits[split_idx];
        split_header = split.header;
        if (split_header.q_token_base != issued_q_token) {
            issued_q_token = split_header.q_token_base;
            if (is_tma_warp)
                issue_q(issued_q_token, num_issued_q);
            ++ num_issued_q;
        }
        const uint32_t num_kv_blocks = KVSplitHeader::get_num_kv_blocks(split_header.packed_num_kv_blocks);
        const uint32_t own_entry = 2 * lane_idx + pair_idx;
        uint32_t own_physical_block = 0;
        if (lane_idx < kNumEntriesPerWarp and own_entry < kNumEntries) {
            const uint32_t own_block = entry_block(own_entry);
            if (own_block < num_kv_blocks) {
                const KVBlockInfo info = split.kv_block_infos[own_block];
                own_physical_block = info.physical_kv_block_idx;
                smem_kv_infos(stage)[own_block] = info;
            }
        }
        return own_physical_block;
    };

    // Issue the `cp.async` gather of one block of a staged split: the 32 lanes copy its rows (contiguous in the cache) and
    // scale words
    const auto gather_entry = [&](const uint32_t t, const uint32_t stage, const uint32_t num_kv_blocks,
                                  const uint32_t own_physical_block) {
        const uint32_t entry = 2 * t + pair_idx;
        if (entry >= kNumEntries)
            return;
        const uint32_t blk = entry_block(entry);
        if (blk >= num_kv_blocks)
            return;
        const uint32_t phys = __shfl_sync(0xffffffffu, own_physical_block, t);
        const uint8_t* src_kv; const uint32_t* src_sf;
        if constexpr (kIsPaged) {
            paged.block_ptrs(phys, SPARSE_BLOCK_KV, src_kv, src_sf);
        } else {
            src_kv = kv + static_cast<uint64_t>(phys) * kRowBytes;   // phys is a token index for contiguous
            src_sf = sf_kv + phys;
        }
        uint8_t* kv_tile = smem_kv(stage);
        uint32_t* sf_tile = smem_sf_kv(stage);
        const uint32_t dst_row = blk * SPARSE_BLOCK_KV;
        for (uint32_t c = lane_idx; c < kNumChunksPerBlock; c += 32) {
            const uint32_t row = c / kNumChunksPerRow, chunk = c % kNumChunksPerRow;
            auto dst = reinterpret_cast<uint4*>(kv_tile + mma::sm120::get_swizzle_offset<kRowBytes>(dst_row + row, chunk));
            auto src = reinterpret_cast<const uint4*>(src_kv) + c;
            if constexpr (kIsPaged) {
                ptx::cp_async_cg<256>(src, dst);
            } else {
                // The last block of the tensor may run past its end
                const bool valid = phys + row < num_kv_tokens;
                ptx::cp_async_cg_zfill<256>(valid ? src : reinterpret_cast<const uint4*>(kv), dst, valid ? 16 : 0);
            }
        }
        if (lane_idx < SPARSE_BLOCK_KV) {
            if constexpr (kIsPaged) {
                ptx::cp_async_ca_zfill(src_sf + lane_idx, sf_tile + dst_row + lane_idx, 4);
            } else {
                const bool valid = phys + lane_idx < num_kv_tokens;
                ptx::cp_async_ca_zfill(valid ? src_sf + lane_idx : sf_kv, sf_tile + dst_row + lane_idx, valid ? 4 : 0);
            }
        }
    };

    // Per-lane Q weights (dense-kernel layout): weights[i][j] packs heads for the reduction
    float weights[kNumQPerWarp][kNumHeads / 4];
    mma::sm120::BlockScaledBFragments<kWarpN, kHeadDim, kIsFP4> q_frags;   // this warp's queries, for the Q block
    const uint32_t q_first = kSplitQ ? pair_idx : 0;   // the query of this warp (its first)
    const bool is_math_warp = kSplitQ or warp_idx < kNumRowGroups;

    KVSplitHeader current_header, next_header;
    {
        const uint32_t own_physical_block = stage_split(split_begin, 0, current_header);
        const uint32_t num_kv_blocks = KVSplitHeader::get_num_kv_blocks(current_header.packed_num_kv_blocks);
        #pragma unroll
        for (uint32_t t = 0; t < kNumEntriesPerWarp; ++ t)
            gather_entry(t, 0, num_kv_blocks, own_physical_block);
        cute::cp_async_fence();
    }

    for (uint32_t split_idx = split_begin; split_idx < split_end; ++ split_idx) {
        const uint32_t stage = kNumStages == 1 ? 0 : (split_idx - split_begin) % 2;
        // Stage the next split (its buffer was freed by the barrier that ended the last iteration); with one buffer it is
        // this split's turn instead
        const bool has_next = kNumStages == 2 and split_idx + 1 < split_end;
        uint32_t next_physical_block = 0, next_num_kv_blocks = 0;
        if constexpr (kNumStages == 2) {
            if (has_next) {
                next_physical_block = stage_split(split_idx + 1, stage ^ 1, next_header);
                next_num_kv_blocks = KVSplitHeader::get_num_kv_blocks(next_header.packed_num_kv_blocks);
            }
        } else if (split_idx != split_begin) {
            const uint32_t own_physical_block = stage_split(split_idx, 0, current_header);
            const uint32_t num_kv_blocks = KVSplitHeader::get_num_kv_blocks(current_header.packed_num_kv_blocks);
            #pragma unroll
            for (uint32_t t = 0; t < kNumEntriesPerWarp; ++ t)
                gather_entry(t, 0, num_kv_blocks, own_physical_block);
            cute::cp_async_fence();
        }
        cute::cp_async_wait<0>();

        if (current_header.q_token_base != current_q_token) {
            current_q_token = current_header.q_token_base;
            q_buf = num_waited_q % 2;
            full_q_barriers[q_buf].wait((num_waited_q / 2) % 2);
            ++ num_waited_q;
            const auto weights_smem = smem_weights(q_buf);
            #pragma unroll
            for (uint32_t i = 0; i < kNumQPerWarp; ++ i)
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = __bfloat162float(weights_smem[(q_first + i) * kNumWeightElems + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2]);
            if (is_math_warp and q_first < (current_header.q1_slot_base == ~0u ? 1u : BLOCK_Q))
                mma::sm120::load_block_scaled_b<kWarpN, kHeadDim, kIsFP4>(
                    mma::sm120::to_smem_addr(smem_q(q_buf)) + q_first * kNumHeads * kRowBytes,
                    smem_sf_q(q_buf) + q_first * kNumHeads, kN, q_frags);
            // This warp has read the Q buffer
            __syncwarp();
            if (lane_idx == 0)
                empty_q_barriers[q_buf].arrive();
        }
        // Publish the row group's gathered rows, scale words and block infos to its two warps
        sync_pair();

        const uint32_t q_token_base = current_header.q_token_base;
        const uint32_t num_q_tokens = current_header.q1_slot_base == ~0u ? 1 : BLOCK_Q;
        const uint32_t num_kv_blocks = KVSplitHeader::get_num_kv_blocks(current_header.packed_num_kv_blocks);
        const uint32_t q0_slot_base = current_header.q0_slot_base;
        const uint32_t q1_slot_base = current_header.q1_slot_base;
        const auto infos = smem_kv_infos(stage);
        const auto sf_kv_tile = smem_sf_kv(stage);

        // Compute: the 8 row groups cover 128 KV rows per pass, so loop over the `SPLIT_KV / 128` sub-tiles; a single-token
        // Q block leaves the second query's warps without scoring work
        static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
        const bool scores = is_math_warp and q_first < num_q_tokens;
        #pragma unroll
        for (uint32_t sub = 0; sub < kNumSubTiles; ++ sub) {
            const uint32_t warp_offset = sub * (kNumRowGroups * 16) + row_group * 16;   // this warp's first KV row
            // The last split of a Q block may end before this tile
            const bool active = scores and warp_offset < num_kv_blocks * SPARSE_BLOCK_KV;
            float accum[kNumAccum];
            if (active)
                mma::sm120::warp_mma_16xNxHd_block_scaled_rs<kWarpN, kHeadDim, kIsFP4>(
                    mma::sm120::to_smem_addr(smem_kv(stage)), warp_offset, sf_kv_tile, SPLIT_KV, q_frags, accum);
            // The next split's blocks, spread over the tiles so that the copy instructions interleave with the MMAs
            if (has_next) {
                #pragma unroll
                for (uint32_t t = 0; t < kNumEntriesPerWarp; ++ t)
                    if (t * kNumSubTiles / kNumEntriesPerWarp == sub)
                        gather_entry(t, stage ^ 1, next_num_kv_blocks, next_physical_block);
            }
            if (!active)
                continue;

            #pragma unroll
            for (uint32_t qq = 0; qq < kNumQPerWarp; ++ qq) {
                const uint32_t i = q_first + qq;
                if (i >= num_q_tokens) continue;
                auto shifted = accum + qq * kNumAccumPerReduce;
                const auto transform = [&](const uint32_t& j) { return fmaxf(shifted[j], 0) * weights[qq][(j / 4) * 2 + (j & 1)]; };
                float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                #pragma unroll
                for (uint32_t j = 1; j < kNumHeads / 8; ++ j)
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++ k) sum[k] += transform(j * 4 + k);
                // Reduce the four lanes of a row group (each holds two columns of rows g and g + 8) so that lane 0
                // ends with row g and lane 1 with row g + 8: lanes swap the row they give away, then add pairs
                const float v_0 = sum[0] + sum[1];
                const float v_1 = sum[2] + sum[3];
                const bool odd = (lane_idx & 1) != 0;
                float v = (odd ? v_1 : v_0) + __shfl_xor_sync(0xffffffffu, odd ? v_0 : v_1, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                const uint32_t kv_row = warp_offset + lane_idx / 4 + (odd ? 8 : 0);
                if (lane_idx % 4 < 2) {
                    const uint32_t blk = kv_row / SPARSE_BLOCK_KV;
                    const uint32_t tok = kv_row % SPARSE_BLOCK_KV;
                    if (blk < num_kv_blocks) {
                        const uint32_t packed_slots = infos[blk].packed_slot_offsets;
                        const uint32_t slot = (packed_slots >> (i * kNumSparseSlotBits)) & kInvalidSparseSlot;
                        if (slot != kInvalidSparseSlot) {
                            const uint32_t slot_base = i == 0 ? q0_slot_base : q1_slot_base;
                            const uint32_t col = (slot_base + slot) * SPARSE_BLOCK_KV + tok;
                            logits[(q_token_base + i) * static_cast<uint64_t>(logits_stride) + col] = static_cast<nv_bfloat16>(v);
                        }
                    }
                }
            }
        }
        if (has_next)
            cute::cp_async_fence();
        // Both warps of the group are done with this buffer before the next gather overwrites it
        sync_pair();
        if (has_next)
            current_header = next_header;
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
