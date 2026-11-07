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

// SM120 block-scaled GEMM with packed scale factors ("1D1D"): FP4 (E2M1) with UE8M0 (MXFP4, 32-element groups) or UE4M3
// (NVFP4, 16-element groups) scales, FP8 (E4M3) with UE8M0 scales, and BF16 without scales.
//
// Same host contract as `sm100_fp8_fp4_gemm_1d1d_impl`: SFA/SFB are int32 tensors holding 4 consecutive-K scale bytes
// per word (K granularity `kGranK` of 32 or 128 for UE8M0, 16 for UE4M3), laid out `[(G,) MN, ceil(K / (kGranK * 4))]`,
// MN-major and
// TMA-aligned. Unlike SM100 there is no TMEM: the scales are TMA-loaded into shared memory with each stage (on the
// same barrier as A/B, so their latency is hidden by the pipeline like the operands') and read into registers by the
// math warps:
//  - The tensor core applies the scales (`mma.sync ... block_scale`), so unlike the FP32 "1D2D" kernel there is no
//    second accumulator and no per-K-block promotion on CUDA cores
//  - FP4 (`kGranK == 32`): a TMA tile row is 128 bytes = 256 FP4 elements, consumed as 4 `m16n8k64` MMAs, each taking
//    the 2 scale bytes of its 64 K
//  - NVFP4 (`kGranK == 16`): the same MMAs with `scale_vec::4X`, each taking one whole word (the 4 UE4M3 bytes of its
//    64 K)
//  - FP8: a tile row is 128 bytes = 128 elements, consumed as 4 `m16n8k32` MMAs. With `kGranK == 32` MMA `j` takes
//    byte `j` of the word, with `kGranK == 128` all MMAs of a K block share one byte
//  - A lane supplies the SFA scale of row `g` (lane `t == 0`) or `g + 8` (lane `t == 1`) and the SFB scale of column `g`
//    (lane `t == 0`), where `g = lane / 4`, `t = lane % 4`
// Narrow tiles (`BLOCK_N == 32`, for thin-N decode GEMMs that otherwise launch too few CTAs to saturate DRAM) split
// the 4 MMA steps of every K block over two K-warps per output sub-tile, as the 128-thread math group must stay a whole
// warpgroup for `setmaxnreg`; the K-warps' accumulators are summed through shared memory before the epilogue.
// MN-major operands (`kMajorA` / `kMajorB`): BF16 is read with the 16-bit `ldmatrix.trans`, FP8 with the native 8-bit
// transposing `ldmatrix.m16n16.trans.b8`; both give the same fragments as a K-major operand. FP4 stays K-major (there is no
// sub-byte transposing load)
// `GemmType::Batched` runs `kNumGroups` independent GEMMs `D[g] = A[g] @ B[g]^T` in one launch: A/B are loaded through 3D tensor
// maps with the batch as the outer dimension (so batches may be strided views, e.g. the heads of a `[b, h, d]` tensor), the
// scales are stacked along their outer dimension like grouped GEMMs', and D is offset by `d_stride_g` per batch
// The TMA pipeline, scheduler, warp tiling, stream-K and epilogue are shared with `sm120_fp8_gemm_1d2d_impl`: with
// `kUseStreamK` (normal GEMMs whose tiles fill the GPU poorly, e.g. decoding with a long K), the
// `num_tiles x num_k_blocks` iteration space is split evenly over the CTAs, and a tile covered by several CTAs is
// reduced in split order through an FP32 `workspace` by the last one to finish (tracked by self-resetting `counters`)
template <bool kIsFP4, bool kIsBF16, uint32_t kGranK,
          cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs, GemmType kGemmType,
          bool kUseStreamK, bool kWithAccumulation,
          typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm120_fp8_fp4_gemm_1d1d_impl(int* grouped_layout, cd_dtype_t* d, uint32_t ld_d, uint64_t d_stride_g, float alpha,
                             float* workspace, int* counters,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    static constexpr bool kIsNVFP4 = kIsFP4 and kGranK == 16;
    using MMA = cute::conditional_t<kIsBF16, mma::sm120::BF16MMA,
                cute::conditional_t<kIsNVFP4, mma::sm120::NVFP4BlockScaledMMA,
                cute::conditional_t<kIsFP4, mma::sm120::FP4BlockScaledMMA, mma::sm120::FP8BlockScaledMMA>>>;
    // TMA element type of A/B: packed FP4 counts two elements per byte; BF16 rows hold 64 elements of 2 bytes
    using ab_dtype_t = cute::conditional_t<kIsBF16, cutlass::bfloat16_t,
                       cute::conditional_t<kIsFP4, cutlass::float_e2m1_t, uint8_t>>;
    DG_STATIC_ASSERT(not (kIsFP4 and kIsBF16), "Invalid operand kind");
    // MN-major operands are stored as TMA-swizzled atoms of `kElemsPerBlock` K rows by 128 MN bytes (64 BF16 or 128 FP8
    // elements; FP8 tiles of 64 MN use 64-byte atoms), and read with `ldmatrix.trans` (BF16, 16-bit) or its 8-bit form (FP8)
    static constexpr bool kIsMNMajorA = kMajorA == cute::UMMA::Major::MN;
    static constexpr bool kIsMNMajorB = kMajorB == cute::UMMA::Major::MN;
    static constexpr uint32_t kElemBytes = kIsBF16 ? 2 : 1;
    static constexpr uint32_t kMNAtomBytesA = BLOCK_M * kElemBytes >= 128 ? 128 : 64;
    static constexpr uint32_t kMNAtomBytesB = BLOCK_N * kElemBytes >= 128 ? 128 : 64;
    DG_STATIC_ASSERT(not kIsFP4 or (not kIsMNMajorA and not kIsMNMajorB), "FP4 operands must be K-major");
    DG_STATIC_ASSERT(not kIsMNMajorA or (BLOCK_M % 64 == 0 and (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched)),
                     "MN-major A needs whole 64-row atoms and no groups");
    static constexpr bool kIsBatchedMM = kGemmType == GemmType::Batched;
    DG_STATIC_ASSERT(not kIsMNMajorB or BLOCK_N % 64 == 0, "MN-major B needs whole 64-column atoms");
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Checks. `BLOCK_K` is the byte width of a tile row (one swizzle atom): 128 bytes (128 FP8 or 256 FP4 elements), or
    // for FP4 64 bytes (128 elements), which halves the stage size so that twice as many stages are in flight
    DG_STATIC_ASSERT(BLOCK_K == 128 or (BLOCK_K == 64 and kIsFP4), "A K block is one 128-byte (or, for FP4, 64-byte) swizzle atom");
    DG_STATIC_ASSERT(kSwizzleAMode == (kIsMNMajorA ? kMNAtomBytesA : BLOCK_K) and
                     kSwizzleBMode == (kIsMNMajorB ? kMNAtomBytesB : BLOCK_K),
                     "Swizzling must match the K block (one atom per row), or the MN atom of an MN-major operand");
    // D = alpha * (A @ B^T) (+ D when accumulating; the host copies a distinct C into D first), BF16 or FP32
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t> or cute::is_same_v<cd_dtype_t, float>,
                     "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::Batched or kGemmType == GemmType::MGroupedContiguous or
                     kGemmType == GemmType::MGroupedContiguousWithPsumLayout or
                     kGemmType == GemmType::MGroupedMasked, "Unsupported GEMM type");
    DG_STATIC_ASSERT(not kUseStreamK or kGemmType == GemmType::Normal, "Stream-K is only for normal GEMMs");
    DG_STATIC_ASSERT(kIsBF16 or kGranK == 32 or (kIsFP4 and kGranK == 16) or (not kIsFP4 and kGranK == 128),
                     "FP4 takes 32- (UE8M0) or 16-element (UE4M3) scales; FP8 32 or 128 (UE8M0)");
    static constexpr uint32_t kElemsPerBlock = kIsFP4 ? BLOCK_K * 2 : (kIsBF16 ? BLOCK_K / 2 : BLOCK_K);
    static constexpr uint32_t kNumKSteps = BLOCK_K / 32;   // 32 bytes of A/B per MMA, for both FP8 and FP4
    // Packed scale words covering one K block (one scale byte per `kGranK` elements, 4 per word); BF16 has no scales,
    // and an FP8 K block (128 elements) takes one word for both granularities (a quarter of a 128-granularity word)
    static constexpr uint32_t kNumSFWords = kIsBF16 ? 0 : (kIsFP4 ? BLOCK_K * 2 / kGranK / 4 : 1);
    DG_STATIC_ASSERT(kIsBF16 or kNumSFWords >= 1, "A K block must cover whole scale words");

    // Warp tiling
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    // One 64-row warp per 64 rows, except 8-warp 64-row tiles (64x128 / 64x256), which use two 32-row warps so A is not
    // re-read by every warp; mirrors `SM120ArchSpec::get_num_warps_m`
    static constexpr uint32_t kNumWarpsM = (BLOCK_M == 64 and kNumMathWarps == 8) ? 2 : (BLOCK_M >= 64 ? BLOCK_M / 64 : 1);
    static constexpr uint32_t kNumKWarps = BLOCK_N == 32 ? 2 : 1;
    static constexpr uint32_t kNumWarpsN = kNumMathWarps / (kNumWarpsM * kNumKWarps);
    static constexpr uint32_t WARP_M = BLOCK_M / kNumWarpsM;
    static constexpr uint32_t WARP_N = BLOCK_N / kNumWarpsN;
    static constexpr uint32_t kNumMTiles = WARP_M / 16;
    static constexpr uint32_t kNumNTiles = WARP_N / 8;
    static constexpr uint32_t kNumAccum = kNumMTiles * kNumNTiles * 4;
    DG_STATIC_ASSERT(kNumWarpsM * kNumWarpsN * kNumKWarps == kNumMathWarps, "Invalid warp grid");
    DG_STATIC_ASSERT(kNumKWarps == 1 or (BLOCK_M < 64 and kNumKSteps % kNumKWarps == 0 and (not kIsFP4 or kNumKWarps == 2)),
                     "K-warps are for narrow, short tiles");
    // Scratch for summing the K-warps' accumulators: `[kNumAccum][threads of one K-warp group]` floats
    static constexpr uint32_t kNumReduceThreads = kNumMathThreads / kNumKWarps;
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
    // A K-warp covers `kNumLocalKSteps` contiguous MMA steps of each block, and keeps only the scale words of those steps
    // in registers: NVFP4 one word per step, MXFP4 one per 2 steps, FP8 one per block. Indexing them by the local step
    // keeps every register-array index a compile-time constant (a runtime index spills the array to local memory)
    static constexpr uint32_t kNumLocalKSteps = kNumKSteps / kNumKWarps;
    static constexpr uint32_t kNumLocalSFWords = kIsBF16 ? 0 : (kIsNVFP4 ? kNumLocalKSteps :
                                                 (kIsFP4 ? math::constexpr_ceil_div(kNumLocalKSteps, 2u) : 1));
    DG_STATIC_ASSERT(not kIsFP4 or kIsNVFP4 or kNumLocalKSteps == 1 or kNumLocalKSteps % 2 == 0,
                     "MXFP4 K-warps must own whole scale words");
    // Scale-word arrays need a non-zero extent even when BF16 has none
    static constexpr uint32_t kNumSFRegs = kNumLocalSFWords > 0 ? kNumLocalSFWords : 1;

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
        if constexpr (not kIsBF16) {
            cute::prefetch_tma_descriptor(&tensor_map_sfa);
            cute::prefetch_tma_descriptor(&tensor_map_sfb);
        }
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
    constexpr uint32_t SMEM_REDUCE_OFFSET = SMEM_SFB_OFFSET + kNumStages * SMEM_SFB_SIZE_PER_STAGE;
    constexpr uint32_t SMEM_REDUCE_SIZE = kNumKWarps > 1 ? math::constexpr_align(kNumAccum * kNumReduceThreads * 4u, 128u) : 0;
    auto smem_reduce = reinterpret_cast<float*>(smem_buffer + SMEM_REDUCE_OFFSET);
    constexpr uint32_t SMEM_BARRIER_OFFSET = SMEM_REDUCE_OFFSET + SMEM_REDUCE_SIZE;
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
    constexpr uint32_t kNumTMARegisters = kNumMathThreads == 128 ? 56 : 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // PDL: the weights (B, SFB) do not depend on the previous kernel. Before waiting for it, the producer issues the
    // first `kNumStages` K blocks of B / SFB of its first tile straight into their pipeline stages (see the producer).
    // A CTA resident while the previous kernel finishes thus starts with a full pipeline of weights, and only the
    // activations (A, SFA) wait for the previous kernel. (Also prefetching the next stages into L2 was neutral to
    // slightly slower: it is early DRAM traffic competing with the previous kernel's tail.) Only normal
    // data-parallel GEMMs know their first tile without reading `grouped_layout`, which the previous kernel may write
    static constexpr bool kPreissueWeights = kGemmType == GemmType::Normal and not kUseStreamK;
    const uint32_t num_preissued_k_blocks = kPreissueWeights and blockIdx.x < scheduler.num_blocks ?
                                            cute::min(num_k_blocks, kNumStages) : 0;

    // Task iteration: a task is a tile, or with stream-K, a K range of a tile
    uint32_t tile_idx = 0, k_block_begin = 0, k_block_end = num_k_blocks;
    uint64_t sk_iter = 0, sk_iter_end = 0, sk_num_iters = 0;
    // NOTES: called by each warp role after its register reconfiguration. Computed before it, the 64-bit range is live
    // across `setmaxnreg.dealloc` and ptxas spills it to local memory (seen on BF16 16x64 stream-K tiles)
    const auto init_stream_k_range = [&]() {
        if constexpr (kUseStreamK) {
            sk_num_iters = static_cast<uint64_t>(scheduler.num_blocks) * num_k_blocks;
            sk_iter = sk_num_iters * blockIdx.x / gridDim.x;
            sk_iter_end = sk_num_iters * (blockIdx.x + 1) / gridDim.x;
        }
    };
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
        init_stream_k_range();

        if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
            constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
            // A 128-element FP8 K block uses a quarter of a 128-granularity scale word, so that word is re-fetched
            const auto get_sf_block_idx = [](const uint32_t& k_block_idx) {
                return (kIsFP4 or kGranK == 32) ? k_block_idx : k_block_idx / 4;
            };
            constexpr uint32_t kNumBTMABytes = SMEM_B_SIZE_PER_STAGE + kNumSFWords * BLOCK_N * 4;
            constexpr uint32_t kNumATMABytes = SMEM_A_SIZE_PER_STAGE + kNumSFWords * BLOCK_M * 4;
            if constexpr (kPreissueWeights) {
                if (num_preissued_k_blocks > 0) {
                    uint32_t first_m_block_idx, first_n_block_idx;
                    scheduler.get_swizzled_block_idx(blockIdx.x, first_m_block_idx, first_n_block_idx);
                    const uint32_t n_idx = first_n_block_idx * BLOCK_N;
                    for (uint32_t k_block_idx = 0; k_block_idx < num_preissued_k_blocks; ++ k_block_idx) {
                        // The stages start empty. The transaction is expected without an arrival, so the B bytes landing
                        // cannot complete the phase: the single arrival comes with A, after the wait
                        auto& full_barrier = *full_barriers[k_block_idx];
                        full_barrier.expect_transaction(kNumBTMABytes);
                        if constexpr (kIsMNMajorB) {
                            tma::copy<BLOCK_N, kElemsPerBlock, kSwizzleBMode, ab_dtype_t>(&tensor_map_b, &full_barrier,
                                     reinterpret_cast<ab_dtype_t*>(smem_b[k_block_idx]), n_idx, k_block_idx * kElemsPerBlock);
                        } else {
                            tma::copy<kElemsPerBlock, BLOCK_N, kSwizzleBMode, ab_dtype_t>(&tensor_map_b, &full_barrier,
                                     reinterpret_cast<ab_dtype_t*>(smem_b[k_block_idx]), k_block_idx * kElemsPerBlock, n_idx);
                        }
                        if constexpr (not kIsBF16) {
                            tma::copy<BLOCK_N, kNumSFRegs, 0, uint32_t>(&tensor_map_sfb, &full_barrier, smem_sfb[k_block_idx],
                                     n_idx, get_sf_block_idx(k_block_idx) * kNumSFWords);
                        }
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
                    // NOTES: for FP4 the TMA K coordinate counts elements (16U4 data type), and the packed
                    // dtype makes one 256-element box a single 128B swizzle atom
                    const uint32_t k_idx = k_block_idx * kElemsPerBlock;
                    const uint32_t sf_block_idx = get_sf_block_idx(k_block_idx);
                    const uint32_t m_idx = scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx);
                    const uint32_t batch_idx = kIsBatchedMM ? scheduler.current_group_idx : 0;
                    if constexpr (kIsMNMajorA) {
                        tma::copy<BLOCK_M, kElemsPerBlock, kSwizzleAMode, ab_dtype_t, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                                 reinterpret_cast<ab_dtype_t*>(smem_a[stage_idx]), m_idx, k_idx, 1, batch_idx);
                    } else {
                        tma::copy<kElemsPerBlock, BLOCK_M, kSwizzleAMode, ab_dtype_t, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                                 reinterpret_cast<ab_dtype_t*>(smem_a[stage_idx]), k_idx, m_idx, 1, batch_idx);
                    }
                    // Scales: inner coordinate is the row/column, outer is the packed word (plus the group offset)
                    if constexpr (not kIsBF16) {
                        tma::copy<BLOCK_M, kNumSFRegs, 0, uint32_t>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx],
                                 m_block_idx * BLOCK_M,
                                 scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(num_sf_words, kNumSFWords, sf_block_idx));
                    }
                    // B / SFB of the first stages of the first task were issued before the wait
                    if (is_first_task and k_block_idx < num_preissued_k_blocks) {
                        full_barrier.arrive_and_expect_tx(kNumATMABytes);
                    } else {
                        // A grouped B is `[G, N, K]` (K-major) or `[G, K, N]` (MN-major): the group offset goes to the outer
                        // dimension, N or K respectively (as in the SM90 BF16 kernel)
                        if constexpr (kIsMNMajorB) {
                            tma::copy<BLOCK_N, kElemsPerBlock, kSwizzleBMode, ab_dtype_t, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                                     reinterpret_cast<ab_dtype_t*>(smem_b[stage_idx]),
                                     scheduler.get_global_idx<false>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                                     scheduler.template get_global_idx<true, sched::IndexType::K>(shape_k, kElemsPerBlock, k_block_idx, m_block_idx),
                                     1, batch_idx);
                        } else {
                            tma::copy<kElemsPerBlock, BLOCK_N, kSwizzleBMode, ab_dtype_t, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                                     reinterpret_cast<ab_dtype_t*>(smem_b[stage_idx]), k_idx,
                                     scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                                     1, batch_idx);
                        }
                        if constexpr (not kIsBF16) {
                            tma::copy<BLOCK_N, kNumSFRegs, 0, uint32_t>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx],
                                     n_block_idx * BLOCK_N,
                                     scheduler.template get_global_idx<true, sched::IndexType::SF_K>(num_sf_words, kNumSFWords, sf_block_idx, m_block_idx));
                        }
                        full_barrier.arrive_and_expect_tx(kNumATMABytes + kNumBTMABytes);
                    }
                }
                is_first_task = false;
            }
            // PDL: let the next kernel launch (it still waits for this grid to complete and flush before reading anything
            // it produced), once its early weight loads no longer compete with ours for DRAM:
            //  - short K (all K blocks fit in the pipeline): every load is issued now, so trigger here
            //  - otherwise the math warps trigger when their mainloop ends (below)
            // Measured (FP4 64x128, graph replay): triggering at kernel start was slower at every K; after the last load
            // vs. at the end of the mainloop: K = 512 4.40 vs 4.78 us, K = 7168 21.0 vs 19.5 us
            if (num_k_blocks <= kNumStages)
                cudaTriggerProgrammaticLaunchCompletion();
        }
    } else {
        // Math warps
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        init_stream_k_range();

        // Wait for the previous kernel before anything that reads its results or writes memory it may still read
        cudaGridDependencySynchronize();

        const uint32_t warp_k_idx = warp_idx / (kNumWarpsM * kNumWarpsN);
        const uint32_t warp_mn_idx = warp_idx % (kNumWarpsM * kNumWarpsN);
        const uint32_t warp_m_idx = warp_mn_idx / kNumWarpsN;
        const uint32_t warp_n_idx = warp_mn_idx % kNumWarpsN;
        // Only the first K-warp of a sub-tile stores results (it holds the sum after the K-warp reduction)
        const bool is_storing_warp = kNumKWarps == 1 or warp_k_idx == 0;
        const uint32_t group_row = lane_idx / 4, group_col = lane_idx % 4;

        // `ldmatrix` source rows/chunks of this lane (same pattern as the FP32 kernel; an MMA consumes 2 chunks of 16B)
        const uint32_t a_ld_row = warp_m_idx * WARP_M + ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
        const uint32_t a_ld_chunk = lane_idx >> 4;
        const uint32_t b_ld_row = warp_n_idx * WARP_N + (lane_idx >> 4) * 8 + (lane_idx & 7);
        const uint32_t b_ld_chunk = (lane_idx >> 3) & 1;
        // MN-major (`ldmatrix.trans`): a lane points at K row `*_t_k_row` (plus the step's 16) of the 8 MN elements starting
        // at `*_t_mn`; matrix `lane / 8` of A is (M +8 if bit 0, K +8 if bit 1), of B (K +8 if bit 0, N +8 if bit 1), the
        // register order of the K-major loads
        const uint32_t a_t_mn = warp_m_idx * WARP_M + ((lane_idx >> 3) & 1) * 8;
        const uint32_t a_t_k_row = ((lane_idx >> 4) & 1) * 8 + (lane_idx & 7);
        const uint32_t b_t_mn = warp_n_idx * WARP_N + (lane_idx >> 4) * 8;
        const uint32_t b_t_k_row = ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
        // MN-major FP8 (`ldmatrix.trans.b8`, lane `l` points at K row `l` of the step): the warp's first MN byte
        const uint32_t a_b8_mn = warp_m_idx * WARP_M, b_b8_mn = warp_n_idx * WARP_N;
        // Byte offset of the 8 MN elements at `mn` (a multiple of 8) of K row `k_row` in an MN-major stage
        const auto get_mn_major_offset = [](const uint32_t& k_row, const uint32_t& mn) -> uint32_t {
            return (mn / 64) * (kElemsPerBlock * 128) + mma::sm120::get_swizzle_offset<128>(k_row, (mn % 64) / 8);
        };

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

                    // The packed scale words of this K-warp's steps, starting at the block's word of its first step
                    const uint32_t first_k = warp_k_idx * kNumLocalKSteps;
                    const uint32_t first_word = kIsNVFP4 ? first_k : (kIsFP4 ? first_k / 2 : 0);
                    uint32_t sfa_words[kNumMTiles][kNumSFRegs] = {}, sfb_words[kNumNTiles][kNumSFRegs] = {};
                    #pragma unroll
                    for (uint32_t w = 0; w < kNumLocalSFWords; ++ w) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i)
                            sfa_words[i][w] = smem_sfa[stage_idx][(first_word + w) * BLOCK_M + sfa_rows[i]];
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j)
                            sfb_words[j][w] = smem_sfb[stage_idx][(first_word + w) * BLOCK_N + sfb_cols[j]];
                    }

                    const uint32_t a_base = mma::sm120::to_smem_addr(smem_a[stage_idx]);
                    const uint32_t b_base = mma::sm120::to_smem_addr(smem_b[stage_idx]);

                    // The whole stage (every K step's A/B fragments and its scale words) is copied into registers, and the stage
                    // is released *before* the MMAs, so the producer can refill all `kNumStages` stages while this one computes.
                    // Releasing after the MMAs leaves only `kNumStages - 1` in flight; on decode / mid-M shapes the kernel is
                    // latency-bound, and that cost 32% per K block (64x128 FP4: 0.93 vs 0.70 us/block against FlashInfer's b12x,
                    // which stages the same way). Every op is `asm volatile`, so this source order is the issue order
                    uint32_t a_frag[kNumLocalKSteps][kNumMTiles][4], b_frag[kNumLocalKSteps][kNumNTiles][2];
                    #pragma unroll
                    for (uint32_t k_local = 0; k_local < kNumLocalKSteps; ++ k_local) {
                        // This K-warp's share of the block's MMA steps (contiguous, so FP4 steps pair up with scale words)
                        const uint32_t k = warp_k_idx * kNumLocalKSteps + k_local;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                            if constexpr (kIsMNMajorA and not kIsBF16) {
                                mma::sm120::ldmatrix_b8_x2_trans(a_frag[k_local][i][0], a_frag[k_local][i][1], a_frag[k_local][i][2], a_frag[k_local][i][3],
                                                                 a_base + mma::sm120::get_mn_major_offset<kMNAtomBytesA, kElemsPerBlock>(
                                                                     k * 32 + lane_idx, a_b8_mn + i * 16));
                            } else if constexpr (kIsMNMajorA) {
                                mma::sm120::ldmatrix_x4_trans(a_frag[k_local][i][0], a_frag[k_local][i][1], a_frag[k_local][i][2], a_frag[k_local][i][3],
                                                              a_base + get_mn_major_offset(k * 16 + a_t_k_row, a_t_mn + i * 16));
                            } else {
                                mma::sm120::ldmatrix_x4(a_frag[k_local][i][0], a_frag[k_local][i][1], a_frag[k_local][i][2], a_frag[k_local][i][3],
                                                        a_base + mma::sm120::get_swizzle_offset<BLOCK_K>(a_ld_row + i * 16, k * 2 + a_ld_chunk));
                            }
                        }
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles / 2; ++ j) {
                            if constexpr (kIsMNMajorB and not kIsBF16) {
                                // The two 8-column tiles of 16 columns: {b0, b1} = {r0, r2} and {r1, r3}
                                mma::sm120::ldmatrix_b8_x2_trans(b_frag[k_local][j * 2][0], b_frag[k_local][j * 2 + 1][0],
                                                                 b_frag[k_local][j * 2][1], b_frag[k_local][j * 2 + 1][1],
                                                                 b_base + mma::sm120::get_mn_major_offset<kMNAtomBytesB, kElemsPerBlock>(
                                                                     k * 32 + lane_idx, b_b8_mn + j * 16));
                            } else if constexpr (kIsMNMajorB) {
                                mma::sm120::ldmatrix_x4_trans(b_frag[k_local][j * 2][0], b_frag[k_local][j * 2][1],
                                                              b_frag[k_local][j * 2 + 1][0], b_frag[k_local][j * 2 + 1][1],
                                                              b_base + get_mn_major_offset(k * 16 + b_t_k_row, b_t_mn + j * 16));
                            } else {
                                mma::sm120::ldmatrix_x4(b_frag[k_local][j * 2][0], b_frag[k_local][j * 2][1],
                                                        b_frag[k_local][j * 2 + 1][0], b_frag[k_local][j * 2 + 1][1],
                                                        b_base + mma::sm120::get_swizzle_offset<BLOCK_K>(b_ld_row + j * 16, k * 2 + b_ld_chunk));
                            }
                        }
                    }
                    // Release the stage: the fence inside orders the `ldmatrix` reads above before the next TMA write
                    empty_barrier_arrive();

                    #pragma unroll
                    for (uint32_t k_local = 0; k_local < kNumLocalKSteps; ++ k_local) {
                        const uint32_t k = warp_k_idx * kNumLocalKSteps + k_local;
                        // The scale operand of this K step, in the low bits of a register (array indices are local)
                        const auto get_scale = [&](const uint32_t (&words)[kNumSFRegs]) -> uint32_t {
                            if constexpr (kIsBF16)
                                return 0u;
                            else if constexpr (kIsNVFP4)
                                return words[k_local];   // NVFP4: a step's 64 elements are exactly one word
                            else if constexpr (kIsFP4)   // MXFP4: a step takes scale bytes `2k` and `2k + 1` of the block
                                return (words[kNumLocalKSteps >= 2 ? k_local / 2 : 0] >> (16 * (k % 2))) & 0xffffu;
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
                                MMA::mma(a_frag[k_local][i], b_frag[k_local][j], accum + (i * kNumNTiles + j) * 4, scale_a[i], scale_b[j]);
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (uint32_t k_block_idx = k_block_begin; k_block_idx < k_block_end; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // Sum the K-warps' partial accumulators into the storing warp
            if constexpr (kNumKWarps > 1) {
                const uint32_t reduce_tid = warp_mn_idx * 32 + lane_idx;
                if (not is_storing_warp) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        smem_reduce[i * kNumReduceThreads + reduce_tid] = accum[i];
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 2);
                if (is_storing_warp) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAccum; ++ i)
                        accum[i] += smem_reduce[i * kNumReduceThreads + reduce_tid];
                }
                // The scratch is rewritten by the next task
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 2);
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
                            if (row >= shape_m or not is_storing_warp)
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
                                if (row >= shape_m or not is_storing_warp)
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

            // PDL (long K, see the producer): the mainloop of this CTA is done
            if (num_k_blocks > kNumStages)
                cudaTriggerProgrammaticLaunchCompletion();

            // Write back results directly from registers: D = alpha * accum (+ D)
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumNTiles % 2 == 0, "Paired stores need an even number of N tiles");
            const uint32_t m_global_base = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
            // (a batched D starts `d_stride_g` elements after the previous batch's)
            const uint64_t d_batch_offset = kIsBatchedMM ? scheduler.current_group_idx * d_stride_g : 0;
            const uint32_t m_local_base = m_block_idx * BLOCK_M;
            const bool is_odd_lane = group_col & 1;
            // FP32 accumulation: read every old D value of this thread before the first store. Interleaved with the
            // stores (the compiler cannot hoist a load over a store to the same buffer), each load waits out a DRAM round
            // trip behind the previous one, which dominated tiles with a single K block (`bhd,bhr->hdr` at b = 128). The
            // loads also ask L2 for the whole 128-byte line (`ld_global_l2_prefetch`): this thread reads 32 bytes of a row at a
            // time, and fetching sector by sector halved the DRAM bandwidth (282 -> 197 us for h = 8, b = 128, r = 4096, d = 1024)
            constexpr bool kLoadOldFirst = kWithAccumulation and cute::is_same_v<cd_dtype_t, float>;
            float2 old_d[kLoadOldFirst ? kNumMTiles * 2 * kNumNTiles : 1];
            if constexpr (kLoadOldFirst) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                    #pragma unroll
                    for (uint32_t half = 0; half < 2; ++ half) {
                        const uint32_t row = warp_m_idx * WARP_M + i * 16 + group_row + half * 8;
                        const bool is_row_valid = is_storing_warp and m_local_base + row < shape_m;
                        auto d_row = reinterpret_cast<const float*>(d) + d_batch_offset + static_cast<uint64_t>(m_global_base + row) * ld_d;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                            const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + j * 8 + group_col * 2;
                            old_d[(i * 2 + half) * kNumNTiles + j] = (is_row_valid and col < shape_n) ?
                                ptx::ld_global_l2_prefetch<128>(reinterpret_cast<const float2*>(d_row + col)) : make_float2(0, 0);
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                #pragma unroll
                for (uint32_t half = 0; half < 2; ++ half) {
                    const uint32_t row = warp_m_idx * WARP_M + i * 16 + group_row + half * 8;
                    const bool is_row_valid = is_storing_warp and m_local_base + row < shape_m;
                    const uint64_t row_offset = d_batch_offset + static_cast<uint64_t>(m_global_base + row) * ld_d;
                    if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                        // FP32: each lane stores its 2 columns of every N tile (the 4 lanes of a row write 32 bytes)
                        auto d_row = reinterpret_cast<float*>(d) + row_offset;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                            const uint32_t idx = (i * kNumNTiles + j) * 4 + half * 2;
                            const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + j * 8 + group_col * 2;
                            if (not is_row_valid or col >= shape_n)
                                continue;
                            float2 value = make_float2(alpha * accum[idx], alpha * accum[idx + 1]);
                            if constexpr (kLoadOldFirst) {
                                const float2 old = old_d[(i * 2 + half) * kNumNTiles + j];
                                value.x += old.x, value.y += old.y;
                            }
                            *reinterpret_cast<float2*>(d_row + col) = value;
                        }
                    } else {
                        // BF16: lanes `t` and `t ^ 1` exchange halves so that each stores 4 consecutive columns (8
                        // bytes), and the 4 lanes of a row cover 16 consecutive columns, i.e. one full 32-byte sector
                        // per row and store. Even lanes keep tile `j` (and get the odd lane's half), odd lanes tile `j + 1`
                        auto d_row = reinterpret_cast<__nv_bfloat16*>(d) + row_offset;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; j += 2) {
                            const uint32_t idx_lo = (i * kNumNTiles + j) * 4 + half * 2;
                            const uint32_t idx_hi = idx_lo + 4;
                            const uint32_t col = n_block_idx * BLOCK_N + warp_n_idx * WARP_N + (j + is_odd_lane) * 8 + (group_col & 2) * 2;
                            const bool is_valid = is_row_valid and col < shape_n;
                            uint2 value;
                            if constexpr (kWithAccumulation) {
                                // Exchange FP32 values, so the sum with D is rounded to BF16 once
                                const float2 lo = make_float2(alpha * accum[idx_lo], alpha * accum[idx_lo + 1]);
                                const float2 hi = make_float2(alpha * accum[idx_hi], alpha * accum[idx_hi + 1]);
                                const float2 send = is_odd_lane ? lo : hi;
                                const float2 recv = make_float2(__shfl_xor_sync(0xffffffff, send.x, 1),
                                                                __shfl_xor_sync(0xffffffff, send.y, 1));
                                float2 first = is_odd_lane ? recv : lo, second = is_odd_lane ? hi : recv;
                                if (is_valid) {
                                    const uint2 old = *reinterpret_cast<const uint2*>(d_row + col);
                                    const float2 old_first = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old.x));
                                    const float2 old_second = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old.y));
                                    first.x += old_first.x, first.y += old_first.y;
                                    second.x += old_second.x, second.y += old_second.y;
                                }
                                const auto first_bf16 = __float22bfloat162_rn(first);
                                const auto second_bf16 = __float22bfloat162_rn(second);
                                value = make_uint2(*reinterpret_cast<const uint32_t*>(&first_bf16),
                                                   *reinterpret_cast<const uint32_t*>(&second_bf16));
                            } else {
                                const auto lo = __float22bfloat162_rn({alpha * accum[idx_lo], alpha * accum[idx_lo + 1]});
                                const auto hi = __float22bfloat162_rn({alpha * accum[idx_hi], alpha * accum[idx_hi + 1]});
                                const uint32_t lo_u32 = *reinterpret_cast<const uint32_t*>(&lo);
                                const uint32_t hi_u32 = *reinterpret_cast<const uint32_t*>(&hi);
                                const uint32_t recv = __shfl_xor_sync(0xffffffff, is_odd_lane ? lo_u32 : hi_u32, 1);
                                value = is_odd_lane ? make_uint2(recv, hi_u32) : make_uint2(lo_u32, recv);
                            }
                            if (is_valid)
                                *reinterpret_cast<uint2*>(d_row + col) = value;
                        }
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
