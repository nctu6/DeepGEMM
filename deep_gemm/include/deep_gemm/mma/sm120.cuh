#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::mma::sm120 {

// SM120 (consumer/workstation Blackwell) has neither WGMMA nor UMMA/TMEM: tensor cores are
// driven by warp-level `mma.sync`. The FP8 shape is `m16n8k32`, with operands in registers.

/// Shared memory address in the `.shared` state space
CUTLASS_DEVICE uint32_t to_smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/// Load four 8x8 b16 matrices (i.e. four 8-row x 16-byte FP8 tiles)
CUTLASS_DEVICE void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const uint32_t& smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(smem_addr)
                 // NOTES: order against mbarrier arrivals releasing the shared memory being read
                 : "memory");
}

/// Byte offset of the 16-byte chunk `chunk` of row `row` inside a TMA 128B-swizzled tile whose rows are 128 bytes
/// NOTES: the tile base must be 1024-byte aligned
CUTLASS_DEVICE uint32_t get_swizzle_128b_offset(const uint32_t& row, const uint32_t& chunk) {
    return row * 128 + ((chunk ^ (row & 7)) << 4);
}

/// Same for a tile whose rows are `kRowBytes` = 64 or 128 bytes, TMA-swizzled with the matching mode
///  - 128B (`Swizzle<3, 4, 3>`): chunk ^= row % 8
///  - 64B (`Swizzle<2, 4, 3>`, address bits [7, 8] into [4, 5]): chunk ^= (row / 2) % 4
template <uint32_t kRowBytes>
CUTLASS_DEVICE uint32_t get_swizzle_offset(const uint32_t& row, const uint32_t& chunk) {
    DG_STATIC_ASSERT(kRowBytes == 64 or kRowBytes == 128, "Unsupported swizzle");
    if constexpr (kRowBytes == 128)
        return row * 128 + ((chunk ^ (row & 7)) << 4);
    else
        return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
}

/// `D = A * B + D` with E4M3 A (16x32, row) and B (32x8, col), FP32 accumulation
///  - A fragment: `a[0]`: (row g, k 4t..4t+3), `a[1]`: (row g+8, same k), `a[2]`/`a[3]`: k + 16
///  - B fragment: `b[0]`: (k 4t..4t+3, col g), `b[1]`: k + 16
///  - D fragment: `d[0..1]`: (row g, col 2t..2t+1), `d[2..3]`: row g+8
///  where `g = lane / 4` and `t = lane % 4`
struct FP8MMA {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 32;

    CUTLASS_DEVICE static void mma(const uint32_t* a, const uint32_t* b, float* d) {
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]));
    }
};

/// Block-scaled MMAs: the tensor core applies UE8M0 scales (`2^(byte - 127)`), so no CUDA-core promotion is needed.
/// Scale-factor lane mapping (verified on hardware, `scripts/sm120_{blockscale,fp4}_probe.cu`), with `g = lane / 4`
/// and `t = lane % 4`, all selectors 0:
///  - SFA: lane `t == 0` supplies row `g`, lane `t == 1` supplies row `g + 8` (lanes `t >= 2` are ignored)
///  - SFB: lane `t == 0` supplies column `g` (the other lanes are ignored)
/// Fragments are identical to the unscaled `FP8MMA` (A: 4 regs, B: 2 regs, D: 4 floats).
///
/// FP8 (E4M3) `m16n8k32`, `scale_vec::1X`: one scale byte per 32 K, passed in the low byte of `sfa`/`sfb`
struct FP8BlockScaledMMA {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 32;

    CUTLASS_DEVICE static void mma(const uint32_t* a, const uint32_t* b, float* d,
                                   const uint32_t& sfa, const uint32_t& sfb) {
        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                     "%10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]),
                       "r"(sfa), "r"(sfb));
    }
};

/// FP4 (E2M1, two values per byte, low nibble first) `m16n8k64`, `scale_vec::2X`: the low 16 bits of `sfa`/`sfb` hold
/// two scale bytes, byte 0 for K 0-31 and byte 1 for K 32-63. The A/B fragments are loaded with the same `ldmatrix`
/// pattern as FP8 (a 64-element FP4 K step is 32 bytes, like a 32-element FP8 K step)
struct FP4BlockScaledMMA {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 64;

    CUTLASS_DEVICE static void mma(const uint32_t* a, const uint32_t* b, float* d,
                                   const uint32_t& sfa, const uint32_t& sfb) {
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                     "%10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]),
                       "r"(sfa), "r"(sfb));
    }
};

/// NVFP4: E2M1 operands with UE4M3 scales over 16-element K groups (`scale_vec::4X`). A lane's 32-bit scale operand holds
/// the 4 UE4M3 bytes of its row/column for the MMA's 64 K: byte `i` scales K `16i .. 16i + 15`. The fragments and the
/// scale lane map are those of the MXFP4 (`scale_vec::2X`) MMA
struct NVFP4BlockScaledMMA {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 64;

    CUTLASS_DEVICE static void mma(const uint32_t* a, const uint32_t* b, float* d,
                                   const uint32_t& sfa, const uint32_t& sfb) {
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                     "%10, {0, 0}, %11, {0, 0};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]),
                       "r"(sfa), "r"(sfb));
    }
};

/// TF32 `m16n8k8`, FP32 accumulation, no scales. A step consumes 8 TF32 values of each operand row (ptxas rounds the
/// FP32 register inputs to TF32). A: 4 regs (rows `g`, `g + 8` x columns `t`, `t + 4`), B: 2 regs (column `g`, rows `t`,
/// `t + 4`), D: 4 floats, `g = lane / 4`, `t = lane % 4`
struct TF32MMA {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 8;

    CUTLASS_DEVICE static void mma(const uint32_t* a, const uint32_t* b, float* d) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]));
    }
};

/// BF16 `m16n8k16`, FP32 accumulation, no scales. A step consumes 32 bytes of each operand row, like the FP8 `m16n8k32`,
/// and the fragments are the same `ldmatrix` patterns (A: 4 regs, B: 2 regs, D: 4 floats)
struct BF16MMA {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 16;

    CUTLASS_DEVICE static void mma(const uint32_t* a, const uint32_t* b, float* d, const uint32_t&, const uint32_t&) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]));
    }
};

/// One warp computes `accum[16, N] = A[16, kHeadDim] @ B[N, kHeadDim]^T` with FP8 operands in TMA-swizzled shared memory
/// (both K-major; rows are `kHeadDim` bytes, swizzled with the matching mode: 128B for 128, 64B for 64)
///  - `a_base`/`b_base`: shared memory addresses of the 1024B-aligned tiles, `a_row`: the first of the 16 A rows
///  - `accum` follows the SM90 WGMMA `m64nN` per-warp layout: `accum[j * 4 + {0, 1, 2, 3}]` holds
///    `(row g, col 8j + 2t + {0, 1})` and `(row g + 8, col 8j + 2t + {0, 1})`, with `g = lane / 4`, `t = lane % 4`
/// An `ldmatrix.x4` reads a 16x16 FP8 tile (two 16B chunks of K per row); `kHeadDim / 32` of them cover the head dim
template <uint32_t N, uint32_t kHeadDim = 128>
CUTLASS_DEVICE void warp_mma_16xNxHd_ss(const uint32_t& a_base, const uint32_t& a_row,
                                         const uint32_t& b_base, float* accum) {
    DG_STATIC_ASSERT(N % 16 == 0, "Invalid N");
    DG_STATIC_ASSERT(kHeadDim == 128 or kHeadDim == 64, "Unsupported head dim");
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t a_ld_row = a_row + ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
    const uint32_t a_ld_chunk = lane_idx >> 4;
    const uint32_t b_ld_row = (lane_idx >> 4) * 8 + (lane_idx & 7);
    const uint32_t b_ld_chunk = (lane_idx >> 3) & 1;

    #pragma unroll
    for (uint32_t i = 0; i < N / 8 * 4; ++ i)
        accum[i] = 0;

    #pragma unroll
    for (uint32_t k = 0; k < kHeadDim / FP8MMA::K; ++ k) {
        uint32_t a_frag[4];
        ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                    a_base + get_swizzle_offset<kHeadDim>(a_ld_row, k * 2 + a_ld_chunk));
        #pragma unroll
        for (uint32_t j = 0; j < N / 16; ++ j) {
            uint32_t b_frag[2][2];
            ldmatrix_x4(b_frag[0][0], b_frag[0][1], b_frag[1][0], b_frag[1][1],
                        b_base + get_swizzle_offset<kHeadDim>(b_ld_row + j * 16, k * 2 + b_ld_chunk));
            FP8MMA::mma(a_frag, b_frag[0], accum + (j * 2 + 0) * 4);
            FP8MMA::mma(a_frag, b_frag[1], accum + (j * 2 + 1) * 4);
        }
    }
}

/// One warp computes `accum[16, N] = A[16, kHeadDim] @ B[N, kHeadDim]^T` with block-scaled FP8 (E4M3, 32-element UE8M0)
/// or FP4 (E2M1, 32-element UE8M0 / `kIsFP4`) operands, the tensor core applying the scales. Same accumulator layout as
/// `warp_mma_16xNxHd_ss`. `a_sf` / `b_sf` are the per-row packed UE8M0 words in shared memory (one int32 per 4 K groups,
/// i.e. per 128 elements), laid out `[kHeadDim / 128 words][row]`; `a_sf_stride` / `b_sf_stride` are the row counts of
/// those arrays. The SF lane map matches the 1D1D GEMM: a lane supplies the A scale of row `g` (`t == 0`) or `g + 8`
/// (`t == 1`), and the B scale of column `g` (`t == 0`), with `g = lane / 4`, `t = lane % 4`
template <uint32_t N, uint32_t kHeadDim, bool kIsFP4>
CUTLASS_DEVICE void warp_mma_16xNxHd_block_scaled_ss(const uint32_t& a_base, const uint32_t& a_row,
                                                     const uint32_t& b_base,
                                                     const uint32_t* a_sf, const uint32_t a_sf_stride,
                                                     const uint32_t* b_sf, const uint32_t b_sf_stride,
                                                     float* accum) {
    DG_STATIC_ASSERT(N % 16 == 0, "Invalid N");
    DG_STATIC_ASSERT(kHeadDim == 128, "Block-scaled MQA head dim must be 128");
    using MMA = cute::conditional_t<kIsFP4, FP4BlockScaledMMA, FP8BlockScaledMMA>;
    // One K step (one MMA) is `MMA::K` elements, which is 32 bytes of a packed row for both: FP8 32 elems x 1 byte,
    // FP4 64 elems / 2 per byte. So an `ldmatrix.x4` (4 regs = 32 bytes = 2 swizzle chunks) feeds exactly one step
    constexpr uint32_t kRowBytes = kHeadDim / (kIsFP4 ? 2 : 1);  // 128 FP8 / 64 FP4 bytes per row
    constexpr uint32_t kNumSteps = kRowBytes / 32;              // 4 FP8 / 2 FP4
    const uint32_t lane_idx = threadIdx.x % 32, g = lane_idx / 4, t = lane_idx % 4;

    // `ldmatrix` source rows/chunks of this lane (16-byte chunks within a `kRowBytes`-byte swizzled row)
    const uint32_t a_ld_row = a_row + ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
    const uint32_t a_ld_chunk = lane_idx >> 4;
    const uint32_t b_ld_row = (lane_idx >> 4) * 8 + (lane_idx & 7);
    const uint32_t b_ld_chunk = (lane_idx >> 3) & 1;

    // This lane's A scale rows (g, g+8) and, for each 8-wide B column tile `c`, the B scale of column `c*8 + g`
    // (the MMA's SFB lane map takes column `g` from lane `t == 0`), across all `kHeadDim / 128` words
    static constexpr uint32_t kNumWords = kHeadDim / 128;
    static constexpr uint32_t kNumBTiles = N / 8;
    uint32_t a_sf_words[2][kNumWords], b_sf_words[kNumBTiles][kNumWords];
    #pragma unroll
    for (uint32_t w = 0; w < kNumWords; ++ w) {
        a_sf_words[0][w] = a_sf[w * a_sf_stride + a_row + g];
        a_sf_words[1][w] = a_sf[w * a_sf_stride + a_row + g + 8];
        #pragma unroll
        for (uint32_t c = 0; c < kNumBTiles; ++ c)
            b_sf_words[c][w] = b_sf[w * b_sf_stride + c * 8 + g];
    }

    #pragma unroll
    for (uint32_t i = 0; i < N / 8 * 4; ++ i)
        accum[i] = 0;

    #pragma unroll
    for (uint32_t k = 0; k < kNumSteps; ++ k) {
        // The UE8M0 scale byte of step `k`: its K covers `MMA::K` elements, i.e. `MMA::K / 32` scale groups. A word
        // holds 4 groups (128 elements). FP8: group `k`, byte `k % 4` of word `k / 4`. FP4: a step is 64 elements = 2
        // groups, the 16-bit pair `k % 2` of word `k / 2`
        const uint32_t word = kIsFP4 ? k / 2 : k / 4;
        const auto scale = [&](const uint32_t& words) -> uint32_t {
            if constexpr (kIsFP4)
                return (words >> (16 * (k % 2))) & 0xffffu;
            else
                return (words >> (8 * (k % 4))) & 0xffu;
        };
        const uint32_t sfa0 = scale(a_sf_words[0][word]), sfa1 = scale(a_sf_words[1][word]);
        uint32_t a_frag[4];
        ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                    a_base + get_swizzle_offset<kRowBytes>(a_ld_row, k * 2 + a_ld_chunk));
        #pragma unroll
        for (uint32_t j = 0; j < N / 16; ++ j) {
            uint32_t b_frag[2][2];
            ldmatrix_x4(b_frag[0][0], b_frag[0][1], b_frag[1][0], b_frag[1][1],
                        b_base + get_swizzle_offset<kRowBytes>(b_ld_row + j * 16, k * 2 + b_ld_chunk));
            // The MMA reads SFA of its two M-rows via the lane map (t==0 -> row g, t==1 -> row g+8); SFB is the
            // scale of this 8-col tile's column g (b_frag[0] -> tile 2j, b_frag[1] -> tile 2j+1)
            MMA::mma(a_frag, b_frag[0], accum + (j * 2 + 0) * 4, ((t & 1) ? sfa1 : sfa0), scale(b_sf_words[j * 2 + 0][word]));
            MMA::mma(a_frag, b_frag[1], accum + (j * 2 + 1) * 4, ((t & 1) ? sfa1 : sfa0), scale(b_sf_words[j * 2 + 1][word]));
        }
    }
}

} // namespace deep_gemm::mma::sm120
