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

/// One warp computes `accum[16, N] = A[16, 128] @ B[N, 128]^T` with FP8 operands in TMA 128B-swizzled shared memory
/// (both K-major with 128-byte rows, i.e. `kHeadDim = 128`)
///  - `a_base`/`b_base`: shared memory addresses of the 1024B-aligned tiles, `a_row`: the first of the 16 A rows
///  - `accum` follows the SM90 WGMMA `m64nN` per-warp layout: `accum[j * 4 + {0, 1, 2, 3}]` holds
///    `(row g, col 8j + 2t + {0, 1})` and `(row g + 8, col 8j + 2t + {0, 1})`, with `g = lane / 4`, `t = lane % 4`
template <uint32_t N>
CUTLASS_DEVICE void warp_mma_16xNx128_ss(const uint32_t& a_base, const uint32_t& a_row,
                                          const uint32_t& b_base, float* accum) {
    DG_STATIC_ASSERT(N % 16 == 0, "Invalid N");
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t a_ld_row = a_row + ((lane_idx >> 3) & 1) * 8 + (lane_idx & 7);
    const uint32_t a_ld_chunk = lane_idx >> 4;
    const uint32_t b_ld_row = (lane_idx >> 4) * 8 + (lane_idx & 7);
    const uint32_t b_ld_chunk = (lane_idx >> 3) & 1;

    #pragma unroll
    for (uint32_t i = 0; i < N / 8 * 4; ++ i)
        accum[i] = 0;

    #pragma unroll
    for (uint32_t k = 0; k < 128 / FP8MMA::K; ++ k) {
        uint32_t a_frag[4];
        ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                    a_base + get_swizzle_128b_offset(a_ld_row, k * 2 + a_ld_chunk));
        #pragma unroll
        for (uint32_t j = 0; j < N / 16; ++ j) {
            uint32_t b_frag[2][2];
            ldmatrix_x4(b_frag[0][0], b_frag[0][1], b_frag[1][0], b_frag[1][1],
                        b_base + get_swizzle_128b_offset(b_ld_row + j * 16, k * 2 + b_ld_chunk));
            FP8MMA::mma(a_frag, b_frag[0], accum + (j * 2 + 0) * 4);
            FP8MMA::mma(a_frag, b_frag[1], accum + (j * 2 + 1) * 4);
        }
    }
}

} // namespace deep_gemm::mma::sm120
