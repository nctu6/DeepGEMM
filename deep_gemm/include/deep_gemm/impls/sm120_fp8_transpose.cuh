#pragma once

#include <cstdint>

#include <cutlass/cutlass.h>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

// Byte matrix transpose for the SM120 FP8 GEMMs: an MN-major operand (`[K][ld_in]`, MN contiguous within a K row) into a
// compact K-major `[MN][K]` tensor, as the host path for MN-major operands the in-kernel transpose-on-load does not cover
// (and the "explicit transpose, then K-major kernel" baseline of the in-kernel path).
//
// A block moves a `BLOCK_K x BLOCK_MN` tile. Phase 1: a thread loads 4 K rows x 4 MN bytes (a warp reads 128 contiguous
// bytes per row), transposes the 4x4 byte block with `prmt` and stores it as four 32-bit words of the transposed shared
// tile `[BLOCK_MN][BLOCK_K + 4]`. Phase 2: threads read 16 consecutive K bytes of an MN row and store them with one 16-byte
// global store, so writes are in full 32-byte sectors
// Requires `mn % 16 == 0` (TMA-aligned MN-major rows), `k % 16 == 0` and 4-byte aligned `in` / `ld_in`
template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t BLOCK_K>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads)
void sm120_transpose_fp8(const uint8_t* in, uint8_t* out, const uint32_t mn, const uint32_t k,
                         const uint64_t ld_in, const uint64_t in_batch_stride) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    constexpr uint32_t kPitch = BLOCK_K + 4;
    DG_STATIC_ASSERT(BLOCK_MN == 32 * 4 and BLOCK_K % 16 == 0 and kNumThreads % 32 == 0, "Invalid tile");
    __shared__ __align__(16) uint8_t smem[BLOCK_MN * kPitch];

    const uint32_t mn_base = blockIdx.x * BLOCK_MN, k_base = blockIdx.y * BLOCK_K;
    in += static_cast<uint64_t>(blockIdx.z) * in_batch_stride;
    out += static_cast<uint64_t>(blockIdx.z) * mn * k;

    // Phase 1: global -> 4x4 byte transpose -> shared
    const uint32_t lane_idx = threadIdx.x % 32, warp_idx = threadIdx.x / 32;
    const uint32_t col = mn_base + lane_idx * 4;
    for (uint32_t kg = warp_idx; kg < BLOCK_K / 4; kg += kNumThreads / 32) {
        uint32_t w[4];
        #pragma unroll
        for (uint32_t r = 0; r < 4; ++ r) {
            const uint32_t row = k_base + kg * 4 + r;
            w[r] = (row < k and col < mn) ? __ldg(reinterpret_cast<const uint32_t*>(in + static_cast<uint64_t>(row) * ld_in + col)) : 0;
        }
        uint32_t lo01, lo23, hi01, hi23;
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(lo01) : "r"(w[0]), "r"(w[1]));
        asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(lo23) : "r"(w[2]), "r"(w[3]));
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(hi01) : "r"(w[0]), "r"(w[1]));
        asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(hi23) : "r"(w[2]), "r"(w[3]));
        uint32_t o[4];
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(o[0]) : "r"(lo01), "r"(lo23));
        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(o[1]) : "r"(lo01), "r"(lo23));
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(o[2]) : "r"(hi01), "r"(hi23));
        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(o[3]) : "r"(hi01), "r"(hi23));
        #pragma unroll
        for (uint32_t r = 0; r < 4; ++ r)
            *reinterpret_cast<uint32_t*>(smem + (lane_idx * 4 + r) * kPitch + kg * 4) = o[r];
    }
    __syncthreads();

    // Phase 2: shared -> global, 16 bytes of K per thread
    constexpr uint32_t kNumKChunks = BLOCK_K / 16;
    for (uint32_t i = threadIdx.x; i < BLOCK_MN * kNumKChunks; i += kNumThreads) {
        const uint32_t mn_local = i / kNumKChunks, k_local = (i % kNumKChunks) * 16;
        const uint32_t row = mn_base + mn_local, k_idx = k_base + k_local;
        if (row < mn and k_idx < k) {
            const auto* src = reinterpret_cast<const uint32_t*>(smem + mn_local * kPitch + k_local);
            *reinterpret_cast<uint4*>(out + static_cast<uint64_t>(row) * k + k_idx) = make_uint4(src[0], src[1], src[2], src[3]);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120f");
#endif
}

} // namespace deep_gemm
