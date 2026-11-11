#pragma once
#include <cstdint>

namespace deep_gemm {

// Epilogues of the SM120 1D1D GEMM that the mega MoE (`fp8_fp4_mega_moe`) fuses into its grouped GEMMs, so that the
// activation does not run as a kernel of its own (see `sm120_fp8_fp4_mega_moe.cuh`). The m-grouped PSUM layout is the
// only GEMM type that has them
enum class SM120MegaMoEEpilogue : uint32_t {
    // Store `D = alpha * (A @ B^T)`
    None = 0,
    // L1: round the tile to BF16, clamp, SwiGLU, top-k weight, cast to FP8 with a UE8M0 scale per 32 outputs, and store the
    // FP8 rows and the packed scales of the L2 GEMM instead of D (gate/up interleaved in groups of 8 columns)
    SwiGLU = 1,
};

// What the fused epilogues read and write. A field a kind does not use stays null
struct SM120MegaMoEEpilogueArgs {
    // Row `r` of the GEMM serves pair `row_pair[r]` (-1: the tail of an expert's run)
    const int* row_pair;
    // SwiGLU: the top-k weight of each row; the FP8 rows `[rows, N / 2]` and packed scale words `[N / 2 / 128][sf_stride]` of
    // the next GEMM
    const float* row_weights;
    uint8_t* l2_acts;
    int* l2_acts_sf;
    float activation_clamp;
    uint32_t sf_stride;
};

}  // namespace deep_gemm
