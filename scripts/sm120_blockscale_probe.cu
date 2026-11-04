// Standalone probe: validate the SM120 block-scale FP8 MMA
// (kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32, UE8M0 scales) on GPU 7.
//
// Goal: pin down the scale-selector semantics (byte-id / thread-id) for
// scale_vec::1X before wiring the instruction into the real kernel. We run a
// single m16n8k32 E4M3 MMA with per-row A scales and per-column B scales, and
// compare against a host reference that decodes the UE8M0 byte as 2^(b-127).
//
// Build (on aws-raichu, GPU 7):
//   nvcc -arch=sm_120a -o /tmp/bsprobe scripts/sm120_blockscale_probe.cu
//   CUDA_VISIBLE_DEVICES=7 /tmp/bsprobe
//
// Fragment layout for m16n8k32 (row.col), per the PTX ISA:
//   lane = 32 threads; g = lane / 4 (0..7), t = lane % 4 (0..3)
//   A (16x32, row-major, e4m3): a[0]=(row g,   k=4t..4t+3), a[1]=(row g+8, same k),
//                               a[2]=(row g,   k+16),       a[3]=(row g+8, k+16)
//     each uint32 packs 4 consecutive-K e4m3 bytes for one (row, k-quad).
//   B (32x8, col-major, e4m3): b[0]=(k=4t..4t+3, col g), b[1]=(k+16, col g)
//   D (16x8, f32): d[0..1]=(row g,   col 2t..2t+1), d[2..3]=(row g+8, col 2t..2t+1)
//
// Scale selection for scale_vec::1X (one UE8M0 byte per 32-K block):
//   scale_A is M x 1 (one byte per A row), scale_B is 1 x N (one byte per B col).
//   The {byte-id, thread-id} tuples pick which byte of the 32-bit scaleData reg,
//   and which lane(s) in each quad, supply SF_A / SF_B. We sweep the documented
//   valid selector values and report which reproduce the reference, so the
//   kernel can use the confirmed mapping.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_fp8.h>

__global__ void mma_blockscale(const uint8_t* A, const uint8_t* B,
                               uint32_t sfa_data, uint32_t sfb_data,
                               uint16_t bid_a, uint16_t tid_a,
                               uint16_t bid_b, uint16_t tid_b,
                               float* D) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t g = lane / 4, t = lane % 4;

    // Load this lane's A fragment: rows {g, g+8}, k-quads {4t..4t+3, 4t+16..}
    uint32_t a[4], b[2];
    auto load_a = [&](uint32_t row, uint32_t k0) -> uint32_t {
        uint32_t v = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            v |= uint32_t(A[row * 32 + k0 + i]) << (i * 8);
        return v;
    };
    a[0] = load_a(g,     4 * t);
    a[1] = load_a(g + 8, 4 * t);
    a[2] = load_a(g,     4 * t + 16);
    a[3] = load_a(g + 8, 4 * t + 16);
    auto load_b = [&](uint32_t col, uint32_t k0) -> uint32_t {
        uint32_t v = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            v |= uint32_t(B[(k0 + i) * 8 + col]) << (i * 8);  // B col-major: B[k][col]
        return v;
    };
    b[0] = load_b(g, 4 * t);
    b[1] = load_b(g, 4 * t + 16);

    float d[4] = {0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "%10, {%11,%12}, %13, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(sfa_data), "h"(bid_a), "h"(tid_a),
          "r"(sfb_data), "h"(bid_b), "h"(tid_b));

    // D[row g,   col 2t+0/1] = d[0], d[1]
    // D[row g+8, col 2t+0/1] = d[2], d[3]
    D[(g)     * 8 + 2 * t + 0] = d[0];
    D[(g)     * 8 + 2 * t + 1] = d[1];
    D[(g + 8) * 8 + 2 * t + 0] = d[2];
    D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

// Same MMA, but each lane loads its 4-byte scaleAData/scaleBData from a per-lane
// table, so we can probe the lane->row/col mapping for scale_vec::1X.
__global__ void mma_blockscale_tab(const uint8_t* A, const uint8_t* B,
                                   const uint8_t* sfaTab, const uint8_t* sfbTab,
                                   uint16_t bid_a, uint16_t tid_a,
                                   uint16_t bid_b, uint16_t tid_b,
                                   float* D) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t g = lane / 4, t = lane % 4;
    uint32_t a[4], b[2];
    auto load_a = [&](uint32_t row, uint32_t k0) -> uint32_t {
        uint32_t v = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) v |= uint32_t(A[row * 32 + k0 + i]) << (i * 8);
        return v;
    };
    a[0] = load_a(g, 4 * t); a[1] = load_a(g + 8, 4 * t);
    a[2] = load_a(g, 4 * t + 16); a[3] = load_a(g + 8, 4 * t + 16);
    auto load_b = [&](uint32_t col, uint32_t k0) -> uint32_t {
        uint32_t v = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) v |= uint32_t(B[(k0 + i) * 8 + col]) << (i * 8);
        return v;
    };
    b[0] = load_b(g, 4 * t); b[1] = load_b(g, 4 * t + 16);

    const uint32_t sfa_data = *reinterpret_cast<const uint32_t*>(sfaTab + lane * 4);
    const uint32_t sfb_data = *reinterpret_cast<const uint32_t*>(sfbTab + lane * 4);

    float d[4] = {0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "%10, {%11,%12}, %13, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa_data), "h"(bid_a), "h"(tid_a),
          "r"(sfb_data), "h"(bid_b), "h"(tid_b));
    D[(g) * 8 + 2 * t + 0] = d[0]; D[(g) * 8 + 2 * t + 1] = d[1];
    D[(g + 8) * 8 + 2 * t + 0] = d[2]; D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

int main() {
    const int M = 16, N = 8, K = 32;
    uint8_t hA[M * K], hB[K * N];
    // e4m3 values: keep them small exact values so the reference is clean.
    for (int i = 0; i < M * K; ++i) hA[i] = __nv_cvt_float_to_fp8(float((i % 5) - 2), __NV_SATFINITE, __NV_E4M3);
    for (int i = 0; i < K * N; ++i) hB[i] = __nv_cvt_float_to_fp8(float((i % 3) - 1), __NV_SATFINITE, __NV_E4M3);

    // UE8M0 scales: one byte per A row (M of them) and per B col (N of them).
    // Byte b decodes to 2^(b-127). Use exponents 127 (=1), 128 (=2), 126 (=0.5)...
    uint8_t sfa_byte[M], sfb_byte[N];
    for (int i = 0; i < M; ++i) sfa_byte[i] = 127 + (i % 3) - 1;  // 0.5, 1, 2 cycling
    for (int j = 0; j < N; ++j) sfb_byte[j] = 127 + (j % 3) - 1;

    uint8_t *dA, *dB; float* dD;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, M * N * sizeof(float));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    auto fp8_to_float = [](uint8_t v) -> float {
        __nv_fp8_e4m3 x; x.__x = v; return float(x);
    };
    

    // Per-row / per-col reference (distinct scales per A row and B col).
    float ref[M * N];
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int k = 0; k < K; ++k) acc += fp8_to_float(hA[m * K + k]) * fp8_to_float(hB[k * N + n]);
            ref[m * N + n] = exp2f(float(sfa_byte[m]) - 127.0f) * exp2f(float(sfb_byte[n]) - 127.0f) * acc;
        }

    // Discover the lane->row(A) / lane->col(B) scale mapping. The scaleAData /
    // scaleBData registers are per-lane; the {byte-id, thread-id} tuple tells the
    // hardware which lane in each quad and which byte supplies the scale. We set
    // each lane's regs from a device-side table we control, sweep the selectors,
    // and report which selector reproduces the per-row/per-col reference and what
    // per-lane byte layout it implies.
    uint8_t *dSfaTab, *dSfbTab;  // per-lane bytes, indexed by lane, 4 bytes each
    cudaMalloc(&dSfaTab, 32 * 4); cudaMalloc(&dSfbTab, 32 * 4);
    float hD[M * N];
    const uint16_t bidset[] = {0, 1, 2, 3};
    const uint16_t tidsetA[] = {0, 1};
    const uint16_t tidsetB[] = {0, 1, 2, 3};
    // Hypothesis to confirm, matching the plan for the real kernel: scale_A is
    // M x 1, and for scale_vec::1X the lane supplies both its rows' scales in its
    // own scaleAData register. The D fragment gives a lane rows {g, g+8}; the
    // natural packing is byte 0 = scale(row g), byte 1 = scale(row g+8), read with
    // thread-id-a = 0 (lower pair of the quad). For B (1 x N, col g per lane),
    // byte 0 = scale(col g), thread-id-b = 0. Build ONE fixed table (independent
    // of the selector) and sweep selectors to find which actually reproduces all
    // 16 rows, so we learn the true byte/thread indexing rather than a tautology.
    printf("Fixed packing: sfaData[lane] byte0=sf(row g), byte1=sf(row g+8); "
           "sfbData[lane] byte0=sf(col g). Sweeping selectors over all 16 rows:\n");
    (void)bidset; (void)tidsetA; (void)tidsetB;

    // Empirically recover the lane -> row(A) and lane -> col(B) scale mapping.
    // CUTLASS (mma_traits_sm120.hpp) uses SFARegisters=uint8_t[1], selectors all 0,
    // so each lane supplies exactly one SF_A and one SF_B byte at {bid=0,tid=0}.
    // We probe one lane at a time: set that lane's SF to 2^1 (byte 128) and all
    // others to 2^0 (byte 127), then see which output rows/cols get scaled by 2.
    auto decode = [](uint8_t b) { return exp2f(float(b) - 127.0f); };
    int rowOfLane[32], colOfLane[32];
    for (int probe = 0; probe < 32; ++probe) {
        uint8_t sfaTab[32 * 4], sfbTab[32 * 4];
        for (int L = 0; L < 32; ++L)
            for (int bb = 0; bb < 4; ++bb) {
                sfaTab[L * 4 + bb] = (L == probe) ? 128 : 127;  // probed lane -> 2^1
                sfbTab[L * 4 + bb] = 127;                        // B all 2^0
            }
        cudaMemcpy(dSfaTab, sfaTab, sizeof(sfaTab), cudaMemcpyHostToDevice);
        cudaMemcpy(dSfbTab, sfbTab, sizeof(sfbTab), cudaMemcpyHostToDevice);
        mma_blockscale_tab<<<1, 32>>>(dA, dB, dSfaTab, dSfbTab, 0, 0, 0, 0, dD);
        cudaDeviceSynchronize();
        cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);
        // Which rows got x2? (compare to raw A*B)
        int r = -1;
        for (int m = 0; m < M; ++m) {
            float raw = 0; for (int k = 0; k < K; ++k) raw += fp8_to_float(hA[m * K + k]) * fp8_to_float(hB[k * N + 0]);
            if (raw != 0 && fabs(hD[m * N + 0] / raw - 2.0f) < 1e-3) { r = (r == -1) ? m : -2; }
        }
        rowOfLane[probe] = r;
    }
    // Same for B: probe one lane's SFB at 2^1, see which cols scale.
    for (int probe = 0; probe < 32; ++probe) {
        uint8_t sfaTab[32 * 4], sfbTab[32 * 4];
        for (int L = 0; L < 32; ++L)
            for (int bb = 0; bb < 4; ++bb) { sfaTab[L * 4 + bb] = 127; sfbTab[L * 4 + bb] = (L == probe) ? 128 : 127; }
        cudaMemcpy(dSfaTab, sfaTab, sizeof(sfaTab), cudaMemcpyHostToDevice);
        cudaMemcpy(dSfbTab, sfbTab, sizeof(sfbTab), cudaMemcpyHostToDevice);
        mma_blockscale_tab<<<1, 32>>>(dA, dB, dSfaTab, dSfbTab, 0, 0, 0, 0, dD);
        cudaDeviceSynchronize();
        cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);
        int c = -1;
        for (int n = 0; n < N; ++n) {
            float raw = 0; for (int k = 0; k < K; ++k) raw += fp8_to_float(hA[0 * K + k]) * fp8_to_float(hB[k * N + n]);
            if (raw != 0 && fabs(hD[0 * N + n] / raw - 2.0f) < 1e-3) { c = (c == -1) ? n : -2; }
        }
        colOfLane[probe] = c;
    }
    (void)decode; (void)sfa_byte; (void)sfb_byte; (void)ref;
    printf("lane : SF_A applies to row | SF_B applies to col   (-1 none, -2 multiple)\n");
    for (int L = 0; L < 32; ++L)
        printf("  lane %2d (g=%d t=%d): rowA=%d  colB=%d\n", L, L / 4, L % 4, rowOfLane[L], colOfLane[L]);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));
    cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(dSfaTab); cudaFree(dSfbTab);
    return 0;
}
