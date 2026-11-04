// Standalone probe for the SM120 FP4 block-scale MMA
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
// Verifies on GPU 7:
//   1. the packed-e2m1 A/B fragment layout and nibble order (vs a host reference)
//   2. the lane / byte mapping of the 16-bit UE8M0 scale registers (2 bytes = the two 32-K halves)
//
// Build: nvcc -gencode arch=compute_120f,code=sm_120f -o /tmp/fp4probe scripts/sm120_fp4_probe.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void mma_fp4(const uint8_t* A, const uint8_t* B, const uint32_t* sfa, const uint32_t* sfb, float* D) {
    const uint32_t lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    auto ld = [](const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); };
    // A: [16][32 bytes], B: [8][32 bytes] (K-major, two e2m1 per byte, low nibble = even k)
    uint32_t a0 = ld(A + g * 32 + 4 * t), a1 = ld(A + (g + 8) * 32 + 4 * t);
    uint32_t a2 = ld(A + g * 32 + 16 + 4 * t), a3 = ld(A + (g + 8) * 32 + 16 + 4 * t);
    uint32_t b0 = ld(B + g * 32 + 4 * t), b1 = ld(B + g * 32 + 16 + 4 * t);
    float d[4] = {0, 0, 0, 0};
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(sfa[lane]), "h"(z), "h"(z), "r"(sfb[lane]), "h"(z), "h"(z));
    D[g * 8 + 2 * t] = d[0]; D[g * 8 + 2 * t + 1] = d[1];
    D[(g + 8) * 8 + 2 * t] = d[2]; D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float dec(uint8_t nib) { float v = kE2M1[nib & 7]; return (nib & 8) ? -v : v; }
static float elem(const uint8_t* row, int k) { uint8_t b = row[k / 2]; return dec((k & 1) ? (b >> 4) : (b & 0xF)); }

static float* dD; static uint8_t *dA, *dB; static uint32_t *dSfa, *dSfb;
static void run(const uint8_t* A, const uint8_t* B, const uint32_t* sfa, const uint32_t* sfb, float* out) {
    cudaMemcpy(dA, A, 16 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, 8 * 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dSfa, sfa, 32 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dSfb, sfb, 32 * 4, cudaMemcpyHostToDevice);
    mma_fp4<<<1, 32>>>(dA, dB, dSfa, dSfb, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 16 * 8 * 4, cudaMemcpyDeviceToHost);
}

int main() {
    cudaMalloc(&dA, 16 * 32); cudaMalloc(&dB, 8 * 32); cudaMalloc(&dD, 128 * 4);
    cudaMalloc(&dSfa, 128); cudaMalloc(&dSfb, 128);
    uint8_t A[16 * 32], B[8 * 32];
    srand(1);
    for (auto& x : A) x = rand() & 0xFF;
    for (auto& x : B) x = rand() & 0xFF;
    uint32_t sfa[32], sfb[32], one = 0x7F7F;  // both bytes = 2^0
    for (int i = 0; i < 32; ++i) sfa[i] = sfb[i] = one;
    float D[128];
    run(A, B, sfa, sfb, D);
    double maxerr = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            double ref = 0;
            for (int k = 0; k < 64; ++k) ref += double(elem(A + m * 32, k)) * double(elem(B + n * 32, k));
            maxerr = fmax(maxerr, fabs(ref - D[m * 8 + n]));
        }
    printf("[1] fragment layout, uniform scale 2^0: max abs err = %g  (D[0]=%g)\n", maxerr, D[0]);

    // [2] scale lane/byte map. All-ones data (e2m1 code 2 = 1.0 -> byte 0x22), only one K half non-zero.
    for (int half = 0; half < 2; ++half) {
        uint8_t A1[16 * 32], B1[8 * 32];
        for (int r = 0; r < 16; ++r) for (int b = 0; b < 32; ++b) A1[r * 32 + b] = ((b >= 16) == (half == 1)) ? 0x22 : 0;
        for (int r = 0; r < 8; ++r) for (int b = 0; b < 32; ++b) B1[r * 32 + b] = ((b >= 16) == (half == 1)) ? 0x22 : 0;
        for (int side = 0; side < 2; ++side)          // 0 = SFA, 1 = SFB
            for (int probe = 0; probe < 32; ++probe)
                for (int byte = 0; byte < 2; ++byte) {
                    uint32_t ta[32], tb[32];
                    for (int i = 0; i < 32; ++i) ta[i] = tb[i] = one;
                    uint32_t v = one; v = (v & ~(0xFFu << (8 * byte))) | (128u << (8 * byte));
                    (side == 0 ? ta : tb)[probe] = v;
                    run(A1, B1, ta, tb, D);
                    int rows = 0, first = -1;
                    if (side == 0) { for (int m = 0; m < 16; ++m) if (fabs(D[m * 8] - 64.f) < 1e-3) { ++rows; if (first < 0) first = m; } }
                    else           { for (int n = 0; n < 8; ++n)  if (fabs(D[n] - 64.f) < 1e-3)      { ++rows; if (first < 0) first = n; } }
                    if (rows) printf("    K-half %d, SF%c lane %2d (g=%d t=%d) byte %d -> %s %d (x%d)\n",
                                     half, side ? 'B' : 'A', probe, probe / 4, probe % 4, byte, side ? "col" : "row", first, rows);
                }
    }
    return 0;
}
