// Standalone probe for the SM120 mixed FP8 x FP4 block-scale MMA
//   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
// A is all 1.0 (E4M3), scales 1.0; lane 0 sets one B byte (k = 0, n = 0). The printed D[0][0] for each E2M1 code
// 1-7 (0.5 1 1.5 2 3 4 6) shifted left by 0-4 shows where the 8-bit container holds the value: bits [5:2] (shift 2).
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -o sm120_mixed_mma_probe sm120_mixed_mma_probe.cu
#include <cstdio>
#include <cstdint>
__global__ void k(uint32_t bbyte, float* out) {
    const int lane = threadIdx.x;
    uint32_t a[4], b[2]; float d[4] = {0,0,0,0};
    // A: row g (lane/4) k=... all bytes 0x38 (e4m3 1.0)
    for (int i = 0; i < 4; ++i) a[i] = 0x38383838u;
    // B: only lane 0, byte 0 of b0 set (k = 0, n = 0)
    b[0] = (lane == 0) ? bbyte : 0u; b[1] = 0u;
    uint32_t sf = 127; // ue8m0 1.0
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sf), "r"(sf));
    if (lane == 0) out[0] = d[0];
}
int main() {
    float* o; cudaMallocManaged(&o, 4);
    const char* names[] = {"nibble<<0","nibble<<1","nibble<<2","nibble<<3","nibble<<4"};
    for (int shift = 0; shift <= 4; ++shift) {
        printf("%s:", names[shift]);
        for (uint32_t nib = 1; nib < 8; ++nib) {  // 0.5 1 1.5 2 3 4 6
            k<<<1,32>>>(nib << shift, o); cudaDeviceSynchronize();
            printf(" %g", o[0]);
        }
        printf("\n");
    }
    // sign bit
    k<<<1,32>>>(0x9u, o); cudaDeviceSynchronize(); printf("0x09 (neg 0.5 if low): %g\n", o[0]);
    k<<<1,32>>>(0x9u << 2, o); cudaDeviceSynchronize(); printf("0x09<<2: %g\n", o[0]);
    k<<<1,32>>>(0x9u << 4, o); cudaDeviceSynchronize(); printf("0x09<<4: %g\n", o[0]);
    return 0;
}
