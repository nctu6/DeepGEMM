// Standalone probe for MN-major FP8 operands on SM120.
//
// Design (d), the native instruction: sm_120a (and sm_120f) have `ldmatrix.sync.aligned.m16n16.x{1,2}.trans.shared.b8`
// (SASS `LDSM.8.MT1616`). It loads a 16 K-row x 16 MN-byte block of an MN-major tile and hands every thread the
// standard m16n8k32 E4M3 fragment registers, no permutation needed. Per matrix (x1), with `g = lane / 4`, `t = lane % 4`:
//   r0 = [K 4t, 4t+1, 4t+2, 4t+3] of MN byte g        (A: a0 / B: b0 of the first 8 columns)
//   r1 = [K 4t .. 4t+3]           of MN byte g + 8    (A: a1 / B: b0 of the second 8 columns)
// Lane `l < 16` supplies the address of K row `l` (16 contiguous MN bytes); with `.x2`, lanes 16-31 supply the rows of the
// second matrix (K rows 16-31 of the 32-K step), giving r2, r3 = the same for K + 16 (A: a2, a3 / B: b1). So one `.x2`
// is one A fragment (16 MN x 32 K), or the B fragments of two 8-column tiles ({r0, r2} and {r1, r3}).
//
// Part 1, correctness (bit-exact) of (d), and of design (a): `ldmatrix.x4.trans` of 16-bit elements on the same tile, then
// `prmt` byte permutes (thread (g, t) receives K rows 2t, 2t + 1 and MN bytes 2g, 2g + 1 of each 8x8 b16 matrix; the K rows
// the lanes point at are chosen to keep the standard K order and to hit 8 distinct swizzle chunks per phase, and the MN
// order inside each 16 comes out permuted: fragment row g is MN byte 2g, row g + 8 is MN byte 2g + 1; the `prmt` selector
// depends on whether t < 2: 0x6420 / 0x2064 even byte, 0x7531 / 0x3175 odd byte).
//
// Part 2, throughput of the candidate designs for the operand loads of a 64x32 warp tile (the 128x128 / 8-warp tile of
// the 1D2D kernel), 188 CTAs, data resident in shared memory (no TMA, no scales):
//   0: K-major `ldmatrix` (the existing path, upper bound)
//   1: (a) `ldmatrix.trans` (b16) + `prmt`
//   2: (c) plain 32-bit `ld.shared` of 4 MN bytes x 4 K rows, 4x4 byte transpose with `prmt` (fragment contents are not
//      checked here; the real kernel spilled with it and was 8-18% slower than (a))
//   3: (b) 4 helper warps transpose the next MN-major tile into a K-major tile (double buffered), the math warps then
//      use the K-major `ldmatrix`
//   4: (d) `ldmatrix.m16n16.x2.trans.b8`
//
// Build and run (on aws-raichu, GPU 7 only):
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o /tmp/fp8_trans_probe scripts/sm120_fp8_trans_probe.cu
//   CUDA_VISIBLE_DEVICES=7 /tmp/fp8_trans_probe
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp8.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// Offset of 16-byte chunk `chunk` of row `row` in a TMA-swizzled tile with `W`-byte rows (W = 128 or 64)
template <int W>
__device__ __forceinline__ uint32_t swz(uint32_t row, uint32_t chunk) {
    if constexpr (W == 128)
        return row * 128 + ((chunk ^ (row & 7)) << 4);
    else
        return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm_b8_x2_trans(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
// 16 MN x 32 K of an MN-major tile with the native 8-bit transposing `ldmatrix`: lane `l` points at K row `l` of the step
template <int W, int KROWS>
__device__ __forceinline__ void load_b8(uint32_t base, uint32_t k_step, uint32_t mn_base, uint32_t lane, uint32_t* r) {
    ldsm_b8_x2_trans(r, base + (mn_base / W) * (KROWS * W) + swz<W>(k_step * 32 + lane, (mn_base % W) / 16));
}
__device__ __forceinline__ uint32_t lds32(const void* p) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(smem_addr(p)) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
__device__ __forceinline__ void mma_e4m3(const uint32_t* a, const uint32_t* b, float* d) {
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Byte offset of MN element `mn` of K row `k` in an MN-major stage (atoms of W MN bytes x KROWS rows)
template <int W, int KROWS>
__device__ __forceinline__ uint32_t mn_major_off(uint32_t k, uint32_t mn) {
    return (mn / W) * (KROWS * W) + swz<W>(k, (mn % W) / 16) + mn % 16;
}

// The lane's K row (within a 32-K step) for `ldmatrix.x4.trans`, and its `prmt` selectors
__device__ __forceinline__ uint32_t trans_k_row(uint32_t lane) {
    const uint32_t m = lane / 8, r = lane % 8, tr = r / 2, e = r % 2, is_y = m & 1;
    return 16 * (m >> 1) + 4 * tr + ((is_y ^ (tr >= 2)) ? 2 : 0) + e;
}

// 16 MN x 32 K of an MN-major tile into the 4 registers (X, Y, Z, W), then the fragments
template <int W, int KROWS>
__device__ __forceinline__ void load_trans(uint32_t base, uint32_t k_step, uint32_t mn_base, uint32_t lane, uint32_t* xyzw) {
    ldsm_x4_trans(xyzw, base + (mn_base / W) * (KROWS * W) + swz<W>(k_step * 32 + trans_k_row(lane), (mn_base % W) / 16));
}
__device__ __forceinline__ void frag_a(const uint32_t* xyzw, uint32_t lane, uint32_t* a) {
    const bool lo = (lane % 4) < 2;
    const uint32_t se = lo ? 0x6420u : 0x2064u, so = lo ? 0x7531u : 0x3175u;
    a[0] = prmt(xyzw[0], xyzw[1], se); a[1] = prmt(xyzw[0], xyzw[1], so);
    a[2] = prmt(xyzw[2], xyzw[3], se); a[3] = prmt(xyzw[2], xyzw[3], so);
}
__device__ __forceinline__ void frag_b(const uint32_t* xyzw, uint32_t lane, uint32_t* b_even, uint32_t* b_odd) {
    const bool lo = (lane % 4) < 2;
    const uint32_t se = lo ? 0x6420u : 0x2064u, so = lo ? 0x7531u : 0x3175u;
    b_even[0] = prmt(xyzw[0], xyzw[1], se); b_even[1] = prmt(xyzw[2], xyzw[3], se);
    b_odd[0]  = prmt(xyzw[0], xyzw[1], so); b_odd[1]  = prmt(xyzw[2], xyzw[3], so);
}

// ---------------------------------------------------------------------------------------------------------------------
// Part 1: correctness. One warp: D[16 x 16] = A[16 x 32] * B[16 x 32]^T from MN-major tiles with MN extent `W`
// (K = 32 rows), MN bases `a_base`, `b_base` inside the tile. D is written in fragment space, mapped back to the
// physical (m, n) with the MN permutation.
// ---------------------------------------------------------------------------------------------------------------------
template <int W>
__global__ void correctness_kernel(const uint8_t* a_src, const uint8_t* b_src, int a_base, int b_base, float* out) {
    // a_src / b_src: [W][32] (MN x K, K-major in global memory)
    __shared__ __align__(1024) uint8_t sa[32 * W];
    __shared__ __align__(1024) uint8_t sb[32 * W];
    for (int i = threadIdx.x; i < W * 32; i += 32) {
        const int mn = i / 32, k = i % 32;
        sa[mn_major_off<W, 32>(k, mn)] = a_src[i];
        sb[mn_major_off<W, 32>(k, mn)] = b_src[i];
    }
    __syncwarp();
    const uint32_t lane = threadIdx.x, g = lane / 4, t = lane % 4;
    uint32_t x[4], a[4], b_even[2], b_odd[2];
    load_trans<W, 32>(smem_addr(sa), 0, a_base, lane, x);
    frag_a(x, lane, a);
    load_trans<W, 32>(smem_addr(sb), 0, b_base, lane, x);
    frag_b(x, lane, b_even, b_odd);
    float d_even[4] = {0, 0, 0, 0}, d_odd[4] = {0, 0, 0, 0};
    mma_e4m3(a, b_even, d_even);
    mma_e4m3(a, b_odd, d_odd);
    // d[0..1]: fragment row g (physical 2g), columns 2t, 2t + 1; d[2..3]: row g + 8 (physical 2g + 1)
    // tile e (even/odd), fragment column c is physical 2c + e
    for (int half = 0; half < 2; ++ half)
        for (int e = 0; e < 2; ++ e)
            for (int x_ = 0; x_ < 2; ++ x_) {
                const int m = 2 * g + half, n = 2 * (2 * t + x_) + e;
                out[m * 16 + n] = (e == 0 ? d_even : d_odd)[half * 2 + x_];
            }
}

// Same test for the native instruction: the fragments are the standard ones, no MN permutation
template <int W>
__global__ void correctness_b8_kernel(const uint8_t* a_src, const uint8_t* b_src, int a_base, int b_base, float* out) {
    __shared__ __align__(1024) uint8_t sa[32 * W];
    __shared__ __align__(1024) uint8_t sb[32 * W];
    for (int i = threadIdx.x; i < W * 32; i += 32) {
        const int mn = i / 32, k = i % 32;
        sa[mn_major_off<W, 32>(k, mn)] = a_src[i];
        sb[mn_major_off<W, 32>(k, mn)] = b_src[i];
    }
    __syncwarp();
    const uint32_t lane = threadIdx.x, g = lane / 4, t = lane % 4;
    uint32_t a[4], b[4];
    load_b8<W, 32>(smem_addr(sa), 0, a_base, lane, a);
    load_b8<W, 32>(smem_addr(sb), 0, b_base, lane, b);
    // B of the 8-column tiles: {r0, r2} and {r1, r3}
    const uint32_t b_lo[2] = {b[0], b[2]}, b_hi[2] = {b[1], b[3]};
    float d_lo[4] = {0, 0, 0, 0}, d_hi[4] = {0, 0, 0, 0};
    mma_e4m3(a, b_lo, d_lo);
    mma_e4m3(a, b_hi, d_hi);
    for (int half = 0; half < 2; ++ half)
        for (int e = 0; e < 2; ++ e)
            for (int x_ = 0; x_ < 2; ++ x_)
                out[(g + 8 * half) * 16 + e * 8 + 2 * t + x_] = (e == 0 ? d_lo : d_hi)[half * 2 + x_];
}

static float decode(uint8_t v) { return static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3*>(&v)); }

template <int W, bool kNative>
static bool run_correctness(int a_base, int b_base) {
    std::vector<uint8_t> a(W * 32), b(W * 32);
    for (auto& v : a) v = static_cast<uint8_t>(__nv_fp8_e4m3(static_cast<float>((rand() % 33) - 16) / 8.f).__x);
    for (auto& v : b) v = static_cast<uint8_t>(__nv_fp8_e4m3(static_cast<float>((rand() % 33) - 16) / 8.f).__x);
    uint8_t *da, *db; float* dout;
    CHECK(cudaMalloc(&da, a.size())); CHECK(cudaMalloc(&db, b.size())); CHECK(cudaMalloc(&dout, 256 * 4));
    CHECK(cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
    if constexpr (kNative)
        correctness_b8_kernel<W><<<1, 32>>>(da, db, a_base, b_base, dout);
    else
        correctness_kernel<W><<<1, 32>>>(da, db, a_base, b_base, dout);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> out(256);
    CHECK(cudaMemcpy(out.data(), dout, 1024, cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (int m = 0; m < 16; ++ m)
        for (int n = 0; n < 16; ++ n) {
            double ref = 0;
            for (int k = 0; k < 32; ++ k)
                ref += static_cast<double>(decode(a[(a_base + m) * 32 + k])) * decode(b[(b_base + n) * 32 + k]);
            max_err = std::max(max_err, std::fabs(ref - out[m * 16 + n]));
        }
    printf("  %s W=%3d a_base=%3d b_base=%3d: max abs err %.3g %s\n", kNative ? "(d) ldmatrix.b8   " : "(a) trans + prmt   ", W, a_base, b_base, max_err, max_err < 1e-4 ? "OK" : "FAIL");
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dout));
    return max_err < 1e-4;
}

// ---------------------------------------------------------------------------------------------------------------------
// Part 2: throughput. A CTA holds a 128 (MN) x 128 (K) tile of A and of B; 8 math warps in a 2 x 4 grid, 64x32 warp tiles
// ---------------------------------------------------------------------------------------------------------------------
constexpr int kTile = 128 * 128;

template <int VARIANT>
__global__ void __launch_bounds__(VARIANT == 3 ? 384 : 256, 1) bench_kernel(const uint8_t* src, float* out, int iters) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* a_mn = smem;                  // MN-major tiles (what TMA delivers)
    uint8_t* b_mn = smem + kTile;
    uint8_t* a_k = smem + 2 * kTile;       // K-major tiles (variants 0 and 3)
    uint8_t* b_k = smem + 3 * kTile;
    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int i = tid; i < kTile / 4; i += blockDim.x) {
        // Fill both MN-major and K-major copies from the same bytes (the values only matter for the sink)
        const uint32_t v = reinterpret_cast<const uint32_t*>(src)[i];
        reinterpret_cast<uint32_t*>(a_mn)[i] = v; reinterpret_cast<uint32_t*>(b_mn)[i] = v ^ 0x01010101u;
        reinterpret_cast<uint32_t*>(a_k)[i] = v;  reinterpret_cast<uint32_t*>(b_k)[i] = v ^ 0x01010101u;
        if constexpr (VARIANT == 3) {
            reinterpret_cast<uint32_t*>(a_k + 2 * kTile)[i] = v; reinterpret_cast<uint32_t*>(b_k + 2 * kTile)[i] = v;
        }
    }
    __syncthreads();

    float acc[4][4][4] = {};
    const uint32_t warp_m = (warp % 8) / 4, warp_n = warp % 4;
    const uint32_t a_ld_row = warp_m * 64 + ((lane >> 3) & 1) * 8 + (lane & 7), a_ld_chunk = lane >> 4;
    const uint32_t b_ld_row = warp_n * 32 + (lane >> 4) * 8 + (lane & 7), b_ld_chunk = (lane >> 3) & 1;
    const uint32_t g = lane / 4, t = lane % 4;

    for (int it = 0; it < iters; ++ it) {
        const uint32_t buf = VARIANT == 3 ? (it & 1) : 0;
        if (warp < 8) {
            const uint32_t a_k_base = smem_addr(a_k) + buf * 2 * kTile, b_k_base = smem_addr(b_k) + buf * 2 * kTile;
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++ k) {
                uint32_t a_frag[4][4], b_frag[4][2];
                if constexpr (VARIANT == 0 || VARIANT == 3) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++ i)
                        ldsm_x4(a_frag[i], a_k_base + swz<128>(a_ld_row + i * 16, k * 2 + a_ld_chunk));
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        uint32_t r[4];
                        ldsm_x4(r, b_k_base + swz<128>(b_ld_row + j * 16, k * 2 + b_ld_chunk));
                        b_frag[j * 2][0] = r[0]; b_frag[j * 2][1] = r[1]; b_frag[j * 2 + 1][0] = r[2]; b_frag[j * 2 + 1][1] = r[3];
                    }
                } else if constexpr (VARIANT == 4) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++ i)
                        load_b8<128, 128>(smem_addr(a_mn), k, warp_m * 64 + i * 16, lane, a_frag[i]);
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        uint32_t r[4];
                        load_b8<128, 128>(smem_addr(b_mn), k, warp_n * 32 + j * 16, lane, r);
                        b_frag[j * 2][0] = r[0]; b_frag[j * 2][1] = r[2]; b_frag[j * 2 + 1][0] = r[1]; b_frag[j * 2 + 1][1] = r[3];
                    }
                } else if constexpr (VARIANT == 1) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++ i) {
                        uint32_t x[4];
                        load_trans<128, 128>(smem_addr(a_mn), k, warp_m * 64 + i * 16, lane, x);
                        frag_a(x, lane, a_frag[i]);
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        uint32_t x[4];
                        load_trans<128, 128>(smem_addr(b_mn), k, warp_n * 32 + j * 16, lane, x);
                        frag_b(x, lane, b_frag[j * 2], b_frag[j * 2 + 1]);
                    }
                } else {
                    // (c): thread (g, t) takes MN bytes 4q .. 4q + 3 (q = g / 2) of 4 K rows 4t .. 4t + 3 and of 16 + 4t ..
                    // with 32-bit loads, and transposes the 4x4 byte block with prmt. Two of the 4 MN bytes are kept
                    const uint32_t q = g / 2, p = g % 2;
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++ i) {
                        uint32_t w[8];
                        #pragma unroll
                        for (uint32_t r = 0; r < 8; ++ r) {
                            const uint32_t kk = (r < 4 ? 4 * t + r : 16 + 4 * t + (r - 4)) + k * 32;
                            w[r] = lds32(a_mn + mn_major_off<128, 128>(kk, warp_m * 64 + i * 16 + 4 * q));
                        }
                        #pragma unroll
                        for (uint32_t h = 0; h < 2; ++ h) {
                            const uint32_t* v = w + h * 4;
                            // Byte p of every word, then byte p + 2
                            const uint32_t lo01 = prmt(v[0], v[1], 0x5140 + p * 0x0202), lo23 = prmt(v[2], v[3], 0x5140 + p * 0x0202);
                            const uint32_t hi01 = prmt(v[0], v[1], 0x5140 + (p + 2) * 0x0202), hi23 = prmt(v[2], v[3], 0x5140 + (p + 2) * 0x0202);
                            a_frag[i][h * 2]     = prmt(lo01, lo23, 0x5410);
                            a_frag[i][h * 2 + 1] = prmt(hi01, hi23, 0x5410);
                        }
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++ j) {
                        uint32_t w[8];
                        #pragma unroll
                        for (uint32_t r = 0; r < 8; ++ r) {
                            const uint32_t kk = (r < 4 ? 4 * t + r : 16 + 4 * t + (r - 4)) + k * 32;
                            w[r] = lds32(b_mn + mn_major_off<128, 128>(kk, warp_n * 32 + j * 8 + (g / 4) * 4));
                        }
                        #pragma unroll
                        for (uint32_t h = 0; h < 2; ++ h) {
                            const uint32_t* v = w + h * 4;
                            const uint32_t lo01 = prmt(v[0], v[1], 0x5140 + (g % 4 & 1) * 0x0202), lo23 = prmt(v[2], v[3], 0x5140 + (g % 4 & 1) * 0x0202);
                            b_frag[j][h] = prmt(lo01, lo23, 0x5410);
                        }
                    }
                }
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++ i)
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++ j)
                        mma_e4m3(a_frag[i], b_frag[j], acc[i][j]);
            }
        }
        if constexpr (VARIANT == 3) {
            if (warp >= 8) {
                // Helper warps: transpose the MN-major A and B tiles into the other K-major buffer, 4x4 byte blocks at a time
                const uint32_t h = tid - 256, nbuf = buf ^ 1;
                uint8_t* dst_a = a_k + nbuf * 2 * kTile; uint8_t* dst_b = b_k + nbuf * 2 * kTile;
                #pragma unroll 4
                for (uint32_t blk = h; blk < 2 * (128 / 4) * (128 / 4); blk += 128) {
                    const bool is_b = blk >= 1024;
                    const uint32_t bi = blk % 1024, kq = bi / 32, mq = bi % 32;   // 4 K rows, 4 MN bytes
                    const uint8_t* srcp = is_b ? b_mn : a_mn;
                    uint32_t w[4];
                    #pragma unroll
                    for (uint32_t r = 0; r < 4; ++ r)
                        w[r] = lds32(srcp + mn_major_off<128, 128>(kq * 4 + r, mq * 4));
                    const uint32_t l01 = prmt(w[0], w[1], 0x5140), l23 = prmt(w[2], w[3], 0x5140);
                    const uint32_t h01 = prmt(w[0], w[1], 0x7362), h23 = prmt(w[2], w[3], 0x7362);
                    uint32_t o[4];
                    o[0] = prmt(l01, l23, 0x5410); o[1] = prmt(l01, l23, 0x7632);
                    o[2] = prmt(h01, h23, 0x5410); o[3] = prmt(h01, h23, 0x7632);
                    #pragma unroll
                    for (uint32_t r = 0; r < 4; ++ r) {
                        const uint32_t mn = mq * 4 + r, kb = kq * 4;
                        *reinterpret_cast<uint32_t*>((is_b ? dst_b : dst_a) + swz<128>(mn, kb / 16) + kb % 16) = o[r];
                    }
                }
            }
            __syncthreads();
        }
    }
    float s = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++ i)
        #pragma unroll
        for (int j = 0; j < 4; ++ j)
            #pragma unroll
            for (int x = 0; x < 4; ++ x)
                s += acc[i][j][x];
    out[blockIdx.x * blockDim.x + tid] = s;
}

template <int VARIANT>
static float run_bench(const uint8_t* dsrc, float* dout, int iters, int blocks) {
    const int smem = (VARIANT == 3 ? 6 : 4) * kTile;
    CHECK(cudaFuncSetAttribute(bench_kernel<VARIANT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const int threads = VARIANT == 3 ? 384 : 256;
    bench_kernel<VARIANT><<<blocks, threads, smem>>>(dsrc, dout, 16);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < 5; ++ r) {
        cudaEventRecord(e0);
        bench_kernel<VARIANT><<<blocks, threads, smem>>>(dsrc, dout, iters);
        cudaEventRecord(e1); CHECK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
    }
    return best * 1e6f / iters;   // ns per K block (128x128x128 per CTA... 8 warps x 16 MMAs x 4 steps)
}

int main() {
    srand(1);
    printf("Correctness (a) ldmatrix.x4.trans + prmt, (d) ldmatrix.m16n16.b8:\n");
    bool ok = true;
    for (int a_base : {0, 16, 48, 112}) for (int b_base : {0, 32, 96}) { ok &= run_correctness<128, false>(a_base, b_base); ok &= run_correctness<128, true>(a_base, b_base); }
    for (int a_base : {0, 16, 48}) for (int b_base : {0, 32, 16}) { ok &= run_correctness<64, false>(a_base, b_base); ok &= run_correctness<64, true>(a_base, b_base); }
    printf(ok ? "ALL OK\n" : "FAILED\n");
    if (!ok) return 1;

    std::vector<uint8_t> src(kTile);
    for (auto& v : src) v = static_cast<uint8_t>(__nv_fp8_e4m3(static_cast<float>((rand() % 17) - 8) / 64.f).__x);
    uint8_t* dsrc; float* dout;
    CHECK(cudaMalloc(&dsrc, kTile)); CHECK(cudaMalloc(&dout, 188 * 384 * 4));
    CHECK(cudaMemcpy(dsrc, src.data(), kTile, cudaMemcpyHostToDevice));
    const int iters = 4096, blocks = 188;
    printf("Throughput, ns per 128x128x128 K block per CTA (8 math warps, 188 CTAs, smem resident):\n");
    const float t0 = run_bench<0>(dsrc, dout, iters, blocks);
    const float t1 = run_bench<1>(dsrc, dout, iters, blocks);
    const float t2 = run_bench<2>(dsrc, dout, iters, blocks);
    const float t3 = run_bench<3>(dsrc, dout, iters, blocks);
    const float t4 = run_bench<4>(dsrc, dout, iters, blocks);
    printf("  0 K-major ldmatrix (bound)       : %7.1f ns\n", t0);
    printf("  1 (a) ldmatrix.trans + prmt      : %7.1f ns (%.2fx of bound)\n", t1, t1 / t0);
    printf("  2 (c) ld.shared.b32 + prmt       : %7.1f ns (%.2fx of bound)\n", t2, t2 / t0);
    printf("  3 (b) helper-warp smem transpose : %7.1f ns (%.2fx of bound)\n", t3, t3 / t0);
    printf("  4 (d) ldmatrix.m16n16.b8         : %7.1f ns (%.2fx of bound)\n", t4, t4 / t0);
    return 0;
}
