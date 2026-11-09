// SM120 distributed-shared-memory (DSMEM) probe for the cluster tile sharing of `sm120_fp8_fp4_gemm_1d1d` (task 23).
//
// Measures how fast one CTA of a cluster can push shared memory into a peer CTA's shared memory, and the latency of a
// push-and-signal round trip, with the three ways of doing it:
//   mode 0: `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes` (one instruction per slice)
//   mode 1: `st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32` from the 32 lanes of a warp
//   mode 3: pull, `ld.shared::cluster.v4.b32` of the predecessor's 4096 bytes by each of the last `num_senders` warps (8 loads in flight per lane)
//   mode 4: latency of one dependent `ld.shared::cluster` (pointer chase)
//   mode 5: mbarrier ping-pong between ranks 0 and 1 (remote arrive + wait), cycles per round trip
//   mode 2: `st.shared::cluster.v4.b32` from a warp, then one release arrive on the peer's mbarrier
// Every CTA of a cluster pushes to its ring neighbour `(rank + 1) % C`. `--push-bytes` is the size of one push, `--iters`
// the number of pushes per CTA; all pushes complete on one mbarrier per receiver (transaction count `iters * bytes`).
// `iters = 1` with a small push measures the latency of one push.
//
// Build and run (GPU 7):
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 scripts/sm120_cluster_probe.cu -o tmp/cluster_probe
//   tmp/cluster_probe <mode> <cluster size> <grid size> <push bytes> <iters> [warps pushing at once, default 1]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t map_rank(uint32_t a, uint32_t r) {
    uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o;
}
__device__ __forceinline__ uint64_t globaltimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release;\n\tbarrier.cluster.wait.acquire;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n)); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
    asm volatile("{\n\t.reg .pred p;\n\tL:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t@!p bra L;\n\t}" :: "r"(a), "r"(parity) : "memory");
}

// Keeps every SM busy for `ms` milliseconds, so that the clocks are at their running value before a measurement
__global__ void warm_up(uint64_t ns, float* sink) {
    const uint64_t t0 = globaltimer();
    float x = threadIdx.x;
    while (globaltimer() - t0 < ns)
        x = x * 1.0001f + 0.5f;
    if (x == 12345.f)
        sink[0] = x;
}

extern __shared__ __align__(1024) uint8_t smem[];

// smem: [0, kMax) source, [kMax, 2 * kMax) destination of the peer's pushes, then the barrier
__global__ void probe(int mode, uint32_t push_bytes, uint32_t iters, uint32_t num_senders, uint64_t* out) {
    constexpr uint32_t kMax = 32768;
    uint32_t rank, size;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(size));
    const uint32_t src = smem_addr(smem), dst = smem_addr(smem + kMax), bar = smem_addr(smem + 2 * kMax);
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        mbar_init(bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    for (uint32_t i = threadIdx.x; i < kMax / 4; i += blockDim.x)
        reinterpret_cast<uint32_t*>(smem)[i] = i + rank;
    __syncthreads();
    cluster_sync();
    if (mode >= 4) {
        // Latency modes, rank 0 / thread 0 only (rank 1 answers in mode 5)
        uint64_t cycles = 0, ns = 0;
        const uint32_t other = rank ^ 1;
        if (mode == 4 and rank == 0 and threadIdx.x == 0) {
            // Dependent `ld.shared::cluster` chain
            const uint32_t peer_src = map_rank(src, other);
            uint32_t v = 0;
            const uint64_t c0 = clock64(), t0 = globaltimer();
            for (uint32_t i = 0; i < iters; ++ i)
                asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(v) : "r"(peer_src + (v & 4)) : "memory");
            cycles = clock64() - c0; ns = globaltimer() - t0;
            if (v == 0x7654321u) smem[0] = 1;
        } else if (mode == 5 and rank < 2 and threadIdx.x == 0) {
            // mbarrier ping-pong: a remote arrive, and the wait for the answer's arrive
            const uint32_t other_bar = map_rank(bar, other);
            const uint64_t c0 = clock64(), t0 = globaltimer();
            for (uint32_t i = 0; i < iters; ++ i) {
                if (rank == 0) {
                    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(other_bar) : "memory");
                    mbar_wait(bar, i & 1);
                } else {
                    mbar_wait(bar, i & 1);
                    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(other_bar) : "memory");
                }
            }
            cycles = clock64() - c0; ns = globaltimer() - t0;
        }
        cluster_sync();
        if (threadIdx.x == 0) { out[blockIdx.x * 8 + 2] = cycles; out[blockIdx.x * 8 + 3] = ns; }
        return;
    }
    const uint32_t peer = (rank + 1) % size;
    const uint32_t peer_dst = map_rank(dst, peer), peer_bar = map_rank(bar, peer);
    uint64_t t_begin = 0, t_end = 0, t_issue = 0, t_issue_ns = 0;
    // The receiver expects the whole stream of its ring predecessor (modes 0 and 1: `iters * bytes` of transaction count;
    // mode 2: one plain arrival after the last store)
    if (threadIdx.x == 0) {
        if (mode < 2)
            mbar_expect_tx(bar, iters * push_bytes * num_senders);
    }
    cluster_sync();
    if (warp < num_senders and mode != 3) {
        t_begin = globaltimer();
        const uint64_t c0 = clock64();
        for (uint32_t i = 0; i < iters; ++ i) {
            if (mode == 0) {
                if (lane == 0)
                    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                 :: "r"(peer_dst), "r"(src), "r"(push_bytes), "r"(peer_bar) : "memory");
            } else {
                // 4096-byte pushes only: all 8 loads, then all 8 stores, so that nothing orders a store behind a load
                uint4 v[8];
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++ j)
                    v[j] = *reinterpret_cast<const uint4*>(smem + j * 512 + lane * 16);
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++ j) {
                    if (mode == 1)
                        asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                     :: "r"(peer_dst + j * 512 + lane * 16), "r"(v[j].x), "r"(v[j].y), "r"(v[j].z), "r"(v[j].w), "r"(peer_bar) : "memory");
                    else
                        asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};"
                                     :: "r"(peer_dst + j * 512 + lane * 16), "r"(v[j].x), "r"(v[j].y), "r"(v[j].z), "r"(v[j].w) : "memory");
                }
            }
        }
        if (mode == 2) {
            __syncwarp();
            if (lane == 0)
                asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(peer_bar) : "memory");
        }
        t_issue = clock64() - c0;
        t_issue_ns = globaltimer() - t_begin;
    }
    if (mode == 3) {
        // Pull: the last `num_senders` warps read the predecessor's 4096-byte source `iters` times (`ld.shared::cluster`), 8 loads in flight
        if (warp >= 4 - num_senders) {
            const uint32_t pred = (rank + size - 1) % size, peer_src = map_rank(src, pred);
            uint32_t acc = 0;
            t_begin = globaltimer();
            for (uint32_t i = 0; i < iters; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++ j) {
                    uint4 v;
                    asm volatile("ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(peer_src + j * 512 + lane * 16) : "memory");
                    acc ^= v.x ^ v.y ^ v.z ^ v.w;
                }
            }
            t_end = globaltimer();
            if (acc == 0x12345678u)
                smem[0] = 1;
        }
    } else if (threadIdx.x == 96) {
        mbar_wait(bar, 0);
        t_end = globaltimer();
    }
    cluster_sync();
    if (threadIdx.x == 0 or threadIdx.x == 96) {
        const uint32_t slot = (blockIdx.x * 2 + threadIdx.x / 96) * 4;
        out[slot] = t_begin; out[slot + 1] = t_end; out[slot + 2] = t_issue; out[slot + 3] = t_issue_ns;
    }
}

int main(int argc, char** argv) {
    if (argc < 6) { printf("usage: %s mode cluster grid push_bytes iters\n", argv[0]); return 1; }
    const int mode = atoi(argv[1]), cluster = atoi(argv[2]), grid = atoi(argv[3]);
    const uint32_t push_bytes = atoi(argv[4]), iters = atoi(argv[5]);
    const uint32_t num_senders = argc > 6 ? atoi(argv[6]) : 1;   // warps pushing at the same time (<= 3)
    const int smem_bytes = 2 * 32768 + 64;
    CHECK(cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CHECK(cudaFuncSetAttribute(probe, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    uint64_t* out; CHECK(cudaMalloc(&out, grid * 8 * sizeof(uint64_t)));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = smem_bytes;
    cudaLaunchAttribute attr; attr.id = cudaLaunchAttributeClusterDimension; attr.val.clusterDim = {(unsigned)cluster, 1, 1};
    cfg.attrs = &attr; cfg.numAttrs = 1;
    int max_clusters = 0;
    CHECK(cudaOccupancyMaxActiveClusters(&max_clusters, (void*)probe, &cfg));
    printf("max active clusters of %d with %d B smem: %d (%d CTAs)\n", cluster, smem_bytes, max_clusters, max_clusters * cluster);
    float* sink; CHECK(cudaMalloc(&sink, 4));
    for (int rep = 0; rep < 3; ++ rep) {
        // 300 ms of load on every SM first: an idle GPU runs the SMs at a fraction of their clock
        warm_up<<<1024, 256>>>(300000000ull, sink);
        CHECK(cudaMemset(out, 0, grid * 8 * sizeof(uint64_t)));
        CHECK(cudaLaunchKernelEx(&cfg, probe, mode, push_bytes, iters, num_senders, out));
        CHECK(cudaDeviceSynchronize());
            uint64_t* h = (uint64_t*)malloc(grid * 8 * sizeof(uint64_t));
        CHECK(cudaMemcpy(h, out, grid * 8 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (mode >= 4) {
            printf("mode %d, cluster %d: %.0f cycles (%.0f ns) per %s\n", mode, cluster, (double)h[2] / iters, (double)h[3] / iters,
                   mode == 4 ? "dependent remote load" : "round trip (remote arrive + wait, twice)");
            free(h);
            continue;
        }
        // Per CTA: stream time = receiver's end (its predecessor's stream) - the sender's begin of that predecessor
        double sum_gbps = 0, worst_us = 0, issue_cycles = 0, issue_ns = 0; int n = 0;
        for (int b = 0; b < grid; ++ b) {
            const int base = b - b % cluster, pred = base + (b % cluster + cluster - 1) % cluster;
            const uint64_t begin = mode == 3 ? h[(b * 2 + 1) * 4] : h[(pred * 2) * 4], end = h[(b * 2 + 1) * 4 + 1];
            if (end > begin) {
                const double us = (end - begin) / 1e3;
                sum_gbps += (double)iters * push_bytes * num_senders / (end - begin);   // bytes per ns = GB/s
                worst_us = us > worst_us ? us : worst_us; ++ n;
                issue_cycles += h[(pred * 2) * 4 + 2]; issue_ns += h[(pred * 2) * 4 + 3];
            }
        }
        printf("mode %d, cluster %d, grid %d, %u x %u B: mean %.1f GB/s per CTA (aggregate %.2f TB/s), slowest stream %.1f us, "
               "sender issue %.0f cycles per push (%.0f ns)\n", mode, cluster, grid, iters, push_bytes, n ? sum_gbps / n : 0.0,
               sum_gbps / 1e3, worst_us, n ? issue_cycles / n / iters : 0.0, n ? issue_ns / n / iters : 0.0);
        free(h);
    }
    return 0;
}
