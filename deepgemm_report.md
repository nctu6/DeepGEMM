# DeepGEMM on SM120 (RTX PRO 6000 Blackwell): report

## 1. Summary

DeepGEMM now runs on SM120 (compute capability 12.0, consumer and workstation
Blackwell). TokenSpeed uses it for DeepSeek-style block-FP8 models on that
hardware.

- **New kernels:**
  - Dense FP8 GEMM with FP32 scales (normal, m-grouped contiguous, m-grouped
    masked), and the V3.2 DSA indexer kernels (`fp8_mqa_logits`,
    `fp8_paged_mqa_logits`). FP32 1x128 activation scales and 128x128 weight
    scales, the same numerics as DeepSeek V3/R1/V3.2 checkpoints.
  - **Block-scaled GEMM with packed UE8M0 scales** (same three GEMM shapes,
    including the PSUM contiguous layout): **FP4** (E2M1, 32-element scales)
    and **FP8** (32- or 128-element scales). They run on the hardware
    block-scaled `mma.sync` (`kind::mxf4nvf4` / `kind::mxf8f6f4`), so the
    tensor core applies the scale directly instead of a CUDA-core promote
    step.
  - **BF16 GEMMs** on the same kernel without scales: normal, m-grouped
    contiguous (incl. PSUM) and m-grouped masked, with K-major or MN-major
    operands (`bf16_gemm_nn/tn/tt`), plus `k_grouped_bf16_gemm_tn_contiguous`.
    Plain `bf16_gemm_nt` still goes to cuBLASLt by default.
  - Both GEMM kernels use **PDL**: weight loads of the first pipeline stages
    are issued before waiting on the previous kernel.
- **Speed vs TokenSpeed's SM120 fallbacks** (Triton and FlashInfer):
  - Dense decode (M <= 128) is 1.2-4.0x faster at TP8 shapes. Dense prefill
    (M >= 512) is on par with FlashInfer: 0.98-1.37x, except the N = 512
    shared gate_up (up to 2.6x).
  - The DSA indexer is 3.4-19x faster for decode and 7.8-52x faster for
    prefill.
  - FP4 is a further 1.3-2.0x faster than the best FP8 kernel at nearly every
    dense shape, and 1.07-9.1x faster than the best Triton/FlashInfer
    block-FP8 kernel.
  - End to end on a block-FP8 27B model, TPOT is 1.18-1.55x better.
- **FP4 vs FlashInfer's MXFP4** (the faster of its `cudnn` and `b12x`
  backends, the two that run MXFP4 on SM120): after the cost-model refit
  (`1b065a2`) DeepGEMM wins 59 of 63 shapes, 1.33x geomean; the other four are
  ties at 0.96-0.99x. See section 7.5.
- **Also on SM120** (`eff4cea`..`38ef2a2`): NVFP4 (UE4M3 16-element scales);
  BF16/FP4 `alpha`, C accumulation and FP32 output; `tf32_hc_prenorm_gemm`;
  `einsum bmk,bnk->mn` and `fp8_einsum bhr,hdr->bhd`; MQA logits head dim 64
  (prefill). Since then: MN-major BF16 operands, native BF16 einsum
  (`bhr,hdr->bhd`, `bhd,hdr->bhr`), k-grouped BF16, and a packed-UE8M0 / FP32
  scale pair on the block-scaled GEMMs (sections 3.4 and 5). An interim FP4-only cost-model fit regressed UE8M0 FP8 at large M;
  fitting all modes jointly restored parity (dense gate_up M=4096: 406 vs 402
  us). See sections 7.5-7.6.
- **Not supported on SM120:** k-grouped FP8 GEMMs and MN-major FP8 operands
  (including `fp8_einsum bhd,hdr->bhr`). They need a byte-granular
  transpose-on-load, and `ldmatrix.b8` doesn't exist on sm_120. The host
  rejects them with a clear error (section 9).
- **Integration:** TokenSpeed selects DeepGEMM on SM120 only when the
  installed build has SM120 kernels. It carries DeepGEMM as a submodule, and
  both `scripts/install.sh` and the UnieAI Docker image build it from source.
  The install auto-detects SM120 and builds from source there with no flag;
  on other GPUs it keeps the published wheel. TokenSpeed calls the FP8
  block-scaled GEMM, the MXFP4 linear and SwiGLU leaves, and the mHC kernels
  (section 8).
- **Test status** (GPU 7, `DG_JIT_CHECK_NO_SPILLS=1`): `test_fp8_fp4`,
  `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection` and
  `test_attention` pass. `test_lazy_init` passes on 2 GPUs. The sanitizer
  reports 0 errors across 21 runs. Three failures are environment-only:
  `test_locality_domain` (MLOPart), `test_mega_moe` (needs NVLink) and the
  sanitizer's `test_legacy` entry (no triton in the venv).

| Repo | Branch | Commits | State |
|---|---|---|---|
| DeepGEMM (`https://github.com/nctu6/DeepGEMM`) | `nctu6` | `970b868` kernels, host code, tests, scripts; `3070f4c` setup.py header fallback; `e350464`, `49f3fe1`, `b5adf0e`, `f6bd900` block-scaled FP4 / UE8M0 FP8 GEMMs; `b419c02` PDL, narrow tiles, 8 warps, stage staging, 64-byte FP4 K, PSUM, BF16; `b19d5e8` sweep / fit / check scripts; `eff4cea` joint-mode refit, `eb61610`+`4659181` NVFP4 + alpha/accumulate/FP32, `bfb637c` tf32 hc, `5dfd734` einsum, `38ef2a2` MQA head dim 64; `1b065a2` cost-model refit; `74e1f61` MN-major BF16, native einsum, k-grouped BF16; `9704db0` BF16 / einsum tests on SM120; `4a95a00` stream-K range after `setmaxnreg`; `a765e73` mixed UE8M0 / FP32 scale pair; `9f19e0a` 1D2D SFA row offsets | `nctu6` head: `e7c0d33` (docs) |
| TokenSpeed | `nctu6` | `1c7a96a73` kernel integration; `0623d1552` PDL sync; `9617cda2c` submodule + install.sh + Docker; `b4cf963fc` SM120 auto-detect in install; `f798e68b4`, `f07f3cc54`, `8222224b8`, `63aaaf957` MXFP4 leaves; `5a532c016` mHC on SM120; `8a33e3341` SM100-only leaves capped; `019b65c04`, `469fc9c4a` NVFP4 fixes | `nctu6` head: `f2f6310aa` |

Test machine: aws-raichu, 8x RTX PRO 6000 Blackwell Server Edition (96 GB,
188 SMs), CUDA 13.2, driver 595.99.02, PCIe only (no NVLink). All tests and
benchmarks ran on GPU 7 only.

## 2. Why upstream DeepGEMM can't run on SM120

**Missing hardware.** Every upstream kernel uses tensor-core hardware that
SM120 doesn't have:

| Feature | SM90 kernels | SM100 kernels | SM120 |
|---|---|---|---|
| `wgmma.mma_async` | all | - | no |
| `tcgen05` / TMEM / UMMA / 2-CTA | - | all | no |
| `mma.sync kind::f8f6f4 m16n8k32` (E4M3) | - | - | yes |
| `mma.sync kind::mxf8f6f4.block_scale` (UE8M0) | - | - | yes |
| TMA, mbarrier, `ldmatrix`/`stmatrix`, `setmaxnreg` | yes | yes | yes |
| Shared memory per block | 227 KB | 227 KB | ~99 KB (101376 B opt-in) |

**Host dispatch.** The host code compares `arch_major == 9` or `== 10`
everywhere. Arch 12 either hit `DG_HOST_UNREACHABLE` (recipes, GEMM dispatch,
SF layout) or silently fell into SM90 choices (attention dtypes and block
sizes). The SM90/SM100 heuristics also hard-code 232448 B of shared memory.

**Toolchain.** No changes were needed. DeepJIT already compiles for
`sm_120f`, and CUTLASS 4.2.1 ships the SM120 MMA atoms.

One gotcha for standalone probes: `nvcc -c -arch=sm_120a` also embeds generic
`compute_120` PTX, so it rejects these instructions. Use `--cubin` or an
explicit `-gencode arch=compute_120a,code=sm_120a`.

So SM120 needs new kernels built on TMA + mbarrier pipelines + `mma.sync`,
plus explicit arch-12 host paths. Porting the SM90 or SM100 kernels isn't
possible.

## 3. FP8 GEMM kernel: `sm120_fp8_gemm_1d2d`

Files:
- `deep_gemm/include/deep_gemm/impls/sm120_fp8_gemm_1d2d.cuh` (device)
- `deep_gemm/include/deep_gemm/mma/sm120.cuh` (MMA and `ldmatrix` helpers)
- `csrc/jit_kernels/impls/sm120_fp8_gemm_1d2d.hpp` (launcher)
- `csrc/jit_kernels/heuristics/sm120.hpp` (`SM120ArchSpec`)

### 3.1 Contract

It has the same contract as `sm90_fp8_gemm_1d2d_impl`:
- A and B are E4M3, both K-major (NT). Output is BF16.
- SFA is FP32 per token per 128 channels. SFB is FP32 per 128x128 block.
- `BLOCK_K` is fixed at 128 (one scale per K block).
- GEMM types: normal, m-grouped contiguous (including psum layout), and
  m-grouped masked. All of them share the existing persistent tile scheduler
  `scheduler/gemm.cuh`, with multicast 1.

### 3.2 Structure

- **Warp specialization:**
  - One TMA warp group loads A and B tiles (128B swizzle) into a
    `kNumStages` shared-memory ring guarded by full/empty mbarriers.
  - Math warps (4 when `BLOCK_M <= 64`, 8 otherwise) form a
    `kNumWarpsM x kNumWarpsN` grid.
  - `setmaxnreg` gives the TMA warps 40 registers and the math warps 232-248.
    The stream-K iteration range is computed after the reconfiguration, in
    each role, so it isn't live across `setmaxnreg.dealloc` (`4a95a00`).
- **MMA:** `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
  - Operands come from shared memory through `ldmatrix.x4`, with addresses
    computed for the TMA 128B swizzle.
  - Each warp accumulates one 128-K block into a partial accumulator.
- **Promotion (1D2D scaling):** after each K block, the partial accumulator is
  folded into the FP32 final accumulator on CUDA cores, multiplied by
  `sfa[row] * sfb[n / 128]`.
  - `WARP_N` divides 128, so a warp sees exactly one SFB value per K block.
  - The partial plus final accumulators are capped at 64 + 64 FP32 per thread.
    A static assert rejects layouts that would spill.
- **SFA without staging or transpose:** math warps read SFA straight from
  global memory (`__ldg`) with runtime strides, prefetched one K block ahead.
  - Both the quantizer's K-major `[M, K/128]` scales and MN-major TMA-aligned
    scales work as-is.
  - Only the first SFA row stays live across the K loop. The clamped row
    offsets of the other rows are recomputed at each load (a few integer ops
    per 64 MMAs). Keeping all of them next to the 128 accumulators and the
    scale prefetch made ptxas spill 20 B in the masked GEMM (N=6144, K=7168,
    128x128, 256 threads, under `DG_JIT_CHECK_NO_SPILLS=1`). Now 0 B, with
    neutral performance (`9f19e0a`, section 6).
  - `transform_sf_into_required_layout` therefore passes FP32 SFA through on
    arch 12. This removed a per-call `transpose_fp32` kernel (~2.4 us plus a
    launch) and freed shared memory for a deeper pipeline.
- **Epilogue:** BF16 is stored straight from registers, so all ~99 KB of
  shared memory goes to the pipeline.
  - Lanes `t` and `t^1` swap halves with one `shfl.xor`. Each lane then
    stores 8 B, and each row writes a full 32 B sector.
  - This halved store requests and sectors (lg_throttle stall 0.88 -> 0.03).
  - Requires `ld_d % 4 == 0`.
- **Stream-K (opt-in, `DG_SM120_FORCE_STREAM_K=1`):** for normal GEMMs that
  fill the GPU poorly.
  - The `num_tiles x num_k_blocks` iteration space is split evenly across
    CTAs.
  - Each contributor writes an FP32 partial into its workspace slot. The last
    CTA to finish (tracked by self-resetting counters) sums the slots in a
    fixed order, so results are deterministic.
  - It's off by default because under CUDA graphs the data-parallel schedule
    won on almost every shape.
  - The counters are a plain `cudaMalloc` zeroed by a memset on the current
    stream, so first use inside a graph capture is safe. The FP32 partial
    workspace is persistent per stream, not allocated per call.
- **PDL (programmatic dependent launch):** before
  `cudaGridDependencySynchronize()`, the producer issues the weight (B) loads
  of the first `kNumStages` K blocks of its first tile, so a CTA resident
  while the previous kernel finishes starts with a full pipeline of weights.
  Only the activations wait. Normal data-parallel GEMMs only: grouped GEMMs
  must read `grouped_layout` first.
  - The next kernel is triggered by the producer when all K blocks fit the
    pipeline, otherwise by the math warps at the end of the mainloop.
  - Rejected: triggering at kernel start (slower everywhere), never
    triggering, and extra L2 prefetch beyond the pre-issued stages (neutral
    to worse; a whole-tile prefetch thrashed L2).
- **Grid:** normal GEMMs launch `min(SMs, tiles)` CTAs instead of a full
  persistent grid. Idle CTAs still pay for dispatch, and under PDL they sit
  between this kernel and the next.

### 3.3 Heuristic (`SM120ArchSpec`)

- **Candidates:**
  - `BLOCK_M` in {16, 32, 64, 128}, limited by the expected M. Grouped
    contiguous uses the layout's M alignment.
  - `BLOCK_N` in {64, 128, 256}, plus 32 for the block-scaled / BF16 kernel
    (section 3.4).
  - The layout must allow a legal warp tiling and at least 2 pipeline stages.
- **Stages:** `(smem capacity - barriers) / bytes per stage`, capped at 16.
  Shared-memory capacity is read from the device, not hard-coded.
- **Cost model:** for each wave, the max of the DRAM share, the L2 share, and
  CTA time plus a fixed overhead. It was fitted with
  `scripts/sm120_fit_heuristic.py` on round-robin CUDA-graph sweeps
  (`scripts/sm120_sweep_graph.py`, 15 shapes x 13 M values).
  - Geomean regret vs the best forced layout: 1.177 -> 1.006.
  - Worst case: 3.2x -> 1.3x.
- The block-scaled / BF16 kernel has its own fitted model (section 3.4).
- **Override for sweeps:** `DG_SM120_FORCE_LAYOUT=<BMxBN>` or `<BMxBNxBK>`,
  and `DG_SM120_FORCE_STREAM_K` (0 never, 1 when the rule says it is a clear
  win, 2 whenever possible). Developer use only.

Tile findings:
- 128x128 with 8 math warps (64x32 warp tiles) wins at large M.
- Small tiles win for single-wave decode shapes, which are bound by per-CTA
  load latency.
- 128x256 and 256x128 tiles aren't possible with FP32 promotion: a 64x64 warp
  tile needs 128 accumulators plus 128 promote registers, which spills.

### 3.4 Block-scaled GEMM: `sm120_fp8_fp4_gemm_1d1d`

Files: `deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh`,
`csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp`, and the
`FP4BlockScaledMMA` / `FP8BlockScaledMMA` wrappers in
`deep_gemm/include/deep_gemm/mma/sm120.cuh`.

**Contract.** Same as SM100's 1D1D path, so SM100 callers port unchanged.
A and B are K-major (BF16 also takes MN-major, below), output is BF16 or FP32. Scales are int32 tensors, four UE8M0
bytes per word (byte `2^(b-127)`), shaped `[(G,) MN, ceil(K / (gran_k * 4))]`,
MN-major and TMA-aligned. Modes:

| Mode | A/B | Scale granularity | MMA | Scale operand per MMA |
|---|---|---|---|---|
| FP4 | E2M1, two per byte | 32 elements | `kind::mxf4nvf4 ... scale_vec::2X m16n8k64` | 16 bits: byte 0 = K 0-31, byte 1 = K 32-63 |
| FP8 g32 | E4M3 | 32 elements | `kind::mxf8f6f4 ... scale_vec::1X m16n8k32` | one byte per MMA |
| FP8 g128 | E4M3 | 128 elements | same | one byte per K block |
| BF16 | BF16 | no scales | `m16n8k16.f32.bf16.bf16.f32` | - |

FP32 scales with `gran_k = 32` are packed to UE8M0 on the fly (the existing
`transform_sf_into_required_layout`, now allowed on arch 12). FP32 scales with
`gran_k = 128` stay on the FP32 kernel; hand over pre-packed int words to use
the 128-element UE8M0 mode. A pair with one packed-UE8M0 side and one FP32
side is accepted too (section 5).

GEMM types: normal, m-grouped contiguous (including the PSUM layout with
`expected_m_for_psum_layout`) and m-grouped masked, for every mode.

**MN-major BF16** (`74e1f61`). The kernel takes `kMajorA` / `kMajorB` template
parameters. An MN-major BF16 operand is TMA-loaded as 128B-swizzled atoms of
64 MN elements by 64 K rows, and read with `ldmatrix.x4.trans` into the same
fragments as the K-major path. A grouped MN-major B takes its group offset on
K, as in the SM90 BF16 kernel. The heuristic only gives MN-major operands tiles
whose MN dimension is a multiple of 64. Host limits: MN-major is BF16 only,
MN-major A has no groups, and a grouped MN-major B needs `K % 64 == 0` (TMA
bounds only the stacked `G*K` rows). This covers `bf16_gemm_nn/tn/tt` and
MN-major grouped B in deterministic mode (which skips cuBLASLt).

**Native einsum and k-grouped BF16** (`74e1f61`). `einsum bhr,hdr->bhd` and
`bhd,hdr->bhr` run one BF16 GEMM per head on strided views.
`k_grouped_bf16_gemm_tn_contiguous` runs one MN-major GEMM per group from the
host-side `ks`, as SM90 does. Both are checked against an FP32 reference
(all four layouts, BF16 and FP32 output, accumulation, K tails 8/136/1000;
diff < 1e-5).

**Register spills.** `DG_JIT_CHECK_NO_SPILLS=1` (set by
`tests/test_sanitizer.py`) rejects a kernel that spills. Two spills were
removed, both in the math or TMA warps after `setmaxnreg`:
- Stream-K range (`4a95a00`): the 64-bit iteration range was computed before
  the warp-role split, so it was live across `setmaxnreg.dealloc` (40
  registers for the TMA warps) and was spilled and reloaded in both roles.
  The BF16 16x64 stream-K tile (N=1024, K=4096) spilled 48 B. Each role now
  computes the range after its reconfiguration: 0 B, same results and decode
  timings. Applies to both SM120 GEMM kernels.
- 1D2D masked FP8 (`9f19e0a`): see section 3.2. The cause was the SFA row
  offsets, not `setmaxnreg`.

**Structure.** It reuses the FP32 kernel's TMA pipeline, scheduler, PDL
scheme and epilogue. `BLOCK_K` counts bytes of a tile row, which is exactly
one swizzle atom: 128 bytes (128 FP8, 256 FP4 or 64 BF16 elements, consumed
as four MMAs in every mode), or for FP4 also 64 bytes (128 elements, twice
the stages; it wins at large M). The differences:
- **No promote step.** The tensor core applies the scales, so there is a
  single accumulator set instead of a partial plus a final one.
- **Scales ride the pipeline stage.** Two extra TMA copies on the stage's full
  barrier land the packed words in per-stage shared memory; the math warps
  read them from there. A lane supplies the A scale of row `g` (`t == 0`) or
  `g + 8` (`t == 1`) and the B scale of column `g` (`t == 0`), where
  `g = lane / 4`, `t = lane % 4`. This was verified on hardware with
  `scripts/sm120_blockscale_probe.cu` and `sm120_fp4_probe.cu`.
- **Narrow tiles.** `BLOCK_N = 32` gives thin-N decode GEMMs twice the CTAs.
  Each K block's MMA steps are split over two K-warps so the math group stays
  a whole 128-thread warpgroup (`setmaxnreg` works per warpgroup); their
  accumulators are summed through shared memory.
- **8 math warps on 64x128 / 64x256** (a 2 x 4 grid of 32x32 warp tiles), as
  FlashInfer's b12x does. With 4 warps, ncu showed `stall_wait` at 2x b12x's.
- **Whole-stage register staging.** A math warp copies all fragments and
  scale words of a stage into registers and releases the stage *before* its
  MMAs, so the producer can refill every stage while one computes (64x128
  FP4: 0.93 -> 0.70 us per K block). Bitwise determinism verified over 200
  runs.
- **Stream-K** (same deterministic scheme as the FP32 kernel) is on by rule:
  only when the data-parallel grid fills at most 1/8 of the SMs and the split
  clearly shortens the per-CTA K loop. Mid-M splits were always slower (each
  CTA pays two pipeline fills).

**Heuristic.** A separate cost model for data-parallel layouts
(`SM120ArchSpec::get_1d1d_cost`). Each wave costs the max of its DRAM share
(inflated for small weight TMA boxes), its L2 share, and one CTA's K loop
(per-layout cost per K block, inflated for pipelines shallower than 6
stages). It was fitted with `scripts/sm120_fit_blockscale.py` on sweeps of
every layout and stream-K choice over the DeepSeek-V3 TP8 dense linears
(`scripts/sm120_sweep_blockscale.py`, CUDA graph, PDL, cold weights), jointly
for FP4, FP8 and NVFP4 so a gain in one mode can't hide a loss in another (an
FP4-only fit had regressed UE8M0 FP8 at large M). The latest refit (`1b065a2`)
re-swept FP4 and UE8M0 FP8: worst-mode geomean regret 1.027 -> 1.014 (worst
1.286 -> 1.135). It trades the steep depth penalty (`depth_exp` 1.58 -> 0.5)
for a higher CTA-cost scale (1.2 -> 1.5) and a recalibrated per-tile cost
table (64x128: 968 -> 476, 128x64: 900 -> 525), so mid-M shapes prefer the
64x128 and 128x64 tiles the sweep measures fastest. The FP32-kernel model is
near-optimal (regret 1.0081); a refit only reached 1.0077 and was not
committed.

A timing autotuner (first call per shape, cached) is available with
`DG_SM120_AUTOTUNE=1`. It is off by default: CUDA events resolve about 2 us
here, and eager timing can't see the PDL overlap with neighboring kernels.

**Why the scales are staged through TMA.** The first version loaded them with
plain global loads one K block ahead. A 128-byte FP4 K step is shorter than a
DRAM round trip, so it could not hide the latency: FP4 ran at 15.7 us on
q_a/kv_a (parity with FP8). With the loads stubbed out it ran in 8.7 us.
Staging them through TMA brought the real kernel to 9.5 us.

**Codegen note.** Block-scaled MMAs need arch-specific codegen
(`-gencode arch=compute_120a,code=sm_120a`, or `compute_120f`, which DeepJIT
uses). `nvcc -arch=sm_120a` still hands ptxas a generic `compute_120` target,
which rejects `.block_scale`.

## 4. DSA indexer kernels: MQA logits

Files:
- `deep_gemm/include/deep_gemm/impls/sm120_fp8_mqa_logits.cuh`
- `deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh`
- `csrc/jit_kernels/impls/sm120_fp8_mqa_logits.hpp`

They were derived from the SM90 sources, so the scheduling, TMA and epilogue
logic is unchanged. Only the MMA block is replaced, by
`mma::sm120::warp_mma_16xNx128_ss`, which keeps the SM90 WGMMA per-warp
accumulator layout. The SM90 paged metadata kernel
(`get_paged_mqa_logits_metadata`) is arch-agnostic and is reused unchanged.

| Kernel | Configuration |
|---|---|
| `fp8_mqa_logits` (prefill) | 8 math warps, `BLOCK_KV = 128` (16 KV rows per warp), 2 Q stages + 3 KV stages, fits 99 KB |
| `fp8_paged_mqa_logits` (decode) | 4 math warp groups, `SPLIT_KV = 256` (same as SM90, so the metadata kernel is reused), page size 64. Stages are chosen to fit shared memory from (3,3), (2,3), (2,2), (2,1); at least 2 Q stages are required |

- Supported: head dim 128, H a divisor of 128 (tested 32 and 64), next_n 1
  and 2.
- Correctness: `scripts/sm120_mqa_check.py` matches a dequantized reference to
  ~1e-14. It covers contiguous; paged with next_n 1/2, H 32/64, context up to
  20K, and B 32/64/128.

**API compatibility.** DeepGEMM 2.8.2 renamed the indexer APIs to
`fp8_fp4_mqa_logits` and `fp8_fp4_paged_mqa_logits`. TokenSpeed calls the old
names, so `deep_gemm/compat.py` provides `fp8_mqa_logits` and
`fp8_paged_mqa_logits` wrappers. `deep_gemm.get_supported_arch_majors()`
returns `(9, 10, 12)`, so callers can tell an SM120-capable build from the
pinned 2.8.0 wheel.

## 5. Host and API changes

- `csrc/utils/layout.hpp`: the arch-12 default recipe is (1, 128, 128) for
  FP32 SFB, and FP8 must be K-major. Packed FP4 operands are accepted on
  arch 12.
- `csrc/apis/layout.hpp`: FP32 (1, 128) SFA passes through on arch 12
  untouched. FP32 scales with `gran_k = 32` are packed to UE8M0 int words
  (as on SM100), and int (UE8M0) scales with `gran_k` 32 or 128 are accepted
  after a layout check. `transform_sf_pair_into_required_layout` also takes a
  **mixed pair** on arch 12: one side packed UE8M0 (int32, e.g. DeepEP
  activations) and the other FP32 (e.g. 128x128 weight scales). Per-side
  handling used to leave the FP32 128x128 SFB unpacked and fail the layout
  check in the UE8M0 kernel. The FP32 side is now cast to packed per-row UE8M0
  (`gran_k` 128), as SM100 does. Pure-FP32 pairs keep the FP32-scale kernel,
  and `disable_ue8m0_cast=True` rejects the mix instead of casting
  (`a765e73`). Normal and m-grouped contiguous GEMMs with an int SFA and an
  FP32 SFB match a dequantized reference (diff ~1e-6).
- `csrc/apis/gemm.hpp`: arch-12 branches for `fp8_fp4_gemm_nt`, m-grouped
  contiguous and m-grouped masked. FP32 scales go to the FP32-scale kernel,
  int scales to the block-scaled kernel. `bf16_gemm_nt` (when cuBLASLt is not
  used), `m_grouped_bf16_gemm_nt_contiguous` and
  `m_grouped_bf16_gemm_nt_masked` go to the 1D1D kernel in BF16 mode, now
  with `kMajorA` / `kMajorB` for `bf16_gemm_nn/tn/tt` and grouped MN-major B
  (BF16 only; MN-major A has no groups; a grouped MN-major B needs
  `K % 64 == 0`). `k_grouped_bf16_gemm_tn_contiguous` runs one MN-major GEMM
  per group. `k_grouped_fp8_gemm_tn_contiguous` and Wgrad 1D1D aren't provided
  (k-grouped FP8 raises `Unsupported architecture`).
- `csrc/apis/einsum.hpp`: arch-12 branches for `bhr,hdr->bhd` and
  `bhd,hdr->bhr` (one BF16 GEMM per head on strided views). `fp8_einsum`
  expressions that need MN-major FP8 operands (`bhd,hdr->bhr`,
  `bhd,bhr->hdr`) now say they need SM100.
- `csrc/jit_kernels/impls/runtime_utils.hpp`: `DG_TMA_L2_PROMOTION`
  (0/64/128/256, default 256) for TMA descriptor experiments; no measured
  effect.
- `csrc/apis/attention.hpp`: arch-12 dispatch for both MQA logits kernels.
- `tests/generators.py`, `tests/test_fp8_fp4.py`, `tests/test_attention.py`,
  `tests/test_bf16.py`, `tests/test_einsum.py`: arch-12 branches. Arch 12 is treated as SM90-like for FP8. Unsupported
  cases are skipped explicitly.
- `setup.py` (`3070f4c`): appends torch's pip `nvidia/*/include` directories
  after `CUDA_HOME`'s. torch's `CUDAContextLight.h` includes cuSPARSE and
  cuSOLVER headers, which trimmed CUDA toolkits (such as the TokenSpeed
  runtime image) lack.

## 6. Correctness bugs found and fixed

1. **Proxy fence race (GEMM and both MQA kernels).** Bitwise-determinism tests
   caught random wrong 8-column tiles with 8 math warps.
   - Cause: math warps read a stage with generic-proxy `ldmatrix`, then
     release it to TMA, whose writes are async-proxy. The reads need a
     `fence.proxy.async.shared::cta` before the empty-barrier arrive.
   - Without the fence, about 1 run in 20 was corrupted (7 in 20 for
     128x64). Moving the arrive, adding a `memory` clobber, or disabling
     `setmaxnreg` didn't help; the fence did. All layouts are now 0/20
     nondeterministic.
2. **Paged MQA hang at B >= 32.** Inserting the fence made a brace-less
   `if (q_iter_idx > 0) arrive();` arrive unconditionally. It's now braced.
3. **Paged MQA deadlock with next_n = 2.** To fit shared memory, the config
   picked a single Q stage. The TMA warp then waits for a free Q stage before
   issuing the KV loads the consumer is blocked on. Configs now require at
   least 2 Q stages.
4. **Split-K / stream-K counters under CUDA graphs.** Per-stream counters with
   a capture assert broke under `torch.cuda.graph`, which captures on its own
   stream. They're now a plain `cudaMalloc` plus a memset on the current
   stream.
5. **`einsum bmk,bnk->mn` segfault (all arches).** With FP32 D and no `c`, the
   host dereferenced an empty optional and crashed. It now asserts that `c`
   aliases `d` (`74e1f61`).
6. **Register spills after `setmaxnreg` (SM120 GEMMs).** Caught by
   `DG_JIT_CHECK_NO_SPILLS=1`. The stream-K range (48 B) and the 1D2D masked
   FP8 SFA row offsets (20 B) were live across the register reconfiguration
   or the K loop. Both are 0 B now (`4a95a00`, `9f19e0a`; sections 3.2, 3.4).
7. **Mixed UE8M0 / FP32 scale pair rejected on SM120** (`a765e73`, section 5).

Tests on GPU 7 (run 2026-11-04, on the tree committed as `b419c02`):
- `tests/test_fp8_fp4.py` passes: normal (forward and Dgrad), m-grouped
  contiguous (psum and non-psum), masked, and bitwise determinism.
- `scripts/sm120_mqa_check.py` passes.
- `scripts/sm120_blockscale_check.py --mode all --grouped --repeat N` passes.
  It covers FP4, FP8 g32, FP8 g128 and BF16; normal, contiguous, PSUM and
  masked; K tails; forced layouts (64x128x64, 16x32, 64x256); forced
  stream-K; and PDL on. The tolerance is 1e-7 against an exact dequantized
  reference.
- `tests/test_bf16.py` and `tests/test_einsum.py` used to divide by zero in
  the perf print on SM120, because they time kernels by SM90/SM100 names
  (the SM120 BF16 kernel is `sm120_fp8_fp4_gemm_1d1d_impl`, and its cuBLASLt
  uses CUTLASS SM75/SM80, `splitKreduce`, `gemvx` and dot kernels, not
  nvjet). `9704db0` fixes this: `bench_kineto(per_call_total=True)` sums all
  matching kernels per call, and the generators know the SM120 kernel names.
  The native einsum is compared to torch at 1e-7 (its `mma.sync` accumulation
  order rounds ~0.02% of elements to the neighboring BF16 value). Stochastic
  rounding and the MN-major `fp8_einsum` tests stay SM100-only.
- `tests/test_attention.py` passes on the cases that fit in memory. Some cases
  OOM on the shared GPU 7. Its kineto perf print needs
  `DG_USE_NVIDIA_TOOLS=1`.

Latest run (GPU 7, `DG_JIT_CHECK_NO_SPILLS=1`): `test_fp8_fp4`, `test_bf16`,
`test_einsum`, `test_layout`, `test_hyperconnection` and `test_attention` pass;
`test_lazy_init` passes on 2 GPUs; the sanitizer reports 0 errors across 21
runs. Environment-only failures: `test_locality_domain` (MLOPart),
`test_mega_moe` (NVLink) and the sanitizer's `test_legacy` entry (no triton in
the venv).

## 7. Performance

Method: CUDA graph, cold weights (L2 flushed between layers), timed at the
same call boundary as TokenSpeed (including SF transforms). Competitors run
through the `tokenspeed_kernel` registry with a forced solution. GPU 7 is
shared with another process, so comparisons are interleaved within a run and
repeated.

### 7.1 Kernel-level

- `mma.sync` E4M3 peak with FP32 accumulate is about 906 TFLOPS. Block-scale
  MMA runs at the same rate.
- Large-M DeepGEMM and FlashInfer both reach 730-750 TFLOPS (81-83% of peak).
  ncu shows ~85% tensor-pipe utilization for both, so the prefill gate was set
  to "no regression" rather than a 10% win.
- Decode GEMMs reach ~1.0 TB/s DRAM, the same as cuBLAS (spec 1.79 TB/s).

| shape (M x N x K) | DeepGEMM | cuBLASLt per-tensor FP8 (no block scales) |
|---|---|---|
| 4096 x 7168 x 16384 | 734 TFLOPS | ~765 TFLOPS |
| 4096 x 4096 x 7168 | 634 TFLOPS | ~690 TFLOPS |
| 4096 x 24576 x 1536 | 597 TFLOPS | ~630 TFLOPS |
| 1 x 7168 x 16384 | 114 us | slower (1.9 vs 2.1 TFLOPS) |

### 7.2 Dense linears vs TokenSpeed fallbacks (DeepSeek-V3, TP8 per-GPU shapes)

Ratio = best(Triton, FlashInfer) time / DeepGEMM time, so > 1 means DeepGEMM
is faster. FlashInfer is excluded where TokenSpeed wouldn't pick it
(17 <= M <= 32, or N % 128 != 0).

| linear (N x K) | M=1 | 8 | 32 | 64 | 128 | 256 | 512 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| q_a/kv_a 2112x7168 | 3.35 | 3.30 | 3.27 | 2.99 | 2.50 | 1.89 | 1.37 | 1.21 | 1.37 |
| q_b 3072x1536 | 2.65 | 2.05 | 2.00 | 1.91 | 1.63 | 1.19 | 1.04 | 1.00 | 1.00 |
| kv_b 4096x512 | 2.65 | 1.88 | 1.74 | 1.63 | 1.32 | 0.95 | 1.07 | 1.02 | 1.06 |
| o_proj 7168x2048 | 1.71 | 1.33 | 1.35 | 1.35 | 1.23 | 1.02 | 1.03 | 0.98 | 1.01 |
| shared gate_up 512x7168 | 3.64 | 3.93 | 3.93 | 3.95 | 3.93 | 3.01 | 2.56 | 1.36 | 1.00 |
| shared down 7168x256 | 3.27 | 2.10 | 1.63 | 1.35 | 1.08 | 1.21 | 1.08 | 1.06 | 1.02 |
| dense gate_up 4608x7168 | 2.28 | 1.96 | 1.97 | 1.91 | 1.77 | 1.35 | 1.00 | 1.01 | 1.01 |
| dense down 7168x2304 | 1.67 | 1.37 | 1.35 | 1.30 | 1.23 | 1.01 | 0.99 | 1.00 | 1.00 |

Measured 2026-11-04 on the current tree (`scripts/sm120_vs_tokenspeed.py
--tp 8`). This script leaves DeepGEMM's PDL off, so it doesn't exercise the
PDL pre-issue (TokenSpeed turns PDL on when serving). Compared with the
earlier run, decode ratios are equal or higher (q_a/kv_a M=1: 3.35 vs 2.89,
shared gate_up M=128: 3.93 vs 2.68). Prefill is unchanged at parity. kv_b
M=256 reads 0.95 here (1.03 before; 6.5 vs Triton's 6.1 us), within the
run-to-run noise of this shared GPU.

Over all 112 TP8 points (`sm120_vs_tokenspeed.py --tp 8`) the FP32-scale path
is 1.61x geomean. Seven points lose, all at 0.94-0.99x (q_b M=2048/4096, kv_b
M=256, o_proj M=1024, shared gate_up M=4096, shared down M=4096/8192). A
re-sweep and refit of the 1D2D model reached regret 1.0077 against 1.0081 and
didn't fix kv_b M=256 (auto 64x128 6.0 us vs best 128x64 5.7 us), so it wasn't
committed. The large-M losses have auto == best in the sweep, so they are
kernel-level parity, not tile choice. The `9f19e0a` spill fix keeps the masked
GEMM within +-0.4% on 20 of 26 configs and 0.8-2.5% faster on 6, none slower
beyond noise.

At TP1, decode is 0.98-3.35x (K = 512/1536 at M = 128 is at Triton parity)
and prefill is 0.95-1.36x.

### 7.3 DSA indexer vs Triton (H = 64, D = 128, page 64, top-k 2048)

Decode (`dsa_plan` + `dsa_decode_topk`), Triton time / DeepGEMM time:

| B x ctx | 1x4K | 8x4K | 32x4K | 128x4K | 1x16K | 32x16K | 128x16K | 1x64K | 32x64K | 1x128K | 8x128K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speedup | 3.4 | 5.4 | 10.7 | 13.4 | 6.0 | 15.2 | 19.3 | 8.1 | 17.2 | 9.2 | 14.7 |

Prefill (DeepGEMM logits + Triton radix top-k vs `triton_dsa_prefill_topk_fp8`):

| T x prefix | 1024x0 | 4096x0 | 4096x4096 | 2048x14336 | 8192x0 |
|---|---|---|---|---|---|
| speedup | 7.8 | 32.5 | 35.3 | 30.1 | 52.3 |

Top-k overlap with Triton is 0.97-1.00. The inputs are random, so near-ties
reorder; DeepGEMM's logits match the dequantized reference to 1e-14.

### 7.4 End to end

Model: Qwen3.8-27B-FP8 (block 128x128 FP32 scales, the same dense block-FP8
path as DeepSeek-V3). TokenSpeed TP1 on GPU 7, `scripts/sm120_e2e.sh`, two
runs each.

| Concurrency | TPOT before | TPOT after | Speedup | Output throughput |
|---|---|---|---|---|
| 1 | 31.9 ms | 20.6 ms | 1.55x | +53% |
| 8 | 32.4 ms | 22.4 ms | 1.45x | +43% |
| 32 | 33.5 ms | 28.4 ms | 1.18x | +16% |

- Long-prompt TTFT (6000 words): 795 -> 781 ms.
- All 260 block-FP8 `mm` selections resolve to
  `deepgemm_sm120_mm_fp8_blockscale`.
- Greedy outputs match the baseline (similarity 0.92-1.00, with only late
  divergence).

### 7.5 Block-scaled GEMMs (FP4 and UE8M0 FP8)

Same method as 7.2 (CUDA graph, cold weights, TP8 DeepSeek-V3 linears). Times
in us per GEMM; the scale layout is what a fused quantizer would emit, so only
the GEMM is timed.

**FP8 with UE8M0 scales** (g128, g32) was at parity with the FP32-scale
kernel (+-5%): same bytes and the same MMA rate. It is a coverage feature for
UE8M0 checkpoints, not a speedup. (An interim FP4-only cost-model fit made it
up to 1.8x slower at large M; fitting all modes jointly, `eff4cea`, restored
parity: dense gate_up M=4096 406 vs 402 us.)

**FP4 vs the best FP8 kernel and the Triton/FlashInfer block-FP8 fallbacks**
(`sm120_bench_blockscale.py --tp 8 --baselines`, 2026-11-04):

| linear (N x K) | M | Triton | FlashInfer | DeepGEMM FP32 | **DeepGEMM FP4** | FP4 vs best FP8 |
|---|---|---|---|---|---|---|
| q_a/kv_a 2112x7168 | 1 | 46.2 | n/a | 14.0 | **8.3** | 1.61x |
| q_a/kv_a 2112x7168 | 4096 | 237.8 | n/a | 176.3 | **88.1** | 2.00x |
| o_proj 7168x2048 | 1 | 37.2 | 21.8 | 12.9 | **8.2** | 1.52x |
| o_proj 7168x2048 | 4096 | 225.3 | 185.7 | 182.8 | **95.0** | 1.92x |
| shared gate_up 512x7168 | 1 | 43.1 | 55.0 | 11.9 | **5.2** | 1.63x |
| dense gate_up 4608x7168 | 1 | 58.9 | 56.1 | 24.6 | **14.3** | 1.72x |
| dense gate_up 4608x7168 | 64 | 47.9 | 56.0 | 25.1 | **14.5** | 1.74x |
| dense gate_up 4608x7168 | 4096 | 507.7 | 408.9 | 402.4 | **198.2** | 2.03x |
| dense down 7168x2304 | 4096 | 253.8 | 207.6 | 204.3 | **105.6** | 1.93x |
| shared down 7168x256 | 1 | 9.7 | 6.8 | 2.2 | **1.9** | 1.17x |

FP4 is 1.3-2.0x faster than the best FP8 kernel at nearly every shape. The
exception is `shared down` (K = 256, a single K block), where fixed
per-launch cost dominates: 0.92-1.48x, slower than FP8 at M=32 and M=2048.
This script runs with DeepGEMM's PDL off. The FlashInfer comparison below
turns PDL on for both sides, as serving does, so its DeepGEMM times are lower
(q_a/kv_a M=1: 6.8 vs 8.3 us).

**FP4 vs FlashInfer MXFP4.** FlashInfer has two backends that run MXFP4 on
SM120: `cudnn` and `b12x` (`cute-dsl`, `cutlass` and `trtllm` don't). The
comparison takes the faster one per shape, with PDL on both sides, implementations
interleaved, best of 5 rounds, GPU idle (`scripts/sm120_fp4_vs_flashinfer.py`).
Speedup = best FlashInfer time / DeepGEMM time:

| linear | M=1 | 8 | 32 | 64 | 128 | 256 | 512 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| q_a/kv_a 2112x7168 | 1.09 | 1.19 | 1.26 | 1.31 | 1.42 | 1.27 | 1.46 | 1.14 | 1.38 |
| q_b 3072x1536 | 2.19 | 2.20 | 2.04 | 1.86 | 1.20 | **0.91** | 1.37 | 1.43 | 1.36 |
| kv_b 4096x512 | 2.02 | 1.97 | 1.87 | 1.78 | 1.24 | 1.15 | 1.03 | 1.26 | 1.23 |
| o_proj 7168x2048 | 1.29 | 1.33 | 1.17 | 1.13 | **0.97** | 1.48 | 1.69 | 1.42 | 1.33 |
| shared gate_up 512x7168 | 1.39 | 1.40 | 1.07 | 1.21 | 1.49 | 1.28 | 1.25 | 1.39 | 1.37 |
| dense gate_up 4608x7168 | 1.23 | 1.25 | 1.51 | 1.53 | 1.16 | 1.00 | 1.37 | 1.20 | 1.35 |
| dense down 7168x2304 | 1.35 | 1.49 | 1.24 | 1.11 | 1.01 | 1.53 | 1.71 | 1.43 | 1.32 |

The table is the run before the cost-model refit, with DeepGEMM winning 60 of
63 points. After the refit (`1b065a2`, 63 DeepSeek-V3 TP8 points, GPU idle) the
geomean speedup rises from 1.315x to 1.33x at 59 of 63 wins. The four ties are
q_b M=256, o_proj M=128, dense gate_up M=256 and dense down M=128 (0.96-0.99x
vs b12x). Before the refit the first three were within 0.6 us of b12x (q_b
M=256: 6.0 vs 5.4 us; o_proj M=128: 7.3 vs 7.1; dense gate_up M=256: 20.0 vs
19.9), and the best forced layout in the sweep matched b12x (5.3-5.5 and
6.6-7.0 us) with the same CTA count. The analytic model has one cost per tile
shape, so it can't pick a tile per M, and `DG_SM120_AUTOTUNE=1` is worse
(geomean 1.318, q_b M=256 0.94x) because eager timing can't see the graph-mode
PDL overlap. A real margin needs kernel work.

What closed the earlier gaps (cuDNN used to win q_a/kv_a and shared gate_up
decode by up to 1.3x):
- **BLOCK_N = 32 tiles.** Thin-N decode gets twice the CTAs. Each K block's
  MMA steps are split over two K-warps, so the math group stays one 128-thread
  warpgroup for `setmaxnreg`. Their accumulators are summed through shared
  memory.
- **8 math warps on 64x128 / 64x256** (2 x 4 warps of 32x32), as b12x does.
- **Whole-stage register staging.** All fragments of a stage are loaded and
  the stage is released before its MMAs, so every stage refills while one
  computes (64x128 FP4: 0.93 -> 0.70 us per K block).
- **PDL weight pre-issue.** Before waiting on the previous kernel, the
  producer issues the weight (B, SFB) loads of the first `kNumStages` K blocks.
  Only the activations wait. The next kernel is triggered at the end of the
  mainloop, or after the last load when all K blocks fit the pipeline.
- **64-byte FP4 K blocks**: twice the stages, for large M.
- **Refitted cost model** (`get_1d1d_cost`, `sm120_fit_blockscale.py`), fitted
  on a 63-shape sweep of every layout (refit again in `1b065a2`, section 3.4).
  Within 1.2% of the best layout on average, 10.5% worst case in the first fit
  (the FP32-kernel model: 8.4% / 36%). Stream-K is
  an explicit rule: split only when the tiles fill at most 1/8 of the SMs and
  the per-CTA K loop clearly shrinks. That gives the N = 512 decode its
  4.6 us (cuDNN 6.4).

**BF16** uses the same 1D1D kernel without scales (`m16n8k16`). It covers
`bf16_gemm_nt` when cuBLASLt is not used (deterministic mode), plus the
m-grouped contiguous (incl. PSUM) and masked BF16 GEMMs, which had no SM120
path before. `alpha` and C accumulation are supported, with FP32 or BF16
output. MN-major operands, native einsum and k-grouped BF16 (`74e1f61`) are
correctness features (section 3.4), with no speed numbers recorded.

### 7.6 NVFP4 and other feature kernels

**NVFP4** (`scale_vec::4X`, UE4M3 per 16 elements) runs on the same 1D1D
kernel, selected by `gran_k == 16`. On the TP8 DeepSeek shapes it is 0-12%
slower than MXFP4 here, because it streams twice the scale bytes (one UE4M3
byte per 16 elements vs. one UE8M0 byte per 32), so MXFP4 stays the default
where a model allows it. Against FlashInfer's NVFP4 backends (`b12x`,
`cute-dsl`, `cudnn`, and cuBLASLt via `cublaslt_nvfp4_gemm_nt`), DeepGEMM wins
the decode and small-M shapes (up to 3.4x vs cuBLASLt on N=512) and is within
a few percent at mid/large M. Correctness is bit-exact against the dequantized
reference.

**`tf32_hc_prenorm_gemm`** (TF32 `m16n8k8`): the DeepSeek hyper-connection
pre-norm GEMM, `d = a @ b.T` plus the row squared-norm, with optional K-split.
It matches the TF32 PyTorch reference to < 1e-8 and reaches ~1.2 TB/s at large
M (it is memory-bound on the BF16 A operand; N is 24).

**einsum:** `bmk,bnk->mn` (BF16, a GEMM per batch slice accumulated into FP32
D), native BF16 `bhr,hdr->bhd` and `bhd,hdr->bhr` (a GEMM per head), and
`fp8_einsum bhr,hdr->bhd` (a per-batch FP32-scale GEMM) run on SM120. The
MN-major FP8 expressions are not supported (section 9).

**MQA logits head dim 64** (contiguous/prefill): one warp-MMA helper now takes
the head dim (128 or 64) with the matching TMA swizzle. Bit-exact against the
reference.

## 8. TokenSpeed integration

**Kernel registration** (`tokenspeed-kernel`, commit `1c7a96a73`):
- `thirdparty/deepgemm.py`: `deepgemm_supports_sm120()` probes the installed
  build through `get_supported_arch_majors()`. Every SM120 path is gated on
  it, so the pinned 2.8.0 wheel keeps the old behavior.
- `ops/gemm/deepgemm.py`: new `deepgemm_sm120_mm_fp8_blockscale`.
  - Arch 12.0-12.9, priority `SPECIALIZED`, trait `block_scale_layout =
    canonical_blackwell` (the label the `mm` facade uses on SM100+).
- `ops/gemm/flashinfer.py`: `flashinfer_mm_fp8_blockscale` caps its
  `max_arch_version` at 11.9 when DeepGEMM supports SM120.
  - Why not a higher priority: DeepGEMM's shape contract covers FlashInfer's,
    and an in-tree kernel can't go above FlashInfer's band without landing in
    the PLUGIN band.
- `ops/attention/dsa/deep_gemm.py`, `ops/attention/kpool/deep_gemm.py`,
  `ops/moe/deep_ep.py`: the DeepGEMM leaves are capped at 10.3 unless the
  build supports SM120.
  - DSA prefill on SM120 uses `triton_topk_from_logits`, because
    `torch.ops.trtllm.indexer_topk_prefill` has no SM120 kernel image.
- `ops/moe/_deep_gemm/ue8m0.py`: `deep_gemm_requires_ue8m0()` is now
  SM100-family only (`_UE8M0_MAX_ARCH = 10.9`). SM120 uses FP32 scales.
- Tests: `test/test_deepgemm_sm120_gating.py` and
  `test/nvidia/ops/gemm/test_sm120_deepgemm.py`. All 17 passed on GPU 7 at
  `1c7a96a73`.
- `0623d1552`: `thirdparty/deepgemm.py` `block_fp8_mm` now syncs
  `deep_gemm.set_pdl(pdl_enabled())` before each call, as the DSA / MoE
  leaves do. DeepGEMM's PDL setting is process-wide, so without this dense
  GEMMs could break the engine's PDL chain. The TokenSpeed kernel suite on
  SM120 is 7091 passed, 0 failed.

**MXFP4 leaves** (`f798e68b4`, `f07f3cc54`, `8222224b8`, `63aaaf957`). TokenSpeed
had no NVIDIA MXFP4 `mm` leaf (the Triton one is AMD-only), so dense MXFP4
linears had no path on SM120. All leaves are `SPECIALIZED`, SM120-gated and
auto-selected; they match the reference to ~1e-6.
- `deepgemm_sm120_mm_mxfp4` (op `mm`): packed E2M1 `a[M,K/2] @ b[N,K/2].T` with
  raw UE8M0 per-32 scales. It packs the scales per call with a transposing
  copy; callers that reuse a weight should use the linear leaf.
- `deepgemm_sm120_mxfp4_linear` (op `mxfp4_linear`, weight layout
  `deepgemm_sm120`): `prepare_mxfp4_linear` packs the weight scales **once at
  load** (int32 MN-major, four K-group bytes per word), and
  `Mxfp4LinearMethod` releases the checkpoint-order scale afterwards. Each
  call quantizes the activation with `mxfp4_quantize_mn_major_scales`, which
  writes the packed bytes and scales in the GEMM's layout, with the same
  numerics as `mxfp4_quantize`. Nothing is widened or repacked per call.
- `mxfp4_quantize_mn_major_scales` launches as a PDL producer for the GEMM
  (grids up to 1024 blocks): it triggers the dependent launch right after its
  own wait, so the GEMM prefetches weights while it runs. Larger grids launch
  as a plain kernel, because PDL-launched blocks that wait on the previous
  GEMM take its SMs (up to 18% slower at M=4096, K=7168).
- `deepgemm_sm120_mxfp4_swiglu_linear` (op `mxfp4_swiglu_linear`): the packed
  linear, then `silu_and_mul` over the stacked `[gate; up]` rows. Gated MLPs
  mark their gate_up projection `enable_swiglu`, and NVIDIA had no leaf for it.
- The linear leaves declare `n_align=64` and `k_align=128`, so selection
  rejects shapes the GEMM can't tile.
- Whole-op CUDA-graph benchmark (activation quantization included) against
  FlashInfer's quantize plus best `mm_fp4` backend, DeepSeek-V3 TP8 dense
  linears: the first version (per-call quantize and repack) lost on every
  shape (0.10-0.67x). Packed scales: faster on 52 of 63, ~1.15x geomean. With
  the PDL quantizer: **60 of 63, 1.23x geomean**. The three losses are `q_b`
  M=256 (0.90x, the GEMM itself is slower there) and M=1 of `o_proj` and
  `dense down` (0.97x).

**mHC** (`5a532c016`). The DeepGEMM mHC pre/mixes leaves were capped at SM103
because `tf32_hc_prenorm_gemm` had no SM120 kernel. When the installed build
reports SM120 kernels the cap is lifted; otherwise SM120 falls back to Triton.
On RTX PRO 6000 (CUDA graph), DeepGEMM vs Triton: `mhc_pre` 1.5-13.6x faster,
`mhc_mixes` 1.6-15x, with TF32-class error like SM100 (`mhc_mixes` within
3.5e-4 of Triton).

**SM100-only leaves capped off SM120** (`8a33e3341`). These outranked the
portable Triton leaves but declared no upper arch bound, so auto selection
picked them on SM120 and crashed: flashinfer_trtllm DSA decode/prefill (cap
10.7), FlashMLA DSA and DSV4 decode/prefill (cap 10.9), `tokenspeed_mla` decode
(cap 10.7), and `fast_hadamard_transform` (no SM120 cubin; cap 10.3). They are
capped by arch window, not ported.
`test_sm120_excludes_sm100_only_leaves` checks against the real registrations
that none is offered to an SM120 platform and Triton still is. The DeepGEMM DSA
top-k leaves also use a stable-sort fallback for batch-invariant selection
where flashinfer's FilteredTopK can't run (99 KiB shared memory; `a6d0f1682`).

**NVFP4 fixes.** `019b65c04`: a checkpoint with one GEMM1 input scale for all
experts (Qwen3.5-122B-A10B-NVFP4) loads it with shape `(1,)`, but
`cutlass_fused_moe` accepts only a 0-d scalar or one value per local expert, so
every layer with more than one local expert failed in warmup. A single-element
scale is now passed as 0-d. `469fc9c4a`: `fp4_quantize` and `mm_fp4` were
imported only for arch major 10, so the first dense NVFP4 linear of
Qwen3.8-27B-NVFP4 raised "Kernel implementation not found" on SM120. FlashInfer
supports both there.

**End to end on SM120.** Correct output (a factual prompt and a short
reasoning prompt) on Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2),
Qwen3.8-27B-NVFP4 and gemma-4-31B, run in-process with `Engine`.

**Build and install** (commits `9617cda2c`, `b4cf963fc`). This is automatic:
the user passes no flag, and the install picks the right DeepGEMM for the
hardware.
- `DeepGEMM/` submodule: url `https://github.com/nctu6/DeepGEMM`, branch
  `nctu6`, pinned at `de8cef7` (`e01367912`: adds the sparse and block-scaled
  MQA logits). It doesn't yet include `74e1f61`..`9f19e0a`.
- `scripts/install.sh`:
  - **Auto-detects SM120** from the device compute capability (major 12),
    read from `nvidia-smi --query-gpu=compute_cap`, with a torch fallback once
    the venv is resolved (for hosts without the CLI).
  - On SM120, it initializes the submodule recursively (cutlass, deep_jit),
    only when unpopulated, and builds DeepGEMM from source. On any other GPU,
    or when no GPU is visible, it keeps the published wheel and skips the
    submodule init entirely, since the source build only helps SM120.
  - Builds DeepGEMM after the kernel steps, because those re-pull the pinned
    wheel.
  - Uninstalls `tokenspeed-deepgemm`, then runs `pip install ./DeepGEMM
    --no-build-isolation --no-deps` with `DG_FORCE_BUILD=1`, so it never
    downloads the upstream prebuilt wheel.
  - Adds `libdw-dev` to the system deps when building DeepGEMM.
  - Verifies the result through package metadata.
  - Rebuilds on a fresh install, when `deep_gemm` is missing, or when
    `tokenspeed-deepgemm` comes back. `--force-deepgemm` /
    `REINSTALL_DEEPGEMM=1` / `FORCE_SM120=1` force the source build on any GPU
    (for debugging or an off-box SM120 image build).
- `docker/Dockerfile.nvidia.unieai`: adds `libdw-dev`, the DeepGEMM build step
  with `DG_FORCE_BUILD=1`, and a sanity check that the published wheel is gone
  and the SM120 API is present. The image targets SM120 and the kernels JIT at
  runtime (arch 9/10/12), so it always builds from source when present; no
  build-time hardware detection is needed.
- `docker/unieai.nvidia.sh`: populates `smg/` and `DeepGEMM/` when they're
  empty, and prints the DeepGEMM commit. The DeepGEMM init is non-fatal: if it
  can't be populated (offline, or not a git checkout), the image keeps the
  published wheel instead of failing the build.

Verified on aws-raichu, using a scratch checkout and a relocated copy of the
TokenSpeed venv so the live venv was untouched:
- Plain `./scripts/install.sh` on GPU 7 (compute_cap 12.0): auto-detected
  SM120, replaced the wheel, and verified the source build. No flag.
- A stub `nvidia-smi` reporting 9.0: kept the wheel, built nothing, and
  skipped the submodule init.
- `--force-deepgemm` with the stub still active: forced the source build.
- The Dockerfile DeepGEMM step on top of `nctu6/tokenspeed:e27cee86-20261028`:
  builds, the sanity check passes, and `fp8_gemm_nt` inside the container on
  GPU 7 gives calc_diff 1.4e-6.
- `unieai.nvidia.sh` submodule logic with a stub `docker`.

A full image build was not run.

## 9. Limitations

**Added since the first report (SM120 now supports):** NVFP4 (UE4M3,
16-element scales); BF16/FP4 `alpha`, C accumulation and FP32 output;
`tf32_hc_prenorm_gemm`; `einsum bmk,bnk->mn` and `fp8_einsum bhr,hdr->bhd`;
MN-major BF16 operands, native BF16 einsum (`bhr,hdr->bhd`, `bhd,hdr->bhr`)
and `k_grouped_bf16_gemm_tn_contiguous`; a mixed packed-UE8M0 / FP32 scale
pair; and MQA logits head dim 64 (contiguous/prefill); **sparse MQA logits**
(`fp8_fp4_sparse_mqa_logits` + paged + the two metadata entry points, MXFP8 and
MXFP4, contiguous aligned/unaligned ks and paged); and **block-scaled
MXFP8/MXFP4 dense MQA logits** (the MX path of `fp8_fp4_mqa_logits` for arch 12,
matching SM100). Both validated against a dequantized reference, diff ~1e-6.
See section 7.

**Not implemented on SM120.** The host dispatch still rejects these
(`DG_HOST_UNREACHABLE` or an arch assert); none has a silent wrong path:
- FP8-output (quantized D) epilogues. Block-scaled GEMMs need K to be a
  multiple of the scale granularity, and BF16/FP32 output.
- Custom epilogue classes (stochastic rounding, FP8 cast): SM100 only.
- **k-grouped FP8 GEMMs** (`k_grouped_fp8_gemm_tn_contiguous`) and Wgrad, and
  **MN-major FP8/FP4 operands**, including `fp8_einsum bhd,hdr->bhr` and
  `bhd,bhr->hdr`. An MN-major FP8 operand needs a byte-granular
  transpose-on-load, and `ldmatrix.b8` doesn't exist on sm_120. The BF16 path
  works because the 16-bit `ldmatrix.trans` exists (section 3.4). The host
  raises a clear error (`Unsupported architecture` for k-grouped; "needs
  MN-major FP8 operands, supported on SM100 only" for the einsum). Training
  only, with no inference caller.
- Paged MQA with head dim != 128.
- mega MoE/gate/mHC. MoE's point is NVLink symmetric-memory dispatch across
  ranks (pointless on one PCIe GPU); gate/mHC are TMEM/TF32-pipeline rewrites.
  (The DeepGEMM mHC pre/mixes kernels built on `tf32_hc_prenorm_gemm` do run,
  section 8.)

**Implemented on SM120 but unused by TokenSpeed on this box.** The m-grouped
contiguous (including psum) and masked FP8 GEMMs have real arch-12 branches in
`csrc/apis/gemm.hpp` and pass `tests/test_fp8_fp4.py`. TokenSpeed only reaches
them through the DeepEP MoE path, which needs NVLink or RDMA; this box is PCIe
only, so nothing calls them here. The kernels are correct and fast
(contiguous 680-757 TFLOPS, masked up to 1.37 TB/s). The block-scaled
(NVFP4 / UE8M0 FP8) and BF16 GEMMs and the einsum paths are also unused:
TokenSpeed's BF16 and einsum paths don't go through DeepGEMM on SM120.
MXFP4 (the `deepgemm_sm120_mm_mxfp4`, `_mxfp4_linear` and `_mxfp4_swiglu_linear`
leaves) and `tf32_hc_prenorm_gemm` (mHC) are used, see section 8.

Other notes:
- FP4 beats FlashInfer's MXFP4 backends on 59 of 63 points; the other four
  are ties at 0.96-0.99x (section 7.5).
- The V3.2 sparse indexer on SM120 is blocked upstream of DeepGEMM. Its
  kernels work, but TokenSpeed's native DSV4.1 indexer gates on
  `platform.is_blackwell` (arch major 10 only), and the top-k DeepSelect half
  needs ~207 KB of shared memory against SM120's 99 KB (a launch fails with
  `CUDA_ERROR_INVALID_CONFIG`). Enabling it needs an SM120 smem-reduced
  DeepSelect port in `tokenspeed-kernel` and flipping those gates to
  `is_blackwell_plus`.
- Call each GEMM once before CUDA-graph capture, so its counters are
  allocated. TokenSpeed warmup does this.
- TokenSpeed's `AGENTS.md` lists NVIDIA support as sm90/sm100/sm103/sm107.
  SM120 is a fork-side effort, so upstreaming needs that scope to change.

## 10. Open items

1. **Four FP4 ties with FlashInfer's b12x at mid M** (q_b M=256, o_proj M=128,
   dense gate_up M=256, dense down M=128; 0.96-0.99x). Tile choice and the
   cost-model refit are exhausted. Also tried and dropped: two CTAs per SM
   (correct, no gain), stream-K at mid M (slower) and `DG_SM120_AUTOTUNE=1`
   (worse). Needs kernel-level work.
2. **V3.2 sparse indexer in TokenSpeed on SM120**: an SM120 smem-reduced
   DeepSelect port and the `is_blackwell` gates (section 9). Out of DeepGEMM
   scope.
3. DeepSeek-V3/V3.2 TP8 end-to-end run. It needs GPUs 0-7, and only GPU 7 was
   available.
4. Push the DeepGEMM and TokenSpeed commits, then bump the `DeepGEMM`
   submodule pointer from `de8cef7` to include `74e1f61`..`9f19e0a`.
5. Deferred, low value here: k-grouped FP8 / Wgrad and MN-major FP8 operands
   and einsum (`ldmatrix.b8` doesn't exist on sm_120; section 9), and mega
   MoE/gate (NVLink or TMEM/TF32-pipeline rewrites).
6. Optional: publish a `tokenspeed-deepgemm` wheel from this fork. No longer
   required, since the source build replaces the wheel.

## 11. Reproducing

All commands run on aws-raichu with `CUDA_VISIBLE_DEVICES=7` and
`CUDA_HOME=/usr/local/cuda`.

- Build and test DeepGEMM:

  ```bash
  scripts/sm120_remote.sh build
  python tests/test_fp8_fp4.py
  python scripts/sm120_mqa_check.py
  ```

- Benchmarks vs TokenSpeed (run from the TokenSpeed venv):

  ```bash
  python scripts/sm120_vs_tokenspeed.py       # dense
  python scripts/sm120_dsa_vs_tokenspeed.py   # DSA indexer
  ```

- Block-scaled GEMMs (FP4, UE8M0 FP8):

  ```bash
  python scripts/sm120_blockscale_check.py --mode all --grouped --repeat 20  # correctness, incl. BF16
  sm120_remote.sh ts -m sm120_bench_blockscale --tp 8 --baselines  # vs FP32 / Triton / FlashInfer
  sm120_remote.sh ts -m sm120_fp4_vs_flashinfer                    # FP4 vs FlashInfer cudnn / b12x MXFP4
  ```

- Heuristic refit, FP32-scale kernel:

  ```bash
  python scripts/sm120_sweep_graph.py --repeats 5 --json
  python scripts/sm120_fit_heuristic.py
  ```

- Heuristic refit, block-scaled / BF16 kernel (later JSON files replace
  earlier records of the same shape):

  ```bash
  sm120_remote.sh ts -m sm120_sweep_blockscale --out sweep.json [--ms 128,256 --repeats 5]
  python scripts/sm120_fit_blockscale.py sweep.json [more.json ...] --verbose
  ```

- End to end:

  ```bash
  scripts/sm120_e2e.sh base|dg <model> <port> <out>
  ```

- TokenSpeed install, from the repo root. On an SM120 GPU this builds
  DeepGEMM from source automatically; no flag is needed:

  ```bash
  ./scripts/install.sh
  ```

  Force the source build on any GPU (or an off-box SM120 image build) with:

  ```bash
  ./scripts/install.sh --force-deepgemm   # or FORCE_SM120=1 ./scripts/install.sh
  ```
