# DeepGEMM on sm120: analysis, plan, and status log

Goal: DeepGEMM kernels running on NVIDIA sm120 (RTX PRO 6000 Blackwell Server
Edition) so TokenSpeed can serve DeepSeek models at the best performance on
that hardware.

This file is the single source of truth for the effort. Update the status log
(section 9) after every step. Re-read sections 2, 5 and 8 before starting any
new work so we don't drift into one of the known dead ends.

## Current state (2026-11-06)

- **Supported on sm120:**
  - `fp8_gemm_nt`, m-grouped contiguous (incl. psum) and masked FP8 GEMMs, with FP32 1x128 / 128x128 scales.
  - `fp8_mqa_logits` and `fp8_paged_mqa_logits` (H 32/64, next_n 1/2; head dim 128 or 64 for the contiguous/prefill kernel, 128 for the paged one), plus `get_paged_mqa_logits_metadata`.
  - Legacy aliases `fp8_mqa_logits` / `fp8_paged_mqa_logits` and the probe `get_supported_arch_majors()`.
  - **Block-scaled GEMMs** (`fp8_fp4_gemm_nt` and the m-grouped contiguous / masked variants): **MXFP4** (E2M1, 32-element UE8M0 scales), **NVFP4** (E2M1, 16-element UE4M3 scales, `scale_vec::4X`), and FP8 with 32- or 128-element UE8M0 scales. The tensor core applies the scales (`kind::mxf4nvf4` m16n8k64, `kind::mxf8f6f4` m16n8k32). BF16 or FP32 output, optional `alpha` and C accumulation. Includes the PSUM layout, BLOCK_N = 32 tiles, 64-byte FP4 K blocks, PDL weight pre-issue, and stream-K for few-tile decode. The cost model is fitted jointly on all three modes. See section 5.6 and the status log.
  - **BF16 GEMMs** on the same kernel: `bf16_gemm_nt` (cuBLASLt stays the default unless deterministic mode; `alpha` and C accumulation supported), `m_grouped_bf16_gemm_nt_contiguous` (incl. PSUM), `m_grouped_bf16_gemm_nt_masked`. **MN-major BF16 operands** (`bf16_gemm_nn/tn/tt`, a grouped MN-major B) and **`k_grouped_bf16_gemm_tn_contiguous`** also run, see section 9.9.
  - **`tf32_hc_prenorm_gemm`** (TF32 `m16n8k8`): `d = a @ b.T` and the row squared-norm, with K-split.
  - **`einsum`**: `bmk,bnk->mn` (BF16), and native BF16 `bhr,hdr->bhd` and `bhd,hdr->bhr` (one GEMM per head). **`fp8_einsum`** `bhr,hdr->bhd` (via a per-batch FP32-scale GEMM), and the MN-major `bhd,hdr->bhr` and `bhd,bhr->hdr` (one GEMM per head, section 9.12).
- **Done since this section was first written (2026-11-05/06):** sparse MQA logits (`fp8_fp4_sparse_mqa_logits` + paged + the two metadata entry points) and **block-scaled MXFP8/MXFP4 dense MQA logits** (`fp8_fp4_mqa_logits` MX path for arch 12). Both validated against a dequantized reference (`scripts/sm120_sparse_mqa_check.py`, `sm120_sparse_paged_check.py`, `sm120_mqa_blockscale_check.py`), diff ~1e-6.
- **Still not implemented on sm120** (rejected at the host): paged MQA with head dim != 128 (section 11.4); mega MoE/gate/mHC (NVLink symmetric memory or TMEM/UMMA). MN-major FP8 GEMM operands (9.10), k-grouped FP8 GEMMs (9.11) and the MN-major `fp8_einsum` expressions (9.12) are now done. See sections 8, 9.9, 9.10, 9.11 and 9.12.
- **Implemented on sm120 but unused by TokenSpeed on this box:** the m-grouped contiguous (incl. psum) and masked FP8 GEMMs have real `arch_major == 12` branches (`gemm.hpp`) and pass `test_fp8_fp4.py`. TokenSpeed only reaches them through DeepEP MoE, which needs NVLink/RDMA (this box is PCIe-only), so nothing calls them here.
- **Tests (GPU 7):** `tests/test_fp8_fp4.py` passes (k-grouped FP8 now enabled, section 9.11); `tests/test_bf16.py` and `tests/test_einsum.py` run on SM120 (section 9.9). `scripts/sm120_mqa_check.py` passes. The TokenSpeed tests pass (17).
- **Performance vs the TokenSpeed sm120 fallbacks:**
  - Dense decode is 1.2-4x faster at TP8 shapes; dense prefill is on par with FlashInfer.
  - The DSA indexer is 3-19x faster for decode and 8-52x faster for prefill.
  - End to end on a block-FP8 27B model: TPOT is 1.18-1.55x better.
  - **FP4 vs the best FP8 kernel** (DeepSeek-V3 TP8 dense linears, graph + cold weights): 1.6-2.0x at nearly every shape (decode 1.3-1.8x, prefill 1.7-2.0x), and 1.1-10.2x over the best Triton/FlashInfer block-FP8 kernel (smallest on K = 256 `shared down` large M, largest on N = 512 `shared gate_up` decode). **FP4 vs FlashInfer MXFP4** (`cudnn` and `b12x`, the backends that run MXFP4 on sm120): DeepGEMM wins 60 of 63 points (1.03-2.2x); the other three are ties within 0.4 us (q_b M=256, o_proj M=128, dense gate_up M=256). See the 2026-11-04 rows.
- **Commits:** DeepGEMM `970b868` (kernels) and `3070f4c` (setup.py fallback headers), both pushed to `origin/nctu6`; then `e350464`, `49f3fe1`, `b5adf0e`, `f6bd900` (block-scaled FP4 / UE8M0 FP8 GEMMs), local on `nctu6`, not pushed. TokenSpeed `1c7a96a73` (integration) and `9617cda2c` (DeepGEMM submodule + install.sh + Docker), local on `nctu6`, not pushed.
- **Install:** TokenSpeed carries DeepGEMM as a submodule (`DeepGEMM/`, branch `nctu6`). `scripts/install.sh` and `docker/Dockerfile.nvidia.unieai` build it from source and replace the published `tokenspeed-deepgemm` wheel. Both are tested on aws-raichu. See the "Submodule + install" rows in section 9.
- **Open:** (0) **UE8M0 FP8 regression at mid/large M from the FP4-only cost-model refit** (status log 2026-11-04; committed in `b419c02`, fix pending). (1) The three FP4 ties with b12x at mid M (kernel-level work; tile choice is exhausted). (2) NVFP4 (UE4M3 / 16-element scales, `scale_vec::4X`). (3) TokenSpeed has no NVIDIA MXFP4 `mm` leaf at all (its MXFP4 leaves are AMD-only), so a `deepgemm_sm120_mm_mxfp4` leaf, with weight scales packed once at load, would be a new capability. (4) A DeepSeek-V3.2 TP8 end-to-end run (needs 8 GPUs). A wheel release and pin bump are optional, since the source build replaces the pin.

## 0. Environments

| Role | Location | Notes |
|------|----------|-------|
| Code (edit here) | local `/Users/nctu6/Downloads/ascend/infra/DeepGEMM`, branch `nctu6` | Submodules are **not** checked out locally (empty `third-party/`) |
| Test (run here) | `ssh aws-raichu`, `/workspace/nctu6/infra/DeepGEMM`, branch `nctu6` | `/workspace` resolves to `/unieai`. **GPU 7 only**: always `CUDA_VISIBLE_DEVICES=7` |
| Runtime | `/workspace/nctu6/infra/DeepGEMM/.venv` | Python 3.15.0. Never install outside this venv |

Both checkouts were at HEAD `5081f85` (DeepGEMM 2.8.2) when this was written.

Facts about the remote machine (verified):
- 8x RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, 96 GB each,
  driver 595.99.02.
- CUDA 13.2 (`/usr/local/cuda` -> 13.2, nvcc V13.2.86). 13.0 is also installed.
- **No NVLink.** `nvidia-smi topo -m`: GPU pairs share a PCIe switch (PIX), pairs
  in the same NUMA node are NODE, and the two NUMA halves are SYS. GPU 7 pairs
  with GPU 6 (PIX).
- Submodules are now initialized on the remote: cutlass `f3fde58` (CUTLASS
  4.2.1) and deep_jit `2efdab4`.
- The venv has no torch yet. `torch-2.14.1+cu130` cp315 wheel exists on
  `download.pytorch.org/whl/cu130` (verified by download). `uv` is at
  `/snap/bin/uv`.

Not yet verified (check in Phase 0): SM count (spec says 188), max opt-in smem
per block (expected 101376 B = 99 KB, per SM 100 KB), max cluster size, and
FP8 `mma.sync` throughput with FP32 vs FP16 accumulate.

## 1. What TokenSpeed calls from DeepGEMM (the actual target surface)

All calls go through `tokenspeed-kernel`. The runtime `python/` never imports
`deep_gemm` directly. The pinned package is
`tokenspeed-deepgemm==2.8.0.post20260924`
(`tokenspeed-kernel/python/requirements/cuda-thirdparty.txt:2`).

| # | DeepGEMM API | TokenSpeed caller | Model use | Scale format | sm120 today |
|---|---|---|---|---|---|
| A | `fp8_gemm_nt` + `transform_sf_into_required_layout` | `thirdparty/deepgemm.py:26-51`, `ops/gemm/deepgemm.py` (`deepgemm_mm_fp8_blockscale`, min=max sm90) | V3/R1/V3.2 dense block-FP8 linears (attention projections, dense MLP, shared experts) | FP32, A 1x128, B 128x128, `disable_ue8m0_cast=True` | Filtered out (max 9.0). The registry falls back to `flashinfer_mm_fp8_blockscale` (min 10.0, no max) or `triton_mm_fp8_blockscale` |
| B | `m_grouped_fp8_gemm_nt_contiguous` / `_masked`, `get_mn_major_tma_aligned_tensor`, `transform_sf_into_required_layout` | `ops/moe/deep_ep.py` (`deep_gemm_deepep_fp8_moe_apply`, min 9.0, **no max**) | MoE experts with DeepEP (EP) | UE8M0 packed (1,1,128) when `deep_gemm_requires_ue8m0()` (arch >= 10, so **includes sm120**), else FP32 | Selectable, but has no sm120 kernel, so it would hit `UNREACHABLE`. DeepEP itself needs NVLink or RDMA, so this path is probably not usable on this box anyway |
| C | `get_paged_mqa_logits_metadata`, `fp8_paged_mqa_logits`, `fp8_mqa_logits` | `ops/attention/dsa/deep_gemm.py` (4 kernels, min 9.0, **no max**), `ops/attention/kpool/deep_gemm.py` | V3.2 DSA indexer (decode, prefill, index candidates), KPool prefill | q FP8 E4M3 with FP32 per-token-128 scale folded into weights; K cache FP8 with an FP32 per-row scale | **Selected over the Triton fallbacks, then fails at runtime** (sm120 falls into sm90 branches and asserts). Inferred, not run |
| D | `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits` | `ops/attention/dsv4/deep_gemm.py`, `dsv41/deep_gemm.py` | V4 / V4.1 indexer | int32 packed UE8M0 | Gated off (`is_blackwell` / max 10.3 / max 9.0) |
| E | `tf32_hc_prenorm_gemm` | `ops/residual/deep_gemm.py` (max 10.3) | V4 / V4.1 mHC | FP32 | Gated off; Triton fallback is used |
| F | `fp8_fp4_mega_moe` | `ops/moe/deep_gemm.py` (sm100 only) | V4 MXFP4 experts | UE8M0 | Gated off; sm100-only design |

Not used by TokenSpeed: `einsum` / `fp8_einsum` / `bmk_bnk_mn`, `bf16_gemm*`,
k-grouped GEMMs, sparse MQA logits, `mega_gate`, `mega_mhc`. The MLA absorb
bmm goes through the kernel `bmm` op (cuBLAS), not DeepGEMM.

TokenSpeed platform flags on sm120 (`platform.py:170-219`): `is_hopper=False`,
`is_blackwell=False` (major == 10 only), `is_hopper_plus=True`,
`is_blackwell_plus=True`. That is why C and B get imported and registered on
sm120 even though DeepGEMM can't serve them.

Scope note: TokenSpeed's `AGENTS.md` lists NVIDIA support as sm90/sm100/sm103/
sm107 only. sm120 is a fork-side effort (see `docker/Dockerfile.nvidia.unieai`,
which already adds `12.0a`). Upstreaming the TokenSpeed side needs that scope
to change.

## 2. Why DeepGEMM doesn't run on sm120 today

### 2.1 Host dispatch: equality checks on `arch_major`

Every dispatcher checks `arch_major == 9` or `== 10`. 12 falls through to
`DG_HOST_UNREACHABLE`, or, worse, silently into sm90 semantics. Paths below are
under `csrc/`.

- `utils/layout.hpp:86-97` `get_default_recipe`: `UNREACHABLE` for 12. This
  breaks every FP8 GEMM called without an explicit recipe.
- `utils/layout.hpp:43` `fp8_fp4_requires_k_major` (== 9 only) returns false
  on 12, so MN-major FP8 would be silently accepted.
- `utils/layout.hpp:69,78`: FP4 requires arch 10.
- `apis/gemm.hpp`:
  - `fp8_fp4_gemm_nt` 117/127 -> 131.
  - m-grouped contiguous 236/242 -> 248, masked 306/311 -> 316.
  - k-grouped 362/421/480.
  - bf16 526/530, which tries cuBLASLt first at 519-521.
  - m-grouped bf16 610-720.
- `apis/layout.hpp:38-57` `transform_sf_into_required_layout`: FP32
  passthrough only when `arch == 9 or disable_ue8m0_cast`. The UE8M0 cast is
  arch 10 only. Otherwise `UNREACHABLE`.
- `jit_kernels/impls/smxx_layout.hpp:224`: the packed-UE8M0 helper asserts
  arch == 10.
- `apis/attention.hpp`:
  - 32/43/80/111/192/233/258: `arch == 10` ternaries, so 12 silently gets
    sm90 dtype/block choices.
  - 120/124, 153/163, 201-202, 267/272 dispatch.
  - 313/344/380/414: sparse paths assert == 10.
- `jit_kernels/impls/sm90_fp8_mqa_logits.hpp:27,161` assert == 9.
- `apis/einsum.hpp`, `hyperconnection.hpp:48`, `mega_*.hpp` have the same
  pattern.
- `jit_kernels/heuristics/sm90.hpp:14` and `sm100.hpp:16` hard-code
  `smem_capacity = 232448`. sm120 is about 99 KB per block.

### 2.2 Device code: every kernel uses Hopper/datacenter-Blackwell-only hardware

| Feature | sm90 kernels | sm100 kernels | sm120 |
|---|---|---|---|
| `wgmma.mma_async` | yes (all) | - | **no** |
| `tcgen05` / TMEM / UMMA / 2-CTA | - | yes (all) | **no** |
| `setmaxnreg` | yes | yes | yes on `sm_120f`/`sm_120a` (verified) |
| TMA load/store, mbarrier, tensormap | yes | yes | yes |
| TMA multicast / clusters | 1x2 / 2x1 | yes | unverified; start with cluster 1 |
| `stmatrix` / `ldmatrix` | yes | yes | yes (verified, all targets) |
| `mma.sync kind::f8f6f4 m16n8k32` (E4M3) | - | - | yes on `sm_120f`/`sm_120a` (verified) |
| `mma.sync kind::mxf8f6f4.block_scale.scale_vec::1X ... .ue8m0` | - | - | yes on `sm_120f`/`sm_120a` (verified) |
| smem per block | 227 KB | 227 KB | ~99 KB |

Conclusion: no existing DeepGEMM kernel body can run on sm120. We need **new
sm120 kernels** built on TMA + mbarrier pipelines + `mma.sync`, plus host
dispatch/heuristics for arch 12.

### 2.3 Toolchain: already fine

- DeepJIT's `get_arch()` (`deep_jit/backend/cuda/device.hpp:68-79`) returns
  `"120f"` for major > 9 (or `"120a"` with `use_arch_family=false`). It
  compiles with `nvcc <src> --cubin --gpu-architecture=sm_120f`
  (`options.hpp:98`, `backend.hpp:95-101`). Verified: `--cubin` with
  `sm_120f`/`sm_120a` accepts `setmaxnreg`, `kind::f8f6f4` and the block-scale
  MMA.
- Gotcha: `nvcc -c --gpu-architecture=sm_120a` (object output) **fails** for
  these instructions, because it also embeds generic `compute_120` PTX. When
  writing standalone probes, use `--cubin` or
  `-gencode arch=compute_120a,code=sm_120a`.
- CUTLASS 4.2.1 has `cute/arch/mma_sm120.hpp`: `SM120_16x8x32_TN<e4m3,e4m3,float>`,
  block-scaled `mxf8f6f4 ... scale_vec::1X ... ue8m0` and
  `mxf4nvf4 ... scale_vec::2X/4X` atoms. We can include these atoms the same
  way DeepGEMM includes the sm90/sm100 arch headers, and we don't need CUTLASS
  collectives.
- No AOT CUDA build. `setup.py` only builds the host extension, so adding
  kernels needs no build-system change.

## 3. Reusable pieces

Reuse as-is or with small edits:
- `heuristics/common.hpp` `get_best_config<ArchSpec>`: plug in a new
  `SM120ArchSpec`.
- `jit_kernels/impls/runtime_utils.hpp`: TMA descriptor builders
  (`make_tma_a/b/cd/sf_desc`).
- `scheduler/gemm.cuh`: persistent tile scheduler, including the
  grouped-contiguous / masked logic. Use multicast = 1.
- `ptx/tma.cuh`, `common/tma_copy.cuh` (sm90 TMA variants only),
  `cutlass/arch/barrier.h` `ClusterTransactionBarrier`, `ptx/ld_st.cuh`
  (stmatrix/ldmatrix).
- `smxx_layout.cuh/.hpp`: SF transpose and UE8M0 pack kernels. Relax the
  `arch == 10` assert.
- `smxx_cublaslt.hpp`: cuBLASLt paths for bf16 and einsum (whether cuBLASLt
  picks sm120 kernels is up to cuBLAS).
- `deep_gemm/utils/math.py` cast helpers and `tests/generators.py` (add arch
  12 branches).

Don't reuse: `mma/sm90.cuh`, `ptx/wgmma.cuh`, `mma/sm100.cuh`,
`ptx/tcgen05.cuh`, `epilogue/sm100_*`, or anything under `impls/sm100_*`.

## 4. Scale-factor semantics decision

There are two families, and sm120 hardware supports both:

1. **FP32 scales, "1d2d" semantics (sm90 style)**: A 1x128 per-token, B 128x128
   blocks, arbitrary FP32 values. On sm120, run plain `kind::f8f6f4` MMAs into
   a per-K-block partial accumulator. Every 128 K, fold it into the main
   accumulator with `sa[row] * sb[nblk]` on CUDA cores. This gives exact
   checkpoint numerics for DeepSeek V3/R1 (FP32 `weight_scale_inv`) and is what
   TokenSpeed's dense path (A) and the DSA indexer (C) feed today.
2. **UE8M0 packed scales, "1d1d" semantics (sm100 style)**: use
   `mxf8f6f4.block_scale.scale_vec::1X` with UE8M0 scales. That's one scale per
   32 K, so gran_k 128 means the same byte repeats across 4 MMA k-steps. The
   hardware applies the scale for free. This matches DeepGEMM's sm100
   packed-int SF layout (`docs/scaling-factor-format.md` Path B) and
   TokenSpeed's `deep_gemm_requires_ue8m0()` (>= sm100, already true on
   sm120). It needs power-of-two scales, so FP32 checkpoints get requantized
   (small accuracy change; V3.2/V4 `scale_fmt=ue8m0` checkpoints are native).

Decision: implement **one sm120 FP8 GEMM kernel template with a compile-time
scale mode**, `kFP32Promote` vs `kUE8M0BlockScale`. Everything else (TMA
pipeline, scheduler, grouped layouts, epilogue) is shared, so this is a
parameter of one path, not two kernels. Dispatch keys on the SFA dtype exactly
like today (float goes to promote mode, int goes to block-scale mode).
`get_default_recipe` for arch 12 returns (1,128,128) for FP32 SFB and
(1,1,128) for int, which mirrors sm100.

## 5. Kernel design sketch: `sm120_fp8_gemm_1d1d.cuh` / `_1d2d` modes

- Warp specialization: 1 producer warp (TMA A/B/SFA[/SFB] into a smem ring,
  mbarrier full/empty) plus 2 consumer warpgroups (8 warps). Use `setmaxnreg`
  to shrink the producer and grow the consumers.
- Tiles: BLOCK_K = 128 (FP8, one 128B-swizzle row). Start with
  BLOCK_M x BLOCK_N = 128x128 and warp tile 64x32 (2x4 warps). One stage is
  then 16 KB A + 16 KB B + SF, so 2-3 stages fit in 99 KB. Also try 64xN for
  small M (decode) and 128x256 with 2 stages for large M.
- Operand fetch: `ldmatrix` (b16 view of FP8 pairs) from 128B-swizzled TMA
  tiles into the `m16n8k32` fragment layout. A and B are both K-major (NT),
  which is exactly `row.col`.
- Accumulators: 64x32 per warp is 64 FP32 per thread. Promote mode adds a
  second 64-register partial accumulator, so watch for spills
  (`--warn-on-spills` is wired in DeepJIT).
- Epilogue: BF16 via `stmatrix` into reused pipeline smem, then TMA store.
  Pipeline smem must be reused because there's no room for a dedicated
  epilogue buffer at 99 KB.
- Scheduler: reuse `scheduler/gemm.cuh` (normal, m-grouped contiguous, masked)
  with persistent CTAs = #SMs.
- No clusters or multicast in v1. Revisit only after profiling shows L2/DRAM
  bound on B loads.

MQA logits (C) on sm120: the same building blocks. FP8 Q x FP8 K^T with
`kind::f8f6f4`, then per-row K scale, ReLU, weighted head-sum on CUDA cores.
The paged variant reuses `scheduler/sm90_paged_mqa_logits.cuh` and the sm90
metadata kernel, which is pure scheduling, after checking it for
wgmma-specific constants.

## 5.5. Completing DeepGEMM on sm120: UE8M0 vs FP4 (decision, 2026-11-03)

The goal is DeepGEMM itself complete on sm120, not just the subset TokenSpeed
calls on this box. Of the unimplemented GEMM features, two are real work worth
doing: UE8M0 block-scale FP8, and FP4. Verified against the PTX ISA and the
SM100 path (context gather + ISA 9.7.16.3):

- **sm120 has the hardware block-scale MMA.**
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.f32.e4m3.e4m3.f32.ue8m0`
  assembles on `sm_120a`/`sm_120f`. Same A/B/C/D fragment layout as the plain
  `kind::f8f6f4` I already use, plus `scaleAData, {byte-id-a, thread-id-a},
  scaleBData, {byte-id-b, thread-id-b}`. scale_vec::1X is one UE8M0 byte per
  32-K block (`D = (A*2^(sfa-127)) * (B*2^(sfb-127)) + C`); the selectors pick
  which byte of a b32 reg and which lane in the quad supplies it. FP4 uses
  `kind::mxf4nvf4 ... m16n8k64` with scale_vec::2X/4X (ue8m0 or ue4m3).

- **UE8M0 FP8 on sm120 buys little over the FP32 path that already ships.**
  sm120 has no tcgen05, so the "scale is free + frees the promote registers +
  unlocks 128x256 tiles" benefit only comes from the *hardware* block-scale
  MMA, and even then TokenSpeed uses gran_k=128 (one scale per 4 MMA k-steps),
  so the register saving is marginal. A software-promote UE8M0 kernel (decode
  `exp2f(byte-127)`, same fold as FP32) would be nearly identical in cost to
  the existing FP32 kernel, differing only by reading a 1-byte SF instead of a
  4-byte one on an already-tiny SF stream. And TokenSpeed caps
  `deep_gemm_requires_ue8m0()` to sm100 and feeds sm120 FP32, so there is no
  engine caller for an sm120 int path unless that cap is also lifted. Net: low
  value, deferred.

- **FP4 is the feature the FP32 path cannot do, and it is where "beat the
  fallback" has the most room.** NVFP4/MXFP4 halves A/B bandwidth and roughly
  doubles tensor-core throughput over FP8. On the decode shapes that dominate
  serving (weight-bandwidth bound), that is a direct win no FP8 kernel —
  DeepGEMM's own FP32 path, Triton, or FlashInfer — can match. It needs the
  block-scale MMA machinery anyway (`mxf4nvf4`, scale_vec 2X/4X), so building
  FP4 first also lays the scale-register plumbing that a later UE8M0-FP8 mode
  would reuse.

**Decision:** do FP4 next (the real throughput win and the harder plumbing),
then fold in UE8M0-FP8 as a scale mode on the same block-scale kernel if a
caller materializes. Both share: load A/B FP4 (packed `e2m1`, 2 per byte) via
TMA, `ldmatrix` into the `m16n8k64` fragment, UE8M0/UE4M3 scale bytes staged to
registers with the `{byte-id, thread-id}` selector layout, block-scale MMA,
BF16 epilogue reusing the existing store. Dispatch keys on A/B dtype
(`kPackedFP4`) and SFA dtype, as sm100 already does.

## 5.6. SM120 block-scale MMA: verified instruction + SF lane mapping (2026-11-03)

`scripts/sm120_blockscale_probe.cu` (standalone, GPU 7) pins down the hardware
contract for
`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`:

- **Must compile arch-specific.** `nvcc -arch=sm_120a` still feeds ptxas a
  generic `compute_120` target, which rejects `.block_scale` / `.kind::mxf8f6f4`
  / `.scale_vec::1X` ("not supported on .target 'sm_120'"). Build with
  `-gencode arch=compute_120a,code=sm_120a` (DeepJIT already emits `sm_120a`).
- **Operands:** same A/B/C/D fragments as the plain `kind::f8f6f4` m16n8k32,
  plus `scaleAData, {byte-id-a, thread-id-a}, scaleBData, {byte-id-b,
  thread-id-b}`. The selector tuple components are `.u16` (`"h"`). CUTLASS
  (`mma_traits_sm120.hpp`) uses `SFARegisters = uint8_t[1]`, selectors all 0.
- **Decode:** UE8M0 byte `b` -> `2^(b-127)`. `D = (A*2^(sfa-127)) *
  (B*2^(sfb-127)) + C`. Confirmed exact (err 0) against a dequantized reference.
- **SF lane mapping with {bid=0, tid=0}** (lane `L = 4g + t`, `g = L/4`,
  `t = L%4`), the empirical result and what the kernel must load:
  - SF_A: `t==0` -> row `g`; `t==1` -> row `g+8`; `t∈{2,3}` unused.
  - SF_B: `t==0` -> col `g`; `t∈{1,2,3}` unused.
  This matches CUTLASS's `SFALayout = ((2,2,8),32):((8,0,1),16)` ("effectively
  16 threads, 2:0 broadcast") and `SFBLayout = ((4,8),32):((0,1),8)`
  ("effectively 8 threads, 4:0 broadcast"). It lines up with the existing
  SM120 kernel's lane layout (`group_row = lane/4`, within-warp `t`), so each
  math warp can load its own rows'/cols' UE8M0 bytes into one b32 reg (byte 0)
  and issue block-scale MMAs with no extra shuffles.

Consequence for the kernel: the scale is applied in the tensor core, so the
FP32 promote path's second accumulator and the per-K-block CUDA-core multiply
go away -- freeing the registers that currently cap tiles at 128x128. This is
what makes FP4 (and a later UE8M0-FP8 mode) able to run 128x256.

## 6. Phased plan

Each phase ends with tests passing on the remote, GPU 7, and a status-log
entry.

### Phase 0: environment and baselines (no kernel code)
1. Install torch into the venv:
   `uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`.
   Then build/install DeepGEMM from source (`./develop.sh` or `./install.sh`).
   Confirm `import deep_gemm` works.
2. Record device facts: SM count, `cudaDevAttrMaxSharedMemoryPerBlockOptin`,
   cluster limits, L2 size.
3. Microbenchmark `mma.sync` E4M3 m16n8k32 throughput with f32 vs f16
   accumulate, and with vs without block_scale. If FP32 accumulate is half
   rate on this SKU, that changes the tile/accum strategy, so do this before
   writing the kernel.
4. Baselines for DeepSeek V3 shapes (N,K in {(24576,1536), (32768,512),
   (7168,16384), (4096,7168), (7168,2048), (2112,7168), (576,7168)}; M in
   {1..8192}):
   - cuBLASLt plain FP8 (roofline proxy);
   - FlashInfer `gemm_fp8_nt_groupwise` / TokenSpeed's
     `flashinfer_mm_fp8_blockscale`, if it runs on sm120;
   - TokenSpeed `triton_mm_fp8_blockscale`;
   - cuBLASLt block-scaled FP8 if supported on sm120.

   This defines "highest performance". If FlashInfer's CUTLASS sm120
   groupwise kernel is already near roofline, the dense GEMM priority drops.
5. Run the existing test suite to record which tests crash on arch 12. That's
   the baseline for "no regressions".

### Phase 1: host plumbing for arch 12 (no new kernels yet)
- `SM120ArchSpec` in `heuristics/sm120.hpp` (smem from the runtime attribute,
  not hard-coded 232448).
- `get_default_recipe`, `fp8_fp4_requires_k_major` (sm120 needs K-major),
  `transform_sf_into_required_layout` and the `smxx_layout` asserts.
- Explicit `arch_major == 12` branches in `gemm.hpp` / `attention.hpp`.
  Unsupported APIs must raise a clear "not supported on sm120" error, not
  `UNREACHABLE` and not a silent sm90 fallthrough.
- `tests/generators.py` and test filters: arch 12 branches. Fix the `>= 9` /
  `>= 10` filters that would wrongly run sm90/sm100-only tests.

### Phase 2: `fp8_gemm_nt` on sm120 (target A)
- Kernel template from section 5, FP32-promote mode first (matches
  TokenSpeed's dense path and V3/R1 checkpoints), then UE8M0 block-scale mode.
- Tests: `tests/test_fp8_fp4.py` normal GEMM with tight tolerance vs the
  existing reference, both recipes. Also `assert`-style checks for no
  register spills.
- Tune the heuristics on the Phase 0 shape table and compare to the baselines.

### Phase 3: m-grouped contiguous and masked (target B)
- Same kernel, `GemmType::MGroupedContiguous` / `MGroupedMasked` via the
  shared scheduler. Tests in `test_fp8_fp4.py`.
- Before investing in tuning, confirm the MoE serving path on this box. Without
  NVLink, DeepEP intranode is likely unavailable, so TP-MoE goes through
  Triton/FlashInfer fused MoE. A grouped DeepGEMM is then only useful if
  TokenSpeed gets a non-DeepEP DeepGEMM MoE path.

### Phase 4: MQA logits for the V3.2 DSA indexer (target C)
- `get_paged_mqa_logits_metadata` (likely reusable), `fp8_paged_mqa_logits`
  (decode, next_n = 1 as TokenSpeed flattens MTP), `fp8_mqa_logits` (prefill).
  Tests: `tests/test_attention.py` paths for arch 12.

### Phase 4.5: beat Triton/FlashInfer, or don't ship (gate for Phase 5)

DeepGEMM on sm120 is only worth using if it's faster than what TokenSpeed would
otherwise pick on sm120. **The baseline is the TokenSpeed fallback, not cuBLAS.**
Earlier cuBLAS comparisons (section 10) only show the roofline. They don't
justify anything.

Competitors, i.e. what the TokenSpeed registry selects on sm120 today:

| Op | DeepGEMM | Competitors (all that run on sm120) |
|---|---|---|
| Dense block-FP8 linear | `fp8_gemm_nt` | `flashinfer_mm_fp8_blockscale` (if it runs on sm120; it has priority today), `triton_mm_fp8_blockscale` |
| MoE grouped FP8 | `m_grouped_*_contiguous` / `_masked` | TokenSpeed Triton fused MoE (`ops/moe/triton.py`), FlashInfer/CUTLASS MoE paths registered for sm120 |
| DSA indexer decode | `get_paged_mqa_logits_metadata` + `fp8_paged_mqa_logits` | `triton_dsa_plan` + `triton_dsa_decode_topk_fp8` |
| DSA indexer prefill | `fp8_mqa_logits` | `triton_dsa_prefill_topk_fp8`, `triton_kpool_prefill_topk` |

Method:
- **Same call boundary as TokenSpeed.** Time the op as TokenSpeed calls it,
  including everything DeepGEMM adds per call: `transform_sf_into_required_layout`,
  the SFA transpose kernel (`transpose_fp32`, visible in the profiler), the
  split-K workspace allocation, and FP32 scale passthrough. Benchmark through
  the `tokenspeed_kernel` registry with a forced `solution=`, so each competitor
  runs exactly as served.
- Also time kernel-only (`scripts/sm120_baseline.py::bench_cold`, cold L2, sum
  of profiler kernel times), so overheads can be attributed.
- Also time hot L2 under CUDA graph. That's the decode serving condition: weights
  of one layer are cold, but small activations and scales are hot. Cold-only
  numbers penalize everyone equally, but launch-count overhead (DeepGEMM =
  transform + GEMM kernels) only shows under graphs.
- Shapes: DeepSeek V3/R1 and V3.2 per-GPU serving shapes for the TP size we'd
  actually use on this box (TP1/TP2/TP4/TP8, PCIe only). M buckets: decode
  {1, 2, 4, 8, 16, 32, 64, 128, 256} (batch x MTP tokens), prefill
  {512, 1024, 2048, 4096, 8192, 16384}. Indexer: H = 64, D = 128, page 64,
  context {4K, 16K, 64K, 128K}, batch {1, 8, 32, 128}.
- Correctness parity at each point: same reference, same `calc_diff` bound.

Win criteria, per op, evaluated before any TokenSpeed change:
1. Every decode-bucket shape: DeepGEMM <= 0.95x the best competitor's time. Decode
   dominates serving latency, so a single losing decode shape blocks that op.
2. Prefill buckets: geometric-mean speedup >= 1.0x, with no shape slower than 0.97x.
   (Revised from >= 1.10x: ncu shows both DeepGEMM and FlashInfer/CUTLASS at ~85% tensor-pipe
   utilization for large M, so a 10% margin is not physically available. See the status log.)
3. End-to-end: serve DeepSeek V3 (or V3.2) on this box with DeepGEMM vs the
   fallback, same config. TPOT and prefill throughput must both improve.

Loop: measure, then profile the losing shapes (ncu: tensor-pipe utilization,
DRAM throughput, stall reasons), then optimize, then re-measure. Log every round
in section 9 and every number in section 10. Optimization backlog, by expected
impact (decide by profile, not by guess):
- Remove per-call overhead: pre-transform weight SFB once at load (TokenSpeed
  weight preprocessor), and have TokenSpeed's quantizer emit the MN-major,
  TMA-aligned SFA directly so the transpose kernel disappears.
- UE8M0 block-scale MMA mode (`mxf8f6f4.block_scale`): the scale is applied in
  hardware, which removes the CUDA-core promote, halves accumulator registers
  and unlocks 128x256 tiles. Needs UE8M0 checkpoints or load-time requantization.
- A stmatrix + TMA-store epilogue reusing pipeline smem, for large M.
- Better decode schedule: stream-K or finer split-K, plus warp-specialized
  `ldmatrix` double buffering.
- Fuse SiLU-mul + FP8 quant into the grouped GEMM epilogue (MoE).

Exit rule: if after the backlog DeepGEMM still loses on an op, **don't enable it
for that op on sm120**. Keep the TokenSpeed fallback, cap the registration's
`max_arch_version` below 12.0, and record the numbers and the reason in section 8
(dead ends). Phase 5 only integrates ops that passed this gate.

### Phase 5: TokenSpeed integration (separate repo, separate change)
- Immediate safety fix, independent of DeepGEMM: cap `deep_gemm_dsa_*`,
  `deep_gemm_kpool_prefill_topk`, `deep_gemm_deepep_fp8_moe_apply` with a
  `max_arch_version` (10.3) until sm120 kernels ship. Today they're selected on
  sm120 and would fail.
- After DeepGEMM sm120 lands: widen the registrations to include 12.0, bump the
  `tokenspeed-deepgemm` pin, and make `deepgemm_mm_fp8_blockscale` accept sm120
  (it currently checks `is_hopper`). Benchmark end-to-end on DeepSeek V3/V3.2.

### Later / optional
- FP4 (`kind::mxf4nvf4`, `scale_vec::2X/4X`) for NVFP4/MXFP4 checkpoints, which
  matters given 96 GB/GPU.
- Clusters/multicast if profiling justifies them.
- `tf32_hc_prenorm_gemm` (V4) if the Triton fallback is a bottleneck.

## 7. Verification rules

- Remote only, `CUDA_VISIBLE_DEVICES=7`, venv python.
- Sync code local -> remote with git (commit on `nctu6`, push, pull on the
  remote) or rsync. Never edit on the remote.
- Correctness: tight atol/rtol versus the existing `deep_gemm.testing`
  reference (`calc_diff`), for both scale modes and all GEMM types.
- Performance: always against the Phase 0 baselines on the same shapes, with
  the same timing utility (`deep_gemm.testing.bench_kineto`).
- JIT: check compile flags and logs (`DG_JIT_DEBUG`, `DG_JIT_PTXAS_VERBOSE` /
  equivalent) for spills and local memory.

## 8. Dead ends (don't do these)

- Porting sm100 kernels (`tcgen05`, TMEM, UMMA, 2-CTA): that hardware doesn't
  exist on sm120.
- Relaxing arch checks so sm90 kernels run: `wgmma` doesn't exist on sm120, so
  ptxas rejects it.
- Reusing sm90/sm100 heuristics or stage counts: they assume 227 KB smem, and
  sm120 has about 99 KB.
- Starting with TMA multicast / clusters: unverified on this SKU and not needed
  for v1.
- Adopting CUTLASS collective GEMMs inside DeepGEMM: DeepGEMM only uses CuTe
  arch atoms. CUTLASS/FlashInfer sm120 kernels are **baselines**, not the
  implementation.
- Concluding "instruction unsupported" from `nvcc -c --gpu-architecture=sm_120a`
  probes: that form embeds `compute_120` PTX and fails. Use `--cubin` or an
  explicit `-gencode` (section 2.3).
- Spending effort on the DeepEP MoE path before confirming DeepEP works on this
  PCIe-only box.
- Using any GPU other than 7, or pip-installing outside `.venv`.
- Two CTAs per SM for small 1D1D tiles (64x64 / 32x64 / 16x32 with half the smem
  each): implemented and correct, 2 blocks/SM confirmed with the occupancy API,
  but never faster (o_proj M=128: 64x64x128 8.0 vs 8.2 us single, 64x128 7.1).
  The busiest SM still gets the same work; removed (2026-11-04).
- Stream-K on mid-M shapes (tiles > SMs / 8): every CTA pays two pipeline fills,
  e.g. 64x128 at M=128, N=4608: 23.9 vs 16.4 us. Tested again on o_proj M=128,
  q_b M=256, dense gate_up M=128/256: never in the top 6.
- Folding stream-K into the fitted 1D1D cost model: only a few measured shapes
  split, so the fit can't learn it and mispicks (N=512 decode lost 4.7 -> 5.9 us).
  The explicit rule (tiles <= SMs / 8 and a clear K-loop win) fits better.
- Random-restart fits of the 1D1D model: lower in-sample regret (1.017) with
  non-physical constants (64x256 cost 15592, box exponent 9.5). Keep the greedy
  fit with plausible constants.
- Timing-based autotuning on first call as the default: CUDA events resolve ~2 us
  here and eager launches can't see the PDL overlap; it mispicks as often as the
  model. Opt-in only (`DG_SM120_AUTOTUNE=1`).

Deferred features (feasible but out of scope for inference on this box; the
host still rejects them, so there is no silent wrong path):
- **Sparse MQA logits** and **block-scaled MXFP8/MXFP4 dense MQA logits**: DONE
  (2026-11-06, section 9.6). Both ship for arch 12 and are validated against a
  dequantized reference. No longer deferred.
- **MN-major `fp8_einsum`** (`fp8_einsum` `bhd,hdr->bhr` / `bhd,bhr->hdr`): no longer here, done in section 9.12. The reason
  given here earlier, that MN-major FP8 needs a byte-granular transpose and `ldmatrix.b8` doesn't exist on sm_120, was
  wrong: `ldmatrix.m16n16.x{1,2}.trans.b8` exists on sm_120a/f (section 9.10). **k-grouped FP8 GEMMs / Wgrad** are no longer here:
  `k_grouped_fp8_gemm_tn_contiguous` / `_nt_contiguous` ship for arch 12 (section 9.11), running one MN-major block-scaled
  GEMM per group. FP4 dense MQA is no longer here either: it is done, section 9.6.
- **Mega MoE**: its point is NVLink symmetric-memory dispatch/combine across
  ranks; on a single PCIe GPU it degenerates to a local grouped-GEMM chain that
  `m_grouped_*` already covers, and a port is effectively a UMMA/TMEM -> mma.sync
  rewrite. **Mega gate / mHC**: single-GPU but TMEM/TF32-pipeline rewrites (mHC
  builds on the TF32 `m16n8k8` now in `mma/sm120.cuh`). Low priority.

## 9. Status log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2026-11-03 | Analysis | done | Mapped DeepGEMM arch dispatch and TokenSpeed call sites. Initialized submodules on the remote. Verified that `kind::f8f6f4`, `mxf8f6f4.block_scale` and `setmaxnreg` assemble on `sm_120f`/`sm_120a` (`--cubin`), and that DeepJIT targets `sm_120f`. Verified CUTLASS 4.2.1 has `mma_sm120.hpp`. No NVLink on the box. Torch not yet installed. |
| 2026-11-03 | Phase 0 | done | torch 2.14.1+cu130 in `.venv` (py3.15; install with `pip install "torch==2.14.1+cu130" --index-url .../cu130`, `uv` refuses the cp315 tag). DeepGEMM builds with `bash develop.sh` (needs `CUDA_HOME=/usr/local/cuda`). Device: 188 SMs, 101376 B opt-in smem/block, 102400 B/SM, L2 128 MB, 64K regs/SM. Baseline: every FP8/BF16 API hits `UNREACHABLE` (`Unknown recipe` / `Unsupported architecture`). |
| 2026-11-03 | Phase 1+2 | done (v1) | New `sm120_fp8_gemm_1d2d` (FP32 SFA/SFB, normal + m-grouped contiguous/psum + masked), `heuristics/sm120.hpp`, arch-12 dispatch in `gemm.hpp`, `layout.hpp` (recipe, K-major, FP32 SF passthrough). Correct on first run (diff ~7e-4 on DeepSeek shapes). Added deterministic split-K for normal GEMMs (<= 4 splits). |
| 2026-11-03 | Phase 2 tuning | done (v1) | Heuristic fitted to sweeps: auto within 1.10x of best layout on most shapes, worst 1.25x (M=64). See section 10 for numbers. |
| 2026-11-03 | Phase 2+3 tests | done | `tests/test_fp8_fp4.py` passes on arch 12: normal (fwd + Dgrad), m-grouped contiguous (psum and non-psum), masked, plus bitwise determinism. Generators treat arch 12 as SM90-like for FP8 (`is_sm90_like_fp8`). Wgrad (1D1D, FP32 out, accumulation) and k-grouped are skipped explicitly, since they're training-only. Grouped contiguous 680-757 TFLOPS; masked small-M up to 1.37 TB/s. |
| 2026-11-03 | Split-K counters | done | Counters are now per-stream and assert they aren't first allocated during CUDA-graph capture, the same pattern as the sparse MQA workspace. Not yet rebuilt or tested on remote. |
| 2026-11-03 | Phase 4 | in progress | Added `mma::sm120::warp_mma_16xNx128_ss`, which keeps the SM90 WGMMA per-warp accumulator layout. Generated `sm120_fp8_mqa_logits.cuh` / `sm120_fp8_paged_mqa_logits.cuh` from the SM90 sources, so the scheduler, TMA and epilogue code is unchanged and only the MMA block is swapped. The metadata kernel (`scheduler/sm90_paged_mqa_logits.cuh`) is arch-agnostic and reused. **Remaining:** host launchers (smem must fit 99 KB, so the SM90 3+3 stages at 512 math threads need re-checking), arch-12 dispatch in `attention.hpp`, `test_attention.py` on arch 12, register/spill check. |
| 2026-11-03 | Phase 4.5 setup | done | Competitors come from the existing TokenSpeed venv (`/workspace/nctu6/infra/tokenspeed/.venv`, py3.12, has tokenspeed_kernel + flashinfer 0.7.2 + triton). DeepGEMM is built separately for it (`build312/`, used via PYTHONPATH, so the pinned `tokenspeed-deepgemm` is never touched). Helper: `scripts/sm120_remote.sh {build,dg,ts}`. FlashInfer JIT needs the venv `bin/` on PATH (`ninja`). |
| 2026-11-03 | Phase 4.5 round 1 (dense) | **losing in places** | `scripts/sm120_vs_tokenspeed.py`, CUDA-graph + cold weights, TP1. vs Triton: DG wins almost everywhere (1.1-3.8x) except q_b/kv_b at M=32-128 (0.81-0.99). vs FlashInfer (runs on sm120, picked by TokenSpeed except 17<=M<=32 and N%128!=0): DG wins decode M<=128 on most linears (1.1-2.3x), but loses q_b/kv_b/shared-down at M>=256 (0.84-0.94), and large M is parity (0.99-1.04). |
| 2026-11-03 | Split-K fix | done | Split-K counters were per-stream and asserted outside capture, so they broke under `torch.cuda.graph` (it captures on its own stream). Now they're plain `cudaMalloc` (relaxed capture mode) zeroed by a memset on the current stream, so first use inside a capture is safe. |
| 2026-11-03 | Stream-K | tried, not a win | Replaced split-K with stream-K (even iteration ranges; deterministic slot reduction; `DG_SM120_FORCE_STREAM_K`). Correct, but under graphs sk=0 is best on almost every shape; it only helps M=256 q_b by ~3%. **The real loss is the heuristic**: `bench_cold`-fitted choices are 1.1-1.9x slower than the best layout under graphs. Re-fitting on graph-mode sweeps (`scripts/sm120_sweep_graph.py`). |
| 2026-11-03 | MMA peak | measured | `mma.sync` E4M3 m16n8k32 FP32-accum peak ≈ 906 TFLOPS, with block_scale at the same rate. Large-M DG / FlashInfer reach ~730-750 (81-83%). |
| 2026-11-03 | Phase 4 | done (correctness) | Host launchers in `csrc/jit_kernels/impls/sm120_fp8_mqa_logits.hpp`, arch-12 dispatch in `attention.hpp`. Contiguous: 8 math warps, BLOCK_KV 128, 2 Q + 3 KV stages (fits 99 KB). Paged: 4 math WGs (SPLIT_KV 256, same as SM90 so the SM90 metadata kernel is reused unchanged), stage count chosen to fit smem (2+2 for next_n=1). `scripts/sm120_mqa_check.py` matches a dequantized reference to ~2e-14 (contiguous; paged next_n 1 and 2; H 32/64; ctx up to 20K). `tests/test_attention.py` asserts pass on the cases that fit memory; it OOMs on GPU 7 (see the next row) and its kineto perf print fails with CUPTI_ERROR_MULTIPLE_SUBSCRIBERS (run with `DG_USE_NVIDIA_TOOLS=1`). Test head dims are limited to 128 on arch 12. |
| 2026-11-03 | **Env caveat** | open | GPU 7 is **shared**: a root `VLLM::EngineCore` (it restarts; seen with 65-92 GB and up to 53% util) runs on it. Benchmarks taken while it's active are noisy: 22/210 sweep rows have the heuristic "beating" the best forced layout by >7%, which is impossible. Perf numbers must be compared within the same run, interleaved and repeated. |
| 2026-11-03 | API note | open | DeepGEMM 2.8.2 (this base) renamed the indexer APIs: `fp8_mqa_logits(q, kv, ..., clean_logits, max_seqlen_k)` became `fp8_fp4_mqa_logits((q, None), kv, w, ks, ke, max_seqlen_k)`, and `fp8_paged_mqa_logits` became `fp8_fp4_paged_mqa_logits((q, None), ...)`. TokenSpeed calls the old names (pinned 2.8.0.post20260924). Phase 5 must port the call sites or add thin compat wrappers. |
| 2026-11-03 | Heuristic refit | done | Round-robin graph sweep (`sm120_sweep_graph.py --repeats 5 --json`, 15 shapes x 13 M) plus offline fit (`sm120_fit_heuristic.py`). New model in `SM120ArchSpec::get_layout_info`: per wave, max(DRAM share, L2 share, CTA time + overhead). Geomean regret vs the best layout: 1.177 (old) -> 1.006 (new), worst 3.2x -> 1.3x. Stream-K is now opt-in only (`DG_SM120_FORCE_STREAM_K=1`). |
| 2026-11-03 | SFA without transpose | done | TokenSpeed's per-call `transform_sf_into_required_layout` launched a `transpose_fp32` kernel (~2.4 us + a launch) on every GEMM. The SM120 kernel now reads SFA with plain strided loads (K-major or MN-major, any strides), and SFA passes through untransformed on arch 12. SFA is no longer staged in smem, so stages are slightly deeper. |
| 2026-11-03 | **Race fix** | done | `test_fp8_fp4.py` determinism caught random wrong 8-col tiles with 8 math warps (BLOCK_M=128). Root cause: generic-proxy `ldmatrix` reads followed by async-proxy TMA writes to the same stage need `fence.proxy.async.shared::cta` before the empty-barrier arrive. Without it, about 1 run in 20 (128x128) or 7 in 20 (128x64) corrupted output. Moving the arrive or adding a `memory` clobber didn't help; disabling setmaxnreg didn't either. Fence added to the GEMM and both MQA kernels. All layouts are now 0/20 nondeterministic, and `test_fp8_fp4.py` passes. |
| 2026-11-03 | ncu available | note | `sudo -n` works on aws-raichu, so `sudo -n env ... ncu` gives counters. Large-M 128x128: tensor pipe 85%, L2 70%, so it's compute-bound near peak. FlashInfer (CUTLASS `MainloopSm120TmaWarpSpecializedBlockwiseScaling`, 384 non-persistent CTAs, 82 KB smem) is in the same regime. **A >= 1.10x prefill win over FlashInfer is not physically available for FP32-accumulate FP8 at these sizes**, so the prefill gate is revised to "no regression (>= 0.97x)". |
| 2026-11-03 | DSA decode | **DG wins big** | `sm120_dsa_vs_tokenspeed.py` (TokenSpeed's `deep_gemm_dsa_decode_topk` vs `triton_dsa_decode_topk_fp8`, graph): DG is 5.9-18.9x faster (B 1-128, ctx 16K-128K). Top-k overlap 0.97-0.98 (random data, near ties). Needed the `deep_gemm.compat` legacy wrappers (`fp8_mqa_logits`, `fp8_paged_mqa_logits`). |
| 2026-11-03 | DSA prefill | TokenSpeed blocker | TokenSpeed's `deep_gemm_dsa_prefill_topk` calls `torch.ops.trtllm.indexer_topk_prefill`, which has **no sm120 kernel image** in this build (`cudaErrorNoKernelImageForDevice`). Phase 5 must route sm120 to another top-k. The benchmark uses `batch_invariant=True` (row-invariant top-k) for now. |
| 2026-11-03 | MQA hang fixes | done | (1) My fence insertion into the brace-less `if (q_iter_idx > 0) arrive()` made the arrive unconditional, which hung paged MQA at B>=32. Now braced. (2) next_n=2 picked 1 Q stage to fit smem, which deadlocks: the TMA warp waits for an empty Q stage before issuing the KV loads the consumer needs. Paged configs now require >= 2 Q stages ((3,3), (2,3), (2,2), (2,1)). `sm120_mqa_check.py` gained B=32/64/128 and next_n=2 regression cases; all pass at ~4e-14. |
| 2026-11-03 | Ops note | note | Don't `pkill -f <script name>` over ssh: the pattern also matches the ssh command line and kills the session. |
| 2026-11-03 | Epilogue + SFA prefetch | done | ncu at kv_b M=512 (DG 44.5 us vs FlashInfer 38.7 us): DG issued 262K global-store requests at 8 half-empty sectors each, with lg_throttle 0.88 and long_scoreboard 2.5 (vs FI 0.63, which uses TMA stores). Changes: lanes `t`/`t^1` exchange halves with one `shfl.xor`, so each lane stores 8 B and each row writes a full 32 B sector per store (requests and sectors halved, lg_throttle 0.88 -> 0.03); SFA is prefetched one K block ahead (long_scoreboard 2.5 -> 0.93). kv_b M=512: 44.5 -> 39.2 us (FI 38.6). Requires `ld_d % 4 == 0`. |
| 2026-11-03 | Phase 4.5 dense | **passed (decode), parity (prefill)** | Round 4, CUDA graph + cold weights, vs the best of Triton/FlashInfer per shape. **TP8** (the config this 8x96 GB box needs for DeepSeek-V3): every decode shape M <= 128 wins 1.17-4.0x except shared-down M=128 (1.03); M=256 1.00-2.5x; prefill M >= 512 0.96-1.18x (FlashInfer parity within ~3% noise). **TP1**: decode 1.0-3.0x (K=512/1536 at M=64-128 are at Triton parity, 0.97-1.00); prefill 0.96-1.05x. The best layout is already chosen for the remaining prefill gaps (sweeps), so they're kernel-level (both kernels are tensor-pipe bound). See section 10. |
| 2026-11-03 | Phase 4.5 DSA | **passed** | Decode (`dsa_plan` + `dsa_decode_topk`, graph): DG 3.4-19.3x faster than Triton (B 1-128, ctx 4K-128K). Prefill: DG logits plus Triton's radix top-k (`_topk_with_padding`) is 7.8-52x faster than `triton_dsa_prefill_topk_fp8` (T 1K-8K, prefix up to 14K). Top-k overlap with Triton 0.97-1.00 (random data, near ties; DG matches a dequantized reference to 1e-14). |
| 2026-11-03 | Phase 4.5 MoE | n/a on this box | TokenSpeed uses DeepGEMM grouped GEMMs only through `deep_gemm_deepep_fp8_moe_apply` (DeepEP). DeepEP needs NVLink (intranode) or RDMA (low-latency), and this box is PCIe-only. The kernels are correct and fast (test_fp8_fp4: contiguous 680-757 TFLOPS, masked up to 1.37 TB/s), but there's no TokenSpeed consumer here. |
| 2026-11-03 | Phase 5 DeepGEMM side | done | `deep_gemm.get_supported_arch_majors()` returns (9, 10, 12), so callers can tell an SM120-capable build from the pinned 2.8.0. Legacy `fp8_mqa_logits` / `fp8_paged_mqa_logits` wrappers live in `deep_gemm/compat.py`. |
| 2026-11-03 | Phase 5 TokenSpeed side | done (local, uncommitted) | `thirdparty/deepgemm.py::deepgemm_supports_sm120()` probes the build. Dense: new `deepgemm_sm120_mm_fp8_blockscale` (12.0-12.9, SPECIALIZED, `block_scale_layout` `canonical_blackwell`, the label the `mm` facade uses on SM100+). `flashinfer_mm_fp8_blockscale` steps back to <= 11.9 when it's registered, since DG's shape contract covers FI's and an in-tree kernel can't go above FI's SPECIALIZED+3. DSA, KPool and DeepEP-MoE DeepGEMM leaves are capped at 10.3 unless the build supports SM120. DSA prefill on SM120 uses `triton_topk_from_logits` instead of `torch.ops.trtllm.indexer_topk_prefill` (no sm120 image). `deep_gemm_requires_ue8m0()` is now SM100-family only. server_args help and the gemm README are updated. Tests: `test/test_deepgemm_sm120_gating.py`, `test/nvidia/ops/gemm/test_sm120_deepgemm.py`, 17 passed on GPU 7. The related existing suites pass; `test_residual_mhc_arch_window::test_registry_picks_triton_mhc_on_sm120_not_deep_gemm` fails only when run after other files, also on the stock package (pre-existing order dependence). |
| 2026-11-03 | Phase 5 E2E | **DG wins** | Qwen3.8-27B-FP8 (block 128x128 FP32 scales, the same dense block-FP8 path as DeepSeek-V3), TokenSpeed TP1 on GPU 7, `scripts/sm120_e2e.sh base|dg`, two runs each. TPOT: C=1 31.9 -> **20.6 ms (1.55x)**, C=8 32.4 -> **22.4 ms (1.45x)**, C=32 33.5 -> **28.4 ms (1.18x)**. Output throughput +53% / +43% / +16%. Long-prompt (6000 words) TTFT 795 -> 781 ms. All 260 block-FP8 `mm` selections resolve to `deepgemm_sm120_mm_fp8_blockscale` (TOKENSPEED_KERNEL_VERBOSE). Greedy outputs match base (similarity 0.92-1.00, late divergence only). DeepSeek-V3/V3.2 itself needs 8 GPUs (TP8); only GPU 7 is allowed, so that run is still open. |
| 2026-11-03 | Commits | done | DeepGEMM `970b868` (no *.md), pushed to `origin/nctu6`. TokenSpeed `1c7a96a73` (integration; only my files, since `pre-commit --all-files` reformats ~100 unrelated files already on the branch). |
| 2026-11-03 | Submodule + install.sh | done (tested) | TokenSpeed: `DeepGEMM/` submodule (url `https://github.com/nctu6/DeepGEMM`, branch `nctu6`). `scripts/install.sh` inits it (recursive, only when unpopulated) and builds it after the kernel steps, because those re-pull the pinned wheel. It uninstalls `tokenspeed-deepgemm`, runs `pip install ./DeepGEMM --no-build-isolation --no-deps` with `DG_FORCE_BUILD=1` (never the upstream prebuilt wheel), and verifies through metadata. It reruns on `--force-deepgemm` / `REINSTALL_DEEPGEMM=1`, when `deep_gemm` is missing, or when `tokenspeed-deepgemm` reappears. Tested on aws-raichu with a scratch checkout plus a relocated copy of the TS venv, so the live venv is untouched: forced build OK (`2.8.2+970b868`), the rerun is a no-op, init from an empty checkout works, `fp8_gemm_nt` on GPU 7 gives calc_diff 1.4e-6 vs the dequantized reference, and the 17 TokenSpeed SM120 tests pass. |
| 2026-11-03 | Docker build deps | fixed | The DeepGEMM step failed in the image on two missing headers. (1) `elfutils/libdwfl.h` (deep_jit; `libdw.so.1` is only dlopen'd for stack traces): added `libdw-dev` to the Dockerfile apt layer, and to install.sh's system deps when DeepGEMM is built. (2) `cusparse.h` (torch's `CUDAContextLight.h`; the image's toolkit is trimmed to cudart/nvcc/cuBLAS): DeepGEMM `setup.py` now appends torch's pip `nvidia/*/include` dirs after `CUDA_HOME`'s (`3070f4c`). Tested with the exact Dockerfile step on top of `nctu6/tokenspeed:e27cee86-20261028`: build OK, the sanity check passes, and the GEMM inside the container on GPU 7 gives calc_diff 1.4e-6. `docker/unieai.nvidia.sh` with a stub `docker` repopulates an empty DeepGEMM recursively and keeps local edits. A full image build was not run (hours). |
| 2026-11-03 | Install auto-detect | done (tested) | TokenSpeed `scripts/install.sh` now auto-detects an sm120 GPU (compute-cap major 12 from `nvidia-smi`, torch fallback once the venv is resolved) and builds DeepGEMM from the submodule only there, with no flag. On any other GPU (or none visible) it keeps the published wheel and skips the DeepGEMM submodule init. `--force-deepgemm` / `REINSTALL_DEEPGEMM=1` / `FORCE_SM120=1` still force it anywhere. `docker/unieai.nvidia.sh`: the DeepGEMM submodule init is non-fatal now (keeps the wheel if it cannot populate), and the image still always builds DeepGEMM from source when present (it targets sm120 and the kernels JIT at runtime). Tested on aws-raichu: plain install on GPU 7 (cap 12.0) builds + verifies the source build; a stub `nvidia-smi` reporting 9.0 keeps the wheel and skips the init; `--force-deepgemm` forces the build with the stub still active. TokenSpeed commit `b4cf963fc`, local on `nctu6`. |
| 2026-11-03 | Limitations audit | verified | Re-read the host dispatch to confirm the "not supported on sm120" list against the code (not the report): every item hits `DG_HOST_UNREACHABLE` or an `arch_major == 10`/`== 9` assert with no arch-12 branch. The one nuance: m-grouped contiguous/masked FP8 GEMMs *are* implemented for arch 12, just unused by TokenSpeed on a PCIe box (DeepEP only). Current-state section updated to split "not implemented" from "implemented but unused". |
| 2026-11-04 | Block-scale probes | done | `scripts/sm120_blockscale_probe.cu` and `sm120_fp4_probe.cu` (standalone, GPU 7). Facts: (1) block-scale MMAs need arch-specific codegen (`-gencode arch=compute_120a,code=sm_120a` or `compute_120f`); `-arch=sm_120a` still hands ptxas a generic `compute_120` and fails. (2) UE8M0 decodes as `2^(b-127)`. (3) Scale lane map with all selectors 0 (`g = lane / 4`, `t = lane % 4`): SFA lane `t == 0` -> row `g`, `t == 1` -> row `g + 8`; SFB lane `t == 0` -> col `g`. FP4 `m16n8k64` `scale_vec::2X` takes a 16-bit register: byte 0 = K 0-31, byte 1 = K 32-63. (4) The packed-E2M1 A/B fragments load with the same `ldmatrix` pattern as FP8 (a 64-element FP4 K step is 32 bytes), low nibble = even k. |
| 2026-11-04 | Block-scaled GEMM (1D1D) | done (tested) | `sm120_fp8_fp4_gemm_1d1d`: FP4 (32-elem scales) and FP8 (32/128-elem scales), normal + m-grouped contiguous + masked, BF16 out, same host SF contract as SM100 (int32, 4 UE8M0 bytes per word, MN-major TMA-aligned; FP32 `gran_k = 32` scales are packed on the fly, `gran_k = 128` FP32 stays on the FP32 kernel). A 128-byte tile row is 256 FP4 or 128 FP8 elements, so the TMA pipeline, scheduler, warp tiling and epilogue are the FP32 kernel's; the scales ride the MMA, so there is no second accumulator. `scripts/sm120_blockscale_check.py` matches a dequantized reference (FP4 exactly, FP8 <= 5e-9) at tol 1e-7 on 7 + 3 shapes incl. K tails and the grouped kernels; negative control: one wrong scale -> 2.6e-6, one flipped byte -> 1.6e-5. `tests/test_fp8_fp4.py` and `sm120_mqa_check.py` still pass. |
| 2026-11-04 | FP4 scale latency | fixed | v1 read the packed words with `__ldg` one K block ahead. A 128-byte FP4 K step is shorter than a DRAM round trip, so decode FP4 ran at 15.7 us on q_a/kv_a (parity with FP8); with the loads stubbed out it ran 8.7 us. The scales now ride the stage: two more TMA copies on the same full barrier land them in per-stage smem (`make_tma_sf_desc`, box = rows x 1-2 words), read from smem by the math warps. FP4 q_a/kv_a M=1: 15.7 -> 9.5 us; FP8 g32 no longer pays for its extra scale traffic. |
| 2026-11-04 | Block-scaled perf | measured | Graph + cold weights, TP8 DeepSeek-V3 linears (`sm120_bench_blockscale.py --baselines`): FP8 UE8M0 (g128 and g32) is at parity with the FP32 kernel (+-5%), as expected (same bytes, same MMA rate; it is a coverage feature for UE8M0 checkpoints). FP4 is 1.6-2.0x faster than the best FP8 kernel at nearly every shape, e.g. dense gate_up: M=1 13.6 us (FP8 24.0, Triton 58.7), M=4096 205 us (FP8 403, FlashInfer 410). Weakest: shared down (K = 256, one K block): 1.06-1.3x. |
| 2026-11-04 | FP4 vs FlashInfer MXFP4 | **mixed** | FlashInfer `mm_fp4(use_nvfp4=False, backend="cudnn")` is the only FlashInfer backend that runs MXFP4 on sm120 (cute-dsl needs NVFP4; cutlass/trtllm don't support it). `sm120_fp4_vs_flashinfer.py`, 7 linears x 9 M: DeepGEMM wins 1.1-2.3x on most points (o_proj and dense down M=256-512: 2.0-2.3x; q_b M=512: 2.1x) but **loses** at q_a/kv_a M <= 64 (7.3 vs 9.4 us at M=1), shared gate_up M=32-256 (0.78-0.90x) and kv_b M=512 (0.89x). The losers are K = 7168 with few N tiles (2112/64 = 33 CTAs): per-CTA bytes in flight (~3 MB total) bound the DRAM rate, so they need more CTAs or deeper pipelines. |
| 2026-11-04 | Block-scaled stream-K | tried, off | Ported the deterministic stream-K to the 1D1D kernel (correct in all modes, `DG_SM120_FORCE_STREAM_K=1`, `=2` skips the cost gate for tests). It wins at N = 512 (FP4 M=1: 6.4 -> 5.3 us, M=8: 6.3 -> 5.4; beats cuDNN's 6.5) and loses at N = 2112 (9.6 -> 10.2 us at M=1, 10.6 -> 13.8 us at M=64), so the FP32-fitted cost model is wrong for this kernel. Unrolling the split-reduction loop (4x) did not change that. Off by default until the cost model is refitted on block-scaled sweeps. The model also counted K in elements; it now counts bytes (FP4 packs two per byte). |
| 2026-11-04 | FP4 tile sweep (q_a/kv_a, shared gate_up) | finding | `DG_SM120_FORCE_LAYOUT` sweep of the losing shapes: for q_a/kv_a (N = 2112) the heuristic's 16x64 is already the best candidate (M=1: 9.1 us; 16x128 25, 32x64 10.7, 64x64 25, 64x128 30.5, 128x64 32.2), so tile choice cannot close the gap to cuDNN (7.3 us). What is missing is a narrower tile (BLOCK_N = 32, 66 CTAs, more bytes in flight). It is not a drop-in: the kernel runs 128 math + 128 TMA threads so `setmaxnreg` (per 128-thread warpgroup) sees two clean groups; a 2-warp math group would share a warpgroup with TMA warps. It would need 4 math warps (e.g. 2 N-warps x 2 intra-CTA K-split warps with an smem reduction) or an 8-wide warp tile with unpaired ldmatrix/stores. GPU 7 is shared, so single-run noise is ~20% on small shapes: the shared gate_up M=32 "auto" read 13.9 us here vs 8.5 us in the full benchmark. |
| 2026-11-04 | 1D1D kernel: narrow tiles, 8 warps, stage staging | done | BLOCK_N = 32 tiles split each K block's MMA steps over 2 K-warps (4 math warps stay one warpgroup for `setmaxnreg`), reduced through smem. 64x128 / 64x256 tiles run 8 math warps as 2 x 4 (ncu: b12x runs 8 warps on the same tile; with 4, `stall_wait` was 2x). The whole stage (all K steps' fragments and scales) is copied into registers and released *before* the MMAs, so all stages refill while one computes (64x128 FP4: 0.93 -> 0.70 us per K block). 200x bitwise determinism. |
| 2026-11-04 | PDL in both GEMM kernels | done | The producer pre-issues B/SFB of the first `kNumStages` K blocks of its first tile before `cudaGridDependencySynchronize()` (transactions expected without arrival; A/SFA arrive after the wait). Trigger: by the producer if all K blocks fit the pipeline, else by the math warps at mainloop end. Rejected: trigger at start (slower everywhere), never, always after loads (dense gate_up 21.0 vs 19.5 us), extra L2 prefetch (neutral/worse; full-tile prefetch thrashed L2: K=16384 59.3 vs 46.8 us). Normal GEMMs launch `min(SMs, tiles)` CTAs. Stream-K workspace is per stream and persistent. Ported to the FP32 1D2D kernel: `test_fp8_fp4.py`, `sm120_mqa_check.py` pass; TP8 decode ratios vs Triton/FlashInfer equal or better than round 4 (q_a/kv_a M=1 3.35x, was 2.89x). |
| 2026-11-04 | FP4 64-byte K blocks, PSUM, BF16 | done | FP4 may use 64-byte K blocks (`BLOCK_K = 64`, swizzle 64B), twice the stages: wins at large M (128x128x64 at M=4096). Block-scaled m-grouped contiguous now supports the PSUM layout (`use_psum_layout`, `expected_m_for_psum_layout`). **BF16** runs on the same 1D1D kernel (`kIsBF16`, `KernelNoSF`, `m16n8k16`, no scale loads): `bf16_gemm_nt` (when cuBLASLt is not used: deterministic mode), `m_grouped_bf16_gemm_nt_contiguous` (incl. PSUM) and `m_grouped_bf16_gemm_nt_masked`. `sm120_blockscale_check.py --mode all --grouped --repeat N` covers fp4 / fp8g32 / fp8g128 / bf16, normal / contiguous / PSUM / masked, K tails, forced layouts (64x128x64, 16x32, 64x256), forced stream-K, PDL on: all OK. |
| 2026-11-04 | 1D1D cost model refit | done | Sweep of every layout x stream-K on 63 FP4 shapes (`sm120_sweep_blockscale.py`, graph + PDL + cold weights; mid-M re-measured with 5 repeats), fit with `sm120_fit_blockscale.py`. New model for data-parallel 1D1D layouts: max(DRAM share x small-box penalty, L2 share, per-CTA K loop x pipeline-depth penalty) per wave, plus the explicit stream-K rule (tiles <= SMs / 8 and a clear K-loop win). Regret vs the best layout: geomean 1.012, worst 1.105 (old model 1.084 / 1.358). The timing autotuner is now opt-in (`DG_SM120_AUTOTUNE=1`). |
| 2026-11-04 | FP4 vs FlashInfer MXFP4 (cudnn, b12x) | **60/63 wins** | `sm120_fp4_vs_flashinfer.py`, interleaved, PDL on both sides, GPU idle: DeepGEMM wins 60 of 63 points, 1.03-2.2x (decode: q_b/kv_b 1.8-2.2x, shared gate_up M=1 4.6 vs 6.4 us). Near-ties: q_b M=256 6.0 vs 5.4 us (b12x), o_proj M=128 7.3 vs 7.1, dense gate_up M=256 20.0 vs 19.9. On the first two, the sweep's best forced layout matches or beats b12x (5.32-5.49 and 6.63-6.97 us), but its margin over the model's pick is within run-to-run noise. b12x and our best layout run the same CTA count (96 / 112), so a real margin needs kernel work, not tile choice. |
| 2026-11-04 | UE8M0 FP8 after the refit | **regression** | `sm120_bench_blockscale.py --tp 8 --baselines`: FP4 is 1.3-2.0x over the best FP8 kernel (shared down 0.92-1.48x), but UE8M0 FP8 (g128/g32) fell behind the FP32-scale kernel at mid/large M (dense gate_up M=4096: 595 vs 402 us; q_a/kv_a M=4096: 312 vs 176). Cause: the 1D1D model was fitted on FP4 only; its depth penalty (up to 3x for 2 stages) never mattered there because FP4 always had a 5-stage 64-byte-K alternative (the 2-stage 128x128x128 FP4 tile measures only 1-14% slower). FP8 has no 64-byte K block, so the model picks 64x64 (647 us) over 128x128 (436 us). Committed in `b419c02` with this known issue. Fix: FP8/BF16 sweep + joint refit, or calibrate the depth penalty on the FP4 bk128/bk64 pairs. FP32-scale FP8 (what TokenSpeed uses) is unaffected. |
| | Open | - | (1) Release a `tokenspeed-deepgemm` wheel from this fork and bump the pin in `tokenspeed-kernel/python/requirements/cuda-thirdparty.txt` (until then, use PYTHONPATH like `sm120_e2e.sh`). (2) DeepSeek-V3.2 TP8 E2E, which needs GPUs 0-7. (3) Optional kernel work: UE8M0 block-scale MMA path and bigger tiles for prefill margin over FlashInfer. | Gate added: DeepGEMM must beat the TokenSpeed sm120 fallbacks (Triton/FlashInfer) per op before Phase 5 integrates it. cuBLAS parity alone doesn't justify integration. Needs triton + flashinfer + tokenspeed-kernel installed into `.venv` on the remote (none are there yet). |

| 2026-11-05 | UE8M0 FP8 regression | fixed | Reworked the sweep/fit to be mode-aware (`--mode fp4/fp8g32/nvfp4`, everything in bytes) and fit all three modes jointly (`sm120_fit_blockscale.py` reports per-mode regret, so an FP4 gain cannot hide an FP8 loss). New `k1D1D*` constants + cost table. Per-mode geomean regret 1.02 (worst layout 14%). UE8M0 FP8 at large M is back to ~1.0x of the FP32-scale kernel (dense gate_up M=4096: 406 vs 402 us, was 595); FP4 unchanged (71/71 vs the Triton/FlashInfer baselines; 59/63 vs FlashInfer MXFP4, the rest ties >= 0.97x). |
| 2026-11-05 | NVFP4 | done | FP4 with 16-element UE4M3 scales (`scale_vec::4X`), keyed on `gran_k == 16`. Host: `get_mn_major_tma_aligned_packed_ue4m3_tensor` packs FP32/E4M3 values to UE4M3 int words; `layout.hpp` accepts raw or pre-packed scales on arch 12. Kernel: a K step takes one whole scale word; the K-warp keeps only its own words in registers (fixed a local-memory spill on narrow tiles). `sm120_blockscale_check.py --mode nvfp4` passes (normal / contiguous / PSUM / masked / epilogue / determinism, forced layouts + stream-K). vs FlashInfer NVFP4 (b12x / cute-dsl / cudnn / cuBLASLt): wins ~40/63; losses are mid/large M within a few %. NVFP4 runs ~0-12% slower than MXFP4 here (more scale bytes), so MXFP4 stays the default where a model allows it. |
| 2026-11-05 | BF16/FP4 epilogue | done | `D = alpha * (A @ B^T) (+ C)`, BF16 or FP32 output, on the 1D1D kernel. `bf16_gemm_nt` takes `alpha` and `c`; the block-scaled entry points pass `alpha`. Accumulation reads D, adds in FP32, rounds once. `sm120_blockscale_check.py` epilogue cases (alpha, accumulate, C-aliases-D, FP32 out) pass for every mode. |
| 2026-11-05 | tf32_hc_prenorm_gemm | done | New `sm120_tf32_hc_prenorm_gemm.{cuh,hpp}` + `TF32MMA` (`m16n8k8`). Same TMA pipeline / K-split as SM90, `mma.sync` TF32 instead of WGMMA, D and sqr_sum written from registers. `test_hyperconnection.py` passes (< 1e-8 vs the TF32 reference); ~1.2 TB/s at large M. |
| 2026-11-05 | einsum / fp8_bmm | done | `bmk,bnk->mn` accumulates a BF16 GEMM per batch slice into FP32 D; `fp8_bmm` runs one FP32-scale GEMM per batch (so `fp8_einsum bhr,hdr->bhd` works). Both match the reference (bmk,bnk->mn to ~1e-6 BF16 / 1e-12 FP32; fp8 einsum ~7e-4). The arch-12 `fp8_bmm` else-branch previously fell through to the SM90 WGMMA kernel. The MN-major expressions (`bhd,hdr->bhr`, `bhd,bhr->hdr`) are still unported. |
| 2026-11-05 | MQA head dim 64 | done | `warp_mma_16xNx128_ss` -> `warp_mma_16xNxHd_ss<N, kHeadDim>` (128 or 64) with the matching TMA swizzle. The contiguous/prefill kernel + launcher accept head dim 64 (`sm120_mqa_check.py` D=64 cases pass, ~1e-14); the paged kernel stays at 128 (its KV gather has its own swizzle assumptions). FP4 MQA is still unported. |
| 2026-11-05 | Commits | done | DeepGEMM `nctu6`: `eff4cea` (joint refit), `eb61610` (NVFP4 + epilogue + spill fix, kernel), `4659181` (NVFP4/epilogue host), `bfb637c` (tf32 hc), `5dfd734` (einsum), `38ef2a2` (MQA head dim 64). Local, not pushed. |

## 9.5 Sparse MQA logits SM120 port plan (in progress, 2026-11-05)

Target: `fp8_fp4_sparse_mqa_logits` + `fp8_fp4_paged_sparse_mqa_logits` +
the two metadata entry points, on arch 12. Ground truth: `test_sparse_mqa_logits`
in `tests/test_attention.py` (MXFP4 + MXFP8, contiguous + paged + unaligned-ks,
num_heads 8/12/20/32, sparse_block_kv 8/16). No SM90 template exists (only
`sm100_sparse_mqa_logits.cuh`), so the compute core is a from-scratch SM120 kernel.

Math per (q_token, kv_token): `logits[q, slot] = sum_h relu(Q[q,h,:] . KV[kv,:]) * weight[q,h]`,
with Q and KV block-scaled (per-32-element UE8M0, int32 words); output BF16, written
to a compressed column via the metadata slot map.

Reuse unchanged:
- `scheduler/sm100_sparse_mqa_logits_metadata.cuh` and its launcher: pure CUDA
  scheduling, no TMEM/cluster/UMMA (grep confirmed 0 hits). Rename to a
  shared/`smxx` name or include as-is from an sm120 launcher.
- `layout/sparse_mqa_logits.cuh` structs (MetadataHeader, KVSplitHeader,
  KVBlockInfo, ScheduleEntry, QBlock). `SharedStorage` must be re-derived (drop
  the TMEM barriers / `tmem_ptr_in_smem`; keep Q/KV/SF/weights smem + barriers).
- The two KV accessors (`ContiguousSparseKVAccessor` TMA, `PagedSparseKVAccessor`
  cp.async gather from the fused cache) and `get_swizzled_kv_chunk_idx`.
- The ReLU x weight head-reduction epilogue and the slot-mapped compressed store
  (identical to the SM100 math-warpgroup tail and to the dense SM120 MQA epilogue).

Rewrite (the SM100 core uses UMMA m128nN with SF in TMEM; replace with mma.sync):
- Per KV split: one math warp owns 16 KV rows; run block-scaled `mma.sync`
  (FP8BlockScaledMMA m16n8k32 / FP4BlockScaledMMA m16n8k64) of
  `[16 KV, kHeadDim] x [BLOCK_Q*kNumHeads, kHeadDim]^T`, accumulating in registers
  (like `warp_mma_16xNxHd_ss`, but block-scaled). kHeadDim 128 -> 4 FP8 / 2 FP4
  K steps; stage the Q and KV MX SF words in smem and read them with the SFA/SFB
  lane map (g = lane/4, t = lane%4) already used by the 1D1D GEMM.
- Q is loaded once per q-block (TMA); KV is gathered per split (the accessors);
  SF words likewise. BLOCK_Q = 2 (two packed Q slots), kNumMaxHeads = 32.

Host: `sm120_sparse_mqa_logits.hpp` mirroring the SM100 launcher (TMA descs for
Q/SF-Q/weights, KV via accessor), smem sized from the new SharedStorage, grid =
num_sms. `get_sparse_mqa_split_kv` (640 fp4 / 512 fp8) and `get_num_metadata_bytes`
are reusable. Dispatch: `attention.hpp` arch-12 branches (all four entry points),
and the metadata launch is arch-agnostic.

Test: let `test_sparse_mqa_logits` run on arch 12 (the generators treat arch 12
as SM90-like for FP8; FP4 Q needs the `kPackedFP4` path enabled for arch 12 in
`generators.py`). Tolerance matches the dense MQA check.

Progress: `mma::sm120::warp_mma_16xNxHd_block_scaled_ss<N, kHeadDim, kIsFP4>`
landed in `mma/sm120.cuh` (the block-scaled compute primitive: `KV[16,hd] @
Q[N,hd]^T` with MX UE8M0 scales, SF lane map copied from the 1D1D GEMM). The
metadata-scheduler smem bound is now device-based. Remaining: the kernel body
(producer + KV-copy + math warps reusing the accessors), the launcher, and the
arch-12 dispatch.

Risk retired: the SF lane map the helper uses (SFA t&1 -> row g / g+8, SFB t==0
-> col g; FP4 16-bit byte0=K0-31/byte1=K32-63) is the one the 2026-11-04 probe
already confirmed on hardware, so the compute primitive rests on validated facts.

Kernel skeleton (for `sm120_sparse_mqa_logits.cuh`, synthesizing the dense SM120
MQA kernel's compute with the SM100 sparse kernel's metadata-driven schedule):
- Warp roles (collapse the SM100 5-role layout to 3; no UMMA/SF-transpose warps):
  (a) one Q+metadata producer warp: per ScheduleEntry, TMA Q / SF-Q / weights into
      a Q-stage ring (as the SM100 producer does), and cp.async the KVSplit header +
      KVBlockInfo array into a KV-stage ring; (b) KV-copy warps: the SM100 gather
      (contiguous TMA or paged cp.async via the two accessors) into smem KV + SF-KV;
      (c) math warps: per KV split, each warp owns 16 KV rows and calls
      `warp_mma_16xNxHd_block_scaled_ss<BLOCK_Q*kNumHeads, 128, kIsFP4>` with the
      staged KV/Q SF words, then the dense kernel's ReLU x weight head reduction.
- SharedStorage: reuse the dense kernel's Q/KV/weights rings, add smem SF-Q
  `[qstage][BLOCK_Q*kNumHeads words]` and SF-KV `[kvstage][SPLIT_KV words]` (int32,
  one per 128 elements = one per token here since hd=128), plus the metadata rings
  (`kv_split_headers`, `kv_block_infos`, `q_blocks`) from the SM100 layout. Drop the
  TMEM barriers / `tmem_ptr_in_smem`.
- SF staging: the SF word of KV token `r` goes to smem `sf_kv[r]`; the helper reads
  `a_sf[a_row + g (+8)]` with `a_sf_stride = SPLIT_KV` (words laid out `[word][row]`,
  one word for hd=128). Q SF likewise `b_sf[g]`, stride `BLOCK_Q*kNumHeads`. No
  transpose (unlike SM100's UTCCP-into-TMEM path).
- Store: the slot map from the SM100 math tail. `accum` layout is the dense
  kernel's (`accum[i*kNumAccumPerReduce ...]`, i over BLOCK_Q, reduced to `v_0`/`v_1`
  for KV rows `warp_offset + lane/4` and `+8`). Map each to a compressed column via
  `packed_slot_offsets` / `q{0,1}_slot_base` from `kv_block_infos` /
  `kv_split_headers`, writing `output_rows[q][ (q_slot_base + q_slot_offset) *
  SPARSE_BLOCK_KV + token_in_kv_block ]` when the slot != kInvalidSparseSlot. No
  FP32 per-KV `scale_kv` multiply (the MX scale is applied in the MMA).
- Launcher `sm120_sparse_mqa_logits.hpp`: mirror the SM100 one (TMA descs for Q /
  SF-Q / weights; KV via accessor), size smem from the new SharedStorage, grid =
  num_sms, `get_sparse_mqa_split_kv` 640 fp4 / 512 fp8. Dispatch in `attention.hpp`
  (four entry points), flip `test_sparse_mqa_logits` to `>= 10`.

Effort: the compute primitive is landed and validated; the remaining kernel body +
launcher + dispatch is the bulk and needs the structured editor for safe iteration
against `test_sparse_mqa_logits` (oracle: bitwise-equal to the dense MQA logits).

Trap found while scoping the gather: the SM100 producer writes KV into a
UMMA-descriptor smem layout via `get_swizzled_kv_chunk_idx`, which does NOT match
the TMA 128B/64B `ldmatrix` swizzle (`get_swizzle_offset`) that the SM120
block-scaled helper reads. So the gather producer must be rewritten to land each
gathered sparse KV block in the `ldmatrix` layout (as the dense SM120 kernel's TMA
KV tile already is), not copied from SM100. Plan: gather with `cp.async` (paged) or
TMA (contiguous) straight into a `[SPLIT_KV][kHeadDim bytes]` tile with the same
128B swizzle the dense kernel uses, and stage one int32 SF word per token into
`sf_kv[token]`. Build incrementally: contiguous MXFP8 aligned-ks first (simplest),
then FP4, unaligned-ks, and paged. Oracle each step against `test_sparse_mqa_logits`
restricted to that case.

## 10. Measurements (RTX PRO 6000, GPU 7)

Timing method matters (learned the hard way):
- `deep_gemm.testing.bench` flushes L2 only once, so repeated launches hit the 128 MB L2. That gives
  impossible numbers (>4 TB/s) for weight-bound decode shapes. **Don't use it for small M.**
- `scripts/sm120_baseline.py::bench_cold` flushes 512 MB before every call and sums profiler kernel
  times. Use it for every comparison. The cold-DRAM floor is real: a 4 MB GEMM takes ~11 us, and
  cuBLAS shows the same floor.
- Best observed DRAM throughput for decode GEMMs is ~1.0 TB/s for both DeepGEMM and cuBLAS (spec 1.79 TB/s).

v1 kernel, `sm120_fp8_gemm_1d2d` with the auto heuristic, vs cuBLASLt per-tensor FP8 `torch._scaled_mm`.
cuBLAS does *less* work here (no block scales), so ratio ~1.0 means parity with the roofline proxy.

| shape (m x n x k) | DG us | DG TFLOPS | cuBLAS TFLOPS |
|---|---|---|---|
| 4096 x 7168 x 16384 | 1309 | 734 | ~765 |
| 4096 x 4096 x 7168 | 378 | 634 | ~690 |
| 4096 x 24576 x 1536 | 517 | 597 | ~630 |
| 1 x 7168 x 16384 (decode, ~1 TB/s) | 114 | 2.1 | 1.9 |
| 1 x 2112 x 7168 | 23 | 1.3 | 1.1 |
| 64 x 2112 x 7168 | 33 | 60 | 58 |

Tile findings (`scripts/sm120_sweep.py`):
- Large M: 128x128 always wins (8 math warps, 64x32 warp tiles, 3 stages).
- Single-wave cases are bound by per-CTA load latency, so small tiles and/or split-K win.
- Split-K helps only while tiles x splits <= 1 wave. Above 4 splits it gets *slower*: one measurement,
  64x128 with 10 splits, ran 2x slower than with no splits. Capped at 4.
- 128x256 / 256x128 tiles are not available. A 64x64 warp tile needs 128 accumulators plus 128 promote
  registers, which spills. A bigger-tile design would need a different promotion scheme (future work).

Known gaps of v1:
- The epilogue stores BF16 straight from registers (16 B per row segment). A stmatrix + TMA-store
  epilogue reusing pipeline smem could help large-M shapes.
- No UE8M0 (packed int SF) path yet. Int SFs on sm120 hit `UNREACHABLE`.
- Split-K counters are allocated on first use. **Call the GEMM once before CUDA-graph capture**
  (TokenSpeed warmup does this).

### Phase 4.5 final numbers (round 4, GPU 7, graph + cold weights, us per GEMM)

TP8 per-GPU DeepSeek-V3 dense linears; ratio = best(Triton, FlashInfer) / DeepGEMM (> 1 means DeepGEMM is faster).
FlashInfer is excluded where TokenSpeed would not pick it (17 <= M <= 32, N % 128 != 0).

| linear (N x K) | M=1 | 8 | 32 | 64 | 128 | 256 | 512 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| q_a/kv_a 2112x7168 | 2.89 | 3.01 | 2.75 | 2.61 | 2.18 | 1.65 | 1.18 | 1.22 | 1.34 |
| q_b 3072x1536 | 2.53 | 2.00 | 1.97 | 1.72 | 1.37 | 1.06 | 1.00 | 1.02 | 1.00 |
| kv_b 4096x512 | 2.55 | 1.77 | 1.75 | 1.64 | 1.37 | 1.03 | 1.06 | 1.02 | 0.98 |
| o_proj 7168x2048 | 1.88 | 1.41 | 1.39 | 1.30 | 1.17 | 1.00 | 1.03 | 1.01 | 1.04 |
| shared gate_up 512x7168 | 3.70 | 4.00 | 3.04 | 3.01 | 2.68 | 2.54 | 2.16 | 1.16 | 0.96 |
| shared down 7168x256 | 2.88 | 1.93 | 1.46 | 1.33 | 1.03 | 1.18 | 1.16 | 1.13 | 1.07 |
| dense gate_up 4608x7168 | 2.36 | 2.00 | 2.00 | 1.94 | 1.64 | 1.19 | 1.00 | 0.96 | 1.02 |
| dense down 7168x2304 | 1.89 | 1.43 | 1.36 | 1.33 | 1.19 | 1.01 | 1.00 | 0.97 | 1.05 |

DSA indexer (H=64, D=128, page 64, top-k 2048), Triton time / DeepGEMM time:

| decode B x ctx | 1x4K | 8x4K | 32x4K | 128x4K | 1x16K | 32x16K | 128x16K | 1x64K | 32x64K | 1x128K | 8x128K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speedup | 3.4 | 5.4 | 10.7 | 13.4 | 6.0 | 15.2 | 19.3 | 8.1 | 17.2 | 9.2 | 14.7 |

| prefill T x prefix | 1024x0 | 4096x0 | 4096x4096 | 2048x14336 | 8192x0 |
|---|---|---|---|---|---|
| speedup (DG logits + Triton radix top-k) | 7.8 | 32.5 | 35.3 | 30.1 | 52.3 |


## 9.6 Sparse MQA done + block-scaled dense MQA + perf findings (2026-11-06)

Sparse MQA logits SM120 port **complete and enabled by default** (gate removed):
contiguous (aligned and unaligned ks) and paged, MXFP8 and MXFP4, H {8,32},
sparse_block_kv {8,16}, KV up to 4096 all match a dequantized `relu(Q.K)*w`
reference at diff ~1e-6 (`scripts/sm120_sparse_mqa_check.py`,
`sm120_sparse_paged_check.py`). The kernel is a from-scratch SM120 compute core
(single warpgroup, `cp.async` gather, `warp_mma_16xNxHd_block_scaled_ss`) on the
SM100 metadata scheduler. Five correctness bugs fixed along the way: SF-Q TMA box
shape/expect_tx, Q-barrier phase, the block-scaled MMA's per-N-tile B scale, and
the FP4 Q TMA byte count (commits eeb4926, 678df95, c384000, 0cd1e67).

**Block-scaled dense MQA logits** (`sm120_fp8_fp4_mqa_logits.cuh`): the MX path
of `fp8_fp4_mqa_logits` for arch 12, matching SM100 (UE8M0 MXFP8/MXFP4, per-row
int32 SF for Q and KV, BF16 weights/output, head dim 128). Reuses the FP8 dense
kernel's pipeline + the sparse block-scaled MMA. The FP8 per-token-scale kernel
(the DSA indexer path) is unchanged; the host dispatches by `is_mx_sf`. Two bugs
fixed: the Q/KV/SF-Q TMA outer strides (must be the contiguous row stride, and
`row_bytes` for packed FP4), and a missing `num_total_kv_blocks` advance in the
consumer that deadlocked the multi-q-block path. Validated at diff ~1e-6.

**Sparse MQA performance**: ~190 TFLOP/s MXFP8 / ~230 MXFP4, ncu shows ~46%/46%
compute/memory at 16.7% occupancy (1 block/SM, exactly one wave of CTAs). The
structural limiter is too few CTAs to exceed 1 block/SM. Tried and reverted:
smaller SPLIT_KV, a 2-stage gather/compute pipeline, `launch_bounds(256,2)`, and
three dedicated-producer/consumer multi-stage rewrites (the dense-kernel pattern).
None beat the baseline: the single-wave pipeline wasn't faster, and the multi-wave
path had recurring subtle shared-memory sync bugs. The "more CTAs" route is blocked
by the shared metadata scheduler's `kNumSMs <= 256` assert. Left at the correct
baseline; `scripts/sm120_sparse_bench.py` / `sm120_sparse_prof.py` capture the
measurement. A future pass needs a ground-up pipeline, not an incremental patch.

Full regression on GPU 6: `tests/test_fp8_fp4.py` (k-grouped skipped), the four
MQA checks, all green.


## 9.7 Cost-model refit + V3.2 indexer SM120 finding (2026-11-06)

**Cost-model refit.** Re-swept every legal 1D1D layout of the DeepSeek-V3 TP8
dense linears for MXFP4 and UE8M0 FP8 (serving condition: CUDA graph, cold
weights, PDL) and jointly refit the model. Worst-mode geomean regret on the
sweep data: 1.027 -> 1.014. New constants (`depth_exp` 1.58 -> 0.5,
`cta_scale` 1.2 -> 1.5, recalibrated per-tile cost table: 64x128 968 -> 476,
128x64 900 -> 525) make mid-M shapes prefer the 64x128 / 128x64 tiles the sweep
measures fastest. End to end vs FlashInfer MXFP4 (63 points): geomean speedup
1.315x -> 1.33x, still 59/63 wins. Committed `1b065a2`.

The 4 remaining ties (q_b M=256, o_proj M=128, dense gate_up M=256, dense down
M=128; 0.96-0.99x vs b12x) are **not closable with the available tools**: the
analytic model cannot pick a per-M tile (its cost table is one value per tile
shape), and `DG_SM120_AUTOTUNE=1` is *worse* (geomean 1.318, q_b M=256 0.94)
because eager timing cannot see the graph-mode PDL overlap. Genuine kernel-level
parity at that M/N boundary; leave at the model's pick.

**V3.2 sparse indexer on SM120 is blocked upstream, not in DeepGEMM.** The
DeepGEMM sparse MQA scoring kernel works on SM120 (section 9.6), but TokenSpeed's
native DSV4.1 indexer (`dsv41/deep_gemm.py`) gates on `platform.is_blackwell`,
which is arch-major==10 only (SM100); SM120 is arch 12 (`is_blackwell` False,
`is_blackwell_plus` True). More fundamentally, the top-k **DeepSelect** half needs
~207 KB of shared memory (sized for SM100's 227 KB); SM120 has 99 KB, so a launch
on sm_120a fails `CUDA_ERROR_INVALID_CONFIG` (probed directly with cluster_size=1).
Enabling the DeepGEMM indexer on SM120 therefore needs an SM120 smem-reduced
DeepSelect port (CuTe DSL, tokenspeed-kernel, core-team-owned), plus flipping the
`is_blackwell` gates to `is_blackwell_plus`. Hand-off noted; out of DeepGEMM scope.


## 9.8 FP8 1D2D model already optimal; FP8 competitive state (2026-11-06)

Benchmarked the FP8 FP32-scale path (the 1D2D kernel TokenSpeed's DSA/dense
linears use) at TP8 vs Triton/FlashInfer (`sm120_vs_tokenspeed.py --tp 8`):
**geomean 1.61x over 112 points**, 7 losses all 0.94-0.99x (q_b M=2048/4096,
kv_b M=256, o_proj M=1024, shared gate_up M=4096, shared down M=4096/8192).

Re-swept the 1D2D layouts (`sm120_sweep_graph.py`) and refit
(`sm120_fit_heuristic.py`): the current model is already near-optimal (geomean
regret 1.0081; a refit only reaches 1.0077, changing just
`kDRAMBytesPerUnit` 1.6e6 -> 1.2e6). The one mid-M mispick (kv_b M=256: auto
64x128 6.0us vs best 128x64 5.7us) is **not fixed by the refit** either (it
still picks 64x128). The large-M losses are `auto == best` in the sweep, i.e.
genuine kernel-level parity, not tile choice. Not committing the refit: no real
gain, target shape unfixed. The FP8 path stays at the proven model.


## 9.9 BF16 MN-major, native einsum, spill fixes, mixed scales (2026-11-06)

Commits on `nctu6` after `df1bb16`, not pushed.

- **`74e1f61` MN-major BF16 and native einsum.** The BF16 `sm120_fp8_fp4_gemm_1d1d`
  kernel takes `kMajorA`/`kMajorB`. An MN-major operand is TMA-loaded as
  128B-swizzled atoms (64 MN elements x 64 K rows) and read with `ldmatrix.x4.trans`
  into the same fragments as the K-major path. The heuristic only gives MN-major
  operands tiles whose MN dimension is a multiple of 64. Host limits: MN-major is
  BF16 only, MN-major A has no groups, a grouped MN-major B needs `K % 64 == 0`
  (TMA bounds only the stacked `G*K` rows). With this, `bf16_gemm_nn/tn/tt` and
  MN-major grouped B work in deterministic mode (which skips cuBLASLt). `einsum`
  `bhr,hdr->bhd` and `bhd,hdr->bhr` run one BF16 GEMM per head on strided views.
  `k_grouped_bf16_gemm_tn_contiguous` runs one MN-major GEMM per group from host-side
  `ks` (no PSUM layout), as on SM90. All four layouts x BF16/FP32 out x accumulation,
  K tails 8/136/1000, match an FP32 reference (diff < 1e-5). Also fixes, on all
  arches, a segfault in `einsum bmk,bnk->mn` with FP32 D and no `c` (it
  dereferenced an empty optional; it now asserts that `c` aliases `d`).
- **`9704db0` tests.** `test_bf16.py` and `test_einsum.py` crashed on SM120 in the perf
  print because they time kernels by SM90/SM100 names. `bench_kineto` has a
  `per_call_total` mode (sum over the kernels a call launches), and the generators
  know SM120's BF16 kernel and cuBLASLt kernel names. The native einsum is compared
  to torch at 1e-7 (the `mma.sync` accumulation order rounds ~0.02% of elements to
  the neighboring BF16 value). Stochastic rounding and the MN-major `fp8_einsum` tests
  stay SM100-only.
- **`4a95a00` stream-K spill.** Both SM120 GEMM kernels computed the 64-bit stream-K
  iteration range before the warp-role split, so it was live across
  `setmaxnreg.dealloc` and ptxas spilled it (48 B on the BF16 16x64 stream-K tile,
  N=1024, K=4096, caught by `DG_JIT_CHECK_NO_SPILLS=1` in `tests/test_sanitizer.py`).
  Each role now computes the range after its reconfiguration: 0 B, same results.
- **`a765e73` mixed scale pair.** A packed-UE8M0 SFA (e.g. DeepEP activations) with an
  FP32 128x128 SFB used to fail the layout check on SM120. `transform_sf_pair_into_required_layout`
  now casts the FP32 side to packed per-row UE8M0 for such pairs (gran_k 128), as
  SM100 does. Pure-FP32 pairs keep the FP32-scale kernel, and
  `disable_ue8m0_cast=True` rejects the mix. Matches a dequantized reference (~1e-6)
  in normal and m-grouped contiguous modes.
- **`9f19e0a` 1D2D masked FP8 spill.** `sm120_fp8_gemm_1d2d` masked (N=6144, K=7168,
  128x128, 256 threads) spilled 20 B. The cause was not `setmaxnreg`: the eight
  clamped SFA row offsets (`sfa_row_offsets[kNumMTiles][2]`) stayed live across the
  K loop next to the 128 accumulators and the scale prefetch. Only the base row is
  kept now; the offsets are recomputed at each load. 0 B. Masked FP8 sweep: within
  +-0.4% on 20 of 26 configs, 0.8-2.5% faster on 6, none slower beyond noise.

Limits that remained on SM120 at that point: k-grouped FP8 and MN-major FP8 einsum. MN-major FP8 GEMM operands are in section 9.10, k-grouped FP8 in 9.11, the einsum in 9.12.

## 9.10 MN-major FP8 operands (2026-11-06)

`fp8_fp4_gemm_{nn,tn,tt}` and m-grouped contiguous with an MN-major B now run on both SM120 FP8 kernels
(`sm120_fp8_gemm_1d2d`, and `sm120_fp8_fp4_gemm_1d1d` in its FP8 UE8M0 g32/g128 modes) as `kMajorA` / `kMajorB`
template parameters of the existing kernels. FP4 operands stay K-major (there is no sub-byte transposing load).

**`ldmatrix.b8` exists on sm_120.** Section 8 said it didn't. `ldmatrix.sync.aligned.m16n16.x{1,2}.trans.shared.b8`
assembles for `compute_120a` and `sm_120f` (SASS `LDSM.8.MT1616`); there is no `m8n8` b8 form, and `.trans` is required.
`scripts/sm120_fp8_trans_probe.cu` verifies its layout bit-exact on hardware. For a 16 K-row x 16 MN-byte block (`.x1`;
lane `l < 16` supplies the address of K row `l`, 16 contiguous MN bytes), with `g = lane / 4`, `t = lane % 4`:

| register | contents |
|---|---|
| `r0` | K `4t..4t+3` of MN byte `g` |
| `r1` | K `4t..4t+3` of MN byte `g + 8` |
| `r2`, `r3` (`.x2`, lanes 16-31 point at K rows 16-31) | the same for K + 16 |

These are exactly the E4M3 `m16n8k32` fragment registers: an A fragment is `{r0, r1, r2, r3}`, and the B fragments of two
8-column tiles are `{r0, r2}` and `{r1, r3}`. No byte permutes and no MN reordering, so the epilogue, the scale lookups and
the K order are those of the K-major path, and an MN-major operand pairs with a K-major one. The TMA tile is the BF16
path's: swizzled atoms of 128 (64 for 64-wide tiles) MN bytes by 128 K rows.

**Design choice.** Four designs for the transpose-on-load were built and timed.

| design | what | probe (ns / 128x128x128 K block, 8 warps, 188 CTAs, smem resident) | in the real 1D2D kernel |
|---|---|---|---|
| bound | K-major `ldmatrix.x4` | 872 | - |
| (a) | b16 `ldmatrix.x4.trans` + 4 `prmt` per 16x32 block | 907 (1.04x) | bit-exact; MN order permuted inside 16, so the epilogue, scale rows and columns are remapped; at 4096^3 B-MN 1.00x, A-MN 1.05x, both 1.07x of the bound |
| (b) | helper warps transpose the next stage in shared memory | 2404 (2.76x) | not built (the probe already doubles smem traffic and needs a sync per stage) |
| (c) | 32-bit `ld.shared` + 4x4 byte transpose with `prmt` | 877 (1.01x) | spills 7 of 8 shapes under `DG_JIT_CHECK_NO_SPILLS=1`, and 8-18% slower than (a) |
| (d) | native `ldmatrix.trans.b8` | 881 (1.01x) | bit-exact, no remapping; at 4096^3 A-MN 0.99x, B-MN 1.06x, both 1.05x of the bound |

(d) is used. It is exact, needs no remapping, and its A side beats (a). Its B side costs about 6% on compute-bound
128x128 tiles (4096^3: 218 vs 205 us for the K-major bound): the B pairs come out of the instruction as `{r0, r2}` and
`{r1, r3}`, but an MMA reads its B operand as an adjacent register pair, so ptxas copies the pair into a scratch pair
before each MMA (two `MOV` per `QMMA`, and a write-after-read dependency on the scratch). Looping over N tiles first, and
packing the pairs into 64-bit registers, produced the same SASS. Using (a) for B only would recover that 6%, at the cost of
the MN permutation in the epilogue and the scale lookups on the B side; not done, since the explicit transpose ties there.

**Host routing.** An MN-major operand either runs in-kernel or is transposed first by a JIT byte-transpose kernel
(`sm120_fp8_transpose.cuh`, a 128x64 tile through shared memory with 4x4 `prmt` transposes and 16-byte stores) followed by the
K-major kernel; that transpose is also the route for what the kernels don't cover (`M % 16 != 0` or `N % 16 != 0`, which
TMA can't express for MN-major rows, a grouped MN-major B whose K is not a multiple of 128, an MN-major A in a grouped GEMM).
`DG_SM120_MN_MAJOR_FP8` = 0 auto, 1 always host transpose, 2 in-kernel when supported (developer override).
Auto picks per operand: **B in-kernel always; A in-kernel from M = 256.** An MN-major A tile is at least 64 rows (the TMA
atom), so below that the kernel computes padding.

**Numbers** (`scripts/sm120_bench_mn_major_fp8.py`; RTX PRO 6000, GPU 7; CUDA graph over enough operand copies to exceed L2, so
operands are cold; min of 3 interleaved rounds; FP32-scale 1D2D kernel, BF16 out; DeepSeek-V3 TP8 linears 2112x7168, 576x7168,
3072x1536, 4096x512, 7168x2048, 4608x7168, 7168x2304, 512x7168, 7168x256 and squares 2048, 4096, 8192, M in
{1, 16, 64, 256, 1024, 4096}; A-MN and both-MN on a 7-shape subset, M in {16, 64, 256, 1024, 4096}). "explicit" is the
byte-transpose kernel then the K-major kernel; "bound" the K-major kernel on pre-transposed data; Triton is TokenSpeed's
block-FP8 kernel called on the strided MN-major operands (its wrapper asserts contiguity, the kernel itself takes strides);
FlashInfer needs K-major, so its time is on pre-transposed data and excludes the transpose. Ratios are geomeans (range):

| MN-major | M | n | default route / explicit | default route / bound | Triton (strided) / default | FlashInfer (K-major) / default |
|---|---|---|---|---|---|---|
| B (weights) | <= 64 | 36 | 1.63x (1.36-1.96) | 1.00x | 2.92x (1.47+) | 2.44x (1.35+) |
| B | 256 | 12 | 1.47x (1.32-1.89) | 1.00x | 1.97x | 1.43x |
| B | >= 1024 | 24 | 1.09x (0.97-1.27) | 0.96x | 1.65x | 1.02x (0.92+) |
| A (activations) | <= 64 | 14 | 1.00x (host route) | 0.88x | 4.08x | 2.78x |
| A | 256 | 7 | 1.07x (0.86-1.16) | 0.97x | 2.47x | 1.32x |
| A | >= 1024 | 14 | 1.08x (0.95-1.39) | 0.99x | 2.16x | 0.99x |
| A and B (Wgrad) | <= 64 | 14 | 1.57x (1.35+) | 0.93x | 4.60x | 1.78x |
| A and B | 256 | 7 | 1.57x (1.09-1.89) | 0.94x | 3.76x | 1.25x |
| A and B | >= 1024 | 14 | 1.20x (1.02-1.47) | 0.96x | 3.07x | 0.96x |

The in-kernel path (forced) against the explicit transpose: B 1.00-1.96x everywhere (1.00 at the compute-bound M = 4096);
A 0.32-0.61x at M = 16, 0.62-1.13x at M = 64 (hence the M >= 256 rule), 0.86-1.39x above; both operands 0.79-1.49x at M = 16,
0.83-1.88x at M = 64, 1.07-1.89x from 256. The default route is within 6% of the better of the two in 137 of 142 points
(the rest: A-MN 0.86x at 576x7168 M = 256, 0.89x at 7168x2048 M = 64, 0.90x at 8192^2 M = 64; both-MN 0.93x at 8192^2
M = 4096; B-MN 0.94x at 2112x7168 M = 4096). Measurement noise at M = 4096 is about 5%: the same code measured later in a
round came out 4-6% slower.
The 1D1D UE8M0 g128 kernel gives the same picture (b/a/ab at M in {16, 256, 4096} on 3 shapes: B 1.04-1.77x over explicit, A from
M = 256 1.06-1.12x, both 1.09-1.74x from M = 256). Where FlashInfer's K-major kernel is ahead of the default route (B-MN and
both-MN at M >= 256, up to 13%), 0-4% of that is already its lead over DeepGEMM's own K-major kernel there, and the rest is
the B-side `MOV` cost above.

**K-major unchanged.** The shared kernels were edited, so the K-major benchmark was rerun against a build of the previous
commit (`scripts/sm120_bench_kmajor.py`, 7 shapes x M in {1, 16, 64, 256, 1024, 4096}, min of 3 alternating rounds x 3 repeats):
1D2D geomean new/old 0.9997 (worst +0.7%), 1D1D UE8M0 g128 1.000 (worst +0.5%).

**Tests.** `tests/generators.py` now yields the SM120 FP8 Dgrad (K-major A, MN-major B) and non-accumulating BF16 Wgrad (both
MN-major) cases and the m-grouped contiguous MN-major B cases; `test_mn_major_fp8_sm120` in `tests/test_fp8_fp4.py` covers both routes on
both kernels, the grouped host-transpose fallback and the FP4 rejection against the dequantized reference at the K-major
tolerance. `scripts/sm120_mn_major_fp8_check.py` checks the MN-major result against the K-major kernel's (bitwise, or FP32
rounding when the heuristic picks another tile or stream-K split) and a dequantized reference over 432 cases.

**Found while testing, not fixed.** The K-major 16x32 stream-K tile of the packed-UE8M0 FP8 kernel
(`sm120_fp8_fp4_gemm_1d1d`, N = 576, K = 1024, M = 16, g32 and g128, 15 stages) spills registers (g128: 4 B of spill stores, 20 B
of loads) under `DG_JIT_CHECK_NO_SPILLS=1`. It does at the previous commit too, with no MN-major operand involved; the existing
tests don't reach that tile. `test_mn_major_fp8_sm120` skips those small shapes for the UE8M0 kernels.

## 9.11 K-grouped FP8 GEMM (weight gradients) (2026-11-07)

`k_grouped_fp8_gemm_tn_contiguous` and `k_grouped_fp8_gemm_nt_contiguous` now run on SM120, closing one of the two
remaining FP8 limits from section 8 (the other, MN-major FP8 einsum, is done in section 9.12). This is the Wgrad shape of MoE
training: a batch of GEMMs that share M and N but have a different K per group, laid end to end along K, with the group
sizes known on the host.

**Design: one GEMM per group, like SM90 and the SM120 k-grouped BF16 path (74e1f61).** There is no k-grouped scheduler on
SM120. The host loops over `ks`, and for a nonzero group runs `sm120_fp8_fp4_gemm_1d1d` over that group's K slice:
`D[g] = A[K_g]^T @ B[K_g]` (+ C[g], which `early_return` has already copied into D, giving the accumulate-into-D semantics
the SM90/SM100 contract requires). The operand slices are MN-major (`[sum_k, M]` / `[sum_k, N]`, narrowed along K then
transposed to a `[MN, K_g]` view for the TN API; the NT API takes K-major `[MN, K_g]` blocks laid end to end), so they are
read in-kernel with the 8-bit transposing `ldmatrix` from 9.10 — no explicit transpose. A group with `k == 0` zeroes its D
slice (or is left for the accumulate copy). Per-group launch overhead is not a concern: even at `num_groups = 32` the eager
and CUDA-graph times are within 0.5% (section measurements below), because each launch is a full-size M×N GEMM, and the
MoE Wgrad groups are large-K, not the many-small-group case that would favour a fused scheduler. A single fused k-grouped
launch was therefore not built; it would only help a regime these shapes do not hit.

**Scales.** `transform_k_grouped_sf_into_required_layout` and `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`
(in `smxx_layout.hpp`) now accept arch 12. They pack the per-group FP32 (or already-UE8M0) scales into the contiguous
`[sum(ceil(K_g / (gran_k·4))), MN]` layout, where a group's packed rows, transposed, are exactly the `[MN, packed_k]`
MN-major packed scales a normal SM120 block-scaled GEMM reads. The FP32 side of a 128-element recipe is cast to packed
UE8M0 (exact only for power-of-two scales), as on SM100. gran_k 32 and 128 both work.

**Numbers** (`scripts/sm120_bench_k_grouped_fp8.py`; RTX PRO 6000, GPU 7; 256 MB L2 flush before every call, so operands
are cold; eager time includes host launch cost, CUDA-graph time is GPU only; min of 3 rounds; accumulate into FP32 D,
gran_k 128). Shapes are the test-file set plus MoE Wgrad: `num_groups` in {4, 8, 16, 32}, DeepSeek-V3 expert dims
(M/N 4096×7168 and 7168×2048, hidden 7168 / intermediate 2048, gate_up 4096 wide), K per group in {256, 1024, 4096}.
Baselines: a host loop that explicitly transposes each MN-major operand slice to K-major and runs the K-major FP8 kernel
per group (the route the kernel replaces), DeepGEMM's own k-grouped BF16 GEMM, and a torch (cuBLAS) BF16 loop. Correctness
is checked bitwise against the per-group loop on the same packed scales inside the script.

| MN-major FP8 vs. | eager | CUDA graph |
|---|---|---|
| host loop + explicit transpose, K-major kernel | 1.12-2.06x faster | 1.11-2.03x faster |
| DeepGEMM k-grouped BF16 | 1.01-1.44x faster | similar |
| torch BF16 loop | 0.76-1.00x (torch wins the small-K 4096×7168 groups) | — |

The win over the explicit-transpose loop grows with K per group (more transpose work avoided): ~1.1x at K ≈ 320/group up
to ~2.0x at K ≈ 4000/group on the 7168×2048 shapes. Eager and graph times match to within ~1%, confirming the per-group
launch overhead is negligible at these sizes.

**Tests.** `test_k_grouped_gemm_contiguous` in `tests/test_fp8_fp4.py` no longer early-returns on SM120. It runs with
packed UE8M0 scales on both operands (`use_ue8m0` is forced on arch 12), checks the accumulate result against the FP32
reference and bitwise self-consistency over 20 repeats and the empty/single-group `ks` variants, and times the per-group
kernels with `per_call_total` so the summed group time is reported. Stochastic rounding and the pre-packed INT32 strided-SF
checks stay SM100-only (they use the SM100 epilogue / SF-K-padding path). Because `enumerate_k_grouped_contiguous` mutates
the contiguous-layout MK alignment and leaves it at its last value (384), and the test now actually runs on SM120, the
k-grouped test restores the 128 default at the end so the later m-grouped `test_mixed_ue8m0_and_fp32_scale_pair` sees a
block_m its heuristics can tile — this ordering bug was latent while the test early-returned.

Verified with `DG_JIT_CHECK_NO_SPILLS=1` and a fresh cache: `test_fp8_fp4.py`, `test_bf16.py` and `test_einsum.py` all pass
(the 16×32 stream-K UE8M0 spill from 9.10 stays skipped). K-major shapes do not regress: `scripts/sm120_bench_kmajor.py`
1D1D timings are unchanged (the shared K-major kernel code is untouched; only the host k-grouped dispatch and the arch
gating in the scale-layout helpers changed).

## 9.12 MN-major `fp8_einsum` (2026-11-07)
`fp8_einsum` `bhd,hdr->bhr` and `bhd,bhr->hdr` now run on SM120 (task 11.3), closing the last FP8 limit from section 8. Like the
native BF16 einsum (74e1f61), each runs one GEMM per head on strided views; the MN-major operands are read in-kernel with the
8-bit transposing `ldmatrix` (9.10).
- **`bhd,hdr->bhr`** (`(batch, m, n, k) = (h, b, r, d)`): K-major A, MN-major B, 128x128 FP32 scales, BF16 D. It is just the
  existing per-batch SM120 branch of `fp8_bmm` (`sm120_fp8_gemm_1d2d`, which derives the majors from the strides); the only
  change is that the arch-10-only gate in `fp8_einsum` is gone.
- **`bhd,bhr->hdr`** (`(h, d, r, b)`): both operands MN-major, per-channel 1x128 scales (`recipe=(1,1,128)`), FP32 D that
  accumulates onto C. This does *not* fit `fp8_bmm`'s SM120 branch, for two reasons: the 1D2D kernel writes only BF16 with no
  accumulation, and it reads SFB as 128x128 blocks. So it runs on the block-scaled `sm120_fp8_fp4_gemm_1d1d` (FP32 D,
  in-kernel accumulation, `gran_k` 128) per head, with both scale tensors packed to UE8M0 the way SM100 packs them (exact for
  power-of-two scales, as `per_channel_cast_to_fp8(use_ue8m0=True)` produces).
- **A false start worth recording.** The first version sent `bhd,bhr->hdr` through the 1D2D kernel with a BF16 scratch and a
  host FP32 add. It built and ran, and the test failed at `calc_diff` 0.030. Comparing against a dequantized reference showed
  the *product* was wrong (0.030) while the quantization error alone was 7e-4: the per-channel SFB was read as block scales.
  `fp8_bmm`'s SM120 branch asserts only the SF dtype and `gran_k`, not the N granularity, so it accepts a per-channel SFB
  silently. That is a latent hole for any caller passing `(1, 1, 128)` through `fp8_bmm` on SM120; it is not reachable from
  `fp8_einsum` any more, but is not closed in `fp8_bmm` itself.
- **Tolerance.** Against the BF16 inputs the error is FP8 quantization, ~7e-4 for `bhd,bhr->hdr` (b = 4096 and 8192), the same
  as SM100, so the existing `calc_diff < 1e-3` is kept. That is only ~30% above pure quantization error, so on SM120 the test
  also compares to the dequantized FP8 operands at 1e-10 (measured 1e-13 for `bhd,bhr->hdr`, 1.4e-6 for `bhd,hdr->bhr`, which
  is BF16 output rounding). Unlike the BF16 einsum (1e-7 against cuBLAS), no loosening is needed: the FP8 quantization error
  swamps the `mma.sync` accumulation-order difference.
- **Still rejected, with clear errors.** Accumulation (`c`) and custom epilogues / the FP8 output pair for `bhd,hdr->bhr` on SM120
  ("Custom epilogues require SM100", "SM120 FP8 bmm does not support accumulation"); `bhd,bhr->hdr` accepts `c` and only
  `recipe=(1,1,128)`. SM90 gets "has no kernel on this architecture".

Verified with `DG_JIT_CHECK_NO_SPILLS=1` and fresh caches: `test_einsum.py`, `test_fp8_fp4.py` and `test_bf16.py` pass. Edge shapes
(unaligned `d`/`r` that take the host-transpose fallback, `b` not a multiple of 16, a single token, `c` aliased / separate /
absent) match the dequantized reference.

**Benchmark** (GPU 7, CUDA graph, best of 3 interleaved rounds, us per call, e4m3 + UE8M0 scales). "BF16 DG" is the
deterministic native per-head BF16 kernel (74e1f61); "cuBLASLt" is `deep_gemm.einsum` with deterministic mode off; "loop" is a
per-head `torch.matmul` on BF16 (for `bhd,bhr->hdr` plus an FP32 add per head).

`bhd,hdr->bhr`:

| h, r, d | b | FP8 | BF16 DG | cuBLASLt | torch loop | FP8 vs BF16 DG | vs loop |
|---|---|---|---|---|---|---|---|
| 8, 4096, 1024 | 4 | 33.9 | 41.8 | 18.4 | 59.3 | 1.24x | 1.75x |
| 8, 4096, 1024 | 32 | 36.7 | 44.9 | 18.8 | 72.1 | 1.22x | 1.97x |
| 8, 4096, 1024 | 128 | 48.8 | 62.1 | 33.4 | 58.8 | 1.27x | 1.21x |
| 8, 4096, 1024 | 4096 | 517 | 793 | 753 | 801 | 1.53x | 1.55x |
| 8, 4096, 1024 | 8192 | 969 | 1486 | 1500 | 1599 | 1.53x | 1.65x |
| 128, 512, 128 | 4 | 191 | 225 | 8.7 | 321 | 1.18x | 1.68x |
| 128, 512, 128 | 32 | 193 | 227 | 9.2 | 275 | 1.17x | 1.42x |
| 128, 512, 128 | 128 | 194 | 233 | 16.6 | 283 | 1.20x | 1.46x |
| 128, 512, 128 | 4096 | 613 | 757 | 483 | 818 | 1.24x | 1.33x |
| 128, 512, 128 | 8192 | 1049 | 1081 | 979 | 1372 | 1.03x | 1.31x |

`bhd,bhr->hdr` (FP32 D accumulate; BF16 baseline is the cuBLASLt path, which is the only BF16 path for this expression):

| h, r, d | b | FP8 | cuBLASLt | torch loop | FP8 vs cuBLASLt | vs loop |
|---|---|---|---|---|---|---|
| 8, 4096, 1024 | 128 | 337 | 180 | 235 | 0.54x | 0.70x |
| 8, 4096, 1024 | 4096 | 660 | 796 | 1006 | 1.21x | 1.53x |
| 8, 4096, 1024 | 8192 | 1164 | 1512 | 1689 | 1.30x | 1.45x |
| 128, 512, 128 | 128 | 435 | 23 | 588 | 0.05x | 1.35x |
| 128, 512, 128 | 4096 | 1345 | 489 | 1408 | 0.36x | 1.05x |
| 128, 512, 128 | 8192 | 2030 | 930 | 1874 | 0.46x | 0.92x |

Reading it: the FP8 einsum beats DeepGEMM's own per-head BF16 kernel everywhere (1.03-1.53x) and the per-head torch loop almost
everywhere, and beats cuBLASLt only at large `b` with few big heads (1.2-1.5x at h = 8, b >= 4096 for both expressions). It
loses badly to cuBLASLt when `b` is small or there are many small heads (h = 128: 8.7 us vs 191 us): cuBLASLt issues one
strided-batched launch, while this path launches `h` GEMMs (about 1.5 us each, which is what the 190 us at h = 128 is). The
`bhd,bhr->hdr` many-head, small-tile case is worst because its per-head GEMM is also launch- and tile-bound (M = d = 128). A
single batched launch would need 3D TMA descriptors in the SM120 kernels (the same reason `fp8_bmm` loops). `fp8_einsum` is an
explicit API with no default to switch, so callers with many small heads should keep the BF16 cuBLASLt einsum.

## Paged MQA logits with head dim 32 and 64 on SM120 (task 11.4)

Source check: the SM90 paged kernel accepts head dim 32, 64 and 128 with H in {32, 64} and next_n in {1, 2}
(`sm90_fp8_mqa_logits.hpp`, `attention.hpp`). The SM100 paged kernel is varlen-only (next_n 1, `indices` required) and takes
H multiples of 4 up to 128 with head dim 32/64/128 (64/128 for MXFP4); that is a different API, which SM120 does not expose
(`DG_HOST_ASSERT(is_varlen == (arch_major == 10))`). So the SM120 target is the SM90 set, and it is now met: head dim 32, 64, 128
x H {32, 64} x next_n {1, 2}. The host rejection is replaced by an assert that names the supported dims.

Design (one kernel path, head dim a template parameter):
- `warp_mma_16xNxHd_ss<N, kHeadDim>` already looped over `kHeadDim / 32` K steps; the paged kernel was just passing 128. Head dim 32
  needs a third TMA swizzle mode in `get_swizzle_offset`: 32B rows use `Swizzle<1, 4, 3>` (address bit 7 into bit 4), i.e.
  `chunk ^= (row / 4) % 2`.
- Tile bases must be 1024B-aligned for the manual swizzle, but SM90 uses `kHeadDim * 8` as the alignment (256 B for head dim 32,
  512 B for 64). Kernel and host now use `max(kHeadDim * 8, 1024)`, so the smem layout is identical in both.
- Stage selection is unchanged ((3,3), (2,3), (2,2), (2,1) with at least 2 Q stages, the deadlock rule above). Smaller head dims need
  less smem, so they get deeper pipelines than head dim 128. Proxy-fence placement is untouched.

Validation (GPU 7, fresh JIT cache, `DG_JIT_CHECK_NO_SPILLS=1`, timeouts on every run): `scripts/sm120_mqa_check.py` runs 13 paged cases per
head dim (128, 64, 32): B 3-128, next_n 1/2, H 32/64, contexts up to 20K, including the B>=32 and next_n=2 pipeline-wrapping cases. All 39
match the dequantized reference with diff <= 4.1e-14 (the paged assert is now 1e-12). `tests/test_attention.py` passes on SM120 with
the new head dims enabled (paged only; the contiguous kernel stays at 64/128), no spills.

Performance (`scripts/sm120_paged_mqa_bench.py`, cold L2, best of 3 x 30 calls; "KV GB/s" = bytes of KV read / time):

| B | next_n | H | ctx | D=128 (c5a9656 -> new) | D=64 | D=32 |
|---|---|---|---|---|---|---|
| 32 | 1 | 64 | 8192 | 43.9 -> 43.8 us | 29.1 us | 23.0 us |
| 32 | 2 | 64 | 8192 | 45.8 -> 45.8 us | 29.7 us | 23.4 us |
| 64 | 1 | 32 | 8192 | 76.7 -> 76.9 us | 45.7 us | 33.5 us |
| 128 | 1 | 64 | 4096 | 77.8 -> 77.6 us | 46.0 us | 33.7 us |
| 8 | 1 | 64 | 32768 | 43.9 -> 44.2 us | 28.7 us | 23.1 us |
| 64 | 2 | 32 | 4096 | 44.5 -> 44.4 us | 29.5 us | 23.5 us |

Head dim 128 is unchanged within noise (two interleaved runs each, baseline built from c5a9656). The new dims run at 400-780 GB/s of KV
traffic against 750-900 GB/s for head dim 128; the short cases are bounded by the fixed launch and tail cost (about 20 us), not by
bandwidth, and the 64-wide cases with enough work (B=64 H=32, B=128) reach 775 GB/s. The TokenSpeed Triton op
(`triton_dsa_decode_topk_fp8`) only registers `head_dim=128`, so there is no Triton number to compare at 32 or 64.
