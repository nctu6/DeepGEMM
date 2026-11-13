# Serving benchmark of two TokenSpeed images on RTX PRO 6000 (SM120)

Tasks 40-48 of the SM120 spec. Served on 2026-11-12 on aws-raichu, GPUs 6 and 7. This is the detailed record.
The short version is in [`deepgemm_report.md`](deepgemm_report.md) section 3, and the status-log entries are in
[`deepgemm-analysis.md`](deepgemm-analysis.md) 9.44 and 9.45.

The question: what do the SM120 kernel and runtime changes (DeepGEMM SM120 pin, CuTe DSL / Triton attention
leaves, NVFP4 and V4.1 fixes) change for served throughput and latency? Two images, six models, two client
workloads, the same server flags in both images.

Every number below comes from the run's `metrics.json` (AIPerf's `profile_export_aiperf.json`, summarized by
the harness) or from the `DONE` / `FAILED` / `SESSION_FAILED` markers, `session.json` and `gpu_at_launch.txt`.
The tables are generated (section 8). Server log lines are quoted with their path.

## 1. Summary

Ratio convention: output tok/s is after over before, and TTFT p50 and TPOT p50 are before over after, so a
value above 1.00x always means the after image is better. One run per point, no repeats. Cells are
`before / after (ratio)`. "after only" rows have no before value, because the before image does not start
that model (section 4.1). The last row compares the two MoE backends of V4.1 on the after image
(`Marlin / mega_moe`, ratio = mega_moe against Marlin). TTFT and TPOT are in ms.


RAG:

| Model | C=1 | C=16 | C=32 | C=64 | C=128 | C=256 | C=512 |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 (TP1) | out tok/s 31.3 / 47.7 (1.52x)<br>TTFT p50 103 / 103 (1.00x)<br>TPOT p50 31.6 / 20.6 (1.53x) | out tok/s 402.9 / 515.3 (1.28x)<br>TTFT p50 970 / 979 (0.99x)<br>TPOT p50 34.8 / 26.6 (1.31x) | out tok/s 709.3 / 795.1 (1.12x)<br>TTFT p50 1796 / 1803 (1.00x)<br>TPOT p50 37.4 / 32.6 (1.15x) | out tok/s 962.2 / 1066.2 (1.11x)<br>TTFT p50 2822 / 2804 (1.01x)<br>TPOT p50 53.2 / 47.1 (1.13x) | out tok/s 1219.4 / 1282.8 (1.05x)<br>TTFT p50 5798 / 5915 (0.98x)<br>TPOT p50 80.4 / 75.5 (1.06x) | out tok/s 1275.4 / 1324.7 (1.04x)<br>TTFT p50 11421 / 11276 (1.01x)<br>TPOT p50 127.8 / 120.1 (1.06x) | out tok/s 1275.8 / 1323.9 (1.04x)<br>TTFT p50 43074 / 42539 (1.01x)<br>TPOT p50 129.8 / 125.1 (1.04x) |
| Qwen3.6-35B-A3B BF16 (TP1, control) | out tok/s 168.2 / 170.1 (1.01x)<br>TTFT p50 64 / 64 (1.00x)<br>TPOT p50 5.7 / 5.6 (1.01x) | out tok/s 712.5 / 721.0 (1.01x)<br>TTFT p50 296 / 292 (1.01x)<br>TPOT p50 21.0 / 20.8 (1.01x) | out tok/s 960.2 / 962.1 (1.00x)<br>TTFT p50 517 / 515 (1.00x)<br>TPOT p50 31.0 / 30.8 (1.00x) | out tok/s 1358.1 / 1360.2 (1.00x)<br>TTFT p50 862 / 824 (1.05x)<br>TPOT p50 43.0 / 42.9 (1.00x) | out tok/s 1936.8 / 1938.5 (1.00x)<br>TTFT p50 1667 / 1506 (1.11x)<br>TPOT p50 58.4 / 58.4 (1.00x) | out tok/s 1979.9 / 1977.7 (1.00x)<br>TTFT p50 3838 / 3900 (0.98x)<br>TPOT p50 67.2 / 67.1 (1.00x) | out tok/s 1968.9 / 1972.7 (1.00x)<br>TTFT p50 20951 / 20823 (1.01x)<br>TPOT p50 67.5 / 66.9 (1.01x) |
| gemma-4-31B-it BF16 (TP1) | out tok/s 22.2 / 23.0 (1.03x)<br>TTFT p50 175 / 173 (1.01x)<br>TPOT p50 44.4 / 43.0 (1.03x) | out tok/s 256.6 / 263.3 (1.03x)<br>TTFT p50 633 / 648 (0.98x)<br>TPOT p50 58.8 / 57.2 (1.03x) | out tok/s 365.8 / 371.4 (1.02x)<br>TTFT p50 1428 / 1514 (0.94x)<br>TPOT p50 73.6 / 73.0 (1.01x) | out tok/s 369.5 / 370.8 (1.00x)<br>TTFT p50 22836 / 22735 (1.00x)<br>TPOT p50 73.8 / 73.3 (1.01x) | out tok/s 364.8 / 369.3 (1.01x)<br>TTFT p50 64409 / 64567 (1.00x)<br>TPOT p50 73.7 / 73.6 (1.00x) | out tok/s 365.8 / 371.6 (1.02x)<br>TTFT p50 148361 / 146598 (1.01x)<br>TPOT p50 73.6 / 73.3 (1.00x) | out tok/s 361.8 / 370.9 (1.03x)<br>TTFT p50 171128 / 166253 (1.03x)<br>TPOT p50 74.2 / 73.0 (1.02x) |
| Qwen3.8-27B-NVFP4 (TP1) **after only** | out tok/s 55.3<br>TTFT p50 74<br>TPOT p50 17.7 | out tok/s 618.2<br>TTFT p50 729<br>TPOT p50 22.4 | out tok/s 966.1<br>TTFT p50 1346<br>TPOT p50 27.5 | out tok/s 1320.6<br>TTFT p50 2217<br>TPOT p50 38.8 | out tok/s 1554.5<br>TTFT p50 3873<br>TPOT p50 64.6 | out tok/s 1609.5<br>TTFT p50 7733<br>TPOT p50 114.6 | out tok/s 1611.6<br>TTFT p50 36700<br>TPOT p50 117.2 |
| Qwen3.5-122B-A10B-NVFP4 (TP2) **after only** | out tok/s 133.4<br>TTFT p50 83<br>TPOT p50 7.2 | out tok/s 802.6<br>TTFT p50 648<br>TPOT p50 17.0 | out tok/s 1098.1<br>TTFT p50 984<br>TPOT p50 24.8 | out tok/s 1480.1<br>TTFT p50 1527<br>TPOT p50 35.9 | out tok/s 1912.2<br>TTFT p50 2947<br>TPOT p50 53.6 | out tok/s 2281.5<br>TTFT p50 5606<br>TPOT p50 86.7 | out tok/s 2255.7<br>TTFT p50 10803<br>TPOT p50 115.1 |
| DeepSeek-V4.1-Flash 128-of-384 experts, Marlin `auto` (TP2) **after only** | out tok/s 104.5<br>TTFT p50 98<br>TPOT p50 9.0 | out tok/s 612.8<br>TTFT p50 658<br>TPOT p50 22.8 | out tok/s 974.9<br>TTFT p50 815<br>TPOT p50 29.4 | out tok/s 1429.0<br>TTFT p50 1199<br>TPOT p50 39.5 | out tok/s 1901.0<br>TTFT p50 2045<br>TPOT p50 57.9 | out tok/s 1649.8<br>TTFT p50 3185<br>TPOT p50 139.8 | out tok/s 2692.5<br>TTFT p50 6547<br>TPOT p50 160.2 |
| DeepSeek-V4.1-Flash 128-of-384 experts, `mega_moe` (TP2): Marlin / mega_moe, both after image | out tok/s 104.5 / 106.1 (1.01x)<br>TTFT p50 98 / 93 (1.05x)<br>TPOT p50 9.0 / 8.9 (1.01x) | out tok/s 612.8 / 626.3 (1.02x)<br>TTFT p50 658 / 634 (1.04x)<br>TPOT p50 22.8 / 22.2 (1.03x) | out tok/s 974.9 / 1003.7 (1.03x)<br>TTFT p50 815 / 770 (1.06x)<br>TPOT p50 29.4 / 28.6 (1.03x) | out tok/s 1429.0 / 1522.3 (1.07x)<br>TTFT p50 1199 / 1087 (1.10x)<br>TPOT p50 39.5 / 37.0 (1.07x) | out tok/s 1901.0 / 2228.9 (1.17x)<br>TTFT p50 2045 / 1756 (1.16x)<br>TPOT p50 57.9 / 49.7 (1.17x) | out tok/s 1649.8 / 1717.8 (1.04x)<br>TTFT p50 3185 / 3166 (1.01x)<br>TPOT p50 139.8 / 134.0 (1.04x) | out tok/s 2692.5 / 2788.3 (1.04x)<br>TTFT p50 6547 / 5738 (1.14x)<br>TPOT p50 160.2 / 157.0 (1.02x) |

agentX:

| Model | C=1 | C=16 | C=64 |
|---|---|---|---|
| Qwen3.8-27B-FP8 (TP1) | out tok/s 21.2 / 24.2 (1.14x)<br>TTFT p50 3265 / 2703 (1.21x)<br>TPOT p50 39.2 / 27.8 (1.41x) | out tok/s 29.0 / 31.8 (1.10x)<br>TTFT p50 1051 / 2519 (0.42x)<br>TPOT p50 94.1 / 83.0 (1.13x) | out tok/s 10.2 / 10.3 (1.01x)<br>TTFT p50 398631 / 398626 (1.00x)<br>TPOT p50 243.9 / 225.8 (1.08x) |
| Qwen3.6-35B-A3B BF16 (TP1, control) | out tok/s 71.7 / 73.0 (1.02x)<br>TTFT p50 709 / 733 (0.97x)<br>TPOT p50 12.4 / 11.9 (1.04x) | out tok/s 108.1 / 101.1 (0.94x)<br>TTFT p50 848 / 871 (0.97x)<br>TPOT p50 22.7 / 21.6 (1.05x) | out tok/s 43.7 / 44.9 (1.03x)<br>TTFT p50 293753 / 291473 (1.01x)<br>TPOT p50 57.0 / 53.9 (1.06x) |
| gemma-4-31B-it BF16 (TP1) | out tok/s 16.0 / 16.0 (1.00x)<br>TTFT p50 2254 / 2236 (1.01x)<br>TPOT p50 52.0 / 50.5 (1.03x) | FAILED / FAILED | FAILED / FAILED |
| Qwen3.8-27B-NVFP4 (TP1) **after only** | out tok/s 24.2<br>TTFT p50 2581<br>TPOT p50 25.0 | out tok/s 40.2<br>TTFT p50 1009<br>TPOT p50 67.8 | out tok/s 8.6<br>TTFT p50 404731<br>TPOT p50 147.1 |
| Qwen3.5-122B-A10B-NVFP4 (TP2) **after only** | out tok/s 63.7<br>TTFT p50 919<br>TPOT p50 13.3 | out tok/s 118.6<br>TTFT p50 720<br>TPOT p50 18.8 | out tok/s 14.9<br>TTFT p50 123098<br>TPOT p50 583.7 |
| DeepSeek-V4.1-Flash 128-of-384 experts, Marlin `auto` (TP2) **after only** | out tok/s 89.6<br>TTFT p50 187<br>TPOT p50 11.3 | out tok/s 150.8<br>TTFT p50 190<br>TPOT p50 11.5 | out tok/s 524.2<br>TTFT p50 299<br>TPOT p50 34.3 |
| DeepSeek-V4.1-Flash 128-of-384 experts, `mega_moe` (TP2): Marlin / mega_moe, both after image | out tok/s 89.6 / 86.7 (0.97x)<br>TTFT p50 187 / 210 (0.89x)<br>TPOT p50 11.3 / 10.9 (1.04x) | out tok/s 150.8 / 151.1 (1.00x)<br>TTFT p50 190 / 174 (1.09x)<br>TPOT p50 11.5 / 11.2 (1.03x) | out tok/s 524.2 / 541.6 (1.03x)<br>TTFT p50 299 / 299 (1.00x)<br>TPOT p50 34.3 / 32.1 (1.07x) |

Headlines:

- Qwen3.8-27B-FP8 (block FP8) is the only model whose speed changes. RAG output tok/s goes up 1.52x at C=1,
  1.28x at C=16 and 1.11-1.12x at C=32-64, then 1.04-1.05x from C=128. The gain is decode: TPOT p50 is
  31.6 -> 20.6 ms at C=1 (1.53x), 1.04x at C=512. TTFT p50 stays at 0.98-1.01x. agentX: output 1.14x / 1.10x /
  1.01x and TPOT p50 1.41x / 1.13x / 1.08x at C = 1 / 16 / 64.
- The BF16 control (Qwen3.6-35B-A3B, no FP8/FP4 GEMM) and gemma-4-31B-it are at 1.00-1.03x for RAG output
  tok/s and TPOT p50 at every C. So the FP8 gain is not a host, image or GPU effect. In agentX the control's
  TPOT p50 is also 1.04-1.06x better on the after image (cause not investigated), so part of the FP8 agentX
  TPOT ratio is not FP8.
- Three models run only on the after image. Qwen3.8-27B-NVFP4 (TP1), Qwen3.5-122B-A10B-NVFP4 (TP2) and the
  V4.1 128-expert subset (TP2) all exit during startup on the before image (section 4.1).
- V4.1 subset, mega_moe against Marlin (both on the after image): RAG 1.01-1.07x up to C=64, 1.17x at C=128,
  1.04x at C=256 and 512. agentX 0.97x / 1.00x / 1.03x.
- At high C every server saturates for the same reasons in both images: a full KV pool, and decode batches
  above the largest CUDA graph (160; fixed for the default in task 53, section 3.8). V4.1 RAG dipped at C=256 (section 3.6). gemma agentX C=16/64 cannot run
  at TP1 with the default args (the 256K traces exceed `max_req_input_len=199614`, section 4.2); with
  `--kv-cache-dtype fp8` or `--chunked-prefill-size 1024` the limit is 262,143 and C=16/64 run (section 3.3, task 52).
- With the servicer's default config (KV events on), the scheduler aborts in both images under RAG load at
  C=128. Every primary result uses `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` (section 5).

## 2. Testing conditions

### 2.1 Hardware and host

| item | value | source |
|---|---|---|
| host | aws-raichu, 2x Intel Xeon Platinum 8559C (192 threads), 2 TB RAM, Ubuntu 24.04.4 LTS, kernel 7.0.0-1014-aws, Docker 29.1.3 | `lscpu`, `free`, `/etc/os-release`, `docker --version` |
| GPUs | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, compute capability 12.0), 97887 MiB each, PCIe only (no NVLink) | `nvidia-smi` |
| GPUs used | 6 and 7 only: same PCIe switch (`PIX`), NUMA node 1, CPU affinity 48-95,144-191. Other users' jobs ran on GPUs 0-5 | `nvidia-smi topo -m` |
| driver / CUDA | driver 595.99.02, CUDA 13.2 (driver API). The images ship the CUDA 13.0 runtime (`nvidia-cuda-runtime` 13.0.96, `torch 2.14.0+cu130`) | `nvidia-smi`, `pip list` in the images |
| GPU state at launch | 0 MiB used and no compute process on the session's GPUs at every launch | `gpu_at_launch.txt` of every session |

The driver and host values were read on 2026-11-13, the day after the runs. I did not record them during the
runs, and nothing was changed on the host in between as far as I know.

### 2.2 Images and versions

| | before | after |
|---|---|---|
| tag | `nctu6/tokenspeed:e27cee86-20261028` | `nctu6/tokenspeed:ef057889-20261112` |
| image ID | `sha256:2b655629ac34c6723dc8ae6cf3c54a5ee67e354bd1f75fa485f2c5f179bce9d4` | `sha256:349e1a2b79973c6446bb5ece04d7e6cf2b91cb110e0ba7fd0193e93d92c300bf` |
| created | 2026-10-28 10:17:53 UTC | 2026-11-12 02:45:29 UTC |
| TokenSpeed | `e27cee861` (2026-10-28, "feat(deepseek-v4.1-flash): default to multimodal at TP=4") | `ef057889f` (2026-11-12, "build: keep tmp/ scratch folders out of the Docker build context"), 153 commits later |
| DeepGEMM | `tokenspeed-deepgemm 2.8.0.post20260924` wheel (no SM120 support; no DeepGEMM submodule in that tree) | `deep_gemm 2.8.2+local`, built from the TokenSpeed `DeepGEMM` submodule pin `976f212` ("docs: tasks 1-39 are pushed") |
| tokenspeed-scheduler | 0.1.41, from the tree's `tokenspeed-scheduler/` (last change `50bd216dd`) | 0.1.41, from the tree's `tokenspeed-scheduler/` (last change `b39d7ccc9`, pre-commit re-wrapping only, so the scheduler code is the same) |
| smg (gateway) submodule | `b545fb85` | `b545fb85` |
| tokenspeed-kernel | 0.1.4.dev20261028 | 0.1.4.dev20261112 |
| other | torch 2.14.0+cu130, flashinfer-python 0.7.2, triton 3.8.0, cuda-bindings 13.4.4, tokenspeed-deepep 2.1.0.post20260810, tokenspeed-deepselect 1.0.0.post20260910 | same, except cuda-bindings 13.5.0 |

How the commits were checked: the images carry no `.git`, so the commit comes from the tag. I compared the
MD5 of `python/pyproject.toml`, `python/tokenspeed/runtime/models/deepseek_v41.py`,
`tokenspeed-scheduler/csrc/scheduler/scheduler.cpp` (and `DeepGEMM/deepgemm_report.md` for the after image) in
`/workspace` of each image, and of the installed `deepseek_v41.py`, against `git show <commit>:<path>`. All of
them match `e27cee861` and `ef057889f` / `976f212`. The submodule pins and the last scheduler change come from
`git ls-tree` / `git log` in the TokenSpeed repo. The package versions come from `importlib.metadata` /
`pip list` in each image. The scheduler line in the crash message (385 after, 386 before) differs because of the
re-wrapping in `b39d7ccc9`.

### 2.3 Server command per model

One `docker run` per (image, model) server. The arguments are identical in both images (the `server_cmd.txt` of
every session):

```bash
docker run --rm --init -d --name bench-nctu6-g<gpus>-<model>-<image>-<variant> --gpus "device=<gpus>" --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 127.0.0.1:<18061+gpu>:8000 \
  -v /models:/models:ro -v /unieai/.cache/huggingface:/root/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e SMG_TOKENSPEED_SERVICER_KV_EVENTS=0 \
  nctu6/tokenspeed:<tag> \
  tokenspeed serve <model path> --served-model-name <name> --host 0.0.0.0 --port 8000 --trust-remote-code \
    --tensor-parallel-size <tp> --max-num-seqs 512 --gpu-memory-utilization 0.9 --max-model-len 262144 <extra>
```

| model (served name) | path | TP, GPUs | extra arguments | probe |
|---|---|---|---|---|
| `qwen38-27b-fp8` | `/models/Qwen/Qwen3.8-27B-FP8` | 1 | - | gated |
| `qwen36-35b-a3b` | `/models/Qwen/Qwen3.6-35B-A3B` | 1 | - | gated |
| `gemma4-31b-it` | `/models/google/gemma-4-31B-it` | 1 | `--tool-call-parser gemma4`; prompts get a literal `<bos>` prefix (client side, section 2.5) | gated |
| `qwen38-27b-nvfp4` | `/models/nvidia/Qwen3.8-27B-NVFP4` | 1 | - | gated |
| `qwen35-122b-a10b-nvfp4` | `/models/nvidia/Qwen3.5-122B-A10B-NVFP4` | 2, GPUs 6,7 | - | gated |
| `dsv41-flash-1017` | `/models/deepseek-ai/DeepSeek-V4.1-Flash-1017` | 2, GPUs 6,7 | `--engram-host-table --engram-host-table-layout sharded --hf-overrides '{"n_routed_experts": 128, "num_checkpoint_experts": 384}'` (the first 128 of 384 routed experts per layer, needed to fit two GPUs) | recorded, not gated |
| same, variant `mega_moe` (after only) | same | 2, GPUs 6,7 | the above plus `--moe-backend mega_moe` (attention TP2 + MoE TP2) | recorded, not gated |

Effective engine settings, from the `server_args=ServerArgs(...)` line of each `server.log`. They are the same in
both images and for every model unless noted:

| setting | value |
|---|---|
| `max_num_seqs` / `gpu_memory_utilization` / `max_model_len` | 512 / 0.9 / 262144 |
| CUDA graphs | on. `cudagraph_capture_size=160`, `cudagraph_capture_sizes=None` (the default), `cuda_graph_padding=False`. Every server that started logs `Capturing batches: [160, 152, 144, 104, 40]`, so no decode graph above batch 160 while `--max-num-seqs` is 512. The prefill breakable graph is captured as well (`prefill_graph.py` "captured buckets") |
| KV cache dtype | `kv_cache_dtype='auto'` (the model dtype: BF16. The FP8 27B log builds FlashInfer `..._dtype_kv_bf16_...` modules) |
| prefix cache / L2 host cache | `enable_prefix_caching=True`, `disable_kvstore=False` (L2 host cache and its writeback on) |
| KV events | `kv_events_config=None` (with the env variable set; without it the servicer turns them on, section 5) |
| chunked prefill | `chunked_prefill_size=8192` |
| MoE backend | `moe_backend='auto'` (V4.1: Marlin), `'mega_moe'` for the V4.1 variant. `ep_size=1` |
| attention backend | `attention_backend=None` (automatic selection) |
| quantization | `quantization=None` (taken from the checkpoint) |
| speculative decoding | off (`speculative_algorithm=None`) |
| reasoning parser | `None`. V4.1: `'deepseek_v41'` (chosen from the model) |
| tool-call parser | the gateway default, except gemma-4 (`--tool-call-parser gemma4`, section 4.4) |
| container env | `HF_HUB_OFFLINE=1`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` |

Engine capacity at startup (the `TokenSpeed engine ready:` line; equal in both images where both start):

| model | `max_total_num_tokens` | `max_req_input_len` |
|---|---|---|
| Qwen3.8-27B-FP8 | 873,984 | 262,143 |
| Qwen3.6-35B-A3B | 829,440 | 262,143 |
| gemma-4-31B-it | 293,760 | 199,614 (`Single-request token limit: cache=199615 model=262144 effective=199615 max_input=199614`) |
| Qwen3.8-27B-NVFP4 (after) | 1,010,688 | 262,143 |
| Qwen3.5-122B-A10B-NVFP4 (after) | 3,629,056 | 262,143 |
| V4.1 subset, Marlin (after) | 24,465,920 | 262,143 |
| V4.1 subset, mega_moe (after) | 23,391,232 | 262,143 |

### 2.4 Probe and warmup before the measurements

After `/health` and `/v1/models` answer 200 (polled every 5 s, the startup time in section 2.6), the harness
sends three greedy requests (`temperature 0`, 24 tokens) and writes them to `probe.json`:

- `/v1/completions` "Question: What is the capital of France?\nAnswer:" must match `Paris`;
- `/v1/completions` "The first ten prime numbers are: 2, 3, 5," must match `\b7\b.*\b11\b.*\b13\b`;
- `/v1/chat/completions` "What is the capital of France?" (recorded only).

For gated sessions a failed completion probe stops the session. Every gated session that started passed both,
for example FP8 27B "\n\nParis." and " 7, 11, 13, 17, 19, 23, and". The V4.1 subset is recorded only: its
output is not meaningful with 128 of 384 experts (" Paris is the capital of France..." but primes
" 7, 9, 11, 13, 15, ...").

Warmup: the harness sends no client warmup requests for RAG (no AIPerf warmup option). The first traffic on a
fresh server is the three probes, and each later RAG point runs on the server the previous point warmed. The
engine's own startup warmup (CUDA graph capture, JIT) runs before `/health`. agentX runs AIPerf's scenario cache
warmup (one request per trajectory lane, no timeout of its own) before the 900 s window; its length per run is
`warmup s` in the agentX tables (6-2933 s).

Between runs the harness probes the engine once with a 1-token completion (`Server.check(strict=True)`). During a
run a watchdog polls the server log every 5 s for fatal lines and probes when the log is quiet for 120 s. On a
dead engine it kills the client, marks the run `FAILED` (`engine crash`) and restarts the server (at most 3
restarts per session). No primary session needed a restart (`restarts: 0` in every `session.json`).

### 2.5 Clients

Client: AIPerf 0.13.0 (`metrics.json` `aiperf_version`), one venv for every run, on the same host, against
`http://127.0.0.1:<port>`.

RAG. Prompts are built once by `build_rag_prompts.py` from the local HF cache snapshot of
`neural-bridge/rag-dataset-12000` (`586c0d0d...`, train split, read-only, no network). 500 rows are sampled with
`random.Random(0).sample`, the prompt is `Context: <context>\nQuestion: <question>\nAnswer:`, and the output is
`data/rag-12000-500-seed0.jsonl`. For gemma-4 a copy with `<bos>` prepended to every prompt
(`rag-12000-500-seed0-prefix-bos.jsonl`, same rows and order). Mean ISL (server tokens) is 820.0 for the 500 prompts and 832.1 for the first 100 (C=1) on the Qwen
models, 817.7 / 829.0 on gemma-4 (with `<bos>`), and 796.8 / 810.7 on V4.1. The command (`aiperf_cmd.txt` of each run):

```bash
aiperf profile --model <served> --tokenizer <model path> --url http://127.0.0.1:<port> --concurrency <C> \
  --ui simple --random-seed 0 --artifact-dir <run>/artifacts \
  --endpoint-type completions --streaming --use-server-token-count \
  --input-file data/rag-12000-500-seed0.jsonl --custom-dataset-type single_turn \
  --dataset-sampling-strategy sequential --request-count <500, or 100 at C=1> \
  --osl 256 --output-tokens-stddev 0 \
  --extra-inputs max_tokens:256 --extra-inputs ignore_eos:true --extra-inputs temperature:0
```

C = 1, 16, 32, 64, 128, 256, 512. Each point sends 500 requests (100 at C=1) with exactly 256 output tokens
(`ignore_eos`). The run ends when all requests return.

agentX. AIPerf scenario `inferencex-agentx-mvp` in its official full mode. The flags are the same as `MODE=full`
of the reference driver `agentx_run.sh`, with its seed 1110:

```bash
AIPERF_DATASET_CONFIGURATION_TIMEOUT=3600 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=3600 \
aiperf profile --model <served> --tokenizer <model path> --url http://127.0.0.1:<port> --concurrency <C> \
  --ui simple --random-seed 1110 --artifact-dir <run>/artifacts \
  --scenario inferencex-agentx-mvp --endpoint-type chat --use-server-token-count \
  --public-dataset semianalysis_cc_traces_weka_062126_256k --benchmark-duration 900
```

C = 1, 16, 64. There is no `--num-dataset-entries` (full corpus), no `--unsafe-override` and no
`--max-context-length`. The scenario applies its own locks: duration >= 900 s, streaming, `ignore_eos`, no input
truncation, cache bust on the first-turn prefix, trajectory start ratios 0..1 and a 10 s idle-gap cap. Phases:
cache warmup, the 900 s profiling window, then a drain of in-flight requests. "ok" counts the requests in the
window. AIPerf writes `submission_valid` and `submission_invalid_reasons`. Every completed full-mode run has
`submission_valid=true` with no reasons. Mean ISL is 82K-149K tokens. Client kill timeout: duration + 5 h.

### 2.6 Run order

Two workers (one per GPU, `bench_queue.py --gpus 6` / `--gpus 7`, each in tmux) took sessions from `queue.txt`,
where the before and after session of a model sit next to each other. A TP1 session runs on whichever worker
is free. The TP2 sessions run only on the GPU 6 worker, after every TP1 session is done, with GPU 7 idle for
2 min. Within a session, one server serves all runs in this order: RAG C = 1, 16, 32, 64, 128, 256, 512, then
agentX C = 1, 16, 64. The server is removed after the session.

Server launches (2026-11-12, local time of aws-raichu), from `gpu_at_launch.txt` and the runs' `DONE` / `FAILED`
markers (`finished` time):

| # | server launch | session (`<model>/<image>`) | GPUs | startup s | runs served, in order | last run ends |
|---|---|---|---|---|---|---|
| 1 | 06:49 | `qwen38-27b-fp8/e27cee86` | 7 | 417 (progress.log) | rag C=1,16,32,64,128,256,512; agentx_smoke C=1,16,32 FAILED | 08:05 |
| 2 | 07:04 | `qwen38-27b-fp8/ef057889` | 6 | 336 (progress.log) | rag C=1,16,32,64,128,256,512; agentx_smoke C=1,16,32 FAILED | 08:12 |
| 3 | 09:51 | `qwen38-27b-fp8/e27cee86` | 6 | 432 | agentx C=1,16,64 | 11:45 |
| 4 | 09:51 | `qwen38-27b-fp8/ef057889` | 7 | 361 | agentx C=1,16,64 | 11:43 |
| 5 | 11:58 | `qwen38-27b-nvfp4/e27cee86` | 7 | - | none: the server exits during startup (section 4.1) | - |
| 6 | 11:58 | `qwen38-27b-nvfp4/ef057889` | 6 | 376 | rag C=1,16,32,64,128,256,512; agentx C=1,16,64 | 14:07 |
| 7 | 12:02 | `qwen36-35b-a3b/e27cee86` | 7 | 492 | rag C=1,16,32,64,128,256,512; agentx C=1,16,64 | 13:37 |
| 8 | 13:39 | `qwen36-35b-a3b/ef057889` | 7 | 477 | rag C=1,16,32,64,128,256,512; agentx C=1,16,64 | 15:13 |
| 9 | 14:51 | `gemma4-31b-it/e27cee86` | 6 | 201 | rag C=1,16,32,64,128,256,512; agentx C=1,16 FAILED,64 FAILED | 16:37 |
| 10 | 15:15 | `gemma4-31b-it/ef057889` | 7 | 196 | rag C=1,16,32,64,128,256,512; agentx C=1,16 FAILED,64 FAILED | 16:59 |
| 11 | 17:01 | `qwen35-122b-a10b-nvfp4/e27cee86` | 6,7 | - | none: the server exits during startup (section 4.1) | - |
| 12 | 17:09 | `qwen35-122b-a10b-nvfp4/ef057889` | 6,7 | 552 | rag C=1,16,32,64,128,256,512; agentx C=1,16,64 | 18:51 |
| 13 | 18:54 | `dsv41-flash-1017/e27cee86` | 6,7 | - | none: the server exits during startup (section 4.1) | - |
| 14 | 18:56 | `dsv41-flash-1017/ef057889` | 6,7 | 537 | rag C=1,16,32,64,128,256,512; agentx C=1,16,64 | 20:25 |
| 15 | 20:28 | `dsv41-flash-1017/ef057889-mega_moe` | 6,7 | 542 | rag C=1,16,32,64,128,256,512; agentx C=1,16,64 | 21:54 |

Notes:
- Qwen3.8-27B-FP8 has two servers per image. The first served RAG and the agentX smoke runs (section 6). The
  second, three hours later, served agentX full mode. The pair swapped GPUs between the two: RAG before on GPU 7
  and after on GPU 6, agentX before on GPU 6 and after on GPU 7. `session.json` holds only the second launch
  (the harness rewrites it per launch), so the first two startup times come from `progress.log`.
- Concurrent sessions: the FP8 pair ran side by side (launches 1-4), then the NVFP4 after session on GPU 6
  beside the Qwen3.6 before session on GPU 7, the Qwen3.6 after session beside NVFP4 agentX C=64, and the gemma
  pair side by side. Before and after of one model usually ran under similar host load, not identical load.
- The before-image Qwen3.8-27B-NVFP4 session started 1 s before the after one and exited after 67 s.

## 3. Detailed per-run results

Units: req/s; tok/s; TTFT, TPOT and ITL in ms; `wall s` is the client wall time of the run (agentX includes
warmup and drain); `eff. C avg` is AIPerf's time-averaged number of requests in flight; `ok` / `failed` are
AIPerf's success and error counts; `ISL avg` is server-counted input tokens per request. A row with only a
status has no metrics (`FAILED` = the run's marker; section 4).


### 3.1 Qwen3.8-27B-FP8 (TP1)

- The gain is decode: TPOT p50 31.6 -> 20.6 ms at C=1, TTFT unchanged (prefill is at kernel parity). It shrinks
  as the batch grows (TPOT 1.31x at C=16, 1.13x at 64, 1.04-1.06x from 128), because attention, the GatedDeltaNet
  layers and the scheduler take a larger share of the step.
- Throughput stops growing at C=256 (about 1275 / 1324 tok/s). The server log shows the KV pool full (567 of
  569 pages active, 49 requests queued, at most 207 running), and the decode batch is above the 160 graph limit.
- agentX: effective concurrency stays at 6-8 for C=16 and 64 (requests wait for long prefills). At C=64 TTFT
  p50 is 399 s and only 14-15 requests finish in the window. The C=16 TTFT p50 pair (1051 / 2519 ms, "0.42x") is
  one run each of a heavy-tailed distribution (TTFT p99 214 / 209 s, mean 31.1 / 30.9 s); I don't read it as a
  regression.

**before `e27cee86`, RAG** (`qwen38-27b-fp8/e27cee86/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.12 | 101.8 | 31.3 | 107 | 103 | 167 | 31.6 | 31.6 | 32.9 | 31.6 | 94.9 | 100 | 0 | 1.0 | 832 | 822 |
| 16 | 1.57 | 1290.4 | 402.9 | 1039 | 970 | 2083 | 34.9 | 34.8 | 38.8 | 33.4 | 100.9 | 500 | 0 | 15.7 | 820 | 322 |
| 32 | 2.77 | 2272.0 | 709.3 | 1757 | 1796 | 2831 | 37.4 | 37.4 | 44.2 | 34.0 | 103.2 | 500 | 0 | 31.3 | 820 | 184 |
| 64 | 3.76 | 3082.2 | 962.2 | 3000 | 2822 | 5522 | 53.5 | 53.2 | 65.9 | 45.1 | 180.0 | 500 | 0 | 62.6 | 820 | 137 |
| 128 | 4.76 | 3906.0 | 1219.4 | 5601 | 5798 | 10831 | 80.8 | 80.4 | 102.0 | 63.0 | 315.4 | 500 | 0 | 125.0 | 820 | 109 |
| 256 | 4.98 | 4085.2 | 1275.4 | 15084 | 11421 | 43413 | 119.0 | 127.8 | 153.8 | 89.0 | 1447.3 | 500 | 0 | 227.4 | 820 | 104 |
| 512 | 4.98 | 4086.7 | 1275.8 | 38227 | 43074 | 86664 | 120.6 | 129.8 | 153.9 | 89.3 | 2060.2 | 500 | 0 | 344.9 | 820 | 104 |

**before `e27cee86`, agentX** (`qwen38-27b-fp8/e27cee86/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.015 | 2206.9 | 21.2 | 5981 | 3265 | 28154 | 39.0 | 39.2 | 40.0 | 39.3 | 78.6 | 14 | 0 | 0.9 | 144981 | 955 | 31 | yes |
| 16 | 0.056 | 4931.1 | 29.0 | 31142 | 1051 | 214469 | 194.2 | 94.1 | 1424.5 | 72.2 | 509.1 | 53 | 0 | 6.3 | 87458 | 1564 | 619 | yes |
| 64 | 0.015 | 1643.7 | 10.2 | 362898 | 398631 | 641302 | 1643.9 | 243.9 | 16735.2 | 72.5 | 219.2 | 14 | 0 | 8.0 | 110366 | 3878 | 2933 | yes |

**after `ef057889`, RAG** (`qwen38-27b-fp8/ef057889/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.19 | 154.9 | 47.7 | 106 | 103 | 172 | 20.6 | 20.6 | 21.2 | 20.6 | 62.1 | 100 | 0 | 1.0 | 832 | 541 |
| 16 | 2.01 | 1650.7 | 515.3 | 1027 | 979 | 1811 | 26.5 | 26.6 | 30.3 | 25.1 | 75.8 | 500 | 0 | 15.7 | 820 | 252 |
| 32 | 3.11 | 2546.8 | 795.1 | 1760 | 1803 | 2854 | 32.6 | 32.6 | 39.4 | 29.2 | 93.1 | 500 | 0 | 31.4 | 820 | 165 |
| 64 | 4.16 | 3415.2 | 1066.2 | 2996 | 2804 | 5469 | 47.1 | 47.1 | 58.9 | 38.8 | 143.0 | 500 | 0 | 62.6 | 820 | 124 |
| 128 | 5.01 | 4109.1 | 1282.8 | 5653 | 5915 | 10758 | 75.5 | 75.5 | 97.0 | 57.5 | 413.0 | 500 | 0 | 125.0 | 820 | 104 |
| 256 | 5.17 | 4243.3 | 1324.7 | 14778 | 11276 | 42493 | 112.9 | 120.1 | 148.2 | 84.0 | 1296.2 | 500 | 0 | 226.5 | 820 | 101 |
| 512 | 5.17 | 4240.8 | 1323.9 | 37630 | 42539 | 84441 | 116.0 | 125.1 | 147.0 | 84.0 | 1954.9 | 500 | 0 | 348.7 | 820 | 101 |

**after `ef057889`, agentX** (`qwen38-27b-fp8/ef057889/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.018 | 2725.6 | 24.2 | 5279 | 2703 | 27062 | 27.5 | 27.8 | 28.6 | 27.7 | 56.1 | 17 | 0 | 0.8 | 149107 | 1033 | 31 | yes |
| 16 | 0.057 | 5034.7 | 31.8 | 30913 | 2519 | 208692 | 173.9 | 83.0 | 1370.5 | 58.3 | 238.9 | 54 | 0 | 6.2 | 87641 | 1555 | 611 | yes |
| 64 | 0.016 | 1779.6 | 10.3 | 383289 | 398626 | 801530 | 1497.6 | 225.8 | 16220.1 | 59.5 | 71.5 | 15 | 0 | 8.5 | 111522 | 3813 | 2868 | yes |


### 3.2 Qwen3.6-35B-A3B BF16 (TP1, control)

- The control. It uses no FP8/FP4 GEMM and no DeepGEMM path. RAG output tok/s 1.00-1.01x and TPOT 1.00-1.01x at
  every C. The TTFT p50 ratio of 0.98-1.11x is noise on 0.3-4 s values. Plateau from C=256: the KV pool is full
  (403 of 405 pages) with about 158 running requests (server log).

**before `e27cee86`, RAG** (`qwen36-35b-a3b/e27cee86/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.66 | 546.6 | 168.2 | 65 | 64 | 86 | 5.7 | 5.7 | 5.9 | 5.7 | 11.7 | 100 | 0 | 1.0 | 832 | 156 |
| 16 | 2.78 | 2282.4 | 712.5 | 333 | 296 | 798 | 21.0 | 21.0 | 22.2 | 21.1 | 43.4 | 500 | 0 | 15.8 | 820 | 184 |
| 32 | 3.75 | 3075.8 | 960.2 | 509 | 517 | 825 | 30.8 | 31.0 | 32.8 | 30.6 | 63.2 | 500 | 0 | 31.4 | 820 | 138 |
| 64 | 5.30 | 4350.1 | 1358.1 | 849 | 862 | 1502 | 42.9 | 43.0 | 46.0 | 41.6 | 85.1 | 500 | 0 | 62.6 | 820 | 98 |
| 128 | 7.57 | 6204.1 | 1936.8 | 1585 | 1667 | 2925 | 58.5 | 58.4 | 64.4 | 54.3 | 324.3 | 500 | 0 | 125.0 | 820 | 70 |
| 256 | 7.73 | 6342.0 | 1979.9 | 10241 | 3838 | 20848 | 65.5 | 67.2 | 72.9 | 59.5 | 450.0 | 500 | 0 | 208.6 | 820 | 69 |
| 512 | 7.69 | 6306.7 | 1968.9 | 22845 | 20951 | 57259 | 65.8 | 67.5 | 73.2 | 59.5 | 417.9 | 500 | 0 | 305.0 | 820 | 69 |

**before `e27cee86`, agentX** (`qwen36-35b-a3b/e27cee86/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.100 | 9373.3 | 71.7 | 1443 | 709 | 12638 | 13.7 | 12.4 | 35.8 | 10.7 | 21.6 | 93 | 0 | 1.0 | 93733 | 1011 | 11 | yes |
| 16 | 0.120 | 10820.6 | 108.1 | 6971 | 848 | 59702 | 43.9 | 22.7 | 277.0 | 14.8 | 45.4 | 112 | 0 | 3.9 | 89850 | 1218 | 284 | yes |
| 64 | 0.053 | 5651.0 | 43.7 | 299612 | 293753 | 549077 | 182.7 | 57.0 | 2809.3 | 24.6 | 52.6 | 50 | 0 | 19.1 | 106238 | 2217 | 1272 | yes |

**after `ef057889`, RAG** (`qwen36-35b-a3b/ef057889/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.66 | 552.9 | 170.1 | 64 | 64 | 82 | 5.6 | 5.6 | 5.9 | 5.6 | 11.5 | 100 | 0 | 1.0 | 832 | 154 |
| 16 | 2.82 | 2309.5 | 721.0 | 325 | 292 | 788 | 20.7 | 20.8 | 22.1 | 20.8 | 42.8 | 500 | 0 | 15.8 | 820 | 182 |
| 32 | 3.76 | 3081.8 | 962.1 | 506 | 515 | 825 | 30.7 | 30.8 | 32.8 | 30.5 | 63.1 | 500 | 0 | 31.4 | 820 | 137 |
| 64 | 5.31 | 4357.0 | 1360.2 | 858 | 824 | 1506 | 42.8 | 42.9 | 46.2 | 41.5 | 85.0 | 500 | 0 | 62.6 | 820 | 98 |
| 128 | 7.57 | 6209.4 | 1938.5 | 1559 | 1506 | 2902 | 58.6 | 58.4 | 64.7 | 54.3 | 361.8 | 500 | 0 | 125.1 | 820 | 70 |
| 256 | 7.73 | 6334.8 | 1977.7 | 10262 | 3900 | 20939 | 65.4 | 67.1 | 72.7 | 59.4 | 436.2 | 500 | 0 | 208.3 | 820 | 69 |
| 512 | 7.71 | 6319.0 | 1972.7 | 22827 | 20823 | 57314 | 65.3 | 66.9 | 73.5 | 59.4 | 418.6 | 500 | 0 | 304.3 | 820 | 69 |

**after `ef057889`, agentX** (`qwen36-35b-a3b/ef057889/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.102 | 9723.8 | 73.0 | 1787 | 733 | 33205 | 13.1 | 11.9 | 28.4 | 10.4 | 24.7 | 94 | 0 | 1.0 | 95071 | 934 | 11 | yes |
| 16 | 0.120 | 10802.6 | 101.1 | 7764 | 871 | 59098 | 184.5 | 21.6 | 735.4 | 15.1 | 37.0 | 113 | 0 | 3.8 | 89863 | 1227 | 282 | yes |
| 64 | 0.054 | 5792.9 | 44.9 | 297606 | 291473 | 538605 | 175.6 | 53.9 | 2700.1 | 23.6 | 50.3 | 51 | 0 | 19.3 | 106772 | 2210 | 1265 | yes |


### 3.3 gemma-4-31B-it BF16 (TP1)

- RAG prompts start with `<bos>` (section 4.4), so ISL is one token more than the same text without it.
- From C=32 the server runs only about 27 requests at a time (output tok/s x TPOT p50; the log shows at most
  34 running and 451 queued with 2291 of 2295 KV pages active). Throughput stays at 362-372 tok/s and TPOT at
  73-74 ms while TTFT p50 grows with the queue. The cap is the KV pool (293,760 tokens), the same in both images.
  Why a ~1.1K-token request holds that many pages (task 52, arithmetic from the capacity model, not a trace):
  each of the five sliding groups owns its parents, and a request keeps its 1024-token window there (17 blocks of
  64 tokens, one parent per block) in every group, about 85 parents with the full-attention share, and
  2295 / 85 = 27 requests.
- agentX C=16 and 64 fail in both images with the default args (section 4.2). C=1 completes (11 requests each, valid). The rerun with args that lift the limit follows the agentX table below.

**before `e27cee86`, RAG** (`gemma4-31b-it/e27cee86/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.09 | 72.0 | 22.2 | 177 | 175 | 296 | 44.4 | 44.4 | 44.5 | 44.4 | 89.2 | 100 | 0 | 1.0 | 829 | 1156 |
| 16 | 1.00 | 819.8 | 256.6 | 712 | 633 | 3019 | 58.7 | 58.8 | 61.4 | 53.0 | 256.0 | 500 | 0 | 15.7 | 818 | 504 |
| 32 | 1.43 | 1168.3 | 365.8 | 2045 | 1428 | 9625 | 78.5 | 73.6 | 142.3 | 61.4 | 239.4 | 500 | 0 | 31.5 | 818 | 355 |
| 64 | 1.44 | 1180.2 | 369.5 | 22582 | 22836 | 37947 | 77.4 | 73.8 | 131.6 | 61.4 | 236.0 | 500 | 0 | 61.1 | 818 | 352 |
| 128 | 1.43 | 1165.3 | 364.8 | 59939 | 64409 | 82344 | 78.1 | 73.7 | 150.5 | 61.4 | 245.0 | 500 | 0 | 113.8 | 818 | 356 |
| 256 | 1.43 | 1168.4 | 365.8 | 117886 | 148361 | 164578 | 78.3 | 73.6 | 144.3 | 61.4 | 240.2 | 500 | 0 | 197.0 | 818 | 355 |
| 512 | 1.41 | 1155.7 | 361.8 | 169674 | 171128 | 337140 | 79.4 | 74.2 | 145.1 | 61.4 | 245.4 | 500 | 0 | 268.4 | 818 | 359 |

**before `e27cee86`, agentX** (`gemma4-31b-it/e27cee86/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.012 | 1654.4 | 16.0 | 4351 | 2254 | 12284 | 52.1 | 52.0 | 53.5 | 52.5 | 103.3 | 11 | 0 | 0.9 | 139875 | 1054 | 51 | yes |
| 16 | FAILED | | | | | | | | | | | | | | | | | |
| 64 | FAILED | | | | | | | | | | | | | | | | | |

**after `ef057889`, RAG** (`gemma4-31b-it/ef057889/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.09 | 74.4 | 23.0 | 175 | 173 | 291 | 43.0 | 43.0 | 43.2 | 43.0 | 86.4 | 100 | 0 | 1.0 | 829 | 1120 |
| 16 | 1.03 | 840.9 | 263.3 | 740 | 648 | 2712 | 57.0 | 57.2 | 60.0 | 51.6 | 253.1 | 500 | 0 | 15.7 | 818 | 491 |
| 32 | 1.45 | 1186.4 | 371.4 | 2031 | 1514 | 7776 | 77.1 | 73.0 | 129.0 | 61.3 | 234.4 | 500 | 0 | 31.5 | 818 | 350 |
| 64 | 1.45 | 1184.3 | 370.8 | 22537 | 22735 | 38750 | 77.0 | 73.3 | 140.1 | 61.3 | 233.5 | 500 | 0 | 61.1 | 818 | 350 |
| 128 | 1.44 | 1179.5 | 369.3 | 59386 | 64567 | 81279 | 77.2 | 73.6 | 145.8 | 61.3 | 237.6 | 500 | 0 | 114.0 | 818 | 352 |
| 256 | 1.45 | 1187.0 | 371.6 | 115888 | 146598 | 164820 | 76.6 | 73.3 | 125.9 | 61.3 | 236.8 | 500 | 0 | 196.6 | 818 | 349 |
| 512 | 1.45 | 1184.6 | 370.9 | 166337 | 166253 | 329356 | 76.9 | 73.0 | 138.3 | 61.3 | 237.6 | 500 | 0 | 269.4 | 818 | 350 |

**after `ef057889`, agentX** (`gemma4-31b-it/ef057889/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.012 | 1654.4 | 16.0 | 4330 | 2236 | 12226 | 50.7 | 50.5 | 52.2 | 51.1 | 98.4 | 11 | 0 | 0.9 | 139875 | 986 | 51 | yes |
| 16 | FAILED | | | | | | | | | | | | | | | | | |
| 64 | FAILED | | | | | | | | | | | | | | | | | |

**after `ef057889`, agentX rerun with a 262,143-token limit (task 52, `results/fix52/gemma4-31b-it/`)**: same image
(`nctu6/tokenspeed:ef057889-20261112`), same harness and AIPerf command as above (full mode, 900 s, seed 1110,
`--tool-call-parser gemma4`, `--max-num-seqs 512`, `--gpu-memory-utilization 0.9`, `--max-model-len 262144`, GPU 6
or 7, KV events off); only the engine args differ. The image's engine code is unchanged (`ef057889f`); the TokenSpeed
commits `1950e48eb` (test) and `eece48b98` (docs) change no runtime code. All six runs `submission_valid=true`,
0 errors, no restart.

| args | C | req/s | in tok/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--kv-cache-dtype fp8` | 1 | 0.012 | 1654.4 | 16.0 | 2261 | 12599 | 48.4 | 49.6 | 48.8 | 50.3 | 11 | 0 | 0.8 | 139872 | 988 | 53 | yes |
| `--kv-cache-dtype fp8` | 16 | 0.017 | 1582.5 | 13.4 | 302891 | 422396 | 89.1 | 1115.9 | 59.5 | 1464.8 | 16 | 0 | 6.4 | 92971 | 1925 | 980 | yes |
| `--kv-cache-dtype fp8` | 64 | 0.009 | 928.8 | 5.0 | 453702 | 873794 | 149.0 | 1673.9 | 55.4 | 2031.7 | 8 | 0 | 4.9 | 109138 | 5923 | 4978 | yes |
| `--chunked-prefill-size 1024` | 1 | 0.012 | 1654.4 | 16.0 | 2351 | 13302 | 50.5 | 52.1 | 51.1 | 98.6 | 11 | 0 | 0.9 | 139875 | 992 | 56 | yes |
| `--chunked-prefill-size 1024` | 16 | 0.007 | 969.6 | 9.2 | 554379 | 699321 | 54.5 | 208.8 | 55.8 | 362.7 | 7 | 0 | 4.0 | 130199 | 1977 | 1032 | yes |
| `--chunked-prefill-size 1024` | 64 | 0.004 | 547.0 | 3.8 | 267216 | 535074 | 54.4 | 398.3 | 52.6 | 56.2 | 4 | 0 | 1.4 | 128535 | 6249 | 5304 | yes |

Startup lines: fp8 `Cache profile: parent_bytes=5242880 parents=4617`, `Single-request token limit: cache=496831 ...
max_input=262143`; chunk 1024 `parents=2339`, `cache=276927 ... max_input=262143` (default: 2295 parents,
`cache=199615 ... max_input=199614`). Both servers answer "Paris" and "2, 3, 5, 7, 11" in a chat probe.
Read these with care. Only 4 to 16 requests complete in the 900 s window (a 256K prefill takes minutes), the
warmup is 16 to 88 minutes at C=16/64, the traces in the window differ between runs (ISL avg 93K to 140K), there
is one run per point, and two runs shared the host (GPU 6 and 7, CPU, disk). The two args are not a comparison of
each other: `fp8` also changes the KV numerics, chunk 1024 keeps BF16 KV and issues prefill in 1024-token steps.
TPOT p50 at C=1 is 48.4 ms (fp8) and 50.5 ms (chunk 1024), against 50.5 ms in the default-args C=1 run above.
C=16/64 throughput is low (13 and 5 output tok/s at best) because these are 100K-256K-context agent traces that
a 22 GiB pool holds only a few of. This is a capacity result (the runs complete), not a throughput comparison
with the Qwen rows.


### 3.4 Qwen3.8-27B-NVFP4 (TP1)

- For orientation only (different checkpoint): on the after image NVFP4 gives 1.16x the FP8 checkpoint's
  output tok/s at C=1 (55.3 vs 47.7) and 1.22x at C=512 (1611.6 vs 1323.9).

**before `e27cee86`**: no runs, the server does not start (section 4.1).

**after `ef057889`, RAG** (`qwen38-27b-nvfp4/ef057889/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.22 | 179.7 | 55.3 | 81 | 74 | 119 | 17.8 | 17.7 | 18.9 | 17.8 | 70.8 | 100 | 0 | 1.0 | 832 | 467 |
| 16 | 2.41 | 1980.3 | 618.2 | 769 | 729 | 1246 | 22.4 | 22.4 | 25.3 | 21.4 | 85.5 | 500 | 0 | 15.7 | 820 | 211 |
| 32 | 3.77 | 3094.7 | 966.1 | 1282 | 1346 | 2099 | 27.5 | 27.5 | 32.3 | 25.0 | 99.9 | 500 | 0 | 31.4 | 820 | 137 |
| 64 | 5.16 | 4230.0 | 1320.6 | 2175 | 2217 | 3898 | 39.0 | 38.8 | 47.5 | 33.0 | 131.7 | 500 | 0 | 62.6 | 820 | 101 |
| 128 | 6.07 | 4979.3 | 1554.5 | 3993 | 3873 | 7890 | 64.7 | 64.6 | 81.5 | 51.6 | 207.5 | 500 | 0 | 125.1 | 820 | 87 |
| 256 | 6.29 | 5155.7 | 1609.5 | 8927 | 7733 | 36595 | 111.3 | 114.6 | 138.6 | 84.9 | 1110.2 | 500 | 0 | 235.4 | 820 | 84 |
| 512 | 6.30 | 5162.3 | 1611.6 | 27391 | 36700 | 73077 | 113.1 | 117.2 | 138.8 | 84.8 | 1480.9 | 500 | 0 | 355.7 | 820 | 84 |

**after `ef057889`, agentX** (`qwen38-27b-nvfp4/ef057889/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.018 | 2725.6 | 24.2 | 4929 | 2581 | 24432 | 24.9 | 25.0 | 26.6 | 24.9 | 46.6 | 17 | 0 | 0.7 | 149107 | 1029 | 28 | yes |
| 16 | 0.073 | 6560.6 | 40.2 | 20488 | 1009 | 183150 | 140.8 | 67.8 | 1140.9 | 54.4 | 170.9 | 69 | 0 | 6.2 | 89376 | 1510 | 565 | yes |
| 64 | 0.020 | 2083.6 | 8.6 | 463104 | 404731 | 827134 | 1350.5 | 147.1 | 16972.9 | 62.1 | 190.3 | 19 | 0 | 11.6 | 103085 | 3611 | 2666 | yes |


### 3.5 Qwen3.5-122B-A10B-NVFP4 (TP2)

- The 122B MoE keeps scaling to C=256 (2281.5 tok/s). agentX C=64 is saturated: TTFT p50 123 s, TPOT p50
  584 ms, effective concurrency 22.4 of 64, 54 requests in the window.

**before `e27cee86`**: no runs, the server does not start (section 4.1).

**after `ef057889`, RAG** (`qwen35-122b-a10b-nvfp4/ef057889/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.52 | 433.5 | 133.4 | 83 | 83 | 114 | 7.1 | 7.2 | 7.5 | 7.2 | 7.7 | 100 | 0 | 1.0 | 832 | 196 |
| 16 | 3.14 | 2570.8 | 802.6 | 599 | 648 | 1243 | 17.2 | 17.0 | 20.0 | 16.7 | 21.3 | 500 | 0 | 15.8 | 820 | 164 |
| 32 | 4.29 | 3517.3 | 1098.1 | 958 | 984 | 1573 | 24.7 | 24.8 | 28.0 | 23.6 | 32.8 | 500 | 0 | 31.4 | 820 | 121 |
| 64 | 5.78 | 4741.2 | 1480.1 | 1539 | 1527 | 2706 | 36.1 | 35.9 | 41.8 | 32.7 | 64.6 | 500 | 0 | 62.6 | 820 | 90 |
| 128 | 7.47 | 6125.1 | 1912.2 | 2883 | 2947 | 5361 | 53.6 | 53.6 | 64.6 | 45.7 | 92.1 | 500 | 0 | 125.0 | 820 | 71 |
| 256 | 8.91 | 7308.1 | 2281.5 | 5492 | 5606 | 10427 | 87.2 | 86.7 | 107.7 | 69.2 | 383.0 | 500 | 0 | 249.6 | 820 | 60 |
| 512 | 8.81 | 7225.4 | 2255.7 | 16262 | 10803 | 45482 | 108.4 | 115.1 | 155.2 | 93.3 | 955.0 | 500 | 0 | 390.5 | 820 | 61 |

**after `ef057889`, agentX** (`qwen35-122b-a10b-nvfp4/ef057889/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.098 | 9017.0 | 63.7 | 1417 | 919 | 5272 | 15.5 | 13.3 | 43.4 | 12.7 | 16.4 | 91 | 0 | 1.0 | 92152 | 1012 | 15 | yes |
| 16 | 0.137 | 11841.0 | 118.6 | 5245 | 720 | 78942 | 34.2 | 18.8 | 225.2 | 13.9 | 21.2 | 127 | 0 | 3.3 | 86710 | 1278 | 344 | yes |
| 64 | 0.057 | 4710.7 | 14.9 | 161115 | 123098 | 431467 | 1753.7 | 583.7 | 19664.8 | 52.3 | 21879.0 | 54 | 0 | 22.4 | 82002 | 2499 | 1555 | yes |


### 3.6 DeepSeek-V4.1-Flash 128-of-384 experts, Marlin `auto` (TP2)

- Not a real model output: 128 of 384 routed experts per layer, so the probe is recorded and not gated. The
  speed is that of a 128-expert V4.1 topology with real weights for everything else. The MoE cost is about a
  third of the full model's, so don't compare it with a full V4.1 deployment.
- RAG C=256 dips below C=128 (6.44 vs 7.43 req/s, TPOT p50 139.8 vs 57.9 ms), and C=512 is 1.6x C=256. It is not
  noise: mega_moe has the same shape (8.71, 6.71, 10.89 req/s), and effective concurrency is 250 and 500 as set.
  The server log shows decode batches of 116-128 at C=128 (about 2390 generated tok/s, ~49 ms a step), 244-256
  at C=256 (~1900 tok/s, ~130 ms a step) and 500 at C=512 (~3550 tok/s, ~140 ms a step). The step time jumps
  2.7x once the batch passes 160, the largest captured graph, then barely grows with the batch. Task 53 confirmed
  the cause: the default `--max-cudagraph-capture-size` was 160 whatever `--max-num-seqs` said (512 here), and a
  decode batch above the largest graph runs the same refresh path without a graph (section 3.8).
- RAG C=1 TTFT p99 is 1277 ms against p50 98 ms: the first request of the run pays one-time work (C=1 only).
- agentX: 654 requests finish at C=64 against 14-54 for the other models. Its KV pool holds 24.5M tokens against
  0.3-3.6M for the others.

**before `e27cee86`**: no runs, the server does not start (section 4.1).

**after `ef057889`, RAG** (`dsv41-flash-1017/ef057889/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.41 | 331.0 | 104.5 | 149 | 98 | 1277 | 9.0 | 9.0 | 9.1 | 9.0 | 9.5 | 100 | 0 | 1.0 | 811 | 249 |
| 16 | 2.39 | 1907.2 | 612.8 | 731 | 658 | 2236 | 23.0 | 22.8 | 28.9 | 21.6 | 23.4 | 500 | 0 | 15.8 | 797 | 213 |
| 32 | 3.81 | 3034.1 | 974.9 | 624 | 815 | 891 | 29.8 | 29.4 | 32.3 | 29.0 | 31.1 | 500 | 0 | 31.3 | 797 | 135 |
| 64 | 5.58 | 4447.4 | 1429.0 | 1080 | 1199 | 1622 | 39.7 | 39.5 | 44.0 | 38.0 | 40.3 | 500 | 0 | 62.6 | 797 | 94 |
| 128 | 7.43 | 5916.7 | 1901.0 | 2117 | 2045 | 3817 | 57.6 | 57.9 | 65.6 | 53.7 | 57.8 | 500 | 0 | 125.0 | 797 | 71 |
| 256 | 6.44 | 5134.7 | 1649.8 | 3262 | 3185 | 5804 | 139.1 | 139.8 | 150.9 | 128.7 | 162.7 | 500 | 0 | 249.9 | 797 | 82 |
| 512 | 10.52 | 8380.0 | 2692.5 | 6135 | 6547 | 11420 | 161.9 | 160.2 | 184.5 | 140.3 | 339.7 | 500 | 0 | 499.6 | 797 | 52 |

**after `ef057889`, agentX** (`dsv41-flash-1017/ef057889/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.111 | 11458.2 | 89.6 | 314 | 187 | 1483 | 11.8 | 11.3 | 19.8 | 9.2 | 13.3 | 100 | 0 | 0.9 | 103124 | 977 | 7 | yes |
| 16 | 0.188 | 16222.1 | 150.8 | 459 | 190 | 3992 | 12.2 | 11.5 | 26.9 | 10.6 | 18.1 | 175 | 0 | 1.8 | 86209 | 1147 | 213 | yes |
| 64 | 0.696 | 62575.2 | 524.2 | 756 | 299 | 6042 | 38.8 | 34.3 | 110.2 | 24.0 | 228.2 | 654 | 0 | 19.4 | 89940 | 1752 | 806 | yes |


### 3.7 DeepSeek-V4.1-Flash 128-of-384 experts, `mega_moe` (TP2)

- `--moe-backend mega_moe` (`deep_gemm_sm120_mxfp4_mega_moe_apply`, attention TP2 + MoE TP2) against Marlin on
  the same image: RAG 1.01-1.07x up to C=64, 1.17x at C=128 (TPOT 57.9 -> 49.7 ms), 1.04x at C=256 and 512. agentX
  0.97x / 1.00x / 1.03x. This agrees with the in-process result of analysis 9.38 (prefill 1.13x, TPOT within 1%),
  and `auto` stays on Marlin.

**after `ef057889`, RAG** (`dsv41-flash-1017/ef057889-mega_moe/rag/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.41 | 335.9 | 106.1 | 145 | 93 | 1281 | 8.9 | 8.9 | 9.0 | 8.9 | 9.4 | 100 | 0 | 1.0 | 811 | 246 |
| 16 | 2.45 | 1949.4 | 626.3 | 697 | 634 | 2971 | 22.6 | 22.2 | 28.0 | 21.3 | 23.1 | 500 | 0 | 15.8 | 797 | 208 |
| 32 | 3.92 | 3123.9 | 1003.7 | 594 | 770 | 851 | 29.0 | 28.6 | 31.3 | 28.2 | 30.4 | 500 | 0 | 31.3 | 797 | 131 |
| 64 | 5.95 | 4737.9 | 1522.3 | 997 | 1087 | 1508 | 37.3 | 37.0 | 41.1 | 35.7 | 38.2 | 500 | 0 | 62.6 | 797 | 88 |
| 128 | 8.71 | 6937.0 | 2228.9 | 1664 | 1756 | 2786 | 49.7 | 49.7 | 56.5 | 45.7 | 49.8 | 500 | 0 | 125.0 | 797 | 62 |
| 256 | 6.71 | 5346.4 | 1717.8 | 2966 | 3166 | 5230 | 134.2 | 134.0 | 144.4 | 125.2 | 160.7 | 500 | 0 | 249.9 | 797 | 78 |
| 512 | 10.89 | 8678.1 | 2788.3 | 5528 | 5738 | 10257 | 157.8 | 157.0 | 177.8 | 138.5 | 354.2 | 500 | 0 | 499.6 | 797 | 50 |

**after `ef057889`, agentX** (`dsv41-flash-1017/ef057889-mega_moe/agentx/`)

| C | req/s | in tok/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | ok | failed | eff. C avg | ISL avg | wall s | warmup s | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.108 | 11088.6 | 86.7 | 298 | 210 | 1321 | 11.6 | 10.9 | 17.6 | 9.1 | 13.3 | 100 | 0 | 0.9 | 103124 | 941 | 6 | yes |
| 16 | 0.190 | 16396.0 | 151.1 | 428 | 174 | 4010 | 12.1 | 11.2 | 27.3 | 10.5 | 18.1 | 177 | 0 | 1.8 | 86148 | 1135 | 200 | yes |
| 64 | 0.704 | 63451.4 | 541.6 | 750 | 299 | 6546 | 36.2 | 32.1 | 100.5 | 23.6 | 205.1 | 662 | 0 | 18.7 | 90097 | 1693 | 747 | yes |


### 3.8 Task 53 rerun: decode graphs up to `--max-num-seqs` (V4.1 subset, TP2)

The default `--max-cudagraph-capture-size` is now `--max-num-seqs` (TokenSpeed `a0a5484a1`). The rerun uses the
`ef057889-20261112` image with the two changed Python files (`forward_step.py`, `server_args.py`) mounted over the
installed ones, `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` like the original rows, the same server flags (only the
captured sizes change), the same RAG protocol and the full C = 1 .. 512 ramp on a fresh server per variant
(`results/fix53full/`; an earlier C = 128/256/512-only rerun in `results/fix53/` started cold and is not used for
C=128). One run per point, shared host.

- Capture list: `[512, 480, 448, 416, 384, 352, 320, 288, 256, 240, ..., 160, 152, ..., 8, 4, 2, 1]` (37 graphs)
  against `[160, 152, 144, 104, 40]` before. Graph
  memory reserve counted before the KV sizing: KV cache 24.92 -> 22.91 GiB per rank (token capacity 24.47M ->
  22.06M, -9.8%).

Output tok/s and TPOT p50 (ms), original row -> rerun (ratio of output tok/s):

| C | Marlin out tok/s | Marlin TPOT p50 | mega_moe out tok/s | mega_moe TPOT p50 |
|---|---|---|---|---|
| 1 | 104.5 -> 104.1 (1.00x) | 9.0 -> 9.0 | 106.1 -> 105.7 (1.00x) | 8.9 -> 8.9 |
| 16 | 612.8 -> 613.5 (1.00x) | 22.8 -> 22.4 | 626.3 -> 625.8 (1.00x) | 22.2 -> 22.2 |
| 32 | 974.9 -> 977.2 (1.00x) | 29.4 -> 29.7 | 1003.7 -> 1003.0 (1.00x) | 28.6 -> 28.6 |
| 64 | 1429.0 -> 1430.5 (1.00x) | 39.5 -> 39.4 | 1522.3 -> 1525.5 (1.00x) | 37.0 -> 37.3 |
| 128 | 1901.0 -> 1934.0 (1.02x) | 57.9 -> 57.6 | 2228.9 -> 2227.8 (1.00x) | 49.7 -> 48.7 |
| 256 | 1649.8 -> **2656.4 (1.61x)** | 139.8 -> **81.0** | 1717.8 -> **3029.2 (1.76x)** | 134.0 -> **70.6** |
| 512 | 2692.5 -> **3480.9 (1.29x)** | 160.2 -> **120.5** | 2788.3 -> **3827.9 (1.37x)** | 157.0 -> **109.8** |

All requests succeeded at every C (500; 100 at C=1), 0 errors. The curve is now
monotonic (Marlin 1934, 2656, 3481; mega_moe 2228, 3029, 3828) and C=1..64 are unchanged. TTFT p50 at C=256 is
3185 -> 3477 ms (Marlin) and 3166 -> 3096 ms (mega_moe), at C=512 6547 -> 5899 and 5738 -> 5305. mega_moe over Marlin
on the same image: 1.00-1.17x up to C=128, 1.14x at C=256 (was 1.04x), 1.10x at C=512 (was 1.04x). Marlin stays the
`auto` choice.


## 4. Failures

Raw logs are on aws-raichu under `/workspace/nctu6/infra/tokenspeed/tmp/bench_images/` (paths below are
relative to it). The local copy (`tokenspeed/tmp/bench_images_local/results/`) has the markers, `metrics.json`
and the `SESSION_FAILED` files with the log tail, but no server logs or AIPerf artifacts.

### 4.1 Before-image startup failures (3 sessions, 30 runs never started)

All three exit during startup (`ts subprocess exited with rc=-10`). The harness records `RuntimeError('server
container exited before becoming healthy')` and `SESSION_FAILED`, and does not retry.

| model | error line | where | session length | server log |
|---|---|---|---|---|
| Qwen3.8-27B-NVFP4, TP1, GPU 7 | `RuntimeError: Kernel implementation not found` | `tokenspeed/runtime/layers/dense/nvfp4.py:196` (`fp4_quantize(x, layer.input_scale_inv)`, reached from `qwen3_5.py` `forward`) -> `tokenspeed_kernel/registry.py:531` (`error_fn`): no `fp4_quantize` kernel is registered for SM120 in this image | 67 s | `qwen38-27b-nvfp4/e27cee86/server-20261112-115858.log` |
| Qwen3.5-122B-A10B-NVFP4, TP2 | `RuntimeError: Check failed: (fc1_act_global.ndim() == 0 \|\| fc1_act_global.size(0) == num_experts_on_rank) is false: fc1 act global must be scalar or (num_experts_on_rank,)` | `tokenspeed_kernel/ops/moe/flashinfer.py:3172` (`flashinfer_cutlass_nvfp4_moe_apply`), startup forward | 362 s | `qwen35-122b-a10b-nvfp4/e27cee86/server-20261112-170130.log` |
| V4.1 subset, TP2 | `ValueError: Unexpected V4.1 expert tensor: layers.0.ffn.experts.128.w1.scale` | `tokenspeed/runtime/models/deepseek_v41.py:2444` (`_load_text_weights`, from `load_weights` :2677): the loader has no `num_checkpoint_experts`, which landed later in `6d6702bae` | 46 s | `dsv41-flash-1017/e27cee86/server-20261112-185406.log` |

The same NVFP4 failure appears in the default-config set (`default/qwen38-27b-nvfp4/e27cee86/server-20261112-055342.log`)
and in the dry run (`default/dryrun/qwen38-27b-nvfp4/e27cee86/server-20261112-044312.log`). The after image starts
all three. I did not bisect which commit fixed each one.

### 4.2 gemma-4 agentX C=16 and 64 (both images, 4 runs)

| run | marker | wall s | AIPerf | server |
|---|---|---|---|---|
| `gemma4-31b-it/e27cee86/agentx/c16` | FAILED `no profile_export_aiperf.json` | 570.7 | warmup phase, then `ERROR No profile results to export (system_controller.py:2350)` | `[smg] ... Stage WorkerSelection failed with status 400 Bad Request` (16:19:38) |
| `gemma4-31b-it/e27cee86/agentx/c64` | same | 1069.3 | same | same (16:25:24) |
| `gemma4-31b-it/ef057889/agentx/c16` | same | 569.9 | `ServerDisconnectedError('Server disconnected')` at 16:41:14.439, then `No profile results to export` | same (16:41:14) |
| `gemma4-31b-it/ef057889/agentx/c64` | same | 1062.5 | same | same (16:46:58) |

The engine's single-request limit is `max_input=199614` (one GPU, 0.9 memory use, KV pool 293,760 tokens), and the
agentX corpus has traces up to 256K tokens. The gateway answers 400 at `WorkerSelection`, and AIPerf's warmup
phase then fails terminally. The 400 is not logged with the request length, so the link to the 199,614 limit is
inferred from the limit and the trace sizes. C=1 completes in both images (11 requests, valid). I did not check
why its traces did not hit the limit. This is not an image difference. Cause and rerun: task 52, below and
section 3.3.

Cause (task 52). Not a counting error. At 0.90 the KV pool is 2295 parents of 10 MiB (22.4 GiB). The capacity model
charges each of the five sliding groups its lookback plus one whole 8192-token prefill chunk (147 parents each,
735 in all, 32% of the pool), which leaves 1560 parents for the full-attention group: 199,615 tokens. The same model
needs 2784 parents for 262,144. The model reproduces the engine's number from the config alone (2294 usable parents:
199,487). Args that reach 262,143 (startup line `Single-request token limit`):
`--kv-cache-dtype fp8` (5 MiB parents, 4617 of them, cache limit 496,831) and `--chunked-prefill-size 1024`
(2339 parents, 276,927); 2048 would give 260,927 and fall short. A prompt above the limit is already rejected per
request by the engine's input processor (`ValueError`); the 400 comes from the gateway's worker selection (smg, not
in the TokenSpeed repo) and the abort from AIPerf's warmup, neither of which this repo can change.

### 4.3 Other failed runs

- `qwen38-27b-fp8/{e27cee86,ef057889}/agentx_smoke/c32`: FAILED, `no profile_export_aiperf.json` (smoke mode,
  section 6).
- Default-config and repro crash runs: section 5.
- Failed probes of the first gemma sessions (no `<bos>`): section 4.4.

### 4.4 gemma-4: `<bos>` and the tool-call parser (harness settings, identical in both images)

- The first gemma sessions failed the startup probe in both images (France ` the the the ...`, primes
  ` 2, 2, 2, ...`; records in `gemma4-31b-it/<image>/_probe_failed_no_bos/`). The gemma-4 tokenizer adds no BOS
  (`tokenizer_config.json` has no `add_bos_token`), so a raw `/v1/completions` prompt has none and the model
  degenerates. With a literal `<bos>` in front, the same prompts answer " Paris." and " 7, 11, 13, ...". Only
  the chat template adds BOS. The harness prefixes `<bos>` to the probes and RAG prompts (queue key
  `prompt_prefix`). agentX uses the chat endpoint and is unchanged.
- With the default parser selection (`tokenizer_config.json` has a `response_template`, so the gateway gets
  `--parser symphony --reasoning-parser hf --tool-call-parser hf`), every `/v1/chat/completions` request on
  gemma-4 fails with HTTP 400 `invalid_parser_configuration`. An explicit `--tool-call-parser gemma4` avoids it.
  It changes only how the gateway parses output text, not the engine. Task 51 follows it up.

## 5. Default config: KV-event crash runs (`default/`, `repro/`)

The `smg_grpc_servicer` that wraps `tokenspeed serve` turns KV-event publishing on when no `--kv-events-config`
is given. The log says `KV event publishing enabled: no --kv-events-config was given ... set
SMG_TOKENSPEED_SERVICER_KV_EVENTS=0 to leave it off`, and `server_args` shows `kv_events_config='{"enable_kv_cache_events":
true, "publisher": "zmq", ...}'`. With it on, the engine aborts under RAG load on Qwen3.8-27B-FP8 in both images:

```
[critical] Fatal check failed: cache mutation on a KV event boundary with no token descriptor
  at /workspace/tokenspeed-scheduler/csrc/scheduler/scheduler.cpp:385 (after) / :386 (before)
  in void tokenspeed::Scheduler::markKvEventBoundaryForReconcile(const tokenspeed::CacheKey&)
Fatal Python error: Aborted
```

`[L2] writeback queued/started` lines come right before the abort. The gateway keeps the container "Up" and
`/health` green, so these first runs (before the watchdog existed) left the client hanging or getting connection
errors. Task 50 traced it: a hybrid model's state checkpoint outlived the KV of its boundary, the drain erased the
boundary's descriptor, and the checkpoint's later eviction hit the check; fixed in the scheduler (section 5.1 and
`deepgemm-analysis.md` 9.46). These runs use the same flags as the primary set minus the env variable. They are kept as the finding, not compared, and not used in any ratio.

First default-config sessions (`default/<model>/<image>/`, RAG only):

| run | marker | wall s | ok/err | out tok/s | TTFT p50 ms | TPOT p50 ms | eff. C | valid | reason |
|---|---|---|---|---|---|---|---|---|---|
| `default/qwen38-27b-fp8/e27cee86/rag/c1` | DONE | 821.4 | 100/0 | 31.3 | 103 | 31.6 | 1.0 | - |  |
| `default/qwen38-27b-fp8/e27cee86/rag/c16` | DONE | 321.3 | 500/0 | 403.6 | 974 | 34.8 | 15.7 | - |  |
| `default/qwen38-27b-fp8/e27cee86/rag/c32` | DONE | 184.3 | 500/0 | 709.7 | 1789 | 37.4 | 31.3 | - |  |
| `default/qwen38-27b-fp8/e27cee86/rag/c64` | DONE | 136.8 | 500/0 | 963.7 | 3376 | 52.6 | 62.6 | - |  |
| `default/qwen38-27b-fp8/e27cee86/rag/c128` | DONE | 137.7 | 158/342 | 244.8 | 4207 | 82.2 | 49.6 | - |  |
| `default/qwen38-27b-fp8/e27cee86/rag/c256` | FAILED | 3.9 | - | - | - | - | - | - | engine crash |
| `default/qwen38-27b-fp8/ef057889/rag/c1` | DONE | 541.2 | 100/0 | 47.7 | 104 | 20.6 | 1.0 | - |  |
| `default/qwen38-27b-fp8/ef057889/rag/c16` | DONE | 252.3 | 500/0 | 515.8 | 978 | 26.5 | 15.7 | - |  |
| `default/qwen38-27b-fp8/ef057889/rag/c32` | DONE | 165.3 | 500/0 | 793.8 | 1825 | 32.6 | 31.4 | - |  |
| `default/qwen38-27b-fp8/ef057889/rag/c64` | DONE | 124.1 | 500/0 | 1065.7 | 3109 | 46.8 | 62.6 | - |  |
| `default/qwen38-27b-fp8/ef057889/rag/c128` | FAILED | - | - | - | - | - | - | - | engine crash |
| `default/qwen38-27b-nvfp4/ef057889/rag/c1` | DONE | 466.0 | 100/0 | 55.4 | 74 | 17.7 | 1.0 | - |  |
| `default/qwen38-27b-nvfp4/ef057889/rag/c16` | DONE | 210.9 | 500/0 | 619.5 | 720 | 22.4 | 15.7 | - |  |
| `default/qwen38-27b-nvfp4/ef057889/rag/c32` | DONE | 136.8 | 500/0 | 965.8 | 1338 | 27.5 | 31.4 | - |  |

- The before image crashed during C=128 too: its C=128 run has a `DONE` marker because AIPerf finished, but 342
  of 500 requests failed (`503 start_generation_failed`: 98 "Cancelling all calls", 244 "tcp connect error"),
  output 244.8 tok/s, and the fatal line is at 05:49:36, 33 s into that run. C=256 then failed after 3.9 s on
  the dead server. So both images crash at C=128 in this config. The task 47 report and the harness README
  counted C=128 as done and C=256 as the crash; that was wrong.
- The after C=128 `FAILED` marker was written by hand later (the old harness did not detect the crash), and its
  `metrics.json` has no AIPerf metrics. The fatal line (05:44:07) is about 31 s after C=64 finished (05:43:36).
- Crash lines: `default/qwen38-27b-fp8/e27cee86/server-20261112-051737.log` line 9031 (05:49:36),
  `default/qwen38-27b-fp8/ef057889/server-20261112-051951.log` line 9015 (05:44:07).
- `default/qwen38-27b-nvfp4/ef057889` ran C=1, 16 and 32 before its worker moved on (no crash).
  `default/qwen38-27b-fp8/e27cee86/_cascade_not_run/` holds the never-run C=512 / agentX records of the old
  harness.

Repro with the watchdog (`repro_queue.txt`, `restart_queue.txt`, fresh server per directory, 500 prompts):

| run | marker | wall s | ok/err | out tok/s | TTFT p50 ms | TPOT p50 ms | eff. C | valid | reason |
|---|---|---|---|---|---|---|---|---|---|
| `repro/qwen38-27b-fp8/ef057889-default/rag/c64` | DONE | 124.7 | 500/0 | 1060.7 | 3051 | 47.0 | 62.6 | - |  |
| `repro/qwen38-27b-fp8/ef057889-default/rag/c128` | FAILED | 35.2 | - | - | - | - | - | - | engine crash |
| `repro/qwen38-27b-fp8/ef057889-nokv/rag/c64` | DONE | 124.7 | 500/0 | 1061.4 | 2897 | 47.0 | 62.6 | - |  |
| `repro/qwen38-27b-fp8/ef057889-nokv/rag/c128` | DONE | 103.6 | 500/0 | 1285.1 | 5844 | 74.4 | 125.0 | - |  |
| `repro/qwen38-27b-fp8/e27cee86-default/rag/c64` | DONE | 137.3 | 500/0 | 961.1 | 2946 | 53.3 | 62.6 | - |  |
| `repro/qwen38-27b-fp8/e27cee86-default/rag/c128` | FAILED | 35.2 | - | - | - | - | - | - | engine crash |
| `repro/qwen38-27b-fp8/ef057889-restart/rag/c64` | DONE | 125.8 | 500/0 | 1059.7 | 2931 | 46.8 | 62.6 | - |  |
| `repro/qwen38-27b-fp8/ef057889-restart/rag/c128` | FAILED | 35.2 | - | - | - | - | - | - | engine crash |
| `repro/qwen38-27b-fp8/ef057889-restart/rag/c16` | DONE | 251.4 | 500/0 | 517.2 | 969 | 26.5 | 15.7 | - |  |

- Default config: C=128 crashes 32-33 s into the run in both images (fatal lines 06:37:43.9, 06:23:59.8 and
  06:50:02.0 against run starts 06:37:11, 06:23:28 and 06:49:30, the end of the C=64 run before it). The watchdog
  ends the run 2-4 s after the fatal line (wall 35.2 s each; `repro/qwen38-27b-fp8/e27cee86-default/server-20261112-062751.log` line 3163,
  `ef057889-default/server-20261112-061536.log` line 3169). The rows are in run order.
- With `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` (`ef057889-nokv`), C=128 completes (500 ok, 0 errors). The
  before-image `nokv` repro was aborted by an operator error (`e27cee86-nokv-ABORTED/NOTE.txt`, no runs). The
  primary before session covers it (C=128 500 ok).
- Restart test (`ef057889-restart`, default config): C=64 done, C=128 crashed
  (`server-20261112-064147.log` line 3158), the server restarted (336 s, `server-20261112-065158.log`), and
  C=16 completed on the new server (`restart_count=1`).
- Only the FP8 27B model was run with KV events on. The crash is in the scheduler, not in the model.

### 5.1 Rerun with the fixed scheduler (task 50)
The scheduler fix (TokenSpeed `d61649596`) was built as the Python extension from the same source tree and mounted
over `tokenspeed_scheduler_ext` in `nctu6/tokenspeed:ef057889-20261112` (nothing else changed: no scheduler or
runtime diff between that commit and the fix besides it). Default config, KV events on (`kv_events_config` with the
ZMQ publisher, no `SMG_TOKENSPEED_SERVICER_KV_EVENTS`), Qwen3.8-27B-FP8, TP1, GPU 6, RAG 500 prompts, C=64 then
C=128 on a fresh server (`results/fix50/`): C=64 500 ok / 0 errors in 124.7 s (1060.7 output tok/s); C=128 500 ok /
0 errors in 103.6 s (1284.7 output tok/s), no fatal line, where the unfixed image died 35 s into C=128. One run per
point; the numbers show that the run completes, not a comparison (the primary set uses the env variable).

### 5.2 Verification of the fixes (task 54)
Local TokenSpeed head `c4681daa6` (tasks 50-53), GPU 7. C++ scheduler tests 610 passed, scheduler Python tests 251,
`test/cli` 200, runtime suite 9018 passed (top-level files 6798), 0 failed; `pre-commit run --all-files` clean.
In-process probes with KV events on and the scheduler extension of the head: Qwen3.8-27B-FP8, Qwen3.8-27B-NVFP4 and
Qwen3.6-35B-A3B answer "Paris" and "2, 3, 5, 7, 11". gemma-4-31B-it served by `tokenspeed serve` (after image, changed
files mounted, default parser args, KV events on): completions and chat answer correctly and a `get_weather` tool call
is parsed. No TP2 probe: GPU 6 was held by another user's job. Details: `deepgemm-analysis.md` 9.50.

## 6. Excluded agentX smoke runs (`agentx_smoke/`)

The first agentX runs used smoke mode: `--num-dataset-entries 8`, `--benchmark-duration 240`,
`--max-context-length 262144`, `--unsafe-override`, seed 0. With 8 entries only about 2 requests are ever in
flight at C>=16, so C=16..512 measure the same thing, and AIPerf marks them `submission_valid=false`
(`unsafe_override`). They are kept but not compared and not used in any ratio:

| run | marker | wall s | ok/err | out tok/s | TTFT p50 ms | TPOT p50 ms | eff. C | valid | reason |
|---|---|---|---|---|---|---|---|---|---|
| `default/dryrun/qwen38-27b-nvfp4/ef057889/agentx_smoke/c1` | DONE | 307.6 | 6/0 | 19.8 | 5106 | 23.6 | 0.6 | no |  |
| `default/dryrun/qwen38-27b-nvfp4/ef057889/agentx_smoke/c16` | DONE | 726.8 | 5/0 | 15.4 | 96023 | 35.3 | 1.8 | no |  |
| `default/dryrun/qwen38-27b-nvfp4/ef057889/rag/c1` | DONE | 227.6 | 48/0 | 55.2 | 80 | 17.7 | 1.0 | - |  |
| `default/dryrun/qwen38-27b-nvfp4/ef057889/rag/c16` | DONE | 24.3 | 48/0 | 606.1 | 843 | 23.2 | 16.0 | - |  |
| `qwen38-27b-fp8/e27cee86/agentx_smoke/c1` | DONE | 285.4 | 5/0 | 19.0 | 8531 | 37.5 | 0.9 | no |  |
| `qwen38-27b-fp8/e27cee86/agentx_smoke/c16` | DONE | 765.5 | 2/0 | 0.7 | 115692 | 153.3 | 0.9 | no |  |
| `qwen38-27b-fp8/e27cee86/agentx_smoke/c32` | FAILED | 1300.6 | - | - | - | - | - | no | no profile_export_aiperf.json |
| `qwen38-27b-fp8/ef057889/agentx_smoke/c1` | DONE | 303.0 | 6/0 | 19.8 | 5433 | 26.4 | 0.7 | no |  |
| `qwen38-27b-fp8/ef057889/agentx_smoke/c16` | DONE | 760.2 | 2/0 | 0.7 | 113525 | 139.2 | 0.9 | no |  |
| `qwen38-27b-fp8/ef057889/agentx_smoke/c32` | FAILED | 1289.1 | - | - | - | - | - | no | no profile_export_aiperf.json |

C=64 was aborted and its directory removed. The paths inside these runs' `aiperf.log` / `aiperf_cmd.txt` still
name the old `agentx/` directory. The dry-run RAG rows (48 prompts, default config) were the pipeline check
before the workers existed.

## 7. Caveats

- One run per point, no repeats. Differences of a few percent are inside the spread. The BF16 control differs
  by 0-1% in RAG output tok/s and TPOT, but by up to 11% in TTFT p50 (RAG) and 4-6% in agentX TPOT p50. TTFT
  p99 and agentX TTFT are heavy-tailed.
- Shared host. Concurrent TP1 sessions ran on GPUs 6 and 7 (section 2.6), and other users' jobs ran on GPUs
  0-5. They share CPU, PCIe, disk, page cache and the network.
- V4.1 is a 128-of-384-expert subset and after-only. Its output is not meaningful and its MoE cost is about a
  third of the full model's. Compare its two backends with each other, not with a full V4.1 deployment.
- Three models are after-only (the before image does not start them), so they have no ratio.
- gemma-4 at TP1 cannot run the 256K agentX traces (`max_req_input_len=199614`), so agentX C=16/64 have no
  result in either image.
- CUDA graphs cover decode batches up to 160 only (`cudagraph_capture_size=160`, the default), while
  `--max-num-seqs` is 512. This affects every server at C>=256 in both images and causes the V4.1 dip at C=256.
  A run with larger capture sizes was not done (task 53).
- Effective concurrency stays below C in agentX (requests wait on 256K-token prefills and between turns) and at
  C=512 in RAG on every server except V4.1 (268-391 of 512: they queue). At C>=128 the RAG ratios compare saturated
  servers.
- agentX runs only at C = 1, 16, 64 with 900 s windows. At C=64 only 14-54 requests finish for most models.
- The server log does not name the GEMM or attention leaf per layer (`selected_kernels.txt` keeps only matching
  log lines). The FP8 gain is attributed to the DeepGEMM SM120 dense leaf from the kernel benchmarks (decode
  1.2-4.0x, prefill parity) and from the unchanged BF16 controls, not from a per-run kernel trace.
- The driver and host details were read the day after the runs (section 2.1).

## 8. Reproduce

Harness: `tokenspeed/tmp/bench_images_local/` (git-ignored scratch in the TokenSpeed checkout; `README.md` there
documents the harness, the watchdog, agentX full mode, the KV-event crash and the gemma details). The live copy
and all raw results are on aws-raichu in `/workspace/nctu6/infra/tokenspeed/tmp/bench_images/`.

```bash
cd /workspace/nctu6/infra/tokenspeed/tmp/bench_images           # aws-raichu
python3 build_rag_prompts.py                                    # once: data/rag-12000-500-seed0.jsonl
tmux new-session -d -s bench  'TMPDIR=$PWD/../bt/g6 python3 -u bench_queue.py --gpus 6 2>&1 | tee -a logs/worker-gpu6.out'
tmux new-session -d -s bench7 'TMPDIR=$PWD/../bt/g7 python3 -u bench_queue.py --gpus 7 2>&1 | tee -a logs/worker-gpu7.out'
cat STATUS                                   # progress
python3 bench_queue.py --gpus 6 --dry-parse  # done / failed counts from the markers
```

`queue.txt` has one session per line (`model= path= tp= gpus= image= env= args= prompt_prefix= hf_overrides=
variant= probe=`). The primary lines set `env=SMG_TOKENSPEED_SERVICER_KV_EVENTS=0`. Runs with a marker are
skipped on a restart (`--retry-failed` reruns failed ones). Results land in
`<model>/<image-short>[-variant]/{rag,agentx}/c<N>/` (`DONE` or `FAILED`, `metrics.json`, `aiperf.log`,
`aiperf_cmd.txt`, `artifacts/`). Default-config runs are in `default/`, and the KV-event repro in `repro/`.

This document is generated from a copy of the result tree (markers, `metrics.json`, `session.json`,
`gpu_at_launch.txt`, `probe.json`, `server_cmd.txt`; no AIPerf artifacts or server logs):

```bash
cd tokenspeed/tmp/bench_images_local/results
python3 make_tables.py benchmark_template.md ../../../../DeepGEMM/deepgemm_benchmark.md
```

`make_tables.py` expands the `{{FRAG ...}}` lines with the tables of `make_detail_tables.py` (summary, run
order, per-run tables, default / repro / smoke runs). Every ratio is computed from the unrounded JSON values.

Counts, primary set: 70 RAG runs and 26 agentX full-mode runs done, 4 agentX runs failed (gemma C=16/64), and
30 runs of the three failed before-image sessions never started. Every primary session has `restarts: 0`.
