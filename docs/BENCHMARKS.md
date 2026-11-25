# TFQ3 Verified Benchmarks

All numbers below were measured directly on **Apple M5 Ultra** (512 GB unified memory, TensorFold Metal) and **NVIDIA GB10** (SM 12.1, CUDA 13.0, TensorFold CUDA) using the **exact same cross-platform TFQ3 checkpoints**.

---

## 1. Qwen3.8-27B (Dense 64-Layer Hybrid GDN + Attention)

- **Calibration data used:** **None (100% data-free)** — zero calibration prompts, zero Hessians, zero imatrix.
- **Weight SNR (`tff_tfq3_qwen27_convert`):** `3b = 17.17 dB`, `4b = 23.37 dB` (`lm_head = 22.70 dB`), conversion time ~3.2 min on Apple M5 Ultra.
- **Perplexity evaluation:** Exact `fp64` log-softmax over **4,790 tokens** (`1,717` multi-domain + `3,071` completely unseen README/Zig/CHANGELOG).
- **KL divergence evaluation:** Exact `fp64` full-vocab $D_{\text{KL}}(P_{\text{8-bit}} \| P_q)$ across **all 248,320 logits** over **1,022 positions** (511 multi-domain + 511 unseen).

### Quality & KL Divergence vs. MLX 8-bit Gold

| Variant | Disk Size | 4,790-tok PPL | Unseen 3,071-tok PPL | Mean KL | Median KL | p95 KL | p99 KL | Max KL | Rev KL | Same Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLX 8-bit/64 (Gold) | 27.50 GiB | 5.2468 | 6.4485 | — | — | — | — | — | — | — |
| MLX 4-bit/64 | 14.98 GiB | 5.3694 | 6.6021 | 0.03352 | 0.01183 | 0.11810 | 0.27518 | 1.0228 | 0.03444 | 91.19% |
| **TFQ3-4b (Pure BF16 base)** | **14.37 GiB** | **5.3680** | **6.5829** | **0.02316** | **0.00927** | **0.07941** | **0.15037** | **0.7051** | **0.02220** | **91.68%** |
| **TFQ3-4b (8-bit base)** | **13.23 GiB** | **5.3681** | **6.5855** | **0.02308** | **0.00980** | **0.07763** | **0.14865** | **0.6968** | **0.02213** | 91.29% |
| TFQ3-3.5b (8-bit base) | 11.59 GiB | 5.4619 | 6.6478 | 0.03779 | 0.01362 | 0.15305 | 0.25983 | 0.9311 | 0.03843 | 91.39% |
| TFQ3-3.5b (Pure BF16 base) | 12.72 GiB | 5.4646 | 6.6461 | — | — | — | — | — | — | — |
| TFQ3-3b (8-bit base)* | 10.38 GiB | 4.3093* | — | — | — | — | — | — | — | — |

*\*`TFQ3-3b` was evaluated on the initial 1,716-token corpus where `MLX-8bit = 4.0354`, `MLX-4bit = 4.1150`, `TFQ3-4b = 4.1197`, and `TFQ3-3.5b = 4.1689`.*

### Throughput & Startup Load (Apple M5 Ultra)

| Variant | Cold Load (`mmap`) | `p2048` Prefill | `B=1` Decode | `B=4` Decode | 32k×8 Startup Budget |
|---|---:|---:|---:|---:|---:|
| MLX 8-bit/64 | 3.27 s | 955.8 tok/s | 27.8 tok/s | 104.4 tok/s | — |
| MLX 4-bit/64 | 3.72 s | 1,869.8 tok/s | 48.4 tok/s | 179.4 tok/s | 43.90 GiB |
| **TFQ3-4b (Pure BF16 base)** | **0.53 s** | 1,184.4 tok/s | **45.4 tok/s** | 140.4 tok/s | **43.29 GiB** |
| **TFQ3-4b (8-bit base)** | **0.20 s** | 1,178.4 tok/s | 44.4 tok/s | **142.2 tok/s** | **42.15 GiB** |
| TFQ3-3.5b (Pure BF16 base) | 0.31 s | 1,229.3 tok/s | **46.3 tok/s** | 139.9 tok/s | — |
| TFQ3-3.5b (8-bit base) | **0.19 s** | 1,222.9 tok/s | 45.3 tok/s | **142.4 tok/s** | — |

### Direct HF-BF16 Source Verification
Converting a 4-layer + `lm_head` HF-format BF16 checkpoint (`model.language_model.*`, bare `lm_head.weight`, 0-centered norms, `F32` `linear_attn.norm.weight`, `[10240, 1, 4]` `conv1d`) vs. MLX-8bit source:
- All 30 residual tensors byte-identical (non-norm) and `0.000000` max norm roundtrip diff.
- All 26 TFQ3 tensors matched within `0.0023 dB` max SNR diff (`4b = 23.38 dB`, `lm_head = 22.70 dB`).

---

## 2. Qwen3.8-Flash-Next (48L + MTP, 512-Expert MoE, Unlocks 128 GB Macs)

- **MoE compression:** 49 routed layers converted in 17.5 min (`48.98 GiB` at `3.411 bpw` avg; `3b = 17.13 dB`, `4b = 23.33 dB`, `5b = 29.31 dB`). Non-MoE + vision kept at 8-bit/32 (`6.17 GiB`); PLE N-gram embeddings ablated at 3-bit/32 (`23.84 GiB`) vs 4-bit/32 (`29.80 GiB`).

| Checkpoint | Disk Size | Main NLL (1,670 tok) | Held-Out NLL (334 tok) | Combined PPL (2,004 tok) | `p2048` Prefill | `B=1` Decode | `MTP d3` Decode | `d0 == d3` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLX 8-bit/32 (Gold) | 189.04 GiB | 1.7315 | 2.1003 | 6.007 | 3,422 tok/s | 79.2 tok/s | 153.9 tok/s | Bit-exact |
| MLX 4-bit/32 | 105.50 GiB | 1.8324 | 2.2006 | 6.644 (`+0.101`) | 3,914 tok/s | 106.2 tok/s | 212.1 tok/s | Bit-exact |
| **TFQ3 + PLE 3-bit/32 (Shipped)** | **78.99 GiB** | **1.7298** | **2.0978** | **5.996 (`-0.002`)** | **3,855 tok/s** | **82.9 tok/s** | **154.5 tok/s** | **Bit-exact** |
| TFQ3 + PLE 4-bit/32 | 84.95 GiB | 1.7336 | 2.1210 | 6.038 (`+0.005`) | 3,856 tok/s | 82.9 tok/s | 160.1 tok/s | Bit-exact |

---

## 3. GLM-5.3-Flash (45L, 288-Expert MoE, Fits 192 GB / 256 GB Macs)

- **MoE + Non-MoE compression:** `139.55 GB` (`129.97 GiB`) total vs `319.76 GB` FP8 (`-56.4%`) and `180.6 GB` MLX 4-bit (`-41 GB`).

| Metric | Raw FP8 (320 GB) | MLX 4-bit (181 GB) | **TFQ3 (139.6 GB / 130 GiB)** |
|---|---:|---:|---:|
| 4-Prompt Weighted NLL / PPL (235 tok) | 3.1846 / 24.16 | 3.2316 / 25.32 | **3.1184 / 22.61** |
| `p2048` Prefill | — | 1,400 tok/s | **1,567 tok/s (+11.9%)** |
| Decode `B=1` | — | 75.5 tok/s | **69.6 tok/s** |
| Live Streaming (`MTP + copies`) | — | — | **87.0–104.3 tok/s** |
| MTP `depth 3 == depth 0` | Bit-exact | Bit-exact | **Bit-exact** |

---

## 4. NVIDIA CUDA Kernel & Converter Verification (NVIDIA GB10, SM 12.1)

Measured directly on **NVIDIA GB10** (48 SMs, ~273 GB/s LPDDR5X peak bandwidth, CUDA 13.0, `nvcc -arch=sm_121`) via `make test-cuda CUDA_ARCH=sm_121`:

- **Numerical parity (`tfq3-cuda-test`):** `PASS` against exact `fp64` reference across all 3b/4b/5b direct PTX `m16n8k16` split-K=8 (`sk8`) and fused dense kernels, **bit-exact speculative verification (`rows=1 == rows=16 == rows=32`)**, mixed-bit 3b/4b/5b routed MoE (`act_mode=0` clamped and `act_mode=1`), `sr_split` fused `gate+up` / `k+v`, strided outputs, and BF16 `mv1`/`mm` kernels.
- **CUDA BF16 Converter (`tfq3-qwen27-convert-cuda`):** 4 full `Qwen3.8-27B` layers (`2.83 GiB` BF16 -> `0.708 GiB` TFQ3-4b, `4.001 bpw`) quantized in **21.8 s** (~5.5 s/layer) at **23.35 dB** mean SNR (matching Metal's `23.37 dB`), passing `tfq3-inspect` with zero errors.

### Standalone CUDA Kernel Microbenchmarks (`src/kernels/cuda/tfq3_cuda.cuh`, Direct PTX `m16n8k16`, 4-bit TFQ3)

| Qwen3.8-27B Projection | Shape `[N × K]` | `rows=1` Latency | `rows=1` BW | `rows=16` Verify Latency | `rows=16` Verify BW | `rows=256` Prefill |
|---|---|---:|---:|---:|---:|---:|
| `gate+up` | `[34816 × 5120]` | 0.336 ms | **265.2 GB/s** | 0.489 ms | **182.2 GB/s** (11.66 TFLOP/s) | **17.74 TFLOP/s** (5.15 ms) |
| `down` | `[5120 × 17408]` | 0.187 ms | **238.5 GB/s** | **0.198 ms** | **225.0 GB/s** (14.40 TFLOP/s) | **23.56 TFLOP/s** (1.94 ms) |
| `qkv` (L2-hot) | `[10240 × 5120]` | 0.079 ms | **330.1 GB/s** | **0.106 ms** | **248.4 GB/s** (15.89 TFLOP/s) | **26.10 TFLOP/s** (1.03 ms) |
| `lm_head` | `[248320 × 5120]` | 2.713 ms | **234.4 GB/s** | **2.812 ms** | **226.1 GB/s** (14.47 TFLOP/s) | — |

---

## 5. End-to-End Speculative Decoding Benchmarks (Metal & CUDA)

All speculative decoding measurements below were verified **100% bit-exact** (`--no-drafts` SHA == drafted SHA at `temperature = 0.0`) on native TensorFold runtimes.

### Apple Silicon Metal (Apple M5 Ultra)

| Model | Target Size | Drafter | Plain (`--no-drafts`) | Drafted Decode | Speedup | Parity |
|---|---:|---|---:|---:|---:|---|
| **Qwen3.8-27B-TFQ3** | 14.37 GiB | **DFlash2** | 44.2 tok/s | **77.4 tok/s** (5.95 tok/round) | **+75%** | Bit-exact |
| **GLM-5.3-Flash-TFQ3** | 129.97 GiB | **MTP + copies** | 69.2 tok/s | **95.2 tok/s** | **+38%** | Bit-exact |
| **GLM-5.3-Flash-TFQ3** | 129.97 GiB | **DFlash2 (4-bit)** | 69.2 tok/s | **85.3 tok/s** | **+23%** | Bit-exact |
| **Qwen3.8-Flash-Next-TFQ3** | 78.99 GiB | **MTP (`d3`) + copies** | 82.9 tok/s | **154.5 tok/s** | **+86%** | Bit-exact |

### NVIDIA CUDA (NVIDIA GB10, SM 12.1, Direct PTX `m16n8k16` Split-K=8)

- **Live `Qwen3.8-27B-TFQ3` (`tensorfold-native serve`, greedy `temp=0`, 256 completion tokens):**

| Prompt Mode | `--no-drafts` | **+ DFlash2** | **+ DSpine** | Bit-Exact (`nodraft == DFlash2 == DSpine`) |
|---|---:|---:|---:|---|
| Coding (`thinking=False`) | 15.1 tok/s | **52.5 tok/s** (**3.48×**, 4.65 tok/round) | **56.6 tok/s** (**3.75×**, 5.12 tok/round) | **100% bit-exact** (`sha=738f62e12436`) |
| Reasoning (`thinking=True`) | 15.1 tok/s | **43.8 tok/s** (**2.90×**, 3.94 tok/round) | **45.6 tok/s** (**3.02×**, 4.34 tok/round) | **100% bit-exact** (`sha=1454fc526024`) |

- **Real-Shard CUDA Parity & Drafter Verification (`GLM-5.3-Flash` & `Qwen3.8-Flash-Next`):**
  - **GLM-5.3-Flash** (`nonmoe.tffpw` 4b/5b `sk8` + `fused`, `layer_03.tffpw` 288-expert 3b/4b/5b MoE `top_k=8`): `cos = 0.999997–1.000000` vs `fp64`, bit-exact `rows=1 == rows=16` (`0` diffs), `215–244 GB/s` dense; live `GLM-5.3-Flash-DFlash2` (`bits=4` and BF16) verified end-to-end with shared TFQ3 `DraftHead`.
  - **Qwen3.8-Flash-Next** (`layer_00.tffpw` target MoE & `mtp_00.tffpw` MTP drafter MoE, 512 experts, 3b/4b/5b `top_k=10` + 3-bit/32 PLE): `cos = 0.999998–1.000000` vs `fp64`, bit-exact `rows=1 == rows=16` (`0/25,600` diffs).
