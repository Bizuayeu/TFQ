# TFQ3 Verified Benchmarks

All numbers below were measured directly on **Apple M5 Ultra** (512 GB unified memory) running TensorFold native Metal kernels.

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
