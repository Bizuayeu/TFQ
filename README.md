<p align="center">
  <img src="assets/tfq3-header.png" alt="TensorFold TFQ3 — 3 · 4 · 5-Bit Vector Quantization for Metal and CUDA" width="100%">
</p>

# TFQ — TensorFold Native Quantization (`TFQ3` & `TFQ4`)

**Cross-Platform: Apple Silicon (Metal) & NVIDIA GPUs (CUDA `sm_80+`) · Standalone PyTorch / Hugging Face / vLLM & TensorFold**

**TFQ3** (*TensorFold Quantization v3 — supports **3-, 4-, and 5-bit**, including mixed-bit per layer or per expert; `3` is the format version, not 3-bit only*) and **TFQ4-E8P** (`2.001 bpw` 8D E8-initialized Voronoi codebook) provide a fast, **data-free by default** (with **optional Hessian-LDLQ calibration** and **optional trace-guided MoE HeatAlloc**) weight quantization format, GPU converter suite, and standalone inference kernels.

> **One Checkpoint Format, Byte-Identical Conversion Across Platforms:**
> The same `.tffpw`/`.tffp4` checkpoint file runs unmodified on both Apple Silicon (Metal) and NVIDIA GPUs (CUDA `sm_80+`), and both Metal and CUDA converters emit **byte-for-byte identical files** (matching SHA-256) from the same source weights via embedded canonical tables (`reference/tables/`) and deterministic IEEE-754 encoder paths. *(Inference kernels on Metal vs CUDA use each architecture's native MMA and reduction schedules, so cross-vendor inference outputs match within normal fp16/bf16 rounding tolerance.)*

### How TFQ Works (and When to Use Data-Free vs. Calibrated Mode)
1. **Default mode (`100% data-free`, minutes to convert):** Applies a deterministic **randomized 2D Fast Walsh–Hadamard Transform (`2D FWHT-128`)** to every `128×128` tile, sets the `fp16` block scale to `RMS * shrink`, and encodes 2D weight pairs via a **256-wide beam search** over 32-bit trellis states against one fixed **16 KiB canonical Gaussian codebook** (`pw_v3_tables.bin`). For MoE models without `--trace`, expert sensitivity is scored **100% data-free** from router row L2 norms + expert weight-scale energy ([`src/converters/moe_heat_rank.hpp`](src/converters/moe_heat_rank.hpp)) — never index order.
2. **Optional calibration-aware modes (same binary format, zero runtime cost):**
   - **Hessian-LDLQ (`tools/tfq3_hessian.py`):** Optimizes real activation output error $\|(W - \hat{W})X^T\|_F^2$ in TFQ3's rotated domain via block-Cholesky LDLQ + closed-form Hessian-optimal fp16 tile scales, exporting standard `TFPN`/`TFP4` shards.
   - **Trace-guided HeatAlloc (`--trace <counts.txt>`):** Uses calibration routing counts (explicitly **uses calibration data**) to assign higher bit widths to hotter MoE experts.
3. **16 KiB–aligned zero-copy shards + bit-exact speculative decoding:** Fixed reduction order per `(N, K)` guarantees **100% bit-exact** `DFlash2`, `DSpine`, and `MTP` speculative verification (`--no-drafts == drafted` at `temp=0`) across the tested decode ranges: **`rows = 1..32` on Metal** and **`rows = 1..64` on CUDA** (single-weight-read multi-row decode across all Qwen3.8-27B shapes and `N > 34816`).

---

## Get Started

### 1. Install `tfq3` (one command — auto-detects Mac Metal or NVIDIA CUDA)

```bash
curl -fsSL https://raw.githubusercontent.com/ashhart/TFQ/main/install.sh | bash
```

*(Prefer building from a local clone? Run `git clone https://github.com/ashhart/TFQ.git && cd TFQ && make install`.)*

### 2. Convert any Safetensors model (or fit to your exact RAM/VRAM budget)

```bash
# Convert any model (defaults to 4-bit TFQ3 into ./<ModelName>-TFQ3)
tfq3 Qwen/Qwen3.8-27B
tfq3 meta-llama/Llama-3.1-8B-Instruct --target-gib 5.0

# Auto-fit to this machine's available GPU memory (data-free mixed 3b/4b/5b sensitivity planner)
tfq3 /path/to/any-safetensors-model --fit auto

# Quantize vision encoders, speculative drafters, embeddings, or 2.001 bpw TFQ4-E8P
tfq3 /path/to/Qwen3.8-27B-DFlash2 --quantize-drafter --bits 4
tfq3 /path/to/Qwen2.5-1.5B --bits 2 --quantize-embed   # 2.001 bpw TFQ4-E8P -> 0.360 GiB in 1.3 s

# GLM-5.3-Flash: compact (3.39 bpw avg + 4b non-MoE) OR matched uniform-4b + full-precision non-MoE
tfq3 /path/to/GLM-5.3-Flash-FP8 --glm-mode uniform4 --keep-nonmoe-bf16

# ERNIE-5.5-Flash (CUDA): uniform 3b/4b/5b routed experts from the BF16 checkpoint; other tensors stay in the source
tfq3 baidu/ERNIE-5.5-Flash-Preview-BF16 --profile 4b

# Validate any TFQ3/TFQ4 checkpoint & check local toolchain
tfq3 inspect -v ./Qwen3.8-27B-TFQ3
tfq3 doctor
```

### 3. Run Anywhere — PyTorch / Hugging Face, vLLM, or TensorFold

TFQ checkpoints are **not locked to TensorFold**:

```python
# Standard PyTorch + Hugging Face Transformers (zero TensorFold dependency — Metal/MPS, CPU, or CUDA)
from integrations.pytorch.tfq_torch import load_tfq_into_hf
model = load_tfq_into_hf("./Qwen2.5-1.5B-TFQ3", device="mps", cache_decoded=True)
```

```bash
# Optional native PyTorch CUDA extension (`src/kernels/cuda/tfq3_cuda.cuh`, zero BF16 weight expansion):
pip install --no-build-isolation -e integrations/pytorch

# Or serve natively on Mac (Metal) or NVIDIA GPU (CUDA) with TensorFold:
tensorfold serve ./Qwen3.8-27B-TFQ3 --host 0.0.0.0 --port 18650 --context 32768
tensorfold serve ./Qwen3.8-27B-TFQ3 --drafter /path/to/Qwen3.8-27B-DFlash2
```

See [`integrations/README.md`](integrations/README.md) for PyTorch, Hugging Face, and vLLM plugin details.

---

## Controlled Head-to-Head Quality Benchmarks `[MEASURED]`

Controlled comparison on **`Qwen/Qwen2.5-1.5B`** (all **196** transformer linear layers quantized; `embed_tokens` & `lm_head` kept BF16; **`16,352` held-out WikiText-2 test tokens**; **`4,088` full-vocab exact `fp64` KL tokens**; held-out real-activation Output SNR). Full SHA-256 provenance receipt: [`benchmarks/provenance_qwen25_1p5b.json`](benchmarks/provenance_qwen25_1p5b.json) (reproduce with `tools/bench_quantizers.py`).

| Method (`Gold BF16 PPL = 11.1726`) | Eff. bpw | Calib? | Held-Out Out-SNR | WikiText-2 PPL | Mean KL | p99 KL | Conv Time |
|---|---:|---|---:|---:|---:|---:|---:|
| RTN / MLX Affine 4b (`g64`) | 4.500 | No | 21.50 dB | 12.1522 | 0.13772 | 1.0543 | 4.2 s |
| AWQ Salient-Scale 4b (`g128`) | 4.250 | Yes | 22.38 dB | 12.0603 | 0.10383 | 0.9321 | 6.8 s |
| GPTQ / LDLQ Scalar 4b (`g128`) | 4.250 | Yes | 23.96 dB | 11.8600 | 0.07855 | 0.8114 | 62.6 s |
| **TFQ3-4b Data-Free (real `.tffpw`)** | **4.001** | **No** | **24.29 dB** | **11.8615** | **0.07978** | **0.5299** | **26.2 s** |
| **TFQ3-4b + Hessian-LDLQ (`TFPN`)** | **4.001** | Yes | **24.66 dB** | **11.7776** | 0.08310 | 0.6385 | 181.7 s |
| EXL3/QTIP-ref 4b (per-col LDLQ) | 4.001 | Yes | **25.60 dB** | **11.5248** | **0.06385** | 0.6899 | 93.6 s |
| RTN / MLX Affine 3b (`g64`) | 3.500 | No | 14.86 dB | 17.8444 | 0.56950 | 3.4547 | 0.4 s |
| AWQ Salient-Scale 3b (`g128`) | 3.250 | Yes | 15.76 dB | 17.2014 | 0.50336 | 3.7428 | 7.3 s |
| GPTQ / LDLQ Scalar 3b (`g128`) | 3.250 | Yes | 17.22 dB | 15.5764 | 0.43779 | 3.8489 | 62.4 s |
| **TFQ3-3b Data-Free (real `.tffpw`)** | **3.001** | **No** | **18.20 dB** | **14.0925** | **0.32752** | **2.1367** | **24.8 s** |
| EXL3/QTIP-ref 3b (per-col LDLQ) | 3.001 | Yes | **19.93 dB** | **13.0780** | **0.24721** | 2.3660 | 95.1 s |

**Trade-off summary:** Data-free TFQ3 converts in seconds with zero calibration data and beats RTN/MLX-Affine, calibrated AWQ, and calibrated GPTQ (especially at 3-bit: `-1.48 PPL` and `-44%` p99 KL vs GPTQ at `0.25` fewer bpw). Full per-column Hessian LDLQ (`EXL3/QTIP/EAQ`) still achieves lower PPL/KL when you can afford calibration data and slow conversion.

### Flagship TensorFold Runtime Families `[MEASURED — TensorFold runner, no per-sequence JSON receipt in benchmarks/]`

> *Provenance note: The table below summarizes earlier TensorFold hardware measurements; unlike the Qwen2.5-1.5B receipt above, these runs do not yet have committed per-sequence JSON receipts under `benchmarks/`. On `Qwen3.8-Flash-Next` (`2,004` scored tokens), the `0.011` PPL difference between `5.996` and `6.007` is within sampling noise (i.e. parity with 8-bit gold).*

| Model | Recipe & Exact bpw | **TFQ3 Size** | Quality Status | Metal & CUDA Runtime / Drafters |
|---|---|---:|---|---|
| **Qwen3.8-27B** (Dense 64L) | 4.001 bpw linear + BF16 or Q8 residuals (data-free) | **14.37 GiB** (BF16) / **13.23 GiB** (Q8) | **`[MEASURED]`** 4,790-tok PPL `5.368`, −31% mean KL / −46% p99 KL vs MLX 4-bit (`248,320` vocab, 1,022 pos) | Metal [`#4974`](https://github.com/ashhart/TensorFold/pull/4974) (`DFlash2` **+75%**, bit-exact) · CUDA [`#4986`](https://github.com/ashhart/TensorFold/pull/4986) (`BM=64` **1.73×** prefill, `DFlash2` **3.50×**, `DSpine` **3.97×**, bit-exact) |
| **Qwen3.8-Flash-Next** (512E MoE) | 3.411 bpw MoE + 8b non-MoE + 3b PLE | **78.99 GiB** (fits 128 GB Macs) | **`[MEASURED]`** 2,004-tok PPL `5.996` vs `6.007` 8-bit gold (within noise; `MLX 4b = 6.644`) | Metal [`#4964`](https://github.com/ashhart/TensorFold/pull/4964) (`MTP d3` **+86%**, `154.5 tok/s`, bit-exact) · CUDA [`#4986`](https://github.com/ashhart/TensorFold/pull/4986) |
| **GLM-5.3-Flash** (288E MoE) | Compact: `3.39 bpw` MoE (`heat43`=3.33 bpw) + 4b/5b `nonmoe.tffpw` (or `--glm-mode uniform4 --keep-nonmoe-bf16` for 4.00 bpw + BF16 non-MoE) | **139.6 GB** (compact) / **167.5 GB** (uniform 4b + BF16 non-MoE) | **`[MEASURED — 4,094 tok]`** Matched-4b PPL `1.8326` vs `1.7943` FP8 (`+0.0211` nats); full task eval pending | Metal [`#4953`](https://github.com/ashhart/TensorFold/pull/4953) (`p2048 +11.9%`, `MTP` `95.2 tok/s`, `DFlash2` `85.3 tok/s`, bit-exact) · CUDA [`#4986`](https://github.com/ashhart/TensorFold/pull/4986) |

*See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for complete tables, provenance tags, and Metal/CUDA microbenchmarks.*

---

## Requirements, Build & Verification

- **Apple Silicon (macOS, M1–M5 — Metal):** Xcode Command Line Tools (`clang++`, `xcrun metal`, `xcrun metallib`). `[MEASURED]`
- **NVIDIA GPUs (Linux, `sm_80+` — CUDA):** CUDA Toolkit (`nvcc` 12+/13+) and `g++`. `[MEASURED]`
- **AMD ROCm / HIP (`src/kernels/hip/tfq3_hip.h`):** `[SOURCE-ONLY / NOT HARDWARE-VERIFIED]` — portable wave32/wave64 kernels written and syntax-checked only; no AMD GPU hardware verification has been run yet.

```bash
git clone https://github.com/ashhart/TFQ.git
cd TFQ

# macOS (Apple Silicon Metal):
make install           # builds Metal converters (TFQ3 + TFQ4) + metallibs and installs `tfq3` into ~/.local/bin
make test              # runs Python references, golden vectors, data-free MoE ranker, strict validator controls, and GPU E2E round-trips
make test-metal        # fp64 Metal parity (3b/4b/5b bit-exact sk8 r1==r16==r32 + BM=32 prefill + Col-TP/Row-TP=4) + bench
make test-tfq4-metal   # fp64 Metal parity (2.001 bpw 8D E8-initialized Voronoi, bit-exact r1==r16) + bench

# Linux (NVIDIA CUDA — pass CUDA_ARCH=sm_80 / sm_86 / sm_89 / sm_90 / sm_100 / sm_121):
make install CUDA_ARCH=sm_121
make test CUDA_ARCH=sm_121
make test-cuda CUDA_ARCH=sm_121
```

---

## Documentation & Tools

- **[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)** — Mathematical formulation of `2D FWHT-128`, `2D-AffineCoset/half4`, state-hash split-K resume, and Metal/CUDA execution.
- **[`docs/FORMAT_SPEC.md`](docs/FORMAT_SPEC.md)** — Byte-level binary specification for `PWV3`/`PWV4` tables and `TFPN`/`TFPW`/`TFP4` shards.
- **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)** — Full provenance-tagged benchmarks (`[MEASURED]` / `[SOURCE-INSPECTED]` / `[NOT VERIFIED]`).
- **[`docs/TFQ4_SUB3BIT.md`](docs/TFQ4_SUB3BIT.md)** — Sub-3-bit (`2.001 bpw`) `TFQ4-E8P` cold-expert lattice format and benchmarks.
- **[`tools/tfq3_hessian.py`](tools/tfq3_hessian.py)** — Optional calibration-aware Hessian-weighted Beam-Viterbi + LDLQ encoder (`TFPN`/`TFP4` compatible).
- **[`tools/bench_quantizers.py`](tools/bench_quantizers.py)** — Reproducible WikiText-2 PPL, full-vocab KL, and real-activation Output SNR benchmark + SHA-256 provenance generator.
- **[`integrations/`](integrations/README.md)** — Standalone PyTorch, Hugging Face `transformers`, and vLLM integrations.
- **[`reference/tfq3_reference.py`](reference/tfq3_reference.py)** & **[`reference/tfq4_lattice.py`](reference/tfq4_lattice.py)** — Pure-Python NumPy reference decoders.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
