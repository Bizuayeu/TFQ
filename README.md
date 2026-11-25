# TFQ — TensorFold Native Quantization (`TFQ3`)

**TFQ3** is an open, **100% data-free (zero-calibration)** 3-, 4-, and 5-bit weight quantization format and Apple Silicon Metal GPU converter suite built for [TensorFold](https://github.com/ashhart/TensorFold).

Instead of uniform affine quantization or calibration-dependent Hessian fitting (GPTQ/AWQ/imatrix), TFQ3 combines:
1. A deterministic **randomized 2D Fast Walsh–Hadamard Transform (`2D FWHT-128`)** on every `128×128` weight tile to analytically eliminate channel outliers.
2. A single fixed **16 KiB universal Gaussian `2D-AffineCoset/half4` vector codebook** (`pw_v3_tables.bin`) with state-dependent coset hashing and zero-cost `split-K` resume.
3. **16 KiB–aligned zero-copy `.tffpw` shards** that `mmap` directly into Metal (`bufferNoCopy`) for sub-second model loads (`0.2s–0.5s` for 27B) and native Apple Silicon SIMD / M5 NAX execution.

---

## Why TFQ3?

On **Qwen3.8-27B** (exact `fp64` full-vocab KL divergence across all 248,320 logits vs. 8-bit gold, zero calibration data):

| Quantization | Disk Size | Unseen PPL (3,071 tok) | Mean KL vs 8-bit | p99 KL vs 8-bit | Same Top-1 | Load Time |
|---|---:|---:|---:|---:|---:|---:|
| MLX 8-bit (Gold) | 27.50 GiB | 6.4485 | — | — | — | 3.27 s |
| MLX 4-bit (g64 affine) | 14.98 GiB | 6.6021 | `0.03352` | `0.27518` | 91.19% | 3.72 s |
| **TFQ3-4b (Pure BF16 base)** | **14.37 GiB** | **6.5829** | **`0.02316` (−31%)** | **`0.15037` (−45%)** | **91.68%** | **0.53 s** |
| **TFQ3-4b (8-bit residual base)** | **13.23 GiB** | **6.5855** | **`0.02308` (−31%)** | **`0.14865` (−46%)** | **91.29%** | **0.20 s** |
| TFQ3-3.5b (8-bit residual base) | 11.59 GiB | 6.6478 | `0.03779` | `0.25983` | 91.39% | 0.19 s |

### Across Model Families (Apple M5 Ultra)

| Model | Source | MLX 4-bit | **TFQ3 Shipped** | Quality vs Gold | TensorFold Runtime PR |
|---|---|---:|---:|---|---|
| **Qwen3.8-27B** (Dense 64L) | BF16 or MLX 8-bit | 14.98 GiB | **14.37 GiB** (BF16 base) / **13.23 GiB** (Q8 base) | **−31% mean KL / −46% p99 KL** vs MLX 4-bit | [`ashhart/TensorFold#4974`](https://github.com/ashhart/TensorFold/pull/4974) |
| **Qwen3.8-Flash-Next** (512E MoE) | MLX 8-bit (189 GiB) | 105.50 GiB | **78.99 GiB** (fits 128 GB Macs!) | **5.996 PPL** vs 6.007 gold (MLX 4-bit: 6.644) | [`ashhart/TensorFold#4964`](https://github.com/ashhart/TensorFold/pull/4964) |
| **GLM-5.3-Flash** (288E MoE) | FP8 / MLX 8-bit (320 GiB) | 180.6 GB | **139.6 GB** (130 GiB) | **22.61 PPL** vs 24.16 FP8 (MLX 4-bit: 25.32) | [`ashhart/TensorFold#4953`](https://github.com/ashhart/TensorFold/pull/4953) |

*See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for complete ablation tables, throughput (`p2048` / `B=1` / `B=4` / `MTP`), and methodology.*

---

## Requirements & Build

- **macOS on Apple Silicon** (M1–M5) with Xcode Command Line Tools (`clang++`, `xcrun metal`, `xcrun metallib`).
- **Zero external C++ dependencies** — builds cleanly in ~3 seconds with standard `make`.

```bash
git clone https://github.com/ashhart/TFQ.git
cd TFQ
make          # builds all GPU converters, metallibs, and tfq3-inspect into build/
make test     # runs Python mathematical reference self-test & CLI checks
```

Built artifacts in `build/`:
- `tfq3-qwen27-convert` (+ `.metallib`) & `tfq3-qwen27-package` — Dense `Qwen3.8-27B` converter (BF16 & MLX 8-bit/64)
- `tfq3-flashnext-convert` (+ `.metallib`) — `Qwen3.8-Flash-Next` 512-expert MoE + PLE converter
- `tfq3-glm-convert` (+ `.metallib`) — `GLM-5.3-Flash` 288-expert MoE + Non-MoE converter
- `tfq3-inspect` — Standalone validator and inspector for `pw_v3_tables.bin`, `TFPN`, and `TFPW` shards

---

## Quick Start: Quantize Your Own Model

### 1. Qwen3.8-27B (from Official Hugging Face BF16 or MLX 8-bit)

One command converts, packages residuals/config/tokenizer, and validates the standalone checkpoint (~3 minutes on Apple M5 Ultra):

```bash
# Option A: Pure BF16 source (e.g. Qwen/Qwen3.8-27B) -> 14.37 GiB TFQ3-4b (zero MLX quantization anywhere)
./scripts/quantize_qwen27.sh /path/to/Qwen3.8-27B /path/to/Qwen3.8-27B-TFQ3 4b bf16

# Option B: MLX 8-bit/64 source -> 13.23 GiB TFQ3-4b (-1.75 GiB smaller than MLX 4-bit)
./scripts/quantize_qwen27.sh /path/to/Qwen3.8-27B-MLX-8bit /path/to/Qwen3.8-27B-TFQ3-Q8 4b q8

# Option C: Ultra-compact 3.5b hybrid (~11.59 GiB Q8 base / ~12.72 GiB BF16 base) or pure 3b (~10.38 GiB)
./scripts/quantize_qwen27.sh /path/to/Qwen3.8-27B /path/to/Qwen3.8-27B-TFQ3-3p5b 3.5b bf16
```

**What `tfq3-qwen27-convert` + `tfq3-qwen27-package` handle automatically:**
- Auto-detects HF BF16 (`model.language_model.*` + bare `lm_head.weight`), MLX (`language_model.model.*`), or plain (`model.*`) namespaces.
- Auto-detects HF 0-centered norms (`mean < 0.5`) and applies `+1.0` fold into the TensorFold/MLX 1-centered convention (leaving `linear_attn.norm.weight` untouched).
- Converts `F32` scalar norms to `BF16` and reshapes HF `conv1d.weight` from `[10240, 1, 4]` to `[10240, 4, 1]`.
- Emits 4 shards (`<= 3.43 GiB` each for 16 GB Mac `maxBufferLength` safety) with contiguous `gate+up` and `k+v` tensors for fused GPU dispatch.
- Strips `"quantization"` from `config.json` automatically when `--residual bf16` is selected.

Run the output directly in TensorFold:
```bash
tensorfold-native info /path/to/Qwen3.8-27B-TFQ3
tensorfold-native serve /path/to/Qwen3.8-27B-TFQ3 --host 0.0.0.0 --port 18650 --context 32768
```

### 2. Qwen3.8-Flash-Next (512-Expert MoE -> ~79 GiB for 128 GB Macs)

```bash
# Boundary layers (L0-L5, L47) with 5b/4b/3b, deep layers (L6-L46 + MTP) with 4b/3b
./build/tfq3-flashnext-convert --model /path/to/FlashNext-8bit --out /path/to/FlashNext-TFQ3 --mode moe --layers 0 5 --n5 64 --n4 192
./build/tfq3-flashnext-convert --model /path/to/FlashNext-8bit --out /path/to/FlashNext-TFQ3 --mode moe --layers 6 46 --n5 0 --n4 192
./build/tfq3-flashnext-convert --model /path/to/FlashNext-8bit --out /path/to/FlashNext-TFQ3 --mode moe --layers 47 47 --n5 64 --n4 192 --mtp

# Requantize PLE embeddings to 3-bit/32 (23.84 GiB) and package standalone checkpoint
./build/tfq3-flashnext-convert --model /path/to/FlashNext-8bit --out /path/to/FlashNext-TFQ3 --mode requant-ple --ple-bits 3
python3 src/converters/flashnext/package.py --src /path/to/FlashNext-8bit --out /path/to/FlashNext-TFQ3 --extract-nonmoe --write-index --ple-bits 3 --move-ple

# Validate all 49 shards + codebook
./build/tfq3-inspect /path/to/FlashNext-TFQ3
```

*(Pass `--trace <routing_counts.txt>` to enable optional HeatAlloc routing-frequency bit allocation across experts; omit `--trace` for pure index-order or uniform bit allocation.)*

### 3. GLM-5.3-Flash (288-Expert MoE -> 130 GiB)

```bash
# MoE layers (42 routed layers) + unified non-MoE shard
./build/tfq3-glm-convert --model /path/to/GLM-5.3-Flash-FP8 --out /path/to/GLM-TFQ3 --mode heat543 --layers 3 6 [--trace routing.txt]
./build/tfq3-glm-convert --model /path/to/GLM-5.3-Flash-FP8 --out /path/to/GLM-TFQ3 --mode heat43  --layers 7 44 [--trace routing.txt]
./build/tfq3-glm-convert --model /path/to/GLM-5.3-Flash-FP8 --out /path/to/GLM-TFQ3 --mode nonmoe --fast4b
./build/tfq3-inspect /path/to/GLM-TFQ3
```

### 4. Inspect & Validate Any TFQ3 Checkpoint or Reference-Decode a Tile

```bash
# Validate alignment, bounds, bit widths, bpw, and per-tensor SNR
./build/tfq3-inspect -v /path/to/Qwen3.8-27B-TFQ3

# Pure Python + NumPy mathematical reference (exact bit-for-bit decode of any TFPN tensor tile)
python3 reference/tfq3_reference.py --self-test
python3 reference/tfq3_reference.py --shard /path/to/qwen27_tfq3-00001-of-00004.tffpw --tensor model.layers.0.mlp.down_proj
```

---

## Documentation

- **[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)** — Mathematical formulation of `2D FWHT-128`, `2D-AffineCoset/half4` vector quantization, zero-calibration design, state-hash split-K resume, and Apple Silicon Metal tile layout.
- **[`docs/FORMAT_SPEC.md`](docs/FORMAT_SPEC.md)** — Byte-level binary specification for `pw_v3_tables.bin` (`PWV3`), dense/non-MoE shards (`TFPN`), and MoE layer shards (`TFPW`) for implementing loaders in any inference engine.
- **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)** — Full KL divergence, perplexity, SNR, throughput, and memory footprint measurements across `Qwen3.8-27B`, `Qwen3.8-Flash-Next`, and `GLM-5.3-Flash`.
- **[`reference/tfq3_reference.py`](reference/tfq3_reference.py)** — Clean, standalone NumPy reference implementation matching the Metal kernels.

---

## Calibration FAQ

- **Is calibration data used to quantize weights?** **No.** Every dense and MoE weight matrix is quantized 100% data-free (no WikiText/C4/RedPajama pass, no Hessians, no imatrix). The 2D FWHT-128 rotation Gaussianizes arbitrary weight matrices analytically, and the 16 KiB codebook (`pw_v3_tables.bin`) is generated once from a synthetic unit Gaussian prior.
- **What about `HeatAlloc` on MoE models?** Optional. For MoE models with hundreds of experts (288–512), you can optionally pass `--trace` containing per-layer expert routing counts so hot experts receive 4- or 5-bit budgets while cold experts use 3-bit. Even with `HeatAlloc`, the weights themselves are still quantized 100% data-free. Dense models (`Qwen3.8-27B`) use zero data anywhere.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
