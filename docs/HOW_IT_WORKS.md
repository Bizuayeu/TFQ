# How TFQ3 Works

TFQ3 is a **cross-platform, transform-coded 2D vector quantization** scheme designed for both **high weight fidelity** (beating uniform affine 4-bit by ~2.5 dB weight SNR) and **high GPU throughput on both Apple Silicon (Metal) and NVIDIA GPUs (CUDA `sm_80+`)** — the same binary checkpoint file runs unmodified on both platforms, and both the Metal and CUDA converters produce byte-for-byte identical files (matching SHA-256) from the same source weights. *(Runtime inference kernels use each architecture's native MMA/reduction schedules, so Metal and CUDA inference outputs match within normal fp16/bf16 rounding tolerance rather than bit-for-bit across vendors.)*

---

## 1. Why Uniform Affine Quantization Loses Quality

Standard group-wise affine quantization (e.g. MLX 4-bit/g64) partitions each row into 64-element groups and maps weights linearly onto 16 equally spaced levels between `[min, max]`:
- **Outlier dilation:** A single heavy-tailed weight in a 64-element group stretches the entire group's dynamic range, wasting quantization bins for the other 63 weights.
- **1D rectangular grid inefficiency:** Quantizing Gaussian-distributed scalars independently along each coordinate hits the scalar Lloyd–Max ceiling (~20.2 dB at 4-bit for $\mathcal{N}(0, 1)$, ~20.9 dB with per-64-group min/max).
- **Per-group metadata tax:** Storing two `fp16`/`bf16` values (`scale`, `bias`) per 64 weights adds **+0.50 bpw** overhead (4.50 effective bpw for "4-bit").

TFQ3 solves all three simultaneously without requiring any calibration dataset.

---

## 2. Step 1 — Deterministic Randomized 2D FWHT-128

For any weight matrix $W \in \mathbb{R}^{R \times C}$ (with $R, C$ multiples of 128), TFQ3 partitions $W$ into $128 \times 128$ blocks and applies an **orthonormal randomized 2D Walsh–Hadamard Transform**:

$$\widetilde{W} = H_{128} \, D_L \, W_{\text{blk}} \, D_R \, H_{128}$$

where:
- $H_{128} = \frac{1}{\sqrt{128}} H_2^{\otimes 7}$ is the normalized 128-point Walsh–Hadamard matrix ($H_{128}^T = H_{128} = H_{128}^{-1}$).
- $D_L = \text{diag}(s_L)$ and $D_R = \text{diag}(s_R)$ are deterministic Rademacher sign vectors ($\pm 1$) generated purely from global row/column indices using Knuth's 32-bit multiplicative hash:

```c
inline float pw_sgn(uint seed, uint idx) {
    uint h = (seed ^ idx) * 2654435761u;
    return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}
```

with fixed seeds `SEED_COL = 0x12345678u` and `SEED_ROW = 0x9E3779B9u`.

### Why This Is 100% Data-Free and Fast at Inference

1. **Analytical Gaussianization:** By the Central Limit Theorem over 128 independent randomized sign combinations across both rows and columns, every entry of $\widetilde{W}$ becomes near-i.i.d. Gaussian $\mathcal{N}(0, \sigma^2)$ with kurtosis collapsing from $>50$ down to `~3.0` — regardless of weight outliers and **without ever seeing calibration activations**.
2. **Negligible block scale overhead:** Because all 16,384 entries inside a $128 \times 128$ block share the same variance $\sigma^2$, TFQ3 stores only **one `fp16` RMS scale per $128 \times 128$ block** — just **`0.001 bpw`** of metadata (vs `0.50 bpw` for g64 affine)!
3. **Exact runtime equivalence:** Because $D_L H_{128}$ and $H_{128} D_R$ are orthogonal involutions, the matrix-vector product $y = W x$ computes cleanly in the rotated domain:
   $$x_{\text{rot}} = H_{128} D_R \, x, \qquad y_{\text{rot}} = \widetilde{W} \, x_{\text{rot}}, \qquad y = D_L H_{128} \, y_{\text{rot}}$$
   Each 128-point FWHT runs in a single 32-lane SIMD register shuffle (`pw_fwht128_simd4`, 4 elements/lane, 7 butterfly stages, zero shared memory!).

---

## 3. Step 2 — `2D-AffineCoset/half4` Vector Quantization

After dividing each $128 \times 128$ block $\widetilde{W}$ by its single `fp16` RMS scale $\sigma$, the normalized entries follow $\mathcal{N}(0, 1)$.

Instead of quantizing 1 scalar at a time, TFQ3 quantizes **pairs of weights $(w_0, w_1) \in \mathbb{R}^2$** using a **2D vector codebook combined with a lightweight state-dependent affine coset**:

### Universal 16 KiB Codebook (`pw_v3_tables.bin`)
Generated deterministically once from a synthetic $\mathcal{N}(0, I_2)$ prior (identical across every model and every layer):
- **3-bit (`lut3` = 1 KiB):** 4 independent 2D k-means codebooks of 64 points each (`256 × half2`). Each weight pair consumes 6 bits (3 bits/weight).
- **4-bit (`lut4` = 4 KiB):** 4 independent 2D k-means codebooks of 256 points each (`1024 × half2`). Each weight pair consumes 8 bits (4 bits/weight).
- **5-bit (`lut5` = 4 KiB):** 1 shared 2D k-means codebook of 1,024 points (`1024 × half2`). Each weight pair consumes 10 bits (5 bits/weight).
- **64 Affine Coset Transforms (`c64_3`, `c64_4`, `c64_5` = 512 B each):** 64 precomputed `half4(s_x, s_y, t_x, t_y)` scale-and-shift tuples that apply tiny deterministic affine perturbations to the 2D codebook based on a rolling 32-bit state `st`.

Total table footprint: **10,816 bytes** (padded to **16,384 bytes** = 16 KiB), fitting 100% inside the GPU L0 constant cache.

### State-Dependent Coset Hashing & Zero-Cost `split-K` Resume

Each GPU thread (`lane ∈ [0..31]`) decodes 8 weights (4 pairs = two `half4` vectors `q0, q1`) per `16×16` subtile step using a rolling 32-bit state `st`:
- At block column `bc = 0`: `st = 0x13579BDFu ^ (lane * 0x9E3779B1u)`.
- From `st`, 4 multiplicative hashes (`h0..h3`, `× 0x9E3779B1u`) select the sub-codebook/coset XOR bits and affine tuple `c64[(h >> 16) & 63]` for the 4 weight pairs.
- Crucially, **`st` is updated directly to the last 32 bits of the packed weight bitstream** (`st = wc` for 4b, `st = (st << 24) | m24` for 3b, `st = lo32` for 5b).
- **Why this matters for GPU speed:** To start a `split-K` threadgroup at *any* arbitrary 128-column block boundary `bc0 > 0`, the kernel does **not** need to walk from `bc = 0` or store sideband state! It simply loads the single preceding 32-bit weight word at `bc0 - 1` (`q27_pw_init_st`) and immediately has the exact rolling state `st` for `bc0`!

### Encoding (Apple Silicon Metal & NVIDIA CUDA)
At conversion time, both the Metal (`src/converters/metal_encode_common.metal`) and CUDA (`src/converters/cuda_encode_common.cuh`) GPU encoders run the exact same deterministic **256-wide beam search** over 32-bit trellis states (`K_KEEP = 5` for 3b/4b, `6` for 5b) using spatial acceleration grids over `lut3`/`lut4`/`lut5` and a deterministic 64-bin histogram cut with prefix-scan compaction. Before encoding each $128 \times 128$ rotated block $\widetilde{W}$, its `fp16` block scale is set once to $\sigma = \mathrm{fp16}(\mathrm{RMS}(\widetilde{W}) \cdot \text{shrink})$ (`0.92 / 0.94 / 0.96` in the generic converter for `3b / 4b / 5b`, `0.992 / 0.997 / 0.999` in `qwen27`, `1.0` in MoE). Both backends compile encoder kernels without fast-math/FMA contraction, embed the same canonical `PWV3` tables, and emit **byte-for-byte identical `.tffpw` files** (matching SHA-256). *(Optional calibration-aware Hessian-LDLQ encoding with closed-form Hessian-optimal scale polish lives in [`tools/tfq3_hessian.py`](../tools/tfq3_hessian.py).)*

Achieved weight SNR on real transformer weights (zero calibration):
- **3-bit (`3.001 bpw`):** **`17.17 dB`**
- **4-bit (`4.001 bpw`):** **`23.37 dB`** *(+2.5 dB over MLX affine 4-bit/g64 at 4.50 bpw!)*
- **5-bit (`5.001 bpw`):** **`29.43 dB`**

---

## 4. Step 3 — Hardware-Native Tile Layout & Packing

Every $128 \times 128$ block is partitioned into **8 horizontal strips of 16 rows** (`strip = rb * 8 + s`, $s \in [0..7]$), and each `16 × 128` strip is subdivided into **8 subtiles of `16 × 16`**.

Inside each `16 × 16` subtile, 32 lanes (one Apple Silicon SIMD group or one NVIDIA CUDA warp) decode **8 weights per lane** (`q0` = 4 columns in row `home.y`, `q1` = 4 columns in row `home.y + 8`):
- **Apple Silicon (Metal, M1–M5):**
  - During **decode**, each lane reads `half4` of rotated input $x_{\text{rot}}$ once and executes two `half4` FMAs (`fma(q0, xv, acc0)`, `fma(q1, xv, acc1)`).
  - During **prefill**, decoded `q0, q1` populate `tfp::frag<half>` registers directly in place — feeding Apple M5 NAX `tfp::mma_16x32` with zero shared-memory staging (and falling back cleanly to `simdgroup_matrix` via `TF_SIMD_FRAGS` on M1–M4).
- **NVIDIA GPUs (CUDA, `sm_80+` Ampere / Ada / Hopper / Blackwell):**
  - Each warp decodes its `16 × 16` strip into a bank-conflict-free `16 × 24` shared-memory tile (`ldm = 24`, activations at `ldm = 136`) and executes direct PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` Tensor Core instructions.
  - Multi-row decode (`rows = 1..64`) decodes each weight tile **once per CTA** and runs all `MTILES = ceil(rows / 16) ∈ {1..4}` row slices in an inner loop (`tfq3_mma_sk_kernel<BITS, MTILES>` for split-K, `tfq3_mma_fused_mt_kernel<BITS, MTILES>` for `ksplit = 1`). The split-K factor is fixed solely by `(N, K)` (`ks = 1` with zero partial buffer for wide projections `N >= 32768`, `ks = 4/8` with ≤ `10 MiB` partials for narrower `N`), guaranteeing **100% bit-exact** outputs across `rows = 1..64` for `--no-drafts`, `DFlash2`, `DSpine`, and `MTP` speculative verification.
  - Long-context prompt prefill (`rows > 64`) uses `tfq3_mma_fused_mt_kernel<BITS, 4>`, zero-repack multi-GPU Tensor Parallelism uses `tfq3_mma_prefill_tp_kernel`, and routed MoE layers run fused mixed-bit `3b/4b/5b` `gate+up+SwiGLU` and `down` kernels (`tfq3_moe_gu_swiglu_kernel`, `tfq3_moe_down_kernel`).

Word packing per `16 × 128` block (8 subtiles × 32 lanes):
- **4-bit:** `8` uint32 words/lane (`2 × uint4` vectorized AoS loads = `256` uint32 words per strip-block).
- **3-bit:** `6` uint32 words/lane (8 × 24-bit subtile codes packed across two 3-word SoA triples = `192` uint32 words per strip-block).
- **5-bit:** `10` uint32 words/lane (8 × 40-bit subtile codes packed across two 5-word SoA quintuples = `320` uint32 words per strip-block).

### Contiguous Pair Fusion (`sr_split`)
For paired projections sharing the same input (e.g., `mlp.gate_proj` + `mlp.up_proj`, or `self_attn.k_proj` + `self_attn.v_proj`), the converter writes them back-to-back inside the same `.tffpw` shard with contiguous scale and weight offsets. The runtime verifies contiguity (`canFusePair`) and executes both in a single kernel launch by passing `sr_split = N_first`, which resets the row-sign hash index (`r - sr_split`) for rows `>= sr_split` so the output matches separate projections bit-for-bit!

---

## 5. Step 4 — Dual-Base Support (Pure BF16 & MLX Affine) + Optional MoE `HeatAlloc`

### Dense Models (`Qwen3.8-27B`)
- Every dense matrix (`qkv`, `z`, `out`, `q`, `k`, `v`, `o`, `gate`, `up`, `down`, `lm_head`) is an exact multiple of 128 (`H=5120, I=17408, V=248320`), encoded directly into 4 `.tffpw` shards (`<= 3.43 GiB` each).
- Only the non-128-aligned `linear_attn.in_proj_a` and `in_proj_b` (`48 × 5120`, `45 MiB` BF16 total across 48 GDN layers), `embed_tokens`, norms, `conv1d`, `A_log`, and `dt_bias` live in `model.safetensors`.
- Those residuals can be kept as **pure BF16** (zero `quantization` block in `config.json`, `14.37 GiB` total) or **MLX 8-bit/64** (`13.23 GiB` total). Both achieve virtually identical KL (`0.02316` vs `0.02308`) and PPL (`5.3680` vs `5.3681`).

### MoE Models (`Qwen3.8-Flash-Next`, `GLM-5.3-Flash`)
- MoE weight quantization is **100% data-free** by default (or optionally calibration-aware via [`tools/tfq3_hessian.py`](../tools/tfq3_hessian.py) when you want Hessian-weighted beam search + block-Cholesky LDLQ error feedback into the exact same `.tffpw`/`.tffp4` format).
- Because MoE routing exhibits strong power-law skew across 288–512 experts, each expert stores its own `b_gu` and `b_down` bit width (`3`, `4`, or `5`) in the layer's `TFPW` header (`HeatAlloc`). Without `--trace`, expert sensitivity is scored **100% data-free** from router gate row L2 norms + expert weight-scale energy ([`src/converters/moe_heat_rank.hpp`](../src/converters/moe_heat_rank.hpp)) — never index order; passing `--trace <file>` switches to trace-guided routing counts. For matched full-width comparisons, use `--glm-mode uniform4 --keep-nonmoe-bf16`. Hot experts and boundary layers can be assigned 4–5 bits while cold experts use 3 bits — dropping `Qwen3.8-Flash-Next` from `105.5 GiB` (MLX 4-bit) to **`78.99 GiB`** while improving PPL from `6.644` to **`5.996`** (matching 8-bit gold `6.007`).
