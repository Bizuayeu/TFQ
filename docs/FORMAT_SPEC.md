# TFQ3 Binary Format Specification (v3)

TFQ3 is a **single cross-platform binary format**: the exact same `pw_v3_tables.bin` and `.tffpw` files run unmodified on both **Apple Silicon (Metal)** and **NVIDIA GPUs (CUDA `sm_80+`)** with zero re-quantization or repacking.

All integers and floats are **little-endian**. Every GPU-mapped payload section (`scales` and `weights`) is aligned to **16,384 bytes (16 KiB)** so Apple Silicon runtimes can `mmap` files read-only and wrap sections directly with `MTLDevice.newBufferWithBytesNoCopy` (zero copies, zero peak RAM spike), while CUDA runtimes `mmap` and DMA-upload contiguous aligned buffers straight into device memory.

---

## 1. Universal Codebook File: `pw_v3_tables.bin` (`PWV3`)

Fixed size: **16,384 bytes** (active payload: **10,816 bytes**, zero-padded to 16 KiB).

| Offset | Size (bytes) | Type | Field | Description |
|---:|---:|---|---|---|
| `0` | 4 | `u32` | `magic` | `0x33565750` (`"PWV3"`) |
| `4` | 4 | `u32` | `version` | `3` |
| `8` | 24 | `f32[6]` | `a3, b3, a4, b4, a5, b5` | Affine coset scale/shift hyperparameters (`3b/4b/5b`) |
| `32` | 32 | `u32[8]` | `pad0` | Reserved (`0`) |
| `64` | 1,024 | `u32[256]` | `lut3` | 256 packed `half2` 2D centroids (`4` codebooks × `64` entries) |
| `1088` | 4,096 | `u32[1024]` | `lut4` | 1,024 packed `half2` 2D centroids (`4` codebooks × `256` entries) |
| `5184` | 4,096 | `u32[1024]` | `lut5` | 1,024 packed `half2` 2D centroids (`1` codebook × `1,024` entries) |
| `9280` | 512 | `half4[64]` | `c64_3` | 64 affine coset tuples `(sx, sy, tx, ty)` for 3-bit |
| `9792` | 512 | `half4[64]` | `c64_4` | 64 affine coset tuples `(sx, sy, tx, ty)` for 4-bit |
| `10304` | 512 | `half4[64]` | `c64_5` | 64 affine coset tuples `(sx, sy, tx, ty)` for 5-bit |
| `10816` | 5,568 | `u8[5568]` | `pad1` | Zero padding to `16,384` bytes |

**Validation requirements:**
- File size `>= 10,816` bytes (canonically `16,384` bytes).
- `magic == 0x33565750` and `version == 3`.
- All 6 `f32` scalars and all `half` entries in `lut3`, `lut4`, `lut5`, `c64_3`, `c64_4`, `c64_5` must be finite (no NaN/Inf).

---

## 2. Dense / Non-MoE Shard Format: `*.tffpw` (`TFPN`)

Used for dense models (`qwen27_tfq3.tffpw` or sharded `qwen27_tfq3-00001-of-00004.tffpw` … `00004-of-00004.tffpw`) and GLM non-MoE weights (`nonmoe.tffpw`).

### File Layout

```
[0 .. 64)                      TfpnHeader (64 bytes)
[off_meta .. off_meta + N*128) TfpnTensorMeta[num_tensors] (128 bytes each)
[... zero padding ...]
[off_scales .. off_weights)    fp16 block scales (16 KiB aligned start)
[... zero padding ...]
[off_weights .. file_end)      u32 packed weight words (16 KiB aligned start & end)
```

### `TfpnHeader` (64 bytes)

```c
struct TfpnHeader {
    uint32_t magic;              // 0x4E504654 ("TFPN")
    uint32_t version;            // 3
    uint32_t num_tensors;        // number of TensorMeta entries
    uint32_t pad0;               // 0
    uint64_t off_meta;           // byte offset of TensorMeta[] (>= 64)
    uint64_t off_scales;         // byte offset of half[] scales (16 KiB aligned)
    uint64_t off_weights;        // byte offset of u32[] weights (16 KiB aligned)
    uint64_t total_scale_halfs;  // total fp16 scale elements
    uint64_t total_weight_words; // total u32 weight words
    uint64_t pad1;               // 0
};
```

### `TfpnTensorMeta` (128 bytes)

```c
struct TfpnTensorMeta {
    char     name[80];    // NUL-terminated canonical tensor name (e.g. "model.layers.0.mlp.gate_proj", "lm_head")
    uint32_t R;           // output rows (positive multiple of 128)
    uint32_t C;           // input columns (positive multiple of 128)
    uint32_t b_bits;      // 3, 4, or 5
    uint32_t sc_off;      // start offset in fp16 elements inside `scales`
    uint64_t word_off;    // start offset in u32 words inside `weights`
    uint64_t word_count;  // exact word count: (R/16) * (C/128) * (b_bits * 2) * 32
    float    snr_db;      // encoder-measured quantization SNR in dB
    uint32_t pad[3];      // [0] = orig_R (0 -> R), [1] = orig_C (0 -> C), [2] = flags bitmask
};
```

### Backward-Compatible Metadata Extension (`pad[3]`)
Model-agnostic conversions (arbitrary matrix dimensions zero-padded to multiples of 128, plus optional vision / drafter / embedding quantization) record metadata inside the existing `pad[3]` words with zero format break:
- `pad[0]`: unpadded original row count `orig_R` (`0` on legacy shards → treat as `R`).
- `pad[1]`: unpadded original column count `orig_C` (`0` on legacy shards → treat as `C`).
- `pad[2]`: flags bitmask (`0` on legacy shards):
  - `bit 0` (`0x1`): tensor was zero-padded to `[R, C]` from `[orig_R, orig_C]`.
  - `bit 1` (`0x2`): vision encoder / multimodal projector tensor (`--quantize-vision`).
  - `bit 2` (`0x4`): speculative drafter tensor (`--quantize-drafter`).
  - `bit 3` (`0x8`): token embedding table (`--quantize-embed`).

### Zero-Repack Tensor Parallelism (`Col-TP` & `Row-TP`)
Because every `128 × 128` tile is independently rotated and every column block `bc` carries its exact 32-bit state hash inside the final `u32` word of block `bc - 1`, any `TFPN` or `TFPW` weight matrix can be sharded across `TP` ranks at runtime with **zero weight repacking**:
- **Column-TP (`n_start, n_end` multiples of 128):** each rank computes output rows `[n_start, n_end)` independently.
- **Row-TP (`k_start, k_end` multiples of 128):** each rank rotates its input slice `X[:, k_start:k_end]` using sign offset `k_sign_off = k_start`, resumes the per-lane state hash at `bc0 = k_start / 128` directly from `bc0 - 1`'s last packed word (zero overhead), multiplies `[k_start, k_end)`, and sums partial outputs via `AllReduce` before applying the final inverse row-FWHT-128.

### Tensor Scale & Weight Indexing

For a tensor of shape `[R, C]` (`n_rb = R / 128`, `n_bc = C / 128`, `wpb = b_bits * 2`):
- **Scales:** `scales[sc_off + rb * n_bc + bc]` is the `fp16` RMS scale for $128 \times 128$ block `(rb, bc)`. Total scale elements = `n_rb * n_bc`.
- **Weights:** Partitioned into `R / 16` horizontal strips (`strip = rb * 8 + s`, $s \in [0..7]$). Strip `strip`, column block `bc` begins at:
  `word_idx = word_off + (strip * n_bc + bc) * wpb * 32`
  - **4-bit (`wpb = 8`):** Stored as `2` consecutive `uint4[32]` arrays (`h ∈ {0, 1}`, lane-major AoS: `((uint4*)blk)[h * 32 + lane]` holds the 4 `u32` subtile words for columns `[h*64 + 0..64)`).
  - **3-bit (`wpb = 6`):** Stored as `2` consecutive `uint32[3][32]` SoA planes (`h ∈ {0, 1}`, `blk[(h * 3 + p) * 32 + lane]` for $p \in \{0, 1, 2\}$ packing four 24-bit subtile codes across 96 bits).
  - **5-bit (`wpb = 10`):** Stored as `2` consecutive `uint32[5][32]` SoA planes (`h ∈ {0, 1}`, `blk[(h * 5 + p) * 32 + lane]` for $p \in \{0..4\}$ packing four 40-bit subtile codes across 160 bits).

### Contiguous Pair Fusion Rule (`canFusePair`)
Two tensors `A` (`[R0, C]`) and `B` (`[R1, C]`) in the **same shard** can execute in a single fused Metal or CUDA launch (with `sr_split = R0`) if and only if:
1. `A.C == B.C` and `A.b_bits == B.b_bits`
2. `B.sc_off == A.sc_off + (A.R / 128) * (A.C / 128)`
3. `B.word_off == A.word_off + A.word_count`

---

## 3. MoE Layer Shard Format: `layer_XX.tffpw` / `mtp_00.tffpw` (`TFPW`)

Used for routed MoE layers in `GLM-5.3-Flash` (`288` experts) and `Qwen3.8-Flash-Next` (`512` experts).

### `TfpwHeader` (144 bytes)

```c
struct TfpwHeader {
    uint32_t magic;          // 0x57504654 ("TFPW")
    uint32_t version;        // 3
    uint32_t layer_idx;      // layer index
    uint32_t num_experts;    // expert count (e.g. 288 or 512)
    uint32_t num_tensors;    // 3 * num_experts (gate, up, down)
    uint32_t hidden_size;    // model hidden_size (multiple of 128)
    uint32_t moe_inter;      // moe_intermediate_size (multiple of 128)
    uint32_t reserved0;      // 0
    uint64_t raw_fp8_bytes;  // uncompressed reference byte count
    uint64_t stream_words;   // total u32 weight words
    uint64_t scale_halfs;    // total fp16 scale elements
    uint64_t off_meta;       // byte offset of TfpwExpertMeta[num_experts]
    uint64_t off_scales;     // 16 KiB aligned fp16 scales section
    uint64_t off_weights;    // 16 KiB aligned u32 weights section
    uint64_t reserved1[4];   // 0
    char     meta_tag[32];   // NUL-terminated profile/tag label
};
```

### `TfpwExpertMeta` (64 bytes)

```c
struct TfpwExpertMeta {
    uint32_t b_bits;         // bit width for this expert (3, 4, or 5)
    uint32_t heat_rank;      // HeatAlloc routing rank (0 = hottest)
    uint64_t gate_word_off;  // u32 word offset for gate_proj [moe_inter, hidden_size]
    uint64_t up_word_off;    // u32 word offset for up_proj   [moe_inter, hidden_size]
    uint64_t down_word_off;  // u32 word offset for down_proj [hidden_size, moe_inter]
    uint32_t gate_sc_off;    // fp16 scale offset for gate_proj
    uint32_t up_sc_off;      // fp16 scale offset for up_proj
    uint32_t down_sc_off;    // fp16 scale offset for down_proj
    float    gate_snr_db;    // encoder-measured SNR for gate_proj
    float    up_snr_db;      // encoder-measured SNR for up_proj
    float    down_snr_db;    // encoder-measured SNR for down_proj
    uint32_t reserved[2];    // 0
};
```

---

## 4. Mandatory Loader Validation Checks

Any compliant reader (and `build/tfq3-inspect`) enforces:
1. `magic` and `version == 3` match.
2. `off_scales % 16384 == 0` and `off_weights % 16384 == 0`.
3. Strict non-overlapping monotonic sections: `sizeof(Header) <= off_meta <= meta_end <= off_scales <= scales_end <= off_weights <= weights_end <= file_size`.
4. All arithmetic checked against 64-bit integer overflow.
5. Every tensor/expert has `bits ∈ {3, 4, 5}`, `R > 0`, `C > 0`, `R % 128 == 0`, `C % 128 == 0`, exact expected `word_count`, and offsets strictly inside `[0, total_scale_halfs]` and `[0, total_weight_words]`.
6. No duplicate tensor names across multi-shard `TFPN` directories.
