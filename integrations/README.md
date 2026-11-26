# Standalone PyTorch, Hugging Face Transformers & vLLM Integrations

TFQ3 and TFQ4 checkpoints are **not locked to TensorFold**. This directory provides standalone loaders and CUDA bindings to run any `.tffpw` (`TFPN` v3) or `.tffp4` (`TFP4` v4) checkpoint directly in standard PyTorch, Hugging Face `transformers`, or vLLM.

## 1. Hugging Face `transformers` + PyTorch (`[MEASURED]` end-to-end)

```python
from integrations.pytorch.tfq_torch import load_tfq_into_hf
from transformers import AutoTokenizer

model = load_tfq_into_hf("./Qwen2.5-1.5B-TFQ3", device="mps", cache_decoded=True)
tok = AutoTokenizer.from_pretrained("./Qwen2.5-1.5B-TFQ3")
ids = tok("The randomized Hadamard transform", return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**ids, max_new_tokens=32, do_sample=False)[0]))
```

Works identically for both **TFQ3 (3/4/5-bit)** (`Tfq3Linear`) and **TFQ4-E8P (2.001 bpw)** (`Tfq4Linear`).

## 2. Native PyTorch CUDA Kernel Extension (`sm_80+`, zero BF16 weight expansion)

Build the extension binding [`src/kernels/cuda/tfq3_cuda.cuh`](../src/kernels/cuda/tfq3_cuda.cuh) directly to `torch::Tensor`:

```bash
pip install --no-build-isolation -e integrations/pytorch
```

Once installed, `Tfq3Linear.forward()` on CUDA automatically dispatches to `tfq3_cuda_ext.tfq3_linear` (single-weight-read multi-row decode bit-exact for `rows = 1..64`, fused `BM=64, BN=128` Tensor Core prefill for long contexts, plus optional `row_off`, `k_sign_off`, `bc0_buf`, and `n_bc_buf` parameters for zero-repack Col-TP and Row-TP).

## 3. vLLM Quantization Plugin (`[UNIT-TESTED / LIVE SERVING PENDING]`)

[`integrations/vllm/tfq3_quant.py`](vllm/tfq3_quant.py) implements vLLM's `QuantizationConfig` (`Tfq3Config`) and `LinearMethodBase` (`Tfq3LinearMethod`), automatically registering `--quantization tfq3` via `register_quantization_config("tfq3")` when vLLM is installed:
- **Per-tensor bit widths:** `Tfq3Config.from_config()` reads `tensors` from `tfq3_manifest.json` (supporting uniform 3b/4b/5b and mixed-bit checkpoints, skipping unquantized residual layers like `lm_head`).
- **Real `.tffpw` + `pw_v3_tables.bin` loader:** `Tfq3CheckpointReader` and `load_tfq3_checkpoint_weights()` load packed shards directly into layer parameters (with per-parameter `weight_loader` hooks).
- **Merged projections:** `qkv_proj` (3-way), `gate_up_proj` (2-way), and arbitrary N-way fused layers track independent per-partition 128-row padding, bit widths, and absolute row signs (`row_off`).
- **Tensor Parallelism (`TP > 1`):** Column-Parallel (`row_off`) and Row-Parallel (`k_sign_off` + 1-block trellis state resume `bc0_buf`) dispatch directly to `tfq3_cuda_ext.tfq3_linear` on CUDA with an exact portable fallback on CPU/MPS.

Verified end-to-end against `tfq3_reference` in [`tests/vllm_plugin_test.py`](../tests/vllm_plugin_test.py). *(Note: live multi-GPU vLLM server benchmarks require installing `vllm` + `tfq3_cuda_ext` on a CUDA host.)*
