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

Once installed, `Tfq3Linear.forward()` on CUDA automatically dispatches to `tfq3_cuda_ext.tfq3_linear` (split-K=8 bit-exact decode/verify for `rows <= 64`, `BM=64, BN=128` Tensor Core prefill for long contexts).

## 3. vLLM Quantization Plugin (`[SOURCE-INSPECTED / INTERFACE-TESTED]`)

[`integrations/vllm/tfq3_quant.py`](vllm/tfq3_quant.py) implements vLLM's `QuantizationConfig` (`Tfq3Config`) and `LinearMethodBase` (`Tfq3LinearMethod`) against `tfq3_cuda_ext.tfq3_linear`. *(Note: end-to-end vLLM serving requires installing `vllm` + `tfq3_cuda_ext` on a CUDA host; this repo's CI verifies the plugin interface against unit stubs.)*
