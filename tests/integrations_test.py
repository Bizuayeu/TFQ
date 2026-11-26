#!/usr/bin/env python3
"""
Verification tests for standalone PyTorch / Hugging Face / vLLM integrations
(zero TensorFold runtime dependency).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "integrations", "pytorch"))
sys.path.insert(0, os.path.join(REPO_ROOT, "integrations", "vllm"))

from tfq_torch import Tfq3Linear, Tfq4Linear, load_tfq_into_hf
from tfq3_quant import Tfq3Config, Tfq3LinearMethod


def test_vllm_plugin_contract() -> None:
    cfg = Tfq3Config(bits=4)
    assert cfg.get_name() == "tfq3"
    assert cfg.get_min_capability() == 80
    assert cfg.get_supported_act_dtypes() == [torch.bfloat16]
    assert cfg.get_config_filenames() == ["tfq3_manifest.json"]

    dummy = torch.nn.Linear(256, 192, bias=False)
    method = cfg.get_quant_method(dummy, "model.layers.0.self_attn.q_proj")
    assert isinstance(method, Tfq3LinearMethod)
    method.create_weights(
        dummy,
        input_size_per_partition=256,
        output_partition_sizes=[192],
        input_size=256,
        output_size=192,
        params_dtype=torch.bfloat16,
    )
    assert dummy.tfq3_pad_r == 256 and dummy.tfq3_pad_c == 256 and dummy.tfq3_orig_r == 192
    assert dummy.tfq3_words.numel() == (256 // 16) * (256 // 128) * 8 * 32
    assert dummy.tfq3_scales.numel() == 4
    assert dummy.tfq3_tables.numel() == 16384
    print("PASS vLLM Tfq3Config & Tfq3LinearMethod interface contract")
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "tests", "vllm_plugin_test.py")], check=True)


def test_hf_end_to_end(model_dir: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cli = os.path.join(REPO_ROOT, "bin", "tfq3")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt = "The Fast Walsh-Hadamard Transform is an orthogonal linear transform that"
    inp = tok(prompt, return_tensors="pt")
    inp_dev = {k: v.to(device) for k, v in inp.items()}

    # Reference BF16 logits on the prompt
    ref_m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True).to(device)
    ref_m.eval()
    with torch.no_grad():
        ref_logits = ref_m(**inp_dev).logits[0, -1].float().cpu()
    del ref_m

    with tempfile.TemporaryDirectory(prefix="tfq_hf_test_") as tmp:
        # 1. TFQ3 4-bit end-to-end Hugging Face generation + logit cosine check vs BF16 gold
        out3 = os.path.join(tmp, "hf_tfq3")
        subprocess.run([cli, model_dir, "--bits", "4", "--head-bits", "0", "--out", out3], check=True, stdout=subprocess.DEVNULL)
        m3 = load_tfq_into_hf(out3, device=device, dtype=torch.bfloat16, cache_decoded=True)
        m3.eval()
        n_tfq3 = sum(1 for m in m3.modules() if isinstance(m, Tfq3Linear))
        assert n_tfq3 > 0, "Expected Tfq3Linear modules in HF model"
        with torch.no_grad():
            q3_logits = m3(**inp_dev).logits[0, -1].float().cpu()
            cos3 = float(torch.nn.functional.cosine_similarity(ref_logits.unsqueeze(0), q3_logits.unsqueeze(0)).item())
            assert cos3 > 0.99, f"TFQ3-4b HF logit cosine too low vs BF16 gold: {cos3:.6f}"
            g3 = m3.generate(**inp_dev, max_new_tokens=24, do_sample=False)
        txt3 = tok.decode(g3[0], skip_special_tokens=True)
        print(f"PASS HuggingFace + PyTorch TFQ3-4b ({n_tfq3} Tfq3Linear, logit cos={cos3:.6f} vs BF16):\n  {txt3!r}")
        del m3

        # 2. TFQ4-E8P 2-bit end-to-end Hugging Face load & generation
        out4 = os.path.join(tmp, "hf_tfq4")
        subprocess.run([cli, model_dir, "--bits", "2", "--head-bits", "0", "--out", out4], check=True, stdout=subprocess.DEVNULL)
        m4 = load_tfq_into_hf(out4, device=device, dtype=torch.bfloat16, cache_decoded=True)
        m4.eval()
        n_tfq4 = sum(1 for m in m4.modules() if isinstance(m, Tfq4Linear))
        assert n_tfq4 > 0, "Expected Tfq4Linear modules in HF model"
        with torch.no_grad():
            g4 = m4.generate(**inp_dev, max_new_tokens=20, do_sample=False)
        txt4 = tok.decode(g4[0], skip_special_tokens=True)
        print(f"PASS HuggingFace + PyTorch TFQ4-E8P ({n_tfq4} Tfq4Linear, zero TensorFold):\n  {txt4!r}")


if __name__ == "__main__":
    test_vllm_plugin_contract()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        test_hf_end_to_end(sys.argv[1])
    else:
        print("NOTE pass a local HF model directory as argv[1] to run full HF generate() E2E test")
