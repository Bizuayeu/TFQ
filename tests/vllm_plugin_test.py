#!/usr/bin/env python3
"""
Verification suite for the vLLM TFQ3 quantization plugin (`integrations/vllm/tfq3_quant.py`):
  1. Automatic registration under `--quantization tfq3` (`QUANTIZATION_METHODS`)
  2. Per-tensor bit widths parsed from `tfq3_manifest.json` (no single root `bits`, mixed 3b/4b/5b)
  3. Real `.tffpw` (`TFPN`) + `pw_v3_tables.bin` weight loading via `Tfq3CheckpointReader` &
     `load_tfq3_checkpoint_weights`
  4. Merged projections (`gate_up_proj` 2-way, `qkv_proj` 3-way) with non-128-multiple partition
     sizes, mixed per-partition bit widths, independent 128-row padding, and independent row signs
  5. Tensor Parallelism (`TP > 1`): Column-Parallel (`row_off`) and Row-Parallel (`k_sign_off` +
     1-block trellis state resume across `TP=2` and `TP=4`) numerical parity vs `tfq3_reference`
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import types

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))
sys.path.insert(0, os.path.join(REPO_ROOT, "integrations", "vllm"))

import tfq3_reference as tfq3_ref
import tfq3_quant
from tfq3_quant import (
    Tfq3CheckpointReader,
    Tfq3Config,
    Tfq3LinearMethod,
    load_tfq3_checkpoint_weights,
    register_tfq3_quantization,
)

ALIGN_16K = 16384


def _align16k(n: int) -> int:
    return (n + ALIGN_16K - 1) & ~(ALIGN_16K - 1)


def _align128(n: int) -> int:
    return ((n + 127) // 128) * 128


def write_synthetic_tfq3_checkpoint(ckpt_dir: str) -> tuple[dict[str, np.ndarray], dict]:
    """Write a valid mixed-bit (3b/4b/5b) TFPN checkpoint + canonical PWV3 tables + manifest."""
    os.makedirs(ckpt_dir, exist_ok=True)
    canon_tbl = os.path.join(REPO_ROOT, "reference", "tables", "pw_v3_tables.bin")
    tbl_bytes = open(canon_tbl, "rb").read()
    with open(os.path.join(ckpt_dir, "pw_v3_tables.bin"), "wb") as f:
        f.write(tbl_bytes)
    tbl = tfq3_ref.load_tables(canon_tbl)

    # Specs: (name, orig_R, orig_C, bits) — intentionally mixes non-128-multiples & 3/4/5b!
    specs = [
        ("model.layers.0.self_attn.q_proj", 192, 256, 3),
        ("model.layers.0.self_attn.k_proj", 160, 256, 4),
        ("model.layers.0.self_attn.v_proj", 160, 256, 5),
        ("model.layers.0.self_attn.o_proj", 256, 512, 4),
        ("model.layers.0.mlp.gate_proj", 192, 256, 3),
        ("model.layers.0.mlp.up_proj", 160, 256, 5),
        ("model.layers.0.mlp.gate_tp_proj", 256, 512, 4),
        ("model.layers.0.mlp.up_tp_proj", 256, 512, 5),
        ("model.layers.0.mlp.down_proj", 256, 512, 3),
    ]

    rng = np.random.default_rng(2026)
    metas = []
    all_scales = []
    all_words = []
    ref_weights: dict[str, np.ndarray] = {}
    manifest_tensors = []
    sc_off = 0
    w_off = 0

    for name, orig_r, orig_c, bits in specs:
        pad_r, pad_c = _align128(orig_r), _align128(orig_c)
        wpb = bits * 2
        nw = (pad_r // 16) * (pad_c // 128) * wpb * 32
        ns = (pad_r // 128) * (pad_c // 128)
        words = rng.integers(0, 1 << 32, size=(nw,), dtype=np.uint32)
        scales = (0.03 + 0.02 * rng.random((pad_r // 128, pad_c // 128))).astype(np.float16)

        W_full = tfq3_ref.decode_tfpn_fast_np(words, scales, pad_r, pad_c, bits, tbl)
        ref_weights[name] = W_full[:orig_r, :orig_c].copy()

        metas.append((name, pad_r, pad_c, bits, sc_off, w_off, nw, 22.0, orig_r, orig_c, 1 if (pad_r != orig_r or pad_c != orig_c) else 0))
        all_scales.append(scales.reshape(-1))
        all_words.append(words)
        manifest_tensors.append({
            "name": name,
            "role": "attn" if "self_attn" in name else "mlp",
            "orig_shape": [orig_r, orig_c],
            "pad_shape": [pad_r, pad_c],
            "bits": bits,
            "snr_db": 22.0,
        })
        sc_off += ns
        w_off += nw

    scales_cat = np.concatenate(all_scales).astype("<f2")
    words_cat = np.concatenate(all_words).astype("<u4")

    off_meta = 64
    off_scales = _align16k(off_meta + len(metas) * 128)
    off_weights = _align16k(off_scales + scales_cat.nbytes)
    end_sz = _align16k(off_weights + words_cat.nbytes)

    buf = bytearray(end_sz)
    struct.pack_into(
        "<IIIIQQQQQQ", buf, 0,
        0x4E504654, 3, len(metas), 0,
        off_meta, off_scales, off_weights,
        len(scales_cat), len(words_cat), 0,
    )
    for i, (nm, R, C, b, so, wo, wc, snr, oR, oC, flg) in enumerate(metas):
        rec = bytearray(128)
        nb = nm.encode("utf-8")[:79]
        rec[: len(nb)] = nb
        struct.pack_into("<IIIIQQfIII", rec, 80, R, C, b, so, wo, wc, snr, oR, oC, flg)
        buf[off_meta + i * 128 : off_meta + (i + 1) * 128] = rec

    buf[off_scales : off_scales + scales_cat.nbytes] = scales_cat.tobytes()
    buf[off_weights : off_weights + words_cat.nbytes] = words_cat.tobytes()

    shard_name = "tfq3-00001-of-00001.tffpw"
    with open(os.path.join(ckpt_dir, shard_name), "wb") as f:
        f.write(buf)

    # Note: deliberately NO root "bits" key — matches generic converter planner.hpp output!
    manifest = {
        "format": "TFQ3",
        "version": 3,
        "tables": "pw_v3_tables.bin",
        "tfq_shards": [shard_name],
        "residual_shards": [],
        "tfq_bytes": end_sz,
        "residual_bytes": 0,
        "num_quantized_tensors": len(metas),
        "num_residual_tensors": 0,
        "tensors": manifest_tensors,
    }
    with open(os.path.join(ckpt_dir, "tfq3_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return ref_weights, manifest


def test_registration_hook() -> None:
    if tfq3_quant.HAS_VLLM:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS  # type: ignore
        assert "tfq3" in QUANTIZATION_METHODS, "tfq3 not registered in vLLM QUANTIZATION_METHODS"
        print("PASS vLLM native registration ('tfq3' in QUANTIZATION_METHODS)")
        return

    # Test registration path using a lightweight mock vLLM quantization registry module
    registry: dict[str, type] = {}

    def register_quantization_config(name: str):
        def _dec(cls):
            registry[name] = cls
            return cls
        return _dec

    q_mod = types.ModuleType("vllm.model_executor.layers.quantization")
    q_mod.QUANTIZATION_METHODS = registry
    q_mod.register_quantization_config = register_quantization_config

    saved = {}
    for mod_name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
    ):
        saved[mod_name] = sys.modules.get(mod_name)
    try:
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.model_executor"] = types.ModuleType("vllm.model_executor")
        sys.modules["vllm.model_executor.layers"] = types.ModuleType("vllm.model_executor.layers")
        sys.modules["vllm.model_executor.layers.quantization"] = q_mod
        prev_has = tfq3_quant.HAS_VLLM
        tfq3_quant.HAS_VLLM = True
        try:
            assert register_tfq3_quantization() is True
            assert registry.get("tfq3") is Tfq3Config
        finally:
            tfq3_quant.HAS_VLLM = prev_has
    finally:
        for mod_name, prev in saved.items():
            if prev is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = prev
    print("PASS vLLM quantization registration hook ('tfq3' -> Tfq3Config)")


def test_manifest_and_execution() -> None:
    with tempfile.TemporaryDirectory(prefix="tfq3_vllm_test_") as tmp:
        refs, manifest = write_synthetic_tfq3_checkpoint(tmp)

        # 1. Exact Issue #9 repro: no root "bits" key; uniform-3b manifest -> cfg.bits == 3
        m_all3 = json.loads(json.dumps(manifest))
        assert "bits" not in m_all3
        for t in m_all3["tensors"]:
            t["bits"] = 3
        cfg3 = Tfq3Config.from_config(m_all3)
        assert cfg3.bits == 3, f"Expected cfg3.bits == 3, got {cfg3.bits}"

        # 2. Mixed-bit manifest parsing + residual skip (lm_head not in manifest -> None)
        cfg = Tfq3Config.from_config(manifest)
        assert cfg.get_tensor_bits("model.layers.0.self_attn.q_proj") == 3
        assert cfg.get_tensor_bits("model.layers.0.self_attn.k_proj") == 4
        assert cfg.get_tensor_bits("model.layers.0.self_attn.v_proj") == 5
        assert cfg.get_tensor_bits("model.layers.0.mlp.up_proj") == 5
        assert cfg.get_quant_method(torch.nn.Linear(256, 256), "lm_head") is None
        print("PASS manifest per-tensor bits (no root 'bits', uniform-3b + mixed 3b/4b/5b + residual skip)")

        reader = Tfq3CheckpointReader(tmp)
        torch.manual_seed(7)

        # 3. Standalone layer + load_tfq3_checkpoint_weights()
        class MiniModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.o_proj = torch.nn.Linear(512, 256, bias=False)
                self.o_proj.prefix = "model.layers.0.self_attn.o_proj"
                self.o_proj.quant_method = cfg.get_quant_method(self.o_proj, self.o_proj.prefix)
                self.o_proj.quant_method.create_weights(self.o_proj, 512, [256], 512, 256, torch.bfloat16)

        model = MiniModel()
        assert load_tfq3_checkpoint_weights(model, tmp) == 1
        x512 = torch.randn(5, 512, dtype=torch.float32)
        y_o = model.o_proj.quant_method.apply(model.o_proj, x512)
        y_o_ref = x512 @ torch.from_numpy(refs["model.layers.0.self_attn.o_proj"]).T
        assert torch.allclose(y_o, y_o_ref, atol=1e-4, rtol=1e-4), f"o_proj max diff={(y_o - y_o_ref).abs().max()}"
        print("PASS standalone layer load_tfq3_checkpoint_weights + exact apply() parity")

        # 4. Merged 2-way gate_up_proj (non-128-multiple 192+160, mixed 3b + 5b)
        x256 = torch.randn(4, 256, dtype=torch.float32)
        gu_layer = torch.nn.Linear(256, 192 + 160, bias=False)
        gu_method = cfg.get_quant_method(gu_layer, "model.layers.0.mlp.gate_up_proj")
        assert isinstance(gu_method, Tfq3LinearMethod)
        gu_method.create_weights(gu_layer, 256, [192, 160], 256, 192 + 160, torch.bfloat16)
        assert gu_layer.tfq3_partition_bits == [3, 5]
        assert gu_layer.tfq3_partition_pad_r == [256, 256]
        gu_method.load_layer_from_reader(gu_layer, reader)
        y_gu = gu_method.apply(gu_layer, x256)
        y_gu_ref = torch.cat([
            x256 @ torch.from_numpy(refs["model.layers.0.mlp.gate_proj"]).T,
            x256 @ torch.from_numpy(refs["model.layers.0.mlp.up_proj"]).T,
        ], dim=-1)
        assert torch.allclose(y_gu, y_gu_ref, atol=1e-4, rtol=1e-4), f"gate_up max diff={(y_gu - y_gu_ref).abs().max()}"
        print("PASS 2-way merged gate_up_proj (non-128-multiple [192,160], mixed 3b+5b, independent padding & row signs)")

        # 5. Merged 3-way qkv_proj (non-128-multiple 192+160+160, mixed 3b+4b+5b)
        qkv_layer = torch.nn.Linear(256, 192 + 160 + 160, bias=False)
        qkv_method = cfg.get_quant_method(qkv_layer, "model.layers.0.self_attn.qkv_proj")
        assert isinstance(qkv_method, Tfq3LinearMethod)
        qkv_method.create_weights(qkv_layer, 256, [192, 160, 160], 256, 192 + 160 + 160, torch.bfloat16)
        assert qkv_layer.tfq3_partition_bits == [3, 4, 5]
        assert qkv_layer.tfq3_partition_pad_r == [256, 256, 256]
        qkv_method.load_layer_from_reader(qkv_layer, reader)
        y_qkv = qkv_method.apply(qkv_layer, x256)
        y_qkv_ref = torch.cat([
            x256 @ torch.from_numpy(refs["model.layers.0.self_attn.q_proj"]).T,
            x256 @ torch.from_numpy(refs["model.layers.0.self_attn.k_proj"]).T,
            x256 @ torch.from_numpy(refs["model.layers.0.self_attn.v_proj"]).T,
        ], dim=-1)
        assert torch.allclose(y_qkv, y_qkv_ref, atol=1e-4, rtol=1e-4), f"qkv max diff={(y_qkv - y_qkv_ref).abs().max()}"
        print("PASS 3-way merged qkv_proj (non-128-multiple [192,160,160], mixed 3b+4b+5b, independent padding & row signs)")

        # 6. Column-Parallel TP=2 on merged 2-way (gate_tp_proj 4b + up_tp_proj 5b, [256,512] each -> [128,512]/rank)
        col_outs = []
        for rank in range(2):
            cl = torch.nn.Linear(512, 256, bias=False)
            cm = Tfq3LinearMethod(cfg, prefix="model.layers.0.mlp.gate_up_tp")
            cm.create_weights(
                cl, 512, [128, 128], 512, 512, torch.bfloat16,
                tp_rank=rank, tp_size=2,
                partition_names=["model.layers.0.mlp.gate_tp_proj", "model.layers.0.mlp.up_tp_proj"],
            )
            assert cl.tfq3_partition_row_offs == [rank * 128, rank * 128]
            cm.load_layer_from_reader(cl, reader)
            col_outs.append(cm.apply(cl, x512))
        y_gate_cat = torch.cat([col_outs[0][:, :128], col_outs[1][:, :128]], dim=-1)
        y_up_cat = torch.cat([col_outs[0][:, 128:], col_outs[1][:, 128:]], dim=-1)
        assert torch.allclose(y_gate_cat, x512 @ torch.from_numpy(refs["model.layers.0.mlp.gate_tp_proj"]).T, atol=1e-4, rtol=1e-4)
        assert torch.allclose(y_up_cat, x512 @ torch.from_numpy(refs["model.layers.0.mlp.up_tp_proj"]).T, atol=1e-4, rtol=1e-4)
        print("PASS Column-Parallel TP=2 (merged mixed-bit 4b+5b with per-rank row_off signs)")

        # 7. Row-Parallel TP=2 and TP=4 on down_proj ([256, 512], 3b) with trellis state resume + k_sign_off
        y_down_ref = x512 @ torch.from_numpy(refs["model.layers.0.mlp.down_proj"]).T
        for tp in (2, 4):
            k_loc = 512 // tp
            y_sum = torch.zeros_like(y_down_ref)
            for rank in range(tp):
                rl = torch.nn.Linear(k_loc, 256, bias=False)
                rm = cfg.get_quant_method(rl, "model.layers.0.mlp.down_proj")
                assert isinstance(rm, Tfq3LinearMethod)
                rm.create_weights(rl, k_loc, [256], 512, 256, torch.bfloat16, tp_rank=rank, tp_size=tp)
                assert rl.tfq3_k_sign_off == rank * k_loc
                assert rl.tfq3_bc0_buf == (0 if rank == 0 else 1)
                rm.load_layer_from_reader(rl, reader)
                y_sum = y_sum + rm.apply(rl, x512[:, rank * k_loc : (rank + 1) * k_loc])
            assert torch.allclose(y_sum, y_down_ref, atol=1e-4, rtol=1e-4), (
                f"Row-TP={tp} max diff={(y_sum - y_down_ref).abs().max()}"
            )
        print("PASS Row-Parallel TP=2 & TP=4 (compact 1-block state resume + k_sign_off AllReduce parity)")


def main() -> None:
    test_registration_hook()
    test_manifest_and_execution()
    print("ALL VLLM PLUGIN TESTS PASSED")


if __name__ == "__main__":
    main()
