#!/usr/bin/env python3
"""
CPU checks for `integrations/pytorch/tfq_torch.py` on tiny synthetic checkpoints (no GPU, no download,
no `datasets`): every model tensor must come from the checkpoint, or `load_tfq_into_hf` must raise.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "integrations", "pytorch"))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))

import tfq3_reference as tfq3_ref
from tfq_torch import Tfq3Linear, load_tfq_into_hf

MLP = ("gate_proj", "up_proj", "down_proj")


def write_pwv3(path: str, rng: np.random.Generator) -> None:
    raw = bytearray(16384)
    struct.pack_into("<II6f", raw, 0, tfq3_ref.MAGIC_PWV3, tfq3_ref.FORMAT_VER, 1, 0, 1, 0, 1, 0)
    for off, n in ((64, 256 * 2), (1088, 1024 * 2), (5184, 1024 * 2)):
        raw[off:off + n * 2] = rng.standard_normal(n).astype(np.float16).tobytes()
    for off in (9280, 9792, 10304):
        raw[off:off + 64 * 4 * 2] = np.tile(np.array([1, 1, 0, 0], dtype=np.float16), 64).tobytes()
    open(path, "wb").write(bytes(raw))


def write_shard(path: str, magic: int, recs: list[tuple[str, int, int, int]], rng: np.random.Generator) -> None:
    """recs: (name, R, C, bits); random words and scales, TFPN v3 layout (TFPW only differs in magic here)."""
    metas, scales, words = [], [], []
    n_sc = n_wd = 0
    for name, R, C, bits in recs:
        sc = (0.01 + 0.01 * rng.random((R // 128) * (C // 128))).astype(np.float16)
        wd = rng.integers(0, 2**32, (R // 16) * (C // 128) * bits * 2 * 32, dtype=np.uint64).astype(np.uint32)
        metas.append(name.encode().ljust(80, b"\0") + struct.pack("<IIIIQQfIII", R, C, bits, n_sc, n_wd, wd.size, 20.0, R, C, 0))
        scales.append(sc); words.append(wd)
        n_sc += sc.size; n_wd += wd.size
    off_meta = 64
    off_sc = -(-(off_meta + 128 * len(metas)) // 16384) * 16384
    off_wd = -(-(off_sc + 2 * n_sc) // 16384) * 16384
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIQQQQQQ", magic, tfq3_ref.FORMAT_VER, len(metas), 0, off_meta, off_sc, off_wd, n_sc, n_wd, 0))
        f.write(b"".join(metas))
        f.seek(off_sc); f.write(np.concatenate(scales).tobytes())
        f.seek(off_wd); f.write(np.concatenate(words).tobytes())


def write_residuals(path: str, tensors: dict[str, torch.Tensor]) -> None:
    hdr, blobs, off = {}, [], 0
    for k, t in tensors.items():
        b = t.detach().to(torch.float32).contiguous().numpy().tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); f.write(b"".join(blobs))


def make_ckpt(d: str, case: str) -> dict[str, torch.Tensor]:
    """A 1-layer Llama, MLP in .tffpw. Returns the reference state dict."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False)
    cfg.save_pretrained(d)
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).state_dict()
    rng = np.random.default_rng(0)
    write_pwv3(os.path.join(d, "pw_v3_tables.bin"), rng)
    mlp = [(f"model.layers.0.mlp.{p}", 128 if p == "down_proj" else 256, 256 if p == "down_proj" else 128, 4) for p in MLP]
    packed = {n for n, *_ in mlp}
    TFPN, TFPW = 0x4E504654, 0x57504654
    if case == "ok":
        write_shard(os.path.join(d, "nonmoe.tffpw"), TFPN, mlp, rng)
    elif case == "embed":  # --quantize-embed: the embedding table is a packed record
        write_shard(os.path.join(d, "nonmoe.tffpw"), TFPN, mlp + [("model.embed_tokens", 256, 128, 4)], rng)
        packed.add("model.embed_tokens")
    elif case == "tfpw":  # specialized MoE converters write expert weights as TFPW shards
        write_shard(os.path.join(d, "layer_00.tffpw"), TFPW, mlp, rng)
    elif case == "hybrid":  # TFQ3 + TFQ4 in one directory: down_proj only in the .tffp4
        write_shard(os.path.join(d, "nonmoe.tffpw"), TFPN, mlp[:2], rng)
        open(os.path.join(d, "tfq4-00001-of-00001.tffp4"), "wb").write(b"\0" * 64)
    write_residuals(os.path.join(d, "model.safetensors"),
                    {k: v for k, v in ref.items() if k.rsplit(".", 1)[0] not in packed})
    return ref


def expect(case: str, missing: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix=f"tfq_torch_{case}_") as d:
        make_ckpt(d, case)
        try:
            load_tfq_into_hf(d, device="cpu", dtype=torch.float32)
        except RuntimeError as e:
            for k in missing:
                assert f"  {k}\n" in str(e) + "\n", f"{case}: {k} not named in: {e}"
            print(f"PASS {case}: load_tfq_into_hf raised and named {', '.join(missing)}")
            return
        raise AssertionError(f"{case}: load_tfq_into_hf returned a model with {', '.join(missing)} at random init")


def test_complete_checkpoint_loads() -> None:
    with tempfile.TemporaryDirectory(prefix="tfq_torch_ok_") as d:
        ref = make_ckpt(d, "ok")
        m = load_tfq_into_hf(d, device="cpu", dtype=torch.float32)
        sd = m.state_dict()
        assert torch.equal(sd["model.embed_tokens.weight"], ref["model.embed_tokens.weight"])
        assert sum(isinstance(x, Tfq3Linear) for x in m.modules()) == 3
        with torch.no_grad():
            y = m(torch.arange(8)[None]).logits
        assert torch.isfinite(y).all()
        print("PASS ok: complete checkpoint loads, residuals exact, 3 Tfq3Linear, CPU forward finite")


def test_cpu_decode_matches_reference() -> None:
    """The CPU/MPS fallback decoder needs only numpy (no datasets/transformers) and matches the slow reference."""
    with tempfile.TemporaryDirectory(prefix="tfq_torch_dec_") as d:
        rng = np.random.default_rng(1)
        tables = os.path.join(d, "pw_v3_tables.bin")
        write_pwv3(tables, rng)
        for bits in (3, 4, 5):
            sp = os.path.join(d, f"t{bits}.tffpw")
            write_shard(sp, 0x4E504654, [("w", 256, 256, bits)], rng)
            ref_w = tfq3_ref.dequantize_tfpn_tensor(sp, tables, "w", max_blocks=2)
            with open(sp, "rb") as f:
                hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
                f.seek(hdr[5]); sc = np.frombuffer(f.read(4 * 2), dtype=np.float16).reshape(2, 2)
                f.seek(hdr[6]); wd = np.frombuffer(f.read(hdr[8] * 4), dtype=np.uint32)
            lin = Tfq3Linear(256, 256, bits, wd, sc, open(tables, "rb").read(), tfq3_ref.load_tables(tables))
            w = lin.dequantize(dtype=torch.float32).numpy()
            err = float(np.max(np.abs(w - ref_w)))
            assert err < 1e-4, f"{bits}b CPU decode vs reference: {err}"
            x = torch.randn(3, 256)
            assert torch.allclose(lin(x), x @ torch.from_numpy(ref_w).T, atol=1e-3)
    print("PASS cpu: Tfq3Linear 3/4/5b CPU decode == tfq3_reference (max err < 1e-4)")


if __name__ == "__main__":
    print("datasets importable:", __import__("importlib").util.find_spec("datasets") is not None)
    test_cpu_decode_matches_reference()
    test_complete_checkpoint_loads()
    expect("embed", ["model.embed_tokens.weight"])
    expect("tfpw", [f"model.layers.0.mlp.{p}.weight" for p in MLP])
    expect("hybrid", ["model.layers.0.mlp.down_proj.weight"])
    print("ALL PASS")
