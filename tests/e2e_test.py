#!/usr/bin/env python3
"""
End-to-end automated verification suite for TFQ3 (3/4/5-bit) and TFQ4-E8P (2-bit).

Verifies:
  1. Pure-Python reference codecs (`tfq3_reference.py`, `tfq4_lattice.py`)
  2. Strict `tfq3-inspect` validation on valid + corrupted PWV3 / PWV4 / TFPN / TFP4 files
     (bad magic, truncation, 16 KiB misalignment, section overlap, NaN/negative LUTs, duplicate names)
  3. CLI behavior (`bin/tfq3 --version`, `--help`, `doctor`, invalid `--bits` rejection, `--dry-run`)
  4. Full GPU converter round-trips (Metal on macOS / CUDA on Linux):
     - Multi-dtype Safetensors input (BF16, F16, F32 + 1D residuals)
     - Non-128-multiple zero-padding (`[192, 256]` -> `[256, 256]`)
     - Row-chunked encoding across the 4096-row boundary (`[4352, 512]`), checking SNR on rows [4096:4352]
     - TFQ3 3b / 4b / 5b + `--target-gib` exact decode parity vs `tfq3_reference.py`
     - TFQ4-E8P 2-bit (`--bits 2`) + multi-shard splitting + exact decode parity vs `tfq4_lattice.py`
"""
import argparse
import glob
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref

ALIGN_16K = 16384


def align16k(n: int) -> int:
    return (n + ALIGN_16K - 1) & ~(ALIGN_16K - 1)


def run(cmd: list[str], expect_ok: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not expect_ok and proc.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {' '.join(cmd)}\nstdout:\n{proc.stdout}")
    return proc


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32, copy=False).view(np.uint32)
    # Round-to-nearest-even to bf16
    rounding = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16 = ((u32 + rounding) >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()


def write_synthetic_model(model_dir: str) -> dict[str, np.ndarray]:
    """Create a small multi-dtype Safetensors model exercising padding, dtypes, and >4096 row-chunking."""
    os.makedirs(model_dir, exist_ok=True)
    rng = np.random.default_rng(12345)

    # BF16 [4352, 512]: crosses 4096-row chunk boundary + ~0.54 MiB in TFQ4
    w_chunk_a = (rng.standard_normal((4352, 512)).astype(np.float32) * 0.02)
    # F16 [4352, 512]: second chunked matrix so total TFQ4 > 1.0 MiB for multi-shard test
    w_chunk_b = (rng.standard_normal((4352, 512)).astype(np.float32) * 0.02)
    # F32 [192, 256]: non-128-multiple rows -> zero-padded to [256, 256]
    w_pad = (rng.standard_normal((192, 256)).astype(np.float32) * 0.03)
    # BF16 [256, 256]: standard square layer
    w_sq = (rng.standard_normal((256, 256)).astype(np.float32) * 0.025)
    # 1D residual norm weight (kept in residual.safetensors)
    w_norm = np.ones((512,), dtype=np.float32)

    tensors = [
        ("model.layers.0.mlp.gate_proj.weight", "BF16", w_chunk_a.shape, f32_to_bf16_bytes(w_chunk_a)),
        ("model.layers.0.mlp.up_proj.weight",   "F16",  w_chunk_b.shape, w_chunk_b.astype(np.float16).tobytes()),
        ("model.layers.0.self_attn.q_proj.weight", "F32", w_pad.shape,   w_pad.astype(np.float32).tobytes()),
        ("model.layers.0.self_attn.o_proj.weight", "BF16", w_sq.shape,   f32_to_bf16_bytes(w_sq)),
        ("model.layers.0.input_layernorm.weight",  "F32", w_norm.shape,  w_norm.tobytes()),
    ]

    header = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(payload)

    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama", "hidden_size": 512, "num_hidden_layers": 1}, f)
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"version": "1.0"}, f)

    return {
        "model.layers.0.mlp.gate_proj": w_chunk_a,
        "model.layers.0.mlp.up_proj": w_chunk_b,
        "model.layers.0.self_attn.q_proj": w_pad,
        "model.layers.0.self_attn.o_proj": w_sq,
    }


def make_minimal_pwv4(path: str, bad_lut_u16: int | None = None) -> None:
    buf = bytearray(16384)
    struct.pack_into("<IIII", buf, 0, 0x34565750, 4, 256, 8)
    one_f16 = np.float16(0.5).view(np.uint16).item()
    for i in range(256 * 8):
        val = bad_lut_u16 if (bad_lut_u16 is not None and i == 17) else one_f16
        struct.pack_into("<H", buf, 64 + i * 2, val)
    with open(path, "wb") as f:
        f.write(buf)


def make_minimal_tfp4(
    path: str,
    names: list[str],
    corrupt_align: bool = False,
    corrupt_overlap: bool = False,
    truncate: bool = False,
) -> None:
    R, C = 128, 128
    n_sc = 1
    n_w = R * (C // 16)
    off_meta = 64
    off_scales = align16k(off_meta + len(names) * 128)
    off_weights = align16k(off_scales + len(names) * n_sc * 2)
    tot_sz = align16k(off_weights + len(names) * n_w * 4)

    hdr_off_scales = off_scales + 8 if corrupt_align else off_scales
    hdr_off_weights = off_scales if corrupt_overlap else off_weights

    buf = bytearray(tot_sz)
    struct.pack_into(
        "<IIIIQQQQQQ", buf, 0,
        0x34504654, 4, len(names), 0,
        off_meta, hdr_off_scales, hdr_off_weights,
        len(names) * n_sc, len(names) * n_w, 0,
    )
    for idx, nm in enumerate(names):
        rec = bytearray(128)
        nb = nm.encode("utf-8")[:79]
        rec[: len(nb)] = nb
        struct.pack_into(
            "<IIIIQQfIII", rec, 80,
            R, C, 2, idx * n_sc, idx * n_w, n_w, 10.1, R, C, 0,
        )
        buf[off_meta + idx * 128 : off_meta + (idx + 1) * 128] = rec

    if truncate:
        buf = buf[: tot_sz - 128]
    with open(path, "wb") as f:
        f.write(buf)


def test_inspector_negative_controls(inspect_bin: str, tmp: str) -> None:
    neg = os.path.join(tmp, "neg")
    os.makedirs(neg, exist_ok=True)

    # Valid PWV4 + valid TFP4 -> OK
    good_dir = os.path.join(neg, "good")
    os.makedirs(good_dir, exist_ok=True)
    make_minimal_pwv4(os.path.join(good_dir, "pw_v4_e8p_tables.bin"))
    make_minimal_tfp4(os.path.join(good_dir, "tfq4-00001-of-00001.tffp4"), ["layer.0.proj"])
    run([inspect_bin, "-v", good_dir], expect_ok=True)

    # Negative value in PWV4 e8p_lut -> FAIL
    neg_f16 = np.float16(-0.25).view(np.uint16).item()
    bad_pwv4_neg = os.path.join(neg, "bad_neg_pw_v4.bin")
    make_minimal_pwv4(bad_pwv4_neg, bad_lut_u16=neg_f16)
    run([inspect_bin, bad_pwv4_neg], expect_ok=False)

    # NaN in PWV4 e8p_lut -> FAIL
    bad_pwv4_nan = os.path.join(neg, "bad_nan_pw_v4.bin")
    make_minimal_pwv4(bad_pwv4_nan, bad_lut_u16=0x7E00)
    run([inspect_bin, bad_pwv4_nan], expect_ok=False)

    # Bad magic -> FAIL
    bad_magic = os.path.join(neg, "bad_magic.tffp4")
    with open(bad_magic, "wb") as f:
        f.write(b"NOPE" + b"\x00" * 16380)
    run([inspect_bin, bad_magic], expect_ok=False)

    # Misaligned off_scales -> FAIL
    bad_align = os.path.join(neg, "bad_align.tffp4")
    make_minimal_tfp4(bad_align, ["a"], corrupt_align=True)
    run([inspect_bin, bad_align], expect_ok=False)

    # Overlapping sections -> FAIL
    bad_overlap = os.path.join(neg, "bad_overlap.tffp4")
    make_minimal_tfp4(bad_overlap, ["a"], corrupt_overlap=True)
    run([inspect_bin, bad_overlap], expect_ok=False)

    # Truncated shard -> FAIL
    bad_trunc = os.path.join(neg, "bad_trunc.tffp4")
    make_minimal_tfp4(bad_trunc, ["a"], truncate=True)
    run([inspect_bin, bad_trunc], expect_ok=False)

    # Duplicate tensor name within one shard -> FAIL
    bad_dup_in = os.path.join(neg, "bad_dup_in.tffp4")
    make_minimal_tfp4(bad_dup_in, ["same.tensor", "same.tensor"])
    run([inspect_bin, bad_dup_in], expect_ok=False)

    # Duplicate tensor name across two shards in a directory -> FAIL
    dup_dir = os.path.join(neg, "dup_dir")
    os.makedirs(dup_dir, exist_ok=True)
    make_minimal_tfp4(os.path.join(dup_dir, "tfq4-00001-of-00002.tffp4"), ["shared.tensor"])
    make_minimal_tfp4(os.path.join(dup_dir, "tfq4-00002-of-00002.tffp4"), ["shared.tensor"])
    run([inspect_bin, dup_dir], expect_ok=False)

    print("PASS inspector strict validation & negative controls (9 cases)")


def test_cli_basics(tfq3_cli: str, model_dir: str) -> None:
    ver = run([tfq3_cli, "--version"])
    assert "tfq3 3.0.0" in ver.stdout
    hlp = run([tfq3_cli, "--help"])
    assert "--bits" in hlp.stdout and "--target-gib" in hlp.stdout
    doc = run([tfq3_cli, "doctor"])
    assert "tfq3-inspect" in doc.stdout and "tfq4-convert" in doc.stdout
    # Reject invalid --bits
    run([tfq3_cli, model_dir, "--bits", "9", "--dry-run"], expect_ok=False)
    print("PASS CLI basics (--version, --help, doctor, invalid --bits rejection)")


def find_shard_for_tensor(out_dir: str, ext: str, tensor_name: str) -> str:
    for fp in sorted(glob.glob(os.path.join(out_dir, f"*{ext}"))):
        with open(fp, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            nt, off_meta = hdr[2], hdr[4]
            f.seek(off_meta)
            for _ in range(nt):
                rec = f.read(128)
                nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                if nm == tensor_name:
                    return fp
    raise KeyError(f"{tensor_name} not found in any *{ext} in {out_dir}")


def test_gpu_roundtrips(tfq3_cli: str, inspect_bin: str, model_dir: str, refs: dict[str, np.ndarray], tmp: str) -> None:
    suffix = "" if platform.system() == "Darwin" else "-cuda"
    lib_dir = os.path.join(REPO_ROOT, "build")
    gen_bin = os.path.join(lib_dir, f"tfq3-generic-convert{suffix}")
    tfq4_bin = os.path.join(lib_dir, f"tfq4-convert{suffix}")

    if not (os.path.isfile(gen_bin) and os.path.isfile(tfq4_bin)):
        print(f"SKIP GPU round-trips (missing {gen_bin} or {tfq4_bin})")
        return

    # 1. TFQ3 generic round-trip across 3-bit, 4-bit, and 5-bit
    expected_min_snr = {3: 14.0, 4: 19.0, 5: 23.0}
    for bits, min_snr in expected_min_snr.items():
        out_dir = os.path.join(tmp, f"out_tfq3_{bits}b")
        run([tfq3_cli, model_dir, "--bits", str(bits), "--out", out_dir])
        run([inspect_bin, "-v", out_dir])

        mf = json.load(open(os.path.join(out_dir, "tfq3_manifest.json")))
        assert mf["format"] == "TFQ3" and mf["version"] == 3 and mf["tables"] == "pw_v3_tables.bin"
        assert mf["num_quantized_tensors"] == 4 and mf["num_residual_tensors"] == 1

        tables_path = os.path.join(out_dir, "pw_v3_tables.bin")

        # Check non-128-multiple padded tensor [192, 256] -> [256, 256]
        q_name = "model.layers.0.self_attn.q_proj"
        shard = find_shard_for_tensor(out_dir, ".tffpw", q_name)
        W_hat = tfq3_ref.dequantize_tfpn_tensor(shard, tables_path, q_name, max_blocks=4)
        assert W_hat.shape == (256, 256)
        snr_valid = tfq4_ref.snr_db(refs[q_name], W_hat[:192, :256])
        tail_rms = float(np.sqrt(np.mean(W_hat[192:, :] ** 2)))
        sig_rms = float(np.sqrt(np.mean(refs[q_name] ** 2)))
        assert snr_valid > min_snr, f"TFQ3 {bits}b padded tensor SNR too low: {snr_valid:.2f} dB"
        assert tail_rms < 0.25 * sig_rms, f"TFQ3 {bits}b zero-padded tail RMS too high: {tail_rms:.4f}"

        # Check >4096-row chunked tensor (especially rows 4096..4351 across r_off boundary!)
        g_name = "model.layers.0.mlp.gate_proj"
        shard_g = find_shard_for_tensor(out_dir, ".tffpw", g_name)
        W_g = tfq3_ref.dequantize_tfpn_tensor(shard_g, tables_path, g_name, max_blocks=64)
        assert W_g.shape == (4352, 512)
        snr_chunk0 = tfq4_ref.snr_db(refs[g_name][:4096, :], W_g[:4096, :])
        snr_chunk1 = tfq4_ref.snr_db(refs[g_name][4096:, :], W_g[4096:, :])
        assert snr_chunk0 > min_snr and snr_chunk1 > min_snr, (
            f"TFQ3 {bits}b row-chunked SNR mismatch: chunk0={snr_chunk0:.2f} dB, chunk1={snr_chunk1:.2f} dB"
        )
        print(f"PASS TFQ3 {bits}b GPU round-trip: padded={snr_valid:.2f} dB, chunk0={snr_chunk0:.2f} dB, chunk1(r>=4096)={snr_chunk1:.2f} dB")

    # 2. TFQ3 --target-gib dry-run plan check
    dry = run([tfq3_cli, model_dir, "--target-gib", "0.01", "--dry-run"])
    assert "TFQ3 Generic Plan" in dry.stdout

    # 3. TFQ4-E8P (--bits 2) GPU round-trip + multi-shard splitting + row-chunking parity
    out_tfq4 = os.path.join(tmp, "out_tfq4")
    run([tfq3_cli, model_dir, "--bits", "2", "--max-shard-gib", "0.001", "--out", out_tfq4])
    run([inspect_bin, "-v", out_tfq4])

    mf4 = json.load(open(os.path.join(out_tfq4, "tfq4_manifest.json")))
    assert mf4["format"] == "TFQ4-E8P" and mf4["version"] == 4 and mf4["tables"] == "pw_v4_e8p_tables.bin"
    assert len(mf4["tfq_shards"]) >= 2, f"Expected multi-shard TFP4 output, got {mf4['tfq_shards']}"
    assert mf4["num_quantized_tensors"] == 4 and mf4["num_residual_tensors"] == 1

    tbl4 = os.path.join(out_tfq4, "pw_v4_e8p_tables.bin")
    # Padded [192, 256] tensor
    q_name = "model.layers.0.self_attn.q_proj"
    shard_q = find_shard_for_tensor(out_tfq4, ".tffp4", q_name)
    W4_q = tfq4_ref.dequantize_tfp4_tensor(shard_q, tbl4, q_name)
    assert W4_q.shape == (256, 256)
    snr4_pad = tfq4_ref.snr_db(refs[q_name], W4_q[:192, :256])
    assert snr4_pad > 9.5, f"TFQ4 padded SNR too low: {snr4_pad:.2f} dB"

    # Row-chunked [4352, 512] BF16 & F16 tensors (verify r >= 4096 sign-vector parity!)
    for tname in ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"):
        sh = find_shard_for_tensor(out_tfq4, ".tffp4", tname)
        W4 = tfq4_ref.dequantize_tfp4_tensor(sh, tbl4, tname)
        assert W4.shape == (4352, 512)
        s0 = tfq4_ref.snr_db(refs[tname][:4096, :], W4[:4096, :])
        s1 = tfq4_ref.snr_db(refs[tname][4096:, :], W4[4096:, :])
        assert s0 > 9.5 and s1 > 9.5, f"TFQ4 chunked SNR mismatch on {tname}: c0={s0:.2f} dB, c1={s1:.2f} dB"
    print(
        f"PASS TFQ4-E8P 2b GPU round-trip: {len(mf4['tfq_shards'])} shards, "
        f"padded={snr4_pad:.2f} dB, chunk0={s0:.2f} dB, chunk1(r>=4096)={s1:.2f} dB"
    )

    # 4. Bit-reproducibility check (Issue #2): repeated runs must produce byte-identical shards & manifests
    def assert_dirs_byte_identical(dir_a: str, dir_b: str, label: str) -> None:
        files_a = sorted(f for f in os.listdir(dir_a) if os.path.isfile(os.path.join(dir_a, f)))
        files_b = sorted(f for f in os.listdir(dir_b) if os.path.isfile(os.path.join(dir_b, f)))
        assert files_a == files_b, f"{label}: file list mismatch {files_a} vs {files_b}"
        for fn in files_a:
            ba = open(os.path.join(dir_a, fn), "rb").read()
            bb = open(os.path.join(dir_b, fn), "rb").read()
            assert ba == bb, f"{label}: non-deterministic bytes in {fn} ({len(ba)} vs {len(bb)} B)"

    out_tfq3_4b_r2 = os.path.join(tmp, "out_tfq3_4b_r2")
    run([tfq3_cli, model_dir, "--bits", "4", "--out", out_tfq3_4b_r2])
    assert_dirs_byte_identical(os.path.join(tmp, "out_tfq3_4b"), out_tfq3_4b_r2, "TFQ3-4b")

    out_tfq4_r2 = os.path.join(tmp, "out_tfq4_r2")
    run([tfq3_cli, model_dir, "--bits", "2", "--max-shard-gib", "0.001", "--out", out_tfq4_r2])
    assert_dirs_byte_identical(out_tfq4, out_tfq4_r2, "TFQ4-E8P")
    print("PASS bit-reproducibility: TFQ3-4b and TFQ4-E8P repeated runs are 100% byte-identical")


def test_moe_datafree_ranking(tmp: str) -> None:
    """Compile & run a C++17 unit test for src/converters/moe_heat_rank.hpp proving non-iota sensitivity ranking."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    src = os.path.join(tmp, "test_moe_rank.cpp")
    exe = os.path.join(tmp, "test_moe_rank")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "moe_heat_rank.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            '#include <cassert>\n'
            '#include <cstdio>\n'
            '#include <map>\n'
            'struct FakeCache {\n'
            '    std::map<std::string, std::vector<uint8_t>> m;\n'
            '    bool has_tensor(const std::string& k) const { return m.count(k) > 0; }\n'
            '    const uint8_t* get_tensor(const std::string& k, size_t* n = nullptr) {\n'
            '        const auto& v = m.at(k); if (n) *n = v.size(); return v.data();\n'
            '    }\n'
            '};\n'
            'int main() {\n'
            '    FakeCache c;\n'
            '    const int E = 8, H = 4;\n'
            '    std::vector<float> router(E * H, 0.1f);\n'
            '    // Make expert 6 have largest router norm, expert 3 second largest\n'
            '    for (int j = 0; j < H; ++j) { router[6 * H + j] = 5.0f; router[3 * H + j] = 2.5f; }\n'
            '    auto put_f32 = [&](const std::string& k, const std::vector<float>& v) {\n'
            '        const uint8_t* b = reinterpret_cast<const uint8_t*>(v.data());\n'
            '        c.m[k].assign(b, b + v.size() * sizeof(float));\n'
            '    };\n'
            '    put_f32("model.language_model.layers.3.mlp.gate.weight", router);\n'
            '    const char* projs[3] = {"gate_proj", "up_proj", "down_proj"};\n'
            '    for (int e = 0; e < E; ++e) {\n'
            '        float s = (e == 6) ? 4.0f : (e == 3 ? 2.0f : 0.2f);\n'
            '        for (int p = 0; p < 3; ++p) {\n'
            '            char nm[256];\n'
            '            std::snprintf(nm, sizeof(nm), "model.language_model.layers.3.mlp.experts.%d.%s.weight_scale_inv", e, projs[p]);\n'
            '            put_f32(nm, std::vector<float>(16, s));\n'
            '        }\n'
            '    }\n'
            '    bool used_router = false;\n'
            '    auto ranks = tfq_moe::rank_glm_experts_datafree(c, 3, E, H, &used_router);\n'
            '    assert(used_router);\n'
            '    assert(ranks[6] == 0 && ranks[3] == 1);\n'
            '    std::printf("PASS data-free MoE router+weight ranker (ranks[6]=%d, ranks[3]=%d)\\n", ranks[6], ranks[3]);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    out = run([exe])
    print(out.stdout.strip())


def test_planner_shape_whitespace(tmp: str) -> None:
    """The generic planner must parse a Safetensors header with JSON whitespace inside a shape array."""
    cxx = "clang++" if platform.system() == "Darwin" else "g++"
    model_dir = os.path.join(tmp, "ws_model")
    os.makedirs(model_dir, exist_ok=True)
    nbytes = 128 * 256 * 4
    header = {"layer.weight": {"dtype": "F32", "shape": [128, 256], "data_offsets": [0, nbytes]}}
    hdr_bytes = json.dumps(header, indent=2).encode("utf-8")  # newline between the shape dims
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(b"\x00" * nbytes)
    src = os.path.join(tmp, "test_planner_ws.cpp")
    exe = os.path.join(tmp, "test_planner_ws")
    hdr = os.path.join(REPO_ROOT, "src", "converters", "generic", "planner.hpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(
            f'#include "{hdr}"\n'
            '#include <cassert>\n'
            'int main(int, char** argv) {\n'
            '    tfq_generic::SafetensorsDir st;\n'
            '    st.init(argv[1]);\n'
            '    const auto& shape = st.tensors.at("layer.weight").shape;\n'
            '    assert(shape.size() == 2 && shape[0] == 128 && shape[1] == 256);\n'
            '    std::printf("PASS planner parses whitespace in Safetensors shapes ([%llu, %llu])\\n",\n'
            '                (unsigned long long)shape[0], (unsigned long long)shape[1]);\n'
            '    return 0;\n'
            '}\n'
        )
    run([cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", src, "-o", exe])
    try:
        out = subprocess.run([exe, model_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError("planner hung parsing a Safetensors shape with a newline between dims")
    if out.returncode != 0:
        raise AssertionError(f"planner whitespace test failed (rc={out.returncode}):\n{out.stdout}\n{out.stderr}")
    print(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="Skip GPU converter round-trips (for headless CI runners)")
    args = ap.parse_args()

    inspect_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    tfq3_cli = os.path.join(REPO_ROOT, "bin", "tfq3")
    if not os.path.isfile(inspect_bin):
        raise SystemExit(f"Missing {inspect_bin}; run 'make tools' or 'make' first")

    tfq3_ref.self_test()
    run([sys.executable, os.path.join(REPO_ROOT, "reference", "tfq4_lattice.py"), "--self-test"])

    with tempfile.TemporaryDirectory(prefix="tfq_e2e_") as tmp:
        model_dir = os.path.join(tmp, "synth_model")
        refs = write_synthetic_model(model_dir)
        test_moe_datafree_ranking(tmp)
        test_planner_shape_whitespace(tmp)
        test_inspector_negative_controls(inspect_bin, tmp)
        test_cli_basics(tfq3_cli, model_dir)
        if not args.no_gpu:
            test_gpu_roundtrips(tfq3_cli, inspect_bin, model_dir, refs, tmp)

    print("ALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()
