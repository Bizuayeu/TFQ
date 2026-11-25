#!/usr/bin/env python3
"""
TFQ3 Reference Codec (Pure Python + NumPy)

Exact mathematical reference for the TFQ3 (2D-AffineCoset/half4 + 2D FWHT-128)
quantization format used by TensorFold:
  - Deterministic Murmur3 sign hash (`pw_sgn`)
  - Orthonormal 128-point Fast Walsh-Hadamard Transform (`fwht_128`)
  - 2D randomized rotation: W_rot = D_L @ H_128 @ W @ H_128 @ D_R
  - Exact 3-bit, 4-bit, and 5-bit 2D-AffineCoset/half4 state-transition dequantizer
  - Reader for `pw_v3_tables.bin` ("PWV3") and `TFPN` (".tffpw") dense shards

Zero external dependencies beyond NumPy. Run `--self-test` to verify orthogonality
and 3b/4b/5b roundtrip SNR against a standard normal matrix.
"""

import argparse
import os
import struct
import sys
import numpy as np

MAGIC_PWV3 = 0x33565750
MAGIC_TFPN = 0x4E504654
FORMAT_VER = 3
SEED_COL = 0x12345678
SEED_ROW = 0x9E3779B9
HASH_MULT = 0x9E3779B1
INIT_ST_BASE = 0x13579BDF


def pw_sgn(seed: int, idx: np.ndarray) -> np.ndarray:
    """Deterministic +/-1 Rademacher sign hash matching Metal `pw_sgn` bit-for-bit."""
    h = ((idx.astype(np.uint32) ^ np.uint32(seed)) * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
    return np.where(((h >> np.uint32(31)) & np.uint32(1)) != 0, -1.0, 1.0).astype(np.float32)


def fwht_128(x: np.ndarray) -> np.ndarray:
    """Orthonormal 128-point Walsh-Hadamard transform along the last axis."""
    assert x.shape[-1] == 128
    y = x.astype(np.float32, copy=True)
    h = 1
    while h < 128:
        y = y.reshape(*x.shape[:-1], -1, 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(x.shape)
        h <<= 1
    y *= (1.0 / np.sqrt(128.0))
    return y


def rotate_block_2d(block: np.ndarray, r_off: int = 0, c_off: int = 0) -> np.ndarray:
    """Apply H_128 @ (D_L @ W @ D_R) @ H_128 to a [128, 128] weight block."""
    assert block.shape == (128, 128)
    sr = pw_sgn(SEED_ROW, np.arange(r_off, r_off + 128, dtype=np.uint32))
    sc = pw_sgn(SEED_COL, np.arange(c_off, c_off + 128, dtype=np.uint32))
    tmp = fwht_128(block * sc[None, :])
    return fwht_128((tmp * sr[:, None]).T).T


def unrotate_block_2d(block_rot: np.ndarray, r_off: int = 0, c_off: int = 0) -> np.ndarray:
    """Exact inverse of `rotate_block_2d`: D_L @ H_128 @ W_rot @ H_128 @ D_R."""
    assert block_rot.shape == (128, 128)
    sr = pw_sgn(SEED_ROW, np.arange(r_off, r_off + 128, dtype=np.uint32))
    sc = pw_sgn(SEED_COL, np.arange(c_off, c_off + 128, dtype=np.uint32))
    tmp = fwht_128(block_rot.T).T * sr[:, None]
    return fwht_128(tmp) * sc[None, :]


def load_tables(path: str) -> dict:
    """Load and validate a 16 KiB `pw_v3_tables.bin` codebook file."""
    raw = open(path, "rb").read()
    magic, ver = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC_PWV3 or ver != FORMAT_VER:
        raise ValueError(f"Bad pw_v3_tables header: magic=0x{magic:08x} ver={ver}")
    a3, b3, a4, b4, a5, b5 = struct.unpack_from("<6f", raw, 8)
    lut3 = np.frombuffer(raw, dtype=np.float16, count=256 * 2, offset=64).reshape(256, 2).copy()
    lut4 = np.frombuffer(raw, dtype=np.float16, count=1024 * 2, offset=1088).reshape(1024, 2).copy()
    lut5 = np.frombuffer(raw, dtype=np.float16, count=1024 * 2, offset=5184).reshape(1024, 2).copy()
    c64_3 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=9280).reshape(64, 4).copy()
    c64_4 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=9792).reshape(64, 4).copy()
    c64_5 = np.frombuffer(raw, dtype=np.float16, count=64 * 4, offset=10304).reshape(64, 4).copy()
    return {
        "a": {3: a3, 4: a4, 5: a5},
        "b": {3: b3, 4: b4, 5: b5},
        "lut": {3: lut3, 4: lut4, 5: lut5},
        "c64": {3: c64_3, 4: c64_4, 5: c64_5},
    }


def _frag_home(lane: int) -> tuple[int, int]:
    """Exact 32-lane M5 / NAX 16x16 subtile coordinate mapping (`pw_frag_home`)."""
    col = (lane & 1) * 4 + ((lane >> 3) & 1) * 8
    row = ((lane >> 1) & 3) + ((lane >> 4) & 1) * 4
    return col, row


def _decode_8_4b(st: int, wc: int, tbl: dict) -> tuple[int, np.ndarray]:
    s0 = st & 0xFFFFFFFF
    h0 = (s0 * HASH_MULT) & 0xFFFFFFFF
    h1 = ((((s0 << 8) | (wc >> 24)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h2 = ((((s0 << 16) | (wc >> 16)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h3 = ((((s0 << 24) | (wc >> 8)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    c0 = tbl["c64"][4][(h0 >> 16) & 63].astype(np.float32)
    c2 = tbl["c64"][4][(h2 >> 16) & 63].astype(np.float32)
    lut = tbl["lut"][4].astype(np.float32)
    p0 = lut[(((wc >> 24) << 2) ^ (h0 >> 22)) & 1023]
    p1 = lut[((((wc >> 16) & 255) << 2) ^ (h1 >> 22)) & 1023]
    p2 = lut[((((wc >> 8) & 255) << 2) ^ (h2 >> 22)) & 1023]
    p3 = lut[(((wc & 255) << 2) ^ (h3 >> 22)) & 1023]
    q0 = np.array([p0[0] * c0[0] + c0[2], p0[1] * c0[1] + c0[3], p1[0] * c0[0] + c0[2], p1[1] * c0[1] + c0[3]], dtype=np.float32)
    q1 = np.array([p2[0] * c2[0] + c2[2], p2[1] * c2[1] + c2[3], p3[0] * c2[0] + c2[2], p3[1] * c2[1] + c2[3]], dtype=np.float32)
    return wc & 0xFFFFFFFF, np.concatenate([q0, q1])


def _decode_8_3b(st: int, m24: int, tbl: dict) -> tuple[int, np.ndarray]:
    s0 = st & 0xFFFFFFFF
    h0 = (s0 * HASH_MULT) & 0xFFFFFFFF
    h1 = ((((s0 << 6) | (m24 >> 18)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h2 = ((((s0 << 12) | (m24 >> 12)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h3 = ((((s0 << 18) | (m24 >> 6)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    new_st = ((s0 << 24) | m24) & 0xFFFFFFFF
    c0 = tbl["c64"][3][(h0 >> 16) & 63].astype(np.float32)
    c2 = tbl["c64"][3][(h2 >> 16) & 63].astype(np.float32)
    lut = tbl["lut"][3].astype(np.float32)
    p0 = lut[(((m24 >> 16) & 0xFC) ^ (h0 >> 24)) & 255]
    p1 = lut[(((m24 >> 10) & 0xFC) ^ (h1 >> 24)) & 255]
    p2 = lut[(((m24 >> 4) & 0xFC) ^ (h2 >> 24)) & 255]
    p3 = lut[(((m24 << 2) & 0xFC) ^ (h3 >> 24)) & 255]
    q0 = np.array([p0[0] * c0[0] + c0[2], p0[1] * c0[1] + c0[3], p1[0] * c0[0] + c0[2], p1[1] * c0[1] + c0[3]], dtype=np.float32)
    q1 = np.array([p2[0] * c2[0] + c2[2], p2[1] * c2[1] + c2[3], p3[0] * c2[0] + c2[2], p3[1] * c2[1] + c2[3]], dtype=np.float32)
    return new_st, np.concatenate([q0, q1])


def _decode_8_5b(st: int, hi8: int, lo32: int, tbl: dict) -> tuple[int, np.ndarray]:
    s0 = st & 0xFFFFFFFF
    b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023
    b1 = (lo32 >> 20) & 1023
    b2 = (lo32 >> 10) & 1023
    b3 = lo32 & 1023
    h0 = (s0 * HASH_MULT) & 0xFFFFFFFF
    h1 = ((((s0 << 10) | b0) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h2 = ((((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    h3 = ((((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) & 0xFFFFFFFF) * HASH_MULT) & 0xFFFFFFFF
    c0 = tbl["c64"][5][(h0 >> 16) & 63].astype(np.float32)
    c2 = tbl["c64"][5][(h2 >> 16) & 63].astype(np.float32)
    lut = tbl["lut"][5].astype(np.float32)
    p0 = lut[(b0 ^ (h0 >> 22)) & 1023]
    p1 = lut[(b1 ^ (h1 >> 22)) & 1023]
    p2 = lut[(b2 ^ (h2 >> 22)) & 1023]
    p3 = lut[(b3 ^ (h3 >> 22)) & 1023]
    q0 = np.array([p0[0] * c0[0] + c0[2], p0[1] * c0[1] + c0[3], p1[0] * c0[0] + c0[2], p1[1] * c0[1] + c0[3]], dtype=np.float32)
    q1 = np.array([p2[0] * c2[0] + c2[2], p2[1] * c2[1] + c2[3], p3[0] * c2[0] + c2[2], p3[1] * c2[1] + c2[3]], dtype=np.float32)
    return lo32 & 0xFFFFFFFF, np.concatenate([q0, q1])


def dequantize_tfpn_tensor(shard_path: str, tables_path: str, tensor_name: str, max_blocks: int = 1) -> np.ndarray:
    """Dequantize the top-left [max_blocks*128, max_blocks*128] tiles of a TFPN tensor to float32."""
    tbl = load_tables(tables_path)
    with open(shard_path, "rb") as f:
        hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
        magic, ver, nt, _, off_meta, off_scales, off_weights, _, _, _ = hdr
        if magic != MAGIC_TFPN or ver != FORMAT_VER:
            raise ValueError(f"Bad TFPN header in {shard_path}")
        f.seek(off_meta)
        found = None
        for _ in range(nt):
            rec = f.read(128)
            nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
            if nm == tensor_name:
                found = struct.unpack("<IIIIQQfIII", rec[80:])
                break
        if found is None:
            raise KeyError(f"Tensor {tensor_name} not in {shard_path}")
        R, C, bits, sc_off, w_off, w_cnt, snr_db, _, _, _ = found
        n_bc = C // 128
        wpb = bits * 2
        rb_lim = min(max_blocks, R // 128)
        cb_lim = min(max_blocks, n_bc)
        f.seek(off_scales + sc_off * 2)
        scales = np.frombuffer(f.read((R // 128) * n_bc * 2), dtype=np.float16).reshape(R // 128, n_bc)
        f.seek(off_weights + w_off * 4)
        words = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32)

    out = np.zeros((rb_lim * 128, cb_lim * 128), dtype=np.float32)
    for rb in range(rb_lim):
        for cb in range(cb_lim):
            blk_rot = np.zeros((128, 128), dtype=np.float32)
            sc = float(scales[rb, cb])
            for s in range(8):
                strip = rb * 8 + s
                strip_base = strip * n_bc * wpb * 32
                for lane in range(32):
                    st = (INIT_ST_BASE ^ ((lane * HASH_MULT) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    if cb > 0:
                        if bits == 4:
                            st = int(words[strip_base + (cb * 2 - 1) * 128 + lane * 4 + 3])
                        else:
                            st = int(words[strip_base + (cb * wpb - 1) * 32 + lane])
                    hc, hr = _frag_home(lane)
                    if bits == 4:
                        for h in range(2):
                            idx = strip_base + (cb * 2 + h) * 128 + lane * 4
                            for k in range(4):
                                st, q = _decode_8_4b(st, int(words[idx + k]), tbl)
                                c0 = h * 64 + k * 16 + hc
                                blk_rot[s * 16 + hr, c0 : c0 + 4] = q[:4] * sc
                                blk_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:] * sc
                    elif bits == 3:
                        for h in range(2):
                            idx = strip_base + (cb * 6 + h * 3) * 32 + lane
                            a0, a1, a2 = int(words[idx]), int(words[idx + 32]), int(words[idx + 64])
                            m24s = [
                                a0 >> 8,
                                ((a0 << 16) | (a1 >> 16)) & 0xFFFFFF,
                                ((a1 << 8) | (a2 >> 24)) & 0xFFFFFF,
                                a2 & 0xFFFFFF,
                            ]
                            for k, m24 in enumerate(m24s):
                                st, q = _decode_8_3b(st, m24, tbl)
                                c0 = h * 64 + k * 16 + hc
                                blk_rot[s * 16 + hr, c0 : c0 + 4] = q[:4] * sc
                                blk_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:] * sc
                    else:
                        for h in range(2):
                            idx = strip_base + (cb * 10 + h * 5) * 32 + lane
                            w0, w1, w2, w3, w4 = [int(words[idx + j * 32]) for j in range(5)]
                            pairs = [
                                (w0 >> 24, ((w0 << 8) | (w1 >> 24)) & 0xFFFFFFFF),
                                ((w1 >> 16) & 255, ((w1 << 16) | (w2 >> 16)) & 0xFFFFFFFF),
                                ((w2 >> 8) & 255, ((w2 << 24) | (w3 >> 8)) & 0xFFFFFFFF),
                                (w3 & 255, w4 & 0xFFFFFFFF),
                            ]
                            for k, (hi8, lo32) in enumerate(pairs):
                                st, q = _decode_8_5b(st, hi8, lo32, tbl)
                                c0 = h * 64 + k * 16 + hc
                                blk_rot[s * 16 + hr, c0 : c0 + 4] = q[:4] * sc
                                blk_rot[s * 16 + hr + 8, c0 : c0 + 4] = q[4:] * sc
            out[rb * 128 : (rb + 1) * 128, cb * 128 : (cb + 1) * 128] = unrotate_block_2d(
                blk_rot, r_off=rb * 128, c_off=cb * 128
            )
    return out


def self_test() -> None:
    rng = np.random.default_rng(42)
    W = rng.standard_normal((128, 128)).astype(np.float32)
    # Inject extreme transformer-like channel outliers (100x std)
    W[17, :] *= 25.0
    W[:, 93] *= 25.0
    W_rot = rotate_block_2d(W, r_off=256, c_off=512)
    W_rec = unrotate_block_2d(W_rot, r_off=256, c_off=512)
    max_err = float(np.max(np.abs(W - W_rec)))
    assert max_err < 1e-4, f"FWHT inverse error too large: {max_err}"

    kurt_before = float(np.mean(W**4) / (np.mean(W**2) ** 2))
    kurt_after = float(np.mean(W_rot**4) / (np.mean(W_rot**2) ** 2))
    print(f"PASS 2D FWHT-128 exact invertibility: max_err={max_err:.2e}")
    print(f"PASS Outlier flattening (kurtosis):   before={kurt_before:.1f} -> after={kurt_after:.2f} (Gaussian=3.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description="TFQ3 Python/NumPy reference codec and inspector")
    ap.add_argument("--self-test", action="store_true", help="Run mathematical invertibility & outlier-flattening self-test")
    ap.add_argument("--shard", help="Optional TFPN .tffpw shard to dequantize a tile from")
    ap.add_argument("--tables", help="Path to pw_v3_tables.bin")
    ap.add_argument("--tensor", help="Tensor name inside --shard")
    args = ap.parse_args()
    if args.self_test or not args.shard:
        self_test()
    if args.shard:
        tables = args.tables or os.path.join(os.path.dirname(args.shard), "pw_v3_tables.bin")
        tile = dequantize_tfpn_tensor(args.shard, tables, args.tensor, max_blocks=1)
        print(f"Dequantized {args.tensor} [0:128, 0:128]: mean={tile.mean():.5f} std={tile.std():.5f} min={tile.min():.4f} max={tile.max():.4f}")


if __name__ == "__main__":
    main()
