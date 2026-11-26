#!/usr/bin/env python3
"""
Reproducible Head-to-Head Quantizer Quality Benchmark & Provenance Generator.

Compares TFQ3 (Data-Free & Optional Hessian-LDLQ) and TFQ4-E8P against standard quantizers
under strictly controlled, matched-scope conditions (identical BF16 source, identical BF16
embeddings/norms/lm_head, disjoint WikiText-2 train calibration vs test evaluation splits):

  Methods compared at 4-bit, 3-bit, and 2-bit:
    1. BF16 Reference (gold baseline)
    2. RTN / MLX-style Asymmetric Affine (g64 and g128)
    3. AWQ-style Activation-Aware Salient Channel Scaling + Affine (g128)
    4. GPTQ / LDLQ Exact Hessian-Cholesky Error-Feedback Scalar Affine (g128)
    5. EXL3 / QTIP-style 2D-FWHT + Per-Column Hessian-LDLQ + Non-Uniform Optimal Gaussian Grid
    6. TFQ3 Data-Free (real compiled `tfq3-generic-convert` binary -> exact TFPN v3 decode!)
    7. TFQ3 + Hessian-LDLQ (optional calibration mode -> exact same TFPN v3 binary format!)
    8. TFQ4-E8P (2.001 bpw, Data-Free real `tfq4-convert` binary & Hessian-LDLQ)

  Metrics reported for EVERY method (exact fp64 accumulation):
    - Exact effective bits/weight (bpw)
    - Mean Weight SNR (dB) across all quantized linear layers
    - Mean Real-Activation Output SNR (dB): ||W X^T||_F / ||(W - W_hat) X^T||_F on held-out X
    - Full-model WikiText-2 test Perplexity (16,384 held-out tokens) + bootstrap 95% CI
    - Full-vocabulary exact fp64 forward KL D_KL(P_BF16 || P_q) over 4,096 tokens
      (mean, median, p95, p99, max KL, and Top-1 token agreement %)
    - Wall-clock quantization time (seconds)

Emits a SHA-256 provenance receipt (`benchmarks/provenance.json`) pinning model revision,
tokenized calibration & eval corpus hashes, hardware, and exact per-method metrics.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import tfq3_reference as tfq3_ref
import tfq4_lattice as tfq4_ref
import tfq3_hessian as tfq3_h

HASH_MULT = tfq3_ref.HASH_MULT
INIT_ST_BASE = tfq3_ref.INIT_ST_BASE
decode_tfpn_fast_np = tfq3_ref.decode_tfpn_fast_np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_token_chunks(tokenizer, split_texts: list[str], seq_len: int, n_seqs: int) -> list[torch.Tensor]:
    text = "\n\n".join(t for t in split_texts if t.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    need = seq_len * n_seqs
    if len(ids) < need:
        raise RuntimeError(f"Not enough tokens ({len(ids)} < {need})")
    arr = torch.tensor(ids[:need], dtype=torch.long).reshape(n_seqs, seq_len)
    return [arr[i : i + 1] for i in range(n_seqs)]


# ----------------------------------------------------------------------------
# Reference Quantizer Implementations (all matched on exact same W [R, C] & H [C, C])
# ----------------------------------------------------------------------------
def quant_affine_rtn(W: torch.Tensor, bits: int, group: int = 64) -> torch.Tensor:
    """Asymmetric uniform affine quantization (MLX-style g64 / g128)."""
    R, C = W.shape
    pad_C = ((C + group - 1) // group) * group
    Wp = W.to(torch.float32)
    if pad_C != C:
        Wp = torch.nn.functional.pad(Wp, (0, pad_C - C))
    G = Wp.reshape(R, pad_C // group, group)
    qmax = float((1 << bits) - 1)
    lo = G.amin(dim=-1, keepdim=True)
    hi = G.amax(dim=-1, keepdim=True)
    sc = torch.clamp((hi - lo) / qmax, min=1e-12)
    q = torch.clamp(torch.round((G - lo) / sc), 0.0, qmax)
    out = (q * sc + lo).reshape(R, pad_C)[:, :C]
    return out


def quant_awq_salient(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """
    AWQ (Lin et al., MLSys 2024): per-input-channel activation-salient scaling s = x_rms^alpha,
    grid-searching alpha in [0, 1] to minimize real activation output error ||(W - W_hat) X^T||_F^2.
    """
    Wf = W.to(torch.float32)
    x_rms = torch.sqrt(torch.clamp(torch.diagonal(H.to(torch.float32)), min=1e-12))
    x_rms = x_rms / torch.clamp(x_rms.mean(), min=1e-12)
    best_err = float("inf")
    best_W = None
    # Fast exact output error via tr((W - W_hat) H (W - W_hat)^T)
    Hf = H.to(torch.float32)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = torch.clamp(x_rms.pow(alpha), min=1e-3, max=1e3)
        W_scaled = Wf * s[None, :]
        W_q = quant_affine_rtn(W_scaled, bits, group=group) / s[None, :]
        d = Wf - W_q
        err = float(torch.sum((d @ Hf) * d).item())
        if err < best_err:
            best_err = err
            best_W = W_q
    return best_W


def quant_gptq_ldlq(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128, damp: float = 0.01, u_inv: torch.Tensor | None = None) -> torch.Tensor:
    """
    GPTQ / LDLQ (Frantar et al., ICLR 2023): exact block-Cholesky Hessian error feedback
    with asymmetric per-group affine scalar quantization (group=128).
    """
    orig_dev = W.device
    Wf = W.detach().cpu().to(torch.float32).clone()
    R, C = Wf.shape
    if u_inv is None:
        H_inv = tfq3_h.compute_h_inv_cholesky(H.cpu(), damp=damp)
        U_inv = torch.linalg.cholesky(H_inv.to(torch.float64)).T.to(torch.float32)
    else:
        U_inv = u_inv.cpu()
    qmax = float((1 << bits) - 1)
    Q = torch.zeros_like(Wf)

    for c0 in range(0, C, group):
        c1 = min(c0 + group, C)
        blk = Wf[:, c0:c1].clone()
        lo = blk.amin(dim=1, keepdim=True)
        hi = blk.amax(dim=1, keepdim=True)
        sc = torch.clamp((hi - lo) / qmax, min=1e-12)
        U_blk = U_inv[c0:c1, c0:c1]
        Err = torch.zeros_like(blk)

        for i in range(c1 - c0):
            w = blk[:, i]
            d = torch.clamp(U_blk[i, i], min=1e-6)
            q = torch.clamp(torch.round((w - lo[:, 0]) / sc[:, 0]), 0.0, qmax) * sc[:, 0] + lo[:, 0]
            Q[:, c0 + i] = q
            err_i = (w - q) / d
            Err[:, i] = err_i
            if i + 1 < (c1 - c0):
                blk[:, i + 1 :] -= err_i.unsqueeze(1) * U_blk[i, i + 1 :].unsqueeze(0)

        if c1 < C:
            Wf[:, c1:] -= Err @ U_inv[c0:c1, c1:]

    return Q.to(orig_dev)


# Exact 1D Max-Lloyd optimal Gaussian centroids (unit variance) for EXL3/QTIP-style rotated-LDLQ reference
_GAUSS_LLOYD_CACHE: dict[int, np.ndarray] = {}


def _gauss_lloyd_1d(bits: int) -> np.ndarray:
    if bits in _GAUSS_LLOYD_CACHE:
        return _GAUSS_LLOYD_CACHE[bits]
    K = 1 << bits
    rng = np.random.default_rng(2026 + bits)
    x = rng.standard_normal(262144).astype(np.float64)
    qs = np.linspace(0.5 / K, 1.0 - 0.5 / K, K)
    cen = np.quantile(x, qs)
    for _ in range(40):
        mid = 0.5 * (cen[:-1] + cen[1:])
        idx = np.searchsorted(mid, x)
        for k in range(K):
            m = idx == k
            if np.any(m):
                cen[k] = x[m].mean()
    _GAUSS_LLOYD_CACHE[bits] = cen.astype(np.float32)
    return _GAUSS_LLOYD_CACHE[bits]


def quant_hadamard_ldlq_lloyd(
    W: torch.Tensor, H: torch.Tensor, bits: int, damp: float = 0.01,
    precomputed: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    EXL3 / QTIP-style Hessian-Rotated LDLQ Reference (exact 2D FWHT-128 + full-rank per-column
    Cholesky LDLQ + optimal Gaussian Max-Lloyd codebook + Hessian-optimal 128x128 tile scale).
    Exact same effective bpw (bits + 0.001) as TFQ3, isolating the ceiling of column-wise Hessian LDLQ.
    """
    orig_dev = W.device
    W_pad, orig_R, orig_C = tfq3_h.pad_weight_128(W.detach().cpu())
    R, C = W_pad.shape
    n_br, n_bc = R // 128, C // 128

    W_rot0 = tfq3_h.rotate_2d_torch(W_pad)
    W_rot = W_rot0.clone()
    if precomputed is not None:
        H_rot, U_inv = precomputed[0].cpu(), precomputed[1].cpu()
    else:
        H_pad = tfq3_h.pad_hessian_128(H.cpu(), C)
        H_rot = tfq3_h.rotate_hessian_torch(H_pad)
        H_inv = tfq3_h.compute_h_inv_cholesky(H_rot, damp=damp)
        U_inv = torch.linalg.cholesky(H_inv.to(torch.float64)).T.to(torch.float32)

    cen = torch.from_numpy(_gauss_lloyd_1d(bits))
    mids = 0.5 * (cen[:-1] + cen[1:])

    Q_unit = torch.zeros_like(W_rot)
    scales = torch.zeros((n_br, n_bc), dtype=torch.float32)

    for cb in range(n_bc):
        c0, c1 = cb * 128, (cb + 1) * 128
        blk = W_rot[:, c0:c1].clone()
        sc_b = torch.sqrt(torch.mean(blk.reshape(n_br, 128, 128) ** 2, dim=(1, 2)) + 1e-12).to(torch.float16).to(torch.float32)
        scales[:, cb] = sc_b
        sc_rows = sc_b.repeat_interleave(128)
        inv_sc = torch.where(sc_rows > 1e-12, 1.0 / sc_rows, torch.zeros_like(sc_rows))

        U_blk = U_inv[c0:c1, c0:c1]
        Err = torch.zeros_like(blk)
        for i in range(128):
            w = blk[:, i]
            d = torch.clamp(U_blk[i, i], min=1e-6)
            idx = torch.bucketize(w * inv_sc, mids)
            qu = cen[idx]
            q = qu * sc_rows
            Q_unit[:, c0 + i] = qu
            err_i = (w - q) / d
            Err[:, i] = err_i
            if i + 1 < 128:
                blk[:, i + 1 :] -= err_i.unsqueeze(1) * U_blk[i, i + 1 :].unsqueeze(0)

        if c1 < C:
            W_rot[:, c1:] -= Err @ U_inv[c0:c1, c1:]

    for cb in range(n_bc):
        c0, c1 = cb * 128, (cb + 1) * 128
        H_blk = H_rot[c0:c1, c0:c1]
        for rb in range(n_br):
            r0, r1 = rb * 128, (rb + 1) * 128
            W_b = W_rot0[r0:r1, c0:c1]
            Q_b = Q_unit[r0:r1, c0:c1]
            QH = Q_b @ H_blk
            num = float(torch.sum(W_b * QH).item())
            den = float(torch.sum(Q_b * QH).item())
            if den > 1e-12 and num > 0.0:
                scales[rb, cb] = float(torch.tensor(num / den, dtype=torch.float16).item())

    W_hat = tfq3_h.unrotate_2d_torch(
        Q_unit * scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )[:orig_R, :orig_C]
    return W_hat.to(orig_dev)


def compute_layer_snrs(W: torch.Tensor, W_hat: torch.Tensor, H_eval: torch.Tensor) -> tuple[float, float]:
    """Exact Weight SNR (dB) and Held-Out Real-Activation Output SNR (dB) on device."""
    Wf = W.detach().to(torch.float32)
    Df = Wf - W_hat.detach().to(device=Wf.device, dtype=torch.float32)
    Hf = H_eval.detach().to(device=Wf.device, dtype=torch.float32)
    w_sig = float(torch.sum(Wf * Wf).item())
    w_err = float(torch.sum(Df * Df).item())
    w_snr = 10.0 * math.log10(w_sig / max(w_err, 1e-30))
    y_sig = float(torch.sum((Wf @ Hf) * Wf).item())
    y_err = float(torch.sum((Df @ Hf) * Df).item())
    out_snr = 10.0 * math.log10(y_sig / max(y_err, 1e-30))
    return w_snr, out_snr


@torch.no_grad()
def evaluate_model_ppl_and_kl(
    model: torch.nn.Module,
    eval_batches: list[torch.Tensor],
    kl_batches_count: int,
    ref_logprobs: list[torch.Tensor] | None,
    device: torch.device,
) -> dict:
    """
    Evaluate exact fp64 WikiText-2 NLL/PPL over ALL eval_batches (with 95% bootstrap CI),
    plus exact fp64 full-vocab forward KL D_KL(P_ref || P_q) and Top-1 agreement over
    the first `kl_batches_count` sequences.
    """
    model.eval()
    seq_nlls = []
    tot_nll = 0.0
    tot_tok = 0
    kl_vals = []
    top1_match = 0
    top1_tot = 0
    captured_ref = [] if ref_logprobs is None else None

    for bi, ids in enumerate(eval_batches):
        ids_dev = ids.to(device)
        logits = model(ids_dev).logits[:, :-1, :].detach().cpu().to(torch.float64)  # [1, T-1, V]
        targets = ids[:, 1:].to(torch.long)                                          # [1, T-1]
        logp = torch.log_softmax(logits, dim=-1)                                     # [1, T-1, V]
        tok_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)     # [T-1]
        seq_nlls.append(float(tok_nll.mean().item()))
        tot_nll += float(tok_nll.sum().item())
        tot_tok += int(tok_nll.numel())

        if bi < kl_batches_count:
            if ref_logprobs is None:
                captured_ref.append(logp.squeeze(0).to(torch.float32))
            else:
                lp_ref = ref_logprobs[bi].to(torch.float64)
                lp_q = logp.squeeze(0)
                p_ref = torch.exp(lp_ref)
                kl = torch.sum(p_ref * (lp_ref - lp_q), dim=-1)  # [T-1]
                kl_vals.append(kl.numpy())
                top1_match += int((lp_ref.argmax(dim=-1) == lp_q.argmax(dim=-1)).sum().item())
                top1_tot += int(lp_ref.shape[0])

    mean_nll = tot_nll / tot_tok
    ppl = math.exp(mean_nll)
    # 1,000-sample deterministic bootstrap 95% CI on sequence-level NLL -> PPL
    rng = np.random.default_rng(2026)
    arr = np.array(seq_nlls, dtype=np.float64)
    boots = np.exp(rng.choice(arr, size=(1000, len(arr)), replace=True).mean(axis=1))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    res = {
        "eval_tokens": tot_tok,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "ppl_ci95": [ci_lo, ci_hi],
    }
    if ref_logprobs is None:
        res["ref_logprobs"] = captured_ref
    else:
        all_kl = np.concatenate(kl_vals)
        res["kl_tokens"] = int(all_kl.size)
        res["kl_mean"] = float(np.mean(all_kl))
        res["kl_median"] = float(np.median(all_kl))
        res["kl_p95"] = float(np.percentile(all_kl, 95))
        res["kl_p99"] = float(np.percentile(all_kl, 99))
        res["kl_max"] = float(np.max(all_kl))
        res["same_top1_pct"] = 100.0 * top1_match / max(top1_tot, 1)
    return res


def load_real_tfq3_binary_weights(model_dir: str, bits: int, tmp_dir: str) -> tuple[dict[str, np.ndarray], float]:
    """Run the ACTUAL compiled Metal/CUDA TFQ3 binary converter and decode every quantized tensor!"""
    suffix = "" if platform.system() == "Darwin" else "-cuda"
    conv_bin = os.path.join(REPO_ROOT, "build", f"tfq3-generic-convert{suffix}")
    insp_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    out_dir = os.path.join(tmp_dir, f"real_tfq3_{bits}b")
    t0 = time.time()
    subprocess.run(
        [conv_bin, "--model", model_dir, "--out", out_dir, "--bits", str(bits), "--head-bits", "0"],
        check=True, stdout=subprocess.DEVNULL,
    )
    conv_sec = time.time() - t0
    subprocess.run([insp_bin, out_dir], check=True, stdout=subprocess.DEVNULL)

    tbl_path = os.path.join(out_dir, "pw_v3_tables.bin")
    tbl = tfq3_ref.load_tables(tbl_path)
    weights: dict[str, np.ndarray] = {}

    # Fast vectorized PyTorch decode of EVERY tensor from the real binary TFPN shards!
    # Also verify exact parity against tfq3_reference.dequantize_tfpn_tensor on the first tensor!
    verified_ref = False
    for shard_path in sorted(glob.glob(os.path.join(out_dir, "*.tffpw"))):
        with open(shard_path, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            nt, off_meta, off_scales, off_weights = hdr[2], hdr[4], hdr[5], hdr[6]
            f.seek(off_meta)
            recs = [f.read(128) for _ in range(nt)]
            for rec in recs:
                nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
                R, C, b, sc_off, w_off, w_cnt, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
                orig_r = orig_r or R
                orig_c = orig_c or C
                f.seek(off_scales + sc_off * 2)
                sc = np.frombuffer(f.read((R // 128) * (C // 128) * 2), dtype=np.float16).reshape(R // 128, C // 128)
                f.seek(off_weights + w_off * 4)
                wd = np.frombuffer(f.read(w_cnt * 4), dtype=np.uint32)
                W_dec = decode_tfpn_fast_np(wd, sc, R, C, b, tbl)[:orig_r, :orig_c]
                if not verified_ref:
                    W_ref = tfq3_ref.dequantize_tfpn_tensor(shard_path, tbl_path, nm, max_blocks=1)
                    err = float(np.max(np.abs(W_dec[:128, :128] - W_ref)))
                    assert err < 1e-4, f"Fast TFPN decode mismatch vs tfq3_reference: {err}"
                    verified_ref = True
                weights[nm] = W_dec
    return weights, conv_sec


def load_real_tfq4_binary_weights(model_dir: str, tmp_dir: str) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Run the ACTUAL compiled Metal/CUDA TFQ4 converter and decode every quantized tensor!"""
    suffix = "" if platform.system() == "Darwin" else "-cuda"
    conv_bin = os.path.join(REPO_ROOT, "build", f"tfq4-convert{suffix}")
    insp_bin = os.path.join(REPO_ROOT, "build", "tfq3-inspect")
    out_dir = os.path.join(tmp_dir, "real_tfq4_2b")
    t0 = time.time()
    subprocess.run(
        [conv_bin, "--model", model_dir, "--out", out_dir, "--head-bits", "0"],
        check=True, stdout=subprocess.DEVNULL,
    )
    conv_sec = time.time() - t0
    subprocess.run([insp_bin, out_dir], check=True, stdout=subprocess.DEVNULL)

    tbl_path = os.path.join(out_dir, "pw_v4_e8p_tables.bin")
    lut = tfq4_ref.load_pwv4_tables(tbl_path)
    weights: dict[str, np.ndarray] = {}
    for shard_path in sorted(glob.glob(os.path.join(out_dir, "*.tffp4"))):
        with open(shard_path, "rb") as f:
            hdr = struct.unpack("<IIIIQQQQQQ", f.read(64))
            nt, off_meta = hdr[2], hdr[4]
            f.seek(off_meta)
            recs = [f.read(128) for _ in range(nt)]
        for rec in recs:
            nm = rec[:80].split(b"\x00", 1)[0].decode("utf-8")
            R, C, _, _, _, _, _, orig_r, orig_c, _ = struct.unpack("<IIIIQQfIII", rec[80:])
            orig_r = orig_r or R
            orig_c = orig_c or C
            weights[nm] = tfq4_ref.dequantize_tfp4_tensor(shard_path, tbl_path, nm)[:orig_r, :orig_c]
    return weights, lut, conv_sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local HF BF16 model directory (e.g. Qwen2.5-1.5B)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--calib-seqs", type=int, default=64, help="Calibration sequences (seq_len=512 from WikiText-2 train)")
    ap.add_argument("--eval-seqs", type=int, default=32, help="Eval sequences (seq_len=512 from WikiText-2 test = 16,384 tok)")
    ap.add_argument("--kl-seqs", type=int, default=8, help="Full-vocab KL sequences (seq_len=512 = 4,096 tok)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-json", default=os.path.join(REPO_ROOT, "benchmarks", "provenance.json"))
    args = ap.parse_args()

    device = get_device()
    print(f"=== TFQ Controlled Head-to-Head Quantizer Benchmark ===")
    print(f"  model:   {args.model_id} ({args.model})")
    print(f"  device:  {device} ({platform.system()} {platform.machine()})")
    print(f"  calib:   {args.calib_seqs} x {args.seq_len} = {args.calib_seqs * args.seq_len} WikiText-2 train tokens")
    print(f"  eval:    {args.eval_seqs} x {args.seq_len} = {args.eval_seqs * args.seq_len} WikiText-2 test tokens ({args.kl_seqs * args.seq_len} full-vocab KL tokens)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    wt_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
    wt_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    calib_batches = build_token_chunks(tokenizer, wt_train, args.seq_len, args.calib_seqs)
    eval_batches = build_token_chunks(tokenizer, wt_test, args.seq_len, args.eval_seqs)

    calib_sha = sha256_bytes(torch.cat(calib_batches, dim=0).numpy().tobytes())
    eval_sha = sha256_bytes(torch.cat(eval_batches, dim=0).numpy().tobytes())
    print(f"  calib_tokens_sha256: {calib_sha[:16]}... | eval_tokens_sha256: {eval_sha[:16]}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(device)
    model.eval()

    # Identify all transformer linear layers (keep embed_tokens & lm_head in original BF16 for apples-to-apples comparison)
    target_linears = [
        nm for nm, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and nm.startswith("model.layers.") and mod.weight.shape[0] >= 128 and mod.weight.shape[1] >= 128
    ]
    print(f"  quantized linear layers: {len(target_linears)} (embed_tokens, norms, biases, lm_head kept BF16)")

    # Snapshot original BF16 weights on CPU
    orig_weights = {nm: dict(model.named_modules())[nm].weight.detach().cpu().clone() for nm in target_linears}
    mod_map = {nm: mod for nm, mod in model.named_modules() if nm in orig_weights}

    # 1. Gold BF16 evaluation
    t0 = time.time()
    gold = evaluate_model_ppl_and_kl(model, eval_batches, args.kl_seqs, None, device)
    ref_logprobs = gold.pop("ref_logprobs")
    print(f"\n[GOLD BF16] PPL={gold['ppl']:.4f} (CI95 [{gold['ppl_ci95'][0]:.4f}, {gold['ppl_ci95'][1]:.4f}]) | NLL={gold['mean_nll']:.5f} ({time.time()-t0:.1f}s)")

    # 2. Collect strictly disjoint calibration Hessians (train) & held-out evaluation Hessians (test)
    t0 = time.time()
    H_calib = tfq3_h.collect_linear_hessians(model, calib_batches, target_linears, device)
    H_eval = tfq3_h.collect_linear_hessians(model, eval_batches[: args.kl_seqs], target_linears, device)
    print(f"Collected {len(H_calib)} train Hessians & {len(H_eval)} held-out test Hessians in {time.time()-t0:.1f}s")

    results = []
    results.append({
        "name": "BF16 Reference (Gold)",
        "bits": 16,
        "eff_bpw": 16.000,
        "uses_calibration": False,
        "weight_snr_db": None,
        "output_snr_db": None,
        "conv_sec": 0.0,
        **gold,
    })

    def run_variant(name: str, bits: int, eff_bpw: float, uses_calib: bool, quant_fn, conv_sec_override: float | None = None):
        t_q = time.time()
        q_hats: dict[str, torch.Tensor] = {}
        for nm in target_linears:
            W_orig = orig_weights[nm].to(device=device, dtype=torch.float32)
            q_hats[nm] = quant_fn(nm, W_orig)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        conv_s = conv_sec_override if conv_sec_override is not None else (time.time() - t_q)

        w_snrs, o_snrs = [], []
        for nm in target_linears:
            W_orig = orig_weights[nm].to(device=device, dtype=torch.float32)
            W_hat = q_hats[nm]
            ws, osnr = compute_layer_snrs(W_orig, W_hat, H_eval[nm])
            w_snrs.append(ws)
            o_snrs.append(osnr)
            mod_map[nm].weight.data.copy_(W_hat.to(device=device, dtype=torch.bfloat16))
        del q_hats
        ev = evaluate_model_ppl_and_kl(model, eval_batches, args.kl_seqs, ref_logprobs, device)
        row = {
            "name": name,
            "bits": bits,
            "eff_bpw": eff_bpw,
            "uses_calibration": uses_calib,
            "weight_snr_db": float(np.mean(w_snrs)),
            "output_snr_db": float(np.mean(o_snrs)),
            "conv_sec": float(conv_s),
            **ev,
        }
        results.append(row)
        print(
            f"  {name:<36s} | {eff_bpw:5.3f} bpw | W-SNR={row['weight_snr_db']:5.2f} dB | "
            f"Out-SNR={row['output_snr_db']:5.2f} dB | PPL={row['ppl']:6.4f} | "
            f"KL={row['kl_mean']:.5f} (p99={row['kl_p99']:.4f}) | Top1={row['same_top1_pct']:5.2f}% | {conv_s:5.1f}s"
        )
        # Restore BF16 weights cleanly before next variant
        for nm in target_linears:
            mod_map[nm].weight.data.copy_(orig_weights[nm].to(device=device, dtype=torch.bfloat16))

    # Precompute Cholesky factors ONCE per unique layer Hessian (shared cleanly across 4b/3b/2b)
    t_ch = time.time()
    gptq_uinv: dict[str, torch.Tensor] = {}
    rot_h_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rot_uinv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for nm in target_linears:
        Hc = H_calib[nm]
        pad_C = ((Hc.shape[0] + 127) // 128) * 128
        H_inv_raw = tfq3_h.compute_h_inv_cholesky(Hc, damp=0.01)
        gptq_uinv[nm] = torch.linalg.cholesky(H_inv_raw.to(torch.float64)).T.to(torch.float32)

        H_pad = tfq3_h.pad_hessian_128(Hc, pad_C)
        H_rot = tfq3_h.rotate_hessian_torch(H_pad)
        H_rot_inv = tfq3_h.compute_h_inv_cholesky(H_rot, damp=0.01)
        U_rot_inv = torch.linalg.cholesky(H_rot_inv.to(torch.float64)).T.to(torch.float32)
        rot_h_cache[nm] = (H_rot, H_rot_inv)
        rot_uinv_cache[nm] = (H_rot, U_rot_inv)
    cholesky_sec = time.time() - t_ch
    print(f"Precomputed {len(target_linears)} Cholesky & rotated-Hessian factors in {cholesky_sec:.1f}s")

    with tempfile.TemporaryDirectory(prefix="tfq_bench_") as tmp:
        # Load canonical PWV3 tables generated by real tfq3-generic-convert
        tfq3_4b_w, tfq3_4b_s = load_real_tfq3_binary_weights(args.model, 4, tmp)
        tfq3_3b_w, tfq3_3b_s = load_real_tfq3_binary_weights(args.model, 3, tmp)
        tfq4_2b_w, e8p_lut, tfq4_2b_s = load_real_tfq4_binary_weights(args.model, tmp)
        tables = tfq3_ref.load_tables(os.path.join(tmp, "real_tfq3_4b", "pw_v3_tables.bin"))

        print("\n=== 4-BIT HEAD-TO-HEAD (All 196 Linear Layers, BF16 Embed/Head) ===")
        run_variant("RTN / MLX Affine 4b (g64)", 4, 4.500, False, lambda nm, W: quant_affine_rtn(W, 4, 64))
        run_variant("RTN / MLX Affine 4b (g128)", 4, 4.250, False, lambda nm, W: quant_affine_rtn(W, 4, 128))
        run_variant("AWQ Salient-Scale 4b (g128)", 4, 4.250, True, lambda nm, W: quant_awq_salient(W, H_calib[nm].to(device), 4, 128))
        run_variant("GPTQ / LDLQ Scalar 4b (g128)", 4, 4.250, True, lambda nm, W: quant_gptq_ldlq(W, H_calib[nm], 4, 128, u_inv=gptq_uinv[nm]))
        run_variant("TFQ3-4b Data-Free (real .tffpw)", 4, 4.001, False, lambda nm, W: torch.from_numpy(tfq3_4b_w[nm]).to(device), conv_sec_override=tfq3_4b_s)
        run_variant(
            "TFQ3-4b + Hessian-LDLQ (same TFPN)", 4, 4.001, True,
            lambda nm, W: tfq3_h.quantize_tfq3_matrix(W, 4, tables, H=H_calib[nm], beam=8, precomputed_h=rot_h_cache[nm])["W_hat"],
        )
        run_variant(
            "EXL3/QTIP-ref 4b (2D-FWHT+LDLQ)", 4, 4.001, True,
            lambda nm, W: quant_hadamard_ldlq_lloyd(W, H_calib[nm], 4, precomputed=rot_uinv_cache[nm]),
        )

        print("\n=== 3-BIT HEAD-TO-HEAD (All 196 Linear Layers, BF16 Embed/Head) ===")
        run_variant("RTN / MLX Affine 3b (g64)", 3, 3.500, False, lambda nm, W: quant_affine_rtn(W, 3, 64))
        run_variant("AWQ Salient-Scale 3b (g128)", 3, 3.250, True, lambda nm, W: quant_awq_salient(W, H_calib[nm].to(device), 3, 128))
        run_variant("GPTQ / LDLQ Scalar 3b (g128)", 3, 3.250, True, lambda nm, W: quant_gptq_ldlq(W, H_calib[nm], 3, 128, u_inv=gptq_uinv[nm]))
        run_variant("TFQ3-3b Data-Free (real .tffpw)", 3, 3.001, False, lambda nm, W: torch.from_numpy(tfq3_3b_w[nm]).to(device), conv_sec_override=tfq3_3b_s)
        run_variant(
            "TFQ3-3b + Hessian-LDLQ (same TFPN)", 3, 3.001, True,
            lambda nm, W: tfq3_h.quantize_tfq3_matrix(W, 3, tables, H=H_calib[nm], beam=8, precomputed_h=rot_h_cache[nm])["W_hat"],
        )
        run_variant(
            "EXL3/QTIP-ref 3b (2D-FWHT+LDLQ)", 3, 3.001, True,
            lambda nm, W: quant_hadamard_ldlq_lloyd(W, H_calib[nm], 3, precomputed=rot_uinv_cache[nm]),
        )

        print("\n=== 2-BIT HEAD-TO-HEAD (All 196 Linear Layers, BF16 Embed/Head) ===")
        run_variant("RTN / MLX Affine 2b (g64)", 2, 2.500, False, lambda nm, W: quant_affine_rtn(W, 2, 64))
        run_variant("GPTQ / LDLQ Scalar 2b (g128)", 2, 2.250, True, lambda nm, W: quant_gptq_ldlq(W, H_calib[nm], 2, 128, u_inv=gptq_uinv[nm]))
        run_variant("TFQ4-E8P 2b Data-Free (real .tffp4)", 2, 2.001, False, lambda nm, W: torch.from_numpy(tfq4_2b_w[nm]).to(device), conv_sec_override=tfq4_2b_s)
        run_variant(
            "TFQ4-E8P 2b + Hessian-LDLQ (TFP4)", 2, 2.001, True,
            lambda nm, W: tfq3_h.quantize_tfq4_matrix(W, e8p_lut, H=H_calib[nm], precomputed_h=rot_h_cache[nm])["W_hat"],
        )

    # Emit pinned SHA-256 provenance receipt
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    snap_rev = os.path.basename(os.path.normpath(args.model))
    git_rev = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    prov = {
        "schema": "tfq-provenance-v1",
        "claim_status": "MEASURED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": git_rev,
        "benchmark_script": "tools/bench_quantizers.py",
        "benchmark_script_sha256": sha256_file(__file__),
        "hessian_module_sha256": sha256_file(os.path.join(REPO_ROOT, "tools", "tfq3_hessian.py")),
        "hardware": f"{platform.system()} {platform.machine()} (device={device})",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "model": {
            "repo_id": args.model_id,
            "snapshot_revision": snap_rev,
            "quantized_linears": len(target_linears),
            "unquantized_preserved_bf16": ["embed_tokens", "lm_head", "rms_norms", "biases"],
        },
        "dataset": {
            "name": "Salesforce/wikitext (wikitext-2-raw-v1)",
            "calibration_split": "train",
            "calibration_tokens": args.calib_seqs * args.seq_len,
            "calibration_tokens_sha256": calib_sha,
            "eval_split": "test",
            "eval_tokens": args.eval_seqs * args.seq_len,
            "eval_tokens_sha256": eval_sha,
            "kl_tokens": args.kl_seqs * args.seq_len,
        },
        "results": results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"\nWrote SHA-256 provenance receipt -> {args.out_json}")


if __name__ == "__main__":
    main()
