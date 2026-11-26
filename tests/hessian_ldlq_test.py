#!/usr/bin/env python3
"""
CPU tests for the LDLQ error feedback in tools/tfq3_hessian.py (no GPU, no model, no compiled tools).

Correct LDLQ quantizes each block against the target that minimizes the remaining output error
given every earlier block already fixed: t_rest = W_rest + E_done H_done,rest H_rest,rest^-1.
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import tfq3_hessian as tfq3_h


def _random_hessian(C: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * C, C, generator=g) @ (torch.eye(C) + 0.3 * torch.randn(C, C, generator=g))
    return (X.T @ X) / X.shape[0]


def _conditional_targets(W_rot0: torch.Tensor, W_rot_hat: torch.Tensor, H: torch.Tensor, s: int) -> torch.Tensor:
    """Exact targets for columns [s, C) given columns [0, s) are fixed to W_rot_hat (float64)."""
    if s == 0:
        return W_rot0.clone()
    E_d = W_rot0[:, :s] - W_rot_hat[:, :s]
    return W_rot0[:, s:] + E_d @ torch.linalg.solve(H[s:, s:], H[:s, s:].T).T


def _setup(R: int, C: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(R, C, generator=g) * 0.02
    H = _random_hessian(C, seed + 1)
    H_rot = tfq3_h.rotate_hessian_torch(tfq3_h.pad_hessian_128(H, C))
    H_inv = tfq3_h.compute_h_inv_cholesky(H_rot, damp=0.01)
    H_eff = torch.linalg.inv(H_inv.to(torch.float64))  # damped H the feedback works against
    W_rot0 = tfq3_h.rotate_2d_torch(W).to(torch.float64)
    return W, H, H_eff, W_rot0


def _synthetic_tables(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "lut": {4: rng.standard_normal((1024, 2)).astype(np.float16)},
        "c64": {4: rng.standard_normal((64, 4)).astype(np.float16)},
    }


def _tfq3_with_block_targets(W: torch.Tensor, **kwargs) -> tuple[dict, dict[int, torch.Tensor]]:
    """Run quantize_tfq3_matrix and record the W_rot columns each 128-column block is encoded against."""
    seen: dict[int, torch.Tensor] = {}
    encode = tfq3_h._encode_block_viterbi

    def recording_encode(*args, **kw):
        caller = sys._getframe(1).f_locals
        cb = caller["cb"]
        seen[cb] = caller["W_rot"][:, cb * 128 : (cb + 1) * 128].detach().to(torch.float64).clone()
        return encode(*args, **kw)

    tfq3_h._encode_block_viterbi = recording_encode
    try:
        res = tfq3_h.quantize_tfq3_matrix(W, 4, _synthetic_tables(4), beam=4, topk=4, refine_scales=False, **kwargs)
    finally:
        tfq3_h._encode_block_viterbi = encode
    return res, seen


def test_tfq3_gain_three_correlated_coordinates() -> None:
    # One coordinate per block (0, 128, 256) with H^-1 diagonal 1, off-diagonal 0.5, all else independent.
    # After block 0 is fixed, a unit error in block 1 must move block 2 by -1/3 (the full H^-1 gives -1/2).
    C = 384
    H_inv = torch.eye(C, dtype=torch.float64)
    idx = torch.tensor([0, 128, 256])
    H_inv[idx[:, None], idx[None, :]] = 0.5
    H_inv[idx, idx] = 1.0
    H_rot = torch.linalg.inv(H_inv).to(torch.float32)
    W = torch.randn(256, C, generator=torch.Generator().manual_seed(5)) * 0.02
    res, seen = _tfq3_with_block_targets(W, precomputed_h=(H_rot, H_inv.to(torch.float32)))
    W_rot0 = tfq3_h.rotate_2d_torch(W).to(torch.float64)
    W_rot_hat = tfq3_h.rotate_2d_torch(res["W_hat"]).to(torch.float64)
    e0 = W_rot0[:, 0] - W_rot_hat[:, 0]
    e1 = seen[1][:, 0] - W_rot_hat[:, 128]
    shift = seen[2][:, 0] - W_rot0[:, 256] + 0.5 * e0  # block 0's -e0/2 is the same on main and here
    gain = -float(torch.dot(shift, e1) / torch.dot(e1, e1))
    assert abs(gain - 1.0 / 3.0) < 1e-3, f"block 2 moved by {-gain:.4f} per unit error in block 1, expected -1/3"


def test_ldlq_gain_helper() -> None:
    H_inv = torch.full((3, 3), 0.5, dtype=torch.float64) + 0.5 * torch.eye(3, dtype=torch.float64)
    U = tfq3_h.ldlq_upper_factor(H_inv).to(torch.float64)
    assert abs(float(tfq3_h.ldlq_gain(U, 0, 1)[0, 0]) - 0.5) < 1e-6
    assert abs(float(tfq3_h.ldlq_gain(U, 1, 2)[0, 0]) - 1.0 / 3.0) < 1e-6


def test_tfq4_feedback_is_sequential_conditional() -> None:
    R, C = 128, 256
    W, H, H_eff, W_rot0 = _setup(R, C, seed=1)
    lut = np.abs(np.random.default_rng(2).standard_normal((256, 8))).astype(np.float32)
    res = tfq3_h.quantize_tfq4_matrix(W, lut, H=H, refine_scales=False)
    W_rot_hat = tfq3_h.rotate_2d_torch(res["W_hat"]).to(torch.float64)
    sc = torch.from_numpy(res["scales"].astype(np.float32)).to(torch.float64)
    lut_t = torch.from_numpy(lut).to(torch.float64)
    worst = 0.0
    for c0 in range(0, C, 8):
        sc_rows = sc[:, c0 // 128].repeat_interleave(128)[:, None]
        v = _conditional_targets(W_rot0, W_rot_hat, H_eff, c0)[:, :8] / sc_rows
        q = W_rot_hat[:, c0 : c0 + 8] / sc_rows
        d_best = torch.min(torch.cdist(v.abs(), lut_t) ** 2, dim=1).values
        d_chosen = torch.sum((v - q) ** 2, dim=1)
        worst = max(worst, float(torch.max(d_chosen - d_best)))
    # Every 8-column code must be the nearest codeword to its exact conditional target.
    assert worst < 1e-3, f"chosen code is {worst:.4f} farther than the nearest codeword to the conditional target"


def test_tfq3_feedback_is_sequential_conditional() -> None:
    R, C = 128, 384
    W, H, H_eff, W_rot0 = _setup(R, C, seed=3)
    res, seen = _tfq3_with_block_targets(W, H=H)
    W_rot_hat = tfq3_h.rotate_2d_torch(res["W_hat"]).to(torch.float64)
    rms = float(torch.sqrt(torch.mean(W_rot0 * W_rot0)))
    for cb in range(C // 128):
        want = _conditional_targets(W_rot0, W_rot_hat, H_eff, cb * 128)[:, :128]
        err = float(torch.max(torch.abs(seen[cb] - want))) / rms
        assert err < 1e-3, f"block {cb}: encoder target is off the conditional target by {err:.3f} x rms(W)"


def test_cholesky_failure_warns() -> None:
    H = torch.tensor([[1.0, 10.0], [10.0, 1.0]])  # indefinite even after the fallback damping
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H_inv = tfq3_h.compute_h_inv_cholesky(H)
    assert torch.equal(H_inv, torch.eye(2))
    assert any(issubclass(w.category, RuntimeWarning) and "Cholesky failed" in str(w.message) for w in caught)


if __name__ == "__main__":
    torch.manual_seed(0)
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # report every test, then exit non-zero
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
