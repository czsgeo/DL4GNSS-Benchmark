"""classical.py -- classical signal-decomposition baselines for the benchmark.

One uniform surface for the three tasks so the DL methods (added later) plug
into the same harness:

    denoise(t, y, **kw)          -> y_hat          (signal estimate, len == y)
    impute(t, y, mask, **kw)     -> y_hat          (filled where mask is False)
    detect(t, y, **kw)           -> list[epoch]    (offset epochs, decimal year)

Every method is deterministic given (t, y, **kw); no fitting on test data.
Methods implemented here:
    mle      trajectory model (trend + annual + semiannual [+ offsets])
    ssa      singular-spectrum-analysis reconstruction
    wavelet  stationary/decimated wavelet soft-thresholding (PyWavelets)
    rts      harmonic state-space Kalman/RTS smoother
    median   moving-median smoother
    linear   piecewise-linear interpolation (imputation)
    cubic    cubic-spline interpolation (imputation)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_lib as gl  # noqa: E402


# ------------------------------------------------------------------ denoise
def mle_signal(t, y, orders=(1, 2), poly=2, offsets=(), robust=True):
    """Deterministic trajectory model: fitted trend + harmonics (+ offsets)."""
    X, _ = gl.build_design(t, orders=orders, poly=poly, offsets=offsets)
    if robust:
        beta, *_ = gl.robust_fit(t, y, X)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ beta


def ssa_reconstruct(y, L=None, n_components=10):
    """SSA reconstruction from the leading `n_components` by diagonal averaging.

    A full (dense) SVD of the L x K trajectory matrix is kept: on these shapes
    (L <= 120, K ~ 1e4) it is actually *faster* than a randomised or Lanczos
    truncated SVD (numpy's LAPACK gesdd is ~0.4 s vs ~2.4 s for randomised_svd),
    and the whole benchmark's SSA cost is bounded by the number of gap-filling
    iterations instead.
    """
    y = np.asarray(y, float)
    n = len(y)
    if L is None:
        L = min(max(n // 4, 20), 120)
    if n_components >= L:                     # never ask for more than L
        n_components = max(1, L - 1)
    K = n - L + 1
    X = np.lib.stride_tricks.sliding_window_view(y, L)     # K x L
    X = X.T                                                # L x K
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Xr = U[:, :n_components] @ (s[:n_components, None] * Vt[:n_components, :])
    y_hat = np.zeros(n)
    cnt = np.zeros(n)
    for i in range(L):
        y_hat[i:i + K] += Xr[i, :]
        cnt[i:i + K] += 1.0
    return y_hat / cnt


def wavelet_denoise(y, wavelet="db4", level=None, mode="soft"):
    """Wavelet soft-thresholding with the universal (VisuShrink) threshold."""
    import pywt
    y = np.asarray(y, float)
    n = len(y)
    max_lvl = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = max(1, min(max_lvl, 8))
    level = max(1, min(level, max_lvl)) if max_lvl >= 1 else 1
    coeffs = pywt.wavedec(y, wavelet, level=level, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
    coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet, mode="periodization")
    return rec[:n]


def rts_smooth(t, y, orders=(1, 2), q_h=1e-7, R=None):
    """Kalman/RTS smoother of the harmonic state-space model -> smoothed signal."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    t0 = float(np.mean(t))
    X, _ = gl.build_design(t, orders=orders, poly=2)
    if R is None:
        _, resid, _, _ = gl.robust_fit(t, y, X)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        R = float(mad ** 2) if mad > 0 else float(np.var(resid) or 1.0)
    H, Q = gl._ssm_matrices(t, orders, t0, q_h)
    XtXi = np.linalg.pinv(X.T @ X)
    x0 = XtXi @ (X.T @ y)
    P0 = XtXi * R
    xs, Ps, ll = gl.ssm_filter_smoother(y, H, Q, R, x0=x0, P0=P0)
    return np.einsum("ij,ij->i", X, xs)


def moving_median(y, win=31):
    y = np.asarray(y, float)
    win = int(win) | 1                      # force odd
    if win < 3:
        return y.copy()
    pad = win // 2
    yp = np.pad(y, pad, mode="reflect")
    sw = np.lib.stride_tricks.sliding_window_view(yp, win)
    return np.median(sw, axis=1)


# ------------------------------------------------------------------ impute
def _obs_index(t, mask):
    obs = np.where(mask)[0]
    return obs


def linear_impute(t, y, mask):
    y = np.asarray(y, float).copy()
    obs = _obs_index(t, mask)
    miss = np.where(~mask)[0]
    if len(miss) and len(obs) >= 2:
        y[miss] = np.interp(np.asarray(t)[miss], np.asarray(t)[obs], y[obs])
    return y


def pchip_impute(t, y, mask):
    """Shape-preserving cubic (PCHIP) interpolation -- monotone, no overshoot.

    A natural cubic spline is unstable here: over a multi-week gap in a noisy
    trending series it overshoots by >100 mm, which is a property of the
    interpolant, not of the data.
    """
    from scipy.interpolate import PchipInterpolator
    y = np.asarray(y, float).copy()
    t = np.asarray(t, float)
    obs = _obs_index(t, mask)
    miss = np.where(~mask)[0]
    if len(miss) and len(obs) >= 4:
        y[miss] = PchipInterpolator(t[obs], y[obs])(t[miss])
    elif len(miss) and len(obs) >= 2:
        y[miss] = np.interp(t[miss], t[obs], y[obs])
    return y


def ssa_impute(t, y, mask, L=None, n_components=10, n_iter=5):
    """Iterative SSA gap filling: interpolate, reconstruct, replace gaps, repeat.

    Five iterations is enough to converge here and keeps the cost bounded: the
    benchmark calls this once per (gap length x seed x component) per station.
    """
    y = np.asarray(y, float).copy()
    miss = np.where(~mask)[0]
    if len(miss) == 0:
        return y
    y[miss] = np.interp(np.asarray(t)[miss], np.asarray(t)[mask], y[mask])
    for _ in range(n_iter):
        rec = ssa_reconstruct(y, L=L, n_components=n_components)
        y[miss] = rec[miss]
    return y


# ------------------------------------------------------------------ detect
def detect_mle_offsets(t, y, max_offsets=6, thresh=10.0, orders=(1, 2)):
    """Offset detection from gnss_lib (robust fit + residual scan).

    Threshold 10 (not the library default 6) because the scan statistic on a
    flicker-noise residual is inflated by the long-range correlation: at 6 the
    detector returns ~12 spurious epochs on an offset-free series.
    """
    return list(gl.detect_offsets(t, y, max_offsets=max_offsets,
                                  thresh=thresh, orders=orders))


# ------------------------------------------------------------------ registry
DENOISERS = {
    "mle": lambda t, y, **k: mle_signal(t, y),
    "ssa": lambda t, y, **k: ssa_reconstruct(y),
    "wavelet": lambda t, y, **k: wavelet_denoise(y),
    "rts": lambda t, y, **k: rts_smooth(t, y),
    "median": lambda t, y, **k: moving_median(y),
    "none": lambda t, y, **k: np.asarray(y, float).copy(),
}

IMPUTERS = {
    "linear": lambda t, y, m, **k: linear_impute(t, y, m),
    "pchip": lambda t, y, m, **k: pchip_impute(t, y, m),
    "ssa": lambda t, y, m, **k: ssa_impute(t, y, m),
    "mle": lambda t, y, m, **k: _mle_impute(t, y, m),
}

DETECTORS = {
    "mle": lambda t, y, **k: detect_mle_offsets(t, y),
}


def _mle_impute(t, y, mask):
    """Fill gaps with the trajectory model fitted on the observed epochs."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    X, _ = gl.build_design(t, orders=(1, 2), poly=2)
    beta, *_ = gl.robust_fit(t[mask], y[mask], X[mask])
    out = y.copy()
    out[~mask] = X[~mask] @ beta
    return out
