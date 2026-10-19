"""bench.py -- shared injection + evaluation harness for the three benchmark
tasks (denoise / impute / detect), used by every method family (classical and
DL) so the comparison is apples-to-apples.

Ground truth is exact by controlled injection. Two regimes, both reported:

* **hybrid**  : a real station series supplies the *deterministic* signal
  (trend + amplitude-modulated annual + semiannual + offsets + transients,
  recovered by a robust trajectory fit of the real series); known WN+FN+RW
  noise is then injected on top. The truth is the deterministic part, so
  signal-recovery error is exact while the signal shape stays realistic.
* **synthetic**: the same signal family with prescribed amplitudes.

The signal deliberately contains amplitude modulation and transients, which
the constant-seasonal trajectory model (a standard classical tool) cannot fit
exactly -- otherwise the model that defines the truth would be the oracle.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_lib as gl  # noqa: E402
import synth  # noqa: E402

DT = gl.DT_YEAR
CYCLE = 2.0 * np.pi


# ------------------------------------------------------------------ signal
def clean_signal(t, p, t0=None):
    """Deterministic signal with amplitude-modulated annual + semiannual."""
    t = np.asarray(t, float)
    t0 = float(t[0]) if t0 is None else t0
    tt = t - t0
    y = p["trend"] * tt
    env = 1.0 + p.get("am", 0.0) * np.sin(CYCLE * p.get("fm", 0.09) * tt + p.get("amph", 0.0))
    y = y + p.get("A1", 0.0) * env * np.sin(CYCLE * tt + p.get("ph1", 0.0))
    y = y + p.get("A2", 0.0) * np.sin(2 * CYCLE * tt + p.get("ph2", 0.7))
    for e, a in p.get("steps", []):
        y = y + a * (t >= e)
    for (e, a, tau) in p.get("transients", []):
        d = t - e
        y = y + a * np.exp(-np.maximum(d, 0.0) / tau) * (d >= 0)
    # logarithmic post-seismic relaxation; absent from the constant-harmonic
    # trajectory basis (used by the out-of-family truth of Sec.~\ref{...})
    for (e, a, tau) in p.get("logdecay", []):
        d = t - e
        y = y + a * np.log1p(np.maximum(d, 0.0) / tau) * (d >= 0)
    return y


def signal_params_from_real(t, y, seed=0):
    """Realistic signal parameters fit from a real series (robust trajectory)."""
    X, _ = gl.build_design(t, orders=(1, 2), poly=2)
    beta, *_ = gl.robust_fit(t, y, X)
    # beta = [bias, trend, sin1, cos1, sin2, cos2]
    a1 = float(np.hypot(beta[2], beta[3]))
    a2 = float(np.hypot(beta[4], beta[5]))
    rng = np.random.default_rng(seed)
    span = float(t[-1] - t[0])
    return {"trend": float(beta[1]), "A1": a1, "A2": a2,
            "am": 0.35, "fm": 0.12, "amph": float(rng.uniform(0, CYCLE)),
            "ph1": 0.0, "ph2": float(rng.uniform(0, CYCLE)), "steps": [], "transients": []}


def signal_params_out_of_family(t, y_real, seed=0, am_lo=0.6, am_hi=1.0,
                                tau_lo=0.3, tau_hi=1.5):
    """Station-like signal parameters whose deterministic shape leaves the
    constant-harmonic trajectory basis (review point M1).

    Starts from the same robust trajectory fit as `signal_params_from_real`
    (same trend and mean harmonic amplitudes, so the series stays station-like)
    but adds structures the trajectory model cannot represent: a **strong**
    multi-year amplitude modulation of the annual term (Klos-type non-stationary
    seasonality), one or two long-lived exponential transients with
    station-specific decay times, and a logarithmic post-seismic relaxation.
    The truth is this deterministic signal, known exactly.
    """
    X, _ = gl.build_design(t, orders=(1, 2), poly=2)
    beta, *_ = gl.robust_fit(t, y_real, X)
    a1 = float(np.hypot(beta[2], beta[3]))
    a2 = float(np.hypot(beta[4], beta[5]))
    rng = np.random.default_rng(seed)
    span = float(t[-1] - t[0])
    t0 = float(t[0])

    def frac(f):
        return t0 + f * span

    transients = []
    for _ in range(int(rng.integers(1, 3))):
        transients.append((frac(float(rng.uniform(0.2, 0.75))),
                           float(rng.uniform(-15, 15)),
                           float(rng.uniform(tau_lo, tau_hi))))
    logdecay = []
    if rng.random() < 0.6:
        logdecay.append((frac(float(rng.uniform(0.15, 0.6))),
                         float(rng.uniform(-20, 20)),
                         float(rng.uniform(0.02, 0.08))))
    return {"trend": float(beta[1]), "A1": a1, "A2": a2,
            "am": float(rng.uniform(am_lo, am_hi)),
            "fm": float(rng.uniform(0.06, 0.18)),
            "amph": float(rng.uniform(0, CYCLE)),
            "ph1": 0.0, "ph2": float(rng.uniform(0, CYCLE)),
            "steps": [], "transients": transients, "logdecay": logdecay}


# ------------------------------------------------------------------ noise
def make_noise_spec(sigma_wn, sigma_fn, sigma_rw, n, dt=DT):
    return synth.make_spec(sigma_wn, sigma_fn, sigma_rw, n, dt, seed=12345)


def inject(n, nspec, dt=DT, rng=None):
    return synth.noise_from_spec(n, nspec, dt, rng or np.random.default_rng(0))


# ------------------------------------------------------------------ gaps
def random_gaps(n, gap_len, n_gaps, rng, margin=30):
    """Boolean observed-mask with `n_gaps` contiguous gaps of length `gap_len`."""
    obs = np.ones(n, bool)
    L = int(gap_len)
    hi = max(margin + 1, n - L - margin)
    starts = rng.integers(margin, hi, size=n_gaps)
    for s in starts:
        obs[s:s + L] = False
    return obs


def random_epochs(t, n, lo=0.15, hi=0.85, min_sep_years=1.5, rng=None):
    """`n` offset epochs inside the [lo, hi] fraction of the span, at least
    `min_sep_years` apart. Keeps the detection task well posed: two offsets
    closer than the match tolerance are unidentifiable by construction, and both
    the greedy classical detector and the DL read-out assume well-separated
    events."""
    t = np.asarray(t, float)
    rng = rng or np.random.default_rng(0)
    t0, span = float(t[0]), float(t[-1] - t[0])
    out = []
    tries = 0
    while len(out) < n and tries < 1000:
        c = float(t0 + rng.uniform(lo, hi) * span)
        if all(abs(c - x) >= min_sep_years for x in out):
            out.append(c)
        tries += 1
    return sorted(out)


# ------------------------------------------------------------------ metrics
def rmse(a, b, mask=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if mask is not None:
        a = a[mask]; b = b[mask]
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def detect_scores(true_epochs, pred_epochs, tol_years):
    """Greedy match each prediction to the nearest unmatched true epoch."""
    tp = 0
    used = np.zeros(len(true_epochs), bool)
    te = np.asarray(true_epochs, float)
    for pe in pred_epochs:
        if len(te) == 0:
            break
        d = np.abs(te - pe)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_years:
            used[j] = True
            tp += 1
    fp = len(pred_epochs) - tp
    fn = len(true_epochs) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


# ------------------------------------------------------------------ task runners
def run_denoise(t, y, S, methods):
    """Return {method: {'rmse':..., 'rmse_reduction':...}} plus the noise floor."""
    base = rmse(y, S)                     # = std(injected noise), the no-denoise error
    out = {}
    for name, fn in methods.items():
        y_hat = fn(t, y)
        r = rmse(y_hat, S)
        out[name] = {"rmse": r,
                     "rmse_reduction": float(1.0 - r / base) if base > 0 else float("nan")}
    return out, base


def run_impute(t, y, gap_len, n_gaps, rng, methods, S=None):
    """Mask gaps, impute, score against the removed true values.

    `rmse` is against the withheld observation `y` (what the station recorded,
    noise included). When the noise-free signal `S` is supplied (review m9) we
    also return `rmse_signal`, the error against the signal itself, so the two
    can be compared: a filler that reproduces part of the correlated noise looks
    better against `y` than against `S`, one that recovers the underlying signal
    looks better against `S`.
    """
    obs = random_gaps(len(t), gap_len, n_gaps, rng)
    miss = ~obs
    if miss.sum() == 0:
        return {}
    out = {}
    for name, fn in methods.items():
        y_hat = fn(t, y, obs)
        rec = {"rmse": rmse(y_hat, y, mask=miss)}
        if S is not None:
            rec["rmse_signal"] = rmse(y_hat, S, mask=miss)
        out[name] = rec
    return out


def run_detect(t, y, true_epochs, methods, tol_years):
    out = {}
    for name, fn in methods.items():
        pred = fn(t, y)
        out[name] = detect_scores(true_epochs, pred, tol_years)
    return out
