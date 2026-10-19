"""gnss_lib -- core routines for GNSS coordinate time-series analysis.

Contents
--------
* WGS84 XYZ -> geodetic (Bowring) and the local ENU rotation.
* Trajectory-model design matrices (bias, trend, annual/semiannual harmonics,
  offsets).
* Robust iterative least squares (Huber) with MAD-based outlier rejection.
* Simple change-point (offset) detection on the post-fit residual.
* Whittle (frequency-domain) maximum-likelihood estimation of power-law noise
  (white, flicker, random walk) -- the circulant approximation of the exact
  covariance (cf. Bos et al. 2008/2013), O(N log N) in numpy.
* GLS velocity uncertainty under a stationary coloured-noise covariance, also
  computed through the FFT (C^-1 acts diag(1/lambda) in the Fourier domain).
* Sliding-window harmonic estimation -> time series of annual amplitude/phase,
  used as the empirical "seasonal variability" estimator.
* Synthetic series generator (time-varying seasonal signal + power-law noise).

All time arguments are decimal years on a near-daily grid.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
DT_YEAR = 1.0 / 365.25          # one day, in years
CYCLE = 2.0 * np.pi

# numpy 2.x renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------- geometry
def geodetic_enu(x: float, y: float, z: float):
    """WGS84 XYZ -> (lat, lon) in radians and R such that R @ [dx,dy,dz] = [E,N,U]."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    th = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    sl, cl = np.sin(lat), np.cos(lat)
    slo, clo = np.sin(lon), np.cos(lon)
    R = np.array([[-slo, clo, 0.0],
                  [-sl * clo, -sl * slo, cl],
                  [cl * clo, cl * slo, sl]])
    return lat, lon, R


# ---------------------------------------------------------------- design
def build_design(t, orders=(1, 2), poly=1, offsets=(), t0=None):
    """Return (X, names). `t` in decimal years; `offsets` are decimal-year epochs."""
    t = np.asarray(t, float)
    if t0 is None:
        t0 = float(np.mean(t))
    cols, names = [], []
    if poly >= 1:
        cols.append(np.ones_like(t)); names.append("bias")
    if poly >= 2:
        cols.append(t - t0); names.append("trend")
    if poly >= 3:
        cols.append((t - t0) ** 2); names.append("quad")
    for k in orders:
        cols.append(np.sin(CYCLE * k * (t - t0))); names.append(f"sin{k}")
        cols.append(np.cos(CYCLE * k * (t - t0))); names.append(f"cos{k}")
    for oj in offsets:
        cols.append((t >= oj).astype(float)); names.append(f"off{oj:.4f}")
    return np.vstack(cols).T, names


def robust_fit(t, y, X, n_iter=6, huber_k=1.5, mad_k=4.0):
    """Iteratively reweighted least squares with Huber weights + MAD outlier cut.

    Returns (beta, resid, weights, outlier_mask).
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    w = np.ones_like(y)
    beta = np.zeros(X.shape[1])
    resid = y.copy()
    out = np.zeros(len(y), bool)
    for it in range(n_iter):
        Xw = X * w[:, None]
        try:
            beta, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
        except np.linalg.LinAlgError:
            break
        resid = y - X @ beta
        s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        s = s if s > 0 else np.std(resid) or 1.0
        # A denoised series that lies in the design space leaves residuals at
        # floating-point level. Without a floor the MAD scale collapses, every
        # point is cut as an outlier, and the next all-zero weighted system
        # returns beta=0. Floor the scale at 1e-9 of the data RMS.
        s_floor = 1e-9 * (float(np.sqrt(np.mean(y * y))) or 1.0)
        s = max(s, s_floor)
        r = np.abs(resid) / s
        out = r > mad_k
        w = np.where(out, 0.0, np.where(r <= huber_k, 1.0, huber_k / np.maximum(r, 1e-9)))
    return beta, resid, w, out


def detect_offsets(t, y, max_offsets=12, win_days=60, thresh=6.0, orders=(1, 2)):
    """Greedy change-point search on the robust residual. Returns list of epochs."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    offsets: list[float] = []
    for _ in range(max_offsets):
        X, _ = build_design(t, orders=orders, poly=2, offsets=offsets)
        _, resid, _, out = robust_fit(t, y, X)
        # Two-sample statistic at every candidate epoch, evaluated vectorised
        # (the scalar slice loop was O(n) Python and dominated the whole
        # benchmark). Valid i in [win_days, n - win_days]; both windows are the
        # same length win_days, so the pooled-denominator form is unchanged.
        n = len(t)
        cand, best = None, 0.0
        if n >= 2 * win_days + 1:
            idx = np.arange(win_days, n - win_days + 1)
            cs = np.concatenate(([0.0], np.cumsum(resid)))
            cs2 = np.concatenate(([0.0], np.cumsum(resid * resid)))
            ma = (cs[idx] - cs[idx - win_days]) / win_days
            mb = (cs[idx + win_days] - cs[idx]) / win_days
            va = (cs2[idx] - cs2[idx - win_days]) / win_days - ma * ma
            vb = (cs2[idx + win_days] - cs2[idx]) / win_days - mb * mb
            sa = np.sqrt(np.maximum(va, 0.0)); sb = np.sqrt(np.maximum(vb, 0.0))
            denom = np.sqrt((sa * sa + sb * sb) / win_days)
            denom = np.where(denom < 1e-12, 1.0, denom)
            stat = np.abs(mb - ma) / denom
            # suppress a neighbourhood around offsets already accepted, so the
            # greedy search returns distinct epochs instead of re-picking the
            # same step (which would inflate the false-positive count).
            for oe in offsets:
                kp = int(np.argmin(np.abs(t - oe))) - win_days   # -> stat-array position
                stat[max(0, kp - win_days):kp + win_days + 1] = -np.inf
            j = int(np.argmax(stat))
            best = float(stat[j]); cand = float(t[idx[j]])
        if cand is None or best < thresh:
            break
        offsets.append(float(cand))
    offsets.sort()
    return offsets


# ---------------------------------------------------------------- noise (Whittle MLE)
def _psd(f, params, models, f1):
    """Two-sided PSD S(f) [m^2 * year] for the requested noise models."""
    S = np.zeros_like(f)
    j = 0
    for m in models:
        if m == "wn":
            S = S + params[j] * np.ones_like(f); j += 1
        elif m == "fn":
            S = S + params[j] * (f1 / np.maximum(f, 1e-12)); j += 1
        elif m == "rw":
            S = S + params[j] * (f1 / np.maximum(f, 1e-12)) ** 2; j += 1
    return S


def _freq_grid(n, dt):
    f = np.fft.rfftfreq(n, d=dt)      # cycles per year
    return f


def whittle_neg2ll(logp, f, power, models, f1, dt):
    # circulating-covariance eigenvalues are lambda_k = S(f_k) / dt
    params = np.exp(logp)
    lam = _psd(f[1:], params, models, f1) / dt
    pw = power[1:]
    return float(np.sum(np.log(lam) + pw / lam))


def whittle_mle(resid, dt=DT_YEAR, models=("wn", "fn")):
    """Estimate power-law noise by minimizing the Whittle likelihood.

    `resid` : detrended residual (mean removed).
    Returns dict with std deviations (m) per model component and fitted PSD
    parameters (two-sided PSD in m^2 * year).
    """
    from scipy.optimize import minimize

    resid = np.asarray(resid, float)
    n = len(resid)
    r = resid - resid.mean()
    F = np.fft.rfft(r)
    power = (np.abs(F) ** 2) / n        # |R_k|^2 for the unitary DFT
    f = _freq_grid(n, dt)
    f1 = 1.0 / (n * dt)
    nt = len(models)
    best = None
    # Multi-start anchored on the periodogram. The pure-WN MLE level
    # S ~ dt*mean(power) sets the scale, so the start grid adapts to whatever
    # unit the residuals are in (metre or millimetre) instead of hard-coding a
    # range; a few log-spaced starts around it reach the global optimum at a
    # fraction of the cost of a fixed 22-decade sweep. The WN+FN Whittle
    # surface is multimodal -- a single narrow start locks onto the WN-only
    # local optimum whenever the flicker term is large.
    scale = dt * float(np.mean(power[1:]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    bounds = [(-80.0, 40.0)] * nt   # exp() stays finite across the whole range
    for a0 in scale * np.array([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]):
        x0 = np.log(np.full(nt, a0, float))
        res = minimize(whittle_neg2ll, x0, args=(f, power, models, f1, dt),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res
    params = np.exp(best.x)

    fN = 1.0 / (2.0 * dt)
    ff = np.linspace(f1, fN, 4000)
    out = {"models": list(models), "params": params.tolist(),
           "neg2ll": float(best.fun)}
    j = 0
    for m in models:
        col = np.zeros(len(models))
        col[j] = 1.0
        v = 2.0 * _trapz(_psd(ff, params * col, models, f1), ff)
        out[f"sigma_{m}"] = float(np.sqrt(max(v, 0.0)))
        j += 1
    out["sigma_total"] = float(np.sqrt(sum(out[f"sigma_{m}"] ** 2 for m in models)))
    out["_psd_params"] = params
    return out


# ---------------------------------------------------------------- GLS velocity uncertainty
def gls_velocity_sigma(t, X, vel_col, psd_params, models, dt=DT_YEAR):
    """GLS std of the velocity coefficient under a stationary covariance whose
    inverse is diag(1/lambda) in the Fourier domain (circulant approximation),
    with covariance eigenvalues lambda_k = S(f_k)/dt."""
    n = len(t)
    f = _freq_grid(n, dt)
    f1 = 1.0 / (n * dt)
    S = _psd(f, np.asarray(psd_params, float), models, f1)
    lam = S / dt
    lam = np.where(lam <= 0, np.min(lam[lam > 0]), lam)
    # weights for real FFT reconstruction: 1 for DC and Nyquist else 2
    wt = np.full_like(f, 2.0)
    wt[0] = 1.0
    if n % 2 == 0:
        wt[-1] = 1.0
    A = np.zeros((X.shape[1], X.shape[1]))
    Xh = [np.fft.rfft(X[:, k]) for k in range(X.shape[1])]
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            val = np.sum(wt * np.conj(Xh[i]) * Xh[j] / lam) / n
            A[i, j] = val.real
            A[j, i] = val.real
    Ainv = np.linalg.pinv(A)
    return float(np.sqrt(max(Ainv[vel_col, vel_col], 0.0))), A, Ainv


def white_velocity_sigma(t, X, vel_col, sigma_w):
    XtX = X.T @ X
    return float(sigma_w * np.sqrt(np.linalg.pinv(XtX)[vel_col, vel_col]))


def _autocov_wn_fn(n, dt, S_wn, S_fn):
    """Exact autocovariance of the discretely-sampled WN+FN process whose
    two-sided PSD is S(f)=S_wn + S_fn*(f1/f) on the DFT frequency grid.

    gamma(m) = f1 * sum_k S(f_k) exp(2 pi i k m / n)  (inverse DFT of the
    two-sided discrete spectrum). This is the exact covariance of
    `noise_from_psd`, hence consistent with `whittle_mle` and with the
    circulant eigenvalues lambda_k = S(f_k)/dt used by `gls_velocity_sigma`.
    """
    f = np.fft.rfftfreq(n, d=dt)
    f1 = 1.0 / (n * dt)
    Sr = np.full(f.shape, float(S_wn))
    if S_fn > 0:
        Sr[1:] = Sr[1:] + S_fn * (f1 / np.maximum(f[1:], 1e-12))
    Sr = np.maximum(Sr, 0.0)
    nf = n // 2 + 1
    Sfull = np.empty(n)
    Sfull[:nf] = Sr
    Sfull[nf:] = Sr[1:n - nf + 1][::-1]
    return f1 * n * np.real(np.fft.ifft(Sfull))


def build_cov_wn_fn(n, dt, S_wn, S_fn):
    """Toeplitz covariance matrix of WN+FN noise (n x n)."""
    from scipy.linalg import toeplitz
    return toeplitz(_autocov_wn_fn(n, dt, S_wn, S_fn))


def gls_velocity_sigma_exact(t, X, vel_col, S_wn, S_fn, dt=DT_YEAR):
    """Exact (non-circulant) GLS std of the velocity coefficient under a
    stationary WN+FN covariance built from the analytic autocovariance.

    O(N^3) via Cholesky -- call once per design, not per MC realisation.
    Returns (sigma, C, XtCiX, Ainv).
    """
    from scipy.linalg import cho_factor, cho_solve
    n = len(t)
    C = build_cov_wn_fn(n, dt, S_wn, S_fn)
    cf = cho_factor(C, lower=True, check_finite=False)
    CiX = cho_solve(cf, X, check_finite=False)
    XtCiX = X.T @ CiX
    Ainv = np.linalg.pinv(XtCiX)
    return float(np.sqrt(max(Ainv[vel_col, vel_col], 0.0))), C, XtCiX, Ainv


def ols_velocity_sigma_under_cov(t, X, vel_col, C):
    """Std of the *OLS* velocity coefficient under a general covariance C
    (sandwich form (X'X)^-1 X'CX (X'X)^-1)."""
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    XtCX = X.T @ (C @ X)
    V = XtXi @ XtCX @ XtXi
    return float(np.sqrt(max(V[vel_col, vel_col], 0.0)))


# ---------------------------------------------------------------- sliding-window seasonality
def sliding_seasonal(t, y, win_years=3.0, step_years=0.1, orders=(1, 2), min_pts=200):
    """Fit [bias, trend, annual, semiannual] in a sliding window.

    Returns (tc, amp1, amp2) arrays: window-centre times and the annual /
    semiannual amplitudes (sqrt(a^2+b^2)).
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    t0, t1 = t.min(), t.max()
    centers = np.arange(t0 + win_years / 2, t1 - win_years / 2 + 1e-9, step_years)
    tc, a1, a2 = [], [], []
    for c in centers:
        m = (t >= c - win_years / 2) & (t <= c + win_years / 2)
        if m.sum() < min_pts:
            continue
        X, names = build_design(t[m], orders=orders, poly=2, t0=c)
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        def amp(k):
            i = names.index(f"sin{k}")
            return float(np.hypot(beta[i], beta[i + 1]))
        tc.append(c); a1.append(amp(1)); a2.append(amp(2))
    return np.array(tc), np.array(a1), np.array(a2)


def seasonal_variability(tc, amp):
    """Relative variability of a seasonal amplitude: std/|mean| (unitless)."""
    amp = np.asarray(amp, float)
    mu = np.mean(amp)
    if mu <= 0:
        return np.nan
    return float(np.std(amp, ddof=1) / mu)


# ---------------------------------------------------------------- time-varying harmonic SSM
def _ssm_matrices(t, orders, t0, q_h):
    """Observation matrix H (n x m) and per-step process covariance Q (n x m x m).

    State  x = [bias, trend, sin1, cos1, sin2, cos2, ...]:
      * bias, trend      : constant   (no process noise)
      * harmonic coeffs  : random walk, process variance q_h per year
    The random-walk block grows linearly with the elapsed time interval, so
    the model is correct on the irregular daily grid produced by preprocessing.
    """
    t = np.asarray(t, float)
    n = len(t)
    cols = [np.ones_like(t), (t - t0)]
    for k in orders:
        cols.append(np.sin(CYCLE * k * (t - t0)))
        cols.append(np.cos(CYCLE * k * (t - t0)))
    H = np.vstack(cols).T                      # (n, m)
    m = H.shape[1]
    n_harm = m - 2
    Q = np.zeros((n, m, m))
    dt_prev = np.diff(t, prepend=t[0])
    for k in range(1, n):
        blk = q_h * dt_prev[k]                 # random-walk variance over dt
        for j in range(n_harm):
            Q[k, 2 + j, 2 + j] = blk
    return H, Q


def _kalman(y, H, Q, R, F=None, x0=None, P0=None):
    """Kalman filter with a Joseph-form covariance update (numerically PSD-safe).

    The initial state is the constant-seasonal OLS fit and the initial
    covariance its OLS parameter covariance (both supplied by the caller), so
    no ill-conditioned diffuse prior is involved: the filter simply starts from
    a well-scaled constant solution and lets the random walk adapt. Returns the
    filtered and one-step-predicted moments plus the prediction-error
    log-likelihood.
    """
    n, m = H.shape
    F = np.eye(m) if F is None else F
    xp = np.zeros((n, m)); Pp = np.zeros((n, m, m))
    xf = np.zeros((n, m)); Pf = np.zeros((n, m, m))
    x = np.zeros(m) if x0 is None else np.array(x0, float).copy()
    P = np.eye(m) * R if P0 is None else np.array(P0, float).copy()
    ll = 0.0
    I = np.eye(m)
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q[k]
        xp[k] = x; Pp[k] = P
        Hk = H[k:k + 1]
        S = (Hk @ P @ Hk.T).item() + R
        S = max(S, 1e-300)
        K = (P @ Hk.T) / S
        v = y[k] - (Hk @ x).item()
        x = x + (K * v).ravel()
        A = I - K @ Hk                        # Joseph form keeps P PSD
        P = A @ P @ A.T + (K @ K.T) * R
        P = 0.5 * (P + P.T)
        xf[k] = x; Pf[k] = P
        ll += -0.5 * (np.log(S) + v * v / S)
    return xf, Pf, xp, Pp, ll


def ssm_loglik(y, H, Q, R, F=None, x0=None, P0=None):
    """Kalman-filter prediction-error log-likelihood (constants omitted)."""
    return _kalman(y, H, Q, R, F, x0, P0)[4]


def ssm_filter_smoother(y, H, Q, R, F=None, x0=None, P0=None):
    """Kalman filter + Rauch-Tung-Striebel smoother.

    Returns (xs, Ps, ll): smoothed state means (n x m), smoothed covariances
    (n x m x m), and the filter log-likelihood.
    """
    n, m = H.shape
    F = np.eye(m) if F is None else F
    xf, Pf, xp, Pp, ll = _kalman(y, H, Q, R, F, x0, P0)
    xs = xf.copy(); Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        J = Pf[k] @ F.T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xf[k] + J @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + J @ (Ps[k + 1] - Pp[k + 1]) @ J.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
    return xs, Ps, ll


def tv_harmonic_ssm(t, y, orders=(1, 2), R=None, q_h=None, q_grid=None, t0=None):
    """Time-varying annual/semiannual amplitude by a state-space model.

    The harmonic coefficients of the standard trajectory model are allowed to
    follow a random walk (process variance `q_h` per year); `bias` and `trend`
    stay constant. The latent trajectory is recovered with the Kalman filter
    and RTS smoother, giving a continuous amplitude curve a_k(t) =
    sqrt(c_k^2 + s_k^2) with its smoothed standard error (delta method).

    If `R` is None it is set from the MAD of a robust constant-seasonal fit's
    residual (a white proxy for the observation noise). If `q_h` is None it is
    chosen by a 1-D search over `q_grid` maximising the filter log-likelihood
    (an empirical-Bayes choice of the smoothness/adaptivity trade-off).

    Returns dict: t, amp1, amp2, se1, se2, R, q_h, loglik.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    if t0 is None:
        t0 = float(np.mean(t))
    X, _ = build_design(t, orders=orders, poly=2)
    XtXi = np.linalg.pinv(X.T @ X)
    beta_ols = XtXi @ (X.T @ y)
    if R is None:
        _, resid, _, _ = robust_fit(t, y, X)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        R = float(mad ** 2) if mad > 0 else float(np.var(resid))
    x0 = beta_ols
    P0 = XtXi * R                       # OLS parameter covariance (white noise)

    def amps_of(xs, Ps):
        a1 = np.hypot(xs[:, 2], xs[:, 3])
        a2 = np.hypot(xs[:, 4], xs[:, 5])
        se = np.zeros(len(a1)); se2 = np.zeros(len(a1))
        for k in range(len(a1)):
            g1 = np.zeros(xs.shape[1]); g2 = np.zeros(xs.shape[1])
            if a1[k] > 0:
                g1[2] = xs[k, 2] / a1[k]; g1[3] = xs[k, 3] / a1[k]
                se[k] = float(np.sqrt(max(g1 @ Ps[k] @ g1, 0.0)))
            if a2[k] > 0:
                g2[4] = xs[k, 4] / a2[k]; g2[5] = xs[k, 5] / a2[k]
                se2[k] = float(np.sqrt(max(g2 @ Ps[k] @ g2, 0.0)))
        return a1, a2, se, se2

    if q_h is not None:
        H, Q = _ssm_matrices(t, orders, t0, q_h)
        xs, Ps, ll = ssm_filter_smoother(y, H, Q, R, x0=x0, P0=P0)
        a1, a2, se1, se2 = amps_of(xs, Ps)
        return {"t": t, "amp1": a1, "amp2": a2, "se1": se1, "se2": se2,
                "R": R, "q_h": float(q_h), "loglik": float(ll)}

    if q_grid is None:
        q_grid = np.logspace(-12, -3, 19)
    best = None
    for q in q_grid:
        H, Q = _ssm_matrices(t, orders, t0, float(q))
        ll = ssm_loglik(y, H, Q, R, x0=x0, P0=P0)
        if best is None or ll > best[1]:
            best = (float(q), ll)
    q_h = best[0]
    H, Q = _ssm_matrices(t, orders, t0, q_h)
    xs, Ps, ll = ssm_filter_smoother(y, H, Q, R, x0=x0, P0=P0)
    a1, a2, se1, se2 = amps_of(xs, Ps)
    return {"t": t, "amp1": a1, "amp2": a2, "se1": se1, "se2": se2,
            "R": R, "q_h": q_h, "loglik": float(ll)}


# ---------------------------------------------------------------- synthetic generator
def powerlaw_noise(n, sigma, kappa, dt=DT_YEAR, rng=None):
    """Generate stationary power-law noise with spectral index kappa (0,1,2)
    and a target std `sigma` on an n-point series."""
    rng = rng or np.random.default_rng(0)
    f = np.fft.rfftfreq(n, d=dt)
    f1 = 1.0 / (n * dt)
    shape = np.ones_like(f)
    shape[1:] = (f1 / f[1:]) ** (kappa / 2.0)
    shape[0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, len(f))
    Z = shape * np.exp(1j * ph)
    x = np.fft.irfft(Z, n=n)
    x = x - x.mean()
    if x.std() > 0:
        x = x / x.std() * sigma
    return x


def noise_from_psd(n, S_wn, S_fn, dt=DT_YEAR, rng=None):
    """Zero-mean stationary noise whose two-sided PSD is
    S(f) = S_wn + S_fn * (f1/f)  [m^2 * year]  on an n-point grid of spacing dt.

    The construction sets E|X_k|^2 = n * S(f_k) / dt for the unnormalised DFT,
    which is exactly the convention used by `whittle_mle` and
    `gls_velocity_sigma` -- so an MLE run on the output returns S_wn, S_fn and
    the GLS velocity variance it predicts matches the Monte-Carlo scatter.
    """
    rng = rng or np.random.default_rng(0)
    f = np.fft.rfftfreq(n, d=dt)
    f1 = 1.0 / (n * dt)
    S = np.full(f.shape, float(S_wn))
    if S_fn > 0:
        S[1:] = S[1:] + S_fn * (f1 / np.maximum(f[1:], 1e-12))
    S = np.maximum(S, 0.0)
    amp = np.sqrt(n * S / dt)
    ph = rng.uniform(0.0, 2 * np.pi, f.shape)
    X = amp * np.exp(1j * ph)
    X[0] = 0.0                       # remove the mean
    if n % 2 == 0:
        X[-1] = np.real(X[-1])       # Nyquist bin must be real
    x = np.fft.irfft(X, n=n)
    return x - x.mean()


def synth_series(n=3650, sigma_w=0.0015, sigma_f=0.004, trend=0.002,
                 A1=0.004, A2=0.0015, mod_std=0.0, seed=0,
                 t_start=2000.0, dt=DT_YEAR):
    """Build a daily series: trend + time-varying annual + semiannual + WN + FN.

    The annual amplitude is modulated as A1*(1 + m(t)) with a smooth (2-year
    low-pass) unit-std random function scaled by `mod_std`.
    """
    rng = np.random.default_rng(seed)
    t = t_start + np.arange(n) * dt
    x = trend * (t - t_start) + A1 * np.sin(CYCLE * (t - t_start)) \
        + A2 * np.sin(2 * CYCLE * (t - t_start) + 0.7)
    if mod_std > 0:
        # smooth unit-variance modulation via low-pass filtering of white noise
        z = rng.standard_normal(n)
        Z = np.fft.rfft(z)
        f = np.fft.rfftfreq(n, d=dt)
        Z[f > 0.5] = 0.0        # keep periods >= 2 years
        m = np.fft.irfft(Z, n=n)
        m = m / m.std()
        x = x + A1 * mod_std * m * np.sin(CYCLE * (t - t_start))
    if sigma_w > 0:
        x = x + rng.standard_normal(n) * sigma_w
    if sigma_f > 0:
        x = x + powerlaw_noise(n, sigma_f, 1.0, dt, rng)
    return t, x
