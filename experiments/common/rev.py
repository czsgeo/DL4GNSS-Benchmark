"""rev.py -- shared helpers for the v2.0 revision experiments (reviews M1-M8).

Kept out of bench/dl so the v1.0 results (and their locked fingerprints) are not
touched. Nothing here changes an existing experiment's output.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dl as _dl  # noqa: E402


# ------------------------------------------------------------------ nets
def load_nets(nets_dir, pattern, device="cpu"):
    """{'<stem>': (seed, net)} for every checkpoint matching `pattern`."""
    import torch
    torch.backends.cudnn.enabled = False
    out = {}
    for p in sorted(glob.glob(os.path.join(nets_dir, pattern))):
        stem = os.path.basename(p)[:-3]
        ck = torch.load(p, map_location=device)
        in_ch = ck.get("in_ch", 2 if ck.get("task") == "impute" else 1)
        arch = ck.get("arch", "unet")
        if arch == "gru":
            net = _dl.build_gru(in_ch=in_ch, hidden=ck.get("gru_hidden", 64),
                                layers=ck.get("gru_layers", 2))
        elif arch == "tcn":
            import tcn as _tcn
            net = _tcn.build_tcn(in_ch=in_ch,
                                 channels=tuple(ck.get("channels", (40, 56, 72, 80, 80))))
        else:
            net = _dl.build_unet(in_ch=in_ch, base=ck.get("base", 12),
                                 depth=ck.get("depth", 3))
        net.load_state_dict(ck["state_dict"])
        net.to(device).eval()
        out[stem] = (ck.get("seed", -1), net)
    return out


def denoiser_of(net, device="cpu"):
    return lambda t, y: _dl.windowed_infer(net, y, device=device)


def imputer_of(net, device="cpu"):
    return lambda t, y, m: _dl.windowed_infer(net, y, mask=m, device=device)


def windowed_infer_perwin(net, series, mask=None, device="cpu", w=None, stride=None,
                          task="denoise", batch=64):
    """Overlap-add inference that standardises **each window by its own
    statistics**, which is exactly how the nets were trained (`dl.train_net`
    standardises every training window with `norm_stats(window)`), instead of
    `dl.windowed_infer`'s whole-series statistics. Used by review M2 to test
    whether the learned gap filler's in-support failure is a pipeline defect.

    Windows are normalised vectorised and pushed through the network in chunks
    of `batch`; this is numerically the same computation as the per-window loop
    (verified to <1e-5 mm) but several times faster on CPU.
    """
    import torch
    w = int(w or _dl.W)
    y = np.asarray(series, float)
    n = len(y)
    stride = stride or w // 2
    pad = w
    yp = np.pad(y, (pad, pad), mode="reflect")
    mp = None
    if mask is not None:
        mp = np.pad(np.asarray(mask, float), (pad, pad), mode="constant")

    starts, los, his = [], [], []
    for a in range(-pad, n + pad, stride):
        lo, hi = max(0, a), min(n, a + w)
        if hi <= lo:
            continue
        starts.append(a)
        los.append(lo)
        his.append(hi)
    if not starts:
        return np.zeros(n, float)
    segs = np.stack([yp[a + pad:a + pad + w] for a in starts])

    if task == "detect":
        means = np.zeros(len(starts))
        scales = np.full(len(starts), _dl.noise_scale(y))
        segn = segs / scales[:, None]
        sms = None
    elif mask is not None:
        sms = np.stack([mp[a + pad:a + pad + w] for a in starts])
        obs = sms > 0
        nobs = obs.sum(axis=1)
        use_mask = nobs > 2
        sums = np.where(obs, segs, 0.0).sum(axis=1)
        means = np.where(use_mask, sums / np.maximum(nobs, 1), segs.mean(axis=1))
        cent = np.where(obs, segs - means[:, None], 0.0)
        var = (cent ** 2).sum(axis=1) / np.maximum(nobs, 1)
        scales = np.where(use_mask, np.sqrt(var), segs.std(axis=1))
        scales = np.where(~np.isfinite(scales) | (scales < 1e-6), 1.0, scales)
        segn = (segs - means[:, None]) / scales[:, None]
    else:
        sms = None
        means = segs.mean(axis=1)
        scales = segs.std(axis=1)
        scales = np.where(~np.isfinite(scales) | (scales < 1e-6), 1.0, scales)
        segn = (segs - means[:, None]) / scales[:, None]

    nw = len(starts)
    outs = np.empty((nw, w), float)
    net.eval()
    with torch.no_grad():
        for i in range(0, nw, batch):
            sl = slice(i, min(i + batch, nw))
            if sms is not None:
                x = np.stack([segn[sl] * sms[sl], sms[sl]], axis=1)
            else:
                x = segn[sl][:, None, :]
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            outs[sl] = net(xt)[:, 0].detach().cpu().numpy()

    acc = np.zeros(n, float)
    cnt = np.zeros(n, float)
    for j, (a, lo, hi) in enumerate(zip(starts, los, his)):
        acc[lo:hi] += (outs[j] * scales[j] + means[j])[lo - a:hi - a]
        cnt[lo:hi] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


# ------------------------------------------------------------------ stats
def bootstrap_ci(x, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of `stat` over the sample `x`."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (float(stat(x)), float(stat(x)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def mean_std(x):
    x = np.asarray(x, float)
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)


def bin_index(values, n_bins=3):
    """Split a 1-D sample into `n_bins` equal-count bins; return bin labels."""
    v = np.asarray(values, float)
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(v, qs)


# ------------------------------------------------------------ detector curves
def classical_step_field(t, y, win_days=60, orders=(1, 2)):
    """The classical detector's decision statistic as a *field* over epochs.

    Same two-sample standardised step statistic `gnss_lib.detect_offsets` scores
    greedily, but computed in a single pass on the robust residual of the
    offset-free trajectory fit and returned for every candidate epoch, so a
    threshold sweep traces its precision-recall curve. Magnitude is in standard
    errors, directly comparable to the 10-SE threshold the v1.0 detector uses.
    """
    import gnss_lib as gl
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    X, _ = gl.build_design(t, orders=orders, poly=2)
    _, resid, _, _ = gl.robust_fit(t, y, X)
    n = len(t)
    stat = np.full(n, np.nan)
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
        stat[idx] = np.abs(mb - ma) / denom
    return stat


def peaks_from_field(t, field, thresh, min_sep_days=45):
    """Threshold + min-separation peak read-out, identical for both detectors."""
    from scipy.signal import find_peaks
    t = np.asarray(t, float)
    f = np.abs(np.asarray(field, float))
    f = np.where(np.isfinite(f), f, 0.0)
    dt = float(np.median(np.diff(t)))
    dist = max(1, int(round((min_sep_days / 365.25) / dt)))
    idx, _ = find_peaks(f, height=thresh, distance=dist)
    return [float(t[i]) for i in idx]


def pr_curve(fields, true_epochs, t, thresholds, tol_years, min_sep_days=45):
    """Precision/recall/F1 at every threshold for one realisation.

    `fields` : list of decision fields (one per training seed / one for the
    classical detector). The curve is the mean over the supplied fields, so a
    learned curve averages the training-seed variation. Returns arrays.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import bench
    P, R, F = [], [], []
    for th in thresholds:
        ps, rs, fs = [], [], []
        for fld in fields:
            pred = peaks_from_field(t, fld, th, min_sep_days)
            s = bench.detect_scores(true_epochs, pred, tol_years)
            ps.append(s["precision"]); rs.append(s["recall"]); fs.append(s["f1"])
        P.append(float(np.mean(ps))); R.append(float(np.mean(rs))); F.append(float(np.mean(fs)))
    return np.array(P), np.array(R), np.array(F)


def average_precision(precisions, recalls):
    """Area under the PR curve by the step-wise (VOC) rule, recall-ordered."""
    r = np.asarray(recalls, float); p = np.asarray(precisions, float)
    order = np.argsort(r)
    r, p = r[order], p[order]
    # precision envelope
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    ap = 0.0
    for i in range(1, len(r)):
        ap += (r[i] - r[i - 1]) * p[i]
    return float(ap)
