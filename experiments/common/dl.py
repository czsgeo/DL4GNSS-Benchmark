"""dl.py -- compact 1-D networks for the three benchmark tasks, plus the
on-the-fly training-data generator they share with the classical harness.

The benchmark is a *controlled* comparison, so the nets are kept small
(<= ~300 k parameters), fixed-seed, and trained on the same injected
signal/noise family the classical methods are scored on -- and then evaluated
on the held-out real test stations (a deliberate synthetic->real domain shift).

Window convention: every net maps a length-``W`` window to the same length. A
long series is processed window-by-window with 50 % overlap and averaged
(``windowed_infer``).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

DT = bench.DT
W = 1024                                    # training window length (days)
AVG_WIN = 181                               # detection high-pass moving-average window (days)


# ------------------------------------------------------------------ data
def random_signal_params(rng, with_steps=True, with_transients=True, family="in"):
    """Steps/transients are placed by *fraction* of the window (0-1); the caller
    converts them to absolute epochs once `t` is known.

    ``family='oof'`` widens the deterministic shape past the constant-harmonic
    trajectory basis: a strong multi-year amplitude modulation (Klos-type
    non-stationary seasonality), long-lived transients with station-specific
    decay times, and a logarithmic relaxation (review point M1). It is used to
    train the matched out-of-family denoiser of `exp-20261009-truth-families`.
    """
    if family == "oof":
        p = {"trend": float(rng.uniform(-30, 30)),
             "A1": float(rng.uniform(1.0, 10.0)),
             "A2": float(rng.uniform(0.5, 4.0)),
             "am": float(rng.uniform(0.6, 1.1)),
             "fm": float(rng.uniform(0.06, 0.18)),
             "amph": float(rng.uniform(0, 2 * np.pi)),
             "ph1": float(rng.uniform(0, 2 * np.pi)),
             "ph2": float(rng.uniform(0, 2 * np.pi)),
             "steps": [], "logdecay": []}
        p["transients"] = []
        for _ in range(int(rng.integers(1, 3))):
            p["transients"].append((float(rng.uniform(0.2, 0.75)),
                                    float(rng.uniform(-15, 15)),
                                    float(rng.uniform(0.3, 1.5))))
        if rng.random() < 0.6:
            p["logdecay"].append((float(rng.uniform(0.15, 0.6)),
                                  float(rng.uniform(-20, 20)),
                                  float(rng.uniform(0.02, 0.08))))
        return p
    p = {"trend": float(rng.uniform(-30, 30)),
         "A1": float(rng.uniform(1.0, 10.0)),
         "A2": float(rng.uniform(0.5, 4.0)),
         "am": float(rng.uniform(0.0, 0.5)),
         "fm": float(rng.uniform(0.05, 0.2)),
         "amph": float(rng.uniform(0, 2 * np.pi)),
         "ph1": float(rng.uniform(0, 2 * np.pi)),
         "ph2": float(rng.uniform(0, 2 * np.pi)),
         "steps": [], "transients": []}
    if with_steps:
        for _ in range(int(rng.integers(0, 3))):
            p["steps"].append((float(rng.uniform(0.2, 0.8)), float(rng.uniform(-12, 12))))
    if with_transients:
        for _ in range(int(rng.integers(0, 2))):
            p["transients"].append((float(rng.uniform(0.3, 0.8)),
                                    float(rng.uniform(-10, 10)), float(rng.uniform(0.02, 0.15))))
    return p


def _abs_epochs(p, t):
    """Convert fractional step/transient locations to absolute epochs."""
    t0, span = float(t[0]), float(t[-1] - t[0])
    p = dict(p)
    p["steps"] = [(t0 + f * span, a) for f, a in p["steps"]]
    p["transients"] = [(t0 + f * span, a, tau) for f, a, tau in p["transients"]]
    return p


def random_noise_spec(rng, n):
    return bench.make_noise_spec(float(rng.uniform(0.5, 3.0)),
                                 float(rng.uniform(0.0, 5.0)),
                                 float(rng.uniform(0.0, 0.5)), n, DT)


def make_sample(n, rng, task, family="in"):
    """Return (t, y, target, aux) for one training window.

    denoise : target = clean signal S                     ; aux = None
    impute  : target = noisy y                            ; aux = observed mask (bool)
    detect  : target = Gaussian impulse at each offset    ; aux = None
    """
    t = np.arange(n) * DT
    if task == "detect":
        p = _abs_epochs(random_signal_params(rng, with_steps=True, with_transients=False), t)
        S = bench.clean_signal(t, p)
        tar = np.zeros(n)
        # Bump width sets the detector's localisation granularity. A ~3-day
        # spike is destroyed by the U-Net's 2^depth pooling (the net then only
        # under-scales a broad smear); ~15 days survives pooling and still sits
        # well inside the 30-day match tolerance used at scoring time.
        sig = 15.0 * DT
        for (e, a) in p["steps"]:
            tar += a * np.exp(-0.5 * ((t - e) / sig) ** 2)
        nspec = random_noise_spec(rng, n)
        y = S + bench.inject(n, nspec, DT, rng)
        return t, y, tar, None
    p = _abs_epochs(random_signal_params(rng, family=family), t)
    S = bench.clean_signal(t, p)
    nspec = random_noise_spec(rng, n)
    y = S + bench.inject(n, nspec, DT, rng)
    if task == "denoise":
        return t, y, S, None
    mask = np.ones(n, bool)
    for _ in range(int(rng.integers(1, 4))):
        L = int(rng.choice([5, 10, 30, 60, 120]))
        s = int(rng.integers(0, max(1, n - L)))
        mask[s:s + L] = False
    return t, y, y, mask


# ------------------------------------------------------------------ model
def build_unet(in_ch=1, base=16, depth=3):
    """Small 1-D U-Net; returns an nn.Module (length-agnostic for W % 2^depth == 0)."""
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.ReLU(inplace=True),
                             nn.Conv1d(o, o, 7, padding=3), nn.ReLU(inplace=True))

    chs = [base * (2 ** i) for i in range(depth + 1)]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList()
            c = in_ch
            for lv in range(depth + 1):
                self.enc.append(block(c, chs[lv])); c = chs[lv]
            self.pool = nn.MaxPool1d(2)
            self.dec = nn.ModuleList([block(chs[lv + 1] + chs[lv], chs[lv])
                                      for lv in range(depth - 1, -1, -1)])
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.out = nn.Conv1d(chs[0], 1, 7, padding=3)

        def forward(self, x):
            skips = []
            for lv in range(depth + 1):
                x = self.enc[lv](x)
                if lv < depth:
                    skips.append(x)
                    x = self.pool(x)
            for j, lv in enumerate(range(depth - 1, -1, -1)):
                x = self.up(x)
                skip = skips[lv]
                x = x[..., :skip.shape[-1]]
                x = self.dec[j](torch.cat([x, skip], dim=1))
            return self.out(x)

    return Net()


def count_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def build_gru(in_ch=1, hidden=64, layers=2):
    """Bidirectional GRU sequence-to-sequence denoiser -- a materially different
    learned baseline from the convolutional U-Net, at a comparable parameter
    budget (review point M6). A local conv stem, a 2-layer BiGRU over the
    window, and a conv head; length-agnostic like the U-Net."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv1d(in_ch, hidden, 7, padding=3)
            self.gru = nn.GRU(hidden, hidden, num_layers=layers,
                              batch_first=True, bidirectional=True)
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(2 * hidden, hidden, 7, padding=3), nn.ReLU(),
                nn.Conv1d(hidden, 1, 7, padding=3))

        def forward(self, x):
            z = torch.relu(self.stem(x))          # (B, hidden, W)
            z = z.transpose(1, 2)                 # (B, W, hidden)
            o, _ = self.gru(z)                    # (B, W, 2*hidden)
            o = o.transpose(1, 2)                 # (B, 2*hidden, W)
            return self.head(o)                   # (B, 1, W)

    return Net()


# ------------------------------------------------------------------ normalization
def norm_stats(y, mask=None):
    """Per-series (mean, std) from the observed points; unit-scaled at worst."""
    y = np.asarray(y, float)
    m = y[mask].mean() if (mask is not None and mask.sum() > 2) else y.mean()
    s = y[mask].std() if (mask is not None and mask.sum() > 2) else y.std()
    if not np.isfinite(s) or s < 1e-6:
        s = 1.0
    return float(m), float(s)


def noise_scale(y):
    """Robust per-series *noise* scale from successive differences.

    Dividing by the full std (dominated by trend + seasonal) crushes a decimetre
    offset to a fraction of sigma, so the detector cannot see it; the classical
    CUSUM detector works on the detrended residual instead. Scaling by the
    high-frequency noise level puts the offset amplitude on a comparable footing
    (a ``k`` mm step becomes ~k/sigma, i.e. an SNR)."""
    d = np.diff(np.asarray(y, float))
    s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    if not np.isfinite(s) or s < 1e-9:
        s = float(np.std(d) / np.sqrt(2.0)) or 1.0
    return float(s)


def detect_normalize(y, avg_win=AVG_WIN):
    """Detector input: high-pass (minus a moving average) then divide by the
    noise scale. The high-pass removes the trend/seasonal that would otherwise
    swamp a decimetre step, and the noise scale turns the step into an SNR --
    the same two operations the classical CUSUM detector applies internally."""
    from scipy.ndimage import uniform_filter1d
    y = np.asarray(y, float)
    win = int(avg_win) | 1
    hp = y - uniform_filter1d(y, win, mode="nearest")
    s = noise_scale(y)
    return hp / s


# ------------------------------------------------------------------ inference
def windowed_infer(net, series, mask=None, device="cpu", w=W, stride=None,
                   norm=True, rescale=True, scale_mode="std", batch=32):
    """Apply `net` to a long series by overlapping windows, averaging outputs.

    `series` : (n,). `mask` (impute) marks observed epochs. With `norm` the
    window is standardised by the series statistics before the net and the
    output is rescaled back (for denoise/impute); for detect the output stays in
    the normalised units. ``scale_mode='noise'`` divides by the robust noise
    scale (and no mean) -- the normalisation the detect net was trained under,
    so a ``k`` mm step reads as ~``k/sigma``.

    Windows are evaluated in chunks of `batch` so the recurrent baseline, whose
    native (non-cuDNN) kernel is launch-overhead-bound, amortises its per-step
    cost across the batch. The nets carry no batch-dependent layers, so the
    output is numerically identical to a one-window-at-a-time loop.
    """
    import torch
    y = np.asarray(series, float)
    n = len(y)
    stride = stride or w // 2
    pad = w
    if not norm:
        m, s = 0.0, 1.0
    elif scale_mode == "noise":
        m, s = 0.0, noise_scale(y)
    else:
        m, s = norm_stats(y, mask)
    yn = (y - m) / s
    yp = np.pad(yn, (pad, pad), mode="reflect")
    mp = None
    if mask is not None:
        mp = np.pad(np.asarray(mask, float), (pad, pad), mode="constant")
    acc = np.zeros(n, float)
    cnt = np.zeros(n, float)
    starts = [a for a in range(-pad, n + pad, stride) if max(0, a) < min(n, a + w)]
    net.eval()
    with torch.no_grad():
        for i in range(0, len(starts), max(1, int(batch))):
            chunk = starts[i:i + max(1, int(batch))]
            segs = []
            for a in chunk:
                seg = yp[a + pad:a + pad + w]
                if mask is not None:
                    sm = mp[a + pad:a + pad + w]
                    # channel 0 must be *masked* (zero at gaps), exactly as the
                    # training loader builds it -- otherwise the net sees the
                    # very values it is asked to predict.
                    segs.append(np.stack([seg * sm, sm]))    # (2,w)
                else:
                    segs.append(seg[None, :])                # (1,w)
            xt = torch.tensor(np.stack(segs), dtype=torch.float32, device=device)
            outs = net(xt).detach().cpu().numpy()            # (B,1,w)
            for j, a in enumerate(chunk):
                lo, hi = max(0, a), min(n, a + w)
                o = outs[j, 0]
                acc[lo:hi] += o[lo - a:hi - a]
                cnt[lo:hi] += 1.0
    cnt[cnt == 0] = 1.0
    out = acc / cnt
    return out * s + m if (norm and rescale) else out


# ------------------------------------------------------------------ training
def _batch_tensors(batch, task, device, w):
    import torch
    xs, ys = [], []
    for (y, tar, aux) in batch:
        if task == "impute":
            xs.append(np.stack([np.where(aux, y, 0.0), aux.astype(float)]))
        else:
            xs.append(y[None, :])
        ys.append(tar[None, :])
    X = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    Y = torch.tensor(np.stack(ys), dtype=torch.float32, device=device)
    return X, Y


def train_net(net, task, n_steps=3000, batch=32, lr=2e-3, seed=0, device="cpu",
              w=W, log_every=250, family="in"):
    """Train on the injected signal/noise family; targets are per-window
    standardised exactly as `windowed_infer` normalises at test time."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    hist = []
    net.train()
    for step in range(n_steps):
        batch_data = []
        for _ in range(batch):
            _t, y, tar, aux = make_sample(w, rng, task, family=family)
            obs = aux if aux is not None else np.ones(len(y), bool)
            if task == "detect":
                # high-pass + noise-scale so the offset amplitude is an SNR
                # the net can actually resolve (see detect_normalize)
                s = noise_scale(y)
                yn, tn = detect_normalize(y), tar / s
            else:
                m, s = norm_stats(y, obs)
                yn, tn = (y - m) / s, (tar - m) / s
            batch_data.append((yn, tn, aux))
        X, Y = _batch_tensors(batch_data, task, device, w)
        opt.zero_grad()
        o = net(X)
        if task == "detect":
            loss = torch.nn.functional.huber_loss(o, Y, delta=0.5)
        else:
            loss = torch.nn.functional.mse_loss(o, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == n_steps - 1:
            hist.append((step, float(loss.detach().cpu())))
            print(f"    step {step:5d}  loss {float(loss):.5f}", flush=True)
    return hist


# ------------------------------------------------------------------ detection read-out
def detect_field(net, y, device="cpu", w=W, stride=None, avg_win=AVG_WIN):
    """Overlap-add the detect net over a long series after the training-time
    high-pass/noise normalisation (so train and test see the same input)."""
    yn = detect_normalize(y, avg_win=avg_win)
    return windowed_infer(net, yn, device=device, w=w, stride=stride,
                          norm=False, rescale=False)


def detect_peaks(field, t, thresh=3.0, min_sep_days=60):
    """Turn the detect net's impulse field (amplitude / noise-sigma) into epochs.

    The target is a signed Gaussian bump at each offset, so we peak-pick on the
    absolute response; `thresh` is in units of the per-series noise sigma, so a
    step of ``thresh`` x sigma is called an offset. `min_sep_days` enforces the
    one-offset-per-neighbourhood rule the greedy matcher also assumes.
    """
    from scipy.signal import find_peaks
    f = np.abs(np.asarray(field, float))
    t = np.asarray(t, float)
    dt = float(np.median(np.diff(t)))            # years per sample
    dist = max(1, int(round((min_sep_days / 365.25) / dt)))
    idx, _ = find_peaks(f, height=thresh, distance=dist)
    return [float(t[i]) for i in idx]

