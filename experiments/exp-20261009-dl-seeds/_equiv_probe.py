#!/usr/bin/env python3
"""Throwaway: confirm batched windowed_infer == the old one-window-at-a-time
loop, for U-Net and GRU, with and without a mask."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import torch
torch.backends.cudnn.enabled = False
import dl  # noqa: E402


def old_windowed_infer(net, series, mask=None, device="cpu", w=dl.W, stride=None,
                       norm=True, rescale=True, scale_mode="std"):
    y = np.asarray(series, float); n = len(y)
    stride = stride or w // 2; pad = w
    if not norm:
        m, s = 0.0, 1.0
    elif scale_mode == "noise":
        m, s = 0.0, dl.noise_scale(y)
    else:
        m, s = dl.norm_stats(y, mask)
    yn = (y - m) / s
    yp = np.pad(yn, (pad, pad), mode="reflect")
    mp = np.pad(np.asarray(mask, float), (pad, pad), mode="constant") if mask is not None else None
    acc = np.zeros(n); cnt = np.zeros(n)
    net.eval()
    with torch.no_grad():
        for a in range(-pad, n + pad, stride):
            lo, hi = max(0, a), min(n, a + w)
            if hi <= lo:
                continue
            seg = yp[a + pad:a + pad + w]
            if mask is not None:
                sm = mp[a + pad:a + pad + w]
                x = np.stack([seg * sm, sm])[None, :, :]
            else:
                x = seg[None, None, :]
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            o = net(xt)[0, 0].detach().cpu().numpy()
            acc[lo:hi] += o[lo - a:hi - a]; cnt[lo:hi] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt * s + m if (norm and rescale) else acc / cnt


dev = "cuda"
rng = np.random.default_rng(0)
y = np.cumsum(rng.normal(0, 0.4, 4000)) + 6 * np.sin(np.arange(4000) / 365 * 2 * np.pi)
mask = np.ones(4000, bool)
mask[500:560] = False

for name, net in (("unet", dl.build_unet(in_ch=1, base=12, depth=3)),
                  ("gru", dl.build_gru(in_ch=1, hidden=64, layers=2))):
    net = net.to(dev)
    a1 = old_windowed_infer(net, y, device=dev)
    a2 = dl.windowed_infer(net, y, device=dev, batch=32)
    print(f"{name} denoise   max|diff| = {np.max(np.abs(a1 - a2)):.3e}", flush=True)

for name, net in (("unet_imp", dl.build_unet(in_ch=2, base=12, depth=3)),
                  ("gru_imp", dl.build_gru(in_ch=2, hidden=64, layers=2))):
    net = net.to(dev)
    a1 = old_windowed_infer(net, y, mask=mask, device=dev)
    a2 = dl.windowed_infer(net, y, mask=mask, device=dev, batch=32)
    print(f"{name} impute    max|diff| = {np.max(np.abs(a1 - a2)):.3e}", flush=True)
