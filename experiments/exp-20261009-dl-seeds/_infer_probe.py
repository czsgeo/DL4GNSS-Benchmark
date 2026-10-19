#!/usr/bin/env python3
"""Throwaway: time one windowed_infer call for a U-Net vs a GRU denoiser on a
realistic series length, to size the alt-arch experiment's GRU cost."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import torch
torch.backends.cudnn.enabled = False
import dl  # noqa: E402
import rev  # noqa: E402

nets_dir = os.path.join(HERE, "nets")
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
y = np.cumsum(rng.normal(0, 0.5, 7300)) + np.sin(np.arange(7300) / 365 * 2 * np.pi) * 5

for pattern in ("unet_denoise_in_s20261007.pt", "gru_denoise_in_s20261007.pt"):
    (_s, net), = rev.load_nets(nets_dir, pattern, dev).values()
    net.to(dev).eval()
    # warm
    dl.windowed_infer(net, y[:2048], device=dev)
    t0 = time.time()
    dl.windowed_infer(net, y, device=dev)
    torch.cuda.synchronize()
    print(f"{pattern:34s} {time.time()-t0:6.2f} s / {len(y)}-sample series", flush=True)
