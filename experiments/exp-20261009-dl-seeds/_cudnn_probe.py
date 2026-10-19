#!/usr/bin/env python3
"""Throwaway probe: does cuDNN work for (a) conv, (b) GRU on this box?

The training script disables cuDNN globally because the conv engine is reported
broken. But the GRU baseline's cost is dominated by the cuDNN-RNN path, which is
a different engine. This probe measures per-step wall time and whether the loss
actually descends, for both engines, so we can decide whether to re-enable cuDNN
for the GRU jobs only.

Run:  ../.venv/bin/python _cudnn_probe.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import torch
import dl

STEPS = 30
BATCH = 8          # small: we only need the per-step rate, not a real run
W = 1024


def timed_run(arch, cudnn, steps=STEPS, batch=BATCH):
    torch.backends.cudnn.enabled = cudnn
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    dev = "cuda"
    net = dl.build_gru(in_ch=1, hidden=64, layers=2) if arch == "gru" \
        else dl.build_unet(in_ch=1, base=12, depth=3)
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    # fixed batch so conv and gru see the same data
    data = [dl.make_sample(W, rng, "denoise", family="in") for _ in range(batch)]
    bd = []
    for _t, y, tar, aux in data:
        m, s = dl.norm_stats(y, np.ones(len(y), bool))
        bd.append(((y - m) / s, (tar - m) / s, None))
    X, Y = dl._batch_tensors(bd, "denoise", dev, W)
    net.train()
    t0 = time.time()
    first = last = None
    for i in range(steps):
        opt.zero_grad()
        o = net(X)
        loss = torch.nn.functional.mse_loss(o, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        v = float(loss.detach().cpu())
        if first is None:
            first = v
        last = v
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / steps, first, last


if __name__ == "__main__":
    for arch in ("gru", "unet"):
        for cudnn in (False, True):
            try:
                per, first, last = timed_run(arch, cudnn)
                print(f"arch={arch:5s} cudnn={str(cudnn):5s} "
                      f"{per*1000:8.1f} ms/step  loss {first:.4f} -> {last:.4f}  "
                      f"({first-last:+.4f})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"arch={arch:5s} cudnn={str(cudnn):5s} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
