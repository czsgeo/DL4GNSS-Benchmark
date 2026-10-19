#!/usr/bin/env python3
"""Throwaway: does batching the GRU forward amortise the native-RNN launch
overhead? windowed_infer currently runs one window per forward; if throughput
scales with batch, a batched inference path makes alt-arch feasible."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import torch
torch.backends.cudnn.enabled = False
import dl  # noqa: E402

dev = "cuda"
net = dl.build_gru(in_ch=1, hidden=64, layers=2).to(dev).eval()
W = 1024
for B in (1, 8, 32, 64, 128):
    x = torch.randn(B, 1, W, device=dev)
    with torch.no_grad():
        net(x)                       # warm
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    N = 3
    with torch.no_grad():
        for _ in range(N):
            net(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print(f"batch={B:4d}  {dt*1000:8.1f} ms/forward   {dt/B*1000:7.2f} ms/window",
          flush=True)
