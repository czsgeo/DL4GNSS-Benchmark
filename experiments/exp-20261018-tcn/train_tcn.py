#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one TCN (exp-20261018-tcn) under a frozen config.

  python train_tcn.py <task> <arch> <family> <seed>

Mirrors exp-20261009-dl-seeds/train_all.py so the training protocol (steps,
batch, lr, window, generator) is identical across architectures.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import torch  # noqa: E402
torch.backends.cudnn.enabled = False       # V100 + cuDNN9 conv engine is broken here

import dl  # noqa: E402
import tcn  # noqa: E402


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    task = sys.argv[1]
    arch = sys.argv[2]
    family = sys.argv[3]
    seed = int(sys.argv[4])
    assert arch == "tcn", arch
    nets_dir = os.path.join(HERE, cfg.get("nets_dir", "nets"))
    os.makedirs(nets_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = cfg["training"]
    in_ch = 2 if task == "impute" else 1
    net = tcn.build_tcn(in_ch=in_ch)
    npath = os.path.join(nets_dir, f"{arch}_{task}_{family}_s{seed}.pt")
    if os.path.exists(npath) and cfg.get("skip_existing", True):
        print(f"skip {npath}", flush=True)
        return

    print(f"train {arch}/{task}/{family} seed={seed} params={tcn.count_params(net)} "
          f"device={device}", flush=True)
    t0 = time.time()
    dl.train_net(net, task, n_steps=tr["n_steps"], batch=tr["batch"], lr=tr["lr"],
                 seed=seed, device=device, w=tr["w"], log_every=10 ** 9, family=family)
    wall = time.time() - t0
    torch.save({"state_dict": net.state_dict(), "in_ch": in_ch, "arch": arch,
                "task": task, "family": family, "seed": seed, "wall_s": wall,
                "channels": [40, 56, 72, 80, 80]}, npath)
    print(f"done {npath}  wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
