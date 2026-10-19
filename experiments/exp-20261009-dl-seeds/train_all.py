#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261009-dl-seeds -- train the revision's learned baselines.

One process trains one network and writes it to nets/. Everything is driven by
`config.json` so the seed set is frozen and reproducible.

  task in {denoise, impute, detect}
  arch in {unet, gru}
  family in {in, oof}          (in = original injection family; oof = the
                                out-of-family truth of exp-20261009-truth-families)

Why the counts differ by task: the bimodal training-seed distribution that made
the v1.0 detector inconclusive (review M3) is a property of the detection task,
so the detector gets the large seed set (>=20); the denoiser/gap filler feed the
out-of-family and downstream arms and need a stable seed mean, not a tail.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import torch  # noqa: E402
torch.backends.cudnn.enabled = False       # V100 + cuDNN9 conv engine is broken here

import dl  # noqa: E402


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    task = sys.argv[1]
    arch = sys.argv[2]
    family = sys.argv[3]
    seed = int(sys.argv[4])
    nets_dir = os.path.join(HERE, cfg.get("nets_dir", "nets"))
    os.makedirs(nets_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = cfg["training"]
    in_ch = 2 if task == "impute" else 1
    if arch == "unet":
        net = dl.build_unet(in_ch=in_ch, base=tr["base"], depth=tr["depth"])
    elif arch == "gru":
        net = dl.build_gru(in_ch=in_ch, hidden=tr["gru_hidden"], layers=tr["gru_layers"])
    else:
        raise SystemExit(f"unknown arch {arch}")

    npath = os.path.join(nets_dir, f"{arch}_{task}_{family}_s{seed}.pt")
    if os.path.exists(npath) and cfg.get("skip_existing", True):
        print(f"skip {npath}", flush=True)
        return

    print(f"train {arch}/{task}/{family} seed={seed} params={dl.count_params(net)} "
          f"device={device}", flush=True)
    t0 = time.time()
    dl.train_net(net, task, n_steps=tr["n_steps"], batch=tr["batch"], lr=tr["lr"],
                 seed=seed, device=device, w=tr["w"], log_every=10 ** 9, family=family)
    wall = time.time() - t0
    torch.save({"state_dict": net.state_dict(), "in_ch": in_ch, "arch": arch,
                "task": task, "family": family, "seed": seed, "wall_s": wall,
                "base": tr["base"], "depth": tr["depth"],
                "gru_hidden": tr["gru_hidden"], "gru_layers": tr["gru_layers"]}, npath)
    print(f"done {npath}  wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
