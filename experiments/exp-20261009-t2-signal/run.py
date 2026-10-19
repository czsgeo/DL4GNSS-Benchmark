#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261009-t2-signal -- T2 gap filling scored against the signal as well
as the observation (review point m9).

`exp-20261008-classical-baselines` and `exp-20261008-dl-baselines` score the
gap fillers only against the withheld observation y (the station's own value,
correlated noise included). A filler that reproduces part of the correlated
noise can therefore look better than it is: it matches the noise realisation, not
the signal. Because the truth is known by injection, we can also score every
filler against the noise-free signal S x  -- the quantity a downstream trajectory
fit actually cares about.

This script re-runs T2 *verbatim*: same test stations, same components, same
seed arithmetic (rng0 ~ (si, comp); rng2 ~ (si, comp, L)), same gap lengths and
n_gaps, same classical IMPUTERS, and the same five headline U-Net impute nets
(`exp-20261008-dl-baselines/nets/impute_s*.pt`) -- so the vs-observation column
reproduces the registered t2_* numbers exactly, and the vs-signal column is the
new evidence. Both columns come out of one pass through `bench.run_impute`,
whose `S=` argument (added for this point) yields `rmse_signal`.

Writes results.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench  # noqa: E402
import classical as cl  # noqa: E402

DT = bench.DT


def load_stations(cfg):
    npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
    split = json.load(open(os.path.join(ROOT, cfg["split_file"]), encoding="utf-8"))
    want = set(split[cfg["use_split"]])
    sid = [str(s) for s in npz["sites"] if str(s) in want]
    sid.sort()
    if cfg.get("max_stations"):
        k = int(cfg["max_stations"])
        if len(sid) > k:
            idx = np.linspace(0, len(sid) - 1, k).round().astype(int)
            sid = [sid[i] for i in sorted(set(idx))]
    return npz, sid


def load_nets(cfg, device):
    """Load the headline U-Net impute nets (one per training seed)."""
    import torch
    import dl
    ndir = os.path.join(ROOT, cfg["nets_dir"])
    paths = sorted(glob.glob(os.path.join(ndir, cfg["impute_glob"])))
    nets = []
    for p in paths:
        ck = torch.load(p, map_location=device)
        net = dl.build_unet(in_ch=ck.get("in_ch", 2), base=ck.get("base", 12),
                            depth=ck.get("depth", 3))
        net.load_state_dict(ck["state_dict"])
        net.to(device)
        net.eval()
        nets.append((os.path.basename(p), net))
    return nets


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    import torch
    import dl
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz, sids = load_stations(cfg)
    nets = load_nets(cfg, device)
    print(f"device {device}  stations {len(sids)}  impute nets {len(nets)}: "
          f"{[n for n, _ in nets]}", flush=True)

    noise = cfg["noise"]
    base_seed = int(cfg["seed"])
    Ls = cfg["impute"]["gap_lengths_days"]
    n_gaps = cfg["impute"]["n_gaps"]

    # methods: classical (deterministic) + one U-Net per training seed.
    method_names = list(cl.IMPUTERS.keys()) + [f"unet__{n}" for n, _ in nets]

    # acc[(L, method)] = list of per-station-component values
    acc = {}
    for si, sid in enumerate(sids):
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            y_real = enu[:, comp]
            p = bench.signal_params_from_real(t, y_real, seed=base_seed + si)
            S = bench.clean_signal(t, p)
            nspec = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                          noise["sigma_rw_mm"], len(t), DT)
            rng0 = np.random.default_rng(base_seed + 1000 * si + 10 * comp + 0)
            y0 = S + bench.inject(len(t), nspec, DT, rng0)
            for L in Ls:
                rng2 = np.random.default_rng(base_seed + 7 * si + comp + int(L))
                # Build the method dict for this pass: classical + per-seed U-Net.
                methods = dict(cl.IMPUTERS)
                for nname, net in nets:
                    methods[f"unet__{nname}"] = (
                        lambda t, y, m, _n=net: dl.windowed_infer(_n, y, mask=m, device=device))
                o = bench.run_impute(t, y0, L, n_gaps, rng2, methods, S=S)
                for m, dd in o.items():
                    acc.setdefault((int(L), m), []).append((dd["rmse"], dd["rmse_signal"]))
        print(f"  [{si+1}/{len(sids)}] {sid}", flush=True)

    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")

    def std(v):
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    out = {"exp_id": cfg["exp_id"], "n_stations": len(sids),
           "components": cfg["components"], "gap_lengths_days": Ls,
           "n_gaps": n_gaps, "n_impute_nets": len(nets),
           "impute": {}}
    for L in Ls:
        cell = {}
        # classical: pooled over station-components
        for m in cl.IMPUTERS:
            v = acc.get((int(L), m), [])
            if v:
                cell[m] = {"rmse": mean([a for a, b in v]),
                           "rmse_signal": mean([b for a, b in v]), "n": len(v)}
        # unet: aggregate per training seed (mean over station-components) then
        # take the mean and the training-seed std across the seeds.
        per_seed_rmse, per_seed_sig = [], []
        for nname, net in nets:
            v = acc.get((int(L), f"unet__{nname}"), [])
            if v:
                per_seed_rmse.append(mean([a for a, b in v]))
                per_seed_sig.append(mean([b for a, b in v]))
        if per_seed_rmse:
            cell["unet"] = {
                "rmse": mean(per_seed_rmse), "rmse_std": std(per_seed_rmse),
                "rmse_signal": mean(per_seed_sig), "rmse_signal_std": std(per_seed_sig),
                "n_seeds": len(per_seed_rmse),
                "n": len(acc.get((int(L), f"unet__{nets[0][0]}"), []))}
        out["impute"][str(int(L))] = cell

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results.json", flush=True)

    # quick console summary of the vs-observation reproduction + the new column
    print("\nL     method   rmse(vs y)  rmse_signal(vs S)")
    for L in Ls:
        c = out["impute"][str(int(L))]
        for m in list(cl.IMPUTERS) + ["unet"]:
            if m in c:
                print(f"{L:5d} {m:8s} {c[m]['rmse']:9.3f}   {c[m]['rmse_signal']:9.3f}")


if __name__ == "__main__":
    main()
