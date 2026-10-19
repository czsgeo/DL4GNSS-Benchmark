#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261009-truth-families -- review M1 (truth circularity) and M4 (RW).

For every regime (truth family x noise) we score all denoisers on the same
realisations:

  hybrid_in      real-station trend/harmonics + modulation (v1.0 headline truth)
  hybrid_oof     same station, but strong multi-year seasonal modulation +
                 long-lived transients + log relaxation  -> OUTSIDE the
                 constant-harmonic trajectory basis
  synth_in       fully synthetic, in-family (main-text fully-synthetic result)
  synth_oof      fully synthetic, out-of-family
  hybrid_in_rw   in-family truth, noise now contains a random walk (M4)

Two learned denoisers are carried: U-Nets trained on the in-family generator and
U-Nets trained on the out-of-family generator (exp-20261009-dl-seeds), so the
learned method is never penalised merely for being out of its training family.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench  # noqa: E402
import classical as cl  # noqa: E402
import rev  # noqa: E402
import synth  # noqa: E402

DT = bench.DT
CYCLE = 2.0 * np.pi


def load_stations(cfg):
    npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
    split = json.load(open(os.path.join(ROOT, cfg["split_file"]), encoding="utf-8"))
    want = set(split[cfg["use_split"]])
    sid = sorted(str(s) for s in npz["sites"] if str(s) in want)
    k = int(cfg["max_stations"])
    if len(sid) > k:
        idx = np.linspace(0, len(sid) - 1, k).round().astype(int)
        sid = [sid[i] for i in sorted(set(idx))]
    return npz, sid


def synth_truth(cfg, family, i, base_seed):
    """Fully synthetic in-/out-of-family truth of length n_days."""
    s = cfg["synth"]
    rng = np.random.default_rng(base_seed + 100 * i)
    n = int(s["n_days"])
    t = np.arange(n) * DT
    p = {"trend": s["trend"], "A1": s["A1"], "A2": s["A2"],
         "am": s["am_in"] if family == "in" else s["am_oof"],
         "fm": s["fm"], "amph": float(rng.uniform(0, CYCLE)),
         "ph1": float(rng.uniform(0, CYCLE)), "ph2": float(rng.uniform(0, CYCLE)),
         "steps": [], "transients": [], "logdecay": []}
    if family == "oof":
        p["transients"] = [(float(rng.uniform(0.2, 0.75)) * (n - 1) * DT,
                            float(rng.uniform(-15, 15)), float(rng.uniform(0.3, 1.5)))]
        if rng.random() < 0.6:
            p["logdecay"] = [(float(rng.uniform(0.15, 0.6)) * (n - 1) * DT,
                              float(rng.uniform(-20, 20)), float(rng.uniform(0.02, 0.08)))]
    return t, bench.clean_signal(t, p)


def make_methods(nets, tag, device, inference="global"):
    """Classical baselines plus the nets trained on the matching truth family.

    Scoring both net families on both regimes doubles the inference cost without
    answering a question: the comparison of interest is "learned methods given
    their own training family" versus classical, so each regime uses the nets
    trained on that family.
    """
    methods = dict(cl.DENOISERS)
    for stem, (_s, net) in nets.items():
        if inference == "perwin":
            # Match training: every window is standardised by its own
            # statistics (the v1 defect standardised by the whole series).
            methods[f"unet_{tag}__{stem}"] = (
                lambda net: (lambda t, y: rev.windowed_infer_perwin(
                    net, y, device=device)))(net)
        else:
            methods[f"unet_{tag}__{stem}"] = rev.denoiser_of(net, device)
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=None, help="run one regime only")
    ap.add_argument("--inference", default="global", choices=["global", "perwin"],
                    help="global = whole-series normalisation (v1); "
                         "perwin = per-window normalisation (matches training)")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    out_dir = os.environ.get("PAPER_OUT", HERE)
    import torch
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz, sids = load_stations(cfg)
    ndir = os.path.join(ROOT, cfg["nets_dir"])
    nets_in = rev.load_nets(ndir, "unet_denoise_in_s*.pt", device)
    nets_oof = rev.load_nets(ndir, "unet_denoise_oof_s*.pt", device)
    print(f"stations {len(sids)} device {device} unet_in {len(nets_in)} "
          f"unet_oof {len(nets_oof)} inference {args.inference}", flush=True)
    methods = {"in": make_methods(nets_in, "in", device, args.inference),
               "oof": make_methods(nets_oof, "oof", device, args.inference)}
    base_seed = int(cfg["seed"])
    regimes = [r for r in cfg["regimes"]
               if args.regime is None or r["name"] == args.regime]
    if args.regime is not None and not regimes:
        raise SystemExit(f"unknown regime {args.regime}")

    results = {"exp_id": cfg["exp_id"], "n_stations": len(sids),
               "components": cfg["components"], "n_seeds": cfg["n_seeds"],
               "unet_in_seeds": sorted(s for s, _ in nets_in.values()),
               "unet_oof_seeds": sorted(s for s, _ in nets_oof.values()),
               "regimes": {}}
    raw = {}   # regime -> method -> list of reductions (for CIs/csv)

    for reg in regimes:
        name, family, base = reg["name"], reg["family"], reg["base"]
        methods_f = methods[family]
        nspec_cfg = cfg[reg["noise_key"]]
        store = {}
        # ---- hybrid: real station series
        if base == "hybrid":
            for si, sid in enumerate(sids):
                t = np.asarray(npz[f"t__{sid}"], float)
                enu = np.asarray(npz[f"enu__{sid}"], float)
                if enu.ndim == 1:
                    enu = enu[:, None]
                for comp in cfg["components"]:
                    if comp >= enu.shape[1]:
                        continue
                    if family == "in":
                        p = bench.signal_params_from_real(t, enu[:, comp], seed=base_seed + si)
                    else:
                        p = bench.signal_params_out_of_family(t, enu[:, comp], seed=base_seed + si)
                    S = bench.clean_signal(t, p)
                    nsp = bench.make_noise_spec(nspec_cfg["sigma_wn_mm"],
                                                nspec_cfg["sigma_fn_mm"],
                                                nspec_cfg["sigma_rw_mm"], len(t), DT)
                    for k in range(cfg["n_seeds"]):
                        rng = np.random.default_rng(base_seed + 1000 * si + 10 * comp + k)
                        y = S + bench.inject(len(t), nsp, DT, rng)
                        out, _b = bench.run_denoise(t, y, S, methods_f)
                        for m, dd in out.items():
                            store.setdefault(m, []).append(dd["rmse_reduction"])
                print(f"  [{name}] {si+1}/{len(sids)} {sid}", flush=True)
        # ---- fully synthetic
        else:
            for i in range(int(cfg["synth"]["n_synth"])):
                t, S = synth_truth(cfg, family, i, base_seed)
                nsp = bench.make_noise_spec(nspec_cfg["sigma_wn_mm"],
                                            nspec_cfg["sigma_fn_mm"],
                                            nspec_cfg["sigma_rw_mm"], len(t), DT)
                rng = np.random.default_rng(base_seed + 7 * i + 3)
                y = S + bench.inject(len(t), nsp, DT, rng)
                out, _b = bench.run_denoise(t, y, S, methods_f)
                for m, dd in out.items():
                    store.setdefault(m, []).append(dd["rmse_reduction"])
                print(f"  [{name}] synth {i+1}/{cfg['synth']['n_synth']}", flush=True)

        raw[name] = store
        summary = {}
        for m, vals in store.items():
            mu, sd = rev.mean_std(vals)
            lo, hi = rev.bootstrap_ci(vals, np.mean, seed=1)
            summary[m] = {"reduction_mean": mu, "reduction_std": sd,
                          "ci_lo": lo, "ci_hi": hi, "n": len(vals)}
        results["regimes"][name] = summary

    suffix = "" if args.inference == "global" else "_perwin"
    results["inference"] = args.inference
    if args.regime is not None:
        json.dump(results, open(os.path.join(out_dir, f"partial_{args.regime}{suffix}.json"),
                                "w", encoding="utf-8"), indent=2)
        print(f"wrote partial_{args.regime}{suffix}.json", flush=True)
        return
    json.dump(results, open(os.path.join(out_dir, f"results{suffix}.json"), "w",
                            encoding="utf-8"), indent=2)
    with open(os.path.join(out_dir, f"results{suffix}.csv"), "w", encoding="utf-8") as fh:
        fh.write("regime,method,reduction_mean,reduction_std,ci_lo,ci_hi,n\n")
        for name, summ in results["regimes"].items():
            for m, d in summ.items():
                fh.write(f"{name},{m},{d['reduction_mean']},{d['reduction_std']},"
                         f"{d['ci_lo']},{d['ci_hi']},{d['n']}\n")
    for name in results["regimes"]:
        agg = {m: round(d["reduction_mean"], 4) for m, d in results["regimes"][name].items()
               if "__" not in m}
        print(name, json.dumps(agg))


if __name__ == "__main__":
    main()
