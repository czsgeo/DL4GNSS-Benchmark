#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261009-in-distribution -- review M2.

The v1.0 gap filler loses to linear interpolation by ~4x at a 5-day gap, which
lies *inside* its training support. This experiment asks whether that is a
property of the learned method under a synthetic-to-real shift, or a
training/inference pipeline defect, by scoring the trained nets where there is
**no** domain shift at all:

  window     : fresh windows from the exact training generator, scored directly
  longseries : long synthetic series (global statistics differ from the window
               statistics) scored under whole-series vs per-window
               standardisation -- the normalisation the nets were trained under
  real       : the v1.0 domain-shift setting, same two normalisations

If the nets are competitive in-distribution but fail on the real stations under
whole-series normalisation only, the v1.0 result is a pipeline artefact; if they
fail in-distribution too, the learned column must be read as a bug report.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench  # noqa: E402
import classical as cl  # noqa: E402
import dl  # noqa: E402
import rev  # noqa: E402

DT = bench.DT


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


def mask_gaps(n, L, n_gaps, rng, margin=30):
    obs = np.ones(n, bool)
    hi = max(margin + 1, n - L - margin)
    for s in rng.integers(margin, hi, size=n_gaps):
        obs[s:s + L] = False
    return obs


def agg(vals):
    v = np.asarray(vals, float)
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    out_dir = os.environ.get("PAPER_OUT", HERE)
    import torch
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ndir = os.path.join(ROOT, cfg["nets_dir"])
    d_nets = rev.load_nets(ndir, "unet_denoise_in_s*.pt", device)
    i_nets = rev.load_nets(ndir, "unet_impute_in_s*.pt", device)
    print(f"device {device} denoise nets {len(d_nets)} impute nets {len(i_nets)}", flush=True)

    results = {"exp_id": cfg["exp_id"],
               "denoise_seeds": sorted(s for s, _ in d_nets.values()),
               "impute_seeds": sorted(s for s, _ in i_nets.values())}

    # ------------------------------------------------------------------ window
    W = int(cfg["window"]["w"])
    rng = np.random.default_rng(int(cfg["window"]["seed"]))
    den = {}      # method -> [reduction]
    imp = {}      # (L, method) -> [rmse]
    for _i in range(int(cfg["window"]["n_windows"])):
        _t, y, S, _aux = dl.make_sample(W, rng, "denoise")   # exact training generator
        # denoise
        out, _b = bench.run_denoise(np.arange(W) * DT, y, S, cl.DENOISERS)
        for m, dd in out.items():
            den.setdefault(m, []).append(dd["rmse_reduction"])
        for stem, (_s, net) in d_nets.items():
            yhat = dl.windowed_infer(net, y, device=device, w=W, stride=W // 2)
            r = bench.rmse(yhat, S)
            base = bench.rmse(y, S)
            den.setdefault(f"unet__{stem}", []).append(1.0 - r / base)
        # impute
        for L in cfg["window"]["impute_lengths_days"]:
            L = int(L)
            obs = mask_gaps(W, L, cfg["window"]["n_gaps"], rng)
            miss = ~obs
            for name, fn in cl.IMPUTERS.items():
                yhat = fn(np.arange(W) * DT, y, obs)
                imp.setdefault((L, name), []).append(bench.rmse(yhat, y, mask=miss))
            for stem, (_s, net) in i_nets.items():
                yhat = dl.windowed_infer(net, y, mask=obs, device=device, w=W, stride=W // 2)
                imp.setdefault((L, f"unet__{stem}"), []).append(bench.rmse(yhat, y, mask=miss))
    results["window"] = {"denoise": {m: agg(v) for m, v in den.items()}, "impute": {}}
    for (L, m), v in imp.items():
        results["window"]["impute"].setdefault(str(L), {})[m] = agg(v)

    # ------------------------------------------------------------- longseries
    ls = cfg["longseries"]
    n = int(ls["n_days"])
    rng = np.random.default_rng(int(ls["seed"]))
    long_imp = {}
    long_den = {}
    for _i in range(int(ls["n_series"])):
        # One continuous draw from the training generator: a long series that is
        # genuinely in-distribution, unlike concatenated windows whose seams add
        # step discontinuities the nets never saw.
        _t, y, S, _a = dl.make_sample(n, rng, "denoise")
        t = np.arange(n) * DT
        out, _b = bench.run_denoise(t, y, S, cl.DENOISERS)
        for m, dd in out.items():
            long_den.setdefault(m, []).append(dd["rmse_reduction"])
        for L in ls["impute_lengths_days"]:
            L = int(L)
            obs = mask_gaps(n, L, ls["n_gaps"], rng)
            miss = ~obs
            for name, fn in cl.IMPUTERS.items():
                yhat = fn(t, y, obs)
                long_imp.setdefault((L, "global", name), []).append(bench.rmse(yhat, y, mask=miss))
            for stem, (_s, net) in i_nets.items():
                g = dl.windowed_infer(net, y, mask=obs, device=device)
                p = rev.windowed_infer_perwin(net, y, mask=obs, device=device, task="impute")
                long_imp.setdefault((L, "global", f"unet__{stem}"), []).append(bench.rmse(g, y, mask=miss))
                long_imp.setdefault((L, "perwin", f"unet__{stem}"), []).append(bench.rmse(p, y, mask=miss))
    results["longseries"] = {"denoise": {m: agg(v) for m, v in long_den.items()}, "impute": {}}
    for (L, mode, m), v in long_imp.items():
        results["longseries"]["impute"].setdefault(str(L), {}).setdefault(mode, {})[m] = agg(v)

    # ------------------------------------------------------------------- real
    npz, sids = load_stations(cfg)
    noise = cfg["noise"]
    base_seed = int(cfg["seed"])
    real_imp = {}
    for si, sid in enumerate(sids):
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            p = bench.signal_params_from_real(t, enu[:, comp], seed=base_seed + si)
            S = bench.clean_signal(t, p)
            nsp = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                        noise["sigma_rw_mm"], len(t), DT)
            rng = np.random.default_rng(base_seed + 1000 * si + 10 * comp)
            y = S + bench.inject(len(t), nsp, DT, rng)
            for L in cfg["real"]["impute_lengths_days"]:
                L = int(L)
                obs = mask_gaps(len(t), L, cfg["real"]["n_gaps"], np.random.default_rng(base_seed + 7 * si + comp + L))
                miss = ~obs
                for name, fn in cl.IMPUTERS.items():
                    yhat = fn(t, y, obs)
                    real_imp.setdefault((L, "global", name), []).append(bench.rmse(yhat, y, mask=miss))
                for stem, (_s, net) in i_nets.items():
                    g = dl.windowed_infer(net, y, mask=obs, device=device)
                    pw = rev.windowed_infer_perwin(net, y, mask=obs, device=device, task="impute")
                    real_imp.setdefault((L, "global", f"unet__{stem}"), []).append(bench.rmse(g, y, mask=miss))
                    real_imp.setdefault((L, "perwin", f"unet__{stem}"), []).append(bench.rmse(pw, y, mask=miss))
        print(f"  real [{si+1}/{len(sids)}] {sid}", flush=True)
    results["real"] = {"impute": {}}
    for (L, mode, m), v in real_imp.items():
        results["real"]["impute"].setdefault(str(L), {}).setdefault(mode, {})[m] = agg(v)

    json.dump(results, open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8"),
              indent=2)

    def compact(node, prefix=""):
        for k, v in node.items():
            if isinstance(v, dict) and "mean" not in v:
                compact(v, f"{prefix}{k}.")
            else:
                pass
    # console: classical + one unet summary per cell
    def sum_cell(cell):
        out = {}
        for m, d in cell.items():
            out[m] = round(d["mean"], 3) if not m.startswith("unet__") else None
        us = [d for m, d in cell.items() if m.startswith("unet__")]
        if us:
            out["unet(mean of nets)"] = round(float(np.mean([d["mean"] for d in us])), 3)
        return {k: v for k, v in out.items() if v is not None}
    print("WINDOW impute:", json.dumps({L: sum_cell(c) for L, c in results["window"]["impute"].items()}))
    print("WINDOW denoise:", json.dumps(sum_cell(results["window"]["denoise"])))
    print("LONG impute:", json.dumps({L: {mode: sum_cell(c) for mode, c in d.items()}
                                      for L, d in results["longseries"]["impute"].items()}))
    print("REAL impute:", json.dumps({L: {mode: sum_cell(c) for mode, c in d.items()}
                                      for L, d in results["real"]["impute"].items()}))


if __name__ == "__main__":
    main()
