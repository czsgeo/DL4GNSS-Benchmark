#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261018-rw-downstream -- ZQH comment 2.2 (random walk downstream).

The v1.0 downstream study injected WN+FN noise only. Here the same pipeline
is run twice, once with sigma_rw=0 and once with sigma_rw=0.5 mm, with the
noise model estimated as WN+FN+RW on the RAW series (the achievable
propagation arm) and, separately, re-estimated on the denoised/imputed
residual (the naive audit arm that produced the artificial -1 in v1.0).

Reported per (noise level, arm, method): velocity bias/RMSE, reported vs
empirical velocity sigma, calibration ratio, 1/2-sigma coverage, and the
relative bias of each noise component.
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
import gnss_lib as gl  # noqa: E402
import rev  # noqa: E402

DT = bench.DT
MODELS = ("wn", "fn", "rw")
COMP = ("wn", "fn", "rw")


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


def load_nets(cfg, device):
    ndir = os.path.join(ROOT, cfg["nets_dir"])
    cap = int(cfg.get("max_net_seeds", 0))

    def fam(pattern):
        nets = rev.load_nets(ndir, pattern, device)
        out = [(int(stem.split("_s")[1]), net) for stem, (_seed, net) in nets.items()]
        return sorted(out)[:cap] if cap > 0 else sorted(out)

    return fam(cfg["denoise_glob"]), fam(cfg["impute_glob"])


def fit_noise(resid):
    return gl.whittle_mle(resid, DT, models=MODELS)


def estimate(y_out, t, vel_col, nspec, e_orig, s_orig):
    """Velocity and its GLS sigma from a denoised/imputed series.

    `e_orig`/`s_orig` are the noise model and GLS sigma read from the *raw*
    series; they depend only on the station realisation, so the caller fits them
    once and reuses them for every denoiser/imputer of that realisation (this is
    an exact reuse, not an approximation).
    """
    X, _ = gl.build_design(t, orders=(1, 2), poly=2)
    beta, resid, _, _ = gl.robust_fit(t, y_out, X)
    vel = float(beta[vel_col])
    e_naive = fit_noise(resid)
    s_naive, _, _ = gl.gls_velocity_sigma(t, X, vel_col, e_naive["_psd_params"],
                                          MODELS, DT)
    rel_n = {c: float(e_naive[f"sigma_{c}"] / nspec[f"sigma_{c}"] - 1.0)
             if nspec[f"sigma_{c}"] > 0 else float("nan") for c in COMP}
    rel_o = {c: float(e_orig[f"sigma_{c}"] / nspec[f"sigma_{c}"] - 1.0)
             if nspec[f"sigma_{c}"] > 0 else float("nan") for c in COMP}
    return vel, float(s_naive), float(s_orig), rel_n, rel_o


def cal_stats(vel_err, sig):
    v = np.asarray(vel_err, float)
    s = np.asarray(sig, float)
    emp = float(np.std(v))
    rep = float(np.median(s))
    return {"sigma_reported_median": rep, "sigma_empirical": emp,
            "calibration_ratio": (rep / emp) if emp > 0 else None,
            "coverage_1sigma": float(np.mean(np.abs(v) <= s)),
            "coverage_2sigma": float(np.mean(np.abs(v) <= 2.0 * s))}


def boot_ci(vel_err, sig, fn, n_boot, seed):
    v = np.asarray(vel_err, float)
    s = np.asarray(sig, float)
    n = len(v)
    if n < 3:
        val = fn(v, s)
        return {"lo": val, "hi": val}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([fn(v[i], s[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def summarize(rows, src, cfg):
    ve = np.array([r["vel_err"] for r in rows], float)
    sig = np.array([r[f"sig_{src}"] for r in rows], float)
    st = cal_stats(ve, sig)
    st["vel_bias_mm_yr"] = float(np.mean(ve))
    st["vel_rmse_mm_yr"] = float(np.sqrt(np.mean(ve ** 2)))
    st["coverage_1sigma_ci"] = boot_ci(
        ve, sig, lambda v, s: float(np.mean(np.abs(v) <= s)),
        cfg["n_boot"], int(cfg["seed"]) + 1)
    st["calibration_ci"] = boot_ci(
        ve, sig, lambda v, s: (float(np.median(s)) / float(np.std(v))
                               if np.std(v) > 0 else np.nan),
        cfg["n_boot"], int(cfg["seed"]) + 2)
    st["n"] = int(len(ve))
    return st


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", default="global", choices=["global", "perwin"])
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    out_dir = os.environ.get("PAPER_OUT", HERE)
    import torch
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    npz, sids = load_stations(cfg)
    den_nets, imp_nets = load_nets(cfg, device)
    print(f"stations {len(sids)} device {device} den {len(den_nets)} "
          f"imp {len(imp_nets)} inference {args.inference}", flush=True)

    den_methods = dict(cl.DENOISERS)
    imp_methods = dict(cl.IMPUTERS)
    for s, net in den_nets:
        if args.inference == "perwin":
            den_methods[f"unet_s{s}"] = (
                lambda net: (lambda t, y: rev.windowed_infer_perwin(
                    net, y, device=device)))(net)
        else:
            den_methods[f"unet_s{s}"] = (
                lambda net: (lambda t, y: dl.windowed_infer(net, y, device=device)))(net)
    for s, net in imp_nets:
        if args.inference == "perwin":
            imp_methods[f"unet_s{s}"] = (
                lambda net: (lambda t, y, m: rev.windowed_infer_perwin(
                    net, y, mask=m, device=device)))(net)
        else:
            imp_methods[f"unet_s{s}"] = (
                lambda net: (lambda t, y, m: dl.windowed_infer(net, y, mask=m,
                                                               device=device)))(net)

    base_seed = int(cfg["seed"])
    results = {"exp_id": cfg["exp_id"], "n_stations": len(sids),
               "n_seeds": cfg["n_seeds"], "models": list(MODELS),
               "noise_levels": cfg["noise_levels"], "levels": {}}

    for lname, noise in cfg["noise_levels"].items():
        arms = {"clean": {}, "imputed": {}}
        for si, sid in enumerate(sids):
            t = np.asarray(npz[f"t__{sid}"], float)
            enu = np.asarray(npz[f"enu__{sid}"], float)
            if enu.ndim == 1:
                enu = enu[:, None]
            X, names = gl.build_design(t, orders=(1, 2), poly=2)
            vel_col = names.index("trend")
            nspec = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                          noise["sigma_rw_mm"], len(t), DT)
            for comp in cfg["components"]:
                if comp >= enu.shape[1]:
                    continue
                p = bench.signal_params_from_real(t, enu[:, comp], seed=base_seed + si)
                S = bench.clean_signal(t, p)
                true_vel = float(p["trend"])
                for k in range(cfg["n_seeds"]):
                    rng = np.random.default_rng(base_seed + 1000 * si + 10 * comp + k)
                    y = S + bench.inject(len(t), nspec, DT, rng)
                    e_orig = fit_noise(gl.robust_fit(t, y, X)[1])
                    s_orig, _, _ = gl.gls_velocity_sigma(
                        t, X, vel_col, e_orig["_psd_params"], MODELS, DT)
                    v, sn, so, rn, ro = estimate(y, t, vel_col, nspec,
                                                 e_orig, s_orig)
                    arms["clean"].setdefault("raw", []).append(
                        {"vel_err": v - true_vel, "sig_naive": sn, "sig_orig": so,
                         "rel_naive": rn, "rel_orig": ro})
                    for m, fn in den_methods.items():
                        yhat = fn(t, y)
                        v, sn, so, rn, ro = estimate(yhat, t, vel_col, nspec, e_orig, s_orig)
                        arms["clean"].setdefault(m, []).append(
                            {"vel_err": v - true_vel, "sig_naive": sn, "sig_orig": so,
                             "rel_naive": rn, "rel_orig": ro})
                    rng2 = np.random.default_rng(base_seed + 7 * si + comp + 1)
                    obs = bench.random_gaps(len(t), cfg["impute_gap_days"],
                                            cfg["n_gaps"], rng2)
                    for m, fn in imp_methods.items():
                        yhat = fn(t, y, obs)
                        v, sn, so, rn, ro = estimate(yhat, t, vel_col, nspec, e_orig, s_orig)
                        arms["imputed"].setdefault(m, []).append(
                            {"vel_err": v - true_vel, "sig_naive": sn, "sig_orig": so,
                             "rel_naive": rn, "rel_orig": ro})
            print(f"  [{lname}] {si+1}/{len(sids)} {sid}", flush=True)

        lvl = {}
        for arm_name, arm in arms.items():
            blk = {}
            for m, rows in arm.items():
                entry = {}
                for src in ("naive", "orig"):
                    entry[src] = summarize(rows, src, cfg)
                for c in COMP:
                    entry[f"{c}_rel_bias_naive"] = float(np.nanmean(
                        [r["rel_naive"][c] for r in rows]))
                    entry[f"{c}_rel_bias_orig"] = float(np.nanmean(
                        [r["rel_orig"][c] for r in rows]))
                blk[m] = entry
            net_keys = sorted(k for k in arm if k.startswith("unet_s"))
            if net_keys:
                blk["unet"] = {}
                for src in ("naive", "orig"):
                    per = [summarize(arm[k], src, cfg) for k in net_keys]
                    keys = ("vel_bias_mm_yr", "vel_rmse_mm_yr", "sigma_reported_median",
                            "sigma_empirical", "calibration_ratio", "coverage_1sigma",
                            "coverage_2sigma")
                    blk["unet"][src] = {
                        k: float(np.mean([p[k] for p in per if p[k] is not None]))
                        for k in keys}
                    blk["unet"][src]["_std"] = {
                        k: float(np.std([p[k] for p in per if p[k] is not None], ddof=1))
                        if len(per) > 1 else 0.0 for k in keys}
                    blk["unet"][src]["n_train_seeds"] = len(net_keys)
                for c in COMP:
                    blk["unet"][f"{c}_rel_bias_orig"] = float(np.nanmean(
                        [r["rel_orig"][c] for k in net_keys for r in arm[k]]))
            lvl[arm_name] = blk
        results["levels"][lname] = lvl

    suffix = "" if args.inference == "global" else "_perwin"
    results["inference"] = args.inference
    json.dump(results, open(os.path.join(out_dir, f"results{suffix}.json"), "w",
                            encoding="utf-8"), indent=2)
    with open(os.path.join(out_dir, f"results{suffix}.csv"), "w", encoding="utf-8") as fh:
        fh.write("level,arm,method,sigma_source,metric,value\n")
        for lname, lvl in results["levels"].items():
            for arm_name, blk in lvl.items():
                for m, d in blk.items():
                    if m == "unet":
                        for src in ("naive", "orig"):
                            for k, v in d[src].items():
                                if isinstance(v, dict) or k == "n_train_seeds":
                                    continue
                                fh.write(f"{lname},{arm_name},{m},{src},{k},{v}\n")
                        continue
                    for src in ("naive", "orig"):
                        for k, v in d[src].items():
                            if isinstance(v, dict):
                                continue
                            fh.write(f"{lname},{arm_name},{m},{src},{k},{v}\n")
                    for c in COMP:
                        fh.write(f"{lname},{arm_name},{m},orig,{c}_rel_bias,"
                                 f"{d[f'{c}_rel_bias_orig']}\n")
    print(json.dumps({lname: {arm: {m: (round(d["orig"]["coverage_1sigma"], 3),
                                      round(d["orig"]["calibration_ratio"], 3))
                                  for m, d in blk.items() if m in ("raw", "mle", "unet")}
                             for arm, blk in lvl.items()}
                      for lname, lvl in results["levels"].items()}, indent=2))


if __name__ == "__main__":
    main()
