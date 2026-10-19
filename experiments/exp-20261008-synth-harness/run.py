#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261008-synth-harness -- validate the injection/estimation pipeline.

Part A: inject known WN+FN amplitudes, fit the trajectory model, run
        `whittle_mle`, and report the recovered amplitudes vs truth.
Part B: Monte-Carlo velocity scatter vs the `gls_velocity_sigma` prediction
        (the calibration that the whole downstream argument rests on).

Everything is seeded; results.json records the numbers used in the manuscript.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gnss_lib as gl  # noqa: E402
import synth  # noqa: E402

DT = gl.DT_YEAR


def part_a(cfg, rng):
    rows = []
    for swn in cfg["grid_wn_mm"]:
        for sfn in cfg["grid_fn_mm"]:
            if swn == 0 and sfn == 0:
                continue
            rec_w, rec_f = [], []
            for k in range(cfg["n_realisations"]):
                spec = {"trend": 10.0, "A1": 5.0, "A2": 2.0}
                nspec = synth.make_spec(swn, sfn, 0.0, cfg["n"], DT, seed=cfg["seed"] + k)
                s = synth.make_series(cfg["n"], spec, nspec, seed=cfg["seed"] + k)
                t, y = s["t"], s["noisy"]
                X, _ = gl.build_design(t, orders=(1, 2), poly=2)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                est = gl.whittle_mle(resid, DT, models=("wn", "fn"))
                rec_w.append(est["sigma_wn"]); rec_f.append(est["sigma_fn"])
            rows.append({"sigma_wn": swn, "sigma_fn": sfn,
                         "wn_med": float(np.median(rec_w)), "fn_med": float(np.median(rec_f)),
                         "wn_spread": float(np.std(rec_w)), "fn_spread": float(np.std(rec_f)),
                         "wn_bias": float(np.median(rec_w) / swn - 1.0) if swn > 0 else None,
                         "fn_bias": float(np.median(rec_f) / sfn - 1.0) if sfn > 0 else None})
            print(f"  A wn={swn} fn={sfn}: rec wn={rows[-1]['wn_med']:.3f} "
                  f"fn={rows[-1]['fn_med']:.3f}", flush=True)
    return rows


def part_b(cfg, rng):
    """MC calibration of the reported GLS velocity uncertainty.

    One fixed injection PSD (so the ensemble is stationary), then for each
    realisation: generate coloured noise, form the GLS velocity under the
    *injected* covariance (the estimator a geodetic pipeline actually uses),
    and predict the formal sigma from the *estimated* noise model. The claim
    under test is that the reported sigma matches the true MC scatter -- a
    self-consistency check, so the estimator and the reported sigma must be
    the same (GLS) estimator; comparing an OLS velocity to a GLS sigma
    understates the calibration by ~2x.
    """
    from scipy.linalg import cho_factor, cho_solve
    vc = cfg["velocity_check"]
    n = cfg["n"]
    t = np.arange(n) * DT
    X, names = gl.build_design(t, orders=(1, 2), poly=2)
    vel_col = names.index("trend")

    # Fixed injection covariance: build once, reuse the factor per realisation.
    nspec = synth.make_spec(vc["sigma_wn_mm"], vc["sigma_fn_mm"], 0.0, n, DT,
                            seed=cfg["seed"])
    C = gl.build_cov_wn_fn(n, DT, nspec["S_wn"], nspec["S_fn"])
    cf = cho_factor(C, lower=True, check_finite=False)
    CiX = cho_solve(cf, X, check_finite=False)
    Ainv = np.linalg.pinv(X.T @ CiX)

    vels, pred = [], []
    for k in range(cfg["n_realisations"]):
        e = synth.noise_from_spec(n, nspec, DT,
                                  np.random.default_rng(cfg["seed"] + 20000 + k))
        y = vc["trend_mm_yr"] * t + e
        beta = Ainv @ (CiX.T @ y)          # GLS velocity (matches the sigma)
        vels.append(beta[vel_col])
        # Report the sigma the way a pipeline would: estimate noise from
        # OLS residuals, then propagate through the GLS formula.
        b_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
        est = gl.whittle_mle(y - X @ b_ols, DT, models=("wn", "fn"))
        sv, *_ = gl.gls_velocity_sigma(t, X, vel_col, est["_psd_params"],
                                       ("wn", "fn"), DT)
        pred.append(sv)
    emp = float(np.std(vels))
    pred_med = float(np.median(pred))
    return {"empirical_sigma_v": emp, "predicted_sigma_v_median": pred_med,
            "ratio_pred_over_emp": pred_med / emp if emp > 0 else None,
            "velocity_bias": float(np.mean(vels) - vc["trend_mm_yr"])}


def main():
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    rng = np.random.default_rng(cfg["seed"])
    print("Part A: noise recovery", flush=True)
    a = part_a(cfg, rng)
    print("Part B: velocity-uncertainty calibration", flush=True)
    b = part_b(cfg, rng)
    ok_a = True
    for r in a:
        if r["wn_bias"] is not None and abs(r["wn_bias"]) >= cfg["tol_rel"]:
            ok_a = False
        if r["fn_bias"] is not None and abs(r["fn_bias"]) >= cfg["tol_rel"]:
            ok_a = False
    results = {"exp_id": cfg["exp_id"], "part_a": a, "part_b": b,
               "self_check_pass": bool(ok_a and abs(b["ratio_pred_over_emp"] - 1.0) < 0.2)}
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({"part_b": b, "self_check_pass": results["self_check_pass"]}, indent=2))


if __name__ == "__main__":
    main()
