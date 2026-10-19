#!/usr/bin/env python3
"""Recover WN/FN/RW amplitudes on every benchmark series.

The trajectory residual (trend + annual + semiannual, robust fit) is fitted by
Whittle maximum likelihood with the three-component power-law model. Transients
and unmodelled deformation are absorbed by the RW term, so the recovered RW
amplitude is an upper bound on the stochastic random walk.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import gnss_lib as gl  # noqa: E402

cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
sids = sorted(str(s) for s in npz["sites"])

rows = []
for si, sid in enumerate(sids):
    t = np.asarray(npz[f"t__{sid}"], float)
    enu = np.asarray(npz[f"enu__{sid}"], float)
    if enu.ndim == 1:
        enu = enu[:, None]
    X, _ = gl.build_design(t, orders=tuple(cfg["design_orders"]),
                           poly=int(cfg["design_poly"]))
    for comp in cfg["components"]:
        if comp >= enu.shape[1]:
            continue
        beta, resid, _, _ = gl.robust_fit(t, enu[:, comp], X)
        e = gl.whittle_mle(resid, cfg["dt"], models=tuple(cfg["models"]))
        rows.append({"site": sid, "component": ["E", "N", "U"][comp],
                     "n": int(len(t)), "span_yr": float(t[-1] - t[0]),
                     "sigma_wn": float(e["sigma_wn"]),
                     "sigma_fn": float(e["sigma_fn"]),
                     "sigma_rw": float(e["sigma_rw"])})
    if (si + 1) % 25 == 0:
        print(f"{si + 1}/{len(sids)}", flush=True)


def q(v, p):
    return float(np.percentile(v, p))


summary = {"exp_id": cfg["exp_id"], "n_series": len(rows), "by_component": {}}
for comp in ("E", "N", "U", "all"):
    sel = rows if comp == "all" else [r for r in rows if r["component"] == comp]
    if not sel:
        continue
    blk = {"n": len(sel)}
    for k in ("sigma_wn", "sigma_fn", "sigma_rw"):
        v = np.array([r[k] for r in sel])
        blk[k] = {"median": q(v, 50), "q25": q(v, 25), "q75": q(v, 75),
                  "min": float(v.min()), "max": float(v.max()),
                  "frac_gt_injected": float(np.mean(v > {"sigma_wn": 1.5, "sigma_fn": 3.0,
                                                         "sigma_rw": 1.0}[k]))}
    summary["by_component"][comp] = blk
summary["notes"] = ("WN/FN/RW from Whittle MLE on the robust trajectory residual; "
                    "RW absorbs unmodelled transients and is an upper bound.")
json.dump(summary, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"),
          indent=2)
with open(os.path.join(HERE, "results.csv"), "w", encoding="utf-8") as fh:
    fh.write("site,component,n,span_yr,sigma_wn,sigma_fn,sigma_rw\n")
    for r in rows:
        fh.write(f"{r['site']},{r['component']},{r['n']},{r['span_yr']},"
                 f"{r['sigma_wn']},{r['sigma_fn']},{r['sigma_rw']}\n")
print(json.dumps(summary["by_component"]["all"], indent=1))
