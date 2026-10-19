#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External / regional validation arm for the v2.0 benchmark.

Every held-out test series is scored on two tasks -- in-family denoising and
30-day gap filling -- and every score is kept per series, so the same numbers
can be aggregated by analysis centre (NGL vs JPL) and by geographic region.
Protocol matches the main in-family experiment: truth from the station's robust
trajectory fit, WN 1.5 mm + FN 3.0 mm, three realisations per component.

The arm is descriptive: it changes no training or threshold decision. JPL
coordinates are read from the NGL holdings table (the JPL site codes are IGS
station codes), which is why the JPL rows in stations.csv carry no coordinates.
"""
from __future__ import annotations

import csv
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

DT = bench.DT


def coords():
    lat, lon = {}, {}
    for row in csv.DictReader(open(os.path.join(ROOT, cfg["stations_csv"]))):
        try:
            la, lo = float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(la) and np.isfinite(lo):
            lat[row["sid"]], lon[row["sid"]] = la, lo
    for line in open(os.path.join(ROOT, cfg["ngl_holdings"])):
        p = line.split()
        if len(p) < 3:
            continue
        try:
            la, lo = float(p[1]), float(p[2])
        except ValueError:
            continue
        lat.setdefault(p[0], la)
        lon.setdefault(p[0], lo)
    return lat, lon


def region_of(sid, lat, lon, source):
    code = sid.split("@")[0]
    la, lo = lat.get(sid, lat.get(code)), lon.get(sid, lon.get(code))
    if la is None or lo is None or not np.isfinite(la) or not np.isfinite(lo):
        return "other"
    lo = ((lo + 180.0) % 360.0) - 180.0
    if -25.0 <= lo <= 45.0 and 34.0 <= la <= 72.0:
        return "Europe"
    if -170.0 <= lo <= -50.0 and 10.0 <= la <= 75.0:
        return "North America"
    if 45.0 < lo <= 180.0 and -12.0 <= la <= 78.0:
        return "Asia"
    if lo > 150.0 or lo < -150.0 or la < -12.0:
        return "Pacific"
    return "other"


def agg(vals):
    v = np.asarray(vals, float)
    lo, hi = rev.bootstrap_ci(v, np.mean, seed=7)
    mu, sd = rev.mean_std(v)
    return {"mean": mu, "std": sd, "ci_lo": lo, "ci_hi": hi, "n_series": int(len(v))}


cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
import torch  # noqa: E402
import dl  # noqa: E402
torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"

npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
split = json.load(open(os.path.join(ROOT, cfg["split_file"]), encoding="utf-8"))
src = {str(s): str(q) for s, q in zip(npz["sites"], npz["source"])}
sids = [s for s in split[cfg["use_split"]] if str(s) in src]
sids = sorted(str(s) for s in sids)
lat, lon = coords()

den_nets = rev.load_nets(os.path.join(ROOT, cfg["nets_dir"]), cfg["denoise_glob"], device)
imp_nets = rev.load_nets(os.path.join(ROOT, cfg["nets_dir"]), cfg["impute_glob"], device)
print(f"series {len(sids)} device {device} den {len(den_nets)} imp {len(imp_nets)}",
      flush=True)

den_methods = dict(cl.DENOISERS)
imp_methods = dict(cl.IMPUTERS)
for stem, (_s, net) in den_nets.items():
    den_methods[stem] = (lambda net: (lambda t, y: rev.windowed_infer_perwin(
        net, y, device=device)))(net)
for stem, (_s, net) in imp_nets.items():
    imp_methods[stem] = (lambda net: (lambda t, y, m: rev.windowed_infer_perwin(
        net, y, mask=m, device=device)))(net)

base_seed = int(cfg["seed"])
noise = cfg["noise"]
rows = []
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
        nspec = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                      noise["sigma_rw_mm"], len(t), DT)
        den = {k: [] for k in den_methods}
        imp = {k: [] for k in imp_methods}
        for k in range(cfg["n_seeds"]):
            rng = np.random.default_rng(base_seed + 1000 * si + 10 * comp + k)
            y = S + bench.inject(len(t), nspec, DT, rng)
            base = bench.rmse(y, S)
            for m, fn in den_methods.items():
                den[m].append(1.0 - bench.rmse(fn(t, y), S) / base)
            obs = bench.random_gaps(len(t), cfg["impute_gap_days"], cfg["n_gaps"],
                                    np.random.default_rng(base_seed + 7 * si + comp + k))
            miss = ~obs
            for m, fn in imp_methods.items():
                imp[m].append(bench.rmse(fn(t, y, obs), y, mask=miss))
        base_row = {"site": sid, "source": src[sid],
                    "region": region_of(sid, lat, lon, src[sid]), "component": comp}
        for m in den:
            rows.append({**base_row, "task": "denoise", "method": m,
                         "value": float(np.mean(den[m])),
                         "value_std": float(np.std(den[m], ddof=1))})
        for m in imp:
            rows.append({**base_row, "task": "impute30", "method": m,
                         "value": float(np.mean(imp[m])),
                         "value_std": float(np.std(imp[m], ddof=1))})
    print(f"  [{si + 1}/{len(sids)}] {sid}", flush=True)

methods_by_task = {"denoise": list(den_methods), "impute30": list(imp_methods)}
results = {"exp_id": cfg["exp_id"],
           "protocol": {"n_series": len(sids), "n_component_series": len(rows) // (
               len(methods_by_task["denoise"]) + len(methods_by_task["impute30"])),
               "n_realisations": cfg["n_seeds"], "impute_gap_days": cfg["impute_gap_days"],
               "components": cfg["components"], "noise": noise},
           "n_network_seeds": {"denoise": len(den_nets), "impute": len(imp_nets)},
           "by_source": {}, "by_region": {}, "by_source_task": {}}


def summarize(sel, task, methods):
    out = {}
    for m in methods:
        vals = [r["value"] for r in sel if r["task"] == task and r["method"] == m]
        if vals:
            out[m] = agg(vals)
    return out


for group_name, key in (("by_source", "source"), ("by_region", "region")):
    groups = sorted({r[key] for r in rows})
    for g in groups:
        sel = [r for r in rows if r[key] == g]
        results[group_name][g] = {
            "n_series": len({(r["site"], r["component"]) for r in sel if r["task"] == "denoise"}),
            "denoise": summarize(sel, "denoise", methods_by_task["denoise"]),
            "impute30": summarize(sel, "impute30", methods_by_task["impute30"]),
        }
for s in ("ngl", "jpl"):
    sel = [r for r in rows if r["source"] == s]
    results["by_source_task"].setdefault(s, {})["n_series"] = len(
        {(r["site"], r["component"]) for r in sel if r["task"] == "denoise"})

json.dump(results, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"),
          indent=2)
with open(os.path.join(HERE, "results.csv"), "w", encoding="utf-8") as fh:
    fh.write("site,source,region,component,task,method,value,value_std\n")
    for r in rows:
        fh.write(f"{r['site']},{r['source']},{r['region']},{r['component']},"
                 f"{r['task']},{r['method']},{r['value']},{r['value_std']}\n")
print("done", flush=True)
