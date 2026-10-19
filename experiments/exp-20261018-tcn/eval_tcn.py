#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the TCN against the U-Net (and BiGRU where available) on the
frozen test split, and break the results down by analysis centre (NGL/JPL)
and by region (external-validation arm).

One script, three tasks, because they share the station set and the harness:

  denoise : 10 test stations x 3 components x 2 realisations
  impute  : 10 test stations x 3 components x gaps 5/30/120 d
  detect  : 8 test stations x 3 components x amplitudes 2/4/8/16/25 mm,
            station-level validation/test splits for threshold selection

Every learned number is reported per training seed as well as averaged, so the
architecture comparison carries the seed distribution rather than a single run.
"""
from __future__ import annotations

import csv
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
import dl  # noqa: E402
import rev  # noqa: E402

DT = bench.DT


def load_stations(cfg, key):
    npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
    split = json.load(open(os.path.join(ROOT, cfg["split_file"]), encoding="utf-8"))
    want = set(split[cfg["use_split"]])
    sid = sorted(str(s) for s in npz["sites"] if str(s) in want)
    k = int(cfg[key]["max_stations"])
    if len(sid) > k:
        idx = np.linspace(0, len(sid) - 1, k).round().astype(int)
        sid = [sid[i] for i in sorted(set(idx))]
    return npz, sid


def load_meta(cfg):
    rows = list(csv.DictReader(open(os.path.join(ROOT, cfg["stations_csv"]),
                                    encoding="utf-8")))
    meta = {r["sid"]: r for r in rows}
    # JPL rows carry no coordinates in stations.csv; the NGL holdings table
    # supplies them so the regional breakdown is not an NGL-only map.
    path = os.path.join(ROOT, cfg.get("ngl_holdings", ""))
    if cfg.get("ngl_holdings") and os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) < 3:
                continue
            try:
                la, lo = float(p[1]), float(p[2])
            except ValueError:
                continue
            for key in (p[0], p[0].split("@")[0]):
                row = meta.setdefault(key, {"sid": key})
                for field, val in (("lat", la), ("lon", lo)):
                    try:
                        old = float(row.get(field, "nan"))
                    except (TypeError, ValueError):
                        old = float("nan")
                    if not np.isfinite(old):
                        row[field] = str(val)
    return meta


def region_of(lat, lon):
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return "other"
    lon = ((lon + 180.0) % 360.0) - 180.0
    if -25.0 <= lon <= 45.0 and 34.0 <= lat <= 72.0:
        return "Europe"
    if (-170.0 <= lon <= -50.0) and (10.0 <= lat <= 75.0):
        return "North America"
    if 45.0 < lon <= 180.0 and -12.0 <= lat <= 78.0:
        return "Asia"
    if lon > 150.0 or lon < -150.0 or lat < -12.0:
        return "Pacific"
    return "other"


def load_nets(cfg, device):
    def fam(d, pat):
        return rev.load_nets(os.path.join(ROOT, d), pat, device)
    out = {}
    for prefix, d, pat in (
            ("unet", cfg["unet_dir"], "unet_denoise_in_s*.pt"),
            ("gru", cfg["unet_dir"], "gru_denoise_in_s*.pt"),
            ("tcn", cfg["tcn_dir"], "tcn_denoise_in_s*.pt")):
        out[prefix] = fam(d, pat)
    return out


def arch_summary(values_by_key, prefix):
    """values_by_key: {method_key: [values]}. Return per-seed and across-seed stats."""
    per_seed = {}
    for k, vals in values_by_key.items():
        if k.startswith(prefix + "_s"):
            per_seed[int(k.split("_s")[1])] = float(np.mean(vals))
    if not per_seed:
        return None
    arr = np.array([per_seed[s] for s in sorted(per_seed)], float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
            "n_seeds": int(len(arr)), "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)}}


def agg_simple(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def group_means(per_station, meta, key):
    """per_station: {sid: {method: value}} -> {method: {group: mean}}."""
    groups = {}
    for sid, d in per_station.items():
        m = meta.get(sid, {})
        try:
            have_xy = np.isfinite(float(m.get("lat", "nan"))) and \
                np.isfinite(float(m.get("lon", "nan")))
        except (TypeError, ValueError):
            have_xy = False
        if not have_xy:
            m = meta.get(sid.split("@")[0], m)
        if key == "source":
            g = m.get("source", "unknown")
        elif key == "region":
            g = region_of(float(m.get("lat", "nan")), float(m.get("lon", "nan")))
        else:
            g = "all"
        for meth, val in d.items():
            groups.setdefault(meth, {}).setdefault(g, []).append(val)
    out = {}
    for meth, gd in groups.items():
        out[meth] = {g: agg_simple(v) for g, v in gd.items()}
    return out


def denoise(cfg, npz, sids, meta, device):
    dcfg = cfg["denoise"]
    nets = load_nets(cfg, device)
    methods = dict(cl.DENOISERS)
    for prefix, netdict in nets.items():
        for stem, (seed, net) in netdict.items():
            methods[f"{prefix}_s{seed}"] = (
                lambda net: (lambda t, y: rev.windowed_infer_perwin(
                    net, y, device=device)))(net)
    noise = cfg["noise"]
    store, per_station = {}, {}
    for si, sid in enumerate(sids):
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            p = bench.signal_params_from_real(t, enu[:, comp], seed=int(cfg["seed"]) + si)
            S = bench.clean_signal(t, p)
            nsp = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                        noise["sigma_rw_mm"], len(t), DT)
            for k in range(int(dcfg["n_realisations"])):
                rng = np.random.default_rng(int(cfg["seed"]) + 1000 * si + 10 * comp + k)
                y = S + bench.inject(len(t), nsp, DT, rng)
                out, _ = bench.run_denoise(t, y, S, methods)
                for m, dd in out.items():
                    store.setdefault(m, []).append(dd["rmse_reduction"])
                    per_station.setdefault(sid, {}).setdefault(m, []).append(
                        dd["rmse_reduction"])
        print(f"  denoise {si+1}/{len(sids)} {sid}", flush=True)
    per_station_mean = {sid: {m: float(np.mean(v)) for m, v in d.items()}
                        for sid, d in per_station.items()}
    res = {"per_method": {}, "by_source": group_means(per_station_mean, meta, "source"),
           "by_region": group_means(per_station_mean, meta, "region")}
    for m in ("none", "mle", "ssa", "wavelet", "rts", "median"):
        res["per_method"][m] = agg_simple(store[m])
    for prefix in ("unet", "gru", "tcn"):
        s = arch_summary(store, prefix)
        if s:
            res["per_method"][prefix] = s
    return res


def impute(cfg, npz, sids, meta, device):
    icfg = cfg["impute"]
    nets = {}
    for prefix, d, pat in (
            ("unet", cfg["unet_dir"], "unet_impute_in_s*.pt"),
            ("gru", cfg["unet_dir"], "gru_impute_in_s*.pt"),
            ("tcn", cfg["tcn_dir"], "tcn_impute_in_s*.pt")):
        nets[prefix] = rev.load_nets(os.path.join(ROOT, d), pat, device)
    methods = dict(cl.IMPUTERS)
    for prefix, netdict in nets.items():
        for stem, (seed, net) in netdict.items():
            methods[f"{prefix}_s{seed}"] = (
                lambda net: (lambda t, y, m: rev.windowed_infer_perwin(
                    net, y, mask=m, device=device)))(net)
    noise = cfg["noise"]
    store = {L: {} for L in icfg["gap_lengths_days"]}
    for si, sid in enumerate(sids):
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            p = bench.signal_params_from_real(t, enu[:, comp], seed=int(cfg["seed"]) + si)
            S = bench.clean_signal(t, p)
            nsp = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                        noise["sigma_rw_mm"], len(t), DT)
            rng = np.random.default_rng(int(cfg["seed"]) + 1000 * si + 10 * comp + 7)
            y = S + bench.inject(len(t), nsp, DT, rng)
            for L in icfg["gap_lengths_days"]:
                rng2 = np.random.default_rng(int(cfg["seed"]) + 3 * si + comp + int(L))
                out = bench.run_impute(t, y, L, icfg["n_gaps"], rng2, methods, S=S)
                for m, dd in out.items():
                    store[L].setdefault(m, []).append((dd["rmse"], dd["rmse_signal"]))
        print(f"  impute {si+1}/{len(sids)} {sid}", flush=True)
    res = {}
    for L, d in store.items():
        blk = {"per_method": {}}
        for m in ("linear", "pchip", "ssa", "mle"):
            blk["per_method"][m] = {
                "rmse": agg_simple([x[0] for x in d[m]]),
                "rmse_signal": agg_simple([x[1] for x in d[m]])}
        for prefix in ("unet", "gru", "tcn"):
            keys = [k for k in d if k.startswith(prefix + "_s")]
            if not keys:
                continue
            per = {k: [x[0] for x in d[k]] for k in keys}
            persig = {k: [x[1] for x in d[k]] for k in keys}
            blk["per_method"][prefix] = {
                "rmse": arch_summary(per, prefix),
                "rmse_signal": arch_summary(persig, prefix)}
        res[str(L)] = blk
    return res


def detect(cfg, npz, sids, meta, device):
    dcfg = cfg["detect"]
    nets = {}
    for prefix, d, pat in (
            ("unet", cfg["unet_dir"], "unet_detect_in_s*.pt"),
            ("tcn", cfg["tcn_dir"], "tcn_detect_in_s*.pt")):
        nets[prefix] = rev.load_nets(os.path.join(ROOT, d), pat, device)
    ths = np.asarray(dcfg["thresholds"], float)
    tol = dcfg["tol_days"] / 365.25
    msep = dcfg["min_sep_days"]
    noise = cfg["noise"]
    # amplitude -> method -> station -> list of (P,R,F)
    curves = {}
    for si, sid in enumerate(sids):
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            p0 = bench.signal_params_from_real(t, enu[:, comp], seed=int(cfg["seed"]) + si)
            nsp = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                        noise["sigma_rw_mm"], len(t), DT)
            for A in dcfg["amplitudes_mm"]:
                for k in range(int(dcfg["n_realisations"])):
                    rng = np.random.default_rng(int(cfg["seed"]) + 3 * si + comp + int(A) + 1000 * k)
                    epc = bench.random_epochs(t, dcfg["n_offsets"],
                                              min_sep_years=dcfg["min_sep_years"], rng=rng)
                    p = dict(p0)
                    p["steps"] = [(e, float(rng.choice([-1, 1]) * A)) for e in epc]
                    S = bench.clean_signal(t, p)
                    y = S + bench.inject(len(t), nsp, DT, rng)
                    fields = {"classical": rev.classical_step_field(t, y)}
                    for prefix, netdict in nets.items():
                        for stem, (seed, net) in netdict.items():
                            fields[f"{prefix}_s{seed}"] = dl.detect_field(net, y,
                                                                         device=device)
                    for m, fld in fields.items():
                        P, R, F = rev.pr_curve([fld], epc, t, ths, tol, msep)
                        curves.setdefault(int(A), {}).setdefault(m, {}).setdefault(
                            sid, []).append((P, R, F))
        print(f"  detect {si+1}/{len(sids)} {sid}", flush=True)

    rngs = np.random.default_rng(int(cfg["seed"]) + 99)
    res = {}
    for A, by_m in curves.items():
        station_curves = {}
        for m, by_st in by_m.items():
            station_curves[m] = {sid: tuple(np.mean([c[i] for c in cs], axis=0)
                                            for i in range(3))
                                 for sid, cs in by_st.items()}
        stations = sorted(station_curves["classical"].keys())
        blk = {"per_method": {}}

        def mean_curve(m, subset):
            P = np.mean([station_curves[m][s][0] for s in subset], axis=0)
            R = np.mean([station_curves[m][s][1] for s in subset], axis=0)
            F = np.mean([station_curves[m][s][2] for s in subset], axis=0)
            return P, R, F

        for m in station_curves:
            P, R, F = mean_curve(m, stations)
            ent = {"ap": rev.average_precision(P, R),
                   "f1_opt": float(np.max(F)),
                   "f1_opt_threshold": float(ths[int(np.argmax(F))])}
            if m.startswith(("unet_", "tcn_")):
                ent["per_seed"] = {str(int(m.split("_s")[1])): {
                    "ap": rev.average_precision(*mean_curve(m, stations)[:2]),
                    "f1_opt": float(np.max(mean_curve(m, stations)[2]))}}
            blk["per_method"][m] = ent
        # across-seed summaries + station-level validation selection
        for prefix in ("unet", "tcn"):
            keys = [m for m in station_curves if m.startswith(prefix + "_s")]
            if not keys:
                continue
            aps = np.array([blk["per_method"][m]["ap"] for m in keys])
            f1s = np.array([blk["per_method"][m]["f1_opt"] for m in keys])
            blk["per_method"][prefix] = {
                "ap_mean": float(aps.mean()), "ap_std": float(aps.std(ddof=1)),
                "f1_opt_mean": float(f1s.mean()), "f1_opt_std": float(f1s.std(ddof=1)),
                "n_seeds": len(keys)}
            valsel_by = {m: [] for m in keys}
            for _ in range(int(dcfg["val_splits"])):
                perm = rngs.permutation(len(stations))
                nval = max(2, len(stations) // 2)
                vidx = [stations[i] for i in perm[:nval]]
                tidx = [stations[i] for i in perm[nval:]]
                for m in keys:
                    _, _, Fv = mean_curve(m, vidx)
                    j = int(np.argmax(Fv))
                    _, _, Ft = mean_curve(m, tidx)
                    valsel_by[m].append(float(Ft[j]))
            for m in keys:
                blk["per_method"][m]["val_selected_f1_mean"] = float(
                    np.mean(valsel_by[m]))
                blk["per_method"][m]["val_selected_f1_std"] = float(
                    np.std(valsel_by[m], ddof=1))
            allvalsel = [v for vals in valsel_by.values() for v in vals]
            blk["per_method"][prefix]["val_selected_f1_mean"] = float(np.mean(allvalsel))
            blk["per_method"][prefix]["val_selected_f1_std"] = float(np.std(allvalsel, ddof=1))
            blk["per_method"][prefix]["oracle_f1_mean"] = float(np.mean(
                [np.max(mean_curve(m, stations)[2]) for m in keys]))
        # classical val selection
        valsel = []
        for _ in range(int(dcfg["val_splits"])):
            perm = rngs.permutation(len(stations))
            nval = max(2, len(stations) // 2)
            vidx = [stations[i] for i in perm[:nval]]
            tidx = [stations[i] for i in perm[nval:]]
            _, _, Fv = mean_curve("classical", vidx)
            j = int(np.argmax(Fv))
            _, _, Ft = mean_curve("classical", tidx)
            valsel.append(float(Ft[j]))
        blk["per_method"]["classical"]["val_selected_f1_mean"] = float(np.mean(valsel))
        blk["per_method"]["classical"]["val_selected_f1_std"] = float(np.std(valsel, ddof=1))
        blk["per_method"]["classical"]["oracle_f1_mean"] = float(np.max(
            mean_curve("classical", stations)[2]))
        # by source
        src = {}
        for sid_group in ("ngl", "jpl"):
            sub = [s for s in stations if meta.get(s, {}).get("source") == sid_group]
            if not sub:
                continue
            _, _, F = mean_curve("classical", sub)
            src[sid_group] = {"n_stations": len(sub),
                              "classical_f1_opt": float(np.max(F))}
            for prefix in ("unet", "tcn"):
                keys = [m for m in station_curves if m.startswith(prefix + "_s")]
                if keys:
                    src[sid_group][f"{prefix}_f1_opt_mean"] = float(np.mean(
                        [np.max(mean_curve(m, sub)[2]) for m in keys]))
        blk["by_source"] = src
        res[str(A)] = blk
    return res


def main():
    cfg = json.load(open(os.path.join(HERE, "eval_config.json"), encoding="utf-8"))
    out_dir = os.environ.get("PAPER_OUT", HERE)
    import torch
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_meta(cfg)
    print(f"device {device}", flush=True)
    results = {"exp_id": cfg["exp_id"], "device": device}

    npz, sids = load_stations(cfg, "denoise")
    results["denoise"] = denoise(cfg, npz, sids, meta, device)
    print("denoise done", flush=True)

    npz, sids = load_stations(cfg, "impute")
    results["impute"] = impute(cfg, npz, sids, meta, device)
    print("impute done", flush=True)

    npz, sids = load_stations(cfg, "detect")
    results["detect"] = detect(cfg, npz, sids, meta, device)
    print("detect done", flush=True)

    json.dump(results, open(os.path.join(out_dir, "results.json"), "w",
                            encoding="utf-8"), indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8") as fh:
        fh.write("task,condition,method,metric,value\n")
        for m, d in results["denoise"]["per_method"].items():
            for k in ("mean", "std"):
                if k in d:
                    fh.write(f"denoise,all,{m},reduction_{k},{d[k]}\n")
        for L, blk in results["impute"].items():
            for m, d in blk["per_method"].items():
                for metric in ("rmse", "rmse_signal"):
                    if metric in d and isinstance(d[metric], dict):
                        for k in ("mean", "std"):
                            if k in d[metric]:
                                fh.write(f"impute,{L},{m},{metric}_{k},{d[metric][k]}\n")
        for A, blk in results["detect"].items():
            for m, d in blk["per_method"].items():
                for k in ("ap", "ap_mean", "f1_opt", "f1_opt_mean",
                          "val_selected_f1_mean", "oracle_f1_mean"):
                    if k in d:
                        fh.write(f"detect,{A},{m},{k},{d[k]}\n")
    print("written", out_dir, flush=True)


if __name__ == "__main__":
    main()
