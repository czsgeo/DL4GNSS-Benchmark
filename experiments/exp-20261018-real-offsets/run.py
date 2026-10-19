#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261018-real-offsets -- real catalogue labels (ZQH comment 1.2).

Labels. The NGL `steps.txt` catalogue (downloaded 2026-10-18, sha256 recorded
in the results) lists candidate discontinuities: code 1 equipment changes and
code 2 possible earthquake steps. A catalogue date is retained as a *labelled
offset* for a component only when the observed level change at that date is at
least 3 noise sigma; dates below that are dropped (unknown/unidentifiable), not
counted as negatives. Duplicate catalogue entries within 60 days are merged.

Protocol. Both detectors are decision fields read out by the same threshold +
minimum-separation rule. Stations are split into validation and test halves at
the station level (40 random splits). The threshold is chosen on validation
stations and applied unchanged to test stations. Reported: AP, F1, precision,
recall, detection delay (days, predicted minus catalogued date), and recall
stratified by step SNR. No threshold is tuned on the test stations.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench  # noqa: E402
import dl  # noqa: E402
import gnss_lib as gl  # noqa: E402
import rev  # noqa: E402

DT = bench.DT
MONTH = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_date(s):
    """YYMMMDD -> decimal year (two-digit year: >=80 is 19xx else 20xx)."""
    yy = int(s[:2])
    y = 1900 + yy if yy >= 80 else 2000 + yy
    m = MONTH[s[2:5]]
    d = int(s[5:7])
    t0 = dt.date(y, 1, 1)
    t1 = dt.date(y + 1, 1, 1)
    return y + (dt.date(y, m, d) - t0).days / (t1 - t0).days


def load_catalog(path, codes):
    """{site: [(epoch, code, raw_event)]}, sorted and de-duplicated later."""
    out = {}
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            try:
                code = int(p[2])
            except ValueError:
                continue
            if code not in codes:
                continue
            try:
                e = parse_date(p[1])
            except (KeyError, ValueError):
                continue
            out.setdefault(p[0], []).append((e, code, " ".join(p[3:])))
            n += 1
    for s in out:
        out[s].sort()
    return out, n


def dedup(events, days):
    kept = []
    for e in events:
        if not kept or (e[0] - kept[-1][0]) * 365.25 > days:
            kept.append(e)
    return kept


def load_meta(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {r["sid"]: r for r in rows}


def region_of(sid, meta):
    """Five geographic regions used for the regional-holdout report."""
    row = meta.get(sid, {})
    try:
        la, lo = float(row["lat"]), float(row["lon"])
    except (KeyError, TypeError, ValueError):
        return "other"
    if not (np.isfinite(la) and np.isfinite(lo)):
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


def robust_step_amplitudes(t, y, events, cfg):
    """Return [(epoch, amp_mm, snr, code)] for events inside the valid span.

    The deterministic trajectory (trend + annual + semiannual) is fitted on
    the series with +/-0.25 y around every candidate event excluded, so the
    fit is not dragged by the steps. The step is the difference of the median
    residual in the 0.08-0.5 y windows on either side, clipped at the midpoint
    to a neighbouring event. Noise sigma is the MAD of first differences.
    """
    if not events:
        return []
    X, _ = gl.build_design(t, orders=(1, 2), poly=2)
    guard = float(cfg["amp_window_years"])
    keep = np.ones(len(t), bool)
    for e, _c, _r in events:
        keep &= ~((t > e - guard) & (t < e + guard))
    if keep.sum() < 100:
        keep[:] = True
    beta, *_ = gl.robust_fit(t[keep], y[keep], X[keep])
    r = y - X @ beta
    sig = 1.4826 * np.median(np.abs(np.diff(r) - np.median(np.diff(r)))) / np.sqrt(2.0)
    if not np.isfinite(sig) or sig <= 0:
        return []
    e0, e1 = float(t[0]), float(t[-1])
    margin = float(cfg["edge_margin_years"])
    gw = float(cfg["amp_guard_years"])
    win = float(cfg["amp_window_years"])
    out = []
    for i, (e, code, _raw) in enumerate(events):
        if e < e0 + margin or e > e1 - margin:
            continue
        left_lim = max(win, (e - events[i - 1][0]) / 2.0) if i > 0 else win
        right_lim = max(win, (events[i + 1][0] - e) / 2.0) if i + 1 < len(events) else win
        pre = (t >= e - left_lim) & (t <= e - gw)
        post = (t >= e + gw) & (t <= e + right_lim)
        if pre.sum() < 20 or post.sum() < 20:
            continue
        a = float(np.median(r[post]) - np.median(r[pre]))
        out.append((float(e), a, float(abs(a) / sig), int(code)))
    return out


def match(true, pred, tol):
    used = set()
    delays = []
    matched = []
    te = np.asarray(true, float)
    for pe in pred:
        d = np.abs(te - pe)
        for j in used:
            d[j] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used.add(j)
            delays.append(float(pe - te[j]))
            matched.append(j)
    return matched, delays


def curve_with_delay(field, t, true_epochs, ths, tol, msep):
    P, R, F, delays, matched = [], [], [], [], []
    n = len(true_epochs)
    for th in ths:
        pred = rev.peaks_from_field(t, field, float(th), min_sep_days=msep)
        m, dl = match(true_epochs, pred, tol)
        tp = len(m)
        fp = len(pred) - tp
        fn = n - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        P.append(p); R.append(rr)
        F.append(2 * p * rr / (p + rr) if p + rr else 0.0)
        delays.append(dl); matched.append(set(m))
    return {"P": np.array(P), "R": np.array(R), "F": np.array(F),
            "delays": delays, "matched": matched}


def load_nets(cfg, device):
    out = {}
    for prefix, d, pat in (
            ("unet", cfg["unet_dir"], "unet_detect_in_s*.pt"),
            ("tcn", cfg["tcn_dir"], "tcn_detect_in_s*.pt")):
        out[prefix] = rev.load_nets(os.path.join(ROOT, d), pat, device)
    return out


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    out_dir = os.environ.get("PAPER_OUT", HERE)
    import torch
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cpath = os.path.join(ROOT, cfg["catalog"])
    h = hashlib.sha256()
    with open(cpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    catalog, n_raw = load_catalog(cpath, set(cfg["catalog_codes"]))
    print(f"catalog lines used {n_raw}; sha {sha}", flush=True)

    npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
    split = json.load(open(os.path.join(ROOT, cfg["split_file"]), encoding="utf-8"))
    meta = load_meta(os.path.join(ROOT, cfg["stations_csv"]))
    want = set(split[cfg["use_split"]])
    sids = sorted(str(s) for s in npz["sites"]
                  if str(s) in want and meta.get(str(s), {}).get("source") == "ngl")
    nets = load_nets(cfg, device)
    ths = np.asarray(cfg["thresholds"], float)
    tol = cfg["tol_days"] / 365.25
    msep = cfg["min_sep_days"]

    # ---- labels
    label_info = []
    series = {}
    for sid in sids:
        site = sid.split("@")[0]
        if site not in catalog:
            continue
        events = dedup(catalog[site], cfg["catalog_dedup_days"])
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        comps = {}
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            amps = robust_step_amplitudes(t, enu[:, comp], events, cfg)
            kept = [a for a in amps if a[2] >= float(cfg["min_snr"])]
            if kept:
                comps[comp] = {"t": t, "y": enu[:, comp], "labels": kept}
                for a in kept:
                    label_info.append({"sid": sid, "comp": comp, "epoch": a[0],
                                       "amp_mm": a[1], "snr": a[2], "code": a[3]})
        if comps:
            series[sid] = comps

    print(f"stations with labels {len(series)}; labels {len(label_info)}", flush=True)
    if not series:
        raise SystemExit("no catalogue labels survived the 3-sigma filter")

    # ---- decision fields and curves
    # curves[method][sid][comp] = curve dict; fields computed once per series
    curves = {}
    for si, (sid, comps) in enumerate(series.items()):
        for comp, d in comps.items():
            t, y = d["t"], d["y"]
            true = [a[0] for a in d["labels"]]
            fields = {"classical": rev.classical_step_field(t, y)}
            for prefix, netdict in nets.items():
                for stem, (seed, net) in netdict.items():
                    fields[f"{prefix}_s{seed}"] = dl.detect_field(net, y, device=device)
            for m, fld in fields.items():
                cues = curve_with_delay(fld, t, true, ths, tol, msep)
                curves.setdefault(m, {}).setdefault(sid, {})[comp] = cues
        print(f"  real offsets {si+1}/{len(series)} {sid}", flush=True)

    stations = sorted(series)
    # station-level curve = mean over components
    station_curve = {m: {} for m in curves}
    for m, by_sid in curves.items():
        for sid, comps in by_sid.items():
            station_curve[m][sid] = {
                k: np.mean([c[k] for c in comps.values()], axis=0)
                for k in ("P", "R", "F")}

    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    REGIONS = ("Europe", "North America", "Asia", "Pacific", "other")
    region = {s: region_of(s, meta) for s in stations}
    region_counts = {r: sum(1 for s in stations if region[s] == r) for r in REGIONS}
    splits = []
    for _ in range(int(cfg["val_splits"])):
        perm = rng.permutation(len(stations))
        nval = max(2, len(stations) // 2)
        splits.append(([stations[i] for i in perm[:nval]],
                       [stations[i] for i in perm[nval:]]))

    def mean_curve(m, subset):
        return tuple(np.mean([station_curve[m][s][k] for s in subset], axis=0)
                     for k in ("P", "R", "F"))

    def evaluate(m, th_idx, subset):
        """Precision/recall/delay at a fixed threshold on `subset`."""
        P, R, F = mean_curve(m, subset)
        delays, n_true, n_tp = [], 0, 0
        bins = {"3-5": [0, 0], "5-10": [0, 0], ">10": [0, 0]}
        for sid in subset:
            for comp, d in series[sid].items():
                cues = curves[m][sid][comp]
                mt = cues["matched"][th_idx]
                dl = cues["delays"][th_idx]
                delays.extend(dl)
                n_true += len(d["labels"])
                n_tp += len(mt)
                for j, a in enumerate(d["labels"]):
                    b = ("3-5" if a[2] < 5 else "5-10" if a[2] < 10 else ">10")
                    bins[b][0] += int(j in mt)
                    bins[b][1] += 1
        return {"precision": float(P[th_idx]), "recall": float(R[th_idx]),
                "f1": float(F[th_idx]),
                "delay_median_days": float(np.median(delays) * 365.25) if delays else None,
                "delay_abs_median_days": float(np.median(np.abs(delays)) * 365.25)
                if delays else None,
                "n_true_events": int(n_true), "n_tp": int(n_tp),
                "recall_by_snr": {b: (v[0] / v[1] if v[1] else None)
                                  for b, v in bins.items()},
                "n_by_snr": {b: v[1] for b, v in bins.items()}}

    # ---- station-level validation selection
    res_methods = {}
    for m in curves:
        vals = []
        reg_vals = {}
        for vidx, tidx in splits:
            _, _, Fv = mean_curve(m, vidx)
            j = int(np.argmax(Fv))
            vals.append(evaluate(m, j, tidx))
            for r in REGIONS:
                sub = [s for s in tidx if region[s] == r]
                if sub:
                    reg_vals.setdefault(r, []).append(evaluate(m, j, sub))
        entry = {
            "val_selected_f1_mean": float(np.mean([v["f1"] for v in vals])),
            "val_selected_f1_std": float(np.std([v["f1"] for v in vals], ddof=1)),
            "precision_mean": float(np.mean([v["precision"] for v in vals])),
            "recall_mean": float(np.mean([v["recall"] for v in vals])),
            "delay_median_days_mean": float(np.mean(
                [v["delay_median_days"] for v in vals if v["delay_median_days"] is not None])),
            "delay_abs_median_days_mean": float(np.mean(
                [v["delay_abs_median_days"] for v in vals
                 if v["delay_abs_median_days"] is not None])),
            "recall_by_snr": {},
            "n_true_events_mean": float(np.mean([v["n_true_events"] for v in vals])),
        }
        for b in ("3-5", "5-10", ">10"):
            r = [v["recall_by_snr"][b] for v in vals if v["recall_by_snr"][b] is not None]
            entry["recall_by_snr"][b] = float(np.mean(r)) if r else None
        P, R, F = mean_curve(m, stations)
        entry["ap"] = float(rev.average_precision(P, R))
        entry["f1_oracle"] = float(np.max(F))
        entry["by_region"] = {}
        for r in REGIONS:
            subs = [s for s in stations if region[s] == r]
            if not subs:
                continue
            f1s = [v["f1"] for v in reg_vals.get(r, [])]
            Pr, Rr, Fr = mean_curve(m, subs)
            entry["by_region"][r] = {
                "n_stations": region_counts[r],
                "val_selected_f1_mean": float(np.mean(f1s)) if f1s else None,
                "val_selected_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "n_splits": len(f1s),
                "ap": float(rev.average_precision(Pr, Rr))}
        res_methods[m] = entry

    # arch summaries
    summary = {}
    for prefix in ("unet", "tcn"):
        keys = [m for m in res_methods if m.startswith(prefix + "_s")]
        if not keys:
            continue
        for k in ("val_selected_f1_mean", "ap", "precision_mean", "recall_mean",
                  "delay_median_days_mean", "delay_abs_median_days_mean", "f1_oracle"):
            summary[f"{prefix}_{k}"] = float(np.mean([res_methods[m][k] for m in keys]))
        summary[f"{prefix}_val_selected_f1_std"] = float(np.std(
            [res_methods[m]["val_selected_f1_mean"] for m in keys], ddof=1))
        summary[f"{prefix}_n_seeds"] = len(keys)
        summary[f"{prefix}_per_seed"] = {
            m: {k: res_methods[m][k] for k in ("val_selected_f1_mean", "ap",
                                               "precision_mean", "recall_mean",
                                               "delay_median_days_mean",
                                               "delay_abs_median_days_mean")}
            for m in keys}
        summary[f"{prefix}_by_region"] = {}
        for r in REGIONS:
            rows_r = [res_methods[m]["by_region"][r] for m in keys
                      if r in res_methods[m]["by_region"]]
            if not rows_r:
                continue
            summary[f"{prefix}_by_region"][r] = {
                "n_stations": region_counts[r],
                "f1_mean": float(np.mean(
                    [x["val_selected_f1_mean"] for x in rows_r])),
                "f1_std_across_seeds": float(np.std(
                    [x["val_selected_f1_mean"] for x in rows_r], ddof=1)),
                "ap_mean": float(np.mean([x["ap"] for x in rows_r]))}
    summary["classical"] = {k: res_methods["classical"][k] for k in
                            ("val_selected_f1_mean", "ap", "precision_mean",
                             "recall_mean", "delay_median_days_mean",
                             "delay_abs_median_days_mean", "f1_oracle",
                             "recall_by_snr", "n_true_events_mean", "by_region")}
    summary["region_counts"] = region_counts

    amps = np.array([a["amp_mm"] for a in label_info], float)
    snrs = np.array([a["snr"] for a in label_info], float)
    label_summary = {
        "n_stations_with_labels": len(series),
        "n_component_series": int(sum(len(v) for v in series.values())),
        "n_labels": len(label_info),
        "n_by_code": {str(c): int(sum(1 for a in label_info if a["code"] == c))
                      for c in cfg["catalog_codes"]},
        "n_by_component": {str(c): int(sum(1 for a in label_info if a["comp"] == c))
                           for c in cfg["components"]},
        "n_by_snr": {"3-5": int((snrs < 5).sum()),
                     "5-10": int(((snrs >= 5) & (snrs < 10)).sum()),
                     ">10": int((snrs >= 10).sum())},
        "amp_mm_median": float(np.median(amps)),
        "amp_mm_iqr": [float(np.percentile(amps, 25)), float(np.percentile(amps, 75))],
        "amp_mm_min": float(amps.min()), "amp_mm_max": float(amps.max()),
    }

    syn = {}
    sp = os.path.join(ROOT, cfg["synthetic_reference"])
    if os.path.exists(sp):
        s = json.load(open(sp, encoding="utf-8"))
        for A, d in s.get("detect", {}).items():
            syn[A] = {"classical": {"ap": d["classical"]["ap"],
                                    "f1_opt": d["classical"]["f1_opt"]["f1"]},
                      "unet": {"ap_mean": d["learned"]["ap_seed_mean"],
                               "f1_opt_mean": d["learned"]["f1_opt_per_seed_mean"],
                               "f1_opt_std": d["learned"]["f1_opt_per_seed_std"]}}
            vs = s.get("validation_selection", {}).get(A, {})
            if vs:
                syn[A]["classical"]["val_selected_f1"] = vs["classical"]["validator_selected_f1_mean"]
                syn[A]["unet"]["val_selected_f1"] = vs["learned"]["validator_selected_f1_mean"]

    results = {
        "exp_id": cfg["exp_id"],
        "catalog": {"url": cfg["catalog_url"],
                    "download_date": cfg["catalog_download_date"],
                    "sha256": sha, "lines_used": n_raw,
                    "codes": cfg["catalog_codes"],
                    "label_rule": f"observed |step| >= {cfg['min_snr']} sigma; "
                                  f"dedup {cfg['catalog_dedup_days']} d; "
                                  f"edge margin {cfg['edge_margin_years']} y"},
        "protocol": {"tol_days": cfg["tol_days"],
                     "min_sep_days": cfg["min_sep_days"],
                     "val_splits": cfg["val_splits"],
                     "station_level": True},
        "labels": label_summary,
        "detection": {"per_method": res_methods, "summary": summary},
        "synthetic_reference": syn,
    }
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w",
                            encoding="utf-8"), indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8") as fh:
        fh.write("method,ap,val_selected_f1,precision,recall,delay_median_days,"
                 "delay_abs_median_days,recall_snr_3_5,recall_snr_5_10,recall_snr_gt10\n")
        for m, d in res_methods.items():
            fh.write(f"{m},{d['ap']},{d['val_selected_f1_mean']},{d['precision_mean']},"
                     f"{d['recall_mean']},{d['delay_median_days_mean']},"
                     f"{d['delay_abs_median_days_mean']},{d['recall_by_snr']['3-5']},"
                     f"{d['recall_by_snr']['5-10']},{d['recall_by_snr']['>10']}\n")
    with open(os.path.join(out_dir, "labels.csv"), "w", encoding="utf-8") as fh:
        fh.write("sid,comp,epoch,amp_mm,snr,code\n")
        for a in label_info:
            fh.write(f"{a['sid']},{a['comp']},{a['epoch']},{a['amp_mm']},{a['snr']},{a['code']}\n")
    print(json.dumps({"labels": label_summary, "summary": {
        k: v for k, v in summary.items() if not k.endswith("per_seed")}}, indent=2))


if __name__ == "__main__":
    main()
