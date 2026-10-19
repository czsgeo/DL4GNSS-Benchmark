#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261008-data-prep -- build the frozen benchmark dataset.

Two real-data sources, kept in one file but tagged and namespaced so the same
4-char site code in both sources never collides:

* **JPL** single-analysis-centre daily ENU series (sibling workspace,
  `clean_jpl.npz`) -> series id ``<SITE>@jpl``, source ``jpl``.
* **NGL** ``tenv3`` long daily records -> series id ``<SITE>``, source ``ngl``.
  The station list is frozen by a deterministic selection from the NGL station
  holdings catalogue (``DataHoldings.txt``): require >= ``min_solutions``
  solutions, record start <= ``t_begin_max_year``, record end >=
  ``t_end_min_year`` and span >= ``min_span_years``; then take the
  ``target_n`` longest, tie-broken by solution count and site code.

Outputs (data/, gitignored)
---------------------------
* series.npz        per-series t (decimal year), enu (mm), sig (mm)
* split.json        station-level train/val/test split (deterministic, by site)
* stations.csv      one row per kept series (id, site, source, span, n, comp)
* manifest.json     provenance: input sha256, holdings sha256, selection rule

results.json holds the summary numbers used in the manuscript.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import ngl  # noqa: E402

DATA = os.path.join(HERE, "data")
NGL_DIR = os.path.join(DATA, "ngl")
HOLDINGS = os.path.join(DATA, "ngl_holdings.txt")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def completeness(t: np.ndarray) -> tuple[float, float, float]:
    """Return (span_years, completeness, max_gap_days) on a daily grid."""
    if len(t) < 2:
        return 0.0, 0.0, 0.0
    span = float(t[-1] - t[0])
    n_expected = span * 365.25 + 1
    comp = float(len(t) / n_expected) if n_expected > 0 else 0.0
    dt = np.diff(t) * 365.25
    return span, comp, float(dt.max()) if len(dt) else 0.0


def load_jpl(cfg) -> dict:
    npz = np.load(cfg["jpl_source_npz"], allow_pickle=True)
    sites = [str(s) for s in npz["sites"]]
    out = {}
    for s in sites:
        t = np.asarray(npz[f"t__{s}"], float)
        enu = np.asarray(npz[f"enu__{s}"], float)
        if enu.ndim == 1:                       # single-component fallback
            enu = enu[:, None]
        o = np.argsort(t)
        out[f"{s}@jpl"] = {"source": "jpl", "site": s,
                           "t": t[o], "enu_mm": enu[o] * 1000.0,   # m -> mm
                           "sig_mm": np.full_like(enu[o], np.nan)}
    return out


def select_ngl(cfg) -> list[str]:
    """Frozen NGL station list = deterministic function of the holdings file."""
    c = cfg["ngl"]
    rows = ngl.load_holdings(HOLDINGS)
    cands = []
    for r in rows:
        span = r["tend_y"] - r["tbeg_y"]
        if r["n"] < c["min_solutions"]:
            continue
        if r["tbeg_y"] > c["t_begin_max_year"] or r["tend_y"] < c["t_end_min_year"]:
            continue
        if span < c["min_span_years"]:
            continue
        cands.append((r["site"], span, r["n"]))
    cands.sort(key=lambda x: (-x[1], -x[2], x[0]))   # span, solutions, site
    return [s for s, _, _ in cands[: int(c["target_n"])]]


def download_ngl(sites: list[str], frame: str, workers: int = 16) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.makedirs(NGL_DIR, exist_ok=True)
    ok, done = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ngl.download_station, s, NGL_DIR, frame): s for s in sites}
        for fut in as_completed(futs):
            done += 1
            if fut.result():
                ok.append(futs[fut])
            if done % 25 == 0:
                print(f"  ngl {done}/{len(sites)} ({len(ok)} ok, {time.time()-t0:.0f}s)", flush=True)
    return ok


def load_ngl(sites: list[str]) -> dict:
    out = {}
    for s in sites:
        p = os.path.join(NGL_DIR, f"{s}.tenv3")
        if not os.path.exists(p):
            continue
        try:
            d = ngl.parse_tenv3(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  parse fail {s}: {exc!r}")
            continue
        if len(d["t"]) < 2:
            continue
        out[s] = {"source": "ngl", "site": s, "t": d["t"], "enu_mm": d["enu_mm"],
                  "sig_mm": d["sig_mm"], "lat": d["lat"], "lon": d["lon"], "h": d["h"]}
    return out


def main():
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    os.makedirs(DATA, exist_ok=True)

    if not os.path.exists(HOLDINGS):
        print("downloading NGL holdings catalogue", flush=True)
        ngl.download_holdings(HOLDINGS)

    print("loading JPL single-AC series", flush=True)
    recs = load_jpl(cfg)

    print("selecting + downloading NGL subset", flush=True)
    targets = select_ngl(cfg)
    with open(os.path.join(HERE, "ngl_targets.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(targets) + "\n")
    got = download_ngl(targets, cfg["ngl"]["frame"], workers=cfg["ngl"].get("workers", 16))
    ngl_recs = load_ngl(got)
    recs.update(ngl_recs)          # ids are namespaced, so nothing is dropped

    # --- QC filter -----------------------------------------------------------
    qc = cfg["qc"]
    keep, rows = {}, []
    for sid, r in sorted(recs.items()):
        span, comp, gap = completeness(r["t"])
        if len(r["t"]) < qc["min_epochs"] or comp < qc["min_completeness"]:
            continue
        if span < qc["min_span_years"]:
            continue
        keep[sid] = r
        rows.append((sid, r["site"], r["source"], len(r["t"]), round(span, 2),
                     round(comp, 3), round(gap, 1),
                     r.get("lat", float("nan")), r.get("lon", float("nan"))))

    # --- split (by site, so a site in both sources stays together) ------------
    rng = np.random.default_rng(cfg["split"]["seed"])
    stations = sorted({keep[s]["site"] for s in keep})
    rng.shuffle(stations)
    f = cfg["split"]["fractions"]
    n = len(stations)
    n_tr = int(round(f[0] * n))
    n_va = int(round(f[1] * n))
    by_site = {"train": stations[:n_tr],
               "val": stations[n_tr:n_tr + n_va],
               "test": stations[n_tr + n_va:]}
    split = {k: sorted(s for s in keep if keep[s]["site"] in set(v))
             for k, v in by_site.items()}

    # --- persist unified npz -------------------------------------------------
    order = sorted(keep)
    kwargs = {"sites": np.array(order),
              "station": np.array([keep[s]["site"] for s in order]),
              "source": np.array([keep[s]["source"] for s in order])}
    for s in order:
        kwargs[f"t__{s}"] = keep[s]["t"]
        kwargs[f"enu__{s}"] = keep[s]["enu_mm"]
        kwargs[f"sig__{s}"] = keep[s]["sig_mm"]
    np.savez_compressed(os.path.join(DATA, "series.npz"), **kwargs)
    with open(os.path.join(DATA, "split.json"), "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    with open(os.path.join(DATA, "stations.csv"), "w", encoding="utf-8") as fh:
        fh.write("sid,site,source,n_epochs,span_years,completeness,max_gap_days,lat,lon\n")
        for row in rows:
            fh.write(",".join(str(x) for x in row) + "\n")

    # --- manifest ------------------------------------------------------------
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ngl_frame": cfg["ngl"]["frame"],
        "ngl_selection": {k: cfg["ngl"][k] for k in
                          ("target_n", "min_solutions", "t_begin_max_year",
                           "t_end_min_year", "min_span_years")},
        "ngl_targets_n": len(targets),
        "ngl_downloaded_n": len(got),
        "jpl_source": {"path": cfg["jpl_source_npz"], "sha256": sha256(cfg["jpl_source_npz"])},
        "ngl_holdings_sha256": sha256(HOLDINGS),
        "ngl_targets_sha256": sha256(os.path.join(HERE, "ngl_targets.txt")),
    }

    def stats(vals):
        v = np.asarray(vals, float)
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max())} if len(v) else {}

    counts = {src: sum(1 for s in keep if keep[s]["source"] == src)
              for src in ("jpl", "ngl")}
    spans = [completeness(keep[s]["t"])[0] for s in keep]
    comps = [completeness(keep[s]["t"])[1] for s in keep]
    results = {
        "exp_id": cfg["exp_id"],
        "n_stations_kept": len(stations),
        "n_series_kept": len(keep),
        "n_jpl": counts["jpl"], "n_ngl": counts["ngl"],
        "span_years": stats(spans),
        "completeness": stats(comps),
        "split": {k: {"series": len(v),
                      "stations": len({keep[s]["site"] for s in v})}
                  for k, v in split.items()},
        "t_begin": float(min(keep[s]["t"][0] for s in keep)),
        "t_end": float(max(keep[s]["t"][-1] for s in keep)),
        "manifest": manifest,
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(DATA, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: results[k] for k in
                      ("n_stations_kept", "n_series_kept", "n_jpl", "n_ngl",
                       "span_years", "completeness", "split")}, indent=2))


if __name__ == "__main__":
    main()
