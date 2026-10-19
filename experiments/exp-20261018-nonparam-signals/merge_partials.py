#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the four per-regime partial_*.json files into results.json/results.csv.

The regimes are run one at a time (run.py --regime) so a long run can be
resumed; the partials carry the same summaries the single-shot run writes.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))

merged = {k: json.load(open(os.path.join(HERE, f"partial_{k}.json"),
                            encoding="utf-8"))
          for k in (r["name"] for r in cfg["regimes"])}
first = merged[cfg["regimes"][0]["name"]]
results = {k: first[k] for k in ("exp_id", "n_stations", "components", "n_seeds",
                                 "unet_in_seeds", "unet_oof_seeds")}
results["regimes"] = {}
for r in cfg["regimes"]:
    results["regimes"][r["name"]] = merged[r["name"]]["regimes"][r["name"]]

json.dump(results, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"),
          indent=2)
with open(os.path.join(HERE, "results.csv"), "w", encoding="utf-8") as fh:
    fh.write("regime,method,reduction_mean,reduction_std,ci_lo,ci_hi,n\n")
    for name, summ in results["regimes"].items():
        for m, d in summ.items():
            fh.write(f"{name},{m},{d['reduction_mean']},{d['reduction_std']},"
                     f"{d['ci_lo']},{d['ci_hi']},{d['n']}\n")
for name, summ in results["regimes"].items():
    agg = {m: round(d["reduction_mean"], 4) for m, d in summ.items() if "__" not in m}
    print(name, json.dumps(agg))
print("regimes", list(results["regimes"]))
