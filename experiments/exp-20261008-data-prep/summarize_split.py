#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the split-level counts quoted in Sec. 3 of the v2 manuscript.

results.json already records series and site counts per fold; the epoch counts
and the test-fold source breakdown are not stored there, and were previously
reported from an ad-hoc count.  This script recomputes them from the frozen
dataset and split file and writes split-summary.json, so the numbers in the
paper point at a file produced by a committed script.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
split = json.load(open(os.path.join(HERE, "data", "split.json"), encoding="utf-8"))
npz = np.load(os.path.join(HERE, "data", "series.npz"), allow_pickle=True)
sites = [str(s) for s in npz["sites"]]
source = {str(s): str(q) for s, q in zip(npz["sites"], npz["source"])}

out = {"folds": {}, "n_series_total": len(sites),
       "n_sites_total": len({s.split("@")[0] for s in sites}),
       "n_dual_source_sites": 0}

by_site = {}
for s in sites:
    by_site.setdefault(s.split("@")[0], set()).add(source[s])
out["n_dual_source_sites"] = sum(1 for v in by_site.values() if len(v) > 1)

for fold, series in split.items():
    epochs = sum(int(len(npz[f"t__{s}"])) for s in series)
    src_counts = {"ngl": 0, "jpl": 0}
    for s in series:
        src_counts[source[s]] += 1
    out["folds"][fold] = {
        "n_series": len(series),
        "n_sites": len({s.split("@")[0] for s in series}),
        "n_epochs": epochs,
        "series_by_source": src_counts,
    }

path = os.path.join(HERE, "split-summary.json")
json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=2))
