#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collapse the per-seed external-validation rows into architecture means.

results.json keeps one entry per trained network (`unet_sXXXXXXXX`). This
script averages those per-seed means into an architecture-level mean and
records the standard deviation across seeds, leaving every classical method
untouched. Writes summary.json next to results.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))


def collapse(methods):
    out, seeds = {}, []
    for k, v in methods.items():
        if k.startswith("unet_"):
            seeds.append(v["mean"])
        else:
            out[k] = v
    if seeds:
        a = np.asarray(seeds, float)
        out["unet"] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                       "n_seeds": int(len(a)),
                       "per_seed": {k: v["mean"] for k, v in methods.items()
                                    if k.startswith("unet_")}}
    return out


summary = {"exp_id": d["exp_id"], "protocol": d["protocol"],
           "n_network_seeds": d["n_network_seeds"],
           "by_source": {}, "by_region": {}}
for group in ("by_source", "by_region"):
    for g, blk in d[group].items():
        summary[group][g] = {
            "n_series": blk["n_series"],
            "denoise": collapse(blk["denoise"]),
            "impute30": collapse(blk["impute30"]),
        }

json.dump(summary, open(os.path.join(HERE, "summary.json"), "w",
                        encoding="utf-8"), indent=2)
for group in ("by_source", "by_region"):
    print(group)
    for g, blk in summary[group].items():
        den, imp = blk["denoise"], blk["impute30"]
        print(f"  {g:16s} n={blk['n_series']:3d} "
              f"den mle={den['mle']['mean']:.3f} wav={den['wavelet']['mean']:.3f} "
              f"unet={den['unet']['mean']:.3f}+-{den['unet']['std']:.3f} | "
              f"imp lin={imp['linear']['mean']:.3f} mle={imp['mle']['mean']:.3f} "
              f"unet={imp['unet']['mean']:.3f}+-{imp['unet']['std']:.3f}")
print("written", os.path.join(HERE, "summary.json"))
