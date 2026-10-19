"""Fig. 14 -- noise amplitudes recovered from the benchmark stations.

The injected levels (WN 1.5 mm, FN 3.0 mm, RW arm 1.0 mm) are drawn against the
distribution recovered by Whittle MLE on every series' robust trajectory
residual. The RW panel is an upper bound because transients are absorbed there.
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261018-station-noise/results.json")))
rows = list(csv.DictReader(open(os.path.join(
    fs.WS, "experiments/exp-20261018-station-noise/results.csv"))))

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35),
                         gridspec_kw=dict(wspace=0.31, left=0.065, right=0.99,
                                          top=0.84, bottom=0.22))
SPEC = [("sigma_wn", "white noise $\\sigma_{\\rm WN}$ (mm)", 1.5, "(a)"),
        ("sigma_fn", "flicker noise $\\sigma_{\\rm FN}$ (mm)", 3.0, "(b)"),
        ("sigma_rw", "random walk $\\sigma_{\\rm RW}$ (mm yr$^{-1/2}$)", 1.0, "(c)")]

for ax, (key, label, inj, tag) in zip(axes, SPEC):
    v = np.array([float(r[key]) for r in rows])
    v = v[np.isfinite(v)]
    if key == "sigma_rw":
        vv = np.clip(v, 0.02, 2000.0)
        bins = np.logspace(np.log10(0.02), np.log10(2000.0), 36)
        ax.hist(vv, bins=bins, color=fs.BLUE, alpha=0.75,
                edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
    else:
        hi = np.percentile(v, 99.5)
        ax.hist(np.clip(v, 0, hi), bins=28, color=fs.BLUE, alpha=0.75,
                edgecolor="white", linewidth=0.3)
    ax.axvline(inj, color=fs.ORANGE, lw=1.0, ls="--")
    med = float(np.median(v))
    ax.axvline(med, color="#333333", lw=0.8, ls=":")
    ax.text(inj, ax.get_ylim()[1] * 0.94, " injected", color=fs.ORANGE,
            fontsize=8, ha="left", va="top")
    ax.text(med, ax.get_ylim()[1] * 0.80, " median %.2f" % med,
            color="#333333", fontsize=8, ha="left", va="top")
    ax.set_xlabel(label)
    ax.set_ylabel("component series")
    ax.set_title(f"{tag} recovered {label.split()[0]}", fontsize=8.4, loc="left")
    ax.tick_params(length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

wn = np.array([float(r["sigma_wn"]) for r in rows])
fn = np.array([float(r["sigma_fn"]) for r in rows])
axes[0].text(0.97, 0.72, f"n = {len(rows)} component series\n"
                         f"WN IQR {np.percentile(wn, 25):.2f}--"
                         f"{np.percentile(wn, 75):.2f} mm\n"
                         f"FN IQR {np.percentile(fn, 25):.2f}--"
                         f"{np.percentile(fn, 75):.2f} mm",
             transform=axes[0].transAxes, fontsize=8, color="#333333",
             va="top", ha="right")
fig.suptitle("Injected white and flicker levels sit inside the recovered distribution",
             fontsize=9.0, x=0.065, ha="left", y=0.99)
fs.save_fig(fig, "fig14_station_noise")
print("saved fig14")
