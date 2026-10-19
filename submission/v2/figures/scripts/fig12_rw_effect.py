"""Fig. 12 -- the random-walk sensitivity arm of the downstream velocity test.

Same six held-out stations, two components and three realisations as Fig. 7,
with and without a random-walk component at 1.0 mm/yr^1/2. The added component
lowers absolute accuracy for every method and leaves the ordering unchanged.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261018-rw-downstream/results_perwin.json")))

METHODS = [("raw", "raw"), ("ssa", "SSA"), ("wavelet", "wavelet"),
           ("rts", "RTS"), ("median", "median"), ("unet", "U-Net")]
LEVELS = [("wn_fn", "WN + FN", 0.42), ("wn_fn_rw", "WN + FN + RW", 1.0)]

fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.9),
                         gridspec_kw=dict(wspace=0.34, hspace=0.52,
                                          left=0.085, right=0.985, top=0.86,
                                          bottom=0.13))
x = np.arange(len(METHODS))

PANELS = [
    ("vel_bias_mm_yr", "(a) velocity bias (mm yr$^{-1}$)", None),
    ("vel_rmse_mm_yr", "(b) velocity RMSE (mm yr$^{-1}$)", None),
    ("calibration_ratio", "(c) reported / empirical error", 1.0),
    ("coverage_1sigma", r"(d) 1$\sigma$ coverage", 0.683),
]

for ax, (key, title, nominal) in zip(axes.ravel(), PANELS):
    for lev, lab, alpha in LEVELS:
        vals = [d["levels"][lev]["clean"][m]["orig"][key] for m, _ in METHODS]
        for i, ((m, _), v) in enumerate(zip(METHODS, vals)):
            c = fs.ORANGE if m == "unet" else fs.BLUE
            ax.bar(x[i] + (-0.19 if lev == "wn_fn" else 0.19), v, width=0.36,
                   color=c, alpha=alpha, linewidth=0)
    ax.axhline(0.0, color="#444444", lw=0.7)
    if nominal is not None:
        ax.axhline(nominal, color="#444444", lw=0.8, ls="--")
    ax.set_xticks(x, [l for _, l in METHODS], rotation=35, ha="right",
                  fontsize=8)
    ax.set_title(title, fontsize=8.3, loc="left")
    ax.tick_params(length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[1, 0].set_ylim(0.0, 1.15)
axes[1, 1].set_ylim(0.0, 0.75)
axes[0, 0].set_ylim(-0.065, 0.006)
axes[0, 1].set_ylim(0.0, 0.085)
axes[1, 0].text(0.98, 0.90, "nominal", transform=axes[1, 0].transAxes,
                ha="right", fontsize=8, color="#444444")
axes[1, 1].text(0.98, 0.925, "nominal", transform=axes[1, 1].transAxes,
                ha="right", fontsize=8, color="#444444")

from matplotlib.patches import Patch
handles = [Patch(facecolor="#777777", alpha=a, label=l)
           for (_, l, a) in LEVELS]
fig.legend(handles=handles, fontsize=8, loc="upper right", ncol=2,
           frameon=False, bbox_to_anchor=(0.985, 0.965), handlelength=1.2,
           columnspacing=1.0)

fig.suptitle("Adding random walk lowers absolute accuracy without changing the method ordering",
             fontsize=8.8, x=0.085, ha="left", y=0.985)
fs.save_fig(fig, "fig12_rw_effect")
print("saved fig12")
