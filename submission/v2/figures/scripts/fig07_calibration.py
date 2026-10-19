"""Fig. 7 -- downstream velocity calibration under the corrected estimator.

Every point uses the noise model estimated on the raw series for all methods
(the "orig" arm). Re-fitting noise on a denoised output is shown separately in
panel (b), where model-based denoisers leave no residual and the reported
uncertainty collapses. Numbers come from exp-20261018-rw-downstream after the
robust-fit scale floor was fixed.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261018-rw-downstream/results_perwin.json")))

lv = d["levels"]["wn_fn"]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7),
                         gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.30,
                                          left=0.075, right=0.985, top=0.82,
                                          bottom=0.17))

# ------------------------------------------------------------ (a) orig noise
ax = axes[0]
CLEAN = ["raw", "ssa", "wavelet", "rts", "median", "unet"]
IMPUTED = ["linear", "pchip", "ssa", "mle", "unet"]
for methods, arm, mk in ((CLEAN, "clean", "o"), (IMPUTED, "imputed", "s")):
    for m in methods:
        v = lv[arm][m]["orig"]
        c = fs.ORANGE if m == "unet" else fs.BLUE
        ax.scatter(v["calibration_ratio"], v["coverage_1sigma"], marker=mk,
                   s=24, facecolors=c, edgecolors="white", linewidths=0.45,
                   zorder=3)
ax.axvline(1.0, color="#444444", lw=0.8, ls="--")
ax.axhline(0.683, color="#444444", lw=0.8, ls=":")
ax.text(0.98, 0.97, "nominal 1:1", ha="right", va="top", fontsize=8,
        color="#444444", transform=ax.transAxes)
raw = lv["clean"]["raw"]["orig"]
un = lv["clean"]["unet"]["orig"]
ui = lv["imputed"]["unet"]["orig"]
arr = dict(arrowstyle="-", lw=0.5, color="#777777",
           shrinkA=1, shrinkB=2)
ax.annotate("raw / MLE", (raw["calibration_ratio"], raw["coverage_1sigma"]),
            xytext=(-42, -26), textcoords="offset points", fontsize=8,
            color="#333333", ha="right", va="top", arrowprops=arr)
ax.annotate("U-Net denoise", (un["calibration_ratio"], un["coverage_1sigma"]),
            xytext=(16, 14), textcoords="offset points", fontsize=8,
            color=fs.ORANGE, ha="left", arrowprops=arr)
ax.annotate("U-Net impute 30 d", (ui["calibration_ratio"], ui["coverage_1sigma"]),
            xytext=(34, -30), textcoords="offset points", fontsize=8,
            color=fs.ORANGE, ha="left", va="top", arrowprops=arr)
ax.set_xlim(0.0, 1.15)
ax.set_ylim(0.08, 0.52)
ax.set_xlabel("reported / empirical velocity error")
ax.set_ylabel(r"1$\sigma$ coverage")
ax.set_title("(a) every method stays near the classical calibration",
             fontsize=8.4, loc="left")

# ------------------------------------------------------------ (b) naive refit
ax = axes[1]
methods = ["raw", "mle", "ssa", "wavelet", "rts", "median", "unet"]
labels = ["raw", "MLE", "SSA", "wavelet", "RTS", "median", "U-Net"]
vals = [lv["clean"][m]["naive"]["calibration_ratio"] for m in methods]
colors = ["#8C8C8C" if m in ("raw", "mle") else
          (fs.ORANGE if m == "unet" else fs.BLUE) for m in methods]
x = np.arange(len(methods))
ax.bar(x, vals, color=colors, width=0.62)
ax.set_yscale("log")
ax.set_ylim(1e-16, 5)
ax.axhline(1.0, color="#444444", lw=0.8, ls="--")
for i, v in enumerate(vals):
    if v < 1e-2:
        ax.text(i, v * 3.0, f"{v:.1e}", ha="center", fontsize=8, rotation=90)
ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("reported / empirical error")
ax.set_title("(b) re-fitting noise collapses the model-based arms",
             fontsize=8.4, loc="left")

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker=m, color="none", markerfacecolor="#777777",
                  markeredgecolor="white", markersize=4,
                  label=l) for m, l in [("o", "denoised (clean)"),
                                        ("s", "imputed, 30 d gaps")]]
axes[0].legend(handles=handles, fontsize=8, loc="lower left",
               handletextpad=0.2, labelspacing=0.25, borderpad=0.2)

for a in axes:
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fig.suptitle("Under protocol-matched inference, velocity calibration is preserved by every method",
             fontsize=8.8, x=0.075, ha="left", y=0.985)
fs.save_fig(fig, "fig07_calibration")
print("saved fig07")
