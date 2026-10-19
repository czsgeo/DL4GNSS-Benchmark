"""Fig. 6 -- gap-filling error on long synthetic records and real stations.

The v1.0 failure came from whole-series normalisation, which mismatches the
window statistics the network was trained under. Scoring the same ten nets per
window removes the failure: the learned filler is within 0.1 mm of SSA at 5 days
and best at 30 days on both record types, while the trajectory model remains
best at 120 days.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261009-in-distribution/results.json")))

GAPS = [5, 30, 120]
METH = [("linear", fs.GREY if hasattr(fs, "GREY") else "#7F7F7F", "--", "linear"),
        ("ssa", fs.SKY, "-.", "SSA"),
        ("mle", fs.BLUE, "-", "trajectory MLE")]


def unet(block, mode, L, key="mean"):
    vals = [v[key] for k, v in block[str(L)][mode].items() if k.startswith("unet")]
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85),
                         gridspec_kw=dict(wspace=0.26, left=0.07, right=0.99,
                                          top=0.80, bottom=0.19))

PANELS = [(axes[0], d["longseries"]["impute"], GAPS,
           "(a) long synthetic records, n = 40"),
          (axes[1], d["real"]["impute"], [5, 30],
           "(b) real held-out stations, n = 30")]

for ax, block, gaps, tag in PANELS:
    for m, color, ls, lab in METH:
        y = [block[str(L)]["global"][m]["mean"] for L in gaps]
        ax.plot(gaps, y, ls, color=color, marker="o", ms=2.8, lw=1.1, label=lab)
    for mode, color, ls, marker, lab in (
            ("global", fs.ORANGE, "--", "s", "U-Net, whole series"),
            ("perwin", fs.ORANGE, "-", "D", "U-Net, per window")):
        ys, es = zip(*[unet(block, mode, L) for L in gaps])
        ax.errorbar(gaps, ys, yerr=es, fmt=marker + ls, color=color, ms=3.0,
                    lw=1.3, capsize=2, elinewidth=0.7,
                    markerfacecolor="white" if mode == "global" else color,
                    label=lab)
    ax.set_xscale("log")
    ax.set_xticks(gaps)
    ax.set_xticklabels([str(g) for g in gaps])
    ax.minorticks_off()
    ax.set_xlabel("gap length (days)")
    ax.set_ylabel("gap RMSE (mm)")
    ax.set_yscale("log")
    ax.set_ylim(2.2, 11.0)
    ax.set_yticks([2.5, 3, 4, 5, 7, 10])
    ax.set_yticklabels(["2.5", "3", "4", "5", "7", "10"])
    ax.minorticks_off()
    ax.set_title(tag, fontsize=8.6, loc="left")
    ax.grid(True, axis="y", which="major", color="#CCCCCC", lw=0.4, alpha=0.6)
    ax.tick_params(length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].legend(fontsize=8, loc="upper left", ncol=2, handlelength=1.6,
               columnspacing=0.7, labelspacing=0.22, frameon=False)
fig.suptitle("Per-window inference removes the learned gap-filler's deficit at 5–30 days",
             fontsize=8.8, x=0.07, ha="left", y=0.99)
fs.save_fig(fig, "fig06_impute_log")
print("saved fig06")
