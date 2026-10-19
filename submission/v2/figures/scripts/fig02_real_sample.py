"""Fig. 2 -- one real station: catalogued offset, gaps, noise, and the controlled
injection used in the benchmark."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

sys.path.insert(0, os.path.join(fs.WS, "experiments", "common"))
import bench  # noqa: E402
import gnss_lib as gl  # noqa: E402

SID = "GOL2"
EVENT = 2019.50
AMP = [46.2, -5.3, 0.0]           # catalogued component steps (mm), E/N/U
COMP = ["E", "N", "U"]

npz = np.load(os.path.join(fs.WS, "experiments/exp-20261008-data-prep/data/series.npz"),
              allow_pickle=True)
t = np.asarray(npz[f"t__{SID}"], float)
enu = np.asarray(npz[f"enu__{SID}"], float)

X, _ = gl.build_design(t, orders=(1, 2), poly=2)
keep = np.abs(t - EVENT) > 0.5

fig, axes = plt.subplots(3, 2, figsize=(7.1, 4.0), sharex="col",
                         gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.18,
                                          hspace=0.13, left=0.075, right=0.99,
                                          top=0.90, bottom=0.10))

# gaps from the observation cadence (t is in decimal years)
dt_gap = np.diff(t) * 365.25
gap_idx = np.where(dt_gap > 10)[0]
# widest gap, used for the injection panel
ig = int(np.argmax(dt_gap))
gap_lo, gap_hi = float(t[ig]), float(t[ig + 1])
gap_days = int(round(float(dt_gap[ig])))

rng = np.random.default_rng(7)
nspec = bench.make_noise_spec(1.5, 3.0, 0.0, len(t), bench.DT)
for j in range(3):
    axl, axr = axes[j, 0], axes[j, 1]
    y = enu[:, j]
    Xj, _ = gl.build_design(t, orders=(1, 2), poly=2)
    bj, *_ = gl.robust_fit(t[keep], y[keep], Xj[keep])
    resid = y - Xj @ bj

    axl.plot(t, resid, ".", ms=0.6, color="#666666", alpha=0.6, rasterized=True)
    axl.axvline(EVENT, color=fs.ORANGE, lw=0.9, ls="--")
    for i in gap_idx:
        axl.axvspan(t[i], t[i + 1], color="#DDDDDD", alpha=0.8, lw=0)
    note = f"{AMP[j]:+.0f} mm" if AMP[j] else "no step"
    axl.text(EVENT - 0.3, np.nanpercentile(resid, 92), note, ha="right",
             fontsize=8, color=fs.ORANGE,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6))
    axl.set_ylabel(f"{COMP[j]} residual (mm)")

    # controlled injection, keeping the real observation gap. Both the truth
    # and the injection are shown after removing the fitted linear trend so
    # that the noise band and the step are visible at their own scale.
    truth = Xj @ bj + AMP[j] * (t >= EVENT)
    ysyn = truth + bench.inject(len(t), nspec, bench.DT, rng)
    lin = bj[0] + bj[1] * (t - float(np.mean(t)))
    truth, ysyn = truth - lin, ysyn - lin
    obs = (t < gap_lo) | (t > gap_hi)
    s3 = 3.0 * float(np.hypot(1.5, 3.0))
    axr.fill_between(t, truth - s3, truth + s3, color="#E6E6E6", lw=0,
                     label=r"$\pm3\sigma$ WN+FN")
    axr.plot(t[obs], ysyn[obs], ".", ms=0.6, color=fs.ORANGE, alpha=0.4,
             rasterized=True, label="injected observation")
    axr.plot(t, truth, color="black", lw=1.0, label="known truth", zorder=4)
    axr.axvspan(gap_lo, gap_hi, color="#DDDDDD", alpha=0.8, lw=0)
    axr.text(gap_lo + 0.3, np.nanpercentile(truth, 88), f"{gap_days} d gap",
             fontsize=8, color="#444444")

    axl.set_xlim(t[0] - 0.5, t[-1] + 0.5)
    for a in (axl, axr):
        a.tick_params(length=2, pad=1)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    if j < 2:
        axl.tick_params(labelbottom=False)
        axr.tick_params(labelbottom=False)

axes[0, 0].set_title("(a) real series, catalogued step and gaps",
                     fontsize=8.3, loc="left")
axes[0, 1].set_title("(b) controlled injection", fontsize=8.3, loc="left")
axes[-1, 0].set_xlabel("year")
axes[-1, 1].set_xlabel("year")
axes[0, 1].legend(fontsize=8, loc="upper right", handlelength=1.2,
                  labelspacing=0.2, borderpad=0.15)
fig.text(0.5, 0.955,
         f"{SID}: {int(t[0])}-{int(t[-1])} daily series; WN 1.5 mm + FN 3.0 mm injected",
         ha="center", fontsize=8, color="#333333")

fs.save_fig(fig, "fig02_real_sample")
print("gap spans", [(round(t[i], 2), round(t[i + 1], 2)) for i in gap_idx])
print("t", t[0], t[-1], "n", len(t))
