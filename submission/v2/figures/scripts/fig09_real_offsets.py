"""Fig. 9 -- offset detection against real NGL catalogue labels."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261018-real-offsets/results.json")))
pm = d["detection"]["per_method"]
ref = d["synthetic_reference"]

unet = sorted(k for k in pm if k.startswith("unet_"))
tcn = sorted(k for k in pm if k.startswith("tcn_"))
cl = pm["classical"]


def vals(keys, field):
    out = []
    for k in keys:
        v = pm[k].get(field)
        if v is not None and np.isfinite(v):
            out.append(float(v))
    return np.asarray(out)


fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.1),
                         gridspec_kw=dict(width_ratios=[1.05, 1.15, 1.05],
                                          wspace=0.38, left=0.075, right=0.99,
                                          top=0.88, bottom=0.21))
rng = np.random.default_rng(3)

# ------------------------------------------------------- (a) F1 per seed
ax = axes[0]
groups = [(unet, "o", "U-Net"), (tcn, "^", "TCN")]
for (keys, mk, lab), dx in zip(groups, [0.0, 0.10]):
    v = vals(keys, "val_selected_f1_mean")
    if not len(v):
        continue
    x = 0.72 + dx + 0.05 * rng.standard_normal(len(v))
    ax.scatter(x, v, marker=mk, s=11, color=fs.ORANGE, edgecolors="white",
               linewidths=0.3, alpha=0.85, zorder=3, label=lab)
    ax.hlines(np.median(v), x.min() - 0.06, x.max() + 0.06, color=fs.ORANGE,
              lw=1.1, zorder=4)
ax.errorbar([0.28], [cl["val_selected_f1_mean"]],
            yerr=[cl["val_selected_f1_std"]], fmt="o", ms=5, color=fs.BLUE,
            capsize=2.5, elinewidth=0.9, lw=0, label="classical")
ax.set_xlim(0.05, 0.98)
ax.set_ylim(0.0, 0.56)
ax.set_xticks([0.28, 0.72], ["classical", "learned"], fontsize=8)
ax.set_ylabel(r"$F_1$ (validation-selected)")
ax.set_title("(a) learned F1 is seed-unstable", fontsize=8.5, loc="left")
ax.legend(fontsize=8, loc="upper right", handletextpad=0.2,
          borderpad=0.2, labelspacing=0.25)
ax.text(0.04, 0.04, f"{len(unet) + len(tcn)} learned seeds", transform=ax.transAxes,
        fontsize=8, color="#444444")

# ------------------------------------------------------- (b) recall by SNR
ax = axes[1]
strata = ["3-5", "5-10", ">10"]
cols = [fs.BLUE, fs.ORANGE, "#E69F00"]
x = np.arange(len(strata))
ax.bar(x - 0.27, [cl["recall_by_snr"][s] for s in strata], width=0.24,
       color=fs.BLUE, label="classical")
learnt = []
for k in unet + tcn:
    r = pm[k].get("recall_by_snr")
    if r:
        learnt.append([r[s] for s in strata])
learnt = np.asarray(learnt)
ax.bar(x + 0.03, learnt.mean(axis=0), width=0.24, color=fs.ORANGE,
       yerr=learnt.std(axis=0, ddof=1),
       error_kw=dict(lw=0.7, capsize=2, elinewidth=0.7),
       label="learned (mean±s.d.)")
ax.set_xticks(x, strata)
ax.set_xlabel("step SNR at the catalogued date")
ax.set_ylabel("recall")
ax.set_ylim(0, 1.02)
ax.set_title("(b) the gap widens with SNR", fontsize=8.5, loc="left")
ax.legend(fontsize=8, loc="upper left", handlelength=1.2,
          labelspacing=0.25, borderpad=0.2)

# ------------------------------------------------------- (c) AP and timing
ax = axes[2]
for keys, mk, lab in groups:
    ap = vals(keys, "ap")
    dl = vals(keys, "delay_abs_median_days_mean")
    n = min(len(ap), len(dl))
    if not n:
        continue
    ax.scatter(dl[:n], ap[:n], marker=mk, s=12, color=fs.ORANGE,
               edgecolors="white", linewidths=0.3, alpha=0.85, label=lab,
               zorder=3)
ax.scatter([cl["delay_abs_median_days_mean"]], [cl["ap"]], marker="*", s=90,
           color=fs.BLUE, edgecolors="white", linewidths=0.4, zorder=4,
           label="classical")
ax.set_xlim(0, 90)
ax.set_ylim(-0.02, 0.48)
ax.set_xlabel("|detection delay| (days)")
ax.set_ylabel("average precision")
ax.set_title("(c) lower AP, shorter delay", fontsize=8.5, loc="left")
ax.legend(fontsize=8, loc="upper right", handletextpad=0.2,
          borderpad=0.2, labelspacing=0.25)
ax.text(0.97, 0.05, f"real labels; {d['labels']['n_labels']:,} events\n"
        f"at {d['labels']['n_stations_with_labels']} held-out stations",
        transform=ax.transAxes, ha="right", fontsize=8, color="#444444")

for a in axes:
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fs.save_fig(fig, "fig09_real_offsets")
print("classical", {k: round(cl[k], 3) for k in
                    ["val_selected_f1_mean", "ap", "precision_mean",
                     "recall_mean", "delay_abs_median_days_mean"]})
print("unet n", len(unet), "tcn n", len(tcn))
