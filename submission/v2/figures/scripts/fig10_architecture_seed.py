"""Fig. 10 -- architecture x seed distributions on the shared test split."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261018-tcn/results.json")))

ARCH = [("unet", "U-Net", fs.ORANGE, "o"),
        ("tcn", "TCN", fs.GREEN, "^"),
        ("gru", "BiGRU", "#7a7a7a", "s")]


def seeds(entry):
    if not entry:
        return np.array([])
    return np.asarray([v for _, v in sorted(entry["per_seed"].items(),
                                            key=lambda kv: int(kv[0]))], float)


fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.05),
                         gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.36,
                                          left=0.075, right=0.99, top=0.86,
                                          bottom=0.24))
rng = np.random.default_rng(5)


def panel(ax, entries, classical, ylabel, title, ref_label):
    xt, xl = [], []
    for i, (prefix, label, color, mk) in enumerate(ARCH):
        v = seeds(entries.get(prefix))
        if not len(v):
            continue
        x0 = i * 0.9
        x = x0 + 0.06 * rng.standard_normal(len(v))
        ax.scatter(x, v, marker=mk, s=13, color=color, edgecolors="white",
                   linewidths=0.35, alpha=0.9, zorder=3)
        ax.hlines(v.mean(), x0 - 0.14, x0 + 0.14, color=color, lw=1.4, zorder=4)
        ax.vlines(x0, v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1),
                  color=color, lw=0.9, zorder=2)
        xt.append(x0)
        xl.append(f"{label}\n({len(v)} seeds)")
    ax.axhline(classical, color=fs.BLUE, lw=1.0, ls="--", zorder=1)
    ax.text(0.02, classical, ref_label, color=fs.BLUE, fontsize=8,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.set_xticks(xt, xl, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5, loc="left")


# (a) denoising: RMSE-reduction fraction
den = d["denoise"]["per_method"]
panel(axes[0], {p: den.get(p) for p, *_ in ARCH}, den["mle"]["mean"],
      "injected-error reduction", "(a) all learned arms sit near $0.36$",
      "classical MLE")

# (b) 30-day gap filling: RMSE
imp = d["impute"]["30"]["per_method"]
panel(axes[1], {p: imp.get(p, {}).get("rmse") for p, *_ in ARCH},
      imp["mle"]["rmse"]["mean"], "RMSE (mm)",
      "(b) TCN and BiGRU lead at 30 days", "classical MLE")

# (c) detection at 8 mm: validation-selected F1
det = d["detect"]["8"]
by = {p: {"per_seed": {k.split("_s")[1]: v["val_selected_f1_mean"]
                       for k, v in det["per_method"].items()
                       if k.startswith(p + "_s")}}
      for p, *_ in ARCH}
by = {p: (v if v["per_seed"] else None) for p, v in by.items()}
panel(axes[2], by, det["per_method"]["classical"]["val_selected_f1_mean"],
      r"$F_1$ (validation-selected)",
      "(c) TCN beats U-Net, both below classical", "classical")

for a in axes:
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fs.save_fig(fig, "fig10_architecture_seed")
print("denoise", {p: (round(den[p]["mean"], 3), round(den[p]["std"], 3))
                  for p, *_ in ARCH if p in den})
print("impute30", {p: (round(imp[p]["rmse"]["mean"], 3),
                       round(imp[p]["rmse"]["std"], 3))
                   for p, *_ in ARCH if p in imp})
print("detect8", {p: (round(float(np.mean(list(by[p]["per_seed"].values()))), 3))
                  for p, *_ in ARCH if by.get(p)})
